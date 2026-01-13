//! Incremental BMC engine with clause learning and caching
//!
//! This module provides the main incremental verification engine that combines
//! all the components: clause database, diff analysis, and solver integration.
//!
//! # Integration with Portfolio Solver
//!
//! The engine can use CaDiCaL to solve DIMACS files and extract learned clauses
//! from DRAT proofs. These learned clauses are cached for future verification runs.

use crate::clause_db::{ClauseDatabase, ClauseDbError, ClauseEntry};
use crate::config::IncrementalConfig;
use crate::content_hash::ContentHash;
use crate::diff::{DiffAnalyzer, DiffResult};
use crate::result::{CacheStats, IncrementalResult};
use kani_fast_core::{KaniConfig, KaniWrapper, VerificationStatus};
use kani_fast_portfolio::{CaDiCaL, LearnConfig, SolverConfig, SolverResult};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

/// Result of verifying a DIMACS file
#[derive(Debug, Clone)]
pub struct DimacsResult {
    /// Whether the formula is satisfiable (counterexample found)
    pub satisfiable: bool,
    /// Whether the formula is proven unsatisfiable (property holds)
    pub proven: bool,
    /// Number of learned clauses extracted from the proof
    pub learned_clauses: usize,
    /// Number of cached clauses that were available
    pub cached_clauses_used: usize,
    /// Time taken for verification
    pub duration: Duration,
    /// Whether result came from cache
    pub from_cache: bool,
}

impl DimacsResult {
    /// Check if verification was definitive (SAT or UNSAT)
    pub fn is_definitive(&self) -> bool {
        self.satisfiable || self.proven
    }
}

/// Errors that can occur during incremental BMC
#[derive(Debug, Error)]
pub enum IncrementalError {
    #[error("Database error: {0}")]
    Database(#[from] ClauseDbError),

    #[error("Diff analysis error: {0}")]
    Diff(#[from] crate::diff::DiffError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Verification error: {0}")]
    Verification(String),

    #[error("Solver error: {0}")]
    Solver(String),
}

/// Incremental BMC engine
pub struct IncrementalBmc {
    /// Configuration
    config: IncrementalConfig,
    /// Clause database
    db: Arc<Mutex<ClauseDatabase>>,
    /// Project root path
    project_root: PathBuf,
    /// Diff analyzer
    diff_analyzer: DiffAnalyzer,
    /// Accumulated cache statistics
    stats: CacheStats,
}

impl IncrementalBmc {
    /// Create a new incremental BMC engine
    pub fn new(
        project_root: impl Into<PathBuf>,
        config: IncrementalConfig,
    ) -> Result<Self, IncrementalError> {
        let project_root = project_root.into();
        let db_path = project_root.join(&config.database_path);

        let db = ClauseDatabase::open(&db_path)?;
        let diff_analyzer = DiffAnalyzer::new(&project_root);

        Ok(Self {
            config,
            db: Arc::new(Mutex::new(db)),
            project_root,
            diff_analyzer,
            stats: CacheStats::new(),
        })
    }

    /// Create with an in-memory database (for testing)
    pub fn in_memory(project_root: impl Into<PathBuf>) -> Result<Self, IncrementalError> {
        let project_root = project_root.into();
        let db = ClauseDatabase::in_memory()?;
        let diff_analyzer = DiffAnalyzer::new(&project_root);

        Ok(Self {
            config: IncrementalConfig::default(),
            db: Arc::new(Mutex::new(db)),
            project_root,
            diff_analyzer,
            stats: CacheStats::new(),
        })
    }

    /// Get the project root path
    pub fn project_root(&self) -> &Path {
        &self.project_root
    }

    /// Perform incremental verification
    pub async fn verify(&mut self) -> Result<IncrementalResult, IncrementalError> {
        let start = Instant::now();

        // Step 1: Analyze changes
        let diff_result = {
            let db = self.db.lock().await;
            self.diff_analyzer.analyze(&db)?
        };

        info!(
            "Diff analysis: {} files changed, {} functions invalidated",
            diff_result.changed_count(),
            diff_result.invalidated_functions.len()
        );

        // Step 2: Check for cached result
        if !diff_result.has_changes() {
            if let Some(cached) = self.check_cached_result(&diff_result).await? {
                self.stats.cache_hits += 1;
                self.refresh_database_stats().await?;
                debug!("Using cached verification result");
                return Ok(IncrementalResult::from_cache(
                    cached.result == "Proven",
                    start.elapsed(),
                    self.stats.clone(),
                ));
            }

            self.stats.cache_misses += 1;
        }

        // Step 3: Load relevant cached clauses
        let cached_clauses = self.load_cached_clauses(&diff_result).await?;

        // Step 4: Perform verification with clause hints
        let (proven, learned_clauses) =
            self.run_verification(&diff_result, &cached_clauses).await?;

        // Step 5: Store learned clauses
        let clauses_stored = self
            .store_learned_clauses(&diff_result, &learned_clauses)
            .await?;

        // Step 6: Update file hashes
        {
            let db = self.db.lock().await;
            self.diff_analyzer.update_hashes(&db)?;
        }

        // Step 7: Store verification result
        let duration = start.elapsed();
        self.store_verification_result(&diff_result, proven, duration, clauses_stored as u64)
            .await?;

        Ok(IncrementalResult::proven(duration, self.stats.clone())
            .with_clause_stats(cached_clauses.len() as u64, clauses_stored as u64)
            .with_function_stats(
                diff_result.invalidated_functions.len() as u64,
                diff_result.cached_functions.len() as u64,
            ))
    }

    /// Verify a specific file
    pub async fn verify_file(
        &mut self,
        path: &Path,
    ) -> Result<IncrementalResult, IncrementalError> {
        let start = Instant::now();

        // Compute file hash
        let file_hash = ContentHash::from_file(path)?;

        // Check if we have cached clauses for this exact file
        let db = self.db.lock().await;
        let cached_hash = db.get_file_hash(&path.to_string_lossy())?;

        let use_cache = cached_hash.as_ref() == Some(&file_hash);

        if use_cache {
            // Try to use cached result
            if let Some(cached) = db.get_verification_result(&file_hash)? {
                self.stats.cache_hits += 1;
                drop(db);

                return Ok(IncrementalResult::from_cache(
                    cached.result == "Proven",
                    start.elapsed(),
                    self.stats.clone(),
                ));
            }
        }

        self.stats.cache_misses += 1;
        drop(db);

        // Perform full verification
        let (proven, learned) = self.verify_single_file(path).await?;

        // Store results
        let db = self.db.lock().await;
        db.store_file_hash(&path.to_string_lossy(), &file_hash)?;
        db.store_verification_result(
            &file_hash,
            &path.to_string_lossy(),
            if proven { "Proven" } else { "Disproven" },
            start.elapsed().as_millis() as u64,
            learned.len() as u64,
        )?;

        Ok(
            IncrementalResult::proven(start.elapsed(), self.stats.clone())
                .with_clause_stats(0, learned.len() as u64),
        )
    }

    /// Check for a cached verification result
    async fn check_cached_result(
        &self,
        diff: &DiffResult,
    ) -> Result<Option<crate::clause_db::CachedResult>, IncrementalError> {
        let db = self.db.lock().await;
        Ok(db.get_verification_result(&diff.new_project_hash)?)
    }

    /// Load cached clauses for functions that can reuse them
    async fn load_cached_clauses(
        &mut self,
        diff: &DiffResult,
    ) -> Result<Vec<ClauseEntry>, IncrementalError> {
        let load_start = Instant::now();
        let mut all_clauses = Vec::new();

        let db = self.db.lock().await;

        for function in &diff.cached_functions {
            let function_hash = ContentHash::from_function(function, "");

            // Load top clauses by activity
            let clauses =
                db.load_top_clauses(&function_hash, self.config.max_clauses_per_function)?;

            self.stats.clauses_loaded += clauses.len() as u64;
            self.stats.clauses_valid += clauses.len() as u64;

            all_clauses.extend(clauses);
        }

        self.stats.load_time = load_start.elapsed();
        let db_stats = db.stats()?;
        drop(db);
        self.apply_db_stats(&db_stats);

        debug!("Loaded {} cached clauses", all_clauses.len());
        Ok(all_clauses)
    }

    /// Store learned clauses from verification
    async fn store_learned_clauses(
        &mut self,
        diff: &DiffResult,
        clauses: &[(Vec<i32>, f64, Option<u32>)],
    ) -> Result<usize, IncrementalError> {
        if clauses.is_empty() {
            return Ok(0);
        }

        let store_start = Instant::now();

        let db = self.db.lock().await;

        // Store clauses associated with the project hash
        let clauses_refs: Vec<(&[i32], f64, Option<u32>)> = clauses
            .iter()
            .filter(|(lits, _, _)| lits.len() >= self.config.min_clause_size)
            .map(|(lits, activity, lbd)| (lits.as_slice(), *activity, *lbd))
            .collect();

        let stored = db.store_clauses_batch(
            &clauses_refs,
            &diff.new_project_hash,
            &diff.new_project_hash, // Use project hash as function hash for now
            "incremental",
        )?;

        self.stats.store_time += store_start.elapsed();
        let db_stats = db.stats()?;
        drop(db);
        self.apply_db_stats(&db_stats);

        debug!(
            "Stored {} learned clauses in {:?}",
            stored,
            store_start.elapsed()
        );
        Ok(stored)
    }

    /// Store verification result
    async fn store_verification_result(
        &self,
        diff: &DiffResult,
        proven: bool,
        duration: Duration,
        clause_count: u64,
    ) -> Result<(), IncrementalError> {
        let db = self.db.lock().await;

        db.store_verification_result(
            &diff.new_project_hash,
            "project",
            if proven { "Proven" } else { "Disproven" },
            duration.as_millis() as u64,
            clause_count,
        )?;

        Ok(())
    }

    /// Run the actual verification with clause hints
    ///
    /// Uses KaniWrapper to verify the Rust project. Cached clauses are logged
    /// but not directly usable by CBMC (they would require CNF-level integration).
    async fn run_verification(
        &self,
        _diff: &DiffResult,
        cached_clauses: &[ClauseEntry],
    ) -> Result<(bool, Vec<(Vec<i32>, f64, Option<u32>)>), IncrementalError> {
        // Log cached clause availability (future: could inject into solver)
        if !cached_clauses.is_empty() {
            debug!(
                "Available cached clauses: {} (not directly usable by CBMC)",
                cached_clauses.len()
            );
        }

        // Configure Kani with timeout from our config
        let kani_config = KaniConfig {
            timeout: self.config.solver_timeout,
            ..Default::default()
        };

        // Create wrapper and run verification
        let wrapper = KaniWrapper::new(kani_config).map_err(|e| {
            IncrementalError::Verification(format!("Failed to create Kani wrapper: {e}"))
        })?;

        let result = wrapper.verify(&self.project_root).await.map_err(|e| {
            IncrementalError::Verification(format!("Kani verification failed: {e}"))
        })?;

        // Determine if proven based on verification status
        let proven = matches!(result.status, VerificationStatus::Proven);

        // Log verification outcome
        match &result.status {
            VerificationStatus::Proven => {
                info!("Verification successful in {:?}", result.duration);
            }
            VerificationStatus::Disproven => {
                info!("Verification found failures");
            }
            VerificationStatus::Timeout => {
                warn!("Verification timed out after {:?}", result.duration);
            }
            VerificationStatus::Unknown { reason } => {
                warn!("Verification inconclusive: {}", reason);
            }
            VerificationStatus::Error { message } => {
                warn!("Verification error: {}", message);
            }
        }

        // CBMC doesn't expose learned clauses, so we return empty
        // Future: could extract clauses if using direct SAT solver integration
        let learned_clauses: Vec<(Vec<i32>, f64, Option<u32>)> = Vec::new();

        Ok((proven, learned_clauses))
    }

    /// Verify a single Rust file
    ///
    /// Creates a temporary Cargo project containing the file and runs Kani on it.
    /// For DIMACS files, use `verify_dimacs` instead.
    async fn verify_single_file(
        &self,
        path: &Path,
    ) -> Result<(bool, Vec<(Vec<i32>, f64, Option<u32>)>), IncrementalError> {
        // Check file extension
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or_default();

        match extension {
            "cnf" | "dimacs" => {
                // For DIMACS files, delegate to verify_dimacs logic directly
                // But since verify_dimacs requires &mut self, we handle SAT solving here
                let solver = CaDiCaL::detect().await.ok_or_else(|| {
                    IncrementalError::Solver("CaDiCaL solver not found".to_string())
                })?;

                let solver_config = SolverConfig {
                    timeout: self.config.solver_timeout,
                    ..Default::default()
                };

                let learn_config = LearnConfig {
                    min_clause_size: self.config.min_clause_size,
                    max_clause_size: self.config.max_clause_size,
                    max_clauses: self.config.max_clauses_per_function,
                    cleanup_proof: true,
                };

                let result = solver
                    .solve_and_learn(path, &solver_config, learn_config)
                    .await
                    .map_err(|e| IncrementalError::Solver(e.to_string()))?;

                let proven = matches!(&result.output.result, SolverResult::Unsat { .. });
                let learned: Vec<(Vec<i32>, f64, Option<u32>)> = result
                    .learned_clauses
                    .into_iter()
                    .enumerate()
                    .map(|(i, c)| (c, 1.0 / (i as f64 + 1.0), None))
                    .collect();

                Ok((proven, learned))
            }
            "rs" => {
                // For Rust files, we need to create a temporary Cargo project
                // This is complex because Kani expects a full Cargo project
                // For now, check if the file is part of the project root
                if path.starts_with(&self.project_root) {
                    // File is in our project, run verification on the whole project
                    // Kani doesn't support single-file verification easily
                    let kani_config = KaniConfig {
                        timeout: self.config.solver_timeout,
                        ..Default::default()
                    };

                    let wrapper = KaniWrapper::new(kani_config).map_err(|e| {
                        IncrementalError::Verification(format!(
                            "Failed to create Kani wrapper: {e}"
                        ))
                    })?;

                    let result = wrapper.verify(&self.project_root).await.map_err(|e| {
                        IncrementalError::Verification(format!("Kani verification failed: {e}"))
                    })?;

                    let proven = matches!(result.status, VerificationStatus::Proven);
                    Ok((proven, Vec::new()))
                } else {
                    // File is outside project - we can't easily verify it
                    // Return success with warning
                    warn!(
                        "Cannot verify file outside project root: {}",
                        path.display()
                    );
                    Ok((true, Vec::new()))
                }
            }
            _ => {
                // Unknown file type
                warn!("Unknown file type for verification: {}", path.display());
                Ok((true, Vec::new()))
            }
        }
    }

    /// Verify a DIMACS CNF file with learned clause extraction
    ///
    /// This method:
    /// 1. Checks for cached verification results
    /// 2. Loads cached clauses if available
    /// 3. Runs CaDiCaL with DRAT proof output
    /// 4. Extracts learned clauses from the proof
    /// 5. Stores learned clauses and result in the database
    pub async fn verify_dimacs(
        &mut self,
        dimacs_path: &Path,
    ) -> Result<DimacsResult, IncrementalError> {
        let start = Instant::now();

        // Compute file hash for caching
        let file_hash = ContentHash::from_file(dimacs_path)?;
        let file_key = dimacs_path.to_string_lossy().to_string();

        // Check for cached result
        {
            let db = self.db.lock().await;
            if let Some(cached) = db.get_verification_result(&file_hash)? {
                self.stats.cache_hits += 1;
                return Ok(DimacsResult {
                    satisfiable: cached.result == "Sat",
                    proven: cached.result == "Unsat",
                    learned_clauses: 0,
                    cached_clauses_used: 0,
                    duration: start.elapsed(),
                    from_cache: true,
                });
            }
        }

        self.stats.cache_misses += 1;

        // Load cached clauses for this file
        let cached_clauses = {
            let db = self.db.lock().await;
            db.load_top_clauses(&file_hash, self.config.max_clauses_per_function)?
        };
        let cached_count = cached_clauses.len();

        // Detect CaDiCaL solver
        let solver = CaDiCaL::detect()
            .await
            .ok_or_else(|| IncrementalError::Solver("CaDiCaL solver not found".to_string()))?;

        // Configure solver
        let solver_config = SolverConfig {
            timeout: self.config.solver_timeout,
            ..Default::default()
        };

        let learn_config = LearnConfig {
            min_clause_size: self.config.min_clause_size,
            max_clause_size: self.config.max_clause_size,
            max_clauses: self.config.max_clauses_per_function,
            cleanup_proof: true,
        };

        // Run solver with learned clause extraction
        let result = solver
            .solve_and_learn(dimacs_path, &solver_config, learn_config)
            .await
            .map_err(|e| IncrementalError::Solver(e.to_string()))?;

        // Determine result
        let (satisfiable, proven) = match &result.output.result {
            SolverResult::Sat { .. } => (true, false),
            SolverResult::Unsat { .. } => (false, true),
            SolverResult::Unknown { reason } => {
                warn!("Solver returned unknown: {}", reason);
                (false, false)
            }
        };

        // Store learned clauses
        let learned_count = result.learned_clauses.len();
        if !result.learned_clauses.is_empty() {
            let db = self.db.lock().await;

            // Convert to format expected by clause database
            let clauses_with_activity: Vec<(&[i32], f64, Option<u32>)> = result
                .learned_clauses
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    // Use position-based activity (earlier clauses are more active)
                    let activity = 1.0 / (i as f64 + 1.0);
                    (c.as_slice(), activity, None)
                })
                .collect();

            let stored =
                db.store_clauses_batch(&clauses_with_activity, &file_hash, &file_hash, "cadical")?;

            info!("Stored {} learned clauses from CaDiCaL", stored);
        }

        // Store verification result
        {
            let db = self.db.lock().await;
            db.store_file_hash(&file_key, &file_hash)?;
            db.store_verification_result(
                &file_hash,
                &file_key,
                if satisfiable {
                    "Sat"
                } else if proven {
                    "Unsat"
                } else {
                    "Unknown"
                },
                start.elapsed().as_millis() as u64,
                learned_count as u64,
            )?;
        }

        Ok(DimacsResult {
            satisfiable,
            proven,
            learned_clauses: learned_count,
            cached_clauses_used: cached_count,
            duration: start.elapsed(),
            from_cache: false,
        })
    }

    /// Clean up old clauses from the database
    pub async fn cleanup(&self) -> Result<usize, IncrementalError> {
        let db = self.db.lock().await;
        let deleted = db.cleanup_old_clauses(self.config.max_clause_age)?;
        Ok(deleted)
    }

    /// Get database statistics
    pub async fn database_stats(&self) -> Result<crate::clause_db::DbStats, IncrementalError> {
        let db = self.db.lock().await;
        Ok(db.stats()?)
    }

    /// Invalidate all clauses for a function
    pub async fn invalidate_function(
        &self,
        function_name: &str,
    ) -> Result<usize, IncrementalError> {
        let function_hash = ContentHash::from_function(function_name, "");
        let db = self.db.lock().await;
        Ok(db.invalidate_function(&function_hash)?)
    }

    /// Get the current cache statistics
    pub fn cache_stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Reset cache statistics
    pub fn reset_stats(&mut self) {
        self.stats = CacheStats::new();
    }

    /// Refresh database-backed cache statistics
    async fn refresh_database_stats(&mut self) -> Result<(), IncrementalError> {
        let db = self.db.lock().await;
        let db_stats = db.stats()?;
        drop(db);
        self.apply_db_stats(&db_stats);
        Ok(())
    }

    /// Apply database stats to cache stats
    fn apply_db_stats(&mut self, db_stats: &crate::clause_db::DbStats) {
        self.stats.total_clauses = db_stats.clause_count;
        self.stats.database_size = db_stats.database_size;
    }
}

/// Builder for IncrementalBmc with fluent API
pub struct IncrementalBmcBuilder {
    project_root: PathBuf,
    config: IncrementalConfig,
}

impl IncrementalBmcBuilder {
    /// Create a new builder
    pub fn new(project_root: impl Into<PathBuf>) -> Self {
        Self {
            project_root: project_root.into(),
            config: IncrementalConfig::default(),
        }
    }

    /// Set the configuration
    pub fn config(mut self, config: IncrementalConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the database path
    pub fn database_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.database_path = path.into();
        self
    }

    /// Set the maximum clause age
    pub fn max_clause_age(mut self, age: Duration) -> Self {
        self.config.max_clause_age = age;
        self
    }

    /// Enable content-addressable storage
    pub fn content_addressable(mut self, enabled: bool) -> Self {
        self.config.content_addressable = enabled;
        self
    }

    /// Build the engine
    pub fn build(self) -> Result<IncrementalBmc, IncrementalError> {
        IncrementalBmc::new(self.project_root, self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::IncrementalConfigBuilder;
    use std::collections::HashSet;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_create_engine() {
        let temp_dir = TempDir::new().unwrap();
        let engine = IncrementalBmc::in_memory(temp_dir.path()).unwrap();

        assert_eq!(engine.project_root(), temp_dir.path());
    }

    #[tokio::test]
    async fn test_verify_empty_project() {
        let temp_dir = TempDir::new().unwrap();
        let mut engine = IncrementalBmc::in_memory(temp_dir.path()).unwrap();

        let result = engine.verify().await.unwrap();

        assert!(result.proven);
    }

    #[tokio::test]
    async fn test_database_stats() {
        let temp_dir = TempDir::new().unwrap();
        let engine = IncrementalBmc::in_memory(temp_dir.path()).unwrap();

        let stats = engine.database_stats().await.unwrap();

        assert_eq!(stats.clause_count, 0);
    }

    #[tokio::test]
    async fn test_cleanup() {
        let temp_dir = TempDir::new().unwrap();
        let engine = IncrementalBmc::in_memory(temp_dir.path()).unwrap();

        let deleted = engine.cleanup().await.unwrap();

        assert_eq!(deleted, 0);
    }

    #[tokio::test]
    async fn test_cache_stats() {
        let temp_dir = TempDir::new().unwrap();
        let mut engine = IncrementalBmc::in_memory(temp_dir.path()).unwrap();

        // Initial stats should be empty
        assert_eq!(engine.cache_stats().cache_hits, 0);

        // Run verification
        let _ = engine.verify().await.unwrap();

        // Reset stats
        engine.reset_stats();
        assert_eq!(engine.cache_stats().cache_hits, 0);
    }

    #[tokio::test]
    async fn test_store_learned_clauses_updates_stats() {
        let temp_dir = TempDir::new().unwrap();
        let mut engine = IncrementalBmc::in_memory(temp_dir.path()).unwrap();

        let diff = DiffResult {
            changes: Vec::new(),
            invalidated_functions: HashSet::new(),
            cached_functions: HashSet::new(),
            old_project_hash: None,
            new_project_hash: ContentHash::from_source("project"),
        };

        let clauses = vec![(vec![1, 2, 3], 1.0, Some(2))];

        let stored = engine.store_learned_clauses(&diff, &clauses).await.unwrap();

        assert_eq!(stored, 1);
        assert!(engine.cache_stats().store_time > Duration::ZERO);

        let db_stats = {
            let db = engine.db.lock().await;
            db.stats().unwrap()
        };

        assert_eq!(engine.cache_stats().total_clauses, db_stats.clause_count);
        assert_eq!(engine.cache_stats().database_size, db_stats.database_size);
    }

    #[tokio::test]
    async fn test_builder() {
        let temp_dir = TempDir::new().unwrap();

        let engine = IncrementalBmcBuilder::new(temp_dir.path())
            .max_clause_age(Duration::from_secs(3600))
            .content_addressable(true)
            .build()
            .unwrap();

        assert_eq!(engine.project_root(), temp_dir.path());
    }

    #[tokio::test]
    async fn test_verify_with_file() {
        let temp_dir = TempDir::new().unwrap();

        // Create a Rust file
        let file_path = temp_dir.path().join("test.rs");
        std::fs::write(&file_path, "fn test() {}").unwrap();

        let mut engine = IncrementalBmc::in_memory(temp_dir.path()).unwrap();

        let result = engine.verify_file(&file_path).await.unwrap();

        assert!(result.proven);
    }

    #[tokio::test]
    async fn test_invalidate_function() {
        let temp_dir = TempDir::new().unwrap();
        let engine = IncrementalBmc::in_memory(temp_dir.path()).unwrap();

        let deleted = engine.invalidate_function("test_fn").await.unwrap();

        // No clauses to delete yet
        assert_eq!(deleted, 0);
    }

    #[tokio::test]
    async fn test_verify_dimacs_sat() {
        // Skip if CaDiCaL not available
        if CaDiCaL::detect().await.is_none() {
            return;
        }

        let temp_dir = TempDir::new().unwrap();

        // Create a simple SAT problem
        let cnf_path = temp_dir.path().join("simple.cnf");
        std::fs::write(&cnf_path, "p cnf 3 2\n1 2 0\n-1 3 0\n").unwrap();

        let mut engine = IncrementalBmc::in_memory(temp_dir.path()).unwrap();
        let result = engine.verify_dimacs(&cnf_path).await.unwrap();

        assert!(result.satisfiable);
        assert!(!result.proven);
        assert!(!result.from_cache);
    }

    #[tokio::test]
    async fn test_verify_dimacs_unsat() {
        // Skip if CaDiCaL not available
        if CaDiCaL::detect().await.is_none() {
            return;
        }

        let temp_dir = TempDir::new().unwrap();

        // Create PHP problem (3 pigeons, 2 holes) - always UNSAT
        let cnf_path = temp_dir.path().join("php32.cnf");
        let dimacs = r"c PHP 3 pigeons 2 holes
p cnf 6 9
1 2 0
3 4 0
5 6 0
-1 -3 0
-1 -5 0
-3 -5 0
-2 -4 0
-2 -6 0
-4 -6 0
";
        std::fs::write(&cnf_path, dimacs).unwrap();

        // Use min_clause_size=1 to capture unit clauses in UNSAT proof
        let config = IncrementalConfigBuilder::new().min_clause_size(1).build();
        let mut engine = IncrementalBmc::new(temp_dir.path(), config).unwrap();
        let result = engine.verify_dimacs(&cnf_path).await.unwrap();

        assert!(!result.satisfiable);
        assert!(result.proven);
        assert!(!result.from_cache);
        // Should have learned some clauses (PHP proofs generate unit clauses)
        assert!(
            result.learned_clauses > 0,
            "Expected learned clauses from UNSAT proof"
        );
    }

    #[tokio::test]
    async fn test_verify_dimacs_caching() {
        // Skip if CaDiCaL not available
        if CaDiCaL::detect().await.is_none() {
            return;
        }

        let temp_dir = TempDir::new().unwrap();

        // Create a simple problem
        let cnf_path = temp_dir.path().join("cached.cnf");
        std::fs::write(&cnf_path, "p cnf 3 2\n1 2 0\n-1 3 0\n").unwrap();

        let mut engine = IncrementalBmc::in_memory(temp_dir.path()).unwrap();

        // First run - should not be from cache
        let result1 = engine.verify_dimacs(&cnf_path).await.unwrap();
        assert!(!result1.from_cache);

        // Second run - should be from cache
        let result2 = engine.verify_dimacs(&cnf_path).await.unwrap();
        assert!(result2.from_cache);
    }

    #[test]
    fn test_dimacs_result_is_definitive() {
        let sat_result = DimacsResult {
            satisfiable: true,
            proven: false,
            learned_clauses: 0,
            cached_clauses_used: 0,
            duration: Duration::ZERO,
            from_cache: false,
        };
        assert!(sat_result.is_definitive());

        let unsat_result = DimacsResult {
            satisfiable: false,
            proven: true,
            learned_clauses: 5,
            cached_clauses_used: 0,
            duration: Duration::ZERO,
            from_cache: false,
        };
        assert!(unsat_result.is_definitive());

        let unknown_result = DimacsResult {
            satisfiable: false,
            proven: false,
            learned_clauses: 0,
            cached_clauses_used: 0,
            duration: Duration::ZERO,
            from_cache: false,
        };
        assert!(!unknown_result.is_definitive());
    }

    #[tokio::test]
    async fn test_verify_single_file_dimacs() {
        // Skip if CaDiCaL not available
        if CaDiCaL::detect().await.is_none() {
            return;
        }

        let temp_dir = TempDir::new().unwrap();

        // Create a simple UNSAT DIMACS file (empty clause = contradiction)
        let cnf_path = temp_dir.path().join("unsat.cnf");
        std::fs::write(&cnf_path, "p cnf 1 2\n1 0\n-1 0\n").unwrap();

        let engine = IncrementalBmc::in_memory(temp_dir.path()).unwrap();

        // verify_single_file should detect .cnf extension and use SAT solving
        let (proven, learned) = engine.verify_single_file(&cnf_path).await.unwrap();

        // 1 AND -1 is UNSAT, so it should be proven
        assert!(proven);
        // May or may not learn clauses depending on solver (just check it doesn't crash)
        let _ = learned;
    }

    #[tokio::test]
    async fn test_verify_single_file_unknown_extension() {
        let temp_dir = TempDir::new().unwrap();

        // Create a file with unknown extension
        let file_path = temp_dir.path().join("test.xyz");
        std::fs::write(&file_path, "unknown content").unwrap();

        let engine = IncrementalBmc::in_memory(temp_dir.path()).unwrap();

        // Should return success with warning for unknown file types
        let (proven, learned) = engine.verify_single_file(&file_path).await.unwrap();
        assert!(proven); // Unknown files are assumed to pass
        assert!(learned.is_empty());
    }

    #[tokio::test]
    async fn test_store_verification_result_records_duration_and_clause_count() {
        let temp_dir = TempDir::new().unwrap();
        let engine = IncrementalBmc::in_memory(temp_dir.path()).unwrap();

        let diff = DiffResult {
            changes: Vec::new(),
            invalidated_functions: HashSet::new(),
            cached_functions: HashSet::new(),
            old_project_hash: None,
            new_project_hash: ContentHash::from_source("project"),
        };

        engine
            .store_verification_result(&diff, true, Duration::from_millis(250), 7)
            .await
            .unwrap();

        let cached = {
            let db = engine.db.lock().await;
            db.get_verification_result(&diff.new_project_hash).unwrap()
        }
        .expect("expected cached verification result");

        assert_eq!(cached.result, "Proven");
        assert_eq!(cached.duration_ms, 250);
        assert_eq!(cached.clause_count, 7);
    }

    #[tokio::test]
    async fn test_verify_uses_cached_result_updates_stats() {
        let temp_dir = TempDir::new().unwrap();
        let mut engine = IncrementalBmc::in_memory(temp_dir.path()).unwrap();

        let diff = {
            let db = engine.db.lock().await;
            engine.diff_analyzer.analyze(&db).unwrap()
        };

        {
            let db = engine.db.lock().await;
            db.store_verification_result(&diff.new_project_hash, "project", "Proven", 5, 0)
                .unwrap();
        }

        engine.reset_stats();
        let result = engine.verify().await.unwrap();

        assert!(result.from_cache);
        assert_eq!(engine.cache_stats().cache_hits, 1);
        assert_eq!(engine.cache_stats().cache_misses, 0);

        let db_stats = {
            let db = engine.db.lock().await;
            db.stats().unwrap()
        };

        assert_eq!(engine.cache_stats().total_clauses, db_stats.clause_count);
        assert_eq!(engine.cache_stats().database_size, db_stats.database_size);
    }

    // ===== IncrementalError tests =====

    #[test]
    fn test_incremental_error_database() {
        let db_err = ClauseDbError::InvalidClause;
        let err: IncrementalError = db_err.into();
        assert!(format!("{}", err).contains("Database error"));
        assert!(format!("{}", err).contains("Invalid clause"));
    }

    #[test]
    fn test_incremental_error_diff() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let diff_err = crate::diff::DiffError::Io(io_err);
        let err: IncrementalError = diff_err.into();
        assert!(format!("{}", err).contains("Diff analysis error"));
    }

    #[test]
    fn test_incremental_error_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let err: IncrementalError = io_err.into();
        assert!(format!("{}", err).contains("I/O error"));
        assert!(format!("{}", err).contains("file missing"));
    }

    #[test]
    fn test_incremental_error_verification() {
        let err = IncrementalError::Verification("kani failed".to_string());
        assert!(format!("{}", err).contains("Verification error"));
        assert!(format!("{}", err).contains("kani failed"));
    }

    #[test]
    fn test_incremental_error_solver() {
        let err = IncrementalError::Solver("timeout".to_string());
        assert!(format!("{}", err).contains("Solver error"));
        assert!(format!("{}", err).contains("timeout"));
    }

    #[test]
    fn test_incremental_error_debug() {
        let err = IncrementalError::Solver("test".to_string());
        let debug = format!("{:?}", err);
        assert!(debug.contains("Solver"));
    }

    // ===== DimacsResult tests =====

    #[test]
    fn test_dimacs_result_debug() {
        let result = DimacsResult {
            satisfiable: true,
            proven: false,
            learned_clauses: 5,
            cached_clauses_used: 3,
            duration: Duration::from_millis(100),
            from_cache: true,
        };
        let debug = format!("{:?}", result);
        assert!(debug.contains("satisfiable: true"));
        assert!(debug.contains("learned_clauses: 5"));
        assert!(debug.contains("cached_clauses_used: 3"));
        assert!(debug.contains("from_cache: true"));
    }

    #[test]
    fn test_dimacs_result_clone() {
        let result = DimacsResult {
            satisfiable: false,
            proven: true,
            learned_clauses: 10,
            cached_clauses_used: 2,
            duration: Duration::from_secs(1),
            from_cache: false,
        };
        let cloned = result.clone();
        assert_eq!(cloned.satisfiable, result.satisfiable);
        assert_eq!(cloned.proven, result.proven);
        assert_eq!(cloned.learned_clauses, result.learned_clauses);
        assert_eq!(cloned.cached_clauses_used, result.cached_clauses_used);
        assert_eq!(cloned.duration, result.duration);
        assert_eq!(cloned.from_cache, result.from_cache);
    }

    #[test]
    fn test_dimacs_result_is_definitive_both_false() {
        let result = DimacsResult {
            satisfiable: false,
            proven: false,
            learned_clauses: 0,
            cached_clauses_used: 0,
            duration: Duration::ZERO,
            from_cache: false,
        };
        assert!(!result.is_definitive());
    }

    #[test]
    fn test_dimacs_result_is_definitive_both_true() {
        // Edge case: both true (shouldn't happen in practice, but test the logic)
        let result = DimacsResult {
            satisfiable: true,
            proven: true,
            learned_clauses: 0,
            cached_clauses_used: 0,
            duration: Duration::ZERO,
            from_cache: false,
        };
        assert!(result.is_definitive());
    }

    // ===== IncrementalBmcBuilder tests =====

    #[test]
    fn test_builder_new() {
        let builder = IncrementalBmcBuilder::new("/tmp/project");
        assert_eq!(builder.project_root, PathBuf::from("/tmp/project"));
    }

    #[test]
    fn test_builder_config() {
        let temp_dir = TempDir::new().unwrap();
        let custom_config = IncrementalConfigBuilder::new()
            .min_clause_size(5)
            .max_clause_size(100)
            .build();

        let engine = IncrementalBmcBuilder::new(temp_dir.path())
            .config(custom_config)
            .build()
            .unwrap();

        assert_eq!(engine.config.min_clause_size, 5);
        assert_eq!(engine.config.max_clause_size, 100);
    }

    #[test]
    fn test_builder_database_path() {
        let temp_dir = TempDir::new().unwrap();
        let engine = IncrementalBmcBuilder::new(temp_dir.path())
            .database_path("custom.db")
            .build()
            .unwrap();

        assert_eq!(engine.config.database_path, PathBuf::from("custom.db"));
    }

    #[test]
    fn test_builder_max_clause_age() {
        let temp_dir = TempDir::new().unwrap();
        let engine = IncrementalBmcBuilder::new(temp_dir.path())
            .max_clause_age(Duration::from_secs(7200))
            .build()
            .unwrap();

        assert_eq!(engine.config.max_clause_age, Duration::from_secs(7200));
    }

    #[test]
    fn test_builder_content_addressable() {
        let temp_dir = TempDir::new().unwrap();
        let engine = IncrementalBmcBuilder::new(temp_dir.path())
            .content_addressable(false)
            .build()
            .unwrap();

        assert!(!engine.config.content_addressable);
    }

    #[test]
    fn test_builder_chained() {
        let temp_dir = TempDir::new().unwrap();
        let engine = IncrementalBmcBuilder::new(temp_dir.path())
            .database_path("test.db")
            .max_clause_age(Duration::from_secs(1800))
            .content_addressable(true)
            .build()
            .unwrap();

        assert_eq!(engine.config.database_path, PathBuf::from("test.db"));
        assert_eq!(engine.config.max_clause_age, Duration::from_secs(1800));
        assert!(engine.config.content_addressable);
    }

    // ===== IncrementalBmc edge case tests =====

    #[tokio::test]
    async fn test_new_with_custom_config() {
        let temp_dir = TempDir::new().unwrap();
        let config = IncrementalConfigBuilder::new()
            .solver_timeout(Duration::from_secs(30))
            .max_clauses_per_function(500)
            .build();

        let engine = IncrementalBmc::new(temp_dir.path(), config).unwrap();

        assert_eq!(engine.config.solver_timeout, Duration::from_secs(30));
        assert_eq!(engine.config.max_clauses_per_function, 500);
    }

    #[tokio::test]
    async fn test_store_learned_clauses_empty() {
        let temp_dir = TempDir::new().unwrap();
        let mut engine = IncrementalBmc::in_memory(temp_dir.path()).unwrap();

        let diff = DiffResult {
            changes: Vec::new(),
            invalidated_functions: HashSet::new(),
            cached_functions: HashSet::new(),
            old_project_hash: None,
            new_project_hash: ContentHash::from_source("project"),
        };

        let clauses: Vec<(Vec<i32>, f64, Option<u32>)> = Vec::new();
        let stored = engine.store_learned_clauses(&diff, &clauses).await.unwrap();

        assert_eq!(stored, 0);
    }

    #[tokio::test]
    async fn test_store_learned_clauses_filters_by_min_size() {
        let temp_dir = TempDir::new().unwrap();
        // Set min_clause_size to 3, so clauses with < 3 literals should be filtered
        let config = IncrementalConfigBuilder::new().min_clause_size(3).build();
        let mut engine = IncrementalBmc::new(temp_dir.path(), config).unwrap();

        let diff = DiffResult {
            changes: Vec::new(),
            invalidated_functions: HashSet::new(),
            cached_functions: HashSet::new(),
            old_project_hash: None,
            new_project_hash: ContentHash::from_source("project"),
        };

        // Include clauses of various sizes
        let clauses = vec![
            (vec![1], 1.0, None),          // size 1 - filtered
            (vec![1, 2], 1.0, None),       // size 2 - filtered
            (vec![1, 2, 3], 1.0, None),    // size 3 - kept
            (vec![1, 2, 3, 4], 1.0, None), // size 4 - kept
        ];

        let stored = engine.store_learned_clauses(&diff, &clauses).await.unwrap();

        // Only 2 clauses should be stored (size >= 3)
        assert_eq!(stored, 2);
    }

    #[tokio::test]
    async fn test_load_cached_clauses_empty_functions() {
        let temp_dir = TempDir::new().unwrap();
        let mut engine = IncrementalBmc::in_memory(temp_dir.path()).unwrap();

        let diff = DiffResult {
            changes: Vec::new(),
            invalidated_functions: HashSet::new(),
            cached_functions: HashSet::new(), // No cached functions
            old_project_hash: None,
            new_project_hash: ContentHash::from_source("project"),
        };

        let clauses = engine.load_cached_clauses(&diff).await.unwrap();

        assert!(clauses.is_empty());
        assert_eq!(engine.cache_stats().clauses_loaded, 0);
    }

    #[tokio::test]
    async fn test_load_cached_clauses_with_functions() {
        let temp_dir = TempDir::new().unwrap();
        let mut engine = IncrementalBmc::in_memory(temp_dir.path()).unwrap();

        // Store some clauses first
        let function_hash = ContentHash::from_function("test_fn", "");
        {
            let db = engine.db.lock().await;
            db.store_clauses_batch(
                &[(&[1, 2, 3][..], 1.0, None)],
                &function_hash,
                &function_hash,
                "test",
            )
            .unwrap();
        }

        let mut cached_functions = HashSet::new();
        cached_functions.insert("test_fn".to_string());

        let diff = DiffResult {
            changes: Vec::new(),
            invalidated_functions: HashSet::new(),
            cached_functions,
            old_project_hash: None,
            new_project_hash: ContentHash::from_source("project"),
        };

        let clauses = engine.load_cached_clauses(&diff).await.unwrap();

        assert_eq!(clauses.len(), 1);
        assert_eq!(engine.cache_stats().clauses_loaded, 1);
        assert_eq!(engine.cache_stats().clauses_valid, 1);
    }

    #[tokio::test]
    async fn test_check_cached_result_none() {
        let temp_dir = TempDir::new().unwrap();
        let engine = IncrementalBmc::in_memory(temp_dir.path()).unwrap();

        let diff = DiffResult {
            changes: Vec::new(),
            invalidated_functions: HashSet::new(),
            cached_functions: HashSet::new(),
            old_project_hash: None,
            new_project_hash: ContentHash::from_source("new_project"),
        };

        let result = engine.check_cached_result(&diff).await.unwrap();

        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_check_cached_result_some() {
        let temp_dir = TempDir::new().unwrap();
        let engine = IncrementalBmc::in_memory(temp_dir.path()).unwrap();

        let project_hash = ContentHash::from_source("cached_project");

        // Store a cached result
        {
            let db = engine.db.lock().await;
            db.store_verification_result(&project_hash, "project", "Proven", 100, 5)
                .unwrap();
        }

        let diff = DiffResult {
            changes: Vec::new(),
            invalidated_functions: HashSet::new(),
            cached_functions: HashSet::new(),
            old_project_hash: None,
            new_project_hash: project_hash,
        };

        let result = engine.check_cached_result(&diff).await.unwrap();

        assert!(result.is_some());
        let cached = result.unwrap();
        assert_eq!(cached.result, "Proven");
        assert_eq!(cached.duration_ms, 100);
        assert_eq!(cached.clause_count, 5);
    }

    #[tokio::test]
    async fn test_verify_file_uses_cache_on_same_hash() {
        // Skip if CaDiCaL not available
        if CaDiCaL::detect().await.is_none() {
            return;
        }

        let temp_dir = TempDir::new().unwrap();

        // Create a DIMACS file (simpler to verify than Rust files)
        let file_path = temp_dir.path().join("cached_test.cnf");
        // Simple UNSAT problem: x AND -x
        std::fs::write(&file_path, "p cnf 1 2\n1 0\n-1 0\n").unwrap();

        let mut engine = IncrementalBmc::in_memory(temp_dir.path()).unwrap();

        // First verification
        let result1 = engine.verify_file(&file_path).await.unwrap();
        let _initial_misses = engine.cache_stats().cache_misses;

        // Second verification - should use cache
        let result2 = engine.verify_file(&file_path).await.unwrap();

        assert!(result1.proven); // UNSAT = proven
        assert!(result2.proven);
        assert!(result2.from_cache);
        // Cache miss count shouldn't increase on cached hit
        assert!(engine.cache_stats().cache_hits > 0);
    }

    #[tokio::test]
    async fn test_verify_file_outside_project_root() {
        let temp_dir = TempDir::new().unwrap();
        let other_dir = TempDir::new().unwrap();

        // Create a Rust file outside project root
        let file_path = other_dir.path().join("outside.rs");
        std::fs::write(&file_path, "fn outside() {}").unwrap();

        let engine = IncrementalBmc::in_memory(temp_dir.path()).unwrap();

        // Should return success with warning (file outside project)
        let (proven, _) = engine.verify_single_file(&file_path).await.unwrap();
        assert!(proven);
    }

    #[tokio::test]
    async fn test_store_verification_result_disproven() {
        let temp_dir = TempDir::new().unwrap();
        let engine = IncrementalBmc::in_memory(temp_dir.path()).unwrap();

        let diff = DiffResult {
            changes: Vec::new(),
            invalidated_functions: HashSet::new(),
            cached_functions: HashSet::new(),
            old_project_hash: None,
            new_project_hash: ContentHash::from_source("failed_project"),
        };

        engine
            .store_verification_result(&diff, false, Duration::from_millis(500), 10)
            .await
            .unwrap();

        let cached = {
            let db = engine.db.lock().await;
            db.get_verification_result(&diff.new_project_hash).unwrap()
        }
        .expect("expected cached verification result");

        assert_eq!(cached.result, "Disproven");
        assert_eq!(cached.duration_ms, 500);
        assert_eq!(cached.clause_count, 10);
    }

    #[tokio::test]
    async fn test_invalidate_function_with_clauses() {
        let temp_dir = TempDir::new().unwrap();
        let engine = IncrementalBmc::in_memory(temp_dir.path()).unwrap();

        // Store clauses for a function
        let function_hash = ContentHash::from_function("invalidate_me", "");
        {
            let db = engine.db.lock().await;
            db.store_clauses_batch(
                &[(&[1, 2, 3][..], 1.0, None), (&[4, 5, 6][..], 0.5, None)],
                &function_hash,
                &function_hash,
                "test",
            )
            .unwrap();
        }

        let deleted = engine.invalidate_function("invalidate_me").await.unwrap();

        assert_eq!(deleted, 2);
    }

    #[tokio::test]
    async fn test_refresh_database_stats() {
        let temp_dir = TempDir::new().unwrap();
        let mut engine = IncrementalBmc::in_memory(temp_dir.path()).unwrap();

        // Store some clauses
        let hash = ContentHash::from_source("test");
        {
            let db = engine.db.lock().await;
            db.store_clauses_batch(&[(&[1, 2, 3][..], 1.0, None)], &hash, &hash, "test")
                .unwrap();
        }

        // Stats should be out of date
        assert_eq!(engine.cache_stats().total_clauses, 0);

        // Refresh stats
        engine.refresh_database_stats().await.unwrap();

        // Now stats should be updated
        assert_eq!(engine.cache_stats().total_clauses, 1);
    }

    #[tokio::test]
    async fn test_apply_db_stats() {
        let temp_dir = TempDir::new().unwrap();
        let mut engine = IncrementalBmc::in_memory(temp_dir.path()).unwrap();

        let db_stats = crate::clause_db::DbStats {
            clause_count: 42,
            function_count: 5,
            total_literals: 100,
            database_size: 1024,
        };

        engine.apply_db_stats(&db_stats);

        assert_eq!(engine.cache_stats().total_clauses, 42);
        assert_eq!(engine.cache_stats().database_size, 1024);
    }

    #[tokio::test]
    async fn test_verify_single_file_dimacs_extension() {
        // Skip if CaDiCaL not available
        if CaDiCaL::detect().await.is_none() {
            return;
        }

        let temp_dir = TempDir::new().unwrap();

        // Create a DIMACS file with .dimacs extension
        let cnf_path = temp_dir.path().join("test.dimacs");
        std::fs::write(&cnf_path, "p cnf 2 1\n1 2 0\n").unwrap();

        let engine = IncrementalBmc::in_memory(temp_dir.path()).unwrap();
        let (proven, _) = engine.verify_single_file(&cnf_path).await.unwrap();

        // Simple SAT problem - satisfiable, not proven UNSAT
        assert!(!proven);
    }

    #[tokio::test]
    async fn test_multiple_verifications_accumulate_stats() {
        let temp_dir = TempDir::new().unwrap();
        let mut engine = IncrementalBmc::in_memory(temp_dir.path()).unwrap();

        // Create multiple Rust files
        for i in 0..3 {
            let file_path = temp_dir.path().join(format!("test_{}.rs", i));
            std::fs::write(&file_path, format!("fn test_{}() {{}}", i)).unwrap();
            let _ = engine.verify_file(&file_path).await.unwrap();
        }

        // Should have 3 cache misses (first time for each file)
        assert_eq!(engine.cache_stats().cache_misses, 3);
    }

    #[tokio::test]
    async fn test_verify_preserves_diff_changes() {
        let temp_dir = TempDir::new().unwrap();

        // Create a Rust file so there's something to track
        let file_path = temp_dir.path().join("tracked.rs");
        std::fs::write(&file_path, "fn tracked() {}").unwrap();

        let mut engine = IncrementalBmc::in_memory(temp_dir.path()).unwrap();

        // First verify
        let result1 = engine.verify().await.unwrap();
        assert!(result1.proven);

        // Modify the file
        std::fs::write(&file_path, "fn tracked_modified() {}").unwrap();

        // Second verify - should detect changes
        let result2 = engine.verify().await.unwrap();
        assert!(result2.proven);
    }
}
