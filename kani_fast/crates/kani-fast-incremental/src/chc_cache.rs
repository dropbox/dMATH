//! Incremental CHC verification cache
//!
//! This module provides function-level caching for CHC verification results,
//! enabling <1s incremental re-verification by reusing results for unchanged
//! functions.
//!
//! # Design
//!
//! Each function is uniquely identified by a hash of:
//! 1. Its MIR representation (the actual code)
//! 2. Its dependencies (other functions it calls)
//! 3. The verification configuration
//!
//! When a function changes:
//! - Its cached result is invalidated
//! - All functions that depend on it are also invalidated (transitively)
//!
//! # Example
//!
//! ```ignore
//! use kani_fast_incremental::{ChcVerificationCache, ContentHash};
//!
//! let cache = ChcVerificationCache::open("./kani_cache.db")?;
//!
//! // Check if we have a cached result
//! let func_hash = ContentHash::from_function("my_func", mir_body);
//! if let Some(cached) = cache.get_verification_result(&func_hash)? {
//!     println!("Cached result: {:?}", cached.outcome);
//!     return Ok(cached);
//! }
//!
//! // Verify and cache
//! let result = verify_function(...).await?;
//! cache.store_verification_result(&func_hash, &result, &dependencies)?;
//! ```

use crate::clause_db::ClauseDbError;
use crate::content_hash::ContentHash;
use rusqlite::{params, Connection, Result as SqlResult};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tracing::{debug, info};

/// Row type for cached CHC results from SQLite
/// Fields: outcome, invariant, counterexample, error_message, duration_ms, backend, computed_at
type CachedResultRow = (
    String,
    Option<String>,
    Option<String>,
    Option<String>,
    i64,
    String,
    i64,
);

/// Verification outcome for a function
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CachedOutcome {
    /// Property proven with an invariant
    Proven,
    /// Property violated with a counterexample trace
    Disproven,
    /// Verification inconclusive (timeout, resource limit, etc.)
    Unknown,
    /// Verification error (solver crash, etc.)
    Error,
}

/// A cached CHC verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedChcResult {
    /// The outcome of verification
    pub outcome: CachedOutcome,
    /// Discovered invariant (if proven), in SMT-LIB2 format
    pub invariant: Option<String>,
    /// Counterexample trace (if disproven), in JSON format
    pub counterexample: Option<String>,
    /// Error message (if error)
    pub error_message: Option<String>,
    /// Verification duration in milliseconds
    pub duration_ms: u64,
    /// Backend used (z3, z4)
    pub backend: String,
    /// Timestamp when this result was computed
    pub computed_at: SystemTime,
}

/// Function dependency information
#[derive(Debug, Clone)]
pub struct FunctionDependency {
    /// Name of the function being depended on
    pub callee: String,
    /// Hash of the callee at the time of verification
    pub callee_hash: ContentHash,
}

impl FunctionDependency {
    /// Create a new function dependency
    pub fn new(callee: impl Into<String>, callee_hash: ContentHash) -> Self {
        Self {
            callee: callee.into(),
            callee_hash,
        }
    }
}

/// Errors that can occur with the CHC cache
#[derive(Debug, Error)]
pub enum ChcCacheError {
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Hex decode error: {0}")]
    HexDecode(#[from] hex::FromHexError),

    #[error("Cache miss for function {0}")]
    CacheMiss(String),

    #[error("Invalid cached data: {0}")]
    InvalidData(String),
}

impl From<ClauseDbError> for ChcCacheError {
    fn from(err: ClauseDbError) -> Self {
        match err {
            ClauseDbError::Database(e) => ChcCacheError::Database(e),
            ClauseDbError::HexDecode(e) => ChcCacheError::HexDecode(e),
            _ => ChcCacheError::InvalidData(err.to_string()),
        }
    }
}

/// SQLite-based CHC verification cache
pub struct ChcVerificationCache {
    conn: Connection,
}

impl ChcVerificationCache {
    /// Open or create a CHC verification cache at the given path
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ChcCacheError> {
        let conn = Connection::open(path)?;
        let cache = Self { conn };
        cache.init_schema()?;
        Ok(cache)
    }

    /// Create an in-memory cache (for testing)
    pub fn in_memory() -> Result<Self, ChcCacheError> {
        let conn = Connection::open_in_memory()?;
        let cache = Self { conn };
        cache.init_schema()?;
        Ok(cache)
    }

    /// Initialize the database schema
    fn init_schema(&self) -> SqlResult<()> {
        self.conn.execute_batch(
            "
            -- CHC verification results cache
            CREATE TABLE IF NOT EXISTS chc_results (
                function_hash TEXT PRIMARY KEY,
                function_name TEXT NOT NULL,
                outcome TEXT NOT NULL,
                invariant TEXT,
                counterexample TEXT,
                error_message TEXT,
                duration_ms INTEGER NOT NULL,
                backend TEXT NOT NULL,
                computed_at INTEGER NOT NULL,
                config_hash TEXT NOT NULL
            );

            -- Function dependency tracking
            CREATE TABLE IF NOT EXISTS function_dependencies (
                caller_hash TEXT NOT NULL,
                callee_name TEXT NOT NULL,
                callee_hash TEXT NOT NULL,
                PRIMARY KEY (caller_hash, callee_name),
                FOREIGN KEY (caller_hash) REFERENCES chc_results(function_hash) ON DELETE CASCADE
            );

            -- Index for reverse dependency lookup (who depends on this function?)
            CREATE INDEX IF NOT EXISTS idx_deps_callee ON function_dependencies(callee_name);

            -- Function metadata (source location, MIR size, etc.)
            CREATE TABLE IF NOT EXISTS function_metadata (
                function_hash TEXT PRIMARY KEY,
                function_name TEXT NOT NULL,
                source_file TEXT,
                line_number INTEGER,
                mir_size INTEGER,
                last_verified INTEGER NOT NULL
            );

            -- Cache statistics
            CREATE TABLE IF NOT EXISTS cache_stats (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            -- Initialize stats
            INSERT OR IGNORE INTO cache_stats (key, value) VALUES
                ('hits', '0'),
                ('misses', '0'),
                ('invalidations', '0');
            ",
        )?;

        info!("CHC verification cache schema initialized");
        Ok(())
    }

    /// Store a verification result
    pub fn store_result(
        &self,
        function_hash: &ContentHash,
        function_name: &str,
        result: &CachedChcResult,
        config_hash: &ContentHash,
        dependencies: &[FunctionDependency],
    ) -> Result<(), ChcCacheError> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let outcome_str = match result.outcome {
            CachedOutcome::Proven => "proven",
            CachedOutcome::Disproven => "disproven",
            CachedOutcome::Unknown => "unknown",
            CachedOutcome::Error => "error",
        };

        // Store the result
        self.conn.execute(
            "
            INSERT OR REPLACE INTO chc_results
            (function_hash, function_name, outcome, invariant, counterexample, error_message,
             duration_ms, backend, computed_at, config_hash)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
            ",
            params![
                function_hash.to_hex(),
                function_name,
                outcome_str,
                result.invariant,
                result.counterexample,
                result.error_message,
                result.duration_ms as i64,
                result.backend,
                now,
                config_hash.to_hex(),
            ],
        )?;

        // Store dependencies
        let func_hex = function_hash.to_hex();
        self.conn.execute(
            "DELETE FROM function_dependencies WHERE caller_hash = ?1",
            [&func_hex],
        )?;

        for dep in dependencies {
            self.conn.execute(
                "
                INSERT INTO function_dependencies (caller_hash, callee_name, callee_hash)
                VALUES (?1, ?2, ?3)
                ",
                params![&func_hex, &dep.callee, dep.callee_hash.to_hex()],
            )?;
        }

        debug!(
            "Stored CHC result for {} ({}) with {} dependencies",
            function_name,
            function_hash.short(),
            dependencies.len()
        );

        Ok(())
    }

    /// Get a cached verification result
    pub fn get_result(
        &self,
        function_hash: &ContentHash,
    ) -> Result<Option<CachedChcResult>, ChcCacheError> {
        let result: SqlResult<CachedResultRow> = self.conn.query_row(
            "
                SELECT outcome, invariant, counterexample, error_message, duration_ms, backend, computed_at
                FROM chc_results
                WHERE function_hash = ?1
                ",
            [function_hash.to_hex()],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                ))
            },
        );

        match result {
            Ok((
                outcome_str,
                invariant,
                counterexample,
                error_message,
                duration_ms,
                backend,
                computed_at,
            )) => {
                let outcome = match outcome_str.as_str() {
                    "proven" => CachedOutcome::Proven,
                    "disproven" => CachedOutcome::Disproven,
                    "unknown" => CachedOutcome::Unknown,
                    "error" => CachedOutcome::Error,
                    _ => {
                        return Err(ChcCacheError::InvalidData(format!(
                            "Unknown outcome: {outcome_str}"
                        )))
                    }
                };

                // Update hit counter
                self.increment_stat("hits")?;

                Ok(Some(CachedChcResult {
                    outcome,
                    invariant,
                    counterexample,
                    error_message,
                    duration_ms: duration_ms as u64,
                    backend,
                    computed_at: UNIX_EPOCH + Duration::from_secs(computed_at as u64),
                }))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => {
                self.increment_stat("misses")?;
                Ok(None)
            }
            Err(e) => Err(ChcCacheError::Database(e)),
        }
    }

    /// Check if a function's cache is valid given current dependency hashes
    ///
    /// Returns true if the function has a cached result AND all its dependencies
    /// still have the same hashes.
    pub fn is_cache_valid(
        &self,
        function_hash: &ContentHash,
        current_dep_hashes: &HashMap<String, ContentHash>,
    ) -> Result<bool, ChcCacheError> {
        // First check if we have a cached result
        let exists: bool = self
            .conn
            .query_row(
                "SELECT 1 FROM chc_results WHERE function_hash = ?1",
                [function_hash.to_hex()],
                |_| Ok(true),
            )
            .unwrap_or(false);

        if !exists {
            return Ok(false);
        }

        // Get stored dependencies
        let mut stmt = self.conn.prepare(
            "SELECT callee_name, callee_hash FROM function_dependencies WHERE caller_hash = ?1",
        )?;

        let deps = stmt
            .query_map([function_hash.to_hex()], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?
            .collect::<SqlResult<Vec<_>>>()?;

        // Check each dependency
        for (callee_name, stored_hash_hex) in deps {
            if let Some(current_hash) = current_dep_hashes.get(&callee_name) {
                if current_hash.to_hex() != stored_hash_hex {
                    debug!(
                        "Cache invalid: dependency {} changed from {} to {}",
                        callee_name,
                        &stored_hash_hex[..8],
                        current_hash.short()
                    );
                    return Ok(false);
                }
            } else {
                // Dependency no longer exists
                debug!("Cache invalid: dependency {} no longer exists", callee_name);
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Invalidate cache for a function and all functions that depend on it
    ///
    /// Returns the number of invalidated entries
    pub fn invalidate_function(&self, function_name: &str) -> Result<usize, ChcCacheError> {
        // Find all functions that depend on this one (directly or transitively)
        let to_invalidate = self.find_dependents(function_name)?;

        let mut invalidated = 0;

        // Invalidate the function itself
        let deleted = self.conn.execute(
            "DELETE FROM chc_results WHERE function_name = ?1",
            [function_name],
        )?;
        invalidated += deleted;

        // Invalidate all dependents
        for dependent in &to_invalidate {
            let deleted = self.conn.execute(
                "DELETE FROM chc_results WHERE function_name = ?1",
                [dependent],
            )?;
            invalidated += deleted;
        }

        if invalidated > 0 {
            self.increment_stat_by("invalidations", invalidated)?;
            info!(
                "Invalidated {} cache entries for {} and its dependents",
                invalidated, function_name
            );
        }

        Ok(invalidated)
    }

    /// Find all functions that depend on the given function (transitively)
    fn find_dependents(&self, function_name: &str) -> Result<HashSet<String>, ChcCacheError> {
        let mut dependents = HashSet::new();
        let mut to_process = vec![function_name.to_string()];

        while let Some(func) = to_process.pop() {
            // Find direct dependents
            let mut stmt = self.conn.prepare(
                "
                SELECT DISTINCT r.function_name
                FROM function_dependencies d
                JOIN chc_results r ON d.caller_hash = r.function_hash
                WHERE d.callee_name = ?1
                ",
            )?;

            let direct: Vec<String> = stmt
                .query_map([&func], |row| row.get(0))?
                .collect::<SqlResult<Vec<_>>>()?;

            for dep in direct {
                if !dependents.contains(&dep) {
                    dependents.insert(dep.clone());
                    to_process.push(dep);
                }
            }
        }

        Ok(dependents)
    }

    /// Get cache statistics
    pub fn stats(&self) -> Result<CacheStats, ChcCacheError> {
        let cached_count: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM chc_results", [], |row| row.get(0))?;

        let proven_count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM chc_results WHERE outcome = 'proven'",
            [],
            |row| row.get(0),
        )?;

        let disproven_count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM chc_results WHERE outcome = 'disproven'",
            [],
            |row| row.get(0),
        )?;

        let hits = self.get_stat("hits")?;
        let misses = self.get_stat("misses")?;
        let invalidations = self.get_stat("invalidations")?;

        let total_duration: i64 = self.conn.query_row(
            "SELECT COALESCE(SUM(duration_ms), 0) FROM chc_results",
            [],
            |row| row.get(0),
        )?;

        Ok(CacheStats {
            cached_functions: cached_count as u64,
            proven_functions: proven_count as u64,
            disproven_functions: disproven_count as u64,
            cache_hits: hits,
            cache_misses: misses,
            invalidations,
            total_verification_time_ms: total_duration as u64,
        })
    }

    /// Clear all cached results
    pub fn clear(&self) -> Result<(), ChcCacheError> {
        self.conn.execute("DELETE FROM chc_results", [])?;
        self.conn.execute("DELETE FROM function_dependencies", [])?;
        self.conn.execute("DELETE FROM function_metadata", [])?;
        info!("CHC verification cache cleared");
        Ok(())
    }

    /// Get functions that need re-verification based on changed functions
    pub fn get_functions_to_verify(
        &self,
        changed_functions: &[String],
    ) -> Result<HashSet<String>, ChcCacheError> {
        let mut to_verify = HashSet::new();

        for func in changed_functions {
            to_verify.insert(func.clone());
            let dependents = self.find_dependents(func)?;
            to_verify.extend(dependents);
        }

        Ok(to_verify)
    }

    /// Store function metadata for faster lookups
    pub fn store_metadata(
        &self,
        function_hash: &ContentHash,
        function_name: &str,
        source_file: Option<&str>,
        line_number: Option<u32>,
        mir_size: Option<u32>,
    ) -> Result<(), ChcCacheError> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        self.conn.execute(
            "
            INSERT OR REPLACE INTO function_metadata
            (function_hash, function_name, source_file, line_number, mir_size, last_verified)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6)
            ",
            params![
                function_hash.to_hex(),
                function_name,
                source_file,
                line_number,
                mir_size,
                now,
            ],
        )?;

        Ok(())
    }

    fn increment_stat(&self, key: &str) -> Result<(), ChcCacheError> {
        self.increment_stat_by(key, 1)
    }

    fn increment_stat_by(&self, key: &str, amount: usize) -> Result<(), ChcCacheError> {
        self.conn.execute(
            "UPDATE cache_stats SET value = CAST((CAST(value AS INTEGER) + ?1) AS TEXT) WHERE key = ?2",
            params![amount as i64, key],
        )?;
        Ok(())
    }

    fn get_stat(&self, key: &str) -> Result<u64, ChcCacheError> {
        let value: String = self.conn.query_row(
            "SELECT value FROM cache_stats WHERE key = ?1",
            [key],
            |row| row.get(0),
        )?;
        Ok(value.parse().unwrap_or(0))
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of cached function results
    pub cached_functions: u64,
    /// Number of proven functions
    pub proven_functions: u64,
    /// Number of disproven functions
    pub disproven_functions: u64,
    /// Number of cache hits
    pub cache_hits: u64,
    /// Number of cache misses
    pub cache_misses: u64,
    /// Number of cache invalidations
    pub invalidations: u64,
    /// Total verification time saved by caching (ms)
    pub total_verification_time_ms: u64,
}

impl CacheStats {
    /// Calculate cache hit rate as a percentage
    pub fn hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            (self.cache_hits as f64 / total as f64) * 100.0
        }
    }

    /// Estimated time saved by cache hits (assuming each hit saves average verification time)
    pub fn estimated_time_saved_ms(&self) -> u64 {
        if self.cached_functions == 0 {
            return 0;
        }
        let avg_time = self.total_verification_time_ms / self.cached_functions;
        self.cache_hits * avg_time
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config_hash() -> ContentHash {
        ContentHash::from_context("test_config", &[])
    }

    #[test]
    fn test_create_cache() {
        let cache = ChcVerificationCache::in_memory().unwrap();
        let stats = cache.stats().unwrap();
        assert_eq!(stats.cached_functions, 0);
    }

    #[test]
    fn test_store_and_retrieve_result() {
        let cache = ChcVerificationCache::in_memory().unwrap();

        let func_hash = ContentHash::from_function("test_func", "fn test() {}");
        let config_hash = test_config_hash();

        let result = CachedChcResult {
            outcome: CachedOutcome::Proven,
            invariant: Some("(>= x 0)".to_string()),
            counterexample: None,
            error_message: None,
            duration_ms: 100,
            backend: "z3".to_string(),
            computed_at: SystemTime::now(),
        };

        cache
            .store_result(&func_hash, "test_func", &result, &config_hash, &[])
            .unwrap();

        let cached = cache.get_result(&func_hash).unwrap();
        assert!(cached.is_some());
        let cached = cached.unwrap();
        assert_eq!(cached.outcome, CachedOutcome::Proven);
        assert_eq!(cached.invariant, Some("(>= x 0)".to_string()));
    }

    #[test]
    fn test_store_with_dependencies() {
        let cache = ChcVerificationCache::in_memory().unwrap();

        let func_hash = ContentHash::from_function("caller", "fn caller() { callee(); }");
        let callee_hash = ContentHash::from_function("callee", "fn callee() {}");
        let config_hash = test_config_hash();

        let result = CachedChcResult {
            outcome: CachedOutcome::Proven,
            invariant: None,
            counterexample: None,
            error_message: None,
            duration_ms: 50,
            backend: "z3".to_string(),
            computed_at: SystemTime::now(),
        };

        let deps = vec![FunctionDependency {
            callee: "callee".to_string(),
            callee_hash: callee_hash.clone(),
        }];

        cache
            .store_result(&func_hash, "caller", &result, &config_hash, &deps)
            .unwrap();

        // Check cache validity with correct hash
        let mut current_hashes = HashMap::new();
        current_hashes.insert("callee".to_string(), callee_hash.clone());
        assert!(cache.is_cache_valid(&func_hash, &current_hashes).unwrap());

        // Check cache validity with changed hash
        let changed_hash = ContentHash::from_function("callee", "fn callee() { /* changed */ }");
        current_hashes.insert("callee".to_string(), changed_hash);
        assert!(!cache.is_cache_valid(&func_hash, &current_hashes).unwrap());
    }

    #[test]
    fn test_invalidate_function() {
        let cache = ChcVerificationCache::in_memory().unwrap();

        let func_hash = ContentHash::from_function("func", "fn func() {}");
        let config_hash = test_config_hash();

        let result = CachedChcResult {
            outcome: CachedOutcome::Proven,
            invariant: None,
            counterexample: None,
            error_message: None,
            duration_ms: 50,
            backend: "z3".to_string(),
            computed_at: SystemTime::now(),
        };

        cache
            .store_result(&func_hash, "func", &result, &config_hash, &[])
            .unwrap();

        assert!(cache.get_result(&func_hash).unwrap().is_some());

        let invalidated = cache.invalidate_function("func").unwrap();
        assert_eq!(invalidated, 1);

        assert!(cache.get_result(&func_hash).unwrap().is_none());
    }

    #[test]
    fn test_transitive_invalidation() {
        let cache = ChcVerificationCache::in_memory().unwrap();
        let config_hash = test_config_hash();

        // Create a dependency chain: a -> b -> c
        let hash_a = ContentHash::from_function("func_a", "fn func_a() { func_b(); }");
        let hash_b = ContentHash::from_function("func_b", "fn func_b() { func_c(); }");
        let hash_c = ContentHash::from_function("func_c", "fn func_c() {}");

        let result = CachedChcResult {
            outcome: CachedOutcome::Proven,
            invariant: None,
            counterexample: None,
            error_message: None,
            duration_ms: 10,
            backend: "z3".to_string(),
            computed_at: SystemTime::now(),
        };

        // Store c (no deps)
        cache
            .store_result(&hash_c, "func_c", &result, &config_hash, &[])
            .unwrap();

        // Store b (depends on c)
        let deps_b = vec![FunctionDependency {
            callee: "func_c".to_string(),
            callee_hash: hash_c.clone(),
        }];
        cache
            .store_result(&hash_b, "func_b", &result, &config_hash, &deps_b)
            .unwrap();

        // Store a (depends on b)
        let deps_a = vec![FunctionDependency {
            callee: "func_b".to_string(),
            callee_hash: hash_b.clone(),
        }];
        cache
            .store_result(&hash_a, "func_a", &result, &config_hash, &deps_a)
            .unwrap();

        // Verify all are cached
        assert!(cache.get_result(&hash_a).unwrap().is_some());
        assert!(cache.get_result(&hash_b).unwrap().is_some());
        assert!(cache.get_result(&hash_c).unwrap().is_some());

        // Invalidate c - should invalidate a and b transitively
        let invalidated = cache.invalidate_function("func_c").unwrap();
        assert_eq!(invalidated, 3);

        // All should be invalidated
        assert!(cache.get_result(&hash_a).unwrap().is_none());
        assert!(cache.get_result(&hash_b).unwrap().is_none());
        assert!(cache.get_result(&hash_c).unwrap().is_none());
    }

    #[test]
    fn test_get_functions_to_verify() {
        let cache = ChcVerificationCache::in_memory().unwrap();
        let config_hash = test_config_hash();

        let hash_a = ContentHash::from_function("func_a", "fn func_a() { func_b(); }");
        let hash_b = ContentHash::from_function("func_b", "fn func_b() {}");

        let result = CachedChcResult {
            outcome: CachedOutcome::Proven,
            invariant: None,
            counterexample: None,
            error_message: None,
            duration_ms: 10,
            backend: "z3".to_string(),
            computed_at: SystemTime::now(),
        };

        // Store b
        cache
            .store_result(&hash_b, "func_b", &result, &config_hash, &[])
            .unwrap();

        // Store a (depends on b)
        let deps_a = vec![FunctionDependency {
            callee: "func_b".to_string(),
            callee_hash: hash_b.clone(),
        }];
        cache
            .store_result(&hash_a, "func_a", &result, &config_hash, &deps_a)
            .unwrap();

        // If b changed, we need to verify both a and b
        let to_verify = cache
            .get_functions_to_verify(&["func_b".to_string()])
            .unwrap();
        assert!(to_verify.contains("func_b"));
        assert!(to_verify.contains("func_a"));
        assert_eq!(to_verify.len(), 2);
    }

    #[test]
    fn test_cache_stats() {
        let cache = ChcVerificationCache::in_memory().unwrap();
        let config_hash = test_config_hash();

        let hash1 = ContentHash::from_function("func1", "fn func1() {}");
        let hash2 = ContentHash::from_function("func2", "fn func2() {}");
        let hash_missing = ContentHash::from_function("missing", "fn missing() {}");

        let result_proven = CachedChcResult {
            outcome: CachedOutcome::Proven,
            invariant: Some("(>= x 0)".to_string()),
            counterexample: None,
            error_message: None,
            duration_ms: 100,
            backend: "z3".to_string(),
            computed_at: SystemTime::now(),
        };

        let result_disproven = CachedChcResult {
            outcome: CachedOutcome::Disproven,
            invariant: None,
            counterexample: Some("{}".to_string()),
            error_message: None,
            duration_ms: 50,
            backend: "z3".to_string(),
            computed_at: SystemTime::now(),
        };

        cache
            .store_result(&hash1, "func1", &result_proven, &config_hash, &[])
            .unwrap();
        cache
            .store_result(&hash2, "func2", &result_disproven, &config_hash, &[])
            .unwrap();

        // Generate some hits and misses
        cache.get_result(&hash1).unwrap(); // hit
        cache.get_result(&hash2).unwrap(); // hit
        cache.get_result(&hash_missing).unwrap(); // miss

        let stats = cache.stats().unwrap();
        assert_eq!(stats.cached_functions, 2);
        assert_eq!(stats.proven_functions, 1);
        assert_eq!(stats.disproven_functions, 1);
        assert_eq!(stats.cache_hits, 2);
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.total_verification_time_ms, 150);
    }

    #[test]
    fn test_hit_rate() {
        let stats = CacheStats {
            cached_functions: 10,
            proven_functions: 8,
            disproven_functions: 2,
            cache_hits: 80,
            cache_misses: 20,
            invalidations: 5,
            total_verification_time_ms: 1000,
        };

        assert!((stats.hit_rate() - 80.0).abs() < 0.01);
    }

    #[test]
    fn test_clear_cache() {
        let cache = ChcVerificationCache::in_memory().unwrap();
        let config_hash = test_config_hash();

        let hash = ContentHash::from_function("func", "fn func() {}");
        let result = CachedChcResult {
            outcome: CachedOutcome::Proven,
            invariant: None,
            counterexample: None,
            error_message: None,
            duration_ms: 10,
            backend: "z3".to_string(),
            computed_at: SystemTime::now(),
        };

        cache
            .store_result(&hash, "func", &result, &config_hash, &[])
            .unwrap();
        assert!(cache.get_result(&hash).unwrap().is_some());

        cache.clear().unwrap();
        // Note: get_result still records misses, but the cache is empty
        let stats = cache.stats().unwrap();
        assert_eq!(stats.cached_functions, 0);
    }

    // ==================== Mutation Coverage Tests ====================

    /// Test From<ClauseDbError> for ChcCacheError - Database variant
    #[test]
    fn test_from_clause_db_error_database() {
        use crate::clause_db::ClauseDbError;

        let db_error = rusqlite::Error::InvalidQuery;
        let clause_err = ClauseDbError::Database(db_error);
        let chc_err: ChcCacheError = clause_err.into();

        // Verify it's a Database variant (not InvalidData)
        match chc_err {
            ChcCacheError::Database(_) => {} // Expected
            _ => panic!("Expected Database variant, got {:?}", chc_err),
        }
    }

    /// Test From<ClauseDbError> for ChcCacheError - HexDecode variant
    #[test]
    fn test_from_clause_db_error_hex_decode() {
        use crate::clause_db::ClauseDbError;

        // Create a hex decode error
        let hex_err = hex::FromHexError::InvalidHexCharacter { c: 'Z', index: 0 };
        let clause_err = ClauseDbError::HexDecode(hex_err);
        let chc_err: ChcCacheError = clause_err.into();

        // Verify it's a HexDecode variant (not InvalidData)
        match chc_err {
            ChcCacheError::HexDecode(_) => {} // Expected
            _ => panic!("Expected HexDecode variant, got {:?}", chc_err),
        }
    }

    /// Test that unknown outcome returns correct CachedOutcome::Unknown
    #[test]
    fn test_get_result_unknown_outcome() {
        let cache = ChcVerificationCache::in_memory().unwrap();
        let config_hash = test_config_hash();
        let func_hash = ContentHash::from_function("unknown_func", "fn unknown_func() {}");

        let result = CachedChcResult {
            outcome: CachedOutcome::Unknown,
            invariant: None,
            counterexample: None,
            error_message: Some("timeout".to_string()),
            duration_ms: 60000,
            backend: "z3".to_string(),
            computed_at: SystemTime::now(),
        };

        cache
            .store_result(&func_hash, "unknown_func", &result, &config_hash, &[])
            .unwrap();

        let cached = cache.get_result(&func_hash).unwrap().unwrap();
        // This catches the "delete match arm 'unknown'" mutant
        assert_eq!(cached.outcome, CachedOutcome::Unknown);
        assert_eq!(cached.error_message, Some("timeout".to_string()));
    }

    /// Test that error outcome returns correct CachedOutcome::Error
    #[test]
    fn test_get_result_error_outcome() {
        let cache = ChcVerificationCache::in_memory().unwrap();
        let config_hash = test_config_hash();
        let func_hash = ContentHash::from_function("error_func", "fn error_func() {}");

        let result = CachedChcResult {
            outcome: CachedOutcome::Error,
            invariant: None,
            counterexample: None,
            error_message: Some("solver crashed".to_string()),
            duration_ms: 1000,
            backend: "z3".to_string(),
            computed_at: SystemTime::now(),
        };

        cache
            .store_result(&func_hash, "error_func", &result, &config_hash, &[])
            .unwrap();

        let cached = cache.get_result(&func_hash).unwrap().unwrap();
        // This catches the "delete match arm 'error'" mutant
        assert_eq!(cached.outcome, CachedOutcome::Error);
        assert_eq!(cached.error_message, Some("solver crashed".to_string()));
    }

    /// Test duration_ms is correctly computed (catches + to - or * mutant)
    #[test]
    fn test_get_result_duration_ms_computation() {
        let cache = ChcVerificationCache::in_memory().unwrap();
        let config_hash = test_config_hash();
        let func_hash = ContentHash::from_function("duration_func", "fn duration_func() {}");

        let result = CachedChcResult {
            outcome: CachedOutcome::Proven,
            invariant: None,
            counterexample: None,
            error_message: None,
            duration_ms: 12345,
            backend: "z3".to_string(),
            computed_at: SystemTime::now(),
        };

        cache
            .store_result(&func_hash, "duration_func", &result, &config_hash, &[])
            .unwrap();

        let cached = cache.get_result(&func_hash).unwrap().unwrap();
        // The computed_at timestamp is stored as seconds since epoch
        // and we add seconds to UNIX_EPOCH. This test ensures the + is correct.
        assert_eq!(cached.duration_ms, 12345);

        // Verify computed_at is reasonable (within last minute)
        let now = SystemTime::now();
        let elapsed = now.duration_since(cached.computed_at).unwrap_or_default();
        assert!(
            elapsed.as_secs() < 60,
            "computed_at should be recent, but was {:?} seconds ago",
            elapsed.as_secs()
        );
    }

    /// Test invalidate_function returns 0 when nothing to invalidate (catches > vs >= mutant)
    #[test]
    fn test_invalidate_function_returns_zero_when_empty() {
        let cache = ChcVerificationCache::in_memory().unwrap();

        // Invalidate non-existent function - should return 0
        let invalidated = cache.invalidate_function("nonexistent").unwrap();
        assert_eq!(
            invalidated, 0,
            "Should return 0 when no entries to invalidate"
        );
    }

    /// Test invalidate_function logs only when invalidated > 0 (catches > vs < or >= mutants)
    #[test]
    fn test_invalidate_function_boundary_one() {
        let cache = ChcVerificationCache::in_memory().unwrap();
        let config_hash = test_config_hash();

        let func_hash = ContentHash::from_function("single_func", "fn single_func() {}");
        let result = CachedChcResult {
            outcome: CachedOutcome::Proven,
            invariant: None,
            counterexample: None,
            error_message: None,
            duration_ms: 10,
            backend: "z3".to_string(),
            computed_at: SystemTime::now(),
        };

        cache
            .store_result(&func_hash, "single_func", &result, &config_hash, &[])
            .unwrap();

        // This returns exactly 1 - tests the > 0 boundary
        let invalidated = cache.invalidate_function("single_func").unwrap();
        assert_eq!(
            invalidated, 1,
            "Should return exactly 1 when one entry invalidated"
        );

        // Verify stats were updated
        let stats = cache.stats().unwrap();
        assert_eq!(
            stats.invalidations, 1,
            "Invalidations stat should be incremented"
        );
    }

    /// Test store_metadata actually stores data (catches returning Ok(()) without storing)
    #[test]
    fn test_store_metadata_actually_stores() {
        let cache = ChcVerificationCache::in_memory().unwrap();
        let func_hash = ContentHash::from_function("meta_func", "fn meta_func() {}");

        cache
            .store_metadata(
                &func_hash,
                "meta_func",
                Some("src/lib.rs"),
                Some(42),
                Some(100),
            )
            .unwrap();

        // Verify the metadata was actually stored by querying the database directly
        let count: i64 = cache
            .conn
            .query_row(
                "SELECT COUNT(*) FROM function_metadata WHERE function_name = ?1",
                ["meta_func"],
                |row| row.get(0),
            )
            .unwrap();

        assert_eq!(count, 1, "store_metadata should actually insert a row");

        // Verify the values
        let (source_file, line_number, mir_size): (Option<String>, Option<i64>, Option<i64>) =
            cache
                .conn
                .query_row(
                    "SELECT source_file, line_number, mir_size FROM function_metadata WHERE function_name = ?1",
                    ["meta_func"],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                )
                .unwrap();

        assert_eq!(source_file, Some("src/lib.rs".to_string()));
        assert_eq!(line_number, Some(42));
        assert_eq!(mir_size, Some(100));
    }

    /// Test estimated_time_saved_ms returns 0 when cached_functions is 0
    #[test]
    fn test_estimated_time_saved_zero_cached_functions() {
        let stats = CacheStats {
            cached_functions: 0,
            proven_functions: 0,
            disproven_functions: 0,
            cache_hits: 100,
            cache_misses: 0,
            invalidations: 0,
            total_verification_time_ms: 0,
        };

        // Should return 0, not panic on division by zero
        assert_eq!(
            stats.estimated_time_saved_ms(),
            0,
            "Should return 0 when cached_functions is 0"
        );
    }

    /// Test estimated_time_saved_ms computes correctly (catches == vs != and arithmetic mutants)
    #[test]
    fn test_estimated_time_saved_ms_computation() {
        let stats = CacheStats {
            cached_functions: 10,
            proven_functions: 8,
            disproven_functions: 2,
            cache_hits: 20,
            cache_misses: 5,
            invalidations: 0,
            total_verification_time_ms: 1000, // avg = 100ms per function
        };

        // Expected: cache_hits * (total_verification_time_ms / cached_functions)
        // = 20 * (1000 / 10) = 20 * 100 = 2000
        let saved = stats.estimated_time_saved_ms();
        assert_eq!(saved, 2000, "Expected 20 * (1000/10) = 2000, got {}", saved);
    }

    /// Test estimated_time_saved_ms with different values to catch / vs % and * vs + mutants
    #[test]
    fn test_estimated_time_saved_ms_arithmetic_correctness() {
        // Use values where wrong operations would give different results
        let stats = CacheStats {
            cached_functions: 4,
            proven_functions: 4,
            disproven_functions: 0,
            cache_hits: 3,
            cache_misses: 1,
            invalidations: 0,
            total_verification_time_ms: 400, // avg = 100ms per function
        };

        // Correct: 3 * (400 / 4) = 3 * 100 = 300
        // If / became %: 3 * (400 % 4) = 3 * 0 = 0
        // If / became *: 3 * (400 * 4) = 3 * 1600 = 4800
        // If * became +: 3 + 100 = 103
        // If * became /: 3 / 100 = 0
        let saved = stats.estimated_time_saved_ms();
        assert_eq!(
            saved, 300,
            "Arithmetic must be correct: 3 * (400/4) = 300, got {}",
            saved
        );
    }
}
