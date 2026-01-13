//! CaDiCaL SAT solver wrapper
//!
//! CaDiCaL is a high-performance CDCL SAT solver that is already used by Kani.
//! This wrapper allows us to run CaDiCaL directly for portfolio solving.
//!
//! # Learned Clause Export
//!
//! CaDiCaL can output DRAT proofs containing learned clauses. Use `solve_with_proof`
//! to get both the solving result and a path to the DRAT proof file.

use crate::drat::{extract_learned_clauses, filter_learned_clauses, DratParser};
use crate::solver::{
    util::{extract_model, extract_number},
    BoxedSolver, Solver, SolverCapability, SolverConfig, SolverError, SolverInfo, SolverOutput,
    SolverResult, SolverStats,
};
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Instant;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::time::timeout;
use tracing::{debug, info, warn};

/// CaDiCaL SAT solver
pub struct CaDiCaL {
    info: SolverInfo,
    binary_path: Option<String>,
}

impl CaDiCaL {
    /// Create a new CaDiCaL instance
    pub fn new() -> Self {
        Self {
            info: SolverInfo {
                id: "cadical".to_string(),
                name: "CaDiCaL".to_string(),
                version: "unknown".to_string(),
                capabilities: vec![
                    SolverCapability::Sat,
                    SolverCapability::Incremental,
                    SolverCapability::Proofs,
                ],
                available: false,
            },
            binary_path: None,
        }
    }

    /// Create with a specific binary path
    pub fn with_binary(path: impl Into<String>) -> Self {
        let mut solver = Self::new();
        solver.binary_path = Some(path.into());
        solver
    }

    /// Detect and initialize CaDiCaL
    pub async fn detect() -> Option<Self> {
        let mut solver = Self::new();

        // Try to find cadical binary
        let binary = solver.find_binary().await?;
        solver.binary_path = Some(binary);

        // Get version
        if let Some(version) = solver.get_version().await {
            solver.info.version = version;
        }

        solver.info.available = true;
        Some(solver)
    }

    async fn find_binary(&self) -> Option<String> {
        if let Some(path) = &self.binary_path {
            return Some(path.clone());
        }

        // Try common binary names
        for name in &["cadical", "CaDiCaL"] {
            if let Some(path) = crate::find_executable(name) {
                return Some(path.to_string_lossy().to_string());
            }
        }

        None
    }

    async fn get_version(&self) -> Option<String> {
        let binary = self.binary_path.as_ref()?;

        let output = Command::new(binary).arg("--version").output().await.ok()?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        // CaDiCaL version output format: "cadical-1.x.x ..."
        stdout.lines().next().map(|s| s.trim().to_string())
    }

    fn parse_output(
        &self,
        stdout: &str,
        stderr: &str,
        duration: std::time::Duration,
    ) -> SolverOutput {
        let mut stats = SolverStats {
            solve_time: duration,
            ..Default::default()
        };

        // Parse CaDiCaL statistics from output
        for line in stdout.lines().chain(stderr.lines()) {
            if line.contains("conflicts:") {
                if let Some(n) = extract_number(line) {
                    stats.conflicts = Some(n);
                }
            } else if line.contains("decisions:") {
                if let Some(n) = extract_number(line) {
                    stats.decisions = Some(n);
                }
            } else if line.contains("propagations:") {
                if let Some(n) = extract_number(line) {
                    stats.propagations = Some(n);
                }
            }
        }

        // Determine result from output
        let result = if stdout.contains("s SATISFIABLE") || stderr.contains("s SATISFIABLE") {
            // Extract model if present
            let model = extract_model(stdout);
            SolverResult::Sat { model }
        } else if stdout.contains("s UNSATISFIABLE") || stderr.contains("s UNSATISFIABLE") {
            SolverResult::Unsat { proof: None }
        } else {
            SolverResult::Unknown {
                reason: "No definitive result in output".to_string(),
            }
        };

        SolverOutput {
            result,
            stats,
            raw_output: Some(format!("STDOUT:\n{stdout}\nSTDERR:\n{stderr}")),
        }
    }

    /// Solve a DIMACS file and output learned clauses to a DRAT proof file
    ///
    /// Returns the solver output and the path to the proof file.
    /// The proof file contains DRAT format learned clauses.
    pub async fn solve_with_proof(
        &self,
        dimacs_path: &Path,
        config: &SolverConfig,
        proof_path: Option<&Path>,
    ) -> Result<(SolverOutput, Option<PathBuf>), SolverError> {
        let binary = self
            .binary_path
            .as_ref()
            .ok_or_else(|| SolverError::NotFound("CaDiCaL binary not found".to_string()))?;

        if !dimacs_path.exists() {
            return Err(SolverError::InvalidInput(format!(
                "DIMACS file not found: {}",
                dimacs_path.display()
            )));
        }

        // Determine proof output path
        let proof_file = proof_path.map(|p| p.to_path_buf()).unwrap_or_else(|| {
            let mut path = dimacs_path.to_path_buf();
            path.set_extension("drat");
            path
        });

        let mut cmd = Command::new(binary);
        cmd.arg(dimacs_path);
        cmd.arg(&proof_file);
        cmd.arg("--binary=false"); // Use text format for easier parsing

        // Add configuration options
        if let Some(seed) = config.seed {
            cmd.arg("--seed").arg(seed.to_string());
        }

        for (key, value) in &config.options {
            cmd.arg(format!("--{key}={value}"));
        }

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        debug!("Running CaDiCaL with proof: {:?}", cmd);
        let start = Instant::now();

        let mut child = cmd
            .spawn()
            .map_err(|e| SolverError::ExecutionFailed(e.to_string()))?;

        let stdout_handle = child.stdout.take();
        let stderr_handle = child.stderr.take();

        let result = timeout(config.timeout, async {
            let mut stdout = String::new();
            let mut stderr = String::new();

            let (status, stdout_result, stderr_result) = tokio::join!(
                child.wait(),
                async {
                    if let Some(mut handle) = stdout_handle {
                        let _ = handle.read_to_string(&mut stdout).await;
                    }
                    stdout
                },
                async {
                    if let Some(mut handle) = stderr_handle {
                        let _ = handle.read_to_string(&mut stderr).await;
                    }
                    stderr
                }
            );

            Ok::<_, std::io::Error>((status?, stdout_result, stderr_result))
        })
        .await;

        let duration = start.elapsed();

        match result {
            Ok(Ok((_status, stdout, stderr))) => {
                let output = self.parse_output(&stdout, &stderr, duration);
                let proof_path = proof_file.exists().then_some(proof_file);
                Ok((output, proof_path))
            }
            Ok(Err(e)) => Err(SolverError::ExecutionFailed(e.to_string())),
            Err(_) => {
                warn!("CaDiCaL timed out after {:?}", config.timeout);
                let _ = child.kill().await;
                Err(SolverError::Timeout(config.timeout))
            }
        }
    }

    /// Solve and extract learned clauses from the proof
    ///
    /// This is a convenience method that combines solving with proof output
    /// and parsing the DRAT proof to extract learned clauses.
    pub async fn solve_and_learn(
        &self,
        dimacs_path: &Path,
        config: &SolverConfig,
        learn_config: LearnConfig,
    ) -> Result<LearnedClausesResult, SolverError> {
        let (output, proof_path) = self.solve_with_proof(dimacs_path, config, None).await?;

        let learned_clauses = if let Some(proof_path) = &proof_path {
            match DratParser::parse_file(proof_path) {
                Ok(drat_clauses) => {
                    let raw_learned = extract_learned_clauses(&drat_clauses);
                    let filtered = filter_learned_clauses(
                        raw_learned,
                        learn_config.min_clause_size,
                        learn_config.max_clause_size,
                        learn_config.max_clauses,
                    );
                    info!("Extracted {} learned clauses from proof", filtered.len());
                    filtered
                }
                Err(e) => {
                    warn!("Failed to parse DRAT proof: {}", e);
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        };

        // Clean up proof file if requested
        if learn_config.cleanup_proof {
            if let Some(proof_path) = proof_path {
                let _ = std::fs::remove_file(proof_path);
            }
        }

        Ok(LearnedClausesResult {
            output,
            learned_clauses,
        })
    }
}

/// Configuration for learned clause extraction
#[derive(Debug, Clone)]
pub struct LearnConfig {
    /// Minimum clause size to keep
    pub min_clause_size: usize,
    /// Maximum clause size to keep
    pub max_clause_size: usize,
    /// Maximum number of clauses to keep
    pub max_clauses: usize,
    /// Whether to delete the proof file after parsing
    pub cleanup_proof: bool,
}

impl Default for LearnConfig {
    fn default() -> Self {
        Self {
            min_clause_size: 2,
            max_clause_size: 20,
            max_clauses: 10000,
            cleanup_proof: true,
        }
    }
}

/// Result of solving with learned clause extraction
#[derive(Debug)]
pub struct LearnedClausesResult {
    /// The solver output
    pub output: SolverOutput,
    /// Learned clauses extracted from the proof
    pub learned_clauses: Vec<Vec<i32>>,
}

impl Default for CaDiCaL {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Solver for CaDiCaL {
    fn info(&self) -> &SolverInfo {
        &self.info
    }

    async fn check_available(&self) -> bool {
        self.find_binary().await.is_some()
    }

    async fn solve_dimacs(
        &self,
        path: &Path,
        config: &SolverConfig,
    ) -> Result<SolverOutput, SolverError> {
        let binary = self
            .binary_path
            .as_ref()
            .ok_or_else(|| SolverError::NotFound("CaDiCaL binary not found".to_string()))?;

        if !path.exists() {
            return Err(SolverError::InvalidInput(format!(
                "DIMACS file not found: {}",
                path.display()
            )));
        }

        let mut cmd = Command::new(binary);
        cmd.arg(path);

        // Add configuration options
        if let Some(seed) = config.seed {
            cmd.arg("--seed").arg(seed.to_string());
        }

        // CaDiCaL-specific options from config
        for (key, value) in &config.options {
            cmd.arg(format!("--{key}={value}"));
        }

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        debug!("Running CaDiCaL: {:?}", cmd);
        let start = Instant::now();

        let mut child = cmd
            .spawn()
            .map_err(|e| SolverError::ExecutionFailed(e.to_string()))?;

        let stdout_handle = child.stdout.take();
        let stderr_handle = child.stderr.take();

        let result = timeout(config.timeout, async {
            let mut stdout = String::new();
            let mut stderr = String::new();

            let (status, stdout_result, stderr_result) = tokio::join!(
                child.wait(),
                async {
                    if let Some(mut handle) = stdout_handle {
                        let _ = handle.read_to_string(&mut stdout).await;
                    }
                    stdout
                },
                async {
                    if let Some(mut handle) = stderr_handle {
                        let _ = handle.read_to_string(&mut stderr).await;
                    }
                    stderr
                }
            );

            Ok::<_, std::io::Error>((status?, stdout_result, stderr_result))
        })
        .await;

        let duration = start.elapsed();

        match result {
            Ok(Ok((_status, stdout, stderr))) => Ok(self.parse_output(&stdout, &stderr, duration)),
            Ok(Err(e)) => Err(SolverError::ExecutionFailed(e.to_string())),
            Err(_) => {
                warn!("CaDiCaL timed out after {:?}", config.timeout);
                let _ = child.kill().await;
                Err(SolverError::Timeout(config.timeout))
            }
        }
    }

    async fn solve_smt2(
        &self,
        _path: &Path,
        _config: &SolverConfig,
    ) -> Result<SolverOutput, SolverError> {
        Err(SolverError::InvalidInput(
            "CaDiCaL does not support SMT-LIB2 format".to_string(),
        ))
    }
}

/// Create a boxed CaDiCaL solver
pub async fn create_cadical() -> Option<BoxedSolver> {
    CaDiCaL::detect().await.map(|s| Box::new(s) as BoxedSolver)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cadical_info() {
        let solver = CaDiCaL::new();
        assert_eq!(solver.info().id, "cadical");
        assert!(solver.supports(SolverCapability::Sat));
        assert!(!solver.supports(SolverCapability::SmtBv));
    }

    #[test]
    fn test_cadical_new() {
        let solver = CaDiCaL::new();
        assert_eq!(solver.info.id, "cadical");
        assert_eq!(solver.info.name, "CaDiCaL");
        assert_eq!(solver.info.version, "unknown");
        assert!(!solver.info.available);
        assert!(solver.binary_path.is_none());
    }

    #[test]
    fn test_cadical_with_binary() {
        let solver = CaDiCaL::with_binary("/usr/local/bin/cadical");
        assert_eq!(
            solver.binary_path,
            Some("/usr/local/bin/cadical".to_string())
        );
        assert_eq!(solver.info.id, "cadical");
    }

    #[test]
    fn test_cadical_with_binary_string() {
        let path = String::from("/opt/solvers/cadical");
        let solver = CaDiCaL::with_binary(path);
        assert_eq!(solver.binary_path, Some("/opt/solvers/cadical".to_string()));
    }

    #[test]
    fn test_cadical_default() {
        let solver = CaDiCaL::default();
        assert_eq!(solver.info.id, "cadical");
        assert!(solver.binary_path.is_none());
    }

    #[test]
    fn test_cadical_capabilities() {
        let solver = CaDiCaL::new();
        assert!(solver.info.capabilities.contains(&SolverCapability::Sat));
        assert!(solver
            .info
            .capabilities
            .contains(&SolverCapability::Incremental));
        assert!(solver.info.capabilities.contains(&SolverCapability::Proofs));
        assert!(!solver.info.capabilities.contains(&SolverCapability::SmtBv));
        assert!(!solver.info.capabilities.contains(&SolverCapability::SmtLia));
    }

    #[test]
    fn test_learn_config_default() {
        let config = LearnConfig::default();
        assert_eq!(config.min_clause_size, 2);
        assert_eq!(config.max_clause_size, 20);
        assert_eq!(config.max_clauses, 10000);
        assert!(config.cleanup_proof);
    }

    #[test]
    fn test_learn_config_custom() {
        let config = LearnConfig {
            min_clause_size: 3,
            max_clause_size: 50,
            max_clauses: 5000,
            cleanup_proof: false,
        };
        assert_eq!(config.min_clause_size, 3);
        assert_eq!(config.max_clause_size, 50);
        assert_eq!(config.max_clauses, 5000);
        assert!(!config.cleanup_proof);
    }

    #[test]
    fn test_learn_config_clone() {
        let config = LearnConfig {
            min_clause_size: 5,
            max_clause_size: 30,
            max_clauses: 1000,
            cleanup_proof: true,
        };
        let cloned = config.clone();
        assert_eq!(config.min_clause_size, cloned.min_clause_size);
        assert_eq!(config.max_clause_size, cloned.max_clause_size);
        assert_eq!(config.max_clauses, cloned.max_clauses);
        assert_eq!(config.cleanup_proof, cloned.cleanup_proof);
    }

    #[test]
    fn test_learn_config_debug() {
        let config = LearnConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("LearnConfig"));
        assert!(debug_str.contains("min_clause_size"));
        assert!(debug_str.contains("max_clause_size"));
    }

    #[test]
    fn test_learned_clauses_result_debug() {
        let result = LearnedClausesResult {
            output: SolverOutput {
                result: SolverResult::Sat { model: None },
                stats: SolverStats::default(),
                raw_output: None,
            },
            learned_clauses: vec![vec![1, 2, -3], vec![-1, 4]],
        };
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("LearnedClausesResult"));
        assert!(debug_str.contains("learned_clauses"));
    }

    #[test]
    fn test_parse_output_sat() {
        let solver = CaDiCaL::new();
        let stdout = "s SATISFIABLE\nv 1 2 -3 0\n";
        let stderr = "";
        let duration = std::time::Duration::from_millis(100);

        let output = solver.parse_output(stdout, stderr, duration);
        assert!(output.result.is_sat());
        assert_eq!(output.stats.solve_time, duration);
    }

    #[test]
    fn test_parse_output_unsat() {
        let solver = CaDiCaL::new();
        let stdout = "s UNSATISFIABLE\n";
        let stderr = "";
        let duration = std::time::Duration::from_millis(50);

        let output = solver.parse_output(stdout, stderr, duration);
        assert!(output.result.is_unsat());
    }

    #[test]
    fn test_parse_output_unknown() {
        let solver = CaDiCaL::new();
        let stdout = "c solving...\nc terminated\n";
        let stderr = "";
        let duration = std::time::Duration::from_millis(1000);

        let output = solver.parse_output(stdout, stderr, duration);
        // Unknown is neither SAT nor UNSAT
        assert!(!output.result.is_sat());
        assert!(!output.result.is_unsat());
        assert!(!output.result.is_definitive());
    }

    #[test]
    fn test_parse_output_sat_in_stderr() {
        let solver = CaDiCaL::new();
        let stdout = "";
        let stderr = "s SATISFIABLE\n";
        let duration = std::time::Duration::from_millis(100);

        let output = solver.parse_output(stdout, stderr, duration);
        assert!(output.result.is_sat());
    }

    #[test]
    fn test_parse_output_unsat_in_stderr() {
        let solver = CaDiCaL::new();
        let stdout = "";
        let stderr = "s UNSATISFIABLE\n";
        let duration = std::time::Duration::from_millis(100);

        let output = solver.parse_output(stdout, stderr, duration);
        assert!(output.result.is_unsat());
    }

    #[test]
    fn test_parse_output_with_statistics() {
        let solver = CaDiCaL::new();
        let stdout = r"c conflicts: 12345
c decisions: 54321
c propagations: 98765
s SATISFIABLE
v 1 -2 3 0
";
        let stderr = "";
        let duration = std::time::Duration::from_millis(200);

        let output = solver.parse_output(stdout, stderr, duration);
        assert!(output.result.is_sat());
        assert_eq!(output.stats.conflicts, Some(12345));
        assert_eq!(output.stats.decisions, Some(54321));
        assert_eq!(output.stats.propagations, Some(98765));
    }

    #[test]
    fn test_parse_output_stats_from_stderr() {
        let solver = CaDiCaL::new();
        let stdout = "s SATISFIABLE\n";
        let stderr = "c conflicts: 100\nc decisions: 200\n";
        let duration = std::time::Duration::from_millis(50);

        let output = solver.parse_output(stdout, stderr, duration);
        assert!(output.result.is_sat());
        assert_eq!(output.stats.conflicts, Some(100));
        assert_eq!(output.stats.decisions, Some(200));
    }

    #[test]
    fn test_parse_output_raw_output_included() {
        let solver = CaDiCaL::new();
        let stdout = "s SATISFIABLE\n";
        let stderr = "c warning: something\n";
        let duration = std::time::Duration::from_millis(50);

        let output = solver.parse_output(stdout, stderr, duration);
        assert!(output.raw_output.is_some());
        let raw = output.raw_output.unwrap();
        assert!(raw.contains("STDOUT:"));
        assert!(raw.contains("STDERR:"));
        assert!(raw.contains("SATISFIABLE"));
        assert!(raw.contains("warning"));
    }

    #[test]
    fn test_parse_output_with_model() {
        let solver = CaDiCaL::new();
        let stdout = "s SATISFIABLE\nv 1 -2 3 -4 5 0\n";
        let stderr = "";
        let duration = std::time::Duration::from_millis(100);

        let output = solver.parse_output(stdout, stderr, duration);
        assert!(output.result.is_sat());
        if let SolverResult::Sat { model } = &output.result {
            // Model may or may not be extracted depending on implementation
            // Just verify the result type is correct
            let _ = model;
        } else {
            panic!("Expected SAT result");
        }
    }

    #[test]
    fn test_parse_output_multiline_model() {
        let solver = CaDiCaL::new();
        let stdout = "s SATISFIABLE\nv 1 2 3\nv 4 5 6 0\n";
        let stderr = "";
        let duration = std::time::Duration::from_millis(100);

        let output = solver.parse_output(stdout, stderr, duration);
        assert!(output.result.is_sat());
    }

    #[test]
    fn test_parse_output_no_stats() {
        let solver = CaDiCaL::new();
        let stdout = "s SATISFIABLE\n";
        let stderr = "";
        let duration = std::time::Duration::from_millis(10);

        let output = solver.parse_output(stdout, stderr, duration);
        assert!(output.stats.conflicts.is_none());
        assert!(output.stats.decisions.is_none());
        assert!(output.stats.propagations.is_none());
    }

    #[test]
    fn test_solver_info_accessors() {
        let solver = CaDiCaL::new();
        let info = solver.info();
        assert_eq!(info.id, "cadical");
        assert_eq!(info.name, "CaDiCaL");
    }

    #[tokio::test]
    async fn test_check_available_without_binary() {
        let solver = CaDiCaL::new();
        // This will check PATH, result depends on system
        let _ = solver.check_available().await;
    }

    #[tokio::test]
    async fn test_check_available_with_binary() {
        let solver = CaDiCaL::with_binary("/nonexistent/path/cadical");
        // Should still work because find_binary returns the set path
        // but actual solving would fail
        let available = solver.check_available().await;
        // Path exists in config, so find_binary returns it
        assert!(available);
    }

    #[tokio::test]
    async fn test_solve_dimacs_missing_file() {
        let solver = CaDiCaL::with_binary("cadical");
        let config = SolverConfig::default();
        let result = solver
            .solve_dimacs(Path::new("/nonexistent/file.cnf"), &config)
            .await;

        assert!(result.is_err());
        match result {
            Err(SolverError::InvalidInput(msg)) => {
                assert!(msg.contains("not found"));
            }
            _ => panic!("Expected InvalidInput error"),
        }
    }

    #[tokio::test]
    async fn test_solve_dimacs_no_binary() {
        let solver = CaDiCaL::new(); // No binary path set
        let config = SolverConfig::default();
        let temp_dir = tempfile::tempdir().unwrap();
        let cnf_path = temp_dir.path().join("test.cnf");
        std::fs::write(&cnf_path, "p cnf 1 1\n1 0\n").unwrap();

        let result = solver.solve_dimacs(&cnf_path, &config).await;
        assert!(result.is_err());
        match result {
            Err(SolverError::NotFound(msg)) => {
                assert!(msg.contains("binary not found"));
            }
            _ => panic!("Expected NotFound error"),
        }
    }

    #[tokio::test]
    async fn test_solve_smt2_returns_error() {
        let solver = CaDiCaL::new();
        let config = SolverConfig::default();
        let result = solver.solve_smt2(Path::new("test.smt2"), &config).await;

        assert!(result.is_err());
        match result {
            Err(SolverError::InvalidInput(msg)) => {
                assert!(msg.contains("does not support SMT-LIB2"));
            }
            _ => panic!("Expected InvalidInput error"),
        }
    }

    #[tokio::test]
    async fn test_solve_with_proof_missing_file() {
        let solver = CaDiCaL::with_binary("cadical");
        let config = SolverConfig::default();
        let result = solver
            .solve_with_proof(Path::new("/nonexistent/file.cnf"), &config, None)
            .await;

        assert!(result.is_err());
        match result {
            Err(SolverError::InvalidInput(msg)) => {
                assert!(msg.contains("not found"));
            }
            _ => panic!("Expected InvalidInput error"),
        }
    }

    #[tokio::test]
    async fn test_solve_with_proof_no_binary() {
        let solver = CaDiCaL::new(); // No binary path set
        let config = SolverConfig::default();
        let temp_dir = tempfile::tempdir().unwrap();
        let cnf_path = temp_dir.path().join("test.cnf");
        std::fs::write(&cnf_path, "p cnf 1 1\n1 0\n").unwrap();

        let result = solver.solve_with_proof(&cnf_path, &config, None).await;
        assert!(result.is_err());
        match result {
            Err(SolverError::NotFound(msg)) => {
                assert!(msg.contains("binary not found"));
            }
            _ => panic!("Expected NotFound error"),
        }
    }

    #[tokio::test]
    async fn test_create_cadical() {
        // This test depends on system having CaDiCaL installed
        let solver = create_cadical().await;
        // Just verify it doesn't panic - result depends on system
        if let Some(s) = solver {
            assert_eq!(s.info().id, "cadical");
        }
    }

    #[tokio::test]
    async fn test_detect_returns_available_solver() {
        // This test depends on system having CaDiCaL installed
        if let Some(solver) = CaDiCaL::detect().await {
            assert!(solver.info.available);
            assert!(solver.binary_path.is_some());
            // Version should be updated from "unknown"
            // (may still be "unknown" if --version fails)
        }
    }

    #[tokio::test]
    async fn test_solve_and_learn_unsat() {
        // Skip if CaDiCaL is not available
        let Some(solver) = CaDiCaL::detect().await else {
            return;
        };

        // Create a pigeon-hole problem (3 pigeons, 2 holes) - always UNSAT
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
        let temp_dir = tempfile::tempdir().unwrap();
        let cnf_path = temp_dir.path().join("php32.cnf");
        std::fs::write(&cnf_path, dimacs).unwrap();

        let config = SolverConfig::default();
        let learn_config = LearnConfig {
            min_clause_size: 1,   // Include unit clauses (common in UNSAT proofs)
            cleanup_proof: false, // Keep proof for verification
            ..Default::default()
        };

        let result = solver
            .solve_and_learn(&cnf_path, &config, learn_config)
            .await
            .unwrap();

        assert!(result.output.result.is_unsat());
        // PHP problems generate learned clauses (including unit clauses in UNSAT proof)
        assert!(
            !result.learned_clauses.is_empty(),
            "Should have learned clauses"
        );
    }

    #[tokio::test]
    async fn test_solve_and_learn_sat() {
        // Skip if CaDiCaL is not available
        let Some(solver) = CaDiCaL::detect().await else {
            return;
        };

        // Simple SAT problem
        let dimacs = "p cnf 3 2\n1 2 0\n-1 3 0\n";
        let temp_dir = tempfile::tempdir().unwrap();
        let cnf_path = temp_dir.path().join("sat.cnf");
        std::fs::write(&cnf_path, dimacs).unwrap();

        let config = SolverConfig::default();
        let learn_config = LearnConfig {
            cleanup_proof: true,
            ..Default::default()
        };

        let result = solver
            .solve_and_learn(&cnf_path, &config, learn_config)
            .await
            .unwrap();

        assert!(result.output.result.is_sat());
        // SAT problems may or may not have learned clauses depending on search
    }

    #[tokio::test]
    async fn test_solve_and_learn_cleanup_proof() {
        // Skip if CaDiCaL is not available
        let Some(solver) = CaDiCaL::detect().await else {
            return;
        };

        let dimacs = "p cnf 2 2\n1 2 0\n-1 -2 0\n";
        let temp_dir = tempfile::tempdir().unwrap();
        let cnf_path = temp_dir.path().join("test.cnf");
        std::fs::write(&cnf_path, dimacs).unwrap();

        let config = SolverConfig::default();
        let learn_config = LearnConfig {
            cleanup_proof: true,
            ..Default::default()
        };

        let _ = solver
            .solve_and_learn(&cnf_path, &config, learn_config)
            .await
            .unwrap();

        // Proof file should be cleaned up
        let proof_path = cnf_path.with_extension("drat");
        assert!(!proof_path.exists(), "Proof file should be deleted");
    }

    #[tokio::test]
    async fn test_solve_with_proof_creates_file() {
        // Skip if CaDiCaL is not available
        let Some(solver) = CaDiCaL::detect().await else {
            return;
        };

        let dimacs = "p cnf 3 2\n1 2 0\n-1 3 0\n";
        let temp_dir = tempfile::tempdir().unwrap();
        let cnf_path = temp_dir.path().join("test.cnf");
        let proof_path = temp_dir.path().join("test.drat");
        std::fs::write(&cnf_path, dimacs).unwrap();

        let config = SolverConfig::default();
        let (output, returned_path) = solver
            .solve_with_proof(&cnf_path, &config, Some(&proof_path))
            .await
            .unwrap();

        // Should be SAT
        assert!(output.result.is_sat());
        // Proof path should exist
        assert!(proof_path.exists());
        assert_eq!(returned_path, Some(proof_path));
    }

    #[tokio::test]
    async fn test_solve_with_proof_default_path() {
        // Skip if CaDiCaL is not available
        let Some(solver) = CaDiCaL::detect().await else {
            return;
        };

        let dimacs = "p cnf 2 1\n1 2 0\n";
        let temp_dir = tempfile::tempdir().unwrap();
        let cnf_path = temp_dir.path().join("auto_proof.cnf");
        std::fs::write(&cnf_path, dimacs).unwrap();

        let config = SolverConfig::default();
        let (output, returned_path) = solver
            .solve_with_proof(&cnf_path, &config, None) // No explicit path
            .await
            .unwrap();

        assert!(output.result.is_sat());
        // Should use default .drat extension
        if let Some(path) = returned_path {
            assert_eq!(path.extension().unwrap(), "drat");
        }
    }

    #[tokio::test]
    async fn test_solve_dimacs_sat() {
        // Skip if CaDiCaL is not available
        let Some(solver) = CaDiCaL::detect().await else {
            return;
        };

        let dimacs = "p cnf 3 2\n1 2 3 0\n-1 -2 0\n";
        let temp_dir = tempfile::tempdir().unwrap();
        let cnf_path = temp_dir.path().join("sat.cnf");
        std::fs::write(&cnf_path, dimacs).unwrap();

        let config = SolverConfig::default();
        let output = solver.solve_dimacs(&cnf_path, &config).await.unwrap();

        assert!(output.result.is_sat());
    }

    #[tokio::test]
    async fn test_solve_dimacs_unsat() {
        // Skip if CaDiCaL is not available
        let Some(solver) = CaDiCaL::detect().await else {
            return;
        };

        // Contradiction: x AND NOT x
        let dimacs = "p cnf 1 2\n1 0\n-1 0\n";
        let temp_dir = tempfile::tempdir().unwrap();
        let cnf_path = temp_dir.path().join("unsat.cnf");
        std::fs::write(&cnf_path, dimacs).unwrap();

        let config = SolverConfig::default();
        let output = solver.solve_dimacs(&cnf_path, &config).await.unwrap();

        assert!(output.result.is_unsat());
    }

    #[tokio::test]
    async fn test_solve_dimacs_with_seed() {
        // Skip if CaDiCaL is not available
        let Some(solver) = CaDiCaL::detect().await else {
            return;
        };

        let dimacs = "p cnf 3 2\n1 2 0\n2 3 0\n";
        let temp_dir = tempfile::tempdir().unwrap();
        let cnf_path = temp_dir.path().join("seeded.cnf");
        std::fs::write(&cnf_path, dimacs).unwrap();

        let config = SolverConfig {
            seed: Some(42),
            ..Default::default()
        };
        let output = solver.solve_dimacs(&cnf_path, &config).await.unwrap();

        assert!(output.result.is_sat());
    }

    #[tokio::test]
    async fn test_solve_dimacs_with_options() {
        // Skip if CaDiCaL is not available
        let Some(solver) = CaDiCaL::detect().await else {
            return;
        };

        let dimacs = "p cnf 2 2\n1 2 0\n-1 -2 0\n";
        let temp_dir = tempfile::tempdir().unwrap();
        let cnf_path = temp_dir.path().join("opts.cnf");
        std::fs::write(&cnf_path, dimacs).unwrap();

        let mut config = SolverConfig::default();
        config.options.insert("quiet".to_string(), "1".to_string());

        let output = solver.solve_dimacs(&cnf_path, &config).await.unwrap();
        // Should still work with options
        assert!(output.result.is_sat() || output.result.is_unsat());
    }

    // ==================== Mutation Coverage Tests ====================

    #[tokio::test]
    async fn test_check_available_returns_bool_value() {
        // Test that check_available returns a meaningful boolean value
        let solver_with_binary = CaDiCaL::with_binary("/bin/echo");
        let available = solver_with_binary.check_available().await;
        assert!(
            available,
            "check_available should return true when binary exists"
        );

        let solver_without_binary = CaDiCaL::new();
        let available = solver_without_binary.check_available().await;
        if CaDiCaL::detect().await.is_some() {
            assert!(available);
        } else {
            assert!(!available);
        }
    }

    #[tokio::test]
    async fn test_create_cadical_returns_option() {
        // Test that create_cadical correctly returns Some or None
        let result = create_cadical().await;
        match result {
            Some(boxed) => {
                assert_eq!(boxed.info().id, "cadical");
                assert!(boxed.info().available);
                assert!(boxed.supports(SolverCapability::Sat));
            }
            None => {
                // Expected when cadical not installed
            }
        }
    }

    #[tokio::test]
    async fn test_find_binary_returns_preset_path() {
        let solver = CaDiCaL::with_binary("/custom/cadical");
        let binary = solver.find_binary().await;
        assert_eq!(binary, Some("/custom/cadical".to_string()));
    }

    #[test]
    fn test_parse_output_solve_time_preserved() {
        let solver = CaDiCaL::new();
        let stdout = "s SATISFIABLE\n";
        let stderr = "";
        let duration = std::time::Duration::from_millis(12345);

        let output = solver.parse_output(stdout, stderr, duration);

        assert_eq!(output.stats.solve_time, duration);
        assert_eq!(output.stats.solve_time.as_millis(), 12345);
    }

    #[tokio::test]
    async fn test_get_version_returns_value() {
        let Some(solver) = CaDiCaL::detect().await else {
            return;
        };

        let version = solver.get_version().await;
        match version {
            Some(v) => {
                assert!(!v.is_empty(), "Version should not be empty");
            }
            None => {
                // Version parsing may fail
            }
        }
    }
}
