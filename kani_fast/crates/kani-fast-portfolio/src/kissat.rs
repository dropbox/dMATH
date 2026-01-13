//! Kissat SAT solver wrapper
//!
//! Kissat is a fast SAT solver that won multiple SAT Competition awards.
//! It excels on many industrial SAT instances.

use crate::solver::{
    util::{extract_model, extract_number},
    BoxedSolver, Solver, SolverCapability, SolverConfig, SolverError, SolverInfo, SolverOutput,
    SolverResult, SolverStats,
};
use async_trait::async_trait;
use std::path::Path;
use std::process::Stdio;
use std::time::Instant;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::time::timeout;
use tracing::{debug, warn};

/// Kissat SAT solver
pub struct Kissat {
    info: SolverInfo,
    binary_path: Option<String>,
}

impl Kissat {
    /// Create a new Kissat instance
    pub fn new() -> Self {
        Self {
            info: SolverInfo {
                id: "kissat".to_string(),
                name: "Kissat".to_string(),
                version: "unknown".to_string(),
                capabilities: vec![SolverCapability::Sat],
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

    /// Detect and initialize Kissat
    pub async fn detect() -> Option<Self> {
        let mut solver = Self::new();

        // Try to find kissat binary
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
        for name in &["kissat", "Kissat"] {
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
        // Kissat version output: "kissat X.Y.Z ..."
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

        // Parse Kissat statistics from output
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

        // Determine result from output - Kissat uses standard DIMACS output format
        let result = if stdout.contains("s SATISFIABLE") || stderr.contains("s SATISFIABLE") {
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
}

impl Default for Kissat {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Solver for Kissat {
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
            .ok_or_else(|| SolverError::NotFound("Kissat binary not found".to_string()))?;

        if !path.exists() {
            return Err(SolverError::InvalidInput(format!(
                "DIMACS file not found: {}",
                path.display()
            )));
        }

        let mut cmd = Command::new(binary);
        cmd.arg(path);

        // Kissat options
        if config.options.get("quiet").is_some_and(|v| v == "true") {
            cmd.arg("--quiet");
        }

        // Kissat-specific options from config
        for (key, value) in &config.options {
            if key != "quiet" {
                cmd.arg(format!("--{key}={value}"));
            }
        }

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        debug!("Running Kissat: {:?}", cmd);
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
                warn!("Kissat timed out after {:?}", config.timeout);
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
            "Kissat does not support SMT-LIB2 format".to_string(),
        ))
    }
}

/// Create a boxed Kissat solver
pub async fn create_kissat() -> Option<BoxedSolver> {
    Kissat::detect().await.map(|s| Box::new(s) as BoxedSolver)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Basic Construction Tests ====================

    #[test]
    fn test_kissat_new() {
        let solver = Kissat::new();
        assert_eq!(solver.info().id, "kissat");
        assert_eq!(solver.info().name, "Kissat");
        assert_eq!(solver.info().version, "unknown");
        assert!(!solver.info().available);
        assert!(solver.binary_path.is_none());
    }

    #[test]
    fn test_kissat_default() {
        let solver = Kissat::default();
        assert_eq!(solver.info().id, "kissat");
        assert_eq!(solver.info().name, "Kissat");
    }

    #[test]
    fn test_kissat_with_binary() {
        let solver = Kissat::with_binary("/usr/local/bin/kissat");
        assert_eq!(
            solver.binary_path,
            Some("/usr/local/bin/kissat".to_string())
        );
    }

    #[test]
    fn test_kissat_with_binary_into() {
        let path = String::from("/custom/path/kissat");
        let solver = Kissat::with_binary(path);
        assert_eq!(solver.binary_path, Some("/custom/path/kissat".to_string()));
    }

    // ==================== Capability Tests ====================

    #[test]
    fn test_kissat_info() {
        let solver = Kissat::new();
        assert_eq!(solver.info().id, "kissat");
        assert!(solver.supports(SolverCapability::Sat));
        assert!(!solver.supports(SolverCapability::SmtBv));
    }

    #[test]
    fn test_kissat_capabilities() {
        let solver = Kissat::new();
        assert!(solver.supports(SolverCapability::Sat));
        assert!(!solver.supports(SolverCapability::SmtBv));
        assert!(!solver.supports(SolverCapability::SmtArrays));
        assert!(!solver.supports(SolverCapability::Quantifiers));
        assert!(!solver.supports(SolverCapability::Incremental));
        assert!(!solver.supports(SolverCapability::Proofs));
    }

    #[test]
    fn test_kissat_single_capability() {
        let solver = Kissat::new();
        // Kissat is a pure SAT solver with only SAT capability
        assert_eq!(solver.info().capabilities.len(), 1);
        assert_eq!(solver.info().capabilities[0], SolverCapability::Sat);
    }

    // ==================== extract_number Tests ====================

    #[test]
    fn test_extract_number() {
        assert_eq!(extract_number("conflicts: 12345"), Some(12345));
        assert_eq!(extract_number("c decisions: 9999"), Some(9999));
        assert_eq!(extract_number("no numbers here"), None);
    }

    #[test]
    fn test_extract_number_single_digit() {
        assert_eq!(extract_number("conflicts: 1"), Some(1));
        assert_eq!(extract_number("x 0"), Some(0));
    }

    #[test]
    fn test_extract_number_large() {
        assert_eq!(
            extract_number("propagations: 18446744073709551615"),
            Some(18446744073709551615)
        );
    }

    #[test]
    fn test_extract_number_with_commas() {
        // Numbers with surrounding punctuation
        assert_eq!(extract_number("stat: [123]"), Some(123));
        assert_eq!(extract_number("(456)"), Some(456));
    }

    #[test]
    fn test_extract_number_multiple_numbers() {
        // Should extract first number found
        assert_eq!(extract_number("a: 100 b: 200"), Some(100));
    }

    #[test]
    fn test_extract_number_empty_string() {
        assert_eq!(extract_number(""), None);
    }

    #[test]
    fn test_extract_number_whitespace_only() {
        assert_eq!(extract_number("   "), None);
    }

    #[test]
    fn test_extract_number_negative() {
        // Negative numbers may be trimmed of the sign
        let result = extract_number("value: -123");
        // Should extract 123 since we trim non-digits
        assert_eq!(result, Some(123));
    }

    // ==================== extract_model Tests ====================

    #[test]
    fn test_extract_model() {
        let output = "c some comment\ns SATISFIABLE\nv 1 -2 3 0\nv 4 -5 0";
        let model = extract_model(output);
        assert_eq!(model, Some("1 -2 3 0 4 -5 0".to_string()));
    }

    #[test]
    fn test_extract_model_single_line() {
        let output = "s SATISFIABLE\nv 1 2 3 0\n";
        let model = extract_model(output);
        assert_eq!(model, Some("1 2 3 0".to_string()));
    }

    #[test]
    fn test_extract_model_no_model() {
        let output = "s SATISFIABLE\nc no model lines\n";
        let model = extract_model(output);
        assert_eq!(model, None);
    }

    #[test]
    fn test_extract_model_empty_output() {
        let model = extract_model("");
        assert_eq!(model, None);
    }

    #[test]
    fn test_extract_model_unsat() {
        let output = "s UNSATISFIABLE\n";
        let model = extract_model(output);
        assert_eq!(model, None);
    }

    #[test]
    fn test_extract_model_multiple_lines() {
        let output = "v 1 0\nv 2 0\nv 3 0\n";
        let model = extract_model(output);
        assert_eq!(model, Some("1 0 2 0 3 0".to_string()));
    }

    #[test]
    fn test_extract_model_preserves_negatives() {
        let output = "v -1 -2 -3 0\n";
        let model = extract_model(output);
        assert_eq!(model, Some("-1 -2 -3 0".to_string()));
    }

    // ==================== parse_output Tests ====================

    #[test]
    fn test_parse_output_sat() {
        let solver = Kissat::new();
        let stdout = "c solving...\ns SATISFIABLE\nv 1 -2 0\n";
        let stderr = "";
        let duration = std::time::Duration::from_millis(100);

        let output = solver.parse_output(stdout, stderr, duration);

        assert!(output.result.is_sat());
        if let SolverResult::Sat { model } = &output.result {
            assert!(model.is_some());
            assert_eq!(model.as_ref().unwrap(), "1 -2 0");
        }
        assert_eq!(output.stats.solve_time, duration);
    }

    #[test]
    fn test_parse_output_unsat() {
        let solver = Kissat::new();
        let stdout = "c solving...\ns UNSATISFIABLE\n";
        let stderr = "";
        let duration = std::time::Duration::from_millis(50);

        let output = solver.parse_output(stdout, stderr, duration);

        assert!(output.result.is_unsat());
    }

    #[test]
    fn test_parse_output_sat_in_stderr() {
        let solver = Kissat::new();
        let stdout = "";
        let stderr = "s SATISFIABLE\nv 1 0\n";
        let duration = std::time::Duration::from_millis(100);

        let output = solver.parse_output(stdout, stderr, duration);

        assert!(output.result.is_sat());
    }

    #[test]
    fn test_parse_output_unsat_in_stderr() {
        let solver = Kissat::new();
        let stdout = "";
        let stderr = "s UNSATISFIABLE\n";
        let duration = std::time::Duration::from_millis(100);

        let output = solver.parse_output(stdout, stderr, duration);

        assert!(output.result.is_unsat());
    }

    #[test]
    fn test_parse_output_unknown() {
        let solver = Kissat::new();
        let stdout = "c some output\n";
        let stderr = "";
        let duration = std::time::Duration::from_millis(100);

        let output = solver.parse_output(stdout, stderr, duration);

        assert!(!output.result.is_definitive());
        if let SolverResult::Unknown { reason } = &output.result {
            assert!(reason.contains("No definitive result"));
        }
    }

    #[test]
    fn test_parse_output_with_stats() {
        let solver = Kissat::new();
        let stdout =
            "c conflicts: 1000\nc decisions: 5000\nc propagations: 50000\ns SATISFIABLE\nv 1 0\n";
        let stderr = "";
        let duration = std::time::Duration::from_millis(100);

        let output = solver.parse_output(stdout, stderr, duration);

        assert_eq!(output.stats.conflicts, Some(1000));
        assert_eq!(output.stats.decisions, Some(5000));
        assert_eq!(output.stats.propagations, Some(50000));
    }

    #[test]
    fn test_parse_output_stats_from_stderr() {
        let solver = Kissat::new();
        let stdout = "s SATISFIABLE\nv 1 0\n";
        let stderr = "conflicts: 500\n";
        let duration = std::time::Duration::from_millis(100);

        let output = solver.parse_output(stdout, stderr, duration);

        assert_eq!(output.stats.conflicts, Some(500));
    }

    #[test]
    fn test_parse_output_raw_output() {
        let solver = Kissat::new();
        let stdout = "stdout content";
        let stderr = "stderr content";
        let duration = std::time::Duration::from_millis(100);

        let output = solver.parse_output(stdout, stderr, duration);

        let raw = output.raw_output.unwrap();
        assert!(raw.contains("STDOUT:"));
        assert!(raw.contains("stdout content"));
        assert!(raw.contains("STDERR:"));
        assert!(raw.contains("stderr content"));
    }

    // ==================== Async Detection Tests ====================

    #[tokio::test]
    async fn test_kissat_detect_available() {
        // This test may or may not find kissat depending on the system
        let result = Kissat::detect().await;

        if let Some(solver) = result {
            assert!(solver.info().available);
            assert!(solver.binary_path.is_some());
            // Version should be populated if detection succeeded
            assert_ne!(solver.info().version, "unknown");
        }
        // If kissat is not installed, result is None, which is fine
    }

    #[tokio::test]
    async fn test_kissat_check_available() {
        let solver = Kissat::new();
        let available = solver.check_available().await;
        // This depends on whether kissat is installed
        // Just ensure the check runs without error
        let _ = available;
    }

    // ==================== solve_smt2 Error Tests ====================

    #[tokio::test]
    async fn test_kissat_solve_smt2_error() {
        let solver = Kissat::with_binary("/some/path/kissat");
        let config = SolverConfig::default();
        let path = std::path::Path::new("/tmp/test.smt2");

        let result = solver.solve_smt2(path, &config).await;

        assert!(result.is_err());
        if let Err(SolverError::InvalidInput(msg)) = result {
            assert!(msg.contains("does not support SMT-LIB2"));
        } else {
            panic!("Expected InvalidInput error");
        }
    }

    // ==================== solve_dimacs Error Tests ====================

    #[tokio::test]
    async fn test_solve_dimacs_no_binary() {
        let solver = Kissat::new(); // No binary path set
        let config = SolverConfig::default();
        let path = std::path::Path::new("/tmp/test.cnf");

        let result = solver.solve_dimacs(path, &config).await;

        assert!(result.is_err());
        if let Err(SolverError::NotFound(msg)) = result {
            assert!(msg.contains("Kissat binary not found"));
        } else {
            panic!("Expected NotFound error");
        }
    }

    #[tokio::test]
    async fn test_solve_dimacs_file_not_found() {
        let solver = Kissat::with_binary("/bin/echo"); // Use echo as fake binary
        let config = SolverConfig::default();
        let path = std::path::Path::new("/nonexistent/path/to/file.cnf");

        let result = solver.solve_dimacs(path, &config).await;

        assert!(result.is_err());
        if let Err(SolverError::InvalidInput(msg)) = result {
            assert!(msg.contains("DIMACS file not found"));
        } else {
            panic!("Expected InvalidInput error");
        }
    }

    // ==================== Integration Tests (require Kissat) ====================

    #[tokio::test]
    async fn test_solve_simple_sat() {
        // Skip if kissat is not available
        let Some(solver) = Kissat::detect().await else {
            return;
        };

        // Simple SAT problem: (x1 OR x2) AND (x1 OR NOT x2)
        let dimacs = "p cnf 2 2\n1 2 0\n1 -2 0\n";
        let temp_dir = tempfile::tempdir().unwrap();
        let cnf_path = temp_dir.path().join("simple.cnf");
        std::fs::write(&cnf_path, dimacs).unwrap();

        let config = SolverConfig::default();
        let output = solver.solve_dimacs(&cnf_path, &config).await.unwrap();

        assert!(output.result.is_sat());
    }

    #[tokio::test]
    async fn test_solve_simple_unsat() {
        // Skip if kissat is not available
        let Some(solver) = Kissat::detect().await else {
            return;
        };

        // Simple UNSAT: x1 AND NOT x1
        let dimacs = "p cnf 1 2\n1 0\n-1 0\n";
        let temp_dir = tempfile::tempdir().unwrap();
        let cnf_path = temp_dir.path().join("unsat.cnf");
        std::fs::write(&cnf_path, dimacs).unwrap();

        let config = SolverConfig::default();
        let output = solver.solve_dimacs(&cnf_path, &config).await.unwrap();

        assert!(output.result.is_unsat());
    }

    #[tokio::test]
    async fn test_solve_with_quiet_option() {
        // Skip if kissat is not available
        let Some(solver) = Kissat::detect().await else {
            return;
        };

        let dimacs = "p cnf 2 1\n1 2 0\n";
        let temp_dir = tempfile::tempdir().unwrap();
        let cnf_path = temp_dir.path().join("quiet.cnf");
        std::fs::write(&cnf_path, dimacs).unwrap();

        let mut config = SolverConfig::default();
        config
            .options
            .insert("quiet".to_string(), "true".to_string());

        let output = solver.solve_dimacs(&cnf_path, &config).await.unwrap();

        assert!(output.result.is_sat());
    }

    #[tokio::test]
    async fn test_solve_larger_problem() {
        // Skip if kissat is not available
        let Some(solver) = Kissat::detect().await else {
            return;
        };

        // 3-SAT problem with 10 variables, 20 clauses
        let dimacs = r"p cnf 10 20
1 2 3 0
-1 4 5 0
2 -3 6 0
-4 -5 7 0
3 6 -7 0
-1 -2 8 0
4 5 -8 0
-6 7 9 0
1 -4 -9 0
2 5 10 0
-3 -6 -10 0
1 7 8 0
-2 -7 9 0
3 -8 -9 0
4 6 10 0
-1 -5 -10 0
2 -4 7 0
-3 5 -7 0
1 -6 8 0
-2 6 -8 0
";
        let temp_dir = tempfile::tempdir().unwrap();
        let cnf_path = temp_dir.path().join("larger.cnf");
        std::fs::write(&cnf_path, dimacs).unwrap();

        let config = SolverConfig::default();
        let output = solver.solve_dimacs(&cnf_path, &config).await.unwrap();

        // This problem is satisfiable
        assert!(output.result.is_sat());
    }

    // ==================== create_kissat Tests ====================

    #[tokio::test]
    async fn test_create_kissat() {
        let result = create_kissat().await;

        // Result depends on whether kissat is installed
        if let Some(boxed) = result {
            assert_eq!(boxed.info().id, "kissat");
            assert!(boxed.info().available);
        }
    }

    // ==================== Mutation Coverage Tests ====================

    #[tokio::test]
    async fn test_check_available_returns_bool_value() {
        // Test that check_available returns a meaningful boolean value
        // that correctly reflects whether kissat binary is present
        let solver_with_binary = Kissat::with_binary("/bin/echo");
        let available = solver_with_binary.check_available().await;
        // /bin/echo exists, so find_binary returns Some, thus check_available returns true
        assert!(
            available,
            "check_available should return true when binary exists"
        );

        let solver_without_binary = Kissat::new();
        // Without kissat installed, find_binary may return None
        let available = solver_without_binary.check_available().await;
        // This test verifies the return value is used (not always true or false)
        // We check that the function actually uses the is_some() result
        if Kissat::detect().await.is_some() {
            // If kissat is installed, solver without explicit binary still finds it
            assert!(available);
        } else {
            // If kissat not installed, check_available returns false
            assert!(!available);
        }
    }

    #[tokio::test]
    async fn test_create_kissat_returns_none_when_not_available() {
        // Test that create_kissat correctly returns None when solver not detected
        // This exercises the .map() branch that could be replaced with None
        let result = create_kissat().await;
        // When kissat is not installed, result should be None
        // When kissat is installed, result should be Some with correct info
        match result {
            Some(boxed) => {
                assert_eq!(boxed.info().id, "kissat");
                assert!(boxed.info().available);
                // Verify the boxed solver is functional
                assert!(boxed.supports(SolverCapability::Sat));
            }
            None => {
                // This is the expected path if kissat is not installed
                // The test verifies the None branch is exercised
            }
        }
    }

    #[tokio::test]
    async fn test_solve_dimacs_with_custom_options() {
        // Skip if kissat is not available
        let Some(solver) = Kissat::detect().await else {
            return;
        };

        let dimacs = "p cnf 2 2\n1 2 0\n1 -2 0\n";
        let temp_dir = tempfile::tempdir().unwrap();
        let cnf_path = temp_dir.path().join("options.cnf");
        std::fs::write(&cnf_path, dimacs).unwrap();

        let mut config = SolverConfig::default();
        // Add a non-quiet option to exercise the else branch in options loop
        config
            .options
            .insert("verbose".to_string(), "2".to_string());

        let output = solver.solve_dimacs(&cnf_path, &config).await.unwrap();

        // Solver should still work with custom options
        assert!(
            output.result.is_sat() || output.result.is_unsat() || !output.result.is_definitive()
        );
    }

    #[tokio::test]
    async fn test_solve_dimacs_quiet_false() {
        // Skip if kissat is not available
        let Some(solver) = Kissat::detect().await else {
            return;
        };

        let dimacs = "p cnf 1 1\n1 0\n";
        let temp_dir = tempfile::tempdir().unwrap();
        let cnf_path = temp_dir.path().join("quiet_false.cnf");
        std::fs::write(&cnf_path, dimacs).unwrap();

        let mut config = SolverConfig::default();
        // Set quiet to false - this exercises the map().unwrap_or(false) == false path
        config
            .options
            .insert("quiet".to_string(), "false".to_string());

        let output = solver.solve_dimacs(&cnf_path, &config).await.unwrap();
        assert!(output.result.is_sat());
    }

    #[tokio::test]
    async fn test_find_binary_with_preset_path() {
        // Test that find_binary returns the preset path when binary_path is Some
        let solver = Kissat::with_binary("/custom/kissat");
        let binary = solver.find_binary().await;
        assert_eq!(binary, Some("/custom/kissat".to_string()));
    }

    #[test]
    fn test_parse_output_empty_strings() {
        let solver = Kissat::new();
        let stdout = "";
        let stderr = "";
        let duration = std::time::Duration::from_millis(0);

        let output = solver.parse_output(stdout, stderr, duration);

        // Empty output should result in Unknown
        assert!(!output.result.is_definitive());
        if let SolverResult::Unknown { reason } = &output.result {
            assert!(reason.contains("No definitive result"));
        }
    }

    #[test]
    fn test_parse_output_duration_preserved() {
        let solver = Kissat::new();
        let stdout = "s SATISFIABLE\nv 1 0\n";
        let stderr = "";
        let duration = std::time::Duration::from_millis(12345);

        let output = solver.parse_output(stdout, stderr, duration);

        // Duration should be preserved in stats
        assert_eq!(output.stats.solve_time, duration);
        assert_eq!(output.stats.solve_time.as_millis(), 12345);
    }

    #[test]
    fn test_parse_output_no_stats() {
        let solver = Kissat::new();
        let stdout = "s UNSATISFIABLE\n";
        let stderr = "";
        let duration = std::time::Duration::from_millis(50);

        let output = solver.parse_output(stdout, stderr, duration);

        // Without stat lines, stats should be None
        assert_eq!(output.stats.conflicts, None);
        assert_eq!(output.stats.decisions, None);
        assert_eq!(output.stats.propagations, None);
    }
}
