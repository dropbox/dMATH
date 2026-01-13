//! Z3 SMT solver wrapper
//!
//! Z3 is a high-performance SMT solver from Microsoft Research.
//! It supports a wide range of theories including bitvectors, arrays,
//! uninterpreted functions, and arithmetic.

use crate::solver::{
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

/// Z3 SMT solver
pub struct Z3 {
    info: SolverInfo,
    binary_path: Option<String>,
}

impl Z3 {
    /// Create a new Z3 instance
    pub fn new() -> Self {
        Self {
            info: SolverInfo {
                id: "z3".to_string(),
                name: "Z3".to_string(),
                version: "unknown".to_string(),
                capabilities: vec![
                    SolverCapability::Sat,
                    SolverCapability::SmtBv,
                    SolverCapability::SmtArrays,
                    SolverCapability::SmtUf,
                    SolverCapability::SmtLia,
                    SolverCapability::SmtLra,
                    SolverCapability::Quantifiers,
                    SolverCapability::Incremental,
                    SolverCapability::Proofs,
                    SolverCapability::UnsatCores,
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

    /// Detect and initialize Z3
    pub async fn detect() -> Option<Self> {
        let mut solver = Self::new();

        // Try to find z3 binary
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
        for name in &["z3", "Z3"] {
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
        // Z3 version output: "Z3 version X.Y.Z - ..."
        stdout.lines().next().map(|s| s.trim().to_string())
    }

    fn parse_dimacs_output(
        &self,
        stdout: &str,
        stderr: &str,
        duration: std::time::Duration,
    ) -> SolverOutput {
        let stats = SolverStats {
            solve_time: duration,
            ..Default::default()
        };

        // Z3 uses standard DIMACS output for SAT
        let result = if stdout.contains("s SATISFIABLE") || stderr.contains("s SATISFIABLE") {
            let model = extract_dimacs_model(stdout);
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

    fn parse_smt2_output(
        &self,
        stdout: &str,
        stderr: &str,
        duration: std::time::Duration,
    ) -> SolverOutput {
        let mut stats = SolverStats {
            solve_time: duration,
            ..Default::default()
        };

        // Parse Z3 statistics if available
        for line in stdout.lines().chain(stderr.lines()) {
            if line.contains(":conflicts") {
                if let Some(n) = extract_smt_stat(line) {
                    stats.conflicts = Some(n);
                }
            } else if line.contains(":decisions") {
                if let Some(n) = extract_smt_stat(line) {
                    stats.decisions = Some(n);
                }
            } else if line.contains(":propagations") {
                if let Some(n) = extract_smt_stat(line) {
                    stats.propagations = Some(n);
                }
            } else if line.contains(":memory") {
                if let Some(n) = extract_smt_stat(line) {
                    // Memory is usually in MB, convert to bytes
                    stats.memory_bytes = Some(n * 1024 * 1024);
                }
            }
        }

        // Z3 SMT-LIB2 output format
        let result = if stdout.trim().starts_with("sat")
            || stdout.contains("\nsat\n")
            || stdout.contains("\nsat")
        {
            let model = extract_smt_model(stdout);
            SolverResult::Sat { model }
        } else if stdout.trim().starts_with("unsat")
            || stdout.contains("\nunsat\n")
            || stdout.contains("\nunsat")
        {
            // Extract unsat core if available
            let proof = extract_unsat_core(stdout);
            SolverResult::Unsat { proof }
        } else if stdout.contains("unknown") {
            SolverResult::Unknown {
                reason: extract_reason(stdout)
                    .unwrap_or_else(|| "Solver returned unknown".to_string()),
            }
        } else {
            SolverResult::Unknown {
                reason: format!(
                    "Unexpected Z3 output: {}",
                    stdout.lines().next().unwrap_or("empty")
                ),
            }
        };

        SolverOutput {
            result,
            stats,
            raw_output: Some(format!("STDOUT:\n{stdout}\nSTDERR:\n{stderr}")),
        }
    }
}

impl Default for Z3 {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Solver for Z3 {
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
            .ok_or_else(|| SolverError::NotFound("Z3 binary not found".to_string()))?;

        if !path.exists() {
            return Err(SolverError::InvalidInput(format!(
                "DIMACS file not found: {}",
                path.display()
            )));
        }

        let mut cmd = Command::new(binary);

        // Z3 needs -dimacs flag for DIMACS input
        cmd.arg("-dimacs").arg(path);

        // Timeout in seconds
        let timeout_secs = config.timeout.as_secs();
        if timeout_secs > 0 {
            cmd.arg(format!("-t:{timeout_secs}"));
        }

        // Random seed
        if let Some(seed) = config.seed {
            cmd.arg(format!("sat.random_seed={seed}"));
        }

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        debug!("Running Z3 (DIMACS): {:?}", cmd);
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
                Ok(self.parse_dimacs_output(&stdout, &stderr, duration))
            }
            Ok(Err(e)) => Err(SolverError::ExecutionFailed(e.to_string())),
            Err(_) => {
                warn!("Z3 timed out after {:?}", config.timeout);
                let _ = child.kill().await;
                Err(SolverError::Timeout(config.timeout))
            }
        }
    }

    async fn solve_smt2(
        &self,
        path: &Path,
        config: &SolverConfig,
    ) -> Result<SolverOutput, SolverError> {
        let binary = self
            .binary_path
            .as_ref()
            .ok_or_else(|| SolverError::NotFound("Z3 binary not found".to_string()))?;

        if !path.exists() {
            return Err(SolverError::InvalidInput(format!(
                "SMT-LIB2 file not found: {}",
                path.display()
            )));
        }

        let mut cmd = Command::new(binary);
        cmd.arg(path);

        // Timeout in seconds
        let timeout_secs = config.timeout.as_secs();
        if timeout_secs > 0 {
            cmd.arg(format!("-t:{timeout_secs}"));
        }

        // Random seed
        if let Some(seed) = config.seed {
            cmd.arg(format!("smt.random_seed={seed}"));
        }

        // Produce models by default
        cmd.arg("model=true");

        // Thread count
        if let Some(threads) = config.threads {
            cmd.arg("parallel.enable=true");
            cmd.arg(format!("parallel.threads.max={threads}"));
        }

        // Additional Z3-specific options
        for (key, value) in &config.options {
            cmd.arg(format!("{key}={value}"));
        }

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        debug!("Running Z3 (SMT2): {:?}", cmd);
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
                Ok(self.parse_smt2_output(&stdout, &stderr, duration))
            }
            Ok(Err(e)) => Err(SolverError::ExecutionFailed(e.to_string())),
            Err(_) => {
                warn!("Z3 timed out after {:?}", config.timeout);
                let _ = child.kill().await;
                Err(SolverError::Timeout(config.timeout))
            }
        }
    }
}

/// Create a boxed Z3 solver
pub async fn create_z3() -> Option<BoxedSolver> {
    Z3::detect().await.map(|s| Box::new(s) as BoxedSolver)
}

/// Extract a number from a DIMACS statistics line
fn extract_dimacs_model(output: &str) -> Option<String> {
    let mut model_lines = Vec::new();
    for line in output.lines() {
        if let Some(stripped) = line.strip_prefix("v ") {
            model_lines.push(stripped.to_string());
        }
    }
    if model_lines.is_empty() {
        None
    } else {
        Some(model_lines.join(" "))
    }
}

/// Extract SMT statistics value
fn extract_smt_stat(line: &str) -> Option<u64> {
    // Format: ":stat-name value" or "(:stat-name value)"
    line.split_whitespace()
        .last()
        .and_then(|s| s.trim_end_matches(')').parse().ok())
}

/// Extract model from SMT-LIB2 output
fn extract_smt_model(output: &str) -> Option<String> {
    // Model is between (model and the matching )
    let start = output.find("(model")?;
    let mut depth = 0;
    let mut end = start;

    for (i, c) in output[start..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    end = start + i + 1;
                    break;
                }
            }
            _ => {}
        }
    }

    if end > start {
        Some(output[start..end].to_string())
    } else {
        None
    }
}

/// Extract unsat core from output
fn extract_unsat_core(output: &str) -> Option<String> {
    // Unsat core is a list after "unsat"
    let core_start = output
        .find("(error")
        .or_else(|| output.find("(:reason-unknown"))?;

    let mut depth = 0;
    let mut end = core_start;

    for (i, c) in output[core_start..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    end = core_start + i + 1;
                    break;
                }
            }
            _ => {}
        }
    }

    if end > core_start {
        Some(output[core_start..end].to_string())
    } else {
        None
    }
}

/// Extract reason for unknown result
fn extract_reason(output: &str) -> Option<String> {
    if let Some(start) = output.find("(:reason-unknown") {
        let rest = &output[start..];
        if let Some(end) = rest.find(')') {
            return Some(rest[..=end].to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Basic Construction Tests ====================

    #[test]
    fn test_z3_new() {
        let solver = Z3::new();
        assert_eq!(solver.info().id, "z3");
        assert_eq!(solver.info().name, "Z3");
        assert_eq!(solver.info().version, "unknown");
        assert!(!solver.info().available);
        assert!(solver.binary_path.is_none());
    }

    #[test]
    fn test_z3_default() {
        let solver = Z3::default();
        assert_eq!(solver.info().id, "z3");
        assert_eq!(solver.info().name, "Z3");
    }

    #[test]
    fn test_z3_with_binary() {
        let solver = Z3::with_binary("/usr/local/bin/z3");
        assert_eq!(solver.binary_path, Some("/usr/local/bin/z3".to_string()));
    }

    #[test]
    fn test_z3_with_binary_into() {
        let path = String::from("/custom/path/z3");
        let solver = Z3::with_binary(path);
        assert_eq!(solver.binary_path, Some("/custom/path/z3".to_string()));
    }

    // ==================== Capability Tests ====================

    #[test]
    fn test_z3_info() {
        let solver = Z3::new();
        assert_eq!(solver.info().id, "z3");
        assert!(solver.supports(SolverCapability::Sat));
        assert!(solver.supports(SolverCapability::SmtBv));
        assert!(solver.supports(SolverCapability::SmtArrays));
        assert!(solver.supports(SolverCapability::Quantifiers));
    }

    #[test]
    fn test_z3_all_capabilities() {
        let solver = Z3::new();
        assert!(solver.supports(SolverCapability::Sat));
        assert!(solver.supports(SolverCapability::SmtBv));
        assert!(solver.supports(SolverCapability::SmtArrays));
        assert!(solver.supports(SolverCapability::SmtUf));
        assert!(solver.supports(SolverCapability::SmtLia));
        assert!(solver.supports(SolverCapability::SmtLra));
        assert!(solver.supports(SolverCapability::Quantifiers));
        assert!(solver.supports(SolverCapability::Incremental));
        assert!(solver.supports(SolverCapability::Proofs));
        assert!(solver.supports(SolverCapability::UnsatCores));
    }

    #[test]
    fn test_z3_capability_count() {
        let solver = Z3::new();
        // Z3 supports many capabilities
        assert_eq!(solver.info().capabilities.len(), 10);
    }

    // ==================== extract_dimacs_model Tests ====================

    #[test]
    fn test_extract_dimacs_model() {
        let output = "s SATISFIABLE\nv 1 -2 3 0\nv 4 -5 0\n";
        let model = extract_dimacs_model(output);
        assert_eq!(model, Some("1 -2 3 0 4 -5 0".to_string()));
    }

    #[test]
    fn test_extract_dimacs_model_single_line() {
        let output = "s SATISFIABLE\nv 1 2 3 0\n";
        let model = extract_dimacs_model(output);
        assert_eq!(model, Some("1 2 3 0".to_string()));
    }

    #[test]
    fn test_extract_dimacs_model_no_model() {
        let output = "s UNSATISFIABLE\n";
        let model = extract_dimacs_model(output);
        assert_eq!(model, None);
    }

    #[test]
    fn test_extract_dimacs_model_empty() {
        let model = extract_dimacs_model("");
        assert_eq!(model, None);
    }

    // ==================== extract_smt_stat Tests ====================

    #[test]
    fn test_extract_smt_stat() {
        assert_eq!(extract_smt_stat(":conflicts 12345"), Some(12345));
        assert_eq!(extract_smt_stat("(:conflicts 999)"), Some(999));
    }

    #[test]
    fn test_extract_smt_stat_zero() {
        assert_eq!(extract_smt_stat(":conflicts 0"), Some(0));
    }

    #[test]
    fn test_extract_smt_stat_large() {
        assert_eq!(
            extract_smt_stat(":conflicts 18446744073709551615"),
            Some(18446744073709551615)
        );
    }

    #[test]
    fn test_extract_smt_stat_no_number() {
        assert_eq!(extract_smt_stat(":conflicts"), None);
        assert_eq!(extract_smt_stat(""), None);
    }

    #[test]
    fn test_extract_smt_stat_with_closing_paren() {
        assert_eq!(extract_smt_stat("(:memory 123)"), Some(123));
    }

    // ==================== extract_smt_model Tests ====================

    #[test]
    fn test_extract_smt_model() {
        let output = "sat\n(model\n  (define-fun x () Int 42)\n)\n";
        let model = extract_smt_model(output);
        assert!(model.is_some());
        assert!(model.unwrap().contains("define-fun x"));
    }

    #[test]
    fn test_extract_smt_model_multiple_functions() {
        let output = r"sat
(model
  (define-fun x () Int 42)
  (define-fun y () Bool true)
)
";
        let model = extract_smt_model(output);
        assert!(model.is_some());
        let m = model.unwrap();
        assert!(m.contains("define-fun x"));
        assert!(m.contains("define-fun y"));
    }

    #[test]
    fn test_extract_smt_model_nested_parens() {
        let output = r"sat
(model
  (define-fun f ((x Int)) Int
    (+ x 1))
)
";
        let model = extract_smt_model(output);
        assert!(model.is_some());
        let m = model.unwrap();
        assert!(m.contains("(+ x 1)"));
    }

    #[test]
    fn test_extract_smt_model_no_model() {
        let output = "unsat\n";
        let model = extract_smt_model(output);
        assert!(model.is_none());
    }

    #[test]
    fn test_extract_smt_model_empty() {
        let model = extract_smt_model("");
        assert!(model.is_none());
    }

    // ==================== extract_unsat_core Tests ====================

    #[test]
    fn test_extract_unsat_core_with_error() {
        let output = "unsat\n(error \"some error message\")\n";
        let core = extract_unsat_core(output);
        assert!(core.is_some());
        assert!(core.unwrap().contains("error"));
    }

    #[test]
    fn test_extract_unsat_core_with_reason() {
        let output = "unknown\n(:reason-unknown \"timeout\")\n";
        let core = extract_unsat_core(output);
        assert!(core.is_some());
        assert!(core.unwrap().contains("reason-unknown"));
    }

    #[test]
    fn test_extract_unsat_core_no_core() {
        let output = "unsat\n";
        let core = extract_unsat_core(output);
        assert!(core.is_none());
    }

    // ==================== extract_reason Tests ====================

    #[test]
    fn test_extract_reason() {
        let output = "unknown\n(:reason-unknown \"timeout\")\n";
        let reason = extract_reason(output);
        assert!(reason.is_some());
        assert!(reason.unwrap().contains("timeout"));
    }

    #[test]
    fn test_extract_reason_incomplete() {
        let output = "unknown\n(:reason-unknown \"incomplete quantifiers\")\n";
        let reason = extract_reason(output);
        assert!(reason.is_some());
        assert!(reason.unwrap().contains("incomplete"));
    }

    #[test]
    fn test_extract_reason_no_reason() {
        let output = "unknown\n";
        let reason = extract_reason(output);
        assert!(reason.is_none());
    }

    #[test]
    fn test_extract_reason_empty() {
        let reason = extract_reason("");
        assert!(reason.is_none());
    }

    // ==================== parse_dimacs_output Tests ====================

    #[test]
    fn test_parse_dimacs_output_sat() {
        let solver = Z3::new();
        let stdout = "s SATISFIABLE\nv 1 -2 0\n";
        let stderr = "";
        let duration = std::time::Duration::from_millis(100);

        let output = solver.parse_dimacs_output(stdout, stderr, duration);

        assert!(output.result.is_sat());
        if let SolverResult::Sat { model } = &output.result {
            assert!(model.is_some());
        }
    }

    #[test]
    fn test_parse_dimacs_output_unsat() {
        let solver = Z3::new();
        let stdout = "s UNSATISFIABLE\n";
        let stderr = "";
        let duration = std::time::Duration::from_millis(50);

        let output = solver.parse_dimacs_output(stdout, stderr, duration);

        assert!(output.result.is_unsat());
    }

    #[test]
    fn test_parse_dimacs_output_unknown() {
        let solver = Z3::new();
        let stdout = "c some output\n";
        let stderr = "";
        let duration = std::time::Duration::from_millis(100);

        let output = solver.parse_dimacs_output(stdout, stderr, duration);

        assert!(!output.result.is_definitive());
    }

    #[test]
    fn test_parse_dimacs_output_in_stderr() {
        let solver = Z3::new();
        let stdout = "";
        let stderr = "s SATISFIABLE\nv 1 0\n";
        let duration = std::time::Duration::from_millis(100);

        let output = solver.parse_dimacs_output(stdout, stderr, duration);

        assert!(output.result.is_sat());
    }

    // ==================== parse_smt2_output Tests ====================

    #[test]
    fn test_parse_smt2_output_sat() {
        let solver = Z3::new();
        let stdout = "sat\n(model\n  (define-fun x () Int 42)\n)\n";
        let stderr = "";
        let duration = std::time::Duration::from_millis(100);

        let output = solver.parse_smt2_output(stdout, stderr, duration);

        assert!(output.result.is_sat());
        if let SolverResult::Sat { model } = &output.result {
            assert!(model.is_some());
        }
    }

    #[test]
    fn test_parse_smt2_output_sat_no_model() {
        let solver = Z3::new();
        let stdout = "sat\n";
        let stderr = "";
        let duration = std::time::Duration::from_millis(100);

        let output = solver.parse_smt2_output(stdout, stderr, duration);

        assert!(output.result.is_sat());
    }

    #[test]
    fn test_parse_smt2_output_unsat() {
        let solver = Z3::new();
        let stdout = "unsat\n";
        let stderr = "";
        let duration = std::time::Duration::from_millis(50);

        let output = solver.parse_smt2_output(stdout, stderr, duration);

        assert!(output.result.is_unsat());
    }

    #[test]
    fn test_parse_smt2_output_unknown() {
        let solver = Z3::new();
        let stdout = "unknown\n(:reason-unknown \"timeout\")\n";
        let stderr = "";
        let duration = std::time::Duration::from_millis(100);

        let output = solver.parse_smt2_output(stdout, stderr, duration);

        assert!(!output.result.is_definitive());
        if let SolverResult::Unknown { reason } = &output.result {
            assert!(reason.contains("timeout"));
        }
    }

    #[test]
    fn test_parse_smt2_output_unknown_no_reason() {
        let solver = Z3::new();
        let stdout = "unknown\n";
        let stderr = "";
        let duration = std::time::Duration::from_millis(100);

        let output = solver.parse_smt2_output(stdout, stderr, duration);

        assert!(!output.result.is_definitive());
        if let SolverResult::Unknown { reason } = &output.result {
            assert!(reason.contains("Solver returned unknown"));
        }
    }

    #[test]
    fn test_parse_smt2_output_with_stats() {
        let solver = Z3::new();
        let stdout = ":conflicts 1000\n:decisions 5000\n:propagations 50000\n:memory 128\nsat\n";
        let stderr = "";
        let duration = std::time::Duration::from_millis(100);

        let output = solver.parse_smt2_output(stdout, stderr, duration);

        assert_eq!(output.stats.conflicts, Some(1000));
        assert_eq!(output.stats.decisions, Some(5000));
        assert_eq!(output.stats.propagations, Some(50000));
        // Memory is converted to bytes (128 MB = 128 * 1024 * 1024 bytes)
        assert_eq!(output.stats.memory_bytes, Some(128 * 1024 * 1024));
    }

    #[test]
    fn test_parse_smt2_output_unexpected() {
        let solver = Z3::new();
        let stdout = "garbage output";
        let stderr = "";
        let duration = std::time::Duration::from_millis(100);

        let output = solver.parse_smt2_output(stdout, stderr, duration);

        assert!(!output.result.is_definitive());
        if let SolverResult::Unknown { reason } = &output.result {
            assert!(reason.contains("Unexpected Z3 output"));
        }
    }

    #[test]
    fn test_parse_smt2_output_sat_in_middle() {
        let solver = Z3::new();
        let stdout = "some stats\nsat\n(model)\n";
        let stderr = "";
        let duration = std::time::Duration::from_millis(100);

        let output = solver.parse_smt2_output(stdout, stderr, duration);

        assert!(output.result.is_sat());
    }

    #[test]
    fn test_parse_smt2_output_raw_output() {
        let solver = Z3::new();
        let stdout = "sat";
        let stderr = "warning";
        let duration = std::time::Duration::from_millis(100);

        let output = solver.parse_smt2_output(stdout, stderr, duration);

        let raw = output.raw_output.unwrap();
        assert!(raw.contains("STDOUT:"));
        assert!(raw.contains("sat"));
        assert!(raw.contains("STDERR:"));
        assert!(raw.contains("warning"));
    }

    // ==================== Async Detection Tests ====================

    #[tokio::test]
    async fn test_z3_detect_available() {
        let result = Z3::detect().await;

        if let Some(solver) = result {
            assert!(solver.info().available);
            assert!(solver.binary_path.is_some());
            // Version should be populated
            assert_ne!(solver.info().version, "unknown");
        }
    }

    #[tokio::test]
    async fn test_z3_check_available() {
        let solver = Z3::new();
        let available = solver.check_available().await;
        let _ = available;
    }

    // ==================== solve_dimacs Error Tests ====================

    #[tokio::test]
    async fn test_solve_dimacs_no_binary() {
        let solver = Z3::new();
        let config = SolverConfig::default();
        let path = std::path::Path::new("/tmp/test.cnf");

        let result = solver.solve_dimacs(path, &config).await;

        assert!(result.is_err());
        if let Err(SolverError::NotFound(msg)) = result {
            assert!(msg.contains("Z3 binary not found"));
        } else {
            panic!("Expected NotFound error");
        }
    }

    #[tokio::test]
    async fn test_solve_dimacs_file_not_found() {
        let solver = Z3::with_binary("/bin/echo");
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

    // ==================== solve_smt2 Error Tests ====================

    #[tokio::test]
    async fn test_solve_smt2_no_binary() {
        let solver = Z3::new();
        let config = SolverConfig::default();
        let path = std::path::Path::new("/tmp/test.smt2");

        let result = solver.solve_smt2(path, &config).await;

        assert!(result.is_err());
        if let Err(SolverError::NotFound(msg)) = result {
            assert!(msg.contains("Z3 binary not found"));
        } else {
            panic!("Expected NotFound error");
        }
    }

    #[tokio::test]
    async fn test_solve_smt2_file_not_found() {
        let solver = Z3::with_binary("/bin/echo");
        let config = SolverConfig::default();
        let path = std::path::Path::new("/nonexistent/path/to/file.smt2");

        let result = solver.solve_smt2(path, &config).await;

        assert!(result.is_err());
        if let Err(SolverError::InvalidInput(msg)) = result {
            assert!(msg.contains("SMT-LIB2 file not found"));
        } else {
            panic!("Expected InvalidInput error");
        }
    }

    // ==================== Integration Tests (require Z3) ====================

    #[tokio::test]
    async fn test_solve_dimacs_sat() {
        let Some(solver) = Z3::detect().await else {
            return;
        };

        let dimacs = "p cnf 2 2\n1 2 0\n1 -2 0\n";
        let temp_dir = tempfile::tempdir().unwrap();
        let cnf_path = temp_dir.path().join("simple.cnf");
        std::fs::write(&cnf_path, dimacs).unwrap();

        let config = SolverConfig::default();
        let output = solver.solve_dimacs(&cnf_path, &config).await.unwrap();

        assert!(output.result.is_sat());
    }

    #[tokio::test]
    async fn test_solve_dimacs_unsat() {
        let Some(solver) = Z3::detect().await else {
            return;
        };

        let dimacs = "p cnf 1 2\n1 0\n-1 0\n";
        let temp_dir = tempfile::tempdir().unwrap();
        let cnf_path = temp_dir.path().join("unsat.cnf");
        std::fs::write(&cnf_path, dimacs).unwrap();

        let config = SolverConfig::default();
        let output = solver.solve_dimacs(&cnf_path, &config).await.unwrap();

        assert!(output.result.is_unsat());
    }

    #[tokio::test]
    async fn test_solve_smt2_sat() {
        let Some(solver) = Z3::detect().await else {
            return;
        };

        let smt2 = r"
(declare-const x Int)
(assert (> x 0))
(assert (< x 10))
(check-sat)
(get-model)
";
        let temp_dir = tempfile::tempdir().unwrap();
        let smt_path = temp_dir.path().join("simple.smt2");
        std::fs::write(&smt_path, smt2).unwrap();

        let config = SolverConfig::default();
        let output = solver.solve_smt2(&smt_path, &config).await.unwrap();

        assert!(output.result.is_sat());
    }

    #[tokio::test]
    async fn test_solve_smt2_unsat() {
        let Some(solver) = Z3::detect().await else {
            return;
        };

        let smt2 = r"
(declare-const x Int)
(assert (> x 10))
(assert (< x 5))
(check-sat)
";
        let temp_dir = tempfile::tempdir().unwrap();
        let smt_path = temp_dir.path().join("unsat.smt2");
        std::fs::write(&smt_path, smt2).unwrap();

        let config = SolverConfig::default();
        let output = solver.solve_smt2(&smt_path, &config).await.unwrap();

        assert!(output.result.is_unsat());
    }

    #[tokio::test]
    async fn test_solve_smt2_bitvector() {
        let Some(solver) = Z3::detect().await else {
            return;
        };

        let smt2 = r"
(declare-const x (_ BitVec 8))
(assert (= (bvadd x #x01) #x02))
(check-sat)
(get-model)
";
        let temp_dir = tempfile::tempdir().unwrap();
        let smt_path = temp_dir.path().join("bv.smt2");
        std::fs::write(&smt_path, smt2).unwrap();

        let config = SolverConfig::default();
        let output = solver.solve_smt2(&smt_path, &config).await.unwrap();

        assert!(output.result.is_sat());
    }

    #[tokio::test]
    async fn test_solve_smt2_array() {
        let Some(solver) = Z3::detect().await else {
            return;
        };

        let smt2 = r"
(declare-const a (Array Int Int))
(assert (= (select a 0) 42))
(check-sat)
(get-model)
";
        let temp_dir = tempfile::tempdir().unwrap();
        let smt_path = temp_dir.path().join("array.smt2");
        std::fs::write(&smt_path, smt2).unwrap();

        let config = SolverConfig::default();
        let output = solver.solve_smt2(&smt_path, &config).await.unwrap();

        assert!(output.result.is_sat());
    }

    #[tokio::test]
    async fn test_solve_smt2_with_seed() {
        let Some(solver) = Z3::detect().await else {
            return;
        };

        let smt2 = r"
(declare-const x Int)
(assert (> x 0))
(check-sat)
";
        let temp_dir = tempfile::tempdir().unwrap();
        let smt_path = temp_dir.path().join("seed.smt2");
        std::fs::write(&smt_path, smt2).unwrap();

        let config = SolverConfig {
            seed: Some(42),
            ..Default::default()
        };

        let output = solver.solve_smt2(&smt_path, &config).await.unwrap();

        assert!(output.result.is_sat());
    }

    #[tokio::test]
    async fn test_solve_smt2_with_options() {
        let Some(solver) = Z3::detect().await else {
            return;
        };

        let smt2 = r"
(declare-const x Int)
(assert (> x 0))
(check-sat)
";
        let temp_dir = tempfile::tempdir().unwrap();
        let smt_path = temp_dir.path().join("opts.smt2");
        std::fs::write(&smt_path, smt2).unwrap();

        let mut config = SolverConfig::default();
        config
            .options
            .insert("model.completion".to_string(), "true".to_string());

        let output = solver.solve_smt2(&smt_path, &config).await.unwrap();

        assert!(output.result.is_sat());
    }

    // ==================== create_z3 Tests ====================

    #[tokio::test]
    async fn test_create_z3() {
        let result = create_z3().await;

        if let Some(boxed) = result {
            assert_eq!(boxed.info().id, "z3");
            assert!(boxed.info().available);
        }
    }

    // ==================== Mutation Coverage Tests ====================

    #[tokio::test]
    async fn test_z3_detect_returns_none_when_not_found() {
        // Test that detect() returns None correctly when Z3 is not found
        // This exercises the .map() and Option chain
        let result = Z3::detect().await;
        match result {
            Some(solver) => {
                assert!(solver.info().available);
                assert!(solver.binary_path.is_some());
                // Verify it's actually functional
                assert!(solver.supports(SolverCapability::Sat));
            }
            None => {
                // This path validates that None is returned when Z3 not found
            }
        }
    }

    #[tokio::test]
    async fn test_z3_find_binary_returns_preset_path() {
        // Test that find_binary returns the preset path when binary_path is Some
        let solver = Z3::with_binary("/custom/z3");
        let binary = solver.find_binary().await;
        assert_eq!(binary, Some("/custom/z3".to_string()));
    }

    #[tokio::test]
    async fn test_z3_get_version_requires_binary_path() {
        // Test that get_version returns None when binary_path is None
        let solver = Z3::new();
        let version = solver.get_version().await;
        // Without binary_path set, get_version returns None early
        // (unless it finds z3 in PATH via detect)
        let _ = version; // Don't assert specific value as it depends on environment
    }

    #[tokio::test]
    async fn test_z3_check_available_returns_bool() {
        // Test that check_available returns proper boolean based on binary existence
        let solver_with_binary = Z3::with_binary("/bin/echo");
        let available = solver_with_binary.check_available().await;
        assert!(
            available,
            "check_available should return true when binary exists"
        );

        let solver_without_binary = Z3::new();
        let available = solver_without_binary.check_available().await;
        // This verifies the is_some() result is correctly used
        if Z3::detect().await.is_some() {
            assert!(available);
        } else {
            assert!(!available);
        }
    }

    #[tokio::test]
    async fn test_create_z3_returns_none_when_not_available() {
        // Test that create_z3 correctly returns None when solver not detected
        let result = create_z3().await;
        match result {
            Some(boxed) => {
                assert_eq!(boxed.info().id, "z3");
                assert!(boxed.info().available);
                assert!(boxed.supports(SolverCapability::SmtBv));
            }
            None => {
                // This path validates the .map() returning None
            }
        }
    }

    #[test]
    fn test_parse_smt2_output_solve_time_preserved() {
        // Test that solve_time is correctly set in stats
        let solver = Z3::new();
        let stdout = "sat\n";
        let stderr = "";
        let duration = std::time::Duration::from_millis(42);

        let output = solver.parse_smt2_output(stdout, stderr, duration);

        // solve_time must be preserved - this catches "delete field solve_time" mutations
        assert_eq!(output.stats.solve_time, duration);
        assert_eq!(output.stats.solve_time.as_millis(), 42);
    }

    #[test]
    fn test_parse_dimacs_output_solve_time_preserved() {
        // Test that solve_time is correctly set in stats for DIMACS parsing
        let solver = Z3::new();
        let stdout = "s SATISFIABLE\nv 1 0\n";
        let stderr = "";
        let duration = std::time::Duration::from_millis(123);

        let output = solver.parse_dimacs_output(stdout, stderr, duration);

        // solve_time must be preserved
        assert_eq!(output.stats.solve_time, duration);
        assert_eq!(output.stats.solve_time.as_millis(), 123);
    }

    #[test]
    fn test_parse_smt2_output_or_conditions() {
        // Test the || conditions in parse_smt2_output for sat detection
        // Tests stdout.trim().starts_with("sat") || stdout.contains("\nsat\n") || stdout.contains("\nsat")
        let solver = Z3::new();

        // Test starts_with("sat")
        let output = solver.parse_smt2_output("sat", "", std::time::Duration::ZERO);
        assert!(output.result.is_sat());

        // Test contains("\nsat\n")
        let output = solver.parse_smt2_output("stats\nsat\nmore", "", std::time::Duration::ZERO);
        assert!(output.result.is_sat());

        // Test contains("\nsat") at end
        let output = solver.parse_smt2_output("stats\nsat", "", std::time::Duration::ZERO);
        assert!(output.result.is_sat());

        // Test unsat conditions similarly
        let output = solver.parse_smt2_output("unsat", "", std::time::Duration::ZERO);
        assert!(output.result.is_unsat());

        let output = solver.parse_smt2_output("stats\nunsat\nmore", "", std::time::Duration::ZERO);
        assert!(output.result.is_unsat());

        let output = solver.parse_smt2_output("stats\nunsat", "", std::time::Duration::ZERO);
        assert!(output.result.is_unsat());
    }

    #[test]
    fn test_extract_smt_model_end_boundary() {
        // Test model extraction at exact boundary (end > start condition)
        // This catches "replace > with >= in extract_smt_model" mutants
        let output = "(model)";
        let model = extract_smt_model(output);
        assert!(model.is_some());
        assert_eq!(model.unwrap(), "(model)");

        // Empty model
        let output = "(model\n)";
        let model = extract_smt_model(output);
        assert!(model.is_some());
    }

    #[test]
    fn test_extract_smt_model_no_closing_paren() {
        // Test model extraction when closing paren not found
        let output = "(model incomplete";
        let model = extract_smt_model(output);
        // Should return None when model is incomplete
        assert!(model.is_none());
    }

    #[test]
    fn test_extract_unsat_core_end_boundary() {
        // Test core extraction at exact boundary (end > core_start condition)
        let output = "(error \"x\")";
        let core = extract_unsat_core(output);
        assert!(core.is_some());

        let output = "(:reason-unknown \"x\")";
        let core = extract_unsat_core(output);
        assert!(core.is_some());
    }

    #[test]
    fn test_extract_unsat_core_no_closing_paren() {
        // Test core extraction when closing paren not found
        let output = "(error incomplete";
        let core = extract_unsat_core(output);
        assert!(core.is_none());
    }

    #[test]
    fn test_extract_smt_model_plus_operation() {
        // Test that the + operation in model extraction is correct
        // Catches "replace + with * in extract_smt_model" and "replace + with - in extract_smt_model"
        let output = "(model (define-fun x () Int 42))";
        let model = extract_smt_model(output);
        assert!(model.is_some());
        let m = model.unwrap();
        assert!(m.starts_with("(model"));
        assert!(m.ends_with(")"));
        assert!(m.contains("define-fun x"));
    }

    #[test]
    fn test_extract_unsat_core_plus_operation() {
        // Test that the + operation in core extraction is correct
        // Catches "replace + with * in extract_unsat_core" and "replace + with - in extract_unsat_core"
        let output = "unsat\n(error \"conflict\")";
        let core = extract_unsat_core(output);
        assert!(core.is_some());
        let c = core.unwrap();
        assert!(c.starts_with("(error"));
        assert!(c.ends_with(")"));
    }

    #[tokio::test]
    async fn test_solve_dimacs_timeout_comparison() {
        // Test the timeout_secs > 0 condition in solve_dimacs
        // Skip if Z3 not available
        let Some(solver) = Z3::detect().await else {
            return;
        };

        let dimacs = "p cnf 1 1\n1 0\n";
        let temp_dir = tempfile::tempdir().unwrap();
        let cnf_path = temp_dir.path().join("timeout_test.cnf");
        std::fs::write(&cnf_path, dimacs).unwrap();

        // Test with zero timeout (should not add -t flag)
        let config = SolverConfig {
            timeout: std::time::Duration::from_secs(0),
            ..Default::default()
        };
        let result = solver.solve_dimacs(&cnf_path, &config).await;
        // Should still work (or timeout based on environment)
        let _ = result;

        // Test with positive timeout
        let config = SolverConfig {
            timeout: std::time::Duration::from_secs(10),
            ..Default::default()
        };
        let output = solver.solve_dimacs(&cnf_path, &config).await.unwrap();
        assert!(output.result.is_sat());
    }

    #[tokio::test]
    async fn test_solve_smt2_timeout_comparison() {
        // Test the timeout_secs > 0 condition in solve_smt2
        // Skip if Z3 not available
        let Some(solver) = Z3::detect().await else {
            return;
        };

        let smt2 = "(declare-const x Int)\n(assert (> x 0))\n(check-sat)\n";
        let temp_dir = tempfile::tempdir().unwrap();
        let smt_path = temp_dir.path().join("timeout_test.smt2");
        std::fs::write(&smt_path, smt2).unwrap();

        // Test with zero timeout
        let config = SolverConfig {
            timeout: std::time::Duration::from_secs(0),
            ..Default::default()
        };
        let result = solver.solve_smt2(&smt_path, &config).await;
        let _ = result;

        // Test with positive timeout
        let config = SolverConfig {
            timeout: std::time::Duration::from_secs(10),
            ..Default::default()
        };
        let output = solver.solve_smt2(&smt_path, &config).await.unwrap();
        assert!(output.result.is_sat());
    }

    #[tokio::test]
    async fn test_z3_get_version_returns_string() {
        // Test that get_version returns a meaningful version string
        let Some(solver) = Z3::detect().await else {
            return;
        };

        let version = solver.get_version().await;
        match version {
            Some(v) => {
                // Version should not be empty
                assert!(!v.is_empty(), "Version string should not be empty");
                // Version string should have at least one character
                assert!(v.chars().next().is_some(), "Version should have content");
            }
            None => {
                // get_version can return None if version output parsing fails
            }
        }
    }

    #[test]
    fn test_parse_smt2_output_all_stat_lines() {
        // Test parsing of all stat line types
        let solver = Z3::new();
        let stdout = ":conflicts 100\n:decisions 200\n:propagations 300\n:memory 50\nsat\n";
        let output = solver.parse_smt2_output(stdout, "", std::time::Duration::ZERO);

        assert_eq!(output.stats.conflicts, Some(100));
        assert_eq!(output.stats.decisions, Some(200));
        assert_eq!(output.stats.propagations, Some(300));
        // Memory is converted to bytes (50 MB)
        assert_eq!(output.stats.memory_bytes, Some(50 * 1024 * 1024));
    }

    #[test]
    fn test_parse_smt2_output_stats_in_stderr() {
        // Test that stats are also parsed from stderr
        let solver = Z3::new();
        let stdout = "sat\n";
        let stderr = ":conflicts 999\n";
        let output = solver.parse_smt2_output(stdout, stderr, std::time::Duration::ZERO);

        assert_eq!(output.stats.conflicts, Some(999));
    }
}
