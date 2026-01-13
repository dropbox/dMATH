//! Alloy backend for bounded model checking
//!
//! This backend compiles USL specifications to Alloy models and executes them
//! via the `alloy exec` command. Alloy uses SAT solving to find instances
//! satisfying constraints or counterexamples violating assertions.
//!
//! Key semantics:
//! - For `check` commands: UNSAT = assertion holds, SAT = counterexample found
//! - Exit code 0 does NOT mean success - must parse SAT/UNSAT from output
//! - Exit code 1 indicates execution error (syntax, type, etc.)

// =============================================
// Kani Proofs for Alloy Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- AlloyConfig Default Tests ----

    /// Verify AlloyConfig::default timeout is 120 seconds
    #[kani::proof]
    fn proof_alloy_config_default_timeout() {
        let config = AlloyConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(120),
            "Default timeout should be 120 seconds",
        );
    }

    /// Verify AlloyConfig::default alloy_path is None
    #[kani::proof]
    fn proof_alloy_config_default_path_none() {
        let config = AlloyConfig::default();
        kani::assert(
            config.alloy_path.is_none(),
            "Default alloy_path should be None",
        );
    }

    /// Verify AlloyConfig::default scope is 5
    #[kani::proof]
    fn proof_alloy_config_default_scope() {
        let config = AlloyConfig::default();
        kani::assert(config.scope == 5, "Default scope should be 5");
    }

    /// Verify AlloyConfig::default solver is None
    #[kani::proof]
    fn proof_alloy_config_default_solver_none() {
        let config = AlloyConfig::default();
        kani::assert(config.solver.is_none(), "Default solver should be None");
    }

    // ---- AlloyBackend Construction Tests ----

    /// Verify AlloyBackend::new uses default config
    #[kani::proof]
    fn proof_alloy_backend_new_defaults() {
        let backend = AlloyBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(120),
            "New backend should use default timeout",
        );
        kani::assert(
            backend.config.scope == 5,
            "New backend should use default scope",
        );
    }

    /// Verify AlloyBackend::with_config preserves custom timeout
    #[kani::proof]
    fn proof_alloy_backend_with_config_timeout() {
        let config = AlloyConfig {
            alloy_path: None,
            timeout: Duration::from_secs(600),
            scope: 10,
            solver: None,
        };
        let backend = AlloyBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "Custom timeout should be preserved",
        );
    }

    /// Verify AlloyBackend::with_config preserves custom scope
    #[kani::proof]
    fn proof_alloy_backend_with_config_scope() {
        let config = AlloyConfig {
            alloy_path: None,
            timeout: Duration::from_secs(120),
            scope: 20,
            solver: Some("MiniSat".to_string()),
        };
        let backend = AlloyBackend::with_config(config);
        kani::assert(
            backend.config.scope == 20,
            "Custom scope should be preserved",
        );
    }

    /// Verify AlloyBackend::default equals AlloyBackend::new
    #[kani::proof]
    fn proof_alloy_backend_default_equals_new() {
        let default_backend = AlloyBackend::default();
        let new_backend = AlloyBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
        kani::assert(
            default_backend.config.scope == new_backend.config.scope,
            "Default and new should have same scope",
        );
    }

    // ---- Backend Trait Implementation Tests ----

    /// Verify id() returns Alloy
    #[kani::proof]
    fn proof_backend_id_is_alloy() {
        let backend = AlloyBackend::new();
        kani::assert(backend.id() == BackendId::Alloy, "ID should be Alloy");
    }

    /// Verify supports() includes Invariant
    #[kani::proof]
    fn proof_alloy_supports_invariant() {
        let backend = AlloyBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Invariant),
            "Should support Invariant",
        );
    }

    /// Verify supports() returns exactly one property type
    #[kani::proof]
    fn proof_alloy_supports_count() {
        let backend = AlloyBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 1,
            "Should support exactly one property type",
        );
    }

    // ---- AlloyDetection Enum Tests ----

    /// Verify AlloyDetection::Available variant stores path
    #[kani::proof]
    fn proof_alloy_detection_available() {
        let path = PathBuf::from("/usr/bin/alloy");
        let detection = AlloyDetection::Available {
            alloy_path: path.clone(),
        };
        match detection {
            AlloyDetection::Available { alloy_path } => {
                kani::assert(alloy_path == path, "Available should preserve alloy_path");
            }
            _ => kani::assert(false, "Should be Available variant"),
        }
    }

    /// Verify AlloyDetection::NotFound variant stores reason
    #[kani::proof]
    fn proof_alloy_detection_not_found() {
        let reason = "alloy not found".to_string();
        let detection = AlloyDetection::NotFound(reason.clone());
        match detection {
            AlloyDetection::NotFound(r) => {
                kani::assert(r == reason, "NotFound should preserve reason");
            }
            _ => kani::assert(false, "Should be NotFound variant"),
        }
    }

    // ---- Error Detection Tests ----

    /// Verify has_error detects ERROR pattern
    #[kani::proof]
    fn proof_has_error_detects_main_error() {
        let backend = AlloyBackend::new();
        kani::assert(
            backend.has_error("[main] ERROR alloy - something went wrong"),
            "Should detect [main] ERROR pattern",
        );
    }

    /// Verify has_error detects syntax error
    #[kani::proof]
    fn proof_has_error_detects_syntax_error() {
        let backend = AlloyBackend::new();
        kani::assert(
            backend.has_error("Syntax error in file.als"),
            "Should detect Syntax error pattern",
        );
    }

    /// Verify has_error detects type mismatch
    #[kani::proof]
    fn proof_has_error_detects_type_mismatch() {
        let backend = AlloyBackend::new();
        kani::assert(
            backend.has_error("Type mismatch: expected int"),
            "Should detect Type mismatch pattern",
        );
    }

    /// Verify has_error detects CompParser syntax error
    #[kani::proof]
    fn proof_has_error_detects_comp_parser_error() {
        let backend = AlloyBackend::new();
        kani::assert(
            backend.has_error("[CompParser.syntax_error] executing CLI:exec"),
            "Should detect CompParser.syntax_error pattern",
        );
    }

    /// Verify has_error returns false for valid output
    #[kani::proof]
    fn proof_has_error_false_for_valid() {
        let backend = AlloyBackend::new();
        kani::assert(
            !backend.has_error("00. check Test 0 UNSAT"),
            "Should not detect error in valid output",
        );
    }

    // ---- Check Results Parsing Tests ----

    /// Verify parse_check_results extracts UNSAT
    #[kani::proof]
    fn proof_parse_check_results_unsat() {
        let backend = AlloyBackend::new();
        let output = "00. check Test 0 UNSAT";
        let results = backend.parse_check_results(output);
        kani::assert(results.len() == 1, "Should parse one result");
        kani::assert(!results[0].2, "UNSAT should be parsed as false (not SAT)");
    }

    /// Verify parse_check_results extracts SAT
    #[kani::proof]
    fn proof_parse_check_results_sat() {
        let backend = AlloyBackend::new();
        let output = "00. check Test 0 1/1 SAT";
        let results = backend.parse_check_results(output);
        kani::assert(results.len() == 1, "Should parse one result");
        kani::assert(results[0].2, "SAT should be parsed as true");
    }

    /// Verify parse_check_results extracts name
    #[kani::proof]
    fn proof_parse_check_results_name() {
        let backend = AlloyBackend::new();
        let output = "00. check MyAssertion 0 UNSAT";
        let results = backend.parse_check_results(output);
        kani::assert(results.len() == 1, "Should parse one result");
        kani::assert(
            results[0].1 == "MyAssertion",
            "Should extract assertion name",
        );
    }

    /// Verify parse_check_results extracts index
    #[kani::proof]
    fn proof_parse_check_results_index() {
        let backend = AlloyBackend::new();
        let output = "05. check Test 0 UNSAT";
        let results = backend.parse_check_results(output);
        kani::assert(results.len() == 1, "Should parse one result");
        kani::assert(results[0].0 == 5, "Should extract index 5");
    }

    /// Verify parse_check_results handles empty input
    #[kani::proof]
    fn proof_parse_check_results_empty() {
        let backend = AlloyBackend::new();
        let results = backend.parse_check_results("");
        kani::assert(results.is_empty(), "Empty input should yield empty results");
    }

    /// Verify parse_check_results handles multiple results
    #[kani::proof]
    fn proof_parse_check_results_multiple() {
        let backend = AlloyBackend::new();
        let output = "00. check A 0 UNSAT\n01. check B 0 1/1 SAT";
        let results = backend.parse_check_results(output);
        kani::assert(results.len() == 2, "Should parse two results");
        kani::assert(!results[0].2, "First should be UNSAT");
        kani::assert(results[1].2, "Second should be SAT");
    }

    // ---- Run Results Parsing Tests ----

    /// Verify parse_run_results extracts SAT
    #[kani::proof]
    fn proof_parse_run_results_sat() {
        let backend = AlloyBackend::new();
        let output = "00. run run$1 0 1/1 SAT";
        let results = backend.parse_run_results(output);
        kani::assert(results.len() == 1, "Should parse one result");
        kani::assert(results[0].2, "SAT should be parsed as true");
    }

    /// Verify parse_run_results extracts UNSAT
    #[kani::proof]
    fn proof_parse_run_results_unsat() {
        let backend = AlloyBackend::new();
        let output = "00. run run$1 0 UNSAT";
        let results = backend.parse_run_results(output);
        kani::assert(results.len() == 1, "Should parse one result");
        kani::assert(!results[0].2, "UNSAT should be parsed as false");
    }

    /// Verify parse_run_results handles empty input
    #[kani::proof]
    fn proof_parse_run_results_empty() {
        let backend = AlloyBackend::new();
        let results = backend.parse_run_results("");
        kani::assert(results.is_empty(), "Empty input should yield empty results");
    }

    // ---- Value Parsing Tests ----

    /// Verify parse_alloy_value handles empty string
    #[kani::proof]
    fn proof_parse_alloy_value_empty_string() {
        let backend = AlloyBackend::new();
        let result = backend.parse_alloy_value("");
        match result {
            CounterexampleValue::Unknown(s) => {
                kani::assert(s.is_empty(), "Empty string should yield empty Unknown");
            }
            _ => kani::assert(false, "Empty string should yield Unknown"),
        }
    }

    /// Verify parse_alloy_value handles empty set
    #[kani::proof]
    fn proof_parse_alloy_value_empty_set() {
        let backend = AlloyBackend::new();
        let result = backend.parse_alloy_value("{}");
        match result {
            CounterexampleValue::Set(v) => {
                kani::assert(v.is_empty(), "Empty set should have no elements");
            }
            _ => kani::assert(false, "Should parse as Set"),
        }
    }

    /// Verify parse_alloy_value handles positive integer
    #[kani::proof]
    fn proof_parse_alloy_value_positive_int() {
        let backend = AlloyBackend::new();
        let result = backend.parse_alloy_value("42");
        match result {
            CounterexampleValue::Int { value, .. } => {
                kani::assert(value == 42, "Should parse 42");
            }
            _ => kani::assert(false, "Should parse as Int"),
        }
    }

    /// Verify parse_alloy_value handles negative integer
    #[kani::proof]
    fn proof_parse_alloy_value_negative_int() {
        let backend = AlloyBackend::new();
        let result = backend.parse_alloy_value("-7");
        match result {
            CounterexampleValue::Int { value, .. } => {
                kani::assert(value == -7, "Should parse -7");
            }
            _ => kani::assert(false, "Should parse as Int"),
        }
    }

    /// Verify parse_alloy_value handles zero
    #[kani::proof]
    fn proof_parse_alloy_value_zero() {
        let backend = AlloyBackend::new();
        let result = backend.parse_alloy_value("0");
        match result {
            CounterexampleValue::Int { value, .. } => {
                kani::assert(value == 0, "Should parse 0");
            }
            _ => kani::assert(false, "Should parse as Int"),
        }
    }

    /// Verify parse_alloy_atom handles Int[n] syntax
    #[kani::proof]
    fn proof_parse_alloy_atom_int_bracket() {
        let backend = AlloyBackend::new();
        let result = backend.parse_alloy_atom("Int[5]");
        match result {
            CounterexampleValue::Int { value, type_hint } => {
                kani::assert(value == 5, "Should parse value 5");
                kani::assert(
                    type_hint == Some("Int".to_string()),
                    "Type hint should be Int",
                );
            }
            _ => kani::assert(false, "Should parse as Int"),
        }
    }

    /// Verify parse_alloy_atom handles atom with $
    #[kani::proof]
    fn proof_parse_alloy_atom_with_dollar() {
        let backend = AlloyBackend::new();
        let result = backend.parse_alloy_atom("Node$0");
        match result {
            CounterexampleValue::String(s) => {
                kani::assert(s == "Node$0", "Should preserve atom name");
            }
            _ => kani::assert(false, "Should parse as String"),
        }
    }

    /// Verify parse_alloy_atom handles unknown format
    #[kani::proof]
    fn proof_parse_alloy_atom_unknown() {
        let backend = AlloyBackend::new();
        let result = backend.parse_alloy_atom("unknown_value");
        match result {
            CounterexampleValue::Unknown(s) => {
                kani::assert(s == "unknown_value", "Should preserve value");
            }
            _ => kani::assert(false, "Should parse as Unknown"),
        }
    }

    // ---- Diagnostics Extraction Tests ----

    /// Verify extract_diagnostics detects warnings
    #[kani::proof]
    fn proof_extract_diagnostics_warning() {
        let backend = AlloyBackend::new();
        let output = "Warning\n  0. line 26, column 13 == is redundant...";
        let diagnostics = backend.extract_diagnostics(output);
        kani::assert(!diagnostics.is_empty(), "Should extract warnings");
    }

    /// Verify extract_diagnostics handles empty input
    #[kani::proof]
    fn proof_extract_diagnostics_empty() {
        let backend = AlloyBackend::new();
        let diagnostics = backend.extract_diagnostics("");
        kani::assert(
            diagnostics.is_empty(),
            "Empty input should yield no diagnostics",
        );
    }

    // ---- extract_error_reason Tests ----

    /// Verify extract_error_reason finds syntax error with location
    #[kani::proof]
    fn proof_extract_error_reason_syntax_with_location() {
        let backend = AlloyBackend::new();
        let output = "Syntax error at file.als at line 5 column 10";
        let reason = backend.extract_error_reason(output);
        kani::assert(
            reason.contains("Syntax error"),
            "Should contain Syntax error",
        );
    }

    /// Verify extract_error_reason finds ERROR line
    #[kani::proof]
    fn proof_extract_error_reason_error_line() {
        let backend = AlloyBackend::new();
        let output = "[main] ERROR alloy - parsing failed";
        let reason = backend.extract_error_reason(output);
        kani::assert(reason.contains("ERROR"), "Should contain ERROR");
    }

    /// Verify extract_error_reason returns default for no error
    #[kani::proof]
    fn proof_extract_error_reason_default() {
        let backend = AlloyBackend::new();
        let output = "00. check Test 0 UNSAT";
        let reason = backend.extract_error_reason(output);
        kani::assert(
            reason == "Alloy execution failed",
            "Should return default message",
        );
    }
}

use crate::traits::*;
use async_trait::async_trait;
use dashprove_usl::{compile_to_alloy, typecheck::TypedSpec};
use regex::Regex;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::process::Command;
use tracing::{debug, info, warn};

/// Configuration for Alloy backend
#[derive(Debug, Clone)]
pub struct AlloyConfig {
    /// Path to `alloy` binary (if not in PATH)
    pub alloy_path: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// Scope for bounded model checking (default: 5)
    pub scope: usize,
    /// SAT solver to use (default: SAT4J)
    pub solver: Option<String>,
}

impl Default for AlloyConfig {
    fn default() -> Self {
        Self {
            alloy_path: None,
            timeout: Duration::from_secs(120),
            scope: 5,
            solver: None,
        }
    }
}

/// Alloy verification backend using `alloy exec`
pub struct AlloyBackend {
    config: AlloyConfig,
}

#[derive(Debug, Clone)]
enum AlloyDetection {
    Available { alloy_path: PathBuf },
    NotFound(String),
}

/// Captured output from alloy exec
struct AlloyOutput {
    stdout: String,
    stderr: String,
    exit_code: Option<i32>,
    duration: Duration,
}

impl AlloyBackend {
    /// Create a new Alloy backend with default configuration
    pub fn new() -> Self {
        Self {
            config: AlloyConfig::default(),
        }
    }

    /// Create a new Alloy backend with custom configuration
    pub fn with_config(config: AlloyConfig) -> Self {
        Self { config }
    }

    /// Detect whether alloy CLI is available
    async fn detect_alloy(&self) -> AlloyDetection {
        let alloy_path = if let Some(ref path) = self.config.alloy_path {
            if path.exists() {
                path.clone()
            } else {
                return AlloyDetection::NotFound(format!(
                    "Configured alloy path does not exist: {:?}",
                    path
                ));
            }
        } else {
            match which::which("alloy") {
                Ok(path) => path,
                Err(_) => {
                    return AlloyDetection::NotFound(
                        "alloy not found. Install via `brew install alloy-analyzer`.".to_string(),
                    )
                }
            }
        };

        // Verify alloy command runs - use `alloy --help` (not `alloy exec --help`)
        // because the Alloy CLI doesn't support --help on subcommands
        let mut cmd = Command::new(&alloy_path);
        cmd.arg("--help")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let result = tokio::time::timeout(Duration::from_secs(10), cmd.output()).await;
        match result {
            Ok(Ok(output)) => {
                // Alloy --help exits with code 1 but outputs usage info to stderr
                // The output contains "_alloy" (the internal command name)
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                let combined = format!("{}{}", stdout, stderr);

                // Check for indicators that this is the Alloy CLI
                if combined.contains("_alloy")
                    || combined.contains("Alloy")
                    || combined.contains("exec")
                    || combined.contains("Execute an Alloy")
                {
                    debug!("Detected alloy at {:?}", alloy_path);
                    AlloyDetection::Available { alloy_path }
                } else {
                    AlloyDetection::NotFound(format!(
                        "alloy --help produced unexpected output: {}",
                        combined.lines().take(5).collect::<Vec<_>>().join("\n")
                    ))
                }
            }
            Ok(Err(e)) => AlloyDetection::NotFound(format!("Failed to execute alloy: {}", e)),
            Err(_) => AlloyDetection::NotFound("alloy --help timed out".to_string()),
        }
    }

    /// Write the Alloy spec to a temporary file
    async fn write_temp_spec(&self, spec: &TypedSpec) -> Result<(TempDir, PathBuf), BackendError> {
        let compiled = compile_to_alloy(spec);

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::CompilationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let spec_path = temp_dir.path().join("spec.als");
        tokio::fs::write(&spec_path, &compiled.code)
            .await
            .map_err(|e| {
                BackendError::CompilationFailed(format!("Failed to write Alloy spec: {}", e))
            })?;

        debug!("Wrote Alloy spec to {:?}", spec_path);
        Ok((temp_dir, spec_path))
    }

    /// Run alloy exec on the given specification
    async fn run_alloy(
        &self,
        detection: &AlloyDetection,
        spec_path: &PathBuf,
    ) -> Result<AlloyOutput, BackendError> {
        let alloy_path = match detection {
            AlloyDetection::Available { alloy_path } => alloy_path,
            AlloyDetection::NotFound(reason) => {
                return Err(BackendError::Unavailable(reason.clone()))
            }
        };

        let mut cmd = Command::new(alloy_path);
        cmd.arg("exec").arg(spec_path);

        // Add solver if specified
        if let Some(ref solver) = self.config.solver {
            cmd.arg("-s").arg(solver);
        }

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let start = Instant::now();
        let output = tokio::time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to execute alloy: {}", e))
            })?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let duration = start.elapsed();

        debug!("alloy exec stdout:\n{}", stdout);
        if !stderr.is_empty() {
            debug!("alloy exec stderr:\n{}", stderr);
        }

        Ok(AlloyOutput {
            stdout,
            stderr,
            exit_code: output.status.code(),
            duration,
        })
    }

    /// Parse alloy exec output into a structured backend result
    fn parse_output(&self, output: &AlloyOutput) -> BackendResult {
        let combined = format!("{}\n{}", output.stdout, output.stderr);
        let diagnostics = self.extract_diagnostics(&combined);

        // Check for errors first (exit code 1 or error patterns)
        if output.exit_code == Some(1) || self.has_error(&combined) {
            let reason = self.extract_error_reason(&combined);
            return BackendResult {
                backend: BackendId::Alloy,
                status: VerificationStatus::Unknown { reason },
                proof: None,
                counterexample: None,
                diagnostics,
                time_taken: output.duration,
            };
        }

        // Parse check command results
        // Format: "00. check AssertionName 0 UNSAT" or "00. check AssertionName 0 1/1 SAT"
        let check_results = self.parse_check_results(&combined);

        if !check_results.is_empty() {
            // If any check has SAT result, assertion is violated (counterexample found)
            let any_sat = check_results.iter().any(|(_, _, sat)| *sat);
            let all_unsat = check_results.iter().all(|(_, _, sat)| !*sat);

            if any_sat {
                let counterexample = self.extract_counterexample(&combined, &check_results);
                return BackendResult {
                    backend: BackendId::Alloy,
                    status: VerificationStatus::Disproven,
                    proof: None,
                    counterexample,
                    diagnostics,
                    time_taken: output.duration,
                };
            }

            if all_unsat {
                let scope_note = format!("All assertions hold within scope {}", self.config.scope);
                return BackendResult {
                    backend: BackendId::Alloy,
                    status: VerificationStatus::Proven,
                    proof: Some(scope_note),
                    counterexample: None,
                    diagnostics,
                    time_taken: output.duration,
                };
            }
        }

        // No check results found - might only have run commands
        let run_results = self.parse_run_results(&combined);
        if !run_results.is_empty() {
            let any_sat = run_results.iter().any(|(_, _, sat)| *sat);
            let proof = if any_sat {
                Some("Model is satisfiable (instances found)".to_string())
            } else {
                Some("No instances found within scope".to_string())
            };

            return BackendResult {
                backend: BackendId::Alloy,
                status: VerificationStatus::Proven,
                proof,
                counterexample: None,
                diagnostics,
                time_taken: output.duration,
            };
        }

        // Could not determine result
        BackendResult {
            backend: BackendId::Alloy,
            status: VerificationStatus::Unknown {
                reason: "Could not parse Alloy output".to_string(),
            },
            proof: None,
            counterexample: None,
            diagnostics,
            time_taken: output.duration,
        }
    }

    /// Check if output contains error patterns
    fn has_error(&self, output: &str) -> bool {
        output.contains("[main] ERROR")
            || output.contains("Syntax error")
            || output.contains("Type mismatch")
            || output.contains("[CompParser.syntax_error]")
    }

    /// Extract error reason from output
    fn extract_error_reason(&self, output: &str) -> String {
        // Look for syntax error with location
        if let Some(caps) = Regex::new(r"Syntax error.*at line (\d+) column (\d+)")
            .ok()
            .and_then(|re| re.captures(output))
        {
            let line = caps.get(1).map_or("?", |m| m.as_str());
            let col = caps.get(2).map_or("?", |m| m.as_str());
            return format!("Syntax error at line {} column {}", line, col);
        }

        // Look for any ERROR line
        for line in output.lines() {
            if line.contains("ERROR") {
                return line.trim().to_string();
            }
        }

        "Alloy execution failed".to_string()
    }

    /// Parse check command results from output
    /// Returns Vec of (index, name, is_sat)
    fn parse_check_results(&self, output: &str) -> Vec<(usize, String, bool)> {
        let mut results = Vec::new();

        // Pattern: "00. check AssertionName 0 UNSAT" or "00. check Name 0 1/1 SAT"
        let re = Regex::new(r"(\d+)\.\s+check\s+(\S+)\s+\d+\s+(?:\d+/\d+\s+)?(SAT|UNSAT)")
            .expect("Invalid regex");

        for caps in re.captures_iter(output) {
            let index: usize = caps.get(1).unwrap().as_str().parse().unwrap_or(0);
            let name = caps.get(2).unwrap().as_str().to_string();
            let is_sat = caps.get(3).unwrap().as_str() == "SAT";
            results.push((index, name, is_sat));
        }

        results
    }

    /// Parse run command results from output
    fn parse_run_results(&self, output: &str) -> Vec<(usize, String, bool)> {
        let mut results = Vec::new();

        let re = Regex::new(r"(\d+)\.\s+run\s+(\S+)\s+\d+\s+(?:\d+/\d+\s+)?(SAT|UNSAT)")
            .expect("Invalid regex");

        for caps in re.captures_iter(output) {
            let index: usize = caps.get(1).unwrap().as_str().parse().unwrap_or(0);
            let name = caps.get(2).unwrap().as_str().to_string();
            let is_sat = caps.get(3).unwrap().as_str() == "SAT";
            results.push((index, name, is_sat));
        }

        results
    }

    /// Extract counterexample information from output
    fn extract_counterexample(
        &self,
        output: &str,
        check_results: &[(usize, String, bool)],
    ) -> Option<StructuredCounterexample> {
        // Find failed assertions (SAT = counterexample found)
        let failed: Vec<_> = check_results
            .iter()
            .filter(|(_, _, sat)| *sat)
            .map(|(_, name, _)| name.clone())
            .collect();

        if failed.is_empty() {
            return None;
        }

        let mut ce = StructuredCounterexample::new();

        // Add failed checks
        for name in &failed {
            ce.failed_checks.push(FailedCheck {
                check_id: name.clone(),
                description: format!("Assertion {} violated (counterexample found)", name),
                location: None,
                function: None,
            });
        }

        // Look for skolem values in output (counterexample witnesses)
        // Format: "skolem $AssertName_var = {Atom$0}"
        let mut raw_parts = vec![format!("Violated assertions: {}", failed.join(", "))];
        for line in output.lines() {
            if line.contains("skolem") && line.contains("$") {
                raw_parts.push(line.trim().to_string());
                // Try to extract witness value
                if let Some(eq_pos) = line.find('=') {
                    let var_part = line[..eq_pos].trim();
                    let val_part = line[eq_pos + 1..].trim();
                    // Extract variable name from format: $AssertionName_varname
                    // The variable name is after the last underscore in the skolem identifier
                    if let Some(dollar_pos) = var_part.find('$') {
                        let skolem_id = &var_part[dollar_pos + 1..];
                        // Variable name is after the last underscore
                        let var_name = if let Some(underscore_pos) = skolem_id.rfind('_') {
                            skolem_id[underscore_pos + 1..].trim().to_string()
                        } else {
                            skolem_id.trim().to_string()
                        };
                        let parsed_value = self.parse_alloy_value(val_part);
                        ce.witness.insert(var_name, parsed_value);
                    }
                }
            }
        }
        ce.raw = Some(raw_parts.join("\n"));

        Some(ce)
    }

    /// Parse an Alloy value string into CounterexampleValue
    ///
    /// Alloy values can be:
    /// - Empty set: `{}`
    /// - Singleton set: `{Node$0}`
    /// - Multi-element set: `{Node$0, Node$1}`
    /// - Nested relation: `{Node$0->Node$1, Node$0->Node$2}`
    /// - Bare atom: `Node$0`
    /// - Integer: `1`, `-5`
    fn parse_alloy_value(&self, value: &str) -> CounterexampleValue {
        let trimmed = value.trim();

        // Empty string
        if trimmed.is_empty() {
            return CounterexampleValue::Unknown(String::new());
        }

        // Set syntax: {...}
        if trimmed.starts_with('{') && trimmed.ends_with('}') {
            let inner = &trimmed[1..trimmed.len() - 1].trim();

            // Empty set
            if inner.is_empty() {
                return CounterexampleValue::Set(Vec::new());
            }

            // Check if this is a relation (contains ->)
            if inner.contains("->") {
                return self.parse_alloy_relation(inner);
            }

            // Regular set: split by comma and parse each element
            let elements: Vec<CounterexampleValue> = inner
                .split(',')
                .map(|e| self.parse_alloy_atom(e.trim()))
                .collect();

            return CounterexampleValue::Set(elements);
        }

        // Integer
        if let Ok(n) = trimmed.parse::<i128>() {
            return CounterexampleValue::Int {
                value: n,
                type_hint: None,
            };
        }

        // Bare atom (e.g., Node$0)
        self.parse_alloy_atom(trimmed)
    }

    /// Parse an Alloy atom (e.g., "Node$0", "Int\\[5\\]")
    fn parse_alloy_atom(&self, value: &str) -> CounterexampleValue {
        let trimmed = value.trim();

        // Handle Int[n] syntax
        if trimmed.starts_with("Int[") && trimmed.ends_with(']') {
            let num_str = &trimmed[4..trimmed.len() - 1];
            if let Ok(n) = num_str.parse::<i128>() {
                return CounterexampleValue::Int {
                    value: n,
                    type_hint: Some("Int".to_string()),
                };
            }
        }

        // Regular atom like Node$0 - store as string
        // Format is typically SigName$Index
        if trimmed.contains('$') {
            return CounterexampleValue::String(trimmed.to_string());
        }

        // Unknown format
        CounterexampleValue::Unknown(trimmed.to_string())
    }

    /// Parse an Alloy relation (set of tuples like "Node$0->Node$1, Node$0->Node$2")
    fn parse_alloy_relation(&self, inner: &str) -> CounterexampleValue {
        let mut mappings = Vec::new();

        // Split by comma, respecting nested structures
        for tuple_str in inner.split(',') {
            let tuple_str = tuple_str.trim();

            // Split on -> to get tuple elements
            let parts: Vec<&str> = tuple_str.split("->").collect();

            if parts.len() == 2 {
                // Binary relation: a -> b becomes Function entry
                let key = self.parse_alloy_atom(parts[0]);
                let val = self.parse_alloy_atom(parts[1]);
                mappings.push((key, val));
            } else if parts.len() > 2 {
                // N-ary relation: a -> b -> c stored as Sequence
                let elements: Vec<CounterexampleValue> =
                    parts.iter().map(|p| self.parse_alloy_atom(p)).collect();
                // Use first element as key, sequence of rest as value
                let key = elements[0].clone();
                let val = CounterexampleValue::Sequence(elements[1..].to_vec());
                mappings.push((key, val));
            }
        }

        CounterexampleValue::Function(mappings)
    }

    /// Extract diagnostic messages from output
    fn extract_diagnostics(&self, output: &str) -> Vec<String> {
        let mut diagnostics = Vec::new();

        // Collect warnings
        for line in output.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("Warning")
                || trimmed.contains("is redundant")
                || (trimmed.starts_with("WARNING:") && !trimmed.contains("java.lang.System"))
            {
                diagnostics.push(trimmed.to_string());
            }
        }

        // Add summary of check/run results
        let checks = self.parse_check_results(output);
        let runs = self.parse_run_results(output);

        if !checks.is_empty() {
            let passed = checks.iter().filter(|(_, _, sat)| !*sat).count();
            let failed = checks.iter().filter(|(_, _, sat)| *sat).count();
            diagnostics.push(format!("Checks: {} passed, {} failed", passed, failed));
        }

        if !runs.is_empty() {
            let sat = runs.iter().filter(|(_, _, s)| *s).count();
            let unsat = runs.iter().filter(|(_, _, s)| !*s).count();
            diagnostics.push(format!("Runs: {} SAT, {} UNSAT", sat, unsat));
        }

        diagnostics
    }
}

impl Default for AlloyBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for AlloyBackend {
    fn id(&self) -> BackendId {
        BackendId::Alloy
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::Invariant]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let detection = self.detect_alloy().await;
        if let AlloyDetection::NotFound(reason) = &detection {
            return Err(BackendError::Unavailable(reason.clone()));
        }

        let (_temp_dir, spec_path) = self.write_temp_spec(spec).await?;
        let output = self.run_alloy(&detection, &spec_path).await?;
        Ok(self.parse_output(&output))
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect_alloy().await {
            AlloyDetection::Available { alloy_path } => {
                info!("alloy available at {:?}", alloy_path);
                HealthStatus::Healthy
            }
            AlloyDetection::NotFound(reason) => {
                warn!("Alloy backend unavailable: {}", reason);
                HealthStatus::Unavailable { reason }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_output(stdout: &str, exit_code: i32) -> AlloyOutput {
        AlloyOutput {
            stdout: stdout.to_string(),
            stderr: String::new(),
            exit_code: Some(exit_code),
            duration: Duration::from_secs(1),
        }
    }

    // Include captured real outputs
    const ALLOY_PASS_OUTPUT: &str = include_str!("../../../examples/alloy/OUTPUT_pass.txt");
    const ALLOY_FAIL_OUTPUT: &str = include_str!("../../../examples/alloy/OUTPUT_fail.txt");
    const ALLOY_ERROR_OUTPUT: &str = include_str!("../../../examples/alloy/OUTPUT_error.txt");

    #[test]
    fn parse_pass_output_marks_proven() {
        let backend = AlloyBackend::new();
        let output = make_output(ALLOY_PASS_OUTPUT, 0);
        let result = backend.parse_output(&output);

        assert!(matches!(result.status, VerificationStatus::Proven));
        assert!(result.counterexample.is_none());
        assert!(result.diagnostics.iter().any(|d| d.contains("1 passed")));
    }

    #[test]
    fn parse_fail_output_marks_disproven() {
        let backend = AlloyBackend::new();
        let output = make_output(ALLOY_FAIL_OUTPUT, 0);
        let result = backend.parse_output(&output);

        assert!(matches!(result.status, VerificationStatus::Disproven));
        assert!(result.counterexample.is_some());
        let ce = result.counterexample.as_ref().unwrap();
        // Check that the failed assertion name is captured
        assert!(
            ce.failed_checks
                .iter()
                .any(|c| c.check_id.contains("NoCycles"))
                || ce
                    .raw
                    .as_ref()
                    .map(|r| r.contains("NoCycles"))
                    .unwrap_or(false)
        );
    }

    #[test]
    fn parse_error_output_marks_unknown() {
        let backend = AlloyBackend::new();
        let output = make_output(ALLOY_ERROR_OUTPUT, 1);
        let result = backend.parse_output(&output);

        assert!(matches!(result.status, VerificationStatus::Unknown { .. }));
        if let VerificationStatus::Unknown { reason } = result.status {
            assert!(reason.contains("Syntax error") || reason.contains("ERROR"));
        }
    }

    #[test]
    fn parse_check_results_extracts_sat_unsat() {
        let backend = AlloyBackend::new();

        let output =
            "00. check InitialNotFinal          0       UNSAT\n01. check NoCycles 0 1/1 SAT";
        let results = backend.parse_check_results(output);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0], (0, "InitialNotFinal".to_string(), false)); // UNSAT = not SAT
        assert_eq!(results[1], (1, "NoCycles".to_string(), true)); // SAT = violated
    }

    #[test]
    fn parse_run_results_extracts_sat_unsat() {
        let backend = AlloyBackend::new();

        let output = "01. run   run$2                    0    1/1     SAT";
        let results = backend.parse_run_results(output);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0], (1, "run$2".to_string(), true));
    }

    #[test]
    fn has_error_detects_error_patterns() {
        let backend = AlloyBackend::new();

        assert!(backend.has_error("[main] ERROR alloy - something went wrong"));
        assert!(backend.has_error("Syntax error in file.als at line 5"));
        assert!(backend.has_error("[CompParser.syntax_error] executing CLI:exec"));
        assert!(!backend.has_error("00. check Test 0 UNSAT"));
    }

    #[test]
    fn extract_diagnostics_collects_warnings() {
        let backend = AlloyBackend::new();

        let output = r#"
00. check Test 0 UNSAT
Warning
  0. line 26, column 13 == is redundant...
"#;
        let diagnostics = backend.extract_diagnostics(output);

        assert!(diagnostics.iter().any(|d| d.contains("Warning")));
        assert!(diagnostics.iter().any(|d| d.contains("redundant")));
    }

    #[test]
    fn counterexample_extraction_includes_assertion_names() {
        let backend = AlloyBackend::new();

        let check_results = vec![
            (0, "PassingAssert".to_string(), false),
            (1, "FailingAssert".to_string(), true),
        ];

        let output = "00. check PassingAssert 0 UNSAT\n01. check FailingAssert 0 1/1 SAT";
        let counterexample = backend.extract_counterexample(output, &check_results);

        assert!(counterexample.is_some());
        let ce = counterexample.unwrap();
        // Check that FailingAssert is captured in failed_checks or raw
        assert!(
            ce.failed_checks
                .iter()
                .any(|c| c.check_id == "FailingAssert")
                || ce
                    .raw
                    .as_ref()
                    .map(|r| r.contains("FailingAssert"))
                    .unwrap_or(false)
        );
    }

    // Tests for Alloy value parsing

    #[test]
    fn parse_alloy_value_empty_set() {
        let backend = AlloyBackend::new();
        let result = backend.parse_alloy_value("{}");
        assert!(matches!(result, CounterexampleValue::Set(v) if v.is_empty()));
    }

    #[test]
    fn parse_alloy_value_singleton_set() {
        let backend = AlloyBackend::new();
        let result = backend.parse_alloy_value("{Node$0}");
        match result {
            CounterexampleValue::Set(v) => {
                assert_eq!(v.len(), 1);
                match &v[0] {
                    CounterexampleValue::String(s) => assert_eq!(s, "Node$0"),
                    _ => panic!("Expected String, got {:?}", v[0]),
                }
            }
            _ => panic!("Expected Set, got {:?}", result),
        }
    }

    #[test]
    fn parse_alloy_value_multi_element_set() {
        let backend = AlloyBackend::new();
        let result = backend.parse_alloy_value("{Node$0, Node$1, Node$2}");
        match result {
            CounterexampleValue::Set(v) => {
                assert_eq!(v.len(), 3);
            }
            _ => panic!("Expected Set, got {:?}", result),
        }
    }

    #[test]
    fn parse_alloy_value_integer() {
        let backend = AlloyBackend::new();

        let result = backend.parse_alloy_value("42");
        match result {
            CounterexampleValue::Int { value, .. } => assert_eq!(value, 42),
            _ => panic!("Expected Int, got {:?}", result),
        }

        let result = backend.parse_alloy_value("-5");
        match result {
            CounterexampleValue::Int { value, .. } => assert_eq!(value, -5),
            _ => panic!("Expected Int, got {:?}", result),
        }
    }

    #[test]
    fn parse_alloy_value_int_bracket_syntax() {
        let backend = AlloyBackend::new();
        let result = backend.parse_alloy_atom("Int[7]");
        match result {
            CounterexampleValue::Int { value, type_hint } => {
                assert_eq!(value, 7);
                assert_eq!(type_hint, Some("Int".to_string()));
            }
            _ => panic!("Expected Int, got {:?}", result),
        }
    }

    #[test]
    fn parse_alloy_value_bare_atom() {
        let backend = AlloyBackend::new();
        let result = backend.parse_alloy_value("Person$3");
        match result {
            CounterexampleValue::String(s) => assert_eq!(s, "Person$3"),
            _ => panic!("Expected String, got {:?}", result),
        }
    }

    #[test]
    fn parse_alloy_value_binary_relation() {
        let backend = AlloyBackend::new();
        let result = backend.parse_alloy_value("{Node$0->Node$1, Node$0->Node$2}");
        match result {
            CounterexampleValue::Function(mappings) => {
                assert_eq!(mappings.len(), 2);
                // First mapping: Node$0 -> Node$1
                match (&mappings[0].0, &mappings[0].1) {
                    (CounterexampleValue::String(k), CounterexampleValue::String(v)) => {
                        assert_eq!(k, "Node$0");
                        assert_eq!(v, "Node$1");
                    }
                    _ => panic!("Expected String keys/values"),
                }
            }
            _ => panic!("Expected Function, got {:?}", result),
        }
    }

    #[test]
    fn parse_alloy_value_ternary_relation() {
        let backend = AlloyBackend::new();
        let result = backend.parse_alloy_value("{A$0->B$0->C$0}");
        match result {
            CounterexampleValue::Function(mappings) => {
                assert_eq!(mappings.len(), 1);
                // Should have key A$0, value Sequence([B$0, C$0])
                match &mappings[0].1 {
                    CounterexampleValue::Sequence(seq) => {
                        assert_eq!(seq.len(), 2);
                    }
                    _ => panic!("Expected Sequence for n-ary relation value"),
                }
            }
            _ => panic!("Expected Function, got {:?}", result),
        }
    }

    #[test]
    fn parse_alloy_value_unknown_format() {
        let backend = AlloyBackend::new();
        let result = backend.parse_alloy_value("some_unknown_value");
        match result {
            CounterexampleValue::Unknown(s) => assert_eq!(s, "some_unknown_value"),
            _ => panic!("Expected Unknown, got {:?}", result),
        }
    }

    #[test]
    fn counterexample_with_skolem_parses_value() {
        let backend = AlloyBackend::new();

        let check_results = vec![(0, "NoCycles".to_string(), true)];
        let output = r#"
00. check NoCycles 0 1/1 SAT
skolem $NoCycles_n = {Node$0}
"#;
        let counterexample = backend.extract_counterexample(output, &check_results);

        assert!(counterexample.is_some());
        let ce = counterexample.unwrap();

        // Check that the witness was parsed as a Set, not Unknown
        if let Some(value) = ce.witness.get("n") {
            match value {
                CounterexampleValue::Set(v) => {
                    assert_eq!(v.len(), 1);
                }
                _ => panic!("Expected Set for skolem value, got {:?}", value),
            }
        } else {
            panic!("Expected 'n' in witness map");
        }
    }
}
