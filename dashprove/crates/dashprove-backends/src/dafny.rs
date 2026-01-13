//! Dafny backend implementation
//!
//! This backend executes Dafny specifications using the dafny command.
//! Dafny is a verification-aware programming language with built-in
//! support for specifications and proofs.

// =============================================
// Kani Proofs for Dafny Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- DafnyConfig Default Tests ----

    /// Verify DafnyConfig::default timeout is 300 seconds
    #[kani::proof]
    fn proof_dafny_config_default_timeout() {
        let config = DafnyConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "Default timeout should be 300 seconds",
        );
    }

    /// Verify DafnyConfig::default dafny_path is None
    #[kani::proof]
    fn proof_dafny_config_default_path_none() {
        let config = DafnyConfig::default();
        kani::assert(
            config.dafny_path.is_none(),
            "Default dafny_path should be None",
        );
    }

    /// Verify DafnyConfig::default target is None
    #[kani::proof]
    fn proof_dafny_config_default_target_none() {
        let config = DafnyConfig::default();
        kani::assert(config.target.is_none(), "Default target should be None");
    }

    // ---- DafnyBackend Construction Tests ----

    /// Verify DafnyBackend::new uses default config
    #[kani::proof]
    fn proof_dafny_backend_new_defaults() {
        let backend = DafnyBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "New backend should use default timeout",
        );
    }

    /// Verify DafnyBackend::with_config preserves custom timeout
    #[kani::proof]
    fn proof_dafny_backend_with_config_timeout() {
        let config = DafnyConfig {
            dafny_path: None,
            timeout: Duration::from_secs(600),
            target: None,
        };
        let backend = DafnyBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "Custom timeout should be preserved",
        );
    }

    /// Verify DafnyBackend::with_config preserves custom target
    #[kani::proof]
    fn proof_dafny_backend_with_config_target() {
        let config = DafnyConfig {
            dafny_path: None,
            timeout: Duration::from_secs(300),
            target: Some("cs".to_string()),
        };
        let backend = DafnyBackend::with_config(config);
        kani::assert(
            backend.config.target == Some("cs".to_string()),
            "Custom target should be preserved",
        );
    }

    /// Verify DafnyBackend::default equals DafnyBackend::new
    #[kani::proof]
    fn proof_dafny_backend_default_equals_new() {
        let default_backend = DafnyBackend::default();
        let new_backend = DafnyBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
    }

    // ---- Backend Trait Implementation Tests ----

    /// Verify id() returns Dafny
    #[kani::proof]
    fn proof_backend_id_is_dafny() {
        let backend = DafnyBackend::new();
        kani::assert(backend.id() == BackendId::Dafny, "ID should be Dafny");
    }

    /// Verify supports() includes Contract
    #[kani::proof]
    fn proof_dafny_supports_contract() {
        let backend = DafnyBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Contract),
            "Should support Contract",
        );
    }

    /// Verify supports() includes Invariant
    #[kani::proof]
    fn proof_dafny_supports_invariant() {
        let backend = DafnyBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Invariant),
            "Should support Invariant",
        );
    }

    /// Verify supports() includes Theorem
    #[kani::proof]
    fn proof_dafny_supports_theorem() {
        let backend = DafnyBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Theorem),
            "Should support Theorem",
        );
    }

    /// Verify supports() returns exactly three property types
    #[kani::proof]
    fn proof_dafny_supports_count() {
        let backend = DafnyBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 3,
            "Should support exactly three property types",
        );
    }

    // ---- DafnyDetection Enum Tests ----

    /// Verify DafnyDetection::Available variant stores path
    #[kani::proof]
    fn proof_dafny_detection_available() {
        let path = PathBuf::from("/usr/bin/dafny");
        let detection = DafnyDetection::Available {
            dafny_path: path.clone(),
        };
        match detection {
            DafnyDetection::Available { dafny_path } => {
                kani::assert(dafny_path == path, "Available should preserve dafny_path");
            }
            _ => kani::assert(false, "Should be Available variant"),
        }
    }

    /// Verify DafnyDetection::NotFound variant stores reason
    #[kani::proof]
    fn proof_dafny_detection_not_found() {
        let reason = "dafny not found".to_string();
        let detection = DafnyDetection::NotFound(reason.clone());
        match detection {
            DafnyDetection::NotFound(r) => {
                kani::assert(r == reason, "NotFound should preserve reason");
            }
            _ => kani::assert(false, "Should be NotFound variant"),
        }
    }

    // ---- extract_failed_checks Tests ----

    /// Verify extract_failed_checks identifies assertion errors
    #[kani::proof]
    fn proof_extract_failed_checks_assertion() {
        let output = "Error: assertion might not hold";
        let checks = DafnyBackend::extract_failed_checks(output);
        kani::assert(checks.len() == 1, "Should find one check");
        kani::assert(
            checks[0].check_id == "dafny_assertion",
            "Should identify as assertion",
        );
    }

    /// Verify extract_failed_checks identifies precondition errors
    #[kani::proof]
    fn proof_extract_failed_checks_precondition() {
        let output = "Error: precondition might not hold";
        let checks = DafnyBackend::extract_failed_checks(output);
        kani::assert(checks.len() == 1, "Should find one check");
        kani::assert(
            checks[0].check_id == "dafny_precondition",
            "Should identify as precondition",
        );
    }

    /// Verify extract_failed_checks identifies postcondition errors
    #[kani::proof]
    fn proof_extract_failed_checks_postcondition() {
        let output = "Error: postcondition might not hold";
        let checks = DafnyBackend::extract_failed_checks(output);
        kani::assert(checks.len() == 1, "Should find one check");
        kani::assert(
            checks[0].check_id == "dafny_postcondition",
            "Should identify as postcondition",
        );
    }

    /// Verify extract_failed_checks identifies ensures errors
    #[kani::proof]
    fn proof_extract_failed_checks_ensures() {
        let output = "Error: ensures clause might not hold";
        let checks = DafnyBackend::extract_failed_checks(output);
        kani::assert(checks.len() == 1, "Should find one check");
        kani::assert(
            checks[0].check_id == "dafny_postcondition",
            "Ensures should map to postcondition",
        );
    }

    /// Verify extract_failed_checks identifies requires errors
    #[kani::proof]
    fn proof_extract_failed_checks_requires() {
        let output = "Error: requires clause might not hold";
        let checks = DafnyBackend::extract_failed_checks(output);
        kani::assert(checks.len() == 1, "Should find one check");
        kani::assert(
            checks[0].check_id == "dafny_precondition",
            "Requires should map to precondition",
        );
    }

    /// Verify extract_failed_checks identifies invariant errors
    #[kani::proof]
    fn proof_extract_failed_checks_invariant() {
        let output = "Error: loop invariant might not hold";
        let checks = DafnyBackend::extract_failed_checks(output);
        kani::assert(checks.len() == 1, "Should find one check");
        kani::assert(
            checks[0].check_id == "dafny_invariant",
            "Should identify as invariant",
        );
    }

    /// Verify extract_failed_checks identifies termination errors
    #[kani::proof]
    fn proof_extract_failed_checks_termination() {
        let output = "Error: decreases expression might not decrease";
        let checks = DafnyBackend::extract_failed_checks(output);
        kani::assert(checks.len() == 1, "Should find one check");
        kani::assert(
            checks[0].check_id == "dafny_termination",
            "Should identify as termination",
        );
    }

    /// Verify extract_failed_checks identifies frame errors
    #[kani::proof]
    fn proof_extract_failed_checks_frame() {
        let output = "Error: modifies clause might not hold";
        let checks = DafnyBackend::extract_failed_checks(output);
        kani::assert(checks.len() == 1, "Should find one check");
        kani::assert(
            checks[0].check_id == "dafny_frame",
            "Should identify as frame",
        );
    }

    /// Verify extract_failed_checks identifies syntax errors
    #[kani::proof]
    fn proof_extract_failed_checks_syntax() {
        let output = "Error: Parsing error: expected identifier";
        let checks = DafnyBackend::extract_failed_checks(output);
        kani::assert(checks.len() == 1, "Should find one check");
        kani::assert(
            checks[0].check_id == "dafny_syntax_error",
            "Should identify as syntax error",
        );
    }

    /// Verify extract_failed_checks identifies resolution errors
    #[kani::proof]
    fn proof_extract_failed_checks_resolution() {
        let output = "Error: undeclared identifier: foo";
        let checks = DafnyBackend::extract_failed_checks(output);
        kani::assert(checks.len() == 1, "Should find one check");
        kani::assert(
            checks[0].check_id == "dafny_resolution_error",
            "Should identify as resolution error",
        );
    }

    /// Verify extract_failed_checks handles empty input
    #[kani::proof]
    fn proof_extract_failed_checks_empty() {
        let checks = DafnyBackend::extract_failed_checks("");
        kani::assert(checks.is_empty(), "Empty input should yield no checks");
    }

    /// Verify extract_failed_checks handles success output
    #[kani::proof]
    fn proof_extract_failed_checks_success() {
        let output = "Dafny program verifier finished with 5 verified, 0 errors";
        let checks = DafnyBackend::extract_failed_checks(output);
        kani::assert(checks.is_empty(), "Success output should yield no checks");
    }

    /// Verify extract_failed_checks handles might_not_hold pattern
    #[kani::proof]
    fn proof_extract_failed_checks_might_not_hold() {
        let output = "assertion might not hold";
        let checks = DafnyBackend::extract_failed_checks(output);
        kani::assert(checks.len() == 1, "Should find one check");
    }

    /// Verify extract_failed_checks handles cannot_be_proved pattern
    #[kani::proof]
    fn proof_extract_failed_checks_cannot_be_proved() {
        let output = "statement cannot be proved";
        let checks = DafnyBackend::extract_failed_checks(output);
        kani::assert(checks.len() == 1, "Should find one check");
    }

    // ---- parse_dafny_error_line Tests ----

    /// Verify parse_dafny_error_line extracts file from location
    #[kani::proof]
    fn proof_parse_dafny_error_line_file() {
        let line = "USLSpec.dfy(42,15): Error: assertion might not hold";
        let (loc, _desc) = DafnyBackend::parse_dafny_error_line(line);
        kani::assert(loc.is_some(), "Should parse location");
        kani::assert(
            loc.unwrap().file == "USLSpec.dfy",
            "Should extract file name",
        );
    }

    /// Verify parse_dafny_error_line extracts line number
    #[kani::proof]
    fn proof_parse_dafny_error_line_line() {
        let line = "USLSpec.dfy(42,15): Error: assertion might not hold";
        let (loc, _desc) = DafnyBackend::parse_dafny_error_line(line);
        kani::assert(loc.is_some(), "Should parse location");
        kani::assert(loc.unwrap().line == 42, "Should extract line 42");
    }

    /// Verify parse_dafny_error_line extracts column
    #[kani::proof]
    fn proof_parse_dafny_error_line_column() {
        let line = "USLSpec.dfy(42,15): Error: assertion might not hold";
        let (loc, _desc) = DafnyBackend::parse_dafny_error_line(line);
        kani::assert(loc.is_some(), "Should parse location");
        kani::assert(loc.unwrap().column == Some(15), "Should extract column 15");
    }

    /// Verify parse_dafny_error_line extracts description
    #[kani::proof]
    fn proof_parse_dafny_error_line_description() {
        let line = "USLSpec.dfy(42,15): Error: assertion might not hold";
        let (_loc, desc) = DafnyBackend::parse_dafny_error_line(line);
        kani::assert(
            desc.contains("assertion"),
            "Description should contain error message",
        );
    }

    /// Verify parse_dafny_error_line handles no location
    #[kani::proof]
    fn proof_parse_dafny_error_line_no_location() {
        let line = "Error: some general error";
        let (loc, desc) = DafnyBackend::parse_dafny_error_line(line);
        kani::assert(loc.is_none(), "Should have no location");
        kani::assert(
            desc.contains("some general error"),
            "Should extract description",
        );
    }

    /// Verify parse_dafny_error_line handles line without column
    #[kani::proof]
    fn proof_parse_dafny_error_line_no_column() {
        let line = "file.dfy(10): Error: test";
        let (loc, _desc) = DafnyBackend::parse_dafny_error_line(line);
        kani::assert(loc.is_some(), "Should parse location");
        let loc = loc.unwrap();
        kani::assert(loc.line == 10, "Should extract line");
        kani::assert(loc.column.is_none(), "Column should be None");
    }

    // ---- extract_function_name Tests ----

    /// Verify extract_function_name finds method
    #[kani::proof]
    fn proof_extract_function_name_method() {
        let lines: Vec<&str> = vec!["method Foo(x: int) returns (y: int)", "  Error: assertion"];
        let func = DafnyBackend::extract_function_name(&lines, 1);
        kani::assert(func == Some("Foo".to_string()), "Should find method Foo");
    }

    /// Verify extract_function_name finds lemma
    #[kani::proof]
    fn proof_extract_function_name_lemma() {
        let lines: Vec<&str> = vec!["lemma MyLemma()", "  Error: test"];
        let func = DafnyBackend::extract_function_name(&lines, 1);
        kani::assert(
            func == Some("MyLemma".to_string()),
            "Should find lemma MyLemma",
        );
    }

    /// Verify extract_function_name finds function
    #[kani::proof]
    fn proof_extract_function_name_function() {
        let lines: Vec<&str> = vec!["function ComputeValue(n: nat): nat", "  Error: test"];
        let func = DafnyBackend::extract_function_name(&lines, 1);
        kani::assert(
            func == Some("ComputeValue".to_string()),
            "Should find function ComputeValue",
        );
    }

    /// Verify extract_function_name finds predicate
    #[kani::proof]
    fn proof_extract_function_name_predicate() {
        let lines: Vec<&str> = vec!["predicate IsValid(s: seq<int>)", "  Error: test"];
        let func = DafnyBackend::extract_function_name(&lines, 1);
        kani::assert(
            func == Some("IsValid".to_string()),
            "Should find predicate IsValid",
        );
    }

    /// Verify extract_function_name returns None when no function
    #[kani::proof]
    fn proof_extract_function_name_no_function() {
        let lines: Vec<&str> = vec!["Error: test", "no declaration here"];
        let func = DafnyBackend::extract_function_name(&lines, 0);
        kani::assert(func.is_none(), "Should return None");
    }

    // ---- extract_first_error Tests ----

    /// Verify extract_first_error finds Error: pattern
    #[kani::proof]
    fn proof_extract_first_error_basic() {
        let output = "Warning: unused variable\nError: this is the error\nmore stuff";
        let err = DafnyBackend::extract_first_error(output);
        kani::assert(err.is_some(), "Should find error");
        kani::assert(
            err.unwrap().contains("this is the error"),
            "Should extract error message",
        );
    }

    /// Verify extract_first_error finds might not hold pattern
    #[kani::proof]
    fn proof_extract_first_error_might_not_hold() {
        let output = "assertion might not hold at line 10";
        let err = DafnyBackend::extract_first_error(output);
        kani::assert(err.is_some(), "Should find error");
        kani::assert(
            err.unwrap().contains("assertion"),
            "Should contain assertion",
        );
    }

    /// Verify extract_first_error returns None when no error
    #[kani::proof]
    fn proof_extract_first_error_no_error() {
        let output = "no errors here";
        let err = DafnyBackend::extract_first_error(output);
        kani::assert(err.is_none(), "Should return None");
    }

    // ---- parse_structured_counterexample Tests ----

    /// Verify parse_structured_counterexample preserves raw output
    #[kani::proof]
    fn proof_structured_counterexample_raw() {
        let ce = DafnyBackend::parse_structured_counterexample("stdout", "stderr");
        kani::assert(ce.raw.is_some(), "Should have raw output");
        kani::assert(
            ce.raw.as_ref().unwrap().contains("stdout"),
            "Raw should contain stdout",
        );
    }

    /// Verify parse_structured_counterexample extracts failed checks
    #[kani::proof]
    fn proof_structured_counterexample_failed_checks() {
        let ce = DafnyBackend::parse_structured_counterexample(
            "",
            "USLSpec.dfy(10,5): Error: assertion might not hold",
        );
        kani::assert(!ce.failed_checks.is_empty(), "Should have failed checks");
        kani::assert(
            ce.failed_checks[0].check_id == "dafny_assertion",
            "Should identify assertion",
        );
    }
}

use crate::counterexample::{FailedCheck, SourceLocation, StructuredCounterexample};
use crate::traits::*;
use crate::util::expand_home_dir;
use async_trait::async_trait;
use dashprove_usl::{compile_to_dafny, typecheck::TypedSpec};
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::process::Command;
use tracing::{debug, info, warn};

/// Configuration for Dafny backend
#[derive(Debug, Clone)]
pub struct DafnyConfig {
    /// Path to dafny executable (if not in PATH)
    pub dafny_path: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// Target language for compilation (if any)
    pub target: Option<String>,
}

impl Default for DafnyConfig {
    fn default() -> Self {
        Self {
            dafny_path: None,
            timeout: Duration::from_secs(300), // 5 minutes
            target: None,                      // Just verify, don't compile
        }
    }
}

/// Dafny verification backend
pub struct DafnyBackend {
    config: DafnyConfig,
}

#[derive(Debug, Clone)]
enum DafnyDetection {
    /// Dafny available
    Available { dafny_path: PathBuf },
    /// Dafny not found
    NotFound(String),
}

impl DafnyBackend {
    /// Create a new Dafny backend with default configuration
    pub fn new() -> Self {
        Self {
            config: DafnyConfig::default(),
        }
    }

    /// Create a new Dafny backend with custom configuration
    pub fn with_config(config: DafnyConfig) -> Self {
        Self { config }
    }

    /// Detect dafny installation
    async fn detect_dafny(&self) -> DafnyDetection {
        let dafny_path = if let Some(ref path) = self.config.dafny_path {
            if path.exists() {
                path.clone()
            } else {
                return DafnyDetection::NotFound(format!(
                    "Configured dafny path does not exist: {:?}",
                    path
                ));
            }
        } else {
            // Check for dafny in PATH
            if let Ok(path) = which::which("dafny") {
                path
            } else {
                // Check common installation locations
                let common_paths = [
                    expand_home_dir("~/.dafny/dafny"),
                    expand_home_dir("~/dafny/dafny"),
                    Some(PathBuf::from("/usr/local/bin/dafny")),
                    Some(PathBuf::from("/opt/dafny/dafny")),
                    Some(PathBuf::from("/opt/homebrew/bin/dafny")),
                ];

                let mut found = None;
                for path in common_paths.into_iter().flatten() {
                    if path.exists() {
                        found = Some(path);
                        break;
                    }
                }

                match found {
                    Some(path) => path,
                    None => {
                        return DafnyDetection::NotFound(
                            "dafny not found. Install Dafny from: https://github.com/dafny-lang/dafny/releases".to_string(),
                        );
                    }
                }
            }
        };

        // Verify dafny works
        let version_check = Command::new(&dafny_path)
            .arg("--version")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await;

        match version_check {
            Ok(output) if output.status.success() => {
                let version = String::from_utf8_lossy(&output.stdout);
                debug!("Found Dafny version: {}", version.trim());
                DafnyDetection::Available { dafny_path }
            }
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                DafnyDetection::NotFound(format!("dafny --version failed: {}", stderr))
            }
            Err(e) => DafnyDetection::NotFound(format!("Failed to execute dafny: {}", e)),
        }
    }

    /// Write Dafny project files to temp directory
    async fn write_project(
        &self,
        spec: &TypedSpec,
        dir: &std::path::Path,
    ) -> Result<PathBuf, BackendError> {
        let compiled = compile_to_dafny(spec);
        let dafny_code = compiled.code;
        let dafny_path = dir.join("USLSpec.dfy");

        tokio::fs::write(&dafny_path, &dafny_code)
            .await
            .map_err(|e| {
                BackendError::CompilationFailed(format!("Failed to write Dafny file: {}", e))
            })?;

        debug!("Written Dafny spec to {:?}", dafny_path);

        Ok(dafny_path)
    }

    /// Execute dafny verify and capture output
    async fn run_dafny_verify(
        &self,
        detection: &DafnyDetection,
        dafny_file: &std::path::Path,
    ) -> Result<DafnyOutput, BackendError> {
        let start = Instant::now();

        let dafny_path = match detection {
            DafnyDetection::Available { dafny_path } => dafny_path,
            DafnyDetection::NotFound(reason) => {
                return Err(BackendError::Unavailable(reason.clone()));
            }
        };

        let mut cmd = Command::new(dafny_path);
        cmd.arg("verify");
        cmd.arg(dafny_file);
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let output = tokio::time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to execute dafny verify: {}", e))
            })?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let duration = start.elapsed();

        debug!("dafny verify stdout:\n{}", stdout);
        if !stderr.is_empty() {
            debug!("dafny verify stderr:\n{}", stderr);
        }

        Ok(DafnyOutput {
            stdout,
            stderr,
            exit_code: output.status.code(),
            duration,
        })
    }

    /// Parse dafny verify output into verification result
    fn parse_output(&self, output: &DafnyOutput) -> BackendResult {
        let combined = format!("{}\n{}", output.stdout, output.stderr);

        // Check for successful verification
        // Dafny reports "Dafny program verifier finished with X verified, 0 errors"
        if output.exit_code == Some(0)
            && (combined.contains("verified, 0 errors") || combined.contains("0 errors"))
        {
            return BackendResult {
                backend: BackendId::Dafny,
                status: VerificationStatus::Proven,
                proof: Some("All assertions verified by Dafny".to_string()),
                counterexample: None,
                diagnostics: self.extract_diagnostics(&combined),
                time_taken: output.duration,
            };
        }

        // Check for verification failures
        if combined.contains("Error:")
            || combined.contains("assertion might not hold")
            || combined.contains("postcondition might not hold")
            || combined.contains("precondition might not hold")
            || combined.contains("invariant might not hold")
        {
            let error_msg = self.extract_error(&combined);

            // Actual verification failures (not syntax errors)
            if combined.contains("might not hold")
                || combined.contains("cannot be proved")
                || combined.contains("verification failed")
            {
                // Generate structured counterexample for verification failures
                let ce = Self::parse_structured_counterexample(&output.stdout, &output.stderr);

                return BackendResult {
                    backend: BackendId::Dafny,
                    status: VerificationStatus::Disproven,
                    proof: None,
                    counterexample: Some(ce),
                    diagnostics: self.extract_diagnostics(&combined),
                    time_taken: output.duration,
                };
            }

            return BackendResult {
                backend: BackendId::Dafny,
                status: VerificationStatus::Unknown {
                    reason: format!("Dafny error: {}", error_msg),
                },
                proof: None,
                counterexample: None,
                diagnostics: self.extract_diagnostics(&combined),
                time_taken: output.duration,
            };
        }

        // Check for partial success
        if combined.contains("verified") && combined.contains("error") {
            let verified_re = regex::Regex::new(r"(\d+) verified").ok();
            let errors_re = regex::Regex::new(r"(\d+) error").ok();

            let verified_count = verified_re
                .and_then(|re| re.captures(&combined))
                .and_then(|caps| caps.get(1))
                .and_then(|m| m.as_str().parse::<usize>().ok())
                .unwrap_or(0);

            let error_count = errors_re
                .and_then(|re| re.captures(&combined))
                .and_then(|caps| caps.get(1))
                .and_then(|m| m.as_str().parse::<usize>().ok())
                .unwrap_or(0);

            if verified_count > 0 && error_count > 0 {
                let total = verified_count + error_count;
                let percentage = (verified_count as f64 / total as f64) * 100.0;

                return BackendResult {
                    backend: BackendId::Dafny,
                    status: VerificationStatus::Partial {
                        verified_percentage: percentage,
                    },
                    proof: Some(format!("{} of {} verified", verified_count, total)),
                    counterexample: None,
                    diagnostics: self.extract_diagnostics(&combined),
                    time_taken: output.duration,
                };
            }
        }

        BackendResult {
            backend: BackendId::Dafny,
            status: VerificationStatus::Unknown {
                reason: "Could not determine verification result".to_string(),
            },
            proof: None,
            counterexample: None,
            diagnostics: vec![combined],
            time_taken: output.duration,
        }
    }

    fn extract_error(&self, output: &str) -> String {
        let mut errors = Vec::new();

        for line in output.lines() {
            if line.contains("Error:") || line.contains("error:") || line.contains("might not hold")
            {
                errors.push(line.to_string());
            }
        }

        if errors.is_empty() {
            "Unknown error".to_string()
        } else {
            errors.join("\n")
        }
    }

    fn extract_diagnostics(&self, output: &str) -> Vec<String> {
        let mut diagnostics = Vec::new();

        // Extract verification summary
        let summary_re = regex::Regex::new(r"(\d+) verified, (\d+) error").ok();
        if let Some(re) = summary_re {
            if let Some(caps) = re.captures(output) {
                let verified = caps.get(1).map(|m| m.as_str()).unwrap_or("?");
                let errors = caps.get(2).map(|m| m.as_str()).unwrap_or("?");
                diagnostics.push(format!("{} verified, {} errors", verified, errors));
            }
        }

        let warning_count = output.matches("Warning:").count();
        if warning_count > 0 {
            diagnostics.push(format!("{} warning(s)", warning_count));
        }

        diagnostics
    }

    /// Parse Dafny output into a structured counterexample with detailed failure info
    fn parse_structured_counterexample(stdout: &str, stderr: &str) -> StructuredCounterexample {
        let mut ce = StructuredCounterexample::new();
        let combined = format!("{}\n{}", stdout, stderr);
        ce.raw = Some(combined.clone());

        // Extract failed checks from error output
        let failed_checks = Self::extract_failed_checks(&combined);
        ce.failed_checks = failed_checks;

        ce
    }

    /// Extract failed checks from Dafny error output
    fn extract_failed_checks(output: &str) -> Vec<FailedCheck> {
        let mut checks = Vec::new();
        let lines: Vec<&str> = output.lines().collect();

        for i in 0..lines.len() {
            let line = lines[i];
            let trimmed = line.trim();

            // Dafny error pattern 1: "Error: message"
            // Dafny error pattern 2: "filename(line,col): Error: message"
            // Dafny error pattern 3: "assertion might not hold"
            if trimmed.contains("Error:")
                || trimmed.contains("might not hold")
                || trimmed.contains("cannot be proved")
            {
                let (location, description) = Self::parse_dafny_error_line(trimmed);

                // Determine error type
                let check_id = if description.contains("assertion")
                    || description.contains("assert")
                {
                    "dafny_assertion".to_string()
                } else if description.contains("precondition") || description.contains("requires") {
                    "dafny_precondition".to_string()
                } else if description.contains("postcondition") || description.contains("ensures") {
                    "dafny_postcondition".to_string()
                } else if description.contains("invariant") {
                    "dafny_invariant".to_string()
                } else if description.contains("decreases") || description.contains("termination") {
                    "dafny_termination".to_string()
                } else if description.contains("modifies") || description.contains("frame") {
                    "dafny_frame".to_string()
                } else if description.contains("Parsing") || description.contains("syntax") {
                    "dafny_syntax_error".to_string()
                } else if description.contains("resolution") || description.contains("undeclared") {
                    "dafny_resolution_error".to_string()
                } else {
                    "dafny_error".to_string()
                };

                // Try to find function name from context
                let function = Self::extract_function_name(&lines, i);

                if !description.is_empty() {
                    checks.push(FailedCheck {
                        check_id,
                        description,
                        location,
                        function,
                    });
                }
            }
        }

        // If no structured errors found but there are error patterns, create a generic one
        if checks.is_empty() && (output.contains("Error") || output.contains("error")) {
            if let Some(first_error) = Self::extract_first_error(output) {
                checks.push(FailedCheck {
                    check_id: "dafny_error".to_string(),
                    description: first_error,
                    location: None,
                    function: None,
                });
            }
        }

        checks
    }

    /// Parse a Dafny error line and extract location and description
    /// Dafny format: "filename(line,col): Error: message" or just "Error: message"
    fn parse_dafny_error_line(line: &str) -> (Option<SourceLocation>, String) {
        // Try pattern: "filename(line,col): Error: message"
        if let Some(paren_pos) = line.find('(') {
            if let Some(colon_pos) = line.find("): ") {
                let file = &line[..paren_pos];
                let loc_str = &line[paren_pos + 1..colon_pos];

                // Parse "line,col" or just "line"
                let parts: Vec<&str> = loc_str.split(',').collect();
                if let Some(line_num) = parts.first().and_then(|s| s.parse::<u32>().ok()) {
                    let column = parts.get(1).and_then(|s| s.parse::<u32>().ok());
                    let description = &line[colon_pos + 3..];

                    // Clean up description by removing "Error:" prefix if present
                    let description = if let Some(stripped) = description.strip_prefix("Error:") {
                        stripped.trim()
                    } else {
                        description.trim()
                    };

                    return (
                        Some(SourceLocation {
                            file: file.to_string(),
                            line: line_num,
                            column,
                        }),
                        description.to_string(),
                    );
                }
            }
        }

        // Try pattern: just "Error: message"
        if let Some(error_pos) = line.find("Error:") {
            let description = &line[error_pos + 6..];
            return (None, description.trim().to_string());
        }

        // Fall back to extracting the whole message
        (None, line.to_string())
    }

    /// Extract function name from context around the error
    fn extract_function_name(lines: &[&str], error_idx: usize) -> Option<String> {
        // Look for "method name", "function name", or "lemma name" in surrounding lines
        let search_range = error_idx.saturating_sub(10)..lines.len().min(error_idx + 5);

        for idx in search_range {
            let line = lines[idx];
            // Dafny uses: method Name, function Name, lemma Name, predicate Name
            for keyword in &["method", "function", "lemma", "predicate", "ghost method"] {
                if let Some(kw_pos) = line.find(keyword) {
                    let after_kw = &line[kw_pos + keyword.len()..];
                    let name: String = after_kw
                        .trim()
                        .chars()
                        .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '\'')
                        .collect();
                    if !name.is_empty() {
                        return Some(name);
                    }
                }
            }
        }
        None
    }

    /// Extract the first error message from output
    fn extract_first_error(output: &str) -> Option<String> {
        for line in output.lines() {
            let trimmed = line.trim();
            if trimmed.contains("Error:") {
                let msg = if let Some(pos) = trimmed.find("Error:") {
                    trimmed[pos + 6..].trim().to_string()
                } else {
                    trimmed.to_string()
                };
                if !msg.is_empty() {
                    return Some(msg);
                }
            }
            // Also check for "might not hold" patterns
            if trimmed.contains("might not hold") || trimmed.contains("cannot be proved") {
                return Some(trimmed.to_string());
            }
        }
        None
    }
}

impl Default for DafnyBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for DafnyBackend {
    fn id(&self) -> BackendId {
        BackendId::Dafny
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::Contract,
            PropertyType::Invariant,
            PropertyType::Theorem,
        ]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let detection = self.detect_dafny().await;

        if let DafnyDetection::NotFound(reason) = &detection {
            return Err(BackendError::Unavailable(reason.clone()));
        }

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::CompilationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let dafny_path = self.write_project(spec, temp_dir.path()).await?;
        let output = self.run_dafny_verify(&detection, &dafny_path).await?;
        let result = self.parse_output(&output);

        Ok(result)
    }

    async fn health_check(&self) -> HealthStatus {
        let detection = self.detect_dafny().await;
        match detection {
            DafnyDetection::Available { dafny_path } => {
                info!("Dafny available: {:?}", dafny_path);
                HealthStatus::Healthy
            }
            DafnyDetection::NotFound(reason) => {
                warn!("Dafny not available: {}", reason);
                HealthStatus::Unavailable { reason }
            }
        }
    }
}

/// Dafny execution output
struct DafnyOutput {
    stdout: String,
    stderr: String,
    exit_code: Option<i32>,
    duration: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;
    use dashprove_usl::{parse, typecheck};

    fn make_typed_spec(input: &str) -> TypedSpec {
        let spec = parse(input).expect("parse failed");
        typecheck(spec).expect("typecheck failed")
    }

    #[test]
    fn test_config_default() {
        let config = DafnyConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert!(config.target.is_none());
    }

    #[test]
    fn test_compile_generates_dafny() {
        let input = r#"
            theorem test {
                forall x: Bool . x or not x
            }
        "#;
        let spec = make_typed_spec(input);
        let compiled = compile_to_dafny(&spec);
        let dafny = compiled.code;

        assert!(dafny.contains("// Generated by DashProve"));
        assert!(dafny.contains("lemma test"));
        assert!(dafny.contains("ensures"));
    }

    #[test]
    fn test_parse_success_output() {
        let backend = DafnyBackend::new();
        let output = DafnyOutput {
            stdout: "Dafny program verifier finished with 5 verified, 0 errors".to_string(),
            stderr: String::new(),
            exit_code: Some(0),
            duration: Duration::from_secs(3),
        };

        let result = backend.parse_output(&output);
        assert!(matches!(result.status, VerificationStatus::Proven));
    }

    #[test]
    fn test_parse_verification_failure() {
        let backend = DafnyBackend::new();
        let output = DafnyOutput {
            stdout: String::new(),
            stderr: "Error: assertion might not hold".to_string(),
            exit_code: Some(1),
            duration: Duration::from_secs(2),
        };

        let result = backend.parse_output(&output);
        assert!(matches!(result.status, VerificationStatus::Disproven));
    }

    #[test]
    fn test_parse_partial_verification() {
        let backend = DafnyBackend::new();
        let output = DafnyOutput {
            stdout: "Dafny program verifier finished with 3 verified, 2 errors".to_string(),
            stderr: String::new(),
            exit_code: Some(1),
            duration: Duration::from_secs(5),
        };

        let result = backend.parse_output(&output);
        match result.status {
            VerificationStatus::Partial {
                verified_percentage,
            } => {
                assert!(verified_percentage > 50.0 && verified_percentage < 70.0);
            }
            _ => panic!("Expected Partial status"),
        }
    }

    #[tokio::test]
    async fn test_health_check_reports_status() {
        let backend = DafnyBackend::new();
        let status = backend.health_check().await;

        match status {
            HealthStatus::Healthy => println!("Dafny is available"),
            HealthStatus::Unavailable { reason } => println!("Dafny not available: {}", reason),
            HealthStatus::Degraded { reason } => println!("Dafny degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_verify_returns_result_or_unavailable() {
        let backend = DafnyBackend::new();
        let input = r#"
            theorem test {
                forall x: Bool . x or not x
            }
        "#;
        let spec = make_typed_spec(input);

        let result = backend.verify(&spec).await;
        match result {
            Ok(r) => println!("Verification result: {:?}", r.status),
            Err(BackendError::Unavailable(reason)) => println!("Backend unavailable: {}", reason),
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    // =============================================
    // Structured counterexample parsing tests
    // =============================================

    #[test]
    fn extract_failed_checks_assertion() {
        let output = "USLSpec.dfy(10,5): Error: assertion might not hold";
        let checks = DafnyBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "dafny_assertion");
        assert!(checks[0].description.contains("assertion"));
        assert!(checks[0].location.is_some());
        let loc = checks[0].location.as_ref().unwrap();
        assert_eq!(loc.file, "USLSpec.dfy");
        assert_eq!(loc.line, 10);
        assert_eq!(loc.column, Some(5));
    }

    #[test]
    fn extract_failed_checks_precondition() {
        let output = "Error: precondition might not hold";
        let checks = DafnyBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "dafny_precondition");
    }

    #[test]
    fn extract_failed_checks_postcondition() {
        let output = "Error: postcondition might not hold";
        let checks = DafnyBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "dafny_postcondition");
    }

    #[test]
    fn extract_failed_checks_ensures() {
        let output = "Error: ensures clause might not hold";
        let checks = DafnyBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "dafny_postcondition");
    }

    #[test]
    fn extract_failed_checks_requires() {
        let output = "Error: requires clause might not hold";
        let checks = DafnyBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "dafny_precondition");
    }

    #[test]
    fn extract_failed_checks_invariant() {
        let output = "Error: loop invariant might not hold";
        let checks = DafnyBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "dafny_invariant");
    }

    #[test]
    fn extract_failed_checks_termination() {
        let output = "Error: decreases expression might not decrease";
        let checks = DafnyBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "dafny_termination");
    }

    #[test]
    fn extract_failed_checks_frame() {
        let output = "Error: modifies clause might not hold";
        let checks = DafnyBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "dafny_frame");
    }

    #[test]
    fn extract_failed_checks_syntax() {
        let output = "Error: Parsing error: expected identifier";
        let checks = DafnyBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "dafny_syntax_error");
    }

    #[test]
    fn extract_failed_checks_resolution() {
        let output = "Error: undeclared identifier: foo";
        let checks = DafnyBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "dafny_resolution_error");
    }

    #[test]
    fn extract_failed_checks_generic() {
        let output = "Error: some other verification error";
        let checks = DafnyBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "dafny_error");
    }

    #[test]
    fn extract_failed_checks_multiple() {
        let output = r#"USLSpec.dfy(10,5): Error: assertion might not hold
USLSpec.dfy(20,3): Error: postcondition might not hold"#;
        let checks = DafnyBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 2);
        assert_eq!(checks[0].check_id, "dafny_assertion");
        assert_eq!(checks[1].check_id, "dafny_postcondition");
    }

    #[test]
    fn extract_failed_checks_empty() {
        let checks = DafnyBackend::extract_failed_checks("");
        assert!(checks.is_empty());
    }

    #[test]
    fn extract_failed_checks_success_output() {
        let output = "Dafny program verifier finished with 5 verified, 0 errors";
        let checks = DafnyBackend::extract_failed_checks(output);
        assert!(checks.is_empty());
    }

    #[test]
    fn extract_failed_checks_might_not_hold() {
        let output = "assertion might not hold";
        let checks = DafnyBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "dafny_assertion");
    }

    #[test]
    fn extract_failed_checks_cannot_be_proved() {
        let output = "statement cannot be proved";
        let checks = DafnyBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
    }

    #[test]
    fn parse_dafny_error_line_with_location() {
        let line = "USLSpec.dfy(42,15): Error: assertion might not hold";
        let (loc, desc) = DafnyBackend::parse_dafny_error_line(line);

        assert!(loc.is_some());
        let loc = loc.unwrap();
        assert_eq!(loc.file, "USLSpec.dfy");
        assert_eq!(loc.line, 42);
        assert_eq!(loc.column, Some(15));
        assert!(desc.contains("assertion"));
    }

    #[test]
    fn parse_dafny_error_line_without_location() {
        let line = "Error: some general error";
        let (loc, desc) = DafnyBackend::parse_dafny_error_line(line);

        assert!(loc.is_none());
        assert!(desc.contains("some general error"));
    }

    #[test]
    fn parse_dafny_error_line_no_column() {
        let line = "file.dfy(10): Error: test";
        let (loc, _desc) = DafnyBackend::parse_dafny_error_line(line);

        assert!(loc.is_some());
        let loc = loc.unwrap();
        assert_eq!(loc.line, 10);
        assert!(loc.column.is_none());
    }

    #[test]
    fn extract_function_name_method() {
        let lines: Vec<&str> = vec!["method Foo(x: int) returns (y: int)", "  Error: assertion"];
        let func = DafnyBackend::extract_function_name(&lines, 1);

        assert_eq!(func, Some("Foo".to_string()));
    }

    #[test]
    fn extract_function_name_lemma() {
        let lines: Vec<&str> = vec!["lemma MyLemma()", "  Error: test"];
        let func = DafnyBackend::extract_function_name(&lines, 1);

        assert_eq!(func, Some("MyLemma".to_string()));
    }

    #[test]
    fn extract_function_name_function() {
        let lines: Vec<&str> = vec!["function ComputeValue(n: nat): nat", "  Error: test"];
        let func = DafnyBackend::extract_function_name(&lines, 1);

        assert_eq!(func, Some("ComputeValue".to_string()));
    }

    #[test]
    fn extract_function_name_predicate() {
        let lines: Vec<&str> = vec!["predicate IsValid(s: seq<int>)", "  Error: test"];
        let func = DafnyBackend::extract_function_name(&lines, 1);

        assert_eq!(func, Some("IsValid".to_string()));
    }

    #[test]
    fn extract_function_name_no_function() {
        let lines: Vec<&str> = vec!["Error: test", "no declaration here"];
        let func = DafnyBackend::extract_function_name(&lines, 0);

        assert!(func.is_none());
    }

    #[test]
    fn structured_counterexample_has_raw() {
        let ce = DafnyBackend::parse_structured_counterexample("stdout", "stderr");
        assert!(ce.raw.is_some());
        assert!(ce.raw.as_ref().unwrap().contains("stdout"));
    }

    #[test]
    fn structured_counterexample_has_failed_checks() {
        let ce = DafnyBackend::parse_structured_counterexample(
            "",
            "USLSpec.dfy(10,5): Error: assertion might not hold",
        );

        assert!(!ce.failed_checks.is_empty());
        assert_eq!(ce.failed_checks[0].check_id, "dafny_assertion");
    }

    #[test]
    fn extract_first_error_basic() {
        let output = "Warning: unused variable\nError: this is the error\nmore stuff";
        let err = DafnyBackend::extract_first_error(output);

        assert!(err.is_some());
        assert!(err.unwrap().contains("this is the error"));
    }

    #[test]
    fn extract_first_error_might_not_hold() {
        let output = "assertion might not hold at line 10";
        let err = DafnyBackend::extract_first_error(output);

        assert!(err.is_some());
        assert!(err.unwrap().contains("assertion"));
    }

    #[test]
    fn extract_first_error_none() {
        let output = "no errors here";
        let err = DafnyBackend::extract_first_error(output);

        assert!(err.is_none());
    }
}
