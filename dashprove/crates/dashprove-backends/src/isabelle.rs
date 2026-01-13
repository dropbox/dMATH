//! Isabelle/HOL backend implementation
//!
//! This backend executes Isabelle/HOL specifications using the isabelle command.
//! Isabelle is an interactive theorem prover for higher-order logic.

// =============================================
// Kani Proofs for Isabelle Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- IsabelleConfig Default Tests ----

    /// Verify IsabelleConfig::default timeout is 300 seconds
    #[kani::proof]
    fn proof_isabelle_config_default_timeout() {
        let config = IsabelleConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "Default timeout should be 300 seconds",
        );
    }

    /// Verify IsabelleConfig::default isabelle_path is None
    #[kani::proof]
    fn proof_isabelle_config_default_path_none() {
        let config = IsabelleConfig::default();
        kani::assert(
            config.isabelle_path.is_none(),
            "Default isabelle_path should be None",
        );
    }

    /// Verify IsabelleConfig::default session is HOL
    #[kani::proof]
    fn proof_isabelle_config_default_session() {
        let config = IsabelleConfig::default();
        kani::assert(config.session == "HOL", "Default session should be HOL");
    }

    // ---- IsabelleBackend Construction Tests ----

    /// Verify IsabelleBackend::new uses default config
    #[kani::proof]
    fn proof_isabelle_backend_new_defaults() {
        let backend = IsabelleBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "New backend should use default timeout",
        );
    }

    /// Verify IsabelleBackend::default equals IsabelleBackend::new
    #[kani::proof]
    fn proof_isabelle_backend_default_equals_new() {
        let default_backend = IsabelleBackend::default();
        let new_backend = IsabelleBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
    }

    /// Verify IsabelleBackend::with_config preserves custom timeout
    #[kani::proof]
    fn proof_isabelle_backend_with_config_timeout() {
        let config = IsabelleConfig {
            isabelle_path: None,
            timeout: Duration::from_secs(600),
            session: "HOL".to_string(),
        };
        let backend = IsabelleBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "Custom timeout should be preserved",
        );
    }

    /// Verify IsabelleBackend::with_config preserves custom session
    #[kani::proof]
    fn proof_isabelle_backend_with_config_session() {
        let config = IsabelleConfig {
            isabelle_path: None,
            timeout: Duration::from_secs(300),
            session: "HOL-Library".to_string(),
        };
        let backend = IsabelleBackend::with_config(config);
        kani::assert(
            backend.config.session == "HOL-Library",
            "Custom session should be preserved",
        );
    }

    // ---- Backend Trait Implementation Tests ----

    /// Verify id() returns Isabelle
    #[kani::proof]
    fn proof_backend_id_is_isabelle() {
        let backend = IsabelleBackend::new();
        kani::assert(backend.id() == BackendId::Isabelle, "ID should be Isabelle");
    }

    /// Verify supports() includes Theorem
    #[kani::proof]
    fn proof_isabelle_supports_theorem() {
        let backend = IsabelleBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Theorem),
            "Should support Theorem",
        );
    }

    /// Verify supports() includes Invariant
    #[kani::proof]
    fn proof_isabelle_supports_invariant() {
        let backend = IsabelleBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Invariant),
            "Should support Invariant",
        );
    }

    /// Verify supports() includes Refinement
    #[kani::proof]
    fn proof_isabelle_supports_refinement() {
        let backend = IsabelleBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Refinement),
            "Should support Refinement",
        );
    }

    /// Verify supports() returns exactly three property types
    #[kani::proof]
    fn proof_isabelle_supports_count() {
        let backend = IsabelleBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 3,
            "Should support exactly three property types",
        );
    }

    // ---- IsabelleDetection Enum Tests ----

    /// Verify IsabelleDetection::Available variant stores path
    #[kani::proof]
    fn proof_isabelle_detection_available() {
        let path = PathBuf::from("/opt/Isabelle2024/bin/isabelle");
        let detection = IsabelleDetection::Available {
            isabelle_path: path.clone(),
        };
        match detection {
            IsabelleDetection::Available { isabelle_path } => {
                kani::assert(
                    isabelle_path == path,
                    "Available should preserve isabelle_path",
                );
            }
            _ => kani::assert(false, "Should be Available variant"),
        }
    }

    /// Verify IsabelleDetection::NotFound variant stores reason
    #[kani::proof]
    fn proof_isabelle_detection_not_found() {
        let reason = "isabelle not found".to_string();
        let detection = IsabelleDetection::NotFound(reason.clone());
        match detection {
            IsabelleDetection::NotFound(r) => {
                kani::assert(r == reason, "NotFound should preserve reason");
            }
            _ => kani::assert(false, "Should be NotFound variant"),
        }
    }

    // ---- extract_failed_checks Tests ----

    /// Verify extract_failed_checks identifies proof failed
    #[kani::proof]
    fn proof_extract_failed_checks_proof_failed() {
        let output = "*** Proof failed";
        let checks = IsabelleBackend::extract_failed_checks(output);
        kani::assert(checks.len() == 1, "Should find one check");
        kani::assert(
            checks[0].check_id == "isabelle_proof_failed",
            "Should identify as proof_failed",
        );
    }

    /// Verify extract_failed_checks identifies unfinished goals
    #[kani::proof]
    fn proof_extract_failed_checks_unfinished_goals() {
        let output = "*** Unfinished goals: 2";
        let checks = IsabelleBackend::extract_failed_checks(output);
        kani::assert(checks.len() == 1, "Should find one check");
        kani::assert(
            checks[0].check_id == "isabelle_unfinished_goals",
            "Should identify as unfinished_goals",
        );
    }

    /// Verify extract_failed_checks identifies type error
    #[kani::proof]
    fn proof_extract_failed_checks_type_error() {
        let output = "*** Type unification failed";
        let checks = IsabelleBackend::extract_failed_checks(output);
        kani::assert(checks.len() == 1, "Should find one check");
        kani::assert(
            checks[0].check_id == "isabelle_type_error",
            "Should identify as type_error",
        );
    }

    /// Verify extract_failed_checks identifies undefined
    #[kani::proof]
    fn proof_extract_failed_checks_undefined() {
        let output = "*** Undefined constant: foo";
        let checks = IsabelleBackend::extract_failed_checks(output);
        kani::assert(checks.len() == 1, "Should find one check");
        kani::assert(
            checks[0].check_id == "isabelle_undefined",
            "Should identify as undefined",
        );
    }

    /// Verify extract_failed_checks handles empty input
    #[kani::proof]
    fn proof_extract_failed_checks_empty() {
        let checks = IsabelleBackend::extract_failed_checks("");
        kani::assert(checks.is_empty(), "Empty input should yield no checks");
    }

    /// Verify extract_failed_checks handles success output
    #[kani::proof]
    fn proof_extract_failed_checks_success() {
        let output = "Finished USLSpec (0:00:05 elapsed time)";
        let checks = IsabelleBackend::extract_failed_checks(output);
        kani::assert(checks.is_empty(), "Success output should yield no checks");
    }

    /// Verify extract_failed_checks handles multiple errors
    #[kani::proof]
    fn proof_extract_failed_checks_multiple() {
        let output = "*** Error 1\n*** Error 2";
        let checks = IsabelleBackend::extract_failed_checks(output);
        kani::assert(checks.len() == 2, "Should find two checks");
    }

    // ---- parse_isabelle_location Tests ----

    /// Verify parse_isabelle_location extracts line number
    #[kani::proof]
    fn proof_parse_isabelle_location_line() {
        let line = "*** At command at line 42";
        let loc = IsabelleBackend::parse_isabelle_location(line);
        kani::assert(loc.is_some(), "Should parse location");
        kani::assert(loc.unwrap().line == 42, "Should extract line 42");
    }

    /// Verify parse_isabelle_location extracts position
    #[kani::proof]
    fn proof_parse_isabelle_location_position() {
        let line = "*** position 123 in theory";
        let loc = IsabelleBackend::parse_isabelle_location(line);
        kani::assert(loc.is_some(), "Should parse location");
        kani::assert(loc.unwrap().line == 123, "Should extract position as line");
    }

    /// Verify parse_isabelle_location returns None when no location
    #[kani::proof]
    fn proof_parse_isabelle_location_no_location() {
        let line = "*** Proof failed";
        let loc = IsabelleBackend::parse_isabelle_location(line);
        kani::assert(loc.is_none(), "Should return None");
    }

    // ---- extract_first_error Tests ----

    /// Verify extract_first_error finds *** pattern
    #[kani::proof]
    fn proof_extract_first_error_asterisk() {
        let output = "Some preamble\n*** This is the error\nMore stuff";
        let error = IsabelleBackend::extract_first_error(output);
        kani::assert(
            error.contains("This is the error"),
            "Should extract error message",
        );
    }

    /// Verify extract_first_error finds Error: pattern
    #[kani::proof]
    fn proof_extract_first_error_error_prefix() {
        let output = "Compiling...\nError: Something failed\nDone.";
        let error = IsabelleBackend::extract_first_error(output);
        kani::assert(
            error.contains("Error: Something failed"),
            "Should extract error message",
        );
    }

    /// Verify extract_first_error returns unknown for no error
    #[kani::proof]
    fn proof_extract_first_error_unknown() {
        let output = "No error pattern here";
        let error = IsabelleBackend::extract_first_error(output);
        kani::assert(error == "Unknown Isabelle error", "Should return unknown");
    }

    // ---- extract_diagnostics Tests ----

    /// Verify extract_diagnostics counts warnings
    #[kani::proof]
    fn proof_extract_diagnostics_warnings() {
        let backend = IsabelleBackend::new();
        let output = "Warning: foo\nWarning: bar";
        let diagnostics = backend.extract_diagnostics(output);
        kani::assert(
            diagnostics.iter().any(|d| d.contains("2 warning")),
            "Should count warnings",
        );
    }

    /// Verify extract_diagnostics counts errors
    #[kani::proof]
    fn proof_extract_diagnostics_errors() {
        let backend = IsabelleBackend::new();
        let output = "*** error 1\n*** error 2\n*** error 3";
        let diagnostics = backend.extract_diagnostics(output);
        kani::assert(
            diagnostics.iter().any(|d| d.contains("3 error")),
            "Should count errors",
        );
    }

    /// Verify extract_diagnostics handles no issues
    #[kani::proof]
    fn proof_extract_diagnostics_empty() {
        let backend = IsabelleBackend::new();
        let output = "All good";
        let diagnostics = backend.extract_diagnostics(output);
        kani::assert(diagnostics.is_empty(), "Should have no diagnostics");
    }

    // ---- parse_output Tests ----

    /// Verify parse_output recognizes success
    #[kani::proof]
    fn proof_parse_output_success() {
        let backend = IsabelleBackend::new();
        let output = IsabelleOutput {
            stdout: "Finished USLSpec".to_string(),
            stderr: String::new(),
            exit_code: Some(0),
            duration: Duration::from_secs(5),
        };
        let result = backend.parse_output(&output);
        kani::assert(
            matches!(result.status, VerificationStatus::Proven),
            "Success should return Proven",
        );
    }

    /// Verify parse_output recognizes proof failure
    #[kani::proof]
    fn proof_parse_output_proof_failure() {
        let backend = IsabelleBackend::new();
        let output = IsabelleOutput {
            stdout: String::new(),
            stderr: "*** Proof failed".to_string(),
            exit_code: Some(1),
            duration: Duration::from_secs(2),
        };
        let result = backend.parse_output(&output);
        kani::assert(
            matches!(result.status, VerificationStatus::Disproven),
            "Proof failed should return Disproven",
        );
    }

    /// Verify parse_output recognizes unfinished goals
    #[kani::proof]
    fn proof_parse_output_unfinished_goals() {
        let backend = IsabelleBackend::new();
        let output = IsabelleOutput {
            stdout: String::new(),
            stderr: "*** Unfinished goals".to_string(),
            exit_code: Some(1),
            duration: Duration::from_secs(2),
        };
        let result = backend.parse_output(&output);
        kani::assert(
            matches!(result.status, VerificationStatus::Disproven),
            "Unfinished goals should return Disproven",
        );
    }

    /// Verify parse_output includes counterexample on failure
    #[kani::proof]
    fn proof_parse_output_has_counterexample() {
        let backend = IsabelleBackend::new();
        let output = IsabelleOutput {
            stdout: String::new(),
            stderr: "*** Proof failed".to_string(),
            exit_code: Some(1),
            duration: Duration::from_secs(2),
        };
        let result = backend.parse_output(&output);
        kani::assert(
            result.counterexample.is_some(),
            "Failure should include counterexample",
        );
    }

    /// Verify parse_output recognizes syntax error
    #[kani::proof]
    fn proof_parse_output_syntax_error() {
        let backend = IsabelleBackend::new();
        let output = IsabelleOutput {
            stdout: String::new(),
            stderr: "*** Syntax error at position 42".to_string(),
            exit_code: Some(1),
            duration: Duration::from_secs(1),
        };
        let result = backend.parse_output(&output);
        kani::assert(
            matches!(result.status, VerificationStatus::Unknown { .. }),
            "Syntax error should return Unknown",
        );
    }

    // ---- StructuredCounterexample Tests ----

    /// Verify parse_structured_counterexample preserves raw output
    #[kani::proof]
    fn proof_structured_counterexample_raw() {
        let backend = IsabelleBackend::new();
        let output = "*** Proof failed";
        let ce = backend.parse_structured_counterexample(output);
        kani::assert(ce.raw.is_some(), "Should have raw output");
        kani::assert(ce.raw.unwrap() == output, "Raw should match input");
    }

    /// Verify parse_structured_counterexample extracts failed checks
    #[kani::proof]
    fn proof_structured_counterexample_failed_checks() {
        let backend = IsabelleBackend::new();
        let output = "*** Proof failed";
        let ce = backend.parse_structured_counterexample(output);
        kani::assert(!ce.failed_checks.is_empty(), "Should have failed checks");
    }
}

use crate::counterexample::{FailedCheck, SourceLocation, StructuredCounterexample};
use crate::traits::*;
use crate::util::expand_home_dir;
use async_trait::async_trait;
use dashprove_usl::{compile_to_isabelle, typecheck::TypedSpec};
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::process::Command;
use tracing::{debug, info, warn};

/// Configuration for Isabelle backend
#[derive(Debug, Clone)]
pub struct IsabelleConfig {
    /// Path to isabelle executable (if not in PATH)
    pub isabelle_path: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// Session to build (e.g., "HOL" for basic HOL)
    pub session: String,
}

impl Default for IsabelleConfig {
    fn default() -> Self {
        Self {
            isabelle_path: None,
            timeout: Duration::from_secs(300), // 5 minutes
            session: "HOL".to_string(),
        }
    }
}

/// Isabelle/HOL verification backend
pub struct IsabelleBackend {
    config: IsabelleConfig,
}

#[derive(Debug, Clone)]
enum IsabelleDetection {
    /// Isabelle available
    Available { isabelle_path: PathBuf },
    /// Isabelle not found
    NotFound(String),
}

impl IsabelleBackend {
    /// Create a new Isabelle backend with default configuration
    pub fn new() -> Self {
        Self {
            config: IsabelleConfig::default(),
        }
    }

    /// Create a new Isabelle backend with custom configuration
    pub fn with_config(config: IsabelleConfig) -> Self {
        Self { config }
    }

    /// Detect isabelle installation
    async fn detect_isabelle(&self) -> IsabelleDetection {
        let isabelle_path = if let Some(ref path) = self.config.isabelle_path {
            if path.exists() {
                path.clone()
            } else {
                return IsabelleDetection::NotFound(format!(
                    "Configured isabelle path does not exist: {:?}",
                    path
                ));
            }
        } else {
            // Check for isabelle in PATH
            if let Ok(path) = which::which("isabelle") {
                path
            } else {
                // Check common installation locations
                let common_paths = [
                    expand_home_dir("~/Isabelle2024/bin/isabelle"),
                    expand_home_dir("~/Isabelle2023/bin/isabelle"),
                    Some(PathBuf::from("/usr/local/bin/isabelle")),
                    Some(PathBuf::from("/opt/Isabelle2024/bin/isabelle")),
                    Some(PathBuf::from("/opt/Isabelle2023/bin/isabelle")),
                    Some(PathBuf::from("/Applications/Isabelle2024.app/bin/isabelle")),
                    Some(PathBuf::from("/Applications/Isabelle2023.app/bin/isabelle")),
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
                        return IsabelleDetection::NotFound(
                            "isabelle not found. Install Isabelle from: https://isabelle.in.tum.de/".to_string(),
                        );
                    }
                }
            }
        };

        // Verify isabelle works
        let version_check = Command::new(&isabelle_path)
            .arg("version")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await;

        match version_check {
            Ok(output) if output.status.success() => {
                let version = String::from_utf8_lossy(&output.stdout);
                debug!("Found Isabelle version: {}", version.trim());
                IsabelleDetection::Available { isabelle_path }
            }
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                IsabelleDetection::NotFound(format!("isabelle version failed: {}", stderr))
            }
            Err(e) => IsabelleDetection::NotFound(format!("Failed to execute isabelle: {}", e)),
        }
    }

    /// Write Isabelle project files to temp directory
    async fn write_project(
        &self,
        spec: &TypedSpec,
        dir: &std::path::Path,
    ) -> Result<PathBuf, BackendError> {
        let compiled = compile_to_isabelle(spec);
        let theory_code = compiled.code;
        let theory_path = dir.join("USLSpec.thy");

        tokio::fs::write(&theory_path, &theory_code)
            .await
            .map_err(|e| {
                BackendError::CompilationFailed(format!("Failed to write theory file: {}", e))
            })?;

        // Write ROOT file for session management
        let root_content = format!(
            "session USLSpec = {} +\n  theories USLSpec\n",
            self.config.session
        );
        let root_path = dir.join("ROOT");
        tokio::fs::write(&root_path, &root_content)
            .await
            .map_err(|e| {
                BackendError::CompilationFailed(format!("Failed to write ROOT file: {}", e))
            })?;

        debug!("Written Isabelle theory to {:?}", theory_path);
        debug!("Written ROOT file to {:?}", root_path);

        Ok(theory_path)
    }

    /// Execute isabelle build and capture output
    async fn run_isabelle_build(
        &self,
        detection: &IsabelleDetection,
        project_dir: &std::path::Path,
    ) -> Result<IsabelleOutput, BackendError> {
        let start = Instant::now();

        let isabelle_path = match detection {
            IsabelleDetection::Available { isabelle_path } => isabelle_path,
            IsabelleDetection::NotFound(reason) => {
                return Err(BackendError::Unavailable(reason.clone()));
            }
        };

        let mut cmd = Command::new(isabelle_path);
        cmd.arg("build");
        cmd.arg("-D");
        cmd.arg(".");
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        cmd.current_dir(project_dir);

        let output = tokio::time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to execute isabelle build: {}", e))
            })?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let duration = start.elapsed();

        debug!("isabelle build stdout:\n{}", stdout);
        if !stderr.is_empty() {
            debug!("isabelle build stderr:\n{}", stderr);
        }

        Ok(IsabelleOutput {
            stdout,
            stderr,
            exit_code: output.status.code(),
            duration,
        })
    }

    /// Parse isabelle build output into verification result
    fn parse_output(&self, output: &IsabelleOutput) -> BackendResult {
        let combined = format!("{}\n{}", output.stdout, output.stderr);

        // Check for successful build
        if output.exit_code == Some(0) {
            return BackendResult {
                backend: BackendId::Isabelle,
                status: VerificationStatus::Proven,
                proof: Some("All theories verified successfully".to_string()),
                counterexample: None,
                diagnostics: self.extract_diagnostics(&combined),
                time_taken: output.duration,
            };
        }

        // Check for proof failures
        if combined.contains("Failed to finish proof")
            || combined.contains("*** Proof failed")
            || combined.contains("*** Unfinished goals")
        {
            return BackendResult {
                backend: BackendId::Isabelle,
                status: VerificationStatus::Disproven,
                proof: None,
                counterexample: Some(self.parse_structured_counterexample(&combined)),
                diagnostics: self.extract_diagnostics(&combined),
                time_taken: output.duration,
            };
        }

        // Check for syntax/type errors
        if combined.contains("*** ") || combined.contains("Error:") {
            let error_msg = self.extract_error(&combined);
            return BackendResult {
                backend: BackendId::Isabelle,
                status: VerificationStatus::Unknown {
                    reason: format!("Isabelle error: {}", error_msg),
                },
                proof: None,
                counterexample: Some(self.parse_structured_counterexample(&combined)),
                diagnostics: self.extract_diagnostics(&combined),
                time_taken: output.duration,
            };
        }

        BackendResult {
            backend: BackendId::Isabelle,
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
            if line.contains("***") || line.contains("Error:") {
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

        let warning_count = output.matches("Warning:").count();
        if warning_count > 0 {
            diagnostics.push(format!("{} warning(s)", warning_count));
        }

        let error_count = output.matches("***").count();
        if error_count > 0 {
            diagnostics.push(format!("{} error(s)", error_count));
        }

        diagnostics
    }

    /// Parse Isabelle output into a structured counterexample with detailed failure info
    fn parse_structured_counterexample(&self, output: &str) -> StructuredCounterexample {
        let mut ce = StructuredCounterexample::new();
        ce.raw = Some(output.to_string());

        // Extract failed checks from error output
        let failed_checks = Self::extract_failed_checks(output);
        ce.failed_checks = failed_checks;

        ce
    }

    /// Extract failed checks from Isabelle error output
    fn extract_failed_checks(output: &str) -> Vec<FailedCheck> {
        let mut checks = Vec::new();
        let lines: Vec<&str> = output.lines().collect();

        for i in 0..lines.len() {
            let line = lines[i];

            // Isabelle error markers: "*** " prefix
            if line.contains("*** ") {
                let description = line
                    .trim()
                    .strip_prefix("***")
                    .unwrap_or(line.trim())
                    .trim();

                // Determine error type
                let check_id = if description.contains("Failed to finish proof")
                    || description.contains("Proof failed")
                {
                    "isabelle_proof_failed".to_string()
                } else if description.contains("Unfinished goals") {
                    "isabelle_unfinished_goals".to_string()
                } else if description.contains("Type unification failed")
                    || description.contains("Type error")
                {
                    "isabelle_type_error".to_string()
                } else if description.contains("Undefined") || description.contains("Unknown fact")
                {
                    "isabelle_undefined".to_string()
                } else if description.contains("At command") || description.contains("position") {
                    // Position/location info, try to parse
                    if let Some(location) = Self::parse_isabelle_location(line) {
                        // This is location info, capture the next error line as description
                        if i + 1 < lines.len() {
                            let next_line = lines[i + 1].trim();
                            if next_line.starts_with("***") {
                                let next_desc =
                                    next_line.strip_prefix("***").unwrap_or(next_line).trim();
                                checks.push(FailedCheck {
                                    check_id: "isabelle_error".to_string(),
                                    description: next_desc.to_string(),
                                    location: Some(location),
                                    function: None,
                                });
                            }
                        }
                        continue;
                    }
                    "isabelle_error".to_string()
                } else {
                    "isabelle_error".to_string()
                };

                // Don't add empty descriptions
                if !description.is_empty() && !description.starts_with("At ") {
                    checks.push(FailedCheck {
                        check_id,
                        description: description.to_string(),
                        location: None,
                        function: None,
                    });
                }
            }

            // Check for "Error:" pattern
            if line.trim().starts_with("Error:") {
                let description = line
                    .trim()
                    .strip_prefix("Error:")
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if !description.is_empty() {
                    checks.push(FailedCheck {
                        check_id: "isabelle_error".to_string(),
                        description,
                        location: None,
                        function: None,
                    });
                }
            }
        }

        // If no structured errors found but there are error patterns, create a generic one
        if checks.is_empty() && (output.contains("***") || output.contains("Error:")) {
            checks.push(FailedCheck {
                check_id: "isabelle_error".to_string(),
                description: Self::extract_first_error(output),
                location: None,
                function: None,
            });
        }

        checks
    }

    /// Parse an Isabelle location string
    /// Format: "*** At command ..." or "position N" or "line N"
    fn parse_isabelle_location(line: &str) -> Option<SourceLocation> {
        // Look for "line N" pattern
        if let Some(pos) = line.find("line ") {
            let after_line = &line[pos + 5..];
            let line_num: u32 = after_line
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse()
                .ok()?;

            return Some(SourceLocation {
                file: "theory".to_string(),
                line: line_num,
                column: None,
            });
        }

        // Look for "position N" pattern
        if let Some(pos) = line.find("position ") {
            let after_pos = &line[pos + 9..];
            let position: u32 = after_pos
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse()
                .ok()?;

            return Some(SourceLocation {
                file: "theory".to_string(),
                line: position, // Use position as line approximation
                column: None,
            });
        }

        None
    }

    /// Extract the first error message from output
    fn extract_first_error(output: &str) -> String {
        for line in output.lines() {
            if line.contains("*** ") {
                return line
                    .trim()
                    .strip_prefix("***")
                    .unwrap_or(line.trim())
                    .trim()
                    .to_string();
            }
            if line.contains("Error:") {
                return line.trim().to_string();
            }
        }
        "Unknown Isabelle error".to_string()
    }
}

impl Default for IsabelleBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for IsabelleBackend {
    fn id(&self) -> BackendId {
        BackendId::Isabelle
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::Theorem,
            PropertyType::Invariant,
            PropertyType::Refinement,
        ]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let detection = self.detect_isabelle().await;

        if let IsabelleDetection::NotFound(reason) = &detection {
            return Err(BackendError::Unavailable(reason.clone()));
        }

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::CompilationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let _theory_path = self.write_project(spec, temp_dir.path()).await?;
        let output = self.run_isabelle_build(&detection, temp_dir.path()).await?;
        let result = self.parse_output(&output);

        Ok(result)
    }

    async fn health_check(&self) -> HealthStatus {
        let detection = self.detect_isabelle().await;
        match detection {
            IsabelleDetection::Available { isabelle_path } => {
                info!("Isabelle available: {:?}", isabelle_path);
                HealthStatus::Healthy
            }
            IsabelleDetection::NotFound(reason) => {
                warn!("Isabelle not available: {}", reason);
                HealthStatus::Unavailable { reason }
            }
        }
    }
}

/// Isabelle build execution output
struct IsabelleOutput {
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
        let config = IsabelleConfig::default();
        assert_eq!(config.session, "HOL");
        assert_eq!(config.timeout, Duration::from_secs(300));
    }

    #[test]
    fn test_compile_generates_theory() {
        let input = r#"
            theorem test {
                forall x: Bool . x or not x
            }
        "#;
        let spec = make_typed_spec(input);
        let compiled = compile_to_isabelle(&spec);

        assert!(compiled.code.contains("theory USLSpec"));
        assert!(compiled.code.contains("imports Main"));
        // Theorems are compiled as lemmas in Isabelle
        assert!(compiled.code.contains("lemma test"));
        assert!(compiled.code.contains("end"));
    }

    #[test]
    fn test_parse_success_output() {
        let backend = IsabelleBackend::new();
        let output = IsabelleOutput {
            stdout: "Finished USLSpec (0:00:05 elapsed time)".to_string(),
            stderr: String::new(),
            exit_code: Some(0),
            duration: Duration::from_secs(5),
        };

        let result = backend.parse_output(&output);
        assert!(matches!(result.status, VerificationStatus::Proven));
    }

    #[test]
    fn test_parse_proof_failure() {
        let backend = IsabelleBackend::new();
        let output = IsabelleOutput {
            stdout: String::new(),
            stderr: "*** Proof failed".to_string(),
            exit_code: Some(1),
            duration: Duration::from_secs(2),
        };

        let result = backend.parse_output(&output);
        assert!(matches!(result.status, VerificationStatus::Disproven));
    }

    #[tokio::test]
    async fn test_health_check_reports_status() {
        let backend = IsabelleBackend::new();
        let status = backend.health_check().await;

        match status {
            HealthStatus::Healthy => println!("Isabelle is available"),
            HealthStatus::Unavailable { reason } => println!("Isabelle not available: {}", reason),
            HealthStatus::Degraded { reason } => println!("Isabelle degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_verify_returns_result_or_unavailable() {
        let backend = IsabelleBackend::new();
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
    fn test_extract_failed_checks_proof_failed() {
        let output = "*** Proof failed";
        let checks = IsabelleBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "isabelle_proof_failed");
        assert!(checks[0].description.contains("Proof failed"));
    }

    #[test]
    fn test_extract_failed_checks_unfinished_goals() {
        let output = "*** Unfinished goals: 2";
        let checks = IsabelleBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "isabelle_unfinished_goals");
        assert!(checks[0].description.contains("Unfinished goals"));
    }

    #[test]
    fn test_extract_failed_checks_type_error() {
        let output = "*** Type unification failed: nat vs bool";
        let checks = IsabelleBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "isabelle_type_error");
        assert!(checks[0].description.contains("Type unification failed"));
    }

    #[test]
    fn test_extract_failed_checks_undefined() {
        let output = "*** Undefined constant: foo";
        let checks = IsabelleBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "isabelle_undefined");
        assert!(checks[0].description.contains("Undefined constant"));
    }

    #[test]
    fn test_extract_failed_checks_generic_error() {
        let output = "*** Some other error message";
        let checks = IsabelleBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "isabelle_error");
        assert!(checks[0].description.contains("Some other error"));
    }

    #[test]
    fn test_extract_failed_checks_multiple_errors() {
        let output = "*** Error 1\n*** Error 2\n*** Error 3";
        let checks = IsabelleBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 3);
    }

    #[test]
    fn test_extract_failed_checks_error_prefix() {
        let output = "Error: Something went wrong";
        let checks = IsabelleBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert!(checks[0].description.contains("Something went wrong"));
    }

    #[test]
    fn test_extract_failed_checks_empty_output() {
        let checks = IsabelleBackend::extract_failed_checks("");
        assert!(checks.is_empty());
    }

    #[test]
    fn test_extract_failed_checks_no_errors() {
        let output = "Session built successfully\nAll theories verified";
        let checks = IsabelleBackend::extract_failed_checks(output);
        assert!(checks.is_empty());
    }

    #[test]
    fn test_parse_isabelle_location_line() {
        let line = "*** At command at line 42";
        let loc = IsabelleBackend::parse_isabelle_location(line).expect("Should parse location");

        assert_eq!(loc.line, 42);
    }

    #[test]
    fn test_parse_isabelle_location_position() {
        let line = "*** position 123 in theory";
        let loc = IsabelleBackend::parse_isabelle_location(line).expect("Should parse location");

        assert_eq!(loc.line, 123); // position used as line approximation
    }

    #[test]
    fn test_parse_isabelle_location_no_location() {
        assert!(IsabelleBackend::parse_isabelle_location("*** Proof failed").is_none());
        assert!(IsabelleBackend::parse_isabelle_location("").is_none());
    }

    #[test]
    fn test_structured_counterexample_has_raw() {
        let backend = IsabelleBackend::new();
        let output = "*** Proof failed";
        let ce = backend.parse_structured_counterexample(output);

        assert!(ce.raw.is_some());
        assert_eq!(ce.raw.unwrap(), output);
    }

    #[test]
    fn test_structured_counterexample_has_failed_checks() {
        let backend = IsabelleBackend::new();
        let output = "*** Proof failed\n*** Unfinished goals: 1";
        let ce = backend.parse_structured_counterexample(output);

        assert!(!ce.failed_checks.is_empty());
    }

    #[test]
    fn test_parse_output_includes_structured_counterexample() {
        let backend = IsabelleBackend::new();
        let output = IsabelleOutput {
            stdout: String::new(),
            stderr: "*** Proof failed".to_string(),
            exit_code: Some(1),
            duration: Duration::from_secs(1),
        };

        let result = backend.parse_output(&output);
        assert!(matches!(result.status, VerificationStatus::Disproven));
        assert!(result.counterexample.is_some());

        let ce = result.counterexample.unwrap();
        assert!(!ce.failed_checks.is_empty());
        assert!(ce
            .failed_checks
            .iter()
            .any(|c| c.check_id.contains("proof_failed")));
    }

    #[test]
    fn test_parse_output_unknown_with_counterexample() {
        let backend = IsabelleBackend::new();
        let output = IsabelleOutput {
            stdout: String::new(),
            stderr: "*** Unknown fact: foo".to_string(),
            exit_code: Some(1),
            duration: Duration::from_secs(1),
        };

        let result = backend.parse_output(&output);
        assert!(matches!(result.status, VerificationStatus::Unknown { .. }));
        // Should still have a counterexample with the error info
        assert!(result.counterexample.is_some());
    }

    #[test]
    fn test_extract_first_error() {
        let output = "Some preamble\n*** This is the error\nMore stuff";
        let error = IsabelleBackend::extract_first_error(output);
        assert!(error.contains("This is the error"));
    }

    #[test]
    fn test_extract_first_error_error_prefix() {
        let output = "Compiling...\nError: Something failed\nDone.";
        let error = IsabelleBackend::extract_first_error(output);
        assert!(error.contains("Error: Something failed"));
    }

    #[test]
    fn test_extract_first_error_unknown() {
        let output = "No error pattern here";
        let error = IsabelleBackend::extract_first_error(output);
        assert_eq!(error, "Unknown Isabelle error");
    }

    #[test]
    fn test_diagnostics_extraction() {
        let backend = IsabelleBackend::new();
        let output = "Warning: implicit\nWarning: another\n*** error\n*** another";
        let diagnostics = backend.extract_diagnostics(output);

        assert!(diagnostics.iter().any(|d| d.contains("2 warning")));
        assert!(diagnostics.iter().any(|d| d.contains("2 error")));
    }

    #[test]
    fn test_failed_to_finish_proof_detection() {
        let output = "*** Failed to finish proof\n*** remaining goals: 2";
        let checks = IsabelleBackend::extract_failed_checks(output);

        assert!(checks.iter().any(|c| c.check_id == "isabelle_proof_failed"));
    }
}
