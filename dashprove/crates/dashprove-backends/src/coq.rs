//! Coq backend implementation
//!
//! This backend executes Coq specifications using the coqc compiler.
//! Coq is an interactive proof assistant based on the Calculus of Constructions.

use crate::counterexample::{FailedCheck, SourceLocation, StructuredCounterexample};
use crate::traits::*;
use crate::util::expand_home_dir;
use async_trait::async_trait;
use dashprove_usl::{compile_to_coq, typecheck::TypedSpec};
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::process::Command;
use tracing::{debug, info, warn};

// =============================================
// Kani Proofs for Coq Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify CoqConfig::default timeout is 300 seconds
    #[kani::proof]
    fn proof_coq_config_default_timeout() {
        let config = CoqConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "Default timeout should be 300 seconds",
        );
    }

    /// Verify CoqConfig::default coqc_path is None
    #[kani::proof]
    fn proof_coq_config_default_path_none() {
        let config = CoqConfig::default();
        kani::assert(
            config.coqc_path.is_none(),
            "Default coqc_path should be None",
        );
    }

    /// Verify CoqConfig::default include_paths is empty
    #[kani::proof]
    fn proof_coq_config_default_include_paths_empty() {
        let config = CoqConfig::default();
        kani::assert(
            config.include_paths.is_empty(),
            "Default include_paths should be empty",
        );
    }

    /// Verify CoqBackend::new uses default config
    #[kani::proof]
    fn proof_coq_backend_new_defaults() {
        let backend = CoqBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "New backend should use default timeout",
        );
    }

    /// Verify CoqBackend::with_config preserves config
    #[kani::proof]
    fn proof_coq_backend_with_config() {
        let config = CoqConfig {
            coqc_path: Some(PathBuf::from("/custom/coqc")),
            timeout: Duration::from_secs(600),
            include_paths: vec![PathBuf::from("/lib/coq")],
        };
        let backend = CoqBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "Custom timeout should be preserved",
        );
        kani::assert(
            backend.config.include_paths.len() == 1,
            "Include paths should be preserved",
        );
    }

    /// Verify id() returns Coq
    #[kani::proof]
    fn proof_backend_id_is_coq() {
        let backend = CoqBackend::new();
        kani::assert(backend.id() == BackendId::Coq, "ID should be Coq");
    }

    /// Verify supports() includes Theorem
    #[kani::proof]
    fn proof_coq_supports_theorem() {
        let backend = CoqBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Theorem),
            "Should support Theorem",
        );
    }

    /// Verify supports() includes Invariant
    #[kani::proof]
    fn proof_coq_supports_invariant() {
        let backend = CoqBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Invariant),
            "Should support Invariant",
        );
    }

    /// Verify supports() includes Refinement
    #[kani::proof]
    fn proof_coq_supports_refinement() {
        let backend = CoqBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Refinement),
            "Should support Refinement",
        );
    }

    /// Verify parse_coq_location extracts file path
    #[kani::proof]
    fn proof_parse_coq_location_file() {
        let line = r#"File "./test.v", line 42, characters 5-15:"#;
        let loc = CoqBackend::parse_coq_location(line);
        kani::assert(loc.is_some(), "Should parse location");
        let loc = loc.unwrap();
        kani::assert(loc.file == "./test.v", "Should extract file path");
    }

    /// Verify parse_coq_location extracts line number
    #[kani::proof]
    fn proof_parse_coq_location_line() {
        let line = r#"File "./test.v", line 42, characters 5-15:"#;
        let loc = CoqBackend::parse_coq_location(line);
        kani::assert(loc.is_some(), "Should parse location");
        kani::assert(loc.unwrap().line == 42, "Should extract line number");
    }

    /// Verify parse_coq_location extracts column
    #[kani::proof]
    fn proof_parse_coq_location_column() {
        let line = r#"File "./test.v", line 42, characters 5-15:"#;
        let loc = CoqBackend::parse_coq_location(line);
        kani::assert(loc.is_some(), "Should parse location");
        kani::assert(loc.unwrap().column == Some(5), "Should extract column");
    }

    /// Verify parse_coq_location returns None for invalid input
    #[kani::proof]
    fn proof_parse_coq_location_invalid() {
        let line = "Error: syntax error";
        let loc = CoqBackend::parse_coq_location(line);
        kani::assert(loc.is_none(), "Should return None for invalid input");
    }

    /// Verify parse_coq_location handles missing characters
    #[kani::proof]
    fn proof_parse_coq_location_no_chars() {
        let line = r#"File "./test.v", line 10:"#;
        let loc = CoqBackend::parse_coq_location(line);
        kani::assert(loc.is_some(), "Should parse location without characters");
        let loc = loc.unwrap();
        kani::assert(loc.column.is_none(), "Column should be None");
    }

    /// Verify extract_first_error finds Error
    #[kani::proof]
    fn proof_extract_first_error_finds_error() {
        let output = "Some preamble\nError: This is the error\nMore stuff";
        let error = CoqBackend::extract_first_error(output);
        kani::assert(error.contains("Error"), "Should find error line");
    }

    /// Verify extract_first_error handles Unable to unify
    #[kani::proof]
    fn proof_extract_first_error_unification() {
        let output = "Compiling...\nUnable to unify X with Y";
        let error = CoqBackend::extract_first_error(output);
        kani::assert(
            error.contains("Unable to unify"),
            "Should find unification error",
        );
    }

    /// Verify extract_first_error returns default for no error
    #[kani::proof]
    fn proof_extract_first_error_default() {
        let output = "All good here";
        let error = CoqBackend::extract_first_error(output);
        kani::assert(
            error == "Unknown Coq error",
            "Should return default message",
        );
    }

    /// Verify extract_failed_checks returns empty for no errors
    #[kani::proof]
    fn proof_extract_failed_checks_empty() {
        let output = "All proofs verified";
        let checks = CoqBackend::extract_failed_checks(output);
        kani::assert(checks.is_empty(), "Should return empty for success");
    }

    /// Verify extract_failed_checks finds unification failure
    #[kani::proof]
    fn proof_extract_failed_checks_unification() {
        let output = r#"File "./Test.v", line 5, characters 0-10:
Error: Unable to unify "nat" with "bool"."#;
        let checks = CoqBackend::extract_failed_checks(output);
        kani::assert(!checks.is_empty(), "Should find failed checks");
        kani::assert(
            checks[0].check_id == "coq_unification_failure",
            "Should identify unification failure",
        );
    }

    /// Verify CoqDetection::NotFound preserves reason
    #[kani::proof]
    fn proof_coq_detection_not_found() {
        let reason = "coqc not found".to_string();
        let detection = CoqDetection::NotFound(reason.clone());
        if let CoqDetection::NotFound(r) = detection {
            kani::assert(r == reason, "NotFound should preserve reason");
        } else {
            kani::assert(false, "Should be NotFound variant");
        }
    }

    /// Verify CoqDetection::Available stores path
    #[kani::proof]
    fn proof_coq_detection_available() {
        let path = PathBuf::from("/bin/coqc");
        let detection = CoqDetection::Available {
            coqc_path: path.clone(),
        };
        if let CoqDetection::Available { coqc_path } = detection {
            kani::assert(coqc_path == path, "Should store coqc_path");
        } else {
            kani::assert(false, "Should be Available variant");
        }
    }
}

/// Configuration for Coq backend
#[derive(Debug, Clone)]
pub struct CoqConfig {
    /// Path to coqc executable (if not in PATH)
    pub coqc_path: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// Additional include paths for Coq libraries
    pub include_paths: Vec<PathBuf>,
}

impl Default for CoqConfig {
    fn default() -> Self {
        Self {
            coqc_path: None,
            timeout: Duration::from_secs(300), // 5 minutes
            include_paths: Vec::new(),
        }
    }
}

/// Coq verification backend
pub struct CoqBackend {
    config: CoqConfig,
}

#[derive(Debug, Clone)]
enum CoqDetection {
    /// Coq available
    Available { coqc_path: PathBuf },
    /// Coq not found
    NotFound(String),
}

impl CoqBackend {
    /// Create a new Coq backend with default configuration
    pub fn new() -> Self {
        Self {
            config: CoqConfig::default(),
        }
    }

    /// Create a new Coq backend with custom configuration
    pub fn with_config(config: CoqConfig) -> Self {
        Self { config }
    }

    /// Detect coqc installation
    async fn detect_coq(&self) -> CoqDetection {
        let coqc_path = if let Some(ref path) = self.config.coqc_path {
            if path.exists() {
                path.clone()
            } else {
                return CoqDetection::NotFound(format!(
                    "Configured coqc path does not exist: {:?}",
                    path
                ));
            }
        } else {
            // Check for coqc in PATH
            if let Ok(path) = which::which("coqc") {
                path
            } else {
                // Check common installation locations
                let common_paths = [
                    expand_home_dir("~/.opam/default/bin/coqc"),
                    Some(PathBuf::from("/usr/local/bin/coqc")),
                    Some(PathBuf::from("/usr/bin/coqc")),
                    Some(PathBuf::from("/opt/homebrew/bin/coqc")),
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
                        return CoqDetection::NotFound(
                            "coqc not found. Install Coq via opam or package manager: https://coq.inria.fr/download".to_string(),
                        );
                    }
                }
            }
        };

        // Verify coqc works
        let version_check = Command::new(&coqc_path)
            .arg("--version")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await;

        match version_check {
            Ok(output) if output.status.success() => {
                let version = String::from_utf8_lossy(&output.stdout);
                debug!("Found Coq version: {}", version.trim());
                CoqDetection::Available { coqc_path }
            }
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                CoqDetection::NotFound(format!("coqc --version failed: {}", stderr))
            }
            Err(e) => CoqDetection::NotFound(format!("Failed to execute coqc: {}", e)),
        }
    }

    /// Write Coq project files to temp directory
    async fn write_project(
        &self,
        spec: &TypedSpec,
        dir: &std::path::Path,
    ) -> Result<PathBuf, BackendError> {
        let compiled = compile_to_coq(spec);
        let coq_code = compiled.code;
        let coq_path = dir.join("USLSpec.v");

        tokio::fs::write(&coq_path, &coq_code).await.map_err(|e| {
            BackendError::CompilationFailed(format!("Failed to write Coq file: {}", e))
        })?;

        debug!("Written Coq spec to {:?}", coq_path);

        Ok(coq_path)
    }

    /// Execute coqc and capture output
    async fn run_coqc(
        &self,
        detection: &CoqDetection,
        coq_file: &std::path::Path,
    ) -> Result<CoqOutput, BackendError> {
        let start = Instant::now();

        let coqc_path = match detection {
            CoqDetection::Available { coqc_path } => coqc_path,
            CoqDetection::NotFound(reason) => {
                return Err(BackendError::Unavailable(reason.clone()));
            }
        };

        let mut cmd = Command::new(coqc_path);
        cmd.arg(coq_file);

        // Add include paths
        for path in &self.config.include_paths {
            cmd.arg("-I");
            cmd.arg(path);
        }

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let output = tokio::time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to execute coqc: {}", e))
            })?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let duration = start.elapsed();

        debug!("coqc stdout:\n{}", stdout);
        if !stderr.is_empty() {
            debug!("coqc stderr:\n{}", stderr);
        }

        Ok(CoqOutput {
            stdout,
            stderr,
            exit_code: output.status.code(),
            duration,
        })
    }

    /// Parse coqc output into verification result
    fn parse_output(&self, output: &CoqOutput) -> BackendResult {
        let combined = format!("{}\n{}", output.stdout, output.stderr);

        // Check for successful compilation (all proofs verified)
        if output.exit_code == Some(0) {
            return BackendResult {
                backend: BackendId::Coq,
                status: VerificationStatus::Proven,
                proof: Some("All proofs verified by Coq".to_string()),
                counterexample: None,
                diagnostics: self.extract_diagnostics(&combined),
                time_taken: output.duration,
            };
        }

        // Check for proof failures
        if combined.contains("Error:")
            || combined.contains("Unable to unify")
            || combined.contains("Proof completed") && combined.contains("Error")
        {
            let error_msg = self.extract_error(&combined);

            // Specific proof failure patterns
            if combined.contains("Unable to unify")
                || combined.contains("No such goal")
                || combined.contains("Proof completed") && combined.contains("declared")
            {
                return BackendResult {
                    backend: BackendId::Coq,
                    status: VerificationStatus::Disproven,
                    proof: None,
                    counterexample: Some(self.parse_structured_counterexample(&combined)),
                    diagnostics: self.extract_diagnostics(&combined),
                    time_taken: output.duration,
                };
            }

            return BackendResult {
                backend: BackendId::Coq,
                status: VerificationStatus::Unknown {
                    reason: format!("Coq error: {}", error_msg),
                },
                proof: None,
                counterexample: Some(self.parse_structured_counterexample(&combined)),
                diagnostics: self.extract_diagnostics(&combined),
                time_taken: output.duration,
            };
        }

        BackendResult {
            backend: BackendId::Coq,
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
        let mut in_error = false;

        for line in output.lines() {
            if line.contains("Error:") {
                in_error = true;
            }
            if in_error {
                errors.push(line.to_string());
                if errors.len() >= 10 {
                    break;
                }
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

        let error_count = output.matches("Error:").count();
        if error_count > 0 {
            diagnostics.push(format!("{} error(s)", error_count));
        }

        diagnostics
    }

    /// Parse Coq output into a structured counterexample with detailed failure info
    fn parse_structured_counterexample(&self, output: &str) -> StructuredCounterexample {
        let mut ce = StructuredCounterexample::new();
        ce.raw = Some(output.to_string());

        // Extract failed checks from error output
        let failed_checks = Self::extract_failed_checks(output);
        ce.failed_checks = failed_checks;

        ce
    }

    /// Extract failed checks from Coq error output
    fn extract_failed_checks(output: &str) -> Vec<FailedCheck> {
        let mut checks = Vec::new();
        let lines: Vec<&str> = output.lines().collect();

        let mut i = 0;
        while i < lines.len() {
            let line = lines[i];

            // Pattern: File "path", line N, characters X-Y:
            if let Some(location) = Self::parse_coq_location(line) {
                // Look for error or proof failure on next lines
                let mut description = String::new();
                let mut check_type = "error".to_string();

                for j in (i + 1)..lines.len().min(i + 10) {
                    let next_line = lines[j].trim();

                    // Check for "Error: Unable to unify" or just "Unable to unify"
                    if next_line.contains("Unable to unify") {
                        check_type = "unification_failure".to_string();
                        description = next_line
                            .strip_prefix("Error:")
                            .map(|s| s.trim())
                            .unwrap_or(next_line)
                            .to_string();
                        // Capture following lines for context
                        for context_line in lines.iter().take(lines.len().min(j + 5)).skip(j + 1) {
                            let trimmed = context_line.trim();
                            if trimmed.is_empty()
                                || trimmed.starts_with("File ")
                                || trimmed.starts_with("Error:")
                            {
                                break;
                            }
                            description.push('\n');
                            description.push_str(trimmed);
                        }
                        break;
                    }
                    // Check for "Error: No such goal" or just "No such goal"
                    if next_line.contains("No such goal") {
                        check_type = "no_such_goal".to_string();
                        description = next_line
                            .strip_prefix("Error:")
                            .map(|s| s.trim())
                            .unwrap_or(next_line)
                            .to_string();
                        break;
                    }
                    if next_line.contains("Proof completed") || next_line.contains("Proof term") {
                        check_type = "proof_incomplete".to_string();
                        description = next_line.to_string();
                        break;
                    }
                    // Generic error without specific type
                    if next_line.starts_with("Error:") {
                        description = next_line
                            .strip_prefix("Error:")
                            .unwrap_or("")
                            .trim()
                            .to_string();
                        break;
                    }
                }

                if !description.is_empty() {
                    checks.push(FailedCheck {
                        check_id: format!("coq_{}", check_type),
                        description,
                        location: Some(location),
                        function: None,
                    });
                }
            }

            // Direct Error: patterns without file location
            if line.trim().starts_with("Error:")
                && !checks
                    .iter()
                    .any(|c| line.contains(c.description.lines().next().unwrap_or("")))
            {
                let description = line
                    .trim()
                    .strip_prefix("Error:")
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if !description.is_empty() {
                    checks.push(FailedCheck {
                        check_id: "coq_error".to_string(),
                        description,
                        location: None,
                        function: None,
                    });
                }
            }

            // "*** " error markers (Coq compilation errors)
            if line.contains("*** ") {
                let description = line
                    .trim()
                    .strip_prefix("***")
                    .unwrap_or(line.trim())
                    .trim()
                    .to_string();
                if !description.is_empty() {
                    checks.push(FailedCheck {
                        check_id: "coq_compilation_error".to_string(),
                        description,
                        location: None,
                        function: None,
                    });
                }
            }

            i += 1;
        }

        // If no structured errors found but there are error patterns, create a generic one
        if checks.is_empty() && (output.contains("Error:") || output.contains("Unable to unify")) {
            checks.push(FailedCheck {
                check_id: "coq_error".to_string(),
                description: Self::extract_first_error(output),
                location: None,
                function: None,
            });
        }

        checks
    }

    /// Parse a Coq file location string
    /// Format: File "path", line N, characters X-Y:
    fn parse_coq_location(line: &str) -> Option<SourceLocation> {
        // Pattern: File "path", line N, characters X-Y:
        if !line.contains("File ") || !line.contains("line ") {
            return None;
        }

        // Extract file path
        let file = line.split("File \"").nth(1)?.split('"').next()?.to_string();

        // Extract line number
        let line_part = line.split("line ").nth(1)?;
        let line_num: u32 = line_part
            .split(|c: char| !c.is_ascii_digit())
            .next()?
            .parse()
            .ok()?;

        // Extract character range if present
        let start_col = if line.contains("characters ") {
            let chars_part = line.split("characters ").nth(1)?;
            let range_str = chars_part.trim_end_matches(':');
            let parts: Vec<&str> = range_str.split('-').collect();
            if !parts.is_empty() {
                parts[0].parse().ok()
            } else {
                None
            }
        } else {
            None
        };

        Some(SourceLocation {
            file,
            line: line_num,
            column: start_col,
        })
    }

    /// Extract the first error message from output
    fn extract_first_error(output: &str) -> String {
        for line in output.lines() {
            if line.contains("Error:") {
                return line.trim().to_string();
            }
            if line.contains("Unable to unify") {
                return line.trim().to_string();
            }
        }
        "Unknown Coq error".to_string()
    }
}

impl Default for CoqBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for CoqBackend {
    fn id(&self) -> BackendId {
        BackendId::Coq
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::Theorem,
            PropertyType::Invariant,
            PropertyType::Refinement,
        ]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let detection = self.detect_coq().await;

        if let CoqDetection::NotFound(reason) = &detection {
            return Err(BackendError::Unavailable(reason.clone()));
        }

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::CompilationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let coq_path = self.write_project(spec, temp_dir.path()).await?;
        let output = self.run_coqc(&detection, &coq_path).await?;
        let result = self.parse_output(&output);

        Ok(result)
    }

    async fn health_check(&self) -> HealthStatus {
        let detection = self.detect_coq().await;
        match detection {
            CoqDetection::Available { coqc_path } => {
                info!("Coq available: {:?}", coqc_path);
                HealthStatus::Healthy
            }
            CoqDetection::NotFound(reason) => {
                warn!("Coq not available: {}", reason);
                HealthStatus::Unavailable { reason }
            }
        }
    }
}

/// Coq execution output
struct CoqOutput {
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
        let config = CoqConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert!(config.include_paths.is_empty());
    }

    #[test]
    fn test_compile_generates_coq() {
        let input = r#"
            theorem test {
                forall x: Bool . x or not x
            }
        "#;
        let spec = make_typed_spec(input);
        let compiled = compile_to_coq(&spec);
        let coq = compiled.code;

        assert!(coq.contains("(* Generated by DashProve"));
        assert!(coq.contains("Theorem test"));
        // Coq proof structure uses Admitted for incomplete proofs
        assert!(coq.contains("Proof.") || coq.contains("Admitted"));
    }

    #[test]
    fn test_parse_success_output() {
        let backend = CoqBackend::new();
        let output = CoqOutput {
            stdout: String::new(),
            stderr: String::new(),
            exit_code: Some(0),
            duration: Duration::from_secs(2),
        };

        let result = backend.parse_output(&output);
        assert!(matches!(result.status, VerificationStatus::Proven));
    }

    #[test]
    fn test_parse_proof_failure() {
        let backend = CoqBackend::new();
        let output = CoqOutput {
            stdout: String::new(),
            stderr: "Error: Unable to unify".to_string(),
            exit_code: Some(1),
            duration: Duration::from_secs(1),
        };

        let result = backend.parse_output(&output);
        assert!(matches!(result.status, VerificationStatus::Disproven));
    }

    #[tokio::test]
    async fn test_health_check_reports_status() {
        let backend = CoqBackend::new();
        let status = backend.health_check().await;

        match status {
            HealthStatus::Healthy => println!("Coq is available"),
            HealthStatus::Unavailable { reason } => println!("Coq not available: {}", reason),
            HealthStatus::Degraded { reason } => println!("Coq degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_verify_returns_result_or_unavailable() {
        let backend = CoqBackend::new();
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
    fn test_parse_coq_location_basic() {
        let line = r#"File "./test.v", line 42, characters 5-15:"#;
        let loc = CoqBackend::parse_coq_location(line).expect("Should parse location");

        assert_eq!(loc.file, "./test.v".to_string());
        assert_eq!(loc.line, 42);
        assert_eq!(loc.column, Some(5));
    }

    #[test]
    fn test_parse_coq_location_no_characters() {
        let line = r#"File "/path/to/file.v", line 10:"#;
        let loc = CoqBackend::parse_coq_location(line).expect("Should parse location");

        assert_eq!(loc.file, "/path/to/file.v".to_string());
        assert_eq!(loc.line, 10);
        assert!(loc.column.is_none());
    }

    #[test]
    fn test_parse_coq_location_complex_path() {
        let line = r#"File "/Users/test/my project/src/Spec.v", line 123, characters 0-50:"#;
        let loc = CoqBackend::parse_coq_location(line).expect("Should parse location");

        assert_eq!(loc.file, "/Users/test/my project/src/Spec.v".to_string());
        assert_eq!(loc.line, 123);
        assert_eq!(loc.column, Some(0));
    }

    #[test]
    fn test_parse_coq_location_invalid() {
        assert!(CoqBackend::parse_coq_location("Error: syntax error").is_none());
        assert!(CoqBackend::parse_coq_location("").is_none());
        assert!(CoqBackend::parse_coq_location("just some text").is_none());
    }

    #[test]
    fn test_extract_failed_checks_unification() {
        let output = r#"File "./USLSpec.v", line 15, characters 0-42:
Error: Unable to unify "true" with "false".
"#;
        let checks = CoqBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "coq_unification_failure");
        assert!(checks[0].description.contains("Unable to unify"));
        assert!(checks[0].location.is_some());
        assert_eq!(checks[0].location.as_ref().unwrap().line, 15);
    }

    #[test]
    fn test_extract_failed_checks_no_such_goal() {
        let output = r#"File "./Proof.v", line 30, characters 2-10:
Error: No such goal. Focus next goal with bullet -.
"#;
        let checks = CoqBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "coq_no_such_goal");
        assert!(checks[0].description.contains("No such goal"));
    }

    #[test]
    fn test_extract_failed_checks_simple_error() {
        let output = "Error: Syntax error: '.' expected after [vernac:gallina_ext].";
        let checks = CoqBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "coq_error");
        assert!(checks[0].description.contains("Syntax error"));
    }

    #[test]
    fn test_extract_failed_checks_multiple_errors() {
        let output = r#"File "./Test.v", line 5, characters 0-10:
Error: Unable to unify "nat" with "bool".
File "./Test.v", line 10, characters 0-15:
Error: Unable to unify "list nat" with "nat".
"#;
        let checks = CoqBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 2);
        assert!(checks
            .iter()
            .all(|c| c.check_id == "coq_unification_failure"));
    }

    #[test]
    fn test_extract_failed_checks_multiline_context() {
        let output = r#"File "./Spec.v", line 20, characters 5-25:
Unable to unify
  "forall n : nat, n = n"
with
  "True".
"#;
        let checks = CoqBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        let desc = &checks[0].description;
        assert!(desc.contains("Unable to unify"));
        assert!(desc.contains("forall n : nat"));
    }

    #[test]
    fn test_extract_failed_checks_empty_output() {
        let checks = CoqBackend::extract_failed_checks("");
        assert!(checks.is_empty());
    }

    #[test]
    fn test_extract_failed_checks_no_errors() {
        let output = "Compiled successfully\nAll proofs verified";
        let checks = CoqBackend::extract_failed_checks(output);
        assert!(checks.is_empty());
    }

    #[test]
    fn test_structured_counterexample_has_raw() {
        let backend = CoqBackend::new();
        let output = "Error: Unable to unify";
        let ce = backend.parse_structured_counterexample(output);

        assert!(ce.raw.is_some());
        assert_eq!(ce.raw.unwrap(), output);
    }

    #[test]
    fn test_structured_counterexample_has_failed_checks() {
        let backend = CoqBackend::new();
        let output = r#"File "./Test.v", line 10, characters 0-20:
Unable to unify "A" with "B"."#;
        let ce = backend.parse_structured_counterexample(output);

        assert!(!ce.failed_checks.is_empty());
        assert_eq!(ce.failed_checks[0].check_id, "coq_unification_failure");
    }

    #[test]
    fn test_parse_output_includes_structured_counterexample() {
        let backend = CoqBackend::new();
        let output = CoqOutput {
            stdout: String::new(),
            stderr: r#"File "./Test.v", line 5, characters 0-10:
Unable to unify "nat" with "bool"."#
                .to_string(),
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
            .any(|c| c.check_id.contains("unification")));
    }

    #[test]
    fn test_extract_first_error() {
        let output = "Some preamble\nError: This is the error\nMore stuff";
        let error = CoqBackend::extract_first_error(output);
        assert!(error.contains("This is the error"));
    }

    #[test]
    fn test_extract_first_error_unification() {
        let output = "Compiling...\nUnable to unify X with Y\nDone.";
        let error = CoqBackend::extract_first_error(output);
        assert!(error.contains("Unable to unify"));
    }

    #[test]
    fn test_extract_first_error_unknown() {
        let output = "No error pattern here";
        let error = CoqBackend::extract_first_error(output);
        assert_eq!(error, "Unknown Coq error");
    }

    #[test]
    fn test_parse_output_unknown_with_counterexample() {
        let backend = CoqBackend::new();
        let output = CoqOutput {
            stdout: String::new(),
            stderr: "Error: Unknown identifier foo.".to_string(),
            exit_code: Some(1),
            duration: Duration::from_secs(1),
        };

        let result = backend.parse_output(&output);
        assert!(matches!(result.status, VerificationStatus::Unknown { .. }));
        // Should still have a counterexample with the error info
        assert!(result.counterexample.is_some());
    }

    #[test]
    fn test_diagnostics_extraction() {
        let backend = CoqBackend::new();
        let output = "Warning: implicit\nWarning: another\nError: failed\nError: again";
        let diagnostics = backend.extract_diagnostics(output);

        assert!(diagnostics.iter().any(|d| d.contains("2 warning")));
        assert!(diagnostics.iter().any(|d| d.contains("2 error")));
    }

    #[test]
    fn test_proof_incomplete_detection() {
        let output = r#"File "./Test.v", line 15, characters 0-20:
Proof completed but type assertion failed.
"#;
        let checks = CoqBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "coq_proof_incomplete");
    }

    #[test]
    fn test_compilation_error_marker() {
        let output = "*** Error: Unbound reference";
        let checks = CoqBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "coq_compilation_error");
        assert!(checks[0].description.contains("Error: Unbound reference"));
    }
}
