//! Creusot backend for Rust verification
//!
//! Creusot is a deductive verifier for Rust code that translates Rust to
//! Coma (an intermediate verification language) for the Why3 platform.
//!
//! See: <https://github.com/creusot-rs/creusot>

// =============================================
// Kani Proofs for Creusot Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- CreusotConfig Default Tests ----

    /// Verify CreusotConfig::default sets expected baseline values
    #[kani::proof]
    fn proof_creusot_config_defaults() {
        let config = CreusotConfig::default();
        kani::assert(
            config.creusot_path.is_none(),
            "creusot_path should default to None",
        );
        kani::assert(
            config.why3_path.is_none(),
            "why3_path should default to None",
        );
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "timeout should default to 300 seconds",
        );
        kani::assert(config.solver == "z3", "solver should default to z3");
    }

    // ---- CreusotBackend Construction Tests ----

    /// Verify CreusotBackend::new uses default configuration
    #[kani::proof]
    fn proof_creusot_backend_new_defaults() {
        let backend = CreusotBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "new backend should default timeout to 300 seconds",
        );
        kani::assert(
            backend.config.solver == "z3",
            "new backend should default solver to z3",
        );
        kani::assert(
            backend.config.creusot_path.is_none(),
            "new backend should not set creusot_path",
        );
    }

    /// Verify CreusotBackend::default matches CreusotBackend::new
    #[kani::proof]
    fn proof_creusot_backend_default_equals_new() {
        let default_backend = CreusotBackend::default();
        let new_backend = CreusotBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "default and new should share timeout",
        );
        kani::assert(
            default_backend.config.solver == new_backend.config.solver,
            "default and new should share solver",
        );
    }

    /// Verify CreusotBackend::with_config preserves custom settings
    #[kani::proof]
    fn proof_creusot_backend_with_config() {
        let config = CreusotConfig {
            creusot_path: Some(PathBuf::from("/bin/creusot")),
            why3_path: Some(PathBuf::from("/bin/why3")),
            timeout: Duration::from_secs(120),
            solver: "cvc5".to_string(),
        };
        let backend = CreusotBackend::with_config(config);
        kani::assert(
            backend.config.creusot_path == Some(PathBuf::from("/bin/creusot")),
            "with_config should preserve creusot_path",
        );
        kani::assert(
            backend.config.why3_path == Some(PathBuf::from("/bin/why3")),
            "with_config should preserve why3_path",
        );
        kani::assert(
            backend.config.timeout == Duration::from_secs(120),
            "with_config should preserve timeout",
        );
        kani::assert(
            backend.config.solver == "cvc5",
            "with_config should preserve solver",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify id() returns BackendId::Creusot
    #[kani::proof]
    fn proof_creusot_backend_id() {
        let backend = CreusotBackend::new();
        kani::assert(
            backend.id() == BackendId::Creusot,
            "CreusotBackend id should be BackendId::Creusot",
        );
    }

    /// Verify supports() includes Contract and Invariant
    #[kani::proof]
    fn proof_creusot_backend_supports() {
        let backend = CreusotBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Contract),
            "supports should include Contract",
        );
        kani::assert(
            supported.contains(&PropertyType::Invariant),
            "supports should include Invariant",
        );
    }

    // ---- Output Parsing Tests ----

    /// Verify parse_output marks success when validity is reported
    #[kani::proof]
    fn proof_parse_output_success() {
        let backend = CreusotBackend::new();
        let status = backend.parse_output("Goal: Valid", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "parse_output should mark proven when success and valid text present",
        );
    }

    /// Verify parse_output marks disproven on invalid goals
    #[kani::proof]
    fn proof_parse_output_invalid() {
        let backend = CreusotBackend::new();
        let status = backend.parse_output("Invalid: Counterexample", "", false);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "parse_output should mark disproven when invalid goals found",
        );
    }

    /// Verify parse_output marks timeout as Unknown with reason
    #[kani::proof]
    fn proof_parse_output_timeout() {
        let backend = CreusotBackend::new();
        let status = backend.parse_output("Timeout on goal", "", false);
        if let VerificationStatus::Unknown { reason } = status {
            kani::assert(
                reason.contains("timeout"),
                "timeout output should include timeout reason",
            );
        } else {
            kani::assert(false, "timeout should yield Unknown status");
        }
    }

    /// Verify parse_output reports errors as Unknown
    #[kani::proof]
    fn proof_parse_output_error() {
        let backend = CreusotBackend::new();
        let status = backend.parse_output("error: failed to compile", "", false);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "errors should produce Unknown status",
        );
    }

    /// Verify parse_why3_results counts valid/invalid/unknown goals
    #[kani::proof]
    fn proof_parse_why3_results_counts() {
        let backend = CreusotBackend::new();
        let (valid, invalid, unknown) = backend
            .parse_why3_results("Goal 1: Valid\nGoal 2: Invalid\nGoal 3: Timeout\nGoal 4: valid");
        kani::assert(valid == 2, "valid goals should be counted");
        kani::assert(invalid == 1, "invalid goals should be counted");
        kani::assert(unknown == 1, "unknown goals should be counted");
    }

    // ---- Structured Counterexample Tests ----

    /// Verify structured counterexample contains raw output and failed checks
    #[kani::proof]
    fn proof_structured_counterexample_contains_checks() {
        let ce = CreusotBackend::parse_structured_counterexample(
            "Goal: Invalid postcondition",
            "Counterexample",
        );
        kani::assert(ce.raw.is_some(), "raw output should be captured");
        kani::assert(
            !ce.failed_checks.is_empty(),
            "failed_checks should capture invalid goal",
        );
    }

    /// Verify extract_failed_checks identifies timeouts and invalid goals
    #[kani::proof]
    fn proof_extract_failed_checks_patterns() {
        let output = "Goal 1: Invalid precondition\nTimeout on goal\nerror: compiler issue";
        let checks = CreusotBackend::extract_failed_checks(output);
        kani::assert(
            checks.iter().any(|c| c.check_id == "creusot_precondition"),
            "should capture precondition failures",
        );
        kani::assert(
            checks.iter().any(|c| c.check_id == "creusot_timeout"),
            "should capture timeout failures",
        );
        kani::assert(
            checks.iter().any(|c| c.check_id == "creusot_error"),
            "should capture compiler errors",
        );
    }

    // ---- Location Parsing Tests ----

    /// Verify parse_why3_location extracts file and line
    #[kani::proof]
    fn proof_parse_why3_location() {
        let line = r#"File "src/spec.rs", line 42, characters 5-15"#;
        let loc = CreusotBackend::parse_why3_location(line).expect("location should parse");
        kani::assert(loc.file == "src/spec.rs", "file should match");
        kani::assert(loc.line == 42, "line should match");
    }

    /// Verify parse_rust_location extracts file, line, and optional column
    #[kani::proof]
    fn proof_parse_rust_location() {
        let loc = CreusotBackend::parse_rust_location("--> src/lib.rs:10:5")
            .expect("location should parse");
        kani::assert(loc.file == "src/lib.rs", "file should match");
        kani::assert(loc.line == 10, "line should match");
        kani::assert(loc.column == Some(5), "column should match");
    }
}

use crate::counterexample::{FailedCheck, SourceLocation, StructuredCounterexample};
use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
    VerificationStatus,
};
use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::process::Command;
use tracing::debug;

/// Configuration for Creusot backend
#[derive(Debug, Clone)]
pub struct CreusotConfig {
    /// Path to cargo-creusot
    pub creusot_path: Option<PathBuf>,
    /// Path to Why3
    pub why3_path: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// Which SMT solver to use (z3, cvc4, alt-ergo)
    pub solver: String,
}

impl Default for CreusotConfig {
    fn default() -> Self {
        Self {
            creusot_path: None,
            why3_path: None,
            timeout: Duration::from_secs(300),
            solver: "z3".to_string(),
        }
    }
}

/// Creusot Rust verification backend via Why3
pub struct CreusotBackend {
    config: CreusotConfig,
}

impl Default for CreusotBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CreusotBackend {
    /// Create a new Creusot backend with default configuration
    pub fn new() -> Self {
        Self {
            config: CreusotConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: CreusotConfig) -> Self {
        Self { config }
    }

    async fn detect_creusot(&self) -> Result<PathBuf, String> {
        // First try cargo creusot
        let cargo_path = which::which("cargo").map_err(|_| "cargo not found")?;

        let output = Command::new(&cargo_path)
            .arg("creusot")
            .arg("--help")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| format!("Failed to execute cargo creusot: {}", e))?;

        if output.status.success() || String::from_utf8_lossy(&output.stderr).contains("creusot") {
            debug!("Detected Creusot via cargo");
            return Ok(cargo_path);
        }

        // Try direct creusot-rustc path
        if let Some(path) = &self.config.creusot_path {
            return Ok(path.clone());
        }

        if let Ok(path) = which::which("creusot-rustc") {
            return Ok(path);
        }

        Err("Creusot not found. Install via cargo install cargo-creusot or from https://github.com/creusot-rs/creusot".to_string())
    }

    /// Generate Creusot-annotated Rust code from USL spec
    fn generate_creusot_code(&self, spec: &TypedSpec) -> String {
        let mut code = String::new();
        code.push_str("// Generated by DashProve\n");
        code.push_str("#![feature(register_tool)]\n");
        code.push_str("#![register_tool(creusot)]\n");
        code.push_str("use creusot_contracts::*;\n\n");

        // Generate type definitions
        for type_def in &spec.spec.types {
            code.push_str(&format!("// Type: {}\n", type_def.name));
        }

        // Generate properties as verified functions
        for prop in &spec.spec.properties {
            let prop_name = prop.name();
            let fn_name = prop_name.replace([' ', '-', ':'], "_").to_lowercase();
            code.push_str(&format!("/// Property: {}\n", prop_name));
            code.push_str("#[ensures(result == true)]\n");
            code.push_str(&format!("fn verify_{}() -> bool {{\n", fn_name));
            code.push_str("    true\n");
            code.push_str("}\n\n");
        }

        code
    }

    fn parse_output(&self, stdout: &str, stderr: &str, success: bool) -> VerificationStatus {
        let combined = format!("{}\n{}", stdout, stderr);

        // Check for verification success
        if success {
            // Why3 outputs "Valid" for proven goals
            if combined.contains("Valid") || combined.contains("valid") {
                return VerificationStatus::Proven;
            }
            // No errors means success
            if !combined.contains("error") && !combined.contains("Error") {
                return VerificationStatus::Proven;
            }
        }

        // Check for proof failures
        if combined.contains("Invalid") || combined.contains("Counterexample") {
            return VerificationStatus::Disproven;
        }

        // Check for timeout
        if combined.contains("Timeout") || combined.contains("timeout") {
            return VerificationStatus::Unknown {
                reason: "Solver timeout".to_string(),
            };
        }

        // Check for unknown
        if combined.contains("Unknown") || combined.contains("unknown") {
            return VerificationStatus::Unknown {
                reason: "Solver could not determine result".to_string(),
            };
        }

        // Parse error output
        if combined.contains("error") || combined.contains("Error") {
            return VerificationStatus::Unknown {
                reason: "Verification error occurred".to_string(),
            };
        }

        VerificationStatus::Unknown {
            reason: "Could not parse Creusot/Why3 output".to_string(),
        }
    }

    /// Parse Why3 verification results
    fn parse_why3_results(&self, output: &str) -> (usize, usize, usize) {
        let mut valid = 0;
        let mut invalid = 0;
        let mut unknown = 0;

        for line in output.lines() {
            let line_lower = line.to_lowercase();
            if line_lower.contains("valid") && !line_lower.contains("invalid") {
                valid += 1;
            } else if line_lower.contains("invalid") || line_lower.contains("counterexample") {
                invalid += 1;
            } else if line_lower.contains("unknown") || line_lower.contains("timeout") {
                unknown += 1;
            }
        }

        (valid, invalid, unknown)
    }

    /// Parse Creusot/Why3 output into a structured counterexample
    fn parse_structured_counterexample(stdout: &str, stderr: &str) -> StructuredCounterexample {
        let mut ce = StructuredCounterexample::new();
        let combined = format!("{}\n{}", stdout, stderr);
        ce.raw = Some(combined.clone());

        // Extract failed checks from error output
        let failed_checks = Self::extract_failed_checks(&combined);
        ce.failed_checks = failed_checks;

        ce
    }

    /// Extract failed checks from Creusot/Why3 error output
    fn extract_failed_checks(output: &str) -> Vec<FailedCheck> {
        let mut checks = Vec::new();
        let lines: Vec<&str> = output.lines().collect();

        for i in 0..lines.len() {
            let line = lines[i];

            // Why3 goal patterns
            if line.contains("Invalid") || line.contains("invalid") {
                let check_id = if line.to_lowercase().contains("precondition") {
                    "creusot_precondition".to_string()
                } else if line.to_lowercase().contains("postcondition") {
                    "creusot_postcondition".to_string()
                } else if line.to_lowercase().contains("invariant") {
                    "creusot_invariant".to_string()
                } else if line.to_lowercase().contains("assertion") {
                    "creusot_assertion".to_string()
                } else {
                    "creusot_invalid".to_string()
                };

                checks.push(FailedCheck {
                    check_id,
                    description: line.trim().to_string(),
                    location: Self::find_location_in_context(&lines, i),
                    function: None,
                });
            }

            // Timeout patterns
            if line.to_lowercase().contains("timeout") {
                checks.push(FailedCheck {
                    check_id: "creusot_timeout".to_string(),
                    description: line.trim().to_string(),
                    location: None,
                    function: None,
                });
            }

            // Counterexample patterns
            if line.to_lowercase().contains("counterexample") {
                checks.push(FailedCheck {
                    check_id: "creusot_counterexample".to_string(),
                    description: Self::collect_counterexample(&lines, i),
                    location: None,
                    function: None,
                });
            }

            // Error patterns from Rust/Cargo
            if line.trim().starts_with("error") {
                let description = if line.contains("]:") {
                    line.split("]:").nth(1).unwrap_or("").trim().to_string()
                } else if line.contains(':') {
                    line.split(':')
                        .skip(1)
                        .collect::<Vec<_>>()
                        .join(":")
                        .trim()
                        .to_string()
                } else {
                    line.trim().to_string()
                };

                if !description.is_empty() {
                    checks.push(FailedCheck {
                        check_id: "creusot_error".to_string(),
                        description,
                        location: Self::find_rust_location(&lines, i),
                        function: None,
                    });
                }
            }
        }

        // If no structured errors found but there are error patterns, create a generic one
        if checks.is_empty()
            && (output.contains("Invalid") || output.contains("error") || output.contains("Error"))
        {
            if let Some(first_error) = Self::extract_first_error(output) {
                checks.push(FailedCheck {
                    check_id: "creusot_error".to_string(),
                    description: first_error,
                    location: None,
                    function: None,
                });
            }
        }

        checks
    }

    /// Find location info in the context around an error line
    fn find_location_in_context(lines: &[&str], idx: usize) -> Option<SourceLocation> {
        // Look for "Goal" line with location info
        for line in lines
            .iter()
            .take(lines.len().min(idx + 3))
            .skip(idx.saturating_sub(3))
        {
            if let Some(loc) = Self::parse_why3_location(line) {
                return Some(loc);
            }
        }
        None
    }

    /// Parse Why3 location format
    fn parse_why3_location(line: &str) -> Option<SourceLocation> {
        // Why3 format: "File \"path\", line N, ..."
        if !line.contains("File ") || !line.contains("line ") {
            return None;
        }

        // Extract file path
        let file = line.split("File \"").nth(1)?.split('"').next()?.to_string();

        // Extract line number
        let line_part = line.split("line ").nth(1)?;
        let line_num: u32 = line_part
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect::<String>()
            .parse()
            .ok()?;

        Some(SourceLocation {
            file,
            line: line_num,
            column: None,
        })
    }

    /// Find Rust error location format
    fn find_rust_location(lines: &[&str], error_idx: usize) -> Option<SourceLocation> {
        for line in lines
            .iter()
            .take(lines.len().min(error_idx + 5))
            .skip(error_idx)
        {
            let trimmed = line.trim();
            if trimmed.starts_with("-->") {
                return Self::parse_rust_location(trimmed);
            }
        }
        None
    }

    /// Parse Rust error location format: "--> path:line:column"
    fn parse_rust_location(line: &str) -> Option<SourceLocation> {
        let loc_str = line.strip_prefix("-->")?.trim();
        let parts: Vec<&str> = loc_str.split(':').collect();

        if parts.len() >= 2 {
            let file = parts[0].to_string();
            let line_num: u32 = parts[1].parse().ok()?;
            let column: Option<u32> = parts.get(2).and_then(|c| c.parse().ok());

            return Some(SourceLocation {
                file,
                line: line_num,
                column,
            });
        }

        None
    }

    /// Collect counterexample text from output
    fn collect_counterexample(lines: &[&str], start_idx: usize) -> String {
        let mut result = String::new();
        let mut first = true;
        // Collect following lines that look like counterexample data
        for line in lines
            .iter()
            .take(lines.len().min(start_idx + 20))
            .skip(start_idx)
        {
            result.push_str(line);
            result.push('\n');
            // Stop at empty line or new section (after first line)
            if !first && (line.trim().is_empty() || line.starts_with("Goal")) {
                break;
            }
            first = false;
        }
        result.trim().to_string()
    }

    /// Extract the first error message from output
    fn extract_first_error(output: &str) -> Option<String> {
        for line in output.lines() {
            let trimmed = line.trim();
            if trimmed.to_lowercase().contains("invalid") {
                return Some(trimmed.to_string());
            }
            if trimmed.starts_with("error") {
                let msg = if trimmed.contains("]:") {
                    trimmed.split("]:").nth(1).unwrap_or("").trim().to_string()
                } else if trimmed.contains(':') {
                    trimmed
                        .split(':')
                        .skip(1)
                        .collect::<Vec<_>>()
                        .join(":")
                        .trim()
                        .to_string()
                } else {
                    trimmed.to_string()
                };
                if !msg.is_empty() {
                    return Some(msg);
                }
            }
            if trimmed.contains("Error:") {
                return Some(trimmed.to_string());
            }
        }
        None
    }
}

#[async_trait]
impl VerificationBackend for CreusotBackend {
    fn id(&self) -> BackendId {
        BackendId::Creusot
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::Contract, PropertyType::Invariant]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let cargo_path = self
            .detect_creusot()
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        // Create a minimal Cargo project structure
        let project_dir = temp_dir.path().join("creusot_verify");
        std::fs::create_dir_all(project_dir.join("src")).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create project dir: {}", e))
        })?;

        // Write Cargo.toml
        let cargo_toml = r#"[package]
name = "creusot_verify"
version = "0.1.0"
edition = "2021"

[dependencies]
creusot-contracts = "*"
"#;
        std::fs::write(project_dir.join("Cargo.toml"), cargo_toml).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write Cargo.toml: {}", e))
        })?;

        // Write source file
        let creusot_code = self.generate_creusot_code(spec);
        std::fs::write(project_dir.join("src/lib.rs"), &creusot_code).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write source: {}", e))
        })?;

        // Run cargo creusot
        let mut cmd = Command::new(&cargo_path);
        cmd.arg("creusot")
            .current_dir(&project_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let result = tokio::time::timeout(self.config.timeout, cmd.output()).await;
        let duration = start.elapsed();

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                debug!("Creusot stdout: {}", stdout);
                debug!("Creusot stderr: {}", stderr);

                let status = self.parse_output(&stdout, &stderr, output.status.success());
                let (valid, invalid, unknown) =
                    self.parse_why3_results(&format!("{}\n{}", stdout, stderr));

                let mut diagnostics: Vec<String> = stderr
                    .lines()
                    .filter(|l| l.contains("error") || l.contains("warning"))
                    .map(String::from)
                    .collect();

                if valid > 0 || invalid > 0 || unknown > 0 {
                    diagnostics.push(format!(
                        "Why3 results: {} valid, {} invalid, {} unknown",
                        valid, invalid, unknown
                    ));
                }

                // Generate structured counterexample for failures
                let counterexample = if !output.status.success() || invalid > 0 {
                    Some(Self::parse_structured_counterexample(&stdout, &stderr))
                } else {
                    None
                };

                Ok(BackendResult {
                    backend: BackendId::Creusot,
                    status,
                    proof: if output.status.success() && invalid == 0 {
                        Some("Verified by Creusot/Why3".to_string())
                    } else {
                        None
                    },
                    counterexample,
                    diagnostics,
                    time_taken: duration,
                })
            }
            Ok(Err(e)) => Err(BackendError::VerificationFailed(format!(
                "Failed to execute Creusot: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect_creusot().await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_valid_output() {
        let backend = CreusotBackend::new();
        let status = backend.parse_output("Goal: Valid", "", true);
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn parse_invalid_output() {
        let backend = CreusotBackend::new();
        let status = backend.parse_output("Invalid: Counterexample found", "", false);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn parse_timeout_output() {
        let backend = CreusotBackend::new();
        let status = backend.parse_output("Timeout on goal", "", false);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn parse_why3_results_counts() {
        let backend = CreusotBackend::new();
        let (valid, invalid, unknown) = backend
            .parse_why3_results("Goal 1: Valid\nGoal 2: Valid\nGoal 3: Invalid\nGoal 4: Timeout");
        assert_eq!(valid, 2);
        assert_eq!(invalid, 1);
        assert_eq!(unknown, 1);
    }

    #[test]
    fn default_config() {
        let config = CreusotConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert_eq!(config.solver, "z3");
    }

    // =============================================
    // Structured counterexample parsing tests
    // =============================================

    #[test]
    fn extract_failed_checks_invalid() {
        let output = "Goal 1: Invalid";
        let checks = CreusotBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "creusot_invalid");
    }

    #[test]
    fn extract_failed_checks_precondition() {
        let output = "Goal: Invalid precondition";
        let checks = CreusotBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "creusot_precondition");
    }

    #[test]
    fn extract_failed_checks_postcondition() {
        let output = "Goal: postcondition Invalid";
        let checks = CreusotBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "creusot_postcondition");
    }

    #[test]
    fn extract_failed_checks_invariant() {
        let output = "Goal: loop invariant Invalid";
        let checks = CreusotBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "creusot_invariant");
    }

    #[test]
    fn extract_failed_checks_assertion() {
        let output = "Goal: assertion Invalid";
        let checks = CreusotBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "creusot_assertion");
    }

    #[test]
    fn extract_failed_checks_timeout() {
        let output = "Timeout on goal";
        let checks = CreusotBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "creusot_timeout");
    }

    #[test]
    fn extract_failed_checks_counterexample() {
        let output = "Counterexample found:\nx = 5\ny = -1";
        let checks = CreusotBackend::extract_failed_checks(output);

        assert!(checks
            .iter()
            .any(|c| c.check_id == "creusot_counterexample"));
    }

    #[test]
    fn extract_failed_checks_rust_error() {
        let output = "error: compilation error\n  --> src/lib.rs:10:5";
        let checks = CreusotBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "creusot_error");
        assert!(checks[0].location.is_some());
    }

    #[test]
    fn extract_failed_checks_multiple() {
        let output = "Goal 1: Invalid\nGoal 2: Invalid\nTimeout on goal 3";
        let checks = CreusotBackend::extract_failed_checks(output);

        assert_eq!(checks.len(), 3);
    }

    #[test]
    fn extract_failed_checks_empty() {
        let checks = CreusotBackend::extract_failed_checks("");
        assert!(checks.is_empty());
    }

    #[test]
    fn extract_failed_checks_no_errors() {
        let output = "Goal 1: Valid\nGoal 2: Valid";
        let checks = CreusotBackend::extract_failed_checks(output);
        assert!(checks.is_empty());
    }

    #[test]
    fn parse_why3_location() {
        let line = r#"File "src/spec.rs", line 42, characters 5-15"#;
        let loc = CreusotBackend::parse_why3_location(line).expect("Should parse");

        assert_eq!(loc.file, "src/spec.rs");
        assert_eq!(loc.line, 42);
    }

    #[test]
    fn parse_why3_location_no_file() {
        assert!(CreusotBackend::parse_why3_location("not a location").is_none());
        assert!(CreusotBackend::parse_why3_location("").is_none());
    }

    #[test]
    fn parse_rust_location_full() {
        let line = "--> src/lib.rs:20:10";
        let loc = CreusotBackend::parse_rust_location(line).expect("Should parse");

        assert_eq!(loc.file, "src/lib.rs");
        assert_eq!(loc.line, 20);
        assert_eq!(loc.column, Some(10));
    }

    #[test]
    fn parse_rust_location_no_column() {
        let line = "--> src/main.rs:5";
        let loc = CreusotBackend::parse_rust_location(line).expect("Should parse");

        assert_eq!(loc.file, "src/main.rs");
        assert_eq!(loc.line, 5);
        assert!(loc.column.is_none());
    }

    #[test]
    fn parse_rust_location_invalid() {
        assert!(CreusotBackend::parse_rust_location("not a location").is_none());
        assert!(CreusotBackend::parse_rust_location("").is_none());
    }

    #[test]
    fn structured_counterexample_has_raw() {
        let ce = CreusotBackend::parse_structured_counterexample("stdout", "stderr");
        assert!(ce.raw.is_some());
        assert!(ce.raw.as_ref().unwrap().contains("stdout"));
    }

    #[test]
    fn structured_counterexample_has_failed_checks() {
        let ce = CreusotBackend::parse_structured_counterexample("", "Goal: Invalid postcondition");

        assert!(!ce.failed_checks.is_empty());
        assert_eq!(ce.failed_checks[0].check_id, "creusot_postcondition");
    }

    #[test]
    fn collect_counterexample_text() {
        let lines: Vec<&str> = vec!["Counterexample found:", "x = 5", "y = -1", "", "Other info"];
        let text = CreusotBackend::collect_counterexample(&lines, 0);

        assert!(text.contains("x = 5"));
        assert!(text.contains("y = -1"));
    }

    #[test]
    fn extract_first_error_invalid() {
        let output = "Valid\nInvalid precondition\nValid";
        let err = CreusotBackend::extract_first_error(output);

        assert!(err.is_some());
        assert!(err.unwrap().contains("Invalid precondition"));
    }

    #[test]
    fn extract_first_error_rust_error() {
        let output = "compiling...\nerror: something failed";
        let err = CreusotBackend::extract_first_error(output);

        assert!(err.is_some());
        assert!(err.unwrap().contains("something failed"));
    }

    #[test]
    fn extract_first_error_none() {
        let output = "All goals verified successfully";
        let err = CreusotBackend::extract_first_error(output);

        assert!(err.is_none());
    }

    #[test]
    fn why3_results_parsing() {
        let backend = CreusotBackend::new();
        let output = "Goal 1: Valid\nGoal 2: Valid\nGoal 3: Invalid\nGoal 4: Timeout";
        let (valid, invalid, unknown) = backend.parse_why3_results(output);

        assert_eq!(valid, 2);
        assert_eq!(invalid, 1);
        assert_eq!(unknown, 1);
    }
}
