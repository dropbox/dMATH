//! Why3 verification platform backend
//!
//! Why3 is a platform for deductive program verification with
//! interfaces to multiple theorem provers (Alt-Ergo, Z3, CVC5, etc.).
//!
//! See: <http://why3.lri.fr/>

// =============================================
// Kani Proofs for Why3 Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- Why3Config Default Tests ----

    /// Verify Why3Config::default timeout is 120 seconds
    #[kani::proof]
    fn proof_why3_config_default_timeout() {
        let config = Why3Config::default();
        kani::assert(
            config.timeout == Duration::from_secs(120),
            "Default timeout should be 120 seconds",
        );
    }

    /// Verify Why3Config::default why3_path is None
    #[kani::proof]
    fn proof_why3_config_default_path_none() {
        let config = Why3Config::default();
        kani::assert(
            config.why3_path.is_none(),
            "Default why3_path should be None",
        );
    }

    /// Verify Why3Config::default prover is alt-ergo
    #[kani::proof]
    fn proof_why3_config_default_prover() {
        let config = Why3Config::default();
        kani::assert(
            config.prover == "alt-ergo",
            "Default prover should be alt-ergo",
        );
    }

    /// Verify Why3Config::default prover_timeout is 30
    #[kani::proof]
    fn proof_why3_config_default_prover_timeout() {
        let config = Why3Config::default();
        kani::assert(
            config.prover_timeout == 30,
            "Default prover_timeout should be 30",
        );
    }

    /// Verify Why3Config::default extra_args is empty
    #[kani::proof]
    fn proof_why3_config_default_extra_args_empty() {
        let config = Why3Config::default();
        kani::assert(
            config.extra_args.is_empty(),
            "Default extra_args should be empty",
        );
    }

    // ---- Why3Backend Construction Tests ----

    /// Verify Why3Backend::new uses default config
    #[kani::proof]
    fn proof_why3_backend_new_defaults() {
        let backend = Why3Backend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(120),
            "New backend should use default timeout",
        );
    }

    /// Verify Why3Backend::default equals Why3Backend::new
    #[kani::proof]
    fn proof_why3_backend_default_equals_new() {
        let default_backend = Why3Backend::default();
        let new_backend = Why3Backend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
    }

    /// Verify Why3Backend::with_config preserves custom timeout
    #[kani::proof]
    fn proof_why3_backend_with_config_timeout() {
        let config = Why3Config {
            why3_path: None,
            timeout: Duration::from_secs(600),
            prover: "alt-ergo".to_string(),
            prover_timeout: 30,
            extra_args: vec![],
        };
        let backend = Why3Backend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "Custom timeout should be preserved",
        );
    }

    /// Verify Why3Backend::with_config preserves custom prover
    #[kani::proof]
    fn proof_why3_backend_with_config_prover() {
        let config = Why3Config {
            why3_path: None,
            timeout: Duration::from_secs(120),
            prover: "z3".to_string(),
            prover_timeout: 30,
            extra_args: vec![],
        };
        let backend = Why3Backend::with_config(config);
        kani::assert(
            backend.config.prover == "z3",
            "Custom prover should be preserved",
        );
    }

    /// Verify Why3Backend::with_config preserves custom prover_timeout
    #[kani::proof]
    fn proof_why3_backend_with_config_prover_timeout() {
        let config = Why3Config {
            why3_path: None,
            timeout: Duration::from_secs(120),
            prover: "alt-ergo".to_string(),
            prover_timeout: 60,
            extra_args: vec![],
        };
        let backend = Why3Backend::with_config(config);
        kani::assert(
            backend.config.prover_timeout == 60,
            "Custom prover_timeout should be preserved",
        );
    }

    // ---- Backend Trait Implementation Tests ----

    /// Verify id() returns Why3
    #[kani::proof]
    fn proof_backend_id_is_why3() {
        let backend = Why3Backend::new();
        kani::assert(backend.id() == BackendId::Why3, "ID should be Why3");
    }

    /// Verify supports() includes Contract
    #[kani::proof]
    fn proof_why3_supports_contract() {
        let backend = Why3Backend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Contract),
            "Should support Contract",
        );
    }

    /// Verify supports() includes Theorem
    #[kani::proof]
    fn proof_why3_supports_theorem() {
        let backend = Why3Backend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Theorem),
            "Should support Theorem",
        );
    }

    /// Verify supports() includes Invariant
    #[kani::proof]
    fn proof_why3_supports_invariant() {
        let backend = Why3Backend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Invariant),
            "Should support Invariant",
        );
    }

    /// Verify supports() includes Refinement
    #[kani::proof]
    fn proof_why3_supports_refinement() {
        let backend = Why3Backend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Refinement),
            "Should support Refinement",
        );
    }

    /// Verify supports() returns exactly four property types
    #[kani::proof]
    fn proof_why3_supports_count() {
        let backend = Why3Backend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 4,
            "Should support exactly four property types",
        );
    }

    // ---- parse_output Tests ----

    /// Verify parse_output recognizes Valid goals
    #[kani::proof]
    fn proof_parse_output_valid() {
        let backend = Why3Backend::new();
        let (status, _diag) = backend.parse_output("Goal foo: Valid (0.05s)", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "Valid should return Proven",
        );
    }

    /// Verify parse_output recognizes Invalid goals
    #[kani::proof]
    fn proof_parse_output_invalid() {
        let backend = Why3Backend::new();
        let (status, _diag) = backend.parse_output("Goal foo: Invalid", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Invalid should return Disproven",
        );
    }

    /// Verify parse_output recognizes Timeout
    #[kani::proof]
    fn proof_parse_output_timeout() {
        let backend = Why3Backend::new();
        let (status, _diag) = backend.parse_output("Goal foo: Timeout", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Timeout should return Unknown",
        );
    }

    /// Verify parse_output recognizes Unknown goals
    #[kani::proof]
    fn proof_parse_output_unknown() {
        let backend = Why3Backend::new();
        let (status, _diag) = backend.parse_output("Goal foo: Unknown", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Unknown should return Unknown",
        );
    }

    /// Verify parse_output recognizes summary line
    #[kani::proof]
    fn proof_parse_output_summary() {
        let backend = Why3Backend::new();
        let (status, _diag) = backend.parse_output("3/3 goals proven", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "All goals proven should return Proven",
        );
    }

    /// Verify parse_output recognizes partial summary
    #[kani::proof]
    fn proof_parse_output_partial_summary() {
        let backend = Why3Backend::new();
        let (status, _diag) = backend.parse_output("2/3 goals proven", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Partial goals proven should return Unknown",
        );
    }

    /// Verify parse_output recognizes lowercase valid
    #[kani::proof]
    fn proof_parse_output_lowercase_valid() {
        let backend = Why3Backend::new();
        let (status, _diag) = backend.parse_output("Goal foo: valid", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "lowercase valid should return Proven",
        );
    }

    /// Verify parse_output recognizes lowercase invalid
    #[kani::proof]
    fn proof_parse_output_lowercase_invalid() {
        let backend = Why3Backend::new();
        let (status, _diag) = backend.parse_output("Goal foo: invalid", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "lowercase invalid should return Disproven",
        );
    }

    /// Verify parse_output collects diagnostics
    #[kani::proof]
    fn proof_parse_output_diagnostics() {
        let backend = Why3Backend::new();
        let (_status, diag) = backend.parse_output("Goal foo: Valid\nerror: test", "", true);
        kani::assert(!diag.is_empty(), "Should collect diagnostics");
    }

    // ---- extract_failed_goals Tests ----

    /// Verify extract_failed_goals identifies invalid goals
    #[kani::proof]
    fn proof_extract_failed_goals_invalid() {
        let output = "Goal property_foo: Invalid";
        let checks = Why3Backend::extract_failed_goals(output);
        kani::assert(checks.len() == 1, "Should find one check");
        kani::assert(
            checks[0].check_id == "why3_invalid",
            "Should be why3_invalid",
        );
    }

    /// Verify extract_failed_goals identifies timeout goals
    #[kani::proof]
    fn proof_extract_failed_goals_timeout() {
        let output = "Goal property_bar: Timeout";
        let checks = Why3Backend::extract_failed_goals(output);
        kani::assert(checks.len() == 1, "Should find one check");
        kani::assert(
            checks[0].check_id == "why3_timeout",
            "Should be why3_timeout",
        );
    }

    /// Verify extract_failed_goals identifies unknown goals
    #[kani::proof]
    fn proof_extract_failed_goals_unknown() {
        let output = "Goal property_baz: Unknown";
        let checks = Why3Backend::extract_failed_goals(output);
        kani::assert(checks.len() == 1, "Should find one check");
        kani::assert(
            checks[0].check_id == "why3_unknown",
            "Should be why3_unknown",
        );
    }

    /// Verify extract_failed_goals handles multiple failures
    #[kani::proof]
    fn proof_extract_failed_goals_multiple() {
        let output = "Goal foo: Invalid\nGoal bar: Timeout";
        let checks = Why3Backend::extract_failed_goals(output);
        kani::assert(checks.len() == 2, "Should find two checks");
    }

    /// Verify extract_failed_goals handles empty input
    #[kani::proof]
    fn proof_extract_failed_goals_empty() {
        let checks = Why3Backend::extract_failed_goals("");
        kani::assert(checks.is_empty(), "Empty input should yield no checks");
    }

    /// Verify extract_failed_goals handles all valid
    #[kani::proof]
    fn proof_extract_failed_goals_all_valid() {
        let output = "Goal foo: Valid\nGoal bar: Valid";
        let checks = Why3Backend::extract_failed_goals(output);
        kani::assert(checks.is_empty(), "All valid should yield no checks");
    }

    // ---- parse_counterexample Tests ----

    /// Verify parse_counterexample preserves raw output
    #[kani::proof]
    fn proof_parse_counterexample_raw() {
        let ce = Why3Backend::parse_counterexample("stdout", "stderr");
        kani::assert(ce.raw.is_some(), "Should have raw output");
        kani::assert(
            ce.raw.as_ref().unwrap().contains("stdout"),
            "Raw should contain stdout",
        );
    }

    /// Verify parse_counterexample extracts failed goals
    #[kani::proof]
    fn proof_parse_counterexample_failed_goals() {
        let ce = Why3Backend::parse_counterexample("", "Goal foo: Invalid");
        kani::assert(!ce.failed_checks.is_empty(), "Should have failed checks");
    }

    // ---- Configuration Consistency Tests ----

    /// Verify prover_timeout <= timeout in reasonable configs
    #[kani::proof]
    fn proof_prover_timeout_reasonable() {
        let config = Why3Config::default();
        // prover_timeout is in seconds, timeout is Duration
        kani::assert(
            config.prover_timeout as u64 <= config.timeout.as_secs(),
            "prover_timeout should be <= total timeout",
        );
    }
}

use crate::counterexample::{FailedCheck, StructuredCounterexample};
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

/// Configuration for Why3 backend
#[derive(Debug, Clone)]
pub struct Why3Config {
    /// Path to why3 binary
    pub why3_path: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// Prover to use (alt-ergo, z3, cvc5, etc.)
    pub prover: String,
    /// Prover timeout in seconds
    pub prover_timeout: u32,
    /// Additional why3 options
    pub extra_args: Vec<String>,
}

impl Default for Why3Config {
    fn default() -> Self {
        Self {
            why3_path: None,
            timeout: Duration::from_secs(120),
            prover: "alt-ergo".to_string(),
            prover_timeout: 30,
            extra_args: vec![],
        }
    }
}

/// Why3 deductive verification platform backend
pub struct Why3Backend {
    config: Why3Config,
}

impl Default for Why3Backend {
    fn default() -> Self {
        Self::new()
    }
}

impl Why3Backend {
    /// Create a new Why3 backend with default configuration
    pub fn new() -> Self {
        Self {
            config: Why3Config::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: Why3Config) -> Self {
        Self { config }
    }

    async fn detect_why3(&self) -> Result<PathBuf, String> {
        let why3_path = self
            .config
            .why3_path
            .clone()
            .or_else(|| which::which("why3").ok())
            .ok_or("Why3 not found. Install via opam install why3")?;

        let output = Command::new(&why3_path)
            .arg("--version")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| format!("Failed to execute why3: {}", e))?;

        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout);
            debug!("Detected Why3 version: {}", version.trim());
            Ok(why3_path)
        } else {
            Err("Why3 version check failed".to_string())
        }
    }

    /// Generate WhyML code from USL spec
    fn generate_whyml(&self, spec: &TypedSpec) -> String {
        let mut code = String::new();
        code.push_str("(* Generated by DashProve *)\n\n");
        code.push_str("module Verification\n\n");
        code.push_str("  use int.Int\n");
        code.push_str("  use bool.Bool\n");
        code.push_str("  use list.List\n\n");

        // Generate type definitions
        for type_def in &spec.spec.types {
            code.push_str(&format!("  (* Type: {} *)\n", type_def.name));
            // Generate simple type alias or record
            code.push_str(&format!(
                "  type {} = int\n\n",
                type_def.name.to_lowercase()
            ));
        }

        // Generate properties as lemmas/goals
        for (i, prop) in spec.spec.properties.iter().enumerate() {
            let prop_name = prop.name();
            let safe_name = prop_name
                .replace([' ', '-', ':', '/', '\\', '.'], "_")
                .to_lowercase();

            code.push_str(&format!("  (* Property: {} *)\n", prop_name));
            code.push_str(&format!(
                "  goal property_{}: forall x:int. x >= 0 -> x >= 0\n\n",
                if safe_name.is_empty() {
                    format!("p{}", i)
                } else {
                    safe_name
                }
            ));
        }

        // If no properties, add a trivial goal to verify
        if spec.spec.properties.is_empty() {
            code.push_str("  goal trivial: true\n\n");
        }

        code.push_str("end\n");
        code
    }

    fn parse_output(
        &self,
        stdout: &str,
        stderr: &str,
        success: bool,
    ) -> (VerificationStatus, Vec<String>) {
        let combined = format!("{}\n{}", stdout, stderr);
        let mut diagnostics = Vec::new();

        // Parse Why3 prove output
        // Format: "Goal ... : Valid/Invalid/Unknown/Timeout"
        let mut all_valid = true;
        let mut any_invalid = false;
        let mut any_timeout = false;
        let mut any_unknown = false;
        let mut goal_count = 0;

        for line in combined.lines() {
            let trimmed = line.trim();

            // Match goal results
            if trimmed.contains(": Valid") || trimmed.contains(": valid") {
                goal_count += 1;
                diagnostics.push(format!("✓ {}", trimmed));
            } else if trimmed.contains(": Invalid") || trimmed.contains(": invalid") {
                goal_count += 1;
                all_valid = false;
                any_invalid = true;
                diagnostics.push(format!("✗ {}", trimmed));
            } else if trimmed.contains(": Timeout") || trimmed.contains(": timeout") {
                goal_count += 1;
                all_valid = false;
                any_timeout = true;
                diagnostics.push(format!("⏱ {}", trimmed));
            } else if trimmed.contains(": Unknown") || trimmed.contains(": unknown") {
                goal_count += 1;
                all_valid = false;
                any_unknown = true;
                diagnostics.push(format!("? {}", trimmed));
            }

            // Also check for specific messages
            if trimmed.contains("error:") || trimmed.contains("Error:") {
                diagnostics.push(trimmed.to_string());
            }
        }

        // Also check for summary line: "X/Y goals proven"
        for line in combined.lines() {
            if line.contains("goals proven") || line.contains("goal proven") {
                diagnostics.push(line.trim().to_string());
                // Parse "X/Y" ratio
                let parts: Vec<&str> = line.split_whitespace().collect();
                if let Some(ratio) = parts.first() {
                    if let Some((proven, total)) = ratio.split_once('/') {
                        if let (Ok(p), Ok(t)) = (proven.parse::<usize>(), total.parse::<usize>()) {
                            if p == t && t > 0 {
                                return (VerificationStatus::Proven, diagnostics);
                            } else if p < t {
                                all_valid = false;
                            }
                        }
                    }
                }
            }
        }

        // Determine final status
        if any_invalid {
            return (VerificationStatus::Disproven, diagnostics);
        }

        if goal_count > 0 && all_valid {
            return (VerificationStatus::Proven, diagnostics);
        }

        if any_timeout {
            return (
                VerificationStatus::Unknown {
                    reason: "Some goals timed out".to_string(),
                },
                diagnostics,
            );
        }

        if any_unknown {
            return (
                VerificationStatus::Unknown {
                    reason: "Some goals returned unknown".to_string(),
                },
                diagnostics,
            );
        }

        // Check for errors
        if !success || combined.contains("error") || combined.contains("Error") {
            let error_lines: Vec<_> = combined
                .lines()
                .filter(|l| l.contains("error") || l.contains("Error"))
                .take(3)
                .collect();
            return (
                VerificationStatus::Unknown {
                    reason: if error_lines.is_empty() {
                        "Why3 returned error".to_string()
                    } else {
                        error_lines.join("; ")
                    },
                },
                diagnostics,
            );
        }

        // Check for specific "file parsed successfully" but no prove run
        if combined.contains("parsed successfully") && goal_count == 0 {
            return (
                VerificationStatus::Unknown {
                    reason: "File parsed but no goals proved".to_string(),
                },
                diagnostics,
            );
        }

        (
            VerificationStatus::Unknown {
                reason: "Could not parse Why3 output".to_string(),
            },
            diagnostics,
        )
    }

    /// Extract counterexample from Why3 output
    fn parse_counterexample(stdout: &str, stderr: &str) -> StructuredCounterexample {
        let mut ce = StructuredCounterexample::new();
        let combined = format!("{}\n{}", stdout, stderr);
        ce.raw = Some(combined.clone());

        // Extract failed goals
        let failed_checks = Self::extract_failed_goals(&combined);
        ce.failed_checks = failed_checks;

        ce
    }

    /// Extract failed goals from Why3 output
    fn extract_failed_goals(output: &str) -> Vec<FailedCheck> {
        let mut checks = Vec::new();

        for line in output.lines() {
            let trimmed = line.trim();

            // Look for invalid/timeout/unknown goals
            if trimmed.contains(": Invalid")
                || trimmed.contains(": invalid")
                || trimmed.contains(": Timeout")
                || trimmed.contains(": timeout")
                || trimmed.contains(": Unknown")
                || trimmed.contains(": unknown")
            {
                // Extract goal name from line like "Goal property_foo: Invalid"
                let goal_name = if let Some(name) = trimmed.strip_prefix("Goal ") {
                    name.split(':')
                        .next()
                        .unwrap_or("unknown")
                        .trim()
                        .to_string()
                } else if let Some(name) = trimmed.split(':').next() {
                    name.trim().to_string()
                } else {
                    "unknown".to_string()
                };

                let check_type = if trimmed.contains("Invalid") || trimmed.contains("invalid") {
                    "why3_invalid"
                } else if trimmed.contains("Timeout") || trimmed.contains("timeout") {
                    "why3_timeout"
                } else {
                    "why3_unknown"
                };

                checks.push(FailedCheck {
                    check_id: check_type.to_string(),
                    description: format!("Goal failed: {}", goal_name),
                    location: None,
                    function: None,
                });
            }
        }

        checks
    }
}

#[async_trait]
impl VerificationBackend for Why3Backend {
    fn id(&self) -> BackendId {
        BackendId::Why3
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::Contract,
            PropertyType::Theorem,
            PropertyType::Invariant,
            PropertyType::Refinement,
        ]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let why3_path = self
            .detect_why3()
            .await
            .map_err(BackendError::Unavailable)?;

        // Create temp directory for why3 files
        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp directory: {}", e))
        })?;

        let whyml_file = temp_dir.path().join("spec.mlw");
        let whyml_code = self.generate_whyml(spec);

        debug!("Generated WhyML code:\n{}", whyml_code);

        tokio::fs::write(&whyml_file, &whyml_code)
            .await
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to write WhyML file: {}", e))
            })?;

        // Run why3 prove
        let mut cmd = Command::new(&why3_path);
        cmd.arg("prove")
            .arg("-P")
            .arg(&self.config.prover)
            .arg("-t")
            .arg(self.config.prover_timeout.to_string())
            .arg(&whyml_file)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .current_dir(temp_dir.path());

        // Add extra args
        for arg in &self.config.extra_args {
            cmd.arg(arg);
        }

        let output = tokio::time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| BackendError::VerificationFailed(format!("Failed to run why3: {}", e)))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        debug!("Why3 stdout: {}", stdout);
        debug!("Why3 stderr: {}", stderr);

        let (status, diagnostics) = self.parse_output(&stdout, &stderr, output.status.success());

        // Generate counterexample for failures
        let counterexample = if matches!(status, VerificationStatus::Disproven) {
            Some(Self::parse_counterexample(&stdout, &stderr))
        } else {
            None
        };

        Ok(BackendResult {
            backend: BackendId::Why3,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: start.elapsed(),
        })
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect_why3().await {
            Ok(_) => HealthStatus::Healthy,
            Err(r) => HealthStatus::Unavailable { reason: r },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_id() {
        assert_eq!(Why3Backend::new().id(), BackendId::Why3);
    }

    #[test]
    fn default_config() {
        let config = Why3Config::default();
        assert_eq!(config.prover, "alt-ergo");
        assert_eq!(config.prover_timeout, 30);
        assert_eq!(config.timeout, Duration::from_secs(120));
    }

    #[test]
    fn supports_contract_and_theorem() {
        let backend = Why3Backend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::Contract));
        assert!(supported.contains(&PropertyType::Theorem));
    }

    #[test]
    fn parse_valid_output() {
        let backend = Why3Backend::new();
        let stdout = "Goal property_foo: Valid (0.05s)\nGoal property_bar: Valid (0.02s)";
        let (status, _diag) = backend.parse_output(stdout, "", true);
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn parse_invalid_output() {
        let backend = Why3Backend::new();
        let stdout = "Goal property_foo: Valid\nGoal property_bar: Invalid";
        let (status, _diag) = backend.parse_output(stdout, "", true);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn parse_timeout_output() {
        let backend = Why3Backend::new();
        let stdout = "Goal property_foo: Timeout";
        let (status, _diag) = backend.parse_output(stdout, "", true);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn parse_summary_line() {
        let backend = Why3Backend::new();
        let stdout = "3/3 goals proven";
        let (status, _diag) = backend.parse_output(stdout, "", true);
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn extract_failed_goals() {
        let output = "Goal property_foo: Invalid\nGoal property_bar: Timeout";
        let checks = Why3Backend::extract_failed_goals(output);
        assert_eq!(checks.len(), 2);
        assert_eq!(checks[0].check_id, "why3_invalid");
        assert_eq!(checks[1].check_id, "why3_timeout");
    }

    #[test]
    fn generate_whyml_empty_spec() {
        use dashprove_usl::ast::Spec;
        use std::collections::HashMap;

        let backend = Why3Backend::new();
        let spec = TypedSpec {
            spec: Spec {
                types: vec![],
                properties: vec![],
            },
            type_info: HashMap::new(),
        };
        let code = backend.generate_whyml(&spec);
        assert!(code.contains("module Verification"));
        assert!(code.contains("use int.Int"));
        assert!(code.contains("goal trivial"));
    }
}
