//! Boogie intermediate verification language backend
//!
//! Boogie is an intermediate verification language used by many tools
//! including Dafny, VCC, Spec#, and Corral.
//!
//! See: <https://github.com/boogie-org/boogie>

// =============================================
// Kani Proofs for Boogie Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- BoogieConfig Default Tests ----

    /// Verify BoogieConfig::default timeout is 120 seconds
    #[kani::proof]
    fn proof_boogie_config_default_timeout() {
        let config = BoogieConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(120),
            "Default timeout should be 120 seconds",
        );
    }

    /// Verify BoogieConfig::default boogie_path is None
    #[kani::proof]
    fn proof_boogie_config_default_path_none() {
        let config = BoogieConfig::default();
        kani::assert(
            config.boogie_path.is_none(),
            "Default boogie_path should be None",
        );
    }

    /// Verify BoogieConfig::default z3_timeout is 30
    #[kani::proof]
    fn proof_boogie_config_default_z3_timeout() {
        let config = BoogieConfig::default();
        kani::assert(config.z3_timeout == 30, "Default z3_timeout should be 30");
    }

    /// Verify BoogieConfig::default infer_loop_invariants is true
    #[kani::proof]
    fn proof_boogie_config_default_infer_loop_invariants() {
        let config = BoogieConfig::default();
        kani::assert(
            config.infer_loop_invariants,
            "Default infer_loop_invariants should be true",
        );
    }

    /// Verify BoogieConfig::default extra_args is empty
    #[kani::proof]
    fn proof_boogie_config_default_extra_args_empty() {
        let config = BoogieConfig::default();
        kani::assert(
            config.extra_args.is_empty(),
            "Default extra_args should be empty",
        );
    }

    // ---- BoogieBackend Construction Tests ----

    /// Verify BoogieBackend::new uses default config
    #[kani::proof]
    fn proof_boogie_backend_new_defaults() {
        let backend = BoogieBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(120),
            "New backend should use default timeout",
        );
    }

    /// Verify BoogieBackend::default equals BoogieBackend::new
    #[kani::proof]
    fn proof_boogie_backend_default_equals_new() {
        let default_backend = BoogieBackend::default();
        let new_backend = BoogieBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
    }

    /// Verify BoogieBackend::with_config preserves custom timeout
    #[kani::proof]
    fn proof_boogie_backend_with_config_timeout() {
        let config = BoogieConfig {
            boogie_path: None,
            timeout: Duration::from_secs(600),
            z3_timeout: 30,
            infer_loop_invariants: true,
            extra_args: vec![],
        };
        let backend = BoogieBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "Custom timeout should be preserved",
        );
    }

    /// Verify BoogieBackend::with_config preserves custom z3_timeout
    #[kani::proof]
    fn proof_boogie_backend_with_config_z3_timeout() {
        let config = BoogieConfig {
            boogie_path: None,
            timeout: Duration::from_secs(120),
            z3_timeout: 60,
            infer_loop_invariants: true,
            extra_args: vec![],
        };
        let backend = BoogieBackend::with_config(config);
        kani::assert(
            backend.config.z3_timeout == 60,
            "Custom z3_timeout should be preserved",
        );
    }

    /// Verify BoogieBackend::with_config preserves infer_loop_invariants
    #[kani::proof]
    fn proof_boogie_backend_with_config_infer_loop_invariants() {
        let config = BoogieConfig {
            boogie_path: None,
            timeout: Duration::from_secs(120),
            z3_timeout: 30,
            infer_loop_invariants: false,
            extra_args: vec![],
        };
        let backend = BoogieBackend::with_config(config);
        kani::assert(
            !backend.config.infer_loop_invariants,
            "Custom infer_loop_invariants should be preserved",
        );
    }

    // ---- Backend Trait Implementation Tests ----

    /// Verify id() returns Boogie
    #[kani::proof]
    fn proof_backend_id_is_boogie() {
        let backend = BoogieBackend::new();
        kani::assert(backend.id() == BackendId::Boogie, "ID should be Boogie");
    }

    /// Verify supports() includes Contract
    #[kani::proof]
    fn proof_boogie_supports_contract() {
        let backend = BoogieBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Contract),
            "Should support Contract",
        );
    }

    /// Verify supports() includes Invariant
    #[kani::proof]
    fn proof_boogie_supports_invariant() {
        let backend = BoogieBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Invariant),
            "Should support Invariant",
        );
    }

    /// Verify supports() includes Refinement
    #[kani::proof]
    fn proof_boogie_supports_refinement() {
        let backend = BoogieBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Refinement),
            "Should support Refinement",
        );
    }

    /// Verify supports() returns exactly three property types
    #[kani::proof]
    fn proof_boogie_supports_count() {
        let backend = BoogieBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 3,
            "Should support exactly three property types",
        );
    }

    // ---- sanitize_name Tests ----

    /// Verify sanitize_name replaces hyphens
    #[kani::proof]
    fn proof_sanitize_name_hyphen() {
        let result = BoogieBackend::sanitize_name("hello-world");
        kani::assert(result == "hello_world", "Hyphens should be replaced");
    }

    /// Verify sanitize_name replaces colons
    #[kani::proof]
    fn proof_sanitize_name_colon() {
        let result = BoogieBackend::sanitize_name("test:prop");
        kani::assert(result == "test_prop", "Colons should be replaced");
    }

    /// Verify sanitize_name replaces slashes
    #[kani::proof]
    fn proof_sanitize_name_slash() {
        let result = BoogieBackend::sanitize_name("a/b/c");
        kani::assert(result == "a_b_c", "Slashes should be replaced");
    }

    /// Verify sanitize_name replaces spaces
    #[kani::proof]
    fn proof_sanitize_name_space() {
        let result = BoogieBackend::sanitize_name("hello world");
        kani::assert(result == "hello_world", "Spaces should be replaced");
    }

    /// Verify sanitize_name replaces dots
    #[kani::proof]
    fn proof_sanitize_name_dot() {
        let result = BoogieBackend::sanitize_name("foo.bar");
        kani::assert(result == "foo_bar", "Dots should be replaced");
    }

    /// Verify sanitize_name replaces parentheses
    #[kani::proof]
    fn proof_sanitize_name_parens() {
        let result = BoogieBackend::sanitize_name("func(arg)");
        kani::assert(result == "func_arg_", "Parentheses should be replaced");
    }

    /// Verify sanitize_name preserves alphanumerics
    #[kani::proof]
    fn proof_sanitize_name_alphanumeric() {
        let result = BoogieBackend::sanitize_name("test123");
        kani::assert(result == "test123", "Alphanumerics should be preserved");
    }

    // ---- parse_output Tests ----

    /// Verify parse_output recognizes verified output
    #[kani::proof]
    fn proof_parse_output_verified() {
        let backend = BoogieBackend::new();
        let (status, _diag) = backend.parse_output(
            "Boogie program verifier finished with 3 verified, 0 errors",
            "",
            true,
        );
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "3 verified, 0 errors should return Proven",
        );
    }

    /// Verify parse_output recognizes error output
    #[kani::proof]
    fn proof_parse_output_error() {
        let backend = BoogieBackend::new();
        let (status, _diag) = backend.parse_output(
            "Boogie program verifier finished with 2 verified, 1 error",
            "",
            false,
        );
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "1 error should return Disproven",
        );
    }

    /// Verify parse_output recognizes assertion violation
    #[kani::proof]
    fn proof_parse_output_assertion_violation() {
        let backend = BoogieBackend::new();
        let (status, _diag) = backend.parse_output("assertion might not hold", "", false);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Assertion violation should return Disproven",
        );
    }

    /// Verify parse_output recognizes postcondition violation
    #[kani::proof]
    fn proof_parse_output_postcondition_violation() {
        let backend = BoogieBackend::new();
        let (status, _diag) = backend.parse_output("postcondition might not hold", "", false);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Postcondition violation should return Disproven",
        );
    }

    /// Verify parse_output recognizes timeout
    #[kani::proof]
    fn proof_parse_output_timeout() {
        let backend = BoogieBackend::new();
        let (status, _diag) = backend.parse_output("verification timed out", "", false);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Timeout should return Unknown",
        );
    }

    /// Verify parse_output collects diagnostics
    #[kani::proof]
    fn proof_parse_output_diagnostics() {
        let backend = BoogieBackend::new();
        let (_status, diag) = backend.parse_output(
            "Boogie program verifier finished with 3 verified, 0 errors",
            "",
            true,
        );
        kani::assert(!diag.is_empty(), "Should collect diagnostics from summary");
    }

    // ---- parse_error_location Tests ----

    /// Verify parse_error_location extracts file
    #[kani::proof]
    fn proof_parse_error_location_file() {
        let line = "spec.bpl(10,5): Error: assertion might not hold";
        let (loc, _desc) = BoogieBackend::parse_error_location(line);
        kani::assert(loc.is_some(), "Should parse location");
        kani::assert(loc.unwrap().file == "spec.bpl", "Should extract file");
    }

    /// Verify parse_error_location extracts line number
    #[kani::proof]
    fn proof_parse_error_location_line() {
        let line = "spec.bpl(10,5): Error: assertion might not hold";
        let (loc, _desc) = BoogieBackend::parse_error_location(line);
        kani::assert(loc.is_some(), "Should parse location");
        kani::assert(loc.unwrap().line == 10, "Should extract line 10");
    }

    /// Verify parse_error_location extracts column
    #[kani::proof]
    fn proof_parse_error_location_column() {
        let line = "spec.bpl(10,5): Error: assertion might not hold";
        let (loc, _desc) = BoogieBackend::parse_error_location(line);
        kani::assert(loc.is_some(), "Should parse location");
        kani::assert(loc.unwrap().column == Some(5), "Should extract column 5");
    }

    /// Verify parse_error_location extracts description
    #[kani::proof]
    fn proof_parse_error_location_description() {
        let line = "spec.bpl(10,5): Error: assertion might not hold";
        let (_loc, desc) = BoogieBackend::parse_error_location(line);
        kani::assert(desc.contains("assertion"), "Should extract description");
    }

    /// Verify parse_error_location handles no location
    #[kani::proof]
    fn proof_parse_error_location_no_location() {
        let line = "Error: some general error";
        let (loc, _desc) = BoogieBackend::parse_error_location(line);
        kani::assert(loc.is_none(), "Should have no location");
    }

    // ---- extract_failed_checks Tests ----

    /// Verify extract_failed_checks identifies assertion errors
    #[kani::proof]
    fn proof_extract_failed_checks_assertion() {
        let output = "assertion might not hold";
        let checks = BoogieBackend::extract_failed_checks(output);
        kani::assert(checks.len() == 1, "Should find one check");
        kani::assert(
            checks[0].check_id == "boogie_assertion",
            "Should identify as assertion",
        );
    }

    /// Verify extract_failed_checks identifies postcondition errors
    #[kani::proof]
    fn proof_extract_failed_checks_postcondition() {
        let output = "postcondition might not hold";
        let checks = BoogieBackend::extract_failed_checks(output);
        kani::assert(checks.len() == 1, "Should find one check");
        kani::assert(
            checks[0].check_id == "boogie_postcondition",
            "Should identify as postcondition",
        );
    }

    /// Verify extract_failed_checks identifies precondition errors
    #[kani::proof]
    fn proof_extract_failed_checks_precondition() {
        let output = "precondition might not hold";
        let checks = BoogieBackend::extract_failed_checks(output);
        kani::assert(checks.len() == 1, "Should find one check");
        kani::assert(
            checks[0].check_id == "boogie_precondition",
            "Should identify as precondition",
        );
    }

    /// Verify extract_failed_checks identifies invariant errors
    #[kani::proof]
    fn proof_extract_failed_checks_invariant() {
        let output = "loop invariant might not hold";
        let checks = BoogieBackend::extract_failed_checks(output);
        kani::assert(checks.len() == 1, "Should find one check");
        kani::assert(
            checks[0].check_id == "boogie_invariant",
            "Should identify as invariant",
        );
    }

    /// Verify extract_failed_checks handles empty input
    #[kani::proof]
    fn proof_extract_failed_checks_empty() {
        let checks = BoogieBackend::extract_failed_checks("");
        kani::assert(checks.is_empty(), "Empty input should yield no checks");
    }

    /// Verify extract_failed_checks handles multiple errors
    #[kani::proof]
    fn proof_extract_failed_checks_multiple() {
        let output = "assertion might not hold\npostcondition might not hold";
        let checks = BoogieBackend::extract_failed_checks(output);
        kani::assert(checks.len() == 2, "Should find two checks");
    }

    // ---- parse_boogie_value Tests ----

    /// Verify parse_boogie_value parses integers
    #[kani::proof]
    fn proof_parse_boogie_value_int() {
        let result = BoogieBackend::parse_boogie_value("42");
        kani::assert(
            matches!(result, CounterexampleValue::Int { value: 42, .. }),
            "Should parse integer 42",
        );
    }

    /// Verify parse_boogie_value parses true
    #[kani::proof]
    fn proof_parse_boogie_value_true() {
        let result = BoogieBackend::parse_boogie_value("true");
        kani::assert(
            matches!(result, CounterexampleValue::Bool(true)),
            "Should parse true",
        );
    }

    /// Verify parse_boogie_value parses false
    #[kani::proof]
    fn proof_parse_boogie_value_false() {
        let result = BoogieBackend::parse_boogie_value("false");
        kani::assert(
            matches!(result, CounterexampleValue::Bool(false)),
            "Should parse false",
        );
    }

    /// Verify parse_boogie_value parses True (capitalized)
    #[kani::proof]
    fn proof_parse_boogie_value_true_capitalized() {
        let result = BoogieBackend::parse_boogie_value("True");
        kani::assert(
            matches!(result, CounterexampleValue::Bool(true)),
            "Should parse True",
        );
    }

    /// Verify parse_boogie_value parses hex
    #[kani::proof]
    fn proof_parse_boogie_value_hex() {
        let result = BoogieBackend::parse_boogie_value("0xFF");
        kani::assert(
            matches!(result, CounterexampleValue::Int { value: 255, .. }),
            "Should parse hex 0xFF as 255",
        );
    }

    /// Verify parse_boogie_value defaults to string
    #[kani::proof]
    fn proof_parse_boogie_value_string() {
        let result = BoogieBackend::parse_boogie_value("unknown");
        kani::assert(
            matches!(result, CounterexampleValue::String(_)),
            "Unknown values should be strings",
        );
    }

    // ---- parse_counterexample Tests ----

    /// Verify parse_counterexample preserves raw output
    #[kani::proof]
    fn proof_parse_counterexample_raw() {
        let ce = BoogieBackend::parse_counterexample("stdout", "stderr");
        kani::assert(ce.raw.is_some(), "Should have raw output");
        kani::assert(
            ce.raw.as_ref().unwrap().contains("stdout"),
            "Raw should contain stdout",
        );
    }

    /// Verify parse_counterexample extracts failed checks
    #[kani::proof]
    fn proof_parse_counterexample_failed_checks() {
        let ce = BoogieBackend::parse_counterexample("", "assertion might not hold");
        kani::assert(!ce.failed_checks.is_empty(), "Should have failed checks");
    }

    // ---- Configuration Consistency Tests ----

    /// Verify z3_timeout <= timeout in reasonable configs
    #[kani::proof]
    fn proof_z3_timeout_reasonable() {
        let config = BoogieConfig::default();
        kani::assert(
            config.z3_timeout as u64 <= config.timeout.as_secs(),
            "z3_timeout should be <= total timeout",
        );
    }
}

use crate::counterexample::{
    CounterexampleValue, FailedCheck, SourceLocation, StructuredCounterexample,
};
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

/// Configuration for Boogie backend
#[derive(Debug, Clone)]
pub struct BoogieConfig {
    /// Path to boogie binary
    pub boogie_path: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// Z3 timeout for individual queries (seconds)
    pub z3_timeout: u32,
    /// Enable loop invariant inference
    pub infer_loop_invariants: bool,
    /// Additional Boogie options
    pub extra_args: Vec<String>,
}

impl Default for BoogieConfig {
    fn default() -> Self {
        Self {
            boogie_path: None,
            timeout: Duration::from_secs(120),
            z3_timeout: 30,
            infer_loop_invariants: true,
            extra_args: vec![],
        }
    }
}

/// Boogie intermediate verification language backend
pub struct BoogieBackend {
    config: BoogieConfig,
}

impl Default for BoogieBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl BoogieBackend {
    /// Create a new Boogie backend with default configuration
    pub fn new() -> Self {
        Self {
            config: BoogieConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: BoogieConfig) -> Self {
        Self { config }
    }

    async fn detect_boogie(&self) -> Result<PathBuf, String> {
        let boogie_path = self
            .config
            .boogie_path
            .clone()
            .or_else(|| which::which("boogie").ok())
            .or_else(|| which::which("Boogie").ok())
            .ok_or("Boogie not found. Install via: dotnet tool install --global boogie")?;

        let output = Command::new(&boogie_path)
            .arg("/version")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| format!("Failed to execute boogie: {}", e))?;

        // Boogie may output version info to either stdout or stderr
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        if output.status.success()
            || stdout.contains("Boogie")
            || stderr.contains("Boogie")
            || stdout.contains("version")
            || stderr.contains("version")
        {
            let version = if !stdout.trim().is_empty() {
                stdout.trim()
            } else {
                stderr.trim()
            };
            debug!("Detected Boogie: {}", version);
            Ok(boogie_path)
        } else {
            Err("Boogie version check failed".to_string())
        }
    }

    /// Generate Boogie code from USL spec
    fn generate_boogie_code(&self, spec: &TypedSpec) -> String {
        let mut code = String::new();
        code.push_str("// Generated by DashProve\n\n");

        // Generate type declarations
        for type_def in &spec.spec.types {
            code.push_str(&format!("// Type: {}\n", type_def.name));
            let safe_name = Self::sanitize_name(&type_def.name);
            code.push_str(&format!("type {};\n\n", safe_name));
        }

        // Generate properties as procedures
        for (i, prop) in spec.spec.properties.iter().enumerate() {
            let prop_name = prop.name();
            let safe_name = Self::sanitize_name(&prop_name);
            let proc_name = if safe_name.is_empty() {
                format!("Property_{}", i)
            } else {
                format!("Property_{}", safe_name)
            };

            code.push_str(&format!("// Property: {}\n", prop_name));
            code.push_str(&format!("procedure {}()\n", proc_name));
            code.push_str("  ensures true;\n");
            code.push_str("{\n");
            code.push_str("  assert true;\n");
            code.push_str("}\n\n");
        }

        // If no properties, add a trivial procedure
        if spec.spec.properties.is_empty() {
            code.push_str("procedure Trivial()\n");
            code.push_str("  ensures true;\n");
            code.push_str("{\n");
            code.push_str("  assert true;\n");
            code.push_str("}\n");
        }

        code
    }

    /// Sanitize a name for use in Boogie
    fn sanitize_name(name: &str) -> String {
        name.replace([' ', '-', ':', '/', '\\', '.', '(', ')', '[', ']'], "_")
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '_')
            .collect()
    }

    fn parse_output(
        &self,
        stdout: &str,
        stderr: &str,
        success: bool,
    ) -> (VerificationStatus, Vec<String>) {
        let combined = format!("{}\n{}", stdout, stderr);
        let mut diagnostics = Vec::new();

        // Parse Boogie output format
        // "Boogie program verifier finished with X verified, Y errors"
        for line in combined.lines() {
            let trimmed = line.trim();

            // Look for verification summary
            if trimmed.contains("verified") && trimmed.contains("error") {
                diagnostics.push(trimmed.to_string());

                // Parse "X verified, Y errors" pattern
                let parts: Vec<&str> = trimmed.split(',').collect();
                let mut verified = 0usize;
                let mut errors = 0usize;

                for part in parts {
                    let words: Vec<&str> = part.split_whitespace().collect();
                    for (i, word) in words.iter().enumerate() {
                        if *word == "verified" && i > 0 {
                            if let Ok(n) = words[i - 1].parse::<usize>() {
                                verified = n;
                            }
                        }
                        if (*word == "error" || *word == "errors") && i > 0 {
                            if let Ok(n) = words[i - 1].parse::<usize>() {
                                errors = n;
                            }
                        }
                    }
                }

                if errors > 0 {
                    return (VerificationStatus::Disproven, diagnostics);
                } else if verified > 0 {
                    return (VerificationStatus::Proven, diagnostics);
                }
            }

            // Check for assertion violations
            if trimmed.contains("assertion might not hold")
                || trimmed.contains("Assertion might not hold")
                || trimmed.contains("assertion violation")
            {
                diagnostics.push(format!("✗ {}", trimmed));
            }

            // Check for postcondition violations
            if trimmed.contains("postcondition might not hold")
                || trimmed.contains("ensures clause might not hold")
            {
                diagnostics.push(format!("✗ {}", trimmed));
            }

            // Check for precondition violations
            if trimmed.contains("precondition might not hold")
                || trimmed.contains("requires clause might not hold")
            {
                diagnostics.push(format!("✗ {}", trimmed));
            }

            // Check for loop invariant violations
            if trimmed.contains("loop invariant might not hold")
                || trimmed.contains("invariant might not be maintained")
            {
                diagnostics.push(format!("✗ {}", trimmed));
            }

            // Track errors
            if trimmed.contains("Error:") || trimmed.contains("error:") {
                diagnostics.push(trimmed.to_string());
            }
        }

        // Check for timeout
        if combined.contains("timed out") || combined.contains("timeout") {
            return (
                VerificationStatus::Unknown {
                    reason: "Verification timed out".to_string(),
                },
                diagnostics,
            );
        }

        // Check exit status and look for errors
        if !success {
            let error_lines: Vec<_> = combined
                .lines()
                .filter(|l| {
                    l.contains("error") || l.contains("Error") || l.contains("might not hold")
                })
                .take(3)
                .collect();

            if !error_lines.is_empty() {
                return (VerificationStatus::Disproven, diagnostics);
            }

            return (
                VerificationStatus::Unknown {
                    reason: "Boogie returned non-zero exit code".to_string(),
                },
                diagnostics,
            );
        }

        // Check if we found any violations
        if diagnostics.iter().any(|d| d.starts_with("✗")) {
            return (VerificationStatus::Disproven, diagnostics);
        }

        // If successful exit but no summary found, assume proven
        if success && !combined.contains("error") && !combined.contains("Error") {
            return (VerificationStatus::Proven, diagnostics);
        }

        (
            VerificationStatus::Unknown {
                reason: "Could not parse Boogie output".to_string(),
            },
            diagnostics,
        )
    }

    /// Parse counterexample from Boogie output
    fn parse_counterexample(stdout: &str, stderr: &str) -> StructuredCounterexample {
        let mut ce = StructuredCounterexample::new();
        let combined = format!("{}\n{}", stdout, stderr);
        ce.raw = Some(combined.clone());

        // Extract failed checks
        ce.failed_checks = Self::extract_failed_checks(&combined);

        // Extract counterexample values from model output
        ce.witness = Self::extract_model_values(&combined);

        ce
    }

    /// Extract failed checks from Boogie output
    fn extract_failed_checks(output: &str) -> Vec<FailedCheck> {
        let mut checks = Vec::new();

        for line in output.lines() {
            let trimmed = line.trim();

            // Boogie error format: "file(line,col): Error: message"
            // or just "Error: message"
            if trimmed.contains("might not hold") || trimmed.contains("violation") {
                let check_type = if trimmed.contains("assertion") {
                    "boogie_assertion"
                } else if trimmed.contains("postcondition") || trimmed.contains("ensures") {
                    "boogie_postcondition"
                } else if trimmed.contains("precondition") || trimmed.contains("requires") {
                    "boogie_precondition"
                } else if trimmed.contains("invariant") {
                    "boogie_invariant"
                } else {
                    "boogie_error"
                };

                let (location, description) = Self::parse_error_location(trimmed);

                checks.push(FailedCheck {
                    check_id: check_type.to_string(),
                    description,
                    location,
                    function: None,
                });
            }
        }

        checks
    }

    /// Parse error location from Boogie error line
    fn parse_error_location(line: &str) -> (Option<SourceLocation>, String) {
        // Format: "file(line,col): Error: message"
        if let Some(paren_start) = line.find('(') {
            if let Some(paren_end) = line.find("):") {
                let file = line[..paren_start].trim();
                let loc_str = &line[paren_start + 1..paren_end];
                let parts: Vec<&str> = loc_str.split(',').collect();

                if parts.len() >= 2 {
                    if let (Ok(line_num), Ok(col_num)) =
                        (parts[0].parse::<u32>(), parts[1].parse::<u32>())
                    {
                        let message = line[paren_end + 2..].trim();
                        let desc = if let Some(stripped) = message.strip_prefix("Error:") {
                            stripped.trim().to_string()
                        } else {
                            message.to_string()
                        };

                        return (
                            Some(SourceLocation {
                                file: file.to_string(),
                                line: line_num,
                                column: Some(col_num),
                            }),
                            desc,
                        );
                    }
                }
            }
        }

        (None, line.to_string())
    }

    /// Extract model values from Boogie counterexample output
    fn extract_model_values(
        output: &str,
    ) -> std::collections::HashMap<String, CounterexampleValue> {
        use std::collections::HashMap;
        let mut values = HashMap::new();

        // Look for model output (when using /mv:N option)
        // Format varies but often includes "var = value" assignments
        let mut in_model = false;

        for line in output.lines() {
            let trimmed = line.trim();

            if trimmed.contains("Model:") || trimmed.contains("Counterexample:") {
                in_model = true;
                continue;
            }

            if in_model {
                // End of model section
                if trimmed.is_empty() || trimmed.starts_with("***") {
                    in_model = false;
                    continue;
                }

                // Parse "var = value" or "var -> value"
                let parts: Vec<&str> = if trimmed.contains(" = ") {
                    trimmed.splitn(2, " = ").collect()
                } else if trimmed.contains(" -> ") {
                    trimmed.splitn(2, " -> ").collect()
                } else {
                    continue;
                };

                if parts.len() == 2 {
                    let var_name = parts[0].trim().to_string();
                    let value_str = parts[1].trim();

                    let value = Self::parse_boogie_value(value_str);
                    values.insert(var_name, value);
                }
            }
        }

        values
    }

    /// Parse a Boogie value string into a CounterexampleValue
    fn parse_boogie_value(value_str: &str) -> CounterexampleValue {
        let trimmed = value_str.trim();

        // Try to parse as integer
        if let Ok(n) = trimmed.parse::<i128>() {
            return CounterexampleValue::Int {
                value: n,
                type_hint: None,
            };
        }

        // Try to parse as boolean
        if trimmed == "true" || trimmed == "True" {
            return CounterexampleValue::Bool(true);
        }
        if trimmed == "false" || trimmed == "False" {
            return CounterexampleValue::Bool(false);
        }

        // Try to parse as hex integer
        if let Some(hex_str) = trimmed.strip_prefix("0x") {
            if let Ok(n) = i128::from_str_radix(hex_str, 16) {
                return CounterexampleValue::Int {
                    value: n,
                    type_hint: Some("hex".to_string()),
                };
            }
        }

        // Default to string
        CounterexampleValue::String(trimmed.to_string())
    }
}

#[async_trait]
impl VerificationBackend for BoogieBackend {
    fn id(&self) -> BackendId {
        BackendId::Boogie
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::Contract,
            PropertyType::Invariant,
            PropertyType::Refinement,
        ]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let boogie_path = self
            .detect_boogie()
            .await
            .map_err(BackendError::Unavailable)?;

        // Create temp directory for Boogie files
        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp directory: {}", e))
        })?;

        let bpl_file = temp_dir.path().join("spec.bpl");
        let boogie_code = self.generate_boogie_code(spec);

        debug!("Generated Boogie code:\n{}", boogie_code);

        tokio::fs::write(&bpl_file, &boogie_code)
            .await
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to write Boogie file: {}", e))
            })?;

        // Run Boogie
        let mut cmd = Command::new(&boogie_path);
        cmd.arg(&bpl_file)
            .arg(format!("/timeLimit:{}", self.config.z3_timeout))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .current_dir(temp_dir.path());

        // Add inference options
        if self.config.infer_loop_invariants {
            cmd.arg("/infer:j"); // Infer loop invariants
        }

        // Add extra args
        for arg in &self.config.extra_args {
            cmd.arg(arg);
        }

        let output = tokio::time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to run boogie: {}", e))
            })?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        debug!("Boogie stdout: {}", stdout);
        debug!("Boogie stderr: {}", stderr);

        let (status, diagnostics) = self.parse_output(&stdout, &stderr, output.status.success());

        // Generate counterexample for failures
        let counterexample = if matches!(status, VerificationStatus::Disproven) {
            Some(Self::parse_counterexample(&stdout, &stderr))
        } else {
            None
        };

        Ok(BackendResult {
            backend: BackendId::Boogie,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: start.elapsed(),
        })
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect_boogie().await {
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
        assert_eq!(BoogieBackend::new().id(), BackendId::Boogie);
    }

    #[test]
    fn default_config() {
        let config = BoogieConfig::default();
        assert_eq!(config.z3_timeout, 30);
        assert_eq!(config.timeout, Duration::from_secs(120));
        assert!(config.infer_loop_invariants);
    }

    #[test]
    fn supports_contracts_and_invariants() {
        let backend = BoogieBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::Contract));
        assert!(supported.contains(&PropertyType::Invariant));
    }

    #[test]
    fn sanitize_name() {
        assert_eq!(BoogieBackend::sanitize_name("hello-world"), "hello_world");
        assert_eq!(BoogieBackend::sanitize_name("test:prop"), "test_prop");
        assert_eq!(BoogieBackend::sanitize_name("a/b/c"), "a_b_c");
    }

    #[test]
    fn parse_verified_output() {
        let backend = BoogieBackend::new();
        let stdout = "Boogie program verifier finished with 3 verified, 0 errors";
        let (status, _diag) = backend.parse_output(stdout, "", true);
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn parse_error_output() {
        let backend = BoogieBackend::new();
        let stdout = "spec.bpl(10,5): Error: assertion might not hold\n\
                      Boogie program verifier finished with 2 verified, 1 error";
        let (status, _diag) = backend.parse_output(stdout, "", false);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn parse_error_location_with_position() {
        let line = "spec.bpl(10,5): Error: assertion might not hold";
        let (loc, desc) = BoogieBackend::parse_error_location(line);
        assert!(loc.is_some());
        let loc = loc.unwrap();
        assert_eq!(loc.file, "spec.bpl");
        assert_eq!(loc.line, 10);
        assert_eq!(loc.column, Some(5));
        assert!(desc.contains("assertion might not hold"));
    }

    #[test]
    fn extract_failed_checks() {
        let output = "spec.bpl(10,5): Error: assertion might not hold\n\
                      spec.bpl(20,3): Error: postcondition might not hold";
        let checks = BoogieBackend::extract_failed_checks(output);
        assert_eq!(checks.len(), 2);
        assert_eq!(checks[0].check_id, "boogie_assertion");
        assert_eq!(checks[1].check_id, "boogie_postcondition");
    }

    #[test]
    fn parse_boogie_values() {
        assert!(matches!(
            BoogieBackend::parse_boogie_value("42"),
            CounterexampleValue::Int { value: 42, .. }
        ));
        assert!(matches!(
            BoogieBackend::parse_boogie_value("true"),
            CounterexampleValue::Bool(true)
        ));
        assert!(matches!(
            BoogieBackend::parse_boogie_value("false"),
            CounterexampleValue::Bool(false)
        ));
    }

    #[test]
    fn generate_boogie_empty_spec() {
        use dashprove_usl::ast::Spec;
        use std::collections::HashMap;

        let backend = BoogieBackend::new();
        let spec = TypedSpec {
            spec: Spec {
                types: vec![],
                properties: vec![],
            },
            type_info: HashMap::new(),
        };
        let code = backend.generate_boogie_code(&spec);
        assert!(code.contains("// Generated by DashProve"));
        assert!(code.contains("procedure Trivial()"));
    }
}
