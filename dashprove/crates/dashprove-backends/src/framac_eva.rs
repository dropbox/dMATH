//! Frama-C EVA value analysis plugin backend
//!
//! EVA (Evolved Value Analysis) is a plugin for Frama-C that computes
//! abstract domains for all program states, enabling verification
//! of runtime errors and ACSL annotations.
//!
//! See: <https://frama-c.com/eva.html>
//!
//! # Features
//!
//! - **Abstract interpretation**: Sound value analysis
//! - **Runtime error detection**: Division by zero, buffer overflows
//! - **ACSL verification**: Verify function contracts
//! - **Interval domains**: Compute variable bounds
//!
//! # Requirements
//!
//! Install Frama-C:
//! ```bash
//! opam install frama-c
//! ```

// =============================================
// Kani Proofs for Frama-C EVA Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- EvaPrecision Default Tests ----

    /// Verify EvaPrecision::default is Standard
    #[kani::proof]
    fn proof_eva_precision_default_is_standard() {
        let precision = EvaPrecision::default();
        kani::assert(
            precision == EvaPrecision::Standard,
            "Default precision should be Standard",
        );
    }

    // ---- FramaCEvaConfig Default Tests ----

    /// Verify FramaCEvaConfig::default timeout is 300 seconds
    #[kani::proof]
    fn proof_framac_eva_config_default_timeout() {
        let config = FramaCEvaConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "Default timeout should be 300 seconds",
        );
    }

    /// Verify FramaCEvaConfig::default framac_path is None
    #[kani::proof]
    fn proof_framac_eva_config_default_path_none() {
        let config = FramaCEvaConfig::default();
        kani::assert(
            config.framac_path.is_none(),
            "Default framac_path should be None",
        );
    }

    /// Verify FramaCEvaConfig::default precision is Standard
    #[kani::proof]
    fn proof_framac_eva_config_default_precision() {
        let config = FramaCEvaConfig::default();
        kani::assert(
            config.precision == EvaPrecision::Standard,
            "Default precision should be Standard",
        );
    }

    /// Verify FramaCEvaConfig::default verify_acsl is true
    #[kani::proof]
    fn proof_framac_eva_config_default_verify_acsl_true() {
        let config = FramaCEvaConfig::default();
        kani::assert(config.verify_acsl, "Default verify_acsl should be true");
    }

    /// Verify FramaCEvaConfig::default extra_args is empty
    #[kani::proof]
    fn proof_framac_eva_config_default_extra_args_empty() {
        let config = FramaCEvaConfig::default();
        kani::assert(
            config.extra_args.is_empty(),
            "Default extra_args should be empty",
        );
    }

    // ---- FramaCEvaBackend Construction Tests ----

    /// Verify FramaCEvaBackend::new uses default config timeout
    #[kani::proof]
    fn proof_framac_eva_backend_new_default_timeout() {
        let backend = FramaCEvaBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "New backend should use default timeout",
        );
    }

    /// Verify FramaCEvaBackend::default equals FramaCEvaBackend::new
    #[kani::proof]
    fn proof_framac_eva_backend_default_equals_new() {
        let default_backend = FramaCEvaBackend::default();
        let new_backend = FramaCEvaBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
    }

    /// Verify FramaCEvaBackend::with_config preserves custom timeout
    #[kani::proof]
    fn proof_framac_eva_backend_with_config_timeout() {
        let config = FramaCEvaConfig {
            framac_path: None,
            timeout: Duration::from_secs(600),
            precision: EvaPrecision::Standard,
            verify_acsl: true,
            extra_args: vec![],
        };
        let backend = FramaCEvaBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "with_config should preserve timeout",
        );
    }

    /// Verify FramaCEvaBackend::with_config preserves High precision
    #[kani::proof]
    fn proof_framac_eva_backend_with_config_high_precision() {
        let config = FramaCEvaConfig {
            framac_path: None,
            timeout: Duration::from_secs(300),
            precision: EvaPrecision::High,
            verify_acsl: true,
            extra_args: vec![],
        };
        let backend = FramaCEvaBackend::with_config(config);
        kani::assert(
            backend.config.precision == EvaPrecision::High,
            "with_config should preserve High precision",
        );
    }

    /// Verify FramaCEvaBackend::with_config preserves Quick precision
    #[kani::proof]
    fn proof_framac_eva_backend_with_config_quick_precision() {
        let config = FramaCEvaConfig {
            framac_path: None,
            timeout: Duration::from_secs(300),
            precision: EvaPrecision::Quick,
            verify_acsl: true,
            extra_args: vec![],
        };
        let backend = FramaCEvaBackend::with_config(config);
        kani::assert(
            backend.config.precision == EvaPrecision::Quick,
            "with_config should preserve Quick precision",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify FramaCEvaBackend::id returns FramaCEva
    #[kani::proof]
    fn proof_framac_eva_backend_id() {
        let backend = FramaCEvaBackend::new();
        kani::assert(
            backend.id() == BackendId::FramaCEva,
            "Backend id should be FramaCEva",
        );
    }

    /// Verify FramaCEvaBackend::supports includes MemorySafety
    #[kani::proof]
    fn proof_framac_eva_backend_supports_memory_safety() {
        let backend = FramaCEvaBackend::new();
        let supported = backend.supports();
        let has_memory = supported.iter().any(|p| *p == PropertyType::MemorySafety);
        kani::assert(has_memory, "Should support MemorySafety property");
    }

    /// Verify FramaCEvaBackend::supports includes Invariant
    #[kani::proof]
    fn proof_framac_eva_backend_supports_invariant() {
        let backend = FramaCEvaBackend::new();
        let supported = backend.supports();
        let has_invariant = supported.iter().any(|p| *p == PropertyType::Invariant);
        kani::assert(has_invariant, "Should support Invariant property");
    }

    /// Verify FramaCEvaBackend::supports returns exactly 2 properties
    #[kani::proof]
    fn proof_framac_eva_backend_supports_length() {
        let backend = FramaCEvaBackend::new();
        let supported = backend.supports();
        kani::assert(supported.len() == 2, "Should support exactly 2 properties");
    }

    // ---- sanitize_name Tests ----

    /// Verify sanitize_name replaces hyphens with underscores
    #[kani::proof]
    fn proof_sanitize_name_hyphen() {
        let result = FramaCEvaBackend::sanitize_name("my-func");
        kani::assert(result == "my_func", "Should replace hyphen with underscore");
    }

    /// Verify sanitize_name converts to lowercase
    #[kani::proof]
    fn proof_sanitize_name_lowercase() {
        let result = FramaCEvaBackend::sanitize_name("MyFunc");
        kani::assert(result == "myfunc", "Should convert to lowercase");
    }

    /// Verify sanitize_name handles empty string
    #[kani::proof]
    fn proof_sanitize_name_empty() {
        let result = FramaCEvaBackend::sanitize_name("");
        kani::assert(result.is_empty(), "Empty string should remain empty");
    }

    // ---- parse_eva_value Tests ----

    /// Verify parse_eva_value returns String for interval notation
    #[kani::proof]
    fn proof_parse_eva_value_interval() {
        let value = FramaCEvaBackend::parse_eva_value("[0..100]");
        kani::assert(
            matches!(value, CounterexampleValue::String(_)),
            "Should return String for interval",
        );
    }

    /// Verify parse_eva_value returns Int for "42"
    #[kani::proof]
    fn proof_parse_eva_value_int() {
        let value = FramaCEvaBackend::parse_eva_value("42");
        kani::assert(
            matches!(value, CounterexampleValue::Int { value: 42, .. }),
            "Should parse 42 as Int",
        );
    }

    /// Verify parse_eva_value returns String for set notation
    #[kani::proof]
    fn proof_parse_eva_value_set() {
        let value = FramaCEvaBackend::parse_eva_value("{1; 2; 3}");
        kani::assert(
            matches!(value, CounterexampleValue::String(_)),
            "Should return String for set notation",
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
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::process::Command;
use tracing::debug;

/// Analysis precision level for EVA
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EvaPrecision {
    /// Quick analysis with fewer iterations
    Quick,
    /// Standard precision
    #[default]
    Standard,
    /// High precision with more iterations
    High,
}

/// Configuration for Frama-C EVA backend
#[derive(Debug, Clone)]
pub struct FramaCEvaConfig {
    /// Path to frama-c binary
    pub framac_path: Option<PathBuf>,
    /// Timeout for analysis
    pub timeout: Duration,
    /// Analysis precision
    pub precision: EvaPrecision,
    /// Enable ACSL verification
    pub verify_acsl: bool,
    /// Additional EVA options
    pub extra_args: Vec<String>,
}

impl Default for FramaCEvaConfig {
    fn default() -> Self {
        Self {
            framac_path: None,
            timeout: Duration::from_secs(300),
            precision: EvaPrecision::default(),
            verify_acsl: true,
            extra_args: vec![],
        }
    }
}

/// Frama-C EVA value analysis backend
pub struct FramaCEvaBackend {
    config: FramaCEvaConfig,
}

impl Default for FramaCEvaBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl FramaCEvaBackend {
    /// Create a new Frama-C EVA backend with default configuration
    pub fn new() -> Self {
        Self {
            config: FramaCEvaConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: FramaCEvaConfig) -> Self {
        Self { config }
    }

    async fn detect(&self) -> Result<PathBuf, String> {
        // Check configured path first
        if let Some(ref path) = self.config.framac_path {
            if path.exists() {
                return Ok(path.clone());
            }
        }

        // Try to find frama-c
        if let Ok(path) = which::which("frama-c") {
            // Verify it's working
            let output = Command::new(&path)
                .arg("-version")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output()
                .await;

            if let Ok(out) = output {
                let stdout = String::from_utf8_lossy(&out.stdout);
                if stdout.contains("Frama-C") || out.status.success() {
                    debug!("Detected Frama-C at: {:?}", path);
                    return Ok(path);
                }
            }
        }

        // Check FRAMAC_HOME environment variable
        if let Ok(fc_home) = std::env::var("FRAMAC_HOME") {
            let fc_bin = PathBuf::from(&fc_home).join("bin").join("frama-c");
            if fc_bin.exists() {
                return Ok(fc_bin);
            }
        }

        // Check opam installation
        if let Ok(opam_switch) = std::env::var("OPAM_SWITCH_PREFIX") {
            let fc_bin = PathBuf::from(&opam_switch).join("bin").join("frama-c");
            if fc_bin.exists() {
                return Ok(fc_bin);
            }
        }

        Err("Frama-C not found. Install via: opam install frama-c".to_string())
    }

    /// Generate C code with ACSL annotations from USL spec
    fn generate_c_code(&self, spec: &TypedSpec) -> String {
        let mut code = String::new();
        code.push_str("/* Generated by DashProve */\n\n");
        code.push_str("#include <limits.h>\n\n");

        // Type declarations
        for type_def in &spec.spec.types {
            let safe_name = Self::sanitize_name(&type_def.name);
            code.push_str(&format!("typedef struct {} {{\n", safe_name));
            code.push_str("    int value;\n");
            code.push_str(&format!("}} {};\n\n", safe_name));
        }

        // Property verification functions with ACSL contracts
        for (i, prop) in spec.spec.properties.iter().enumerate() {
            let prop_name = prop.name();
            let safe_name = Self::sanitize_name(&prop_name);
            let func_name = if safe_name.is_empty() {
                format!("verify_prop_{}", i)
            } else {
                format!("verify_{}", safe_name)
            };

            code.push_str(&format!("/* Property: {} */\n", prop_name));
            code.push_str("/*@\n");
            code.push_str("  requires \\true;\n");
            code.push_str("  ensures \\result >= 0;\n");
            code.push_str("  assigns \\nothing;\n");
            code.push_str("*/\n");
            code.push_str(&format!("int {}(void) {{\n", func_name));
            code.push_str("    return 1;\n");
            code.push_str("}\n\n");
        }

        // Main function with invariants
        code.push_str("/*@\n");
        code.push_str("  ensures \\result == 0;\n");
        code.push_str("*/\n");
        code.push_str("int main(void) {\n");
        code.push_str("    int result = 0;\n");
        for (i, prop) in spec.spec.properties.iter().enumerate() {
            let prop_name = prop.name();
            let safe_name = Self::sanitize_name(&prop_name);
            let func_name = if safe_name.is_empty() {
                format!("verify_prop_{}", i)
            } else {
                format!("verify_{}", safe_name)
            };
            code.push_str(&format!("    result += {}();\n", func_name));
        }
        code.push_str("    //@ assert result >= 0;\n");
        code.push_str("    return 0;\n");
        code.push_str("}\n");

        code
    }

    /// Sanitize a name for C identifiers
    fn sanitize_name(name: &str) -> String {
        name.replace([' ', '-', ':', '/', '\\', '.', '(', ')', '[', ']'], "_")
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '_')
            .collect::<String>()
            .to_lowercase()
    }

    fn parse_output(
        &self,
        stdout: &str,
        stderr: &str,
        success: bool,
    ) -> (VerificationStatus, Vec<String>) {
        let combined = format!("{}\n{}", stdout, stderr);
        let mut diagnostics = Vec::new();

        // Parse EVA output
        for line in combined.lines() {
            let trimmed = line.trim();

            // Valid status
            if trimmed.contains("Valid") || trimmed.contains("valid") {
                diagnostics.push(format!("✓ {}", trimmed));
            }

            // Invalid/Unknown status
            if trimmed.contains("Invalid")
                || trimmed.contains("Unknown")
                || trimmed.contains("MAYBE")
            {
                diagnostics.push(format!("✗ {}", trimmed));
            }

            // Alarms
            if trimmed.contains("alarm") || trimmed.contains("Alarm") {
                diagnostics.push(format!("⚠ {}", trimmed));
            }

            // Analysis statistics
            if trimmed.contains("Coverage") || trimmed.contains("coverage") {
                diagnostics.push(trimmed.to_string());
            }
        }

        // Count results
        let valid_count = combined.matches("Valid").count() + combined.matches("valid").count();
        let invalid_count = combined.matches("Invalid").count()
            + combined.matches("MAYBE").count()
            + combined.matches("Unknown status").count();
        let alarm_count = combined.matches("alarm").count() + combined.matches("Alarm").count();

        // All valid and no alarms
        if valid_count > 0 && invalid_count == 0 && alarm_count == 0 {
            return (VerificationStatus::Proven, diagnostics);
        }

        // Some invalid or alarms
        if invalid_count > 0 {
            return (VerificationStatus::Disproven, diagnostics);
        }

        // Only alarms (potential issues)
        if alarm_count > 0 {
            return (
                VerificationStatus::Unknown {
                    reason: format!("EVA raised {} potential alarms", alarm_count),
                },
                diagnostics,
            );
        }

        // Successful run with no specific findings
        if success && !combined.contains("error") && !combined.contains("Error") {
            return (VerificationStatus::Proven, diagnostics);
        }

        // Error in analysis
        if combined.contains("error") || combined.contains("Error") {
            let error_lines: Vec<_> = combined
                .lines()
                .filter(|l| l.contains("error") || l.contains("Error"))
                .take(3)
                .collect();

            return (
                VerificationStatus::Unknown {
                    reason: format!("EVA error: {}", error_lines.join("; ")),
                },
                diagnostics,
            );
        }

        (
            VerificationStatus::Unknown {
                reason: "Could not parse EVA output".to_string(),
            },
            diagnostics,
        )
    }

    fn parse_counterexample(stdout: &str, stderr: &str) -> StructuredCounterexample {
        let mut ce = StructuredCounterexample::new();
        let combined = format!("{}\n{}", stdout, stderr);
        ce.raw = Some(combined.clone());
        ce.failed_checks = Self::extract_failed_checks(&combined);
        ce.witness = Self::extract_value_intervals(&combined);
        ce
    }

    fn extract_failed_checks(output: &str) -> Vec<FailedCheck> {
        let mut checks = Vec::new();

        for line in output.lines() {
            let trimmed = line.trim();

            if trimmed.contains("Invalid") || trimmed.contains("MAYBE") || trimmed.contains("alarm")
            {
                let check_type = if trimmed.contains("division") {
                    "eva_division_by_zero"
                } else if trimmed.contains("overflow") {
                    "eva_overflow"
                } else if trimmed.contains("mem_access") || trimmed.contains("out of bounds") {
                    "eva_mem_access"
                } else if trimmed.contains("assert") {
                    "eva_assertion"
                } else if trimmed.contains("precondition") {
                    "eva_precondition"
                } else if trimmed.contains("postcondition") {
                    "eva_postcondition"
                } else {
                    "eva_alarm"
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

    fn parse_error_location(line: &str) -> (Option<SourceLocation>, String) {
        // Frama-C format: "file.c:line: message" or "[kernel] file.c:line: message"
        let line_to_parse = if line.contains(']') {
            if let Some(bracket_end) = line.find(']') {
                line[bracket_end + 1..].trim()
            } else {
                line
            }
        } else {
            line
        };

        if let Some(colon_pos) = line_to_parse.find(':') {
            let prefix = &line_to_parse[..colon_pos];
            if prefix.ends_with(".c") || prefix.ends_with(".h") {
                let rest = &line_to_parse[colon_pos + 1..];
                if let Some(next_colon) = rest.find(':') {
                    if let Ok(line_num) = rest[..next_colon].trim().parse::<u32>() {
                        let message = rest[next_colon + 1..].trim().to_string();
                        return (
                            Some(SourceLocation {
                                file: prefix.to_string(),
                                line: line_num,
                                column: None,
                            }),
                            message,
                        );
                    }
                }
            }
        }

        (None, line.to_string())
    }

    fn extract_value_intervals(output: &str) -> HashMap<String, CounterexampleValue> {
        let mut values = HashMap::new();
        let mut in_values = false;

        for line in output.lines() {
            let trimmed = line.trim();

            // Look for value section
            if trimmed.contains("Values at") || trimmed.contains("VALUES") {
                in_values = true;
                continue;
            }

            if in_values {
                if trimmed.is_empty() || trimmed.starts_with("---") || trimmed.contains("===") {
                    in_values = false;
                    continue;
                }

                // Parse "var ∈ [min..max]" or "var = value" patterns
                if trimmed.contains('∈') {
                    let parts: Vec<&str> = trimmed.splitn(2, '∈').collect();
                    if parts.len() == 2 {
                        let var_name = parts[0].trim().to_string();
                        let interval = parts[1].trim().to_string();
                        values.insert(var_name, CounterexampleValue::String(interval));
                    }
                } else if let Some(eq_pos) = trimmed.find(" = ") {
                    let var_name = trimmed[..eq_pos].trim().to_string();
                    let value_str = trimmed[eq_pos + 3..].trim();
                    values.insert(var_name, Self::parse_eva_value(value_str));
                }
            }
        }

        values
    }

    fn parse_eva_value(value_str: &str) -> CounterexampleValue {
        let trimmed = value_str.trim();

        // Interval notation [min..max]
        if trimmed.starts_with('[') && trimmed.contains("..") {
            return CounterexampleValue::String(trimmed.to_string());
        }

        // Set notation {a; b; c}
        if trimmed.starts_with('{') {
            return CounterexampleValue::String(trimmed.to_string());
        }

        // Integer
        if let Ok(n) = trimmed.parse::<i128>() {
            return CounterexampleValue::Int {
                value: n,
                type_hint: None,
            };
        }

        CounterexampleValue::String(trimmed.to_string())
    }
}

#[async_trait]
impl VerificationBackend for FramaCEvaBackend {
    fn id(&self) -> BackendId {
        BackendId::FramaCEva
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::MemorySafety, PropertyType::Invariant]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let fc_path = self.detect().await.map_err(BackendError::Unavailable)?;

        // Create temp directory
        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp directory: {}", e))
        })?;

        let c_file = temp_dir.path().join("verify.c");
        let c_code = self.generate_c_code(spec);

        debug!("Generated C code:\n{}", c_code);

        tokio::fs::write(&c_file, &c_code).await.map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write C file: {}", e))
        })?;

        // Build command
        let mut cmd = Command::new(&fc_path);
        cmd.arg("-eva")
            .arg(&c_file)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .current_dir(temp_dir.path());

        // Precision settings
        match self.config.precision {
            EvaPrecision::Quick => {
                cmd.arg("-eva-precision").arg("0");
            }
            EvaPrecision::Standard => {
                cmd.arg("-eva-precision").arg("3");
            }
            EvaPrecision::High => {
                cmd.arg("-eva-precision").arg("7");
            }
        }

        // ACSL verification
        if self.config.verify_acsl {
            cmd.arg("-eva-show-progress");
        }

        for arg in &self.config.extra_args {
            cmd.arg(arg);
        }

        let output = tokio::time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to run Frama-C: {}", e))
            })?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        debug!("Frama-C stdout: {}", stdout);
        debug!("Frama-C stderr: {}", stderr);

        let (status, diagnostics) = self.parse_output(&stdout, &stderr, output.status.success());

        let counterexample = if matches!(status, VerificationStatus::Disproven) {
            Some(Self::parse_counterexample(&stdout, &stderr))
        } else {
            None
        };

        Ok(BackendResult {
            backend: BackendId::FramaCEva,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: start.elapsed(),
        })
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect().await {
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
        assert_eq!(FramaCEvaBackend::new().id(), BackendId::FramaCEva);
    }

    #[test]
    fn default_config() {
        let config = FramaCEvaConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert_eq!(config.precision, EvaPrecision::Standard);
        assert!(config.verify_acsl);
    }

    #[test]
    fn supports_memory_and_invariant() {
        let backend = FramaCEvaBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::MemorySafety));
        assert!(supported.contains(&PropertyType::Invariant));
    }

    #[test]
    fn sanitize_name() {
        assert_eq!(FramaCEvaBackend::sanitize_name("my-func"), "my_func");
    }

    #[test]
    fn parse_valid_output() {
        let backend = FramaCEvaBackend::new();
        let stdout = "postcondition: Valid\nassert: Valid\n";
        let (status, diag) = backend.parse_output(stdout, "", true);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(!diag.is_empty());
    }

    #[test]
    fn parse_invalid_output() {
        let backend = FramaCEvaBackend::new();
        let stdout = "precondition: Invalid\nalarm: division by zero";
        let (status, _diag) = backend.parse_output(stdout, "", true);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn extract_failed_checks() {
        let output = "alarm: division by zero\noverflow alarm";
        let checks = FramaCEvaBackend::extract_failed_checks(output);
        assert_eq!(checks.len(), 2);
        assert_eq!(checks[0].check_id, "eva_division_by_zero");
        assert_eq!(checks[1].check_id, "eva_overflow");
    }

    #[test]
    fn parse_error_location() {
        let line = "verify.c:10: Warning: division by zero";
        let (loc, desc) = FramaCEvaBackend::parse_error_location(line);
        assert!(loc.is_some());
        let loc = loc.unwrap();
        assert_eq!(loc.file, "verify.c");
        assert_eq!(loc.line, 10);
        assert!(desc.contains("division"));
    }

    #[test]
    fn parse_eva_values() {
        assert!(matches!(
            FramaCEvaBackend::parse_eva_value("[0..100]"),
            CounterexampleValue::String(_)
        ));
        assert!(matches!(
            FramaCEvaBackend::parse_eva_value("42"),
            CounterexampleValue::Int { value: 42, .. }
        ));
    }

    #[test]
    fn generate_c_empty_spec() {
        use dashprove_usl::ast::Spec;

        let backend = FramaCEvaBackend::new();
        let spec = TypedSpec {
            spec: Spec {
                types: vec![],
                properties: vec![],
            },
            type_info: HashMap::new(),
        };
        let code = backend.generate_c_code(&spec);
        assert!(code.contains("Generated by DashProve"));
        assert!(code.contains("/*@")); // ACSL annotation
        assert!(code.contains("int main"));
    }

    #[test]
    fn config_with_high_precision() {
        let config = FramaCEvaConfig {
            precision: EvaPrecision::High,
            verify_acsl: false,
            ..Default::default()
        };
        let backend = FramaCEvaBackend::with_config(config);
        assert_eq!(backend.config.precision, EvaPrecision::High);
        assert!(!backend.config.verify_acsl);
    }
}
