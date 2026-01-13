//! CBMC - C Bounded Model Checker backend
//!
//! CBMC is a bounded model checker for C and C++ programs. It verifies
//! memory safety, assertions, and other properties by converting the
//! program to SAT/SMT formulas.
//!
//! See: <https://www.cprover.org/cbmc/>
//!
//! # Features
//!
//! - **Memory safety**: Buffer overflows, null pointer dereferences
//! - **Assertions**: User-defined assertions (`assert(cond)`)
//! - **Undefined behavior**: Division by zero, shift overflows
//! - **Loop unwinding**: Bounded verification with configurable depth
//! - **Coverage**: Generate tests from counterexamples
//!
//! # Requirements
//!
//! Install CBMC:
//! ```bash
//! # macOS
//! brew install cbmc
//!
//! # Linux (Debian/Ubuntu)
//! apt install cbmc
//!
//! # From source
//! git clone https://github.com/diffblue/cbmc
//! cd cbmc && cmake -S . -B build && cmake --build build
//! ```

// =============================================
// Kani Proofs for CBMC Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- CbmcMode Enum Tests ----

    /// Verify CbmcMode::default is Verify
    #[kani::proof]
    fn proof_cbmc_mode_default() {
        let mode = CbmcMode::default();
        kani::assert(matches!(mode, CbmcMode::Verify), "Default should be Verify");
    }

    /// Verify CbmcMode variants are distinct
    #[kani::proof]
    fn proof_cbmc_mode_variants_distinct() {
        kani::assert(CbmcMode::Verify != CbmcMode::Cover, "Verify != Cover");
        kani::assert(
            CbmcMode::Verify != CbmcMode::ShowProperties,
            "Verify != ShowProperties",
        );
        kani::assert(
            CbmcMode::Cover != CbmcMode::ShowProperties,
            "Cover != ShowProperties",
        );
    }

    // ---- CbmcConfig Default Tests ----

    /// Verify CbmcConfig::default timeout is 300 seconds
    #[kani::proof]
    fn proof_cbmc_config_default_timeout() {
        let config = CbmcConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "Default timeout should be 300 seconds",
        );
    }

    /// Verify CbmcConfig::default unwind is 10
    #[kani::proof]
    fn proof_cbmc_config_default_unwind() {
        let config = CbmcConfig::default();
        kani::assert(config.unwind == Some(10), "Default unwind should be 10");
    }

    /// Verify CbmcConfig::default unwinding_assertions is true
    #[kani::proof]
    fn proof_cbmc_config_default_unwinding_assertions() {
        let config = CbmcConfig::default();
        kani::assert(
            config.unwinding_assertions,
            "Default unwinding_assertions should be true",
        );
    }

    /// Verify CbmcConfig::default bounds_check is true
    #[kani::proof]
    fn proof_cbmc_config_default_bounds_check() {
        let config = CbmcConfig::default();
        kani::assert(config.bounds_check, "Default bounds_check should be true");
    }

    /// Verify CbmcConfig::default pointer_check is true
    #[kani::proof]
    fn proof_cbmc_config_default_pointer_check() {
        let config = CbmcConfig::default();
        kani::assert(config.pointer_check, "Default pointer_check should be true");
    }

    /// Verify CbmcConfig::default memory_leak_check is true
    #[kani::proof]
    fn proof_cbmc_config_default_memory_leak_check() {
        let config = CbmcConfig::default();
        kani::assert(
            config.memory_leak_check,
            "Default memory_leak_check should be true",
        );
    }

    /// Verify CbmcConfig::default div_by_zero_check is true
    #[kani::proof]
    fn proof_cbmc_config_default_div_by_zero_check() {
        let config = CbmcConfig::default();
        kani::assert(
            config.div_by_zero_check,
            "Default div_by_zero_check should be true",
        );
    }

    /// Verify CbmcConfig::default signed_overflow_check is true
    #[kani::proof]
    fn proof_cbmc_config_default_signed_overflow_check() {
        let config = CbmcConfig::default();
        kani::assert(
            config.signed_overflow_check,
            "Default signed_overflow_check should be true",
        );
    }

    /// Verify CbmcConfig::default trace is true
    #[kani::proof]
    fn proof_cbmc_config_default_trace() {
        let config = CbmcConfig::default();
        kani::assert(config.trace, "Default trace should be true");
    }

    /// Verify CbmcConfig::default cbmc_path is None
    #[kani::proof]
    fn proof_cbmc_config_default_cbmc_path() {
        let config = CbmcConfig::default();
        kani::assert(
            config.cbmc_path.is_none(),
            "Default cbmc_path should be None",
        );
    }

    // ---- CbmcConfig Builder Tests ----

    /// Verify with_timeout preserves value
    #[kani::proof]
    fn proof_cbmc_config_with_timeout() {
        let config = CbmcConfig::default().with_timeout(Duration::from_secs(600));
        kani::assert(
            config.timeout == Duration::from_secs(600),
            "with_timeout should set timeout",
        );
    }

    /// Verify with_unwind preserves value
    #[kani::proof]
    fn proof_cbmc_config_with_unwind() {
        let config = CbmcConfig::default().with_unwind(20);
        kani::assert(config.unwind == Some(20), "with_unwind should set unwind");
    }

    /// Verify with_unwinding_assertions preserves value
    #[kani::proof]
    fn proof_cbmc_config_with_unwinding_assertions() {
        let config = CbmcConfig::default().with_unwinding_assertions(false);
        kani::assert(
            !config.unwinding_assertions,
            "with_unwinding_assertions should set value",
        );
    }

    /// Verify with_mode preserves value
    #[kani::proof]
    fn proof_cbmc_config_with_mode() {
        let config = CbmcConfig::default().with_mode(CbmcMode::Cover);
        kani::assert(config.mode == CbmcMode::Cover, "with_mode should set mode");
    }

    /// Verify with_bounds_check preserves value
    #[kani::proof]
    fn proof_cbmc_config_with_bounds_check() {
        let config = CbmcConfig::default().with_bounds_check(false);
        kani::assert(!config.bounds_check, "with_bounds_check should set value");
    }

    /// Verify with_pointer_check preserves value
    #[kani::proof]
    fn proof_cbmc_config_with_pointer_check() {
        let config = CbmcConfig::default().with_pointer_check(false);
        kani::assert(!config.pointer_check, "with_pointer_check should set value");
    }

    /// Verify with_cbmc_path preserves value
    #[kani::proof]
    fn proof_cbmc_config_with_cbmc_path() {
        let path = PathBuf::from("/usr/local/bin/cbmc");
        let config = CbmcConfig::default().with_cbmc_path(path.clone());
        kani::assert(
            config.cbmc_path == Some(path),
            "with_cbmc_path should set value",
        );
    }

    // ---- CbmcBackend Construction Tests ----

    /// Verify CbmcBackend::new uses default config
    #[kani::proof]
    fn proof_cbmc_backend_new_defaults() {
        let backend = CbmcBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "New backend should use default timeout",
        );
    }

    /// Verify CbmcBackend::with_config preserves config
    #[kani::proof]
    fn proof_cbmc_backend_with_config() {
        let config = CbmcConfig::default().with_timeout(Duration::from_secs(600));
        let backend = CbmcBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "Custom config should be preserved",
        );
    }

    /// Verify CbmcBackend::default equals CbmcBackend::new
    #[kani::proof]
    fn proof_cbmc_backend_default_equals_new() {
        let default_backend = CbmcBackend::default();
        let new_backend = CbmcBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
    }

    // ---- Backend Trait Implementation Tests ----

    /// Verify id() returns CBMC
    #[kani::proof]
    fn proof_backend_id_is_cbmc() {
        let backend = CbmcBackend::new();
        kani::assert(backend.id() == BackendId::CBMC, "ID should be CBMC");
    }

    /// Verify supports() includes Contract
    #[kani::proof]
    fn proof_cbmc_supports_contract() {
        let backend = CbmcBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Contract),
            "Should support Contract",
        );
    }

    /// Verify supports() includes Invariant
    #[kani::proof]
    fn proof_cbmc_supports_invariant() {
        let backend = CbmcBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Invariant),
            "Should support Invariant",
        );
    }

    /// Verify supports() includes MemorySafety
    #[kani::proof]
    fn proof_cbmc_supports_memory_safety() {
        let backend = CbmcBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::MemorySafety),
            "Should support MemorySafety",
        );
    }

    /// Verify supports() returns exactly three property types
    #[kani::proof]
    fn proof_cbmc_supports_count() {
        let backend = CbmcBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 3,
            "Should support exactly three property types",
        );
    }

    // ---- CbmcStats Tests ----

    /// Verify CbmcStats::default initializes properly
    #[kani::proof]
    fn proof_cbmc_stats_default() {
        let stats = CbmcStats::default();
        kani::assert(
            stats.verification_result.is_none(),
            "verification_result should be None",
        );
        kani::assert(
            stats.properties_checked == 0,
            "properties_checked should be 0",
        );
        kani::assert(
            stats.properties_failed == 0,
            "properties_failed should be 0",
        );
    }

    // ---- parse_cbmc_output Tests ----

    /// Verify parse_cbmc_output detects success
    #[kani::proof]
    fn proof_parse_cbmc_output_success() {
        let backend = CbmcBackend::new();
        let output = "VERIFICATION SUCCESSFUL";
        let (status, errors, stats) = backend.parse_cbmc_output(output, 0);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "Should be Proven",
        );
        kani::assert(errors.is_empty(), "No errors for success");
        kani::assert(
            stats.verification_result == Some(true),
            "verification_result should be true",
        );
    }

    /// Verify parse_cbmc_output detects failure
    #[kani::proof]
    fn proof_parse_cbmc_output_failure() {
        let backend = CbmcBackend::new();
        let output = "VERIFICATION FAILED";
        let (status, _errors, stats) = backend.parse_cbmc_output(output, 10);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Should be Disproven",
        );
        kani::assert(
            stats.verification_result == Some(false),
            "verification_result should be false",
        );
    }

    /// Verify parse_cbmc_output extracts property failure
    #[kani::proof]
    fn proof_parse_cbmc_output_property_failure() {
        let backend = CbmcBackend::new();
        let output = "property main.assertion.1: FAILURE";
        let (_status, errors, stats) = backend.parse_cbmc_output(output, 10);
        kani::assert(!errors.is_empty(), "Should have errors");
        kani::assert(
            stats.properties_failed == 1,
            "Should have one failed property",
        );
    }

    /// Verify parse_cbmc_output detects unwinding assertion
    #[kani::proof]
    fn proof_parse_cbmc_output_unwinding() {
        let backend = CbmcBackend::new();
        let output = "unwinding assertion loop 1: FAILURE";
        let (_status, errors, _stats) = backend.parse_cbmc_output(output, 10);
        kani::assert(!errors.is_empty(), "Should detect unwinding assertion");
        kani::assert(
            errors[0].contains("Unwinding bound"),
            "Should mention unwinding bound",
        );
    }

    /// Verify parse_cbmc_output handles exit code 0
    #[kani::proof]
    fn proof_parse_cbmc_output_exit_code_0() {
        let backend = CbmcBackend::new();
        let output = "no explicit result marker";
        let (status, _errors, _stats) = backend.parse_cbmc_output(output, 0);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "Exit code 0 should be Proven",
        );
    }

    /// Verify parse_cbmc_output handles exit code 10
    #[kani::proof]
    fn proof_parse_cbmc_output_exit_code_10() {
        let backend = CbmcBackend::new();
        let output = "no explicit result marker";
        let (status, _errors, _stats) = backend.parse_cbmc_output(output, 10);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Exit code 10 should be Disproven",
        );
    }

    /// Verify parse_cbmc_output handles unknown exit code
    #[kani::proof]
    fn proof_parse_cbmc_output_unknown() {
        let backend = CbmcBackend::new();
        let output = "no explicit result marker";
        let (status, _errors, _stats) = backend.parse_cbmc_output(output, 42);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Unknown exit code should be Unknown",
        );
    }
}

use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
    VerificationStatus,
};
use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::process::Command;

/// CBMC verification mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CbmcMode {
    /// Full verification (default)
    #[default]
    Verify,
    /// Cover mode - generate test cases
    Cover,
    /// Show properties only
    ShowProperties,
}

/// Configuration for CBMC backend
#[derive(Debug, Clone)]
pub struct CbmcConfig {
    /// Path to CBMC installation
    pub cbmc_path: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// Loop unwinding bound
    pub unwind: Option<usize>,
    /// Unwind assertions (fail if bound not sufficient)
    pub unwinding_assertions: bool,
    /// Verification mode
    pub mode: CbmcMode,
    /// Check array bounds
    pub bounds_check: bool,
    /// Check pointer dereferences
    pub pointer_check: bool,
    /// Check memory leaks
    pub memory_leak_check: bool,
    /// Check division by zero
    pub div_by_zero_check: bool,
    /// Check signed overflow
    pub signed_overflow_check: bool,
    /// Generate trace on counterexample
    pub trace: bool,
    /// Additional CBMC options
    pub extra_options: Vec<String>,
}

impl Default for CbmcConfig {
    fn default() -> Self {
        Self {
            cbmc_path: None,
            timeout: Duration::from_secs(300),
            unwind: Some(10),
            unwinding_assertions: true,
            mode: CbmcMode::default(),
            bounds_check: true,
            pointer_check: true,
            memory_leak_check: true,
            div_by_zero_check: true,
            signed_overflow_check: true,
            trace: true,
            extra_options: Vec::new(),
        }
    }
}

impl CbmcConfig {
    /// Set CBMC installation path
    pub fn with_cbmc_path(mut self, path: PathBuf) -> Self {
        self.cbmc_path = Some(path);
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set loop unwinding bound
    pub fn with_unwind(mut self, bound: usize) -> Self {
        self.unwind = Some(bound);
        self
    }

    /// Enable/disable unwinding assertions
    pub fn with_unwinding_assertions(mut self, enabled: bool) -> Self {
        self.unwinding_assertions = enabled;
        self
    }

    /// Set verification mode
    pub fn with_mode(mut self, mode: CbmcMode) -> Self {
        self.mode = mode;
        self
    }

    /// Enable/disable bounds checking
    pub fn with_bounds_check(mut self, enabled: bool) -> Self {
        self.bounds_check = enabled;
        self
    }

    /// Enable/disable pointer checking
    pub fn with_pointer_check(mut self, enabled: bool) -> Self {
        self.pointer_check = enabled;
        self
    }
}

/// CBMC bounded model checker backend
pub struct CbmcBackend {
    config: CbmcConfig,
}

impl CbmcBackend {
    /// Create a new CBMC backend with default configuration
    pub fn new() -> Self {
        Self {
            config: CbmcConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: CbmcConfig) -> Self {
        Self { config }
    }

    /// Generate C code from USL spec (placeholder for actual compilation)
    fn generate_c_code(&self, spec: &TypedSpec) -> String {
        let mut code = String::new();

        code.push_str("/* Generated by DashProve from USL spec */\n\n");
        code.push_str("#include <assert.h>\n");
        code.push_str("#include <stdbool.h>\n\n");

        // Generate assertions from properties
        for prop in &spec.spec.properties {
            let prop_name = prop.name();
            code.push_str(&format!("/* Property: {} */\n", prop_name));
        }

        // Generate a simple main function with assertions
        code.push_str("\nint main() {\n");
        code.push_str("    int x;\n");
        code.push_str("    __CPROVER_assume(x >= 0 && x < 100);\n");
        code.push_str("    \n");
        code.push_str("    // Property from USL spec\n");
        code.push_str("    assert(x >= 0);\n");
        code.push_str("    assert(x < 100);\n");
        code.push_str("    \n");
        code.push_str("    return 0;\n");
        code.push_str("}\n");

        code
    }

    /// Parse CBMC output
    fn parse_cbmc_output(
        &self,
        output: &str,
        exit_code: i32,
    ) -> (VerificationStatus, Vec<String>, CbmcStats) {
        let mut stats = CbmcStats::default();
        let mut errors = Vec::new();

        for line in output.lines() {
            if line.contains("VERIFICATION SUCCESSFUL") {
                stats.verification_result = Some(true);
            }
            if line.contains("VERIFICATION FAILED") {
                stats.verification_result = Some(false);
            }
            if line.contains("properties checked") {
                // Extract property count
                if let Some(num) = line.split_whitespace().find(|s| s.parse::<usize>().is_ok()) {
                    stats.properties_checked = num.parse().unwrap_or(0);
                }
            }
            if line.contains("property") && line.contains("FAILURE") {
                errors.push(line.to_string());
                stats.properties_failed += 1;
            }
            if line.contains("unwinding assertion") && line.contains("FAILURE") {
                errors.push("Unwinding bound insufficient".to_string());
            }
            if line.contains("assertion") && line.contains("file") {
                errors.push(line.to_string());
            }
        }

        let status = match stats.verification_result {
            Some(true) => VerificationStatus::Proven,
            Some(false) => VerificationStatus::Disproven,
            None => {
                if exit_code == 0 {
                    VerificationStatus::Proven
                } else if exit_code == 10 {
                    VerificationStatus::Disproven
                } else {
                    VerificationStatus::Unknown {
                        reason: format!("CBMC exited with code {}", exit_code),
                    }
                }
            }
        };

        (status, errors, stats)
    }

    /// Verify a C source file
    pub async fn verify_c_file(&self, c_path: &Path) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let cbmc_cmd = self
            .config
            .cbmc_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("cbmc"));

        let mut args = Vec::new();

        // Verification options
        if let Some(unwind) = self.config.unwind {
            args.push(format!("--unwind={}", unwind));
        }
        if self.config.unwinding_assertions {
            args.push("--unwinding-assertions".to_string());
        }

        // Safety checks
        if self.config.bounds_check {
            args.push("--bounds-check".to_string());
        }
        if self.config.pointer_check {
            args.push("--pointer-check".to_string());
        }
        if self.config.memory_leak_check {
            args.push("--memory-leak-check".to_string());
        }
        if self.config.div_by_zero_check {
            args.push("--div-by-zero-check".to_string());
        }
        if self.config.signed_overflow_check {
            args.push("--signed-overflow-check".to_string());
        }

        // Trace for counterexamples
        if self.config.trace {
            args.push("--trace".to_string());
        }

        // Mode-specific options
        match self.config.mode {
            CbmcMode::Verify => {}
            CbmcMode::Cover => {
                args.push("--cover".to_string());
                args.push("assertion".to_string());
            }
            CbmcMode::ShowProperties => {
                args.push("--show-properties".to_string());
            }
        }

        // Add source file
        args.push(c_path.to_string_lossy().to_string());

        // Extra options
        args.extend(self.config.extra_options.clone());

        // Run CBMC
        let mut cmd = Command::new(&cbmc_cmd);
        cmd.args(&args);

        let output = tokio::time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| BackendError::VerificationFailed(format!("Failed to run CBMC: {}", e)))?;

        let duration = start.elapsed();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined = format!("{}\n{}", stdout, stderr);

        let (status, errors, stats) =
            self.parse_cbmc_output(&combined, output.status.code().unwrap_or(-1));

        let mut diagnostics = Vec::new();

        let summary = match &status {
            VerificationStatus::Proven => format!(
                "CBMC: Verification successful ({} properties checked)",
                stats.properties_checked
            ),
            VerificationStatus::Disproven => {
                format!(
                    "CBMC: Verification failed ({} of {} properties failed)",
                    stats.properties_failed, stats.properties_checked
                )
            }
            VerificationStatus::Unknown { reason } => format!("CBMC: {}", reason),
            VerificationStatus::Partial {
                verified_percentage,
            } => {
                format!("CBMC: Partial verification ({:.1}%)", verified_percentage)
            }
        };
        diagnostics.push(summary);

        for error in &errors {
            diagnostics.push(format!("Error: {}", error));
        }

        let counterexample = if matches!(status, VerificationStatus::Disproven) && self.config.trace
        {
            // Use the formal verification helper to build a structured counterexample
            crate::counterexample::build_bmc_counterexample(&stdout, &stderr, "CBMC", None)
        } else {
            None
        };

        Ok(BackendResult {
            backend: BackendId::CBMC,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: duration,
        })
    }
}

impl Default for CbmcBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// CBMC verification statistics
#[derive(Debug, Default)]
struct CbmcStats {
    verification_result: Option<bool>,
    properties_checked: usize,
    properties_failed: usize,
}

#[async_trait]
impl VerificationBackend for CbmcBackend {
    fn id(&self) -> BackendId {
        BackendId::CBMC
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::Contract,
            PropertyType::Invariant,
            PropertyType::MemorySafety,
        ]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        // Generate C code
        let c_code = self.generate_c_code(spec);

        // Write to temp file
        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let c_path = temp_dir.path().join("spec.c");
        std::fs::write(&c_path, &c_code).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write C file: {}", e))
        })?;

        self.verify_c_file(&c_path).await
    }

    async fn health_check(&self) -> HealthStatus {
        let cbmc_cmd = self
            .config
            .cbmc_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("cbmc"));

        match Command::new(&cbmc_cmd).arg("--version").output().await {
            Ok(output) if output.status.success() => HealthStatus::Healthy,
            Ok(_) => HealthStatus::Degraded {
                reason: "CBMC returned non-zero exit code".to_string(),
            },
            Err(e) => HealthStatus::Unavailable {
                reason: format!("CBMC not found: {}", e),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cbmc_config_defaults() {
        let config = CbmcConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert_eq!(config.unwind, Some(10));
        assert!(config.bounds_check);
        assert!(config.pointer_check);
    }

    #[test]
    fn test_cbmc_config_builder() {
        let config = CbmcConfig::default()
            .with_timeout(Duration::from_secs(600))
            .with_unwind(20)
            .with_bounds_check(false);

        assert_eq!(config.timeout, Duration::from_secs(600));
        assert_eq!(config.unwind, Some(20));
        assert!(!config.bounds_check);
    }

    #[test]
    fn test_parse_cbmc_output_success() {
        let backend = CbmcBackend::new();
        let output = r#"
CBMC version 5.95.1
Parsing spec.c
VERIFICATION SUCCESSFUL
2 properties checked, all satisfied
        "#;

        let (status, errors, _stats) = backend.parse_cbmc_output(output, 0);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(errors.is_empty());
    }

    #[test]
    fn test_parse_cbmc_output_failure() {
        let backend = CbmcBackend::new();
        let output = r#"
CBMC version 5.95.1
Parsing spec.c
** Results:
[main.assertion.1] property main.assertion.1: FAILURE
VERIFICATION FAILED
        "#;

        let (status, errors, _stats) = backend.parse_cbmc_output(output, 10);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_generate_c_code() {
        use dashprove_usl::parse;
        use dashprove_usl::typecheck::typecheck;

        let spec = parse("invariant test { true }").unwrap();
        let typed = typecheck(spec).unwrap();

        let backend = CbmcBackend::new();
        let c_code = backend.generate_c_code(&typed);

        assert!(c_code.contains("Generated by DashProve"));
        assert!(c_code.contains("assert"));
        assert!(c_code.contains("main"));
    }

    #[tokio::test]
    async fn test_health_check_unavailable() {
        let config = CbmcConfig::default().with_cbmc_path(PathBuf::from("/nonexistent/cbmc"));
        let backend = CbmcBackend::with_config(config);

        let health = backend.health_check().await;
        assert!(matches!(health, HealthStatus::Unavailable { .. }));
    }

    #[test]
    fn test_backend_id() {
        let backend = CbmcBackend::new();
        assert_eq!(backend.id(), BackendId::CBMC);
    }
}
