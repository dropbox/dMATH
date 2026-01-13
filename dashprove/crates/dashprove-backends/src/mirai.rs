//! MIRAI backend for abstract interpretation of Rust code
//!
//! This backend runs MIRAI (Facebook's abstract interpreter for Rust) to detect
//! potential bugs and verify program invariants. MIRAI performs modular inter-procedural
//! analysis to find issues like panics, overflows, and assertion failures.

use crate::counterexample::StructuredCounterexample;
use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
    VerificationStatus,
};
use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio::process::Command;

/// Configuration for MIRAI backend
#[derive(Debug, Clone)]
pub struct MiraiConfig {
    /// Path to the crate to analyze
    pub crate_path: Option<PathBuf>,
    /// Timeout for analysis
    pub timeout: Duration,
    /// Whether to enable all features
    pub all_features: bool,
    /// Maximum number of iterations for fixed-point computation
    pub max_iterations: u32,
    /// Whether to enable MIRAI assertions checking
    pub check_assertions: bool,
    /// Whether to enable panic reachability analysis
    pub check_panics: bool,
}

impl Default for MiraiConfig {
    fn default() -> Self {
        Self {
            crate_path: None,
            timeout: Duration::from_secs(300),
            all_features: false,
            max_iterations: 100,
            check_assertions: true,
            check_panics: true,
        }
    }
}

impl MiraiConfig {
    /// Set the crate path
    pub fn with_crate_path(mut self, path: PathBuf) -> Self {
        self.crate_path = Some(path);
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Enable all features
    pub fn with_all_features(mut self) -> Self {
        self.all_features = true;
        self
    }

    /// Set maximum iterations
    pub fn with_max_iterations(mut self, iterations: u32) -> Self {
        self.max_iterations = iterations;
        self
    }

    /// Enable/disable assertion checking
    pub fn with_assertion_checking(mut self, check: bool) -> Self {
        self.check_assertions = check;
        self
    }

    /// Enable/disable panic reachability analysis
    pub fn with_panic_checking(mut self, check: bool) -> Self {
        self.check_panics = check;
        self
    }
}

/// MIRAI verification backend for abstract interpretation
///
/// MIRAI (Modular Inter-procedural Reasoning about Aliasing and Invariants) is
/// Facebook's abstract interpreter for Rust. It performs modular analysis to:
/// - Detect potential panics and assertion failures
/// - Verify preconditions and postconditions
/// - Find integer overflow/underflow
/// - Check array bounds
///
/// # Requirements
///
/// Install MIRAI:
/// ```bash
/// cargo install --git https://github.com/facebookexperimental/MIRAI mirai
/// ```
pub struct MiraiBackend {
    config: MiraiConfig,
}

impl MiraiBackend {
    /// Create a new MIRAI backend with default configuration
    pub fn new() -> Self {
        Self {
            config: MiraiConfig::default(),
        }
    }

    /// Create a new MIRAI backend with custom configuration
    pub fn with_config(config: MiraiConfig) -> Self {
        Self { config }
    }

    /// Run MIRAI on a crate
    pub async fn analyze_crate(&self, crate_path: &Path) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        // Build MIRAI command
        let mut args = vec!["mirai"];

        if self.config.all_features {
            args.push("--all-features");
        }

        // Run MIRAI
        let output = Command::new("cargo")
            .args(&args)
            .current_dir(crate_path)
            .env("MIRAI_FLAGS", self.build_mirai_flags())
            .output()
            .await
            .map_err(|e| BackendError::VerificationFailed(format!("Failed to run MIRAI: {}", e)))?;

        let duration = start.elapsed();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined = format!("{}\n{}", stdout, stderr);

        // Parse MIRAI output
        let (status, findings) = self.parse_mirai_output(&combined, output.status.success());

        // Build diagnostics
        let mut diagnostics = Vec::new();

        // Summary
        let summary = match &status {
            VerificationStatus::Proven => "MIRAI: No issues found".to_string(),
            VerificationStatus::Disproven => {
                format!("MIRAI: {} potential issues found", findings.len())
            }
            VerificationStatus::Partial {
                verified_percentage,
            } => {
                format!("MIRAI: Analysis {:.1}% complete", verified_percentage)
            }
            VerificationStatus::Unknown { reason } => format!("MIRAI: {}", reason),
        };
        diagnostics.push(summary);
        diagnostics.extend(findings.clone());

        // Build counterexample if issues found
        let counterexample = if !findings.is_empty() {
            Some(StructuredCounterexample::from_raw(combined.clone()))
        } else {
            None
        };

        Ok(BackendResult {
            backend: BackendId::Mirai,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: duration,
        })
    }

    fn build_mirai_flags(&self) -> String {
        let mut flags = Vec::new();

        flags.push(format!(
            "--max_analysis_time_for_body={}",
            self.config.timeout.as_secs()
        ));

        if self.config.check_assertions {
            flags.push("--diag=verify".to_string());
        }

        flags.join(" ")
    }

    fn parse_mirai_output(&self, output: &str, success: bool) -> (VerificationStatus, Vec<String>) {
        let mut findings = Vec::new();

        // Look for MIRAI-specific warning patterns
        // MIRAI warnings typically look like:
        // warning: possible panic
        // warning: precondition not satisfied
        // warning: postcondition might not hold
        // error: assertion failed
        for line in output.lines() {
            let is_diagnostic = line.contains("warning:") || line.contains("error:");
            let is_mirai_issue = line.contains("possible panic")
                || line.contains("precondition")
                || line.contains("postcondition")
                || line.contains("assertion")
                || line.contains("overflow")
                || line.contains("underflow")
                || line.contains("out of bounds")
                || line.contains("unreachable");

            if is_diagnostic && is_mirai_issue {
                findings.push(line.trim().to_string());
            }
        }

        let status = if findings.is_empty() && success {
            VerificationStatus::Proven
        } else if !findings.is_empty() {
            VerificationStatus::Disproven
        } else if !success {
            VerificationStatus::Unknown {
                reason: "MIRAI analysis failed".to_string(),
            }
        } else {
            VerificationStatus::Proven
        };

        (status, findings)
    }

    /// Check if MIRAI is installed
    pub async fn check_installed(&self) -> Result<bool, BackendError> {
        let output = Command::new("cargo")
            .args(["mirai", "--version"])
            .output()
            .await;

        match output {
            Ok(out) => Ok(out.status.success()),
            Err(_) => Ok(false),
        }
    }
}

impl Default for MiraiBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for MiraiBackend {
    fn id(&self) -> BackendId {
        BackendId::Mirai
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::MemorySafety, PropertyType::Contract]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let crate_path = self.config.crate_path.clone().ok_or_else(|| {
            BackendError::Unavailable(
                "MIRAI backend requires crate_path pointing to a Rust crate".to_string(),
            )
        })?;

        self.analyze_crate(&crate_path).await
    }

    async fn health_check(&self) -> HealthStatus {
        match self.check_installed().await {
            Ok(true) => HealthStatus::Healthy,
            Ok(false) => HealthStatus::Unavailable {
                reason: "MIRAI not installed. Install with: cargo install --git https://github.com/facebookexperimental/MIRAI mirai".to_string(),
            },
            Err(e) => HealthStatus::Unavailable {
                reason: format!("Failed to check MIRAI: {}", e),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mirai_config_default() {
        let config = MiraiConfig::default();
        assert!(config.crate_path.is_none());
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert!(!config.all_features);
        assert_eq!(config.max_iterations, 100);
        assert!(config.check_assertions);
        assert!(config.check_panics);
    }

    #[test]
    fn test_mirai_config_builder() {
        let config = MiraiConfig::default()
            .with_crate_path(PathBuf::from("/test/path"))
            .with_timeout(Duration::from_secs(120))
            .with_all_features()
            .with_max_iterations(50)
            .with_assertion_checking(false)
            .with_panic_checking(false);

        assert_eq!(config.crate_path, Some(PathBuf::from("/test/path")));
        assert_eq!(config.timeout, Duration::from_secs(120));
        assert!(config.all_features);
        assert_eq!(config.max_iterations, 50);
        assert!(!config.check_assertions);
        assert!(!config.check_panics);
    }

    #[test]
    fn test_mirai_backend_id() {
        let backend = MiraiBackend::new();
        assert_eq!(backend.id(), BackendId::Mirai);
    }

    #[test]
    fn test_mirai_supports_memory_safety() {
        let backend = MiraiBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::MemorySafety));
        assert!(supported.contains(&PropertyType::Contract));
    }

    #[test]
    fn test_mirai_parse_output_clean() {
        let backend = MiraiBackend::new();
        let (status, findings) = backend.parse_mirai_output("Finished analysis", true);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(findings.is_empty());
    }

    #[test]
    fn test_mirai_parse_output_with_warnings() {
        let backend = MiraiBackend::new();
        let output = "warning: possible panic in function foo\nwarning: precondition not satisfied";
        let (status, findings) = backend.parse_mirai_output(output, false);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert_eq!(findings.len(), 2);
    }

    #[tokio::test]
    async fn test_mirai_health_check() {
        let backend = MiraiBackend::new();
        let health = backend.health_check().await;
        // Should return a valid status
        match health {
            HealthStatus::Healthy => {
                // Expected if MIRAI is installed
            }
            HealthStatus::Unavailable { reason } => {
                // Expected if MIRAI not installed
                assert!(reason.contains("MIRAI") || reason.contains("mirai"));
            }
            HealthStatus::Degraded { .. } => {
                // Also acceptable
            }
        }
    }

    #[tokio::test]
    async fn test_mirai_verify_requires_crate_path() {
        use dashprove_usl::{parse, typecheck};

        let backend = MiraiBackend::new();
        let spec = parse("theorem test { true }").expect("parse");
        let typed_spec = typecheck(spec).expect("typecheck");

        let result = backend.verify(&typed_spec).await;
        assert!(result.is_err());
        if let Err(BackendError::Unavailable(msg)) = result {
            assert!(msg.contains("crate_path"));
        } else {
            panic!("Expected Unavailable error");
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ==================== MiraiConfig Default Proofs ====================

    #[kani::proof]
    fn proof_config_default_crate_path_none() {
        let config = MiraiConfig::default();
        kani::assert(
            config.crate_path.is_none(),
            "Default crate_path should be None",
        );
    }

    #[kani::proof]
    fn proof_config_default_timeout_300s() {
        let config = MiraiConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "Default timeout should be 300 seconds",
        );
    }

    #[kani::proof]
    fn proof_config_default_all_features_false() {
        let config = MiraiConfig::default();
        kani::assert(!config.all_features, "Default all_features should be false");
    }

    #[kani::proof]
    fn proof_config_default_max_iterations_100() {
        let config = MiraiConfig::default();
        kani::assert(
            config.max_iterations == 100,
            "Default max_iterations should be 100",
        );
    }

    #[kani::proof]
    fn proof_config_default_check_assertions_true() {
        let config = MiraiConfig::default();
        kani::assert(
            config.check_assertions,
            "Default check_assertions should be true",
        );
    }

    #[kani::proof]
    fn proof_config_default_check_panics_true() {
        let config = MiraiConfig::default();
        kani::assert(config.check_panics, "Default check_panics should be true");
    }

    // ==================== MiraiConfig Builder Proofs ====================

    #[kani::proof]
    fn proof_config_with_crate_path() {
        let config = MiraiConfig::default().with_crate_path(PathBuf::from("/test/path"));
        kani::assert(
            config.crate_path == Some(PathBuf::from("/test/path")),
            "with_crate_path should set path",
        );
    }

    #[kani::proof]
    fn proof_config_with_timeout() {
        let config = MiraiConfig::default().with_timeout(Duration::from_secs(120));
        kani::assert(
            config.timeout == Duration::from_secs(120),
            "with_timeout should set timeout",
        );
    }

    #[kani::proof]
    fn proof_config_with_all_features() {
        let config = MiraiConfig::default().with_all_features();
        kani::assert(
            config.all_features,
            "with_all_features should enable all_features",
        );
    }

    #[kani::proof]
    fn proof_config_with_max_iterations() {
        let iterations: u32 = kani::any();
        kani::assume(iterations > 0 && iterations < 1000);
        let config = MiraiConfig::default().with_max_iterations(iterations);
        kani::assert(
            config.max_iterations == iterations,
            "with_max_iterations should set iterations",
        );
    }

    #[kani::proof]
    fn proof_config_with_assertion_checking_false() {
        let config = MiraiConfig::default().with_assertion_checking(false);
        kani::assert(
            !config.check_assertions,
            "with_assertion_checking(false) should disable",
        );
    }

    #[kani::proof]
    fn proof_config_with_panic_checking_false() {
        let config = MiraiConfig::default().with_panic_checking(false);
        kani::assert(
            !config.check_panics,
            "with_panic_checking(false) should disable",
        );
    }

    #[kani::proof]
    fn proof_config_builder_chaining() {
        let config = MiraiConfig::default()
            .with_crate_path(PathBuf::from("/test"))
            .with_timeout(Duration::from_secs(60))
            .with_all_features()
            .with_max_iterations(50);
        kani::assert(config.crate_path.is_some(), "Should chain crate_path");
        kani::assert(
            config.timeout == Duration::from_secs(60),
            "Should chain timeout",
        );
        kani::assert(config.all_features, "Should chain all_features");
        kani::assert(config.max_iterations == 50, "Should chain max_iterations");
    }

    // ==================== MiraiBackend Construction Proofs ====================

    #[kani::proof]
    fn proof_backend_new_uses_default_config() {
        let backend = MiraiBackend::new();
        kani::assert(
            backend.config.crate_path.is_none(),
            "new() should use default config with crate_path=None",
        );
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "new() should use default timeout of 300s",
        );
    }

    #[kani::proof]
    fn proof_backend_default_equals_new() {
        let b1 = MiraiBackend::new();
        let b2 = MiraiBackend::default();
        kani::assert(
            b1.config.timeout == b2.config.timeout,
            "default() and new() should produce equal timeout",
        );
        kani::assert(
            b1.config.all_features == b2.config.all_features,
            "default() and new() should produce equal all_features",
        );
    }

    #[kani::proof]
    fn proof_backend_with_config_preserves_settings() {
        let config = MiraiConfig::default()
            .with_timeout(Duration::from_secs(120))
            .with_all_features();
        let backend = MiraiBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(120),
            "with_config should preserve timeout",
        );
        kani::assert(
            backend.config.all_features,
            "with_config should preserve all_features",
        );
    }

    // ==================== Backend Trait Implementation Proofs ====================

    #[kani::proof]
    fn proof_backend_id_is_mirai() {
        let backend = MiraiBackend::new();
        kani::assert(
            backend.id() == BackendId::Mirai,
            "Backend ID should be Mirai",
        );
    }

    #[kani::proof]
    fn proof_supports_contains_memory_safety() {
        let backend = MiraiBackend::new();
        let supported = backend.supports();
        let has_memory_safety = supported.iter().any(|p| *p == PropertyType::MemorySafety);
        kani::assert(
            has_memory_safety,
            "Should support MemorySafety property type",
        );
    }

    #[kani::proof]
    fn proof_supports_contains_contract() {
        let backend = MiraiBackend::new();
        let supported = backend.supports();
        let has_contract = supported.iter().any(|p| *p == PropertyType::Contract);
        kani::assert(has_contract, "Should support Contract property type");
    }

    #[kani::proof]
    fn proof_supports_returns_two_types() {
        let backend = MiraiBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 2,
            "Should support exactly 2 property types",
        );
    }

    // ==================== build_mirai_flags Proofs ====================

    #[kani::proof]
    fn proof_build_mirai_flags_not_empty() {
        let backend = MiraiBackend::new();
        let flags = backend.build_mirai_flags();
        kani::assert(!flags.is_empty(), "MIRAI flags should not be empty");
    }

    #[kani::proof]
    fn proof_build_mirai_flags_contains_max_analysis_time() {
        let backend = MiraiBackend::new();
        let flags = backend.build_mirai_flags();
        kani::assert(
            flags.contains("max_analysis_time"),
            "MIRAI flags should contain max_analysis_time",
        );
    }

    #[kani::proof]
    fn proof_build_mirai_flags_with_assertions() {
        let config = MiraiConfig::default().with_assertion_checking(true);
        let backend = MiraiBackend::with_config(config);
        let flags = backend.build_mirai_flags();
        kani::assert(
            flags.contains("verify"),
            "MIRAI flags should contain verify when assertions enabled",
        );
    }

    // ==================== parse_mirai_output Proofs ====================

    #[kani::proof]
    fn proof_parse_mirai_output_clean_is_proven() {
        let backend = MiraiBackend::new();
        let (status, findings) = backend.parse_mirai_output("Finished analysis", true);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "Clean output with success should be Proven",
        );
        kani::assert(findings.is_empty(), "Clean output should have no findings");
    }

    #[kani::proof]
    fn proof_parse_mirai_output_with_panic_is_disproven() {
        let backend = MiraiBackend::new();
        let output = "warning: possible panic in function foo";
        let (status, findings) = backend.parse_mirai_output(output, false);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Output with panic warning should be Disproven",
        );
        kani::assert(!findings.is_empty(), "Should have findings");
    }

    #[kani::proof]
    fn proof_parse_mirai_output_with_precondition_is_disproven() {
        let backend = MiraiBackend::new();
        let output = "warning: precondition not satisfied";
        let (status, findings) = backend.parse_mirai_output(output, false);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Output with precondition warning should be Disproven",
        );
        kani::assert(!findings.is_empty(), "Should have findings");
    }

    #[kani::proof]
    fn proof_parse_mirai_output_with_postcondition_is_disproven() {
        let backend = MiraiBackend::new();
        let output = "warning: postcondition might not hold";
        let (status, findings) = backend.parse_mirai_output(output, false);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Output with postcondition warning should be Disproven",
        );
        kani::assert(!findings.is_empty(), "Should have findings");
    }

    #[kani::proof]
    fn proof_parse_mirai_output_with_overflow_is_disproven() {
        let backend = MiraiBackend::new();
        let output = "warning: possible overflow detected";
        let (status, findings) = backend.parse_mirai_output(output, false);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Output with overflow warning should be Disproven",
        );
    }

    #[kani::proof]
    fn proof_parse_mirai_output_failure_without_findings_is_unknown() {
        let backend = MiraiBackend::new();
        let output = "compilation failed due to timeout";
        let (status, findings) = backend.parse_mirai_output(output, false);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Failure without MIRAI-specific findings should be Unknown",
        );
        kani::assert(
            findings.is_empty(),
            "Should have no MIRAI-specific findings",
        );
    }

    #[kani::proof]
    fn proof_parse_mirai_output_collects_multiple_findings() {
        let backend = MiraiBackend::new();
        let output = "warning: possible panic\nwarning: precondition violation\nwarning: overflow";
        let (_, findings) = backend.parse_mirai_output(output, false);
        kani::assert(findings.len() >= 2, "Should collect multiple findings");
    }
}
