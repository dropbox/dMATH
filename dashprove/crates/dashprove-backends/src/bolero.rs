//! Bolero backend for unified fuzzing and property-based testing
//!
//! This backend runs Bolero on Rust code for unified fuzzing and property-based
//! testing. Bolero provides a single API that can run with multiple backends
//! (libfuzzer, afl, proptest) making it ideal for comprehensive testing.

// =============================================
// Kani Proofs for Bolero Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- BoleroConfig Default Tests ----

    /// Verify BoleroConfig::default sets correct baseline values
    #[kani::proof]
    fn proof_bolero_config_defaults() {
        let config = BoleroConfig::default();
        kani::assert(
            config.crate_path.is_none(),
            "crate_path should default to None",
        );
        kani::assert(config.target.is_none(), "target should default to None");
        kani::assert(
            config.timeout == Duration::from_secs(60),
            "timeout should default to 60 seconds",
        );
        kani::assert(
            config.max_iterations == 0,
            "max_iterations should default to 0",
        );
    }

    // ---- BoleroConfig Builder Tests ----

    /// Verify with_crate_path updates crate_path
    #[kani::proof]
    fn proof_bolero_config_with_crate_path() {
        let config = BoleroConfig::default().with_crate_path(PathBuf::from("/project"));
        kani::assert(
            config.crate_path == Some(PathBuf::from("/project")),
            "with_crate_path should set crate_path",
        );
    }

    /// Verify with_target updates target
    #[kani::proof]
    fn proof_bolero_config_with_target() {
        let config = BoleroConfig::default().with_target("test_property");
        kani::assert(
            config.target == Some("test_property".to_string()),
            "with_target should set target",
        );
    }

    /// Verify with_timeout updates timeout
    #[kani::proof]
    fn proof_bolero_config_with_timeout() {
        let config = BoleroConfig::default().with_timeout(Duration::from_secs(120));
        kani::assert(
            config.timeout == Duration::from_secs(120),
            "with_timeout should set timeout",
        );
    }

    /// Verify with_max_iterations updates max_iterations
    #[kani::proof]
    fn proof_bolero_config_with_max_iterations() {
        let config = BoleroConfig::default().with_max_iterations(10000);
        kani::assert(
            config.max_iterations == 10000,
            "with_max_iterations should set max_iterations",
        );
    }

    /// Verify builder chaining works correctly
    #[kani::proof]
    fn proof_bolero_config_builder_chain() {
        let config = BoleroConfig::default()
            .with_crate_path(PathBuf::from("/crate"))
            .with_target("fuzz_test")
            .with_timeout(Duration::from_secs(300))
            .with_max_iterations(50000);

        kani::assert(
            config.crate_path == Some(PathBuf::from("/crate")),
            "crate_path should be set after chain",
        );
        kani::assert(
            config.target == Some("fuzz_test".to_string()),
            "target should be set after chain",
        );
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "timeout should be set after chain",
        );
        kani::assert(
            config.max_iterations == 50000,
            "max_iterations should be set after chain",
        );
    }

    // ---- BoleroBackend Construction Tests ----

    /// Verify BoleroBackend::new uses default configuration
    #[kani::proof]
    fn proof_bolero_backend_new_defaults() {
        let backend = BoleroBackend::new();
        kani::assert(
            backend.config.crate_path.is_none(),
            "new backend should have no crate_path",
        );
        kani::assert(
            backend.config.target.is_none(),
            "new backend should have no target",
        );
        kani::assert(
            backend.config.timeout == Duration::from_secs(60),
            "new backend should default timeout to 60 seconds",
        );
        kani::assert(
            backend.config.max_iterations == 0,
            "new backend should default max_iterations to 0",
        );
    }

    /// Verify BoleroBackend::default equals BoleroBackend::new
    #[kani::proof]
    fn proof_bolero_backend_default_equals_new() {
        let default_backend = BoleroBackend::default();
        let new_backend = BoleroBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "default and new should share timeout",
        );
        kani::assert(
            default_backend.config.max_iterations == new_backend.config.max_iterations,
            "default and new should share max_iterations",
        );
    }

    /// Verify BoleroBackend::with_config preserves custom configuration
    #[kani::proof]
    fn proof_bolero_backend_with_config() {
        let config = BoleroConfig {
            crate_path: Some(PathBuf::from("/work/project")),
            target: Some("fuzz_parser".to_string()),
            timeout: Duration::from_secs(600),
            max_iterations: 100000,
        };
        let backend = BoleroBackend::with_config(config);
        kani::assert(
            backend.config.crate_path == Some(PathBuf::from("/work/project")),
            "with_config should preserve crate_path",
        );
        kani::assert(
            backend.config.target == Some("fuzz_parser".to_string()),
            "with_config should preserve target",
        );
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "with_config should preserve timeout",
        );
        kani::assert(
            backend.config.max_iterations == 100000,
            "with_config should preserve max_iterations",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify id() returns BackendId::Bolero
    #[kani::proof]
    fn proof_bolero_backend_id() {
        let backend = BoleroBackend::new();
        kani::assert(
            backend.id() == BackendId::Bolero,
            "BoleroBackend id should be BackendId::Bolero",
        );
    }

    /// Verify supports() includes PropertyBased
    #[kani::proof]
    fn proof_bolero_backend_supports() {
        let backend = BoleroBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::PropertyBased),
            "supports should include PropertyBased",
        );
    }

    /// Verify supports() returns exactly one property type
    #[kani::proof]
    fn proof_bolero_backend_supports_length() {
        let backend = BoleroBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 1,
            "Bolero should support exactly one property type",
        );
    }
}

use crate::counterexample::StructuredCounterexample;
use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
    VerificationStatus,
};
use async_trait::async_trait;
use dashprove_fuzz::{FuzzBackend as FuzzRunner, FuzzConfig, FuzzerType};
use dashprove_usl::typecheck::TypedSpec;
use std::path::PathBuf;
use std::time::Duration;

/// Configuration for Bolero backend
#[derive(Debug, Clone)]
pub struct BoleroConfig {
    /// Path to the crate to test
    pub crate_path: Option<PathBuf>,
    /// Name of the test target
    pub target: Option<String>,
    /// Timeout for testing
    pub timeout: Duration,
    /// Maximum number of iterations
    pub max_iterations: u64,
}

impl Default for BoleroConfig {
    fn default() -> Self {
        Self {
            crate_path: None,
            target: None,
            timeout: Duration::from_secs(60),
            max_iterations: 0, // Run until timeout
        }
    }
}

impl BoleroConfig {
    /// Set the crate path
    pub fn with_crate_path(mut self, path: PathBuf) -> Self {
        self.crate_path = Some(path);
        self
    }

    /// Set the test target name
    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.target = Some(target.into());
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set maximum iterations
    pub fn with_max_iterations(mut self, iterations: u64) -> Self {
        self.max_iterations = iterations;
        self
    }
}

/// Bolero verification backend for unified fuzzing and property-based testing
///
/// Bolero is a unified fuzzing and property-based testing framework that allows
/// writing tests once and running them with multiple backends:
/// - LibFuzzer for coverage-guided fuzzing
/// - AFL for mutation-based fuzzing
/// - Proptest for property-based testing
///
/// This makes it ideal for comprehensive testing as the same test can be run
/// with different engines to find different kinds of bugs.
///
/// # Requirements
///
/// - `cargo-bolero` installed (`cargo install cargo-bolero`)
/// - Bolero test targets defined with `#[test]` and `bolero::check!`
pub struct BoleroBackend {
    config: BoleroConfig,
}

impl BoleroBackend {
    /// Create a new Bolero backend with default configuration
    pub fn new() -> Self {
        Self {
            config: BoleroConfig::default(),
        }
    }

    /// Create a new Bolero backend with custom configuration
    pub fn with_config(config: BoleroConfig) -> Self {
        Self { config }
    }

    /// Run Bolero on a crate target
    pub async fn run_tests(
        &self,
        crate_path: &std::path::Path,
        target: &str,
    ) -> Result<BackendResult, BackendError> {
        let fuzz_config = FuzzConfig::default()
            .with_timeout(self.config.timeout)
            .with_max_iterations(self.config.max_iterations);

        let runner = FuzzRunner::new(FuzzerType::Bolero, fuzz_config);

        // Check if bolero is installed first
        match runner.check_installed().await {
            Ok(true) => {}
            Ok(false) => {
                return Err(BackendError::Unavailable(
                    "cargo-bolero not installed. Run: cargo install cargo-bolero".to_string(),
                ));
            }
            Err(e) => {
                return Err(BackendError::Unavailable(format!(
                    "Failed to check cargo-bolero: {}",
                    e
                )));
            }
        }

        let result = runner
            .run_on_target(crate_path, target)
            .await
            .map_err(|e| match e {
                dashprove_fuzz::FuzzError::NotInstalled(name, _) => {
                    BackendError::Unavailable(format!("{} not installed", name))
                }
                dashprove_fuzz::FuzzError::NightlyRequired(name) => {
                    BackendError::Unavailable(format!("Nightly toolchain required for {}", name))
                }
                dashprove_fuzz::FuzzError::TargetNotFound(target) => {
                    BackendError::VerificationFailed(format!("Bolero target not found: {}", target))
                }
                dashprove_fuzz::FuzzError::BuildFailed(msg) => {
                    BackendError::VerificationFailed(format!("Build failed: {}", msg))
                }
                dashprove_fuzz::FuzzError::FuzzingFailed(msg) => {
                    BackendError::VerificationFailed(format!("Bolero test failed: {}", msg))
                }
                dashprove_fuzz::FuzzError::Timeout(d) => BackendError::Timeout(d),
                dashprove_fuzz::FuzzError::Io(e) => {
                    BackendError::VerificationFailed(format!("IO error: {}", e))
                }
            })?;

        // Convert FuzzResult to BackendResult
        let status = if result.passed {
            VerificationStatus::Proven
        } else if !result.crashes.is_empty() {
            VerificationStatus::Disproven
        } else {
            VerificationStatus::Unknown {
                reason: "Bolero testing completed but status unclear".to_string(),
            }
        };

        // Build diagnostics
        let mut diagnostics = Vec::new();

        // Summary
        let summary = if result.passed {
            format!(
                "Bolero: All tests passed ({} executions, {:.1} exec/sec)",
                result.executions, result.exec_per_sec
            )
        } else {
            format!(
                "Bolero: {} failures found ({} executions)",
                result.crashes.len(),
                result.executions
            )
        };
        diagnostics.push(summary);

        // Coverage info
        if result.coverage.regions_covered > 0 {
            diagnostics.push(format!(
                "  Coverage: {} regions covered",
                result.coverage.regions_covered
            ));
        }

        // Add failure details
        for crash in &result.crashes {
            diagnostics.push(format!("  [FAILURE] {:?}", crash.crash_type));
            if let Some(ref trace) = crash.stack_trace {
                for line in trace.lines().take(5) {
                    diagnostics.push(format!("    {}", line));
                }
            }
            if let Some(ref artifact) = crash.artifact_path {
                diagnostics.push(format!("  Artifact: {}", artifact.display()));
            }
        }

        // Build counterexample if failures found
        let counterexample = if !result.crashes.is_empty() {
            let failure_info = result
                .crashes
                .iter()
                .map(|c| {
                    format!(
                        "{:?}: {}",
                        c.crash_type,
                        c.stack_trace.as_deref().unwrap_or("no stack trace")
                    )
                })
                .collect::<Vec<_>>()
                .join("\n---\n");
            Some(StructuredCounterexample::from_raw(failure_info))
        } else {
            None
        };

        Ok(BackendResult {
            backend: BackendId::Bolero,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: result.duration,
        })
    }
}

impl Default for BoleroBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for BoleroBackend {
    fn id(&self) -> BackendId {
        BackendId::Bolero
    }

    fn supports(&self) -> Vec<PropertyType> {
        // Bolero is designed for property-based testing
        vec![PropertyType::PropertyBased]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let crate_path = self.config.crate_path.clone().ok_or_else(|| {
            BackendError::Unavailable(
                "Bolero backend requires crate_path pointing to a Rust crate with bolero tests"
                    .to_string(),
            )
        })?;

        let target = self.config.target.clone().ok_or_else(|| {
            BackendError::Unavailable(
                "Bolero backend requires a target name specifying the test to run".to_string(),
            )
        })?;

        self.run_tests(&crate_path, &target).await
    }

    async fn health_check(&self) -> HealthStatus {
        // Check if cargo-bolero is installed
        let fuzz_config = FuzzConfig::default();
        let runner = FuzzRunner::new(FuzzerType::Bolero, fuzz_config);

        match runner.check_installed().await {
            Ok(true) => HealthStatus::Healthy,
            Ok(false) => HealthStatus::Unavailable {
                reason: "cargo-bolero not installed. Run: cargo install cargo-bolero".to_string(),
            },
            Err(e) => HealthStatus::Unavailable {
                reason: format!("Failed to check cargo-bolero: {}", e),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bolero_config_default() {
        let config = BoleroConfig::default();
        assert!(config.crate_path.is_none());
        assert!(config.target.is_none());
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.max_iterations, 0);
    }

    #[test]
    fn test_bolero_config_builder() {
        let config = BoleroConfig::default()
            .with_crate_path(PathBuf::from("/test/path"))
            .with_target("test_property")
            .with_timeout(Duration::from_secs(120))
            .with_max_iterations(10000);

        assert_eq!(config.crate_path, Some(PathBuf::from("/test/path")));
        assert_eq!(config.target, Some("test_property".to_string()));
        assert_eq!(config.timeout, Duration::from_secs(120));
        assert_eq!(config.max_iterations, 10000);
    }

    #[test]
    fn test_bolero_backend_id() {
        let backend = BoleroBackend::new();
        assert_eq!(backend.id(), BackendId::Bolero);
    }

    #[test]
    fn test_bolero_supports_property_based() {
        let backend = BoleroBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::PropertyBased));
    }

    #[tokio::test]
    async fn test_bolero_health_check() {
        let backend = BoleroBackend::new();
        let health = backend.health_check().await;
        // Health check should return a valid status
        match health {
            HealthStatus::Healthy => {
                // Expected if cargo-bolero is installed
            }
            HealthStatus::Unavailable { reason } => {
                // Expected if cargo-bolero not installed
                assert!(reason.contains("bolero") || reason.contains("not installed"));
            }
            HealthStatus::Degraded { .. } => {
                // Also acceptable
            }
        }
    }

    #[tokio::test]
    async fn test_bolero_verify_requires_crate_path() {
        use dashprove_usl::{parse, typecheck};

        let backend = BoleroBackend::new();
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

    #[tokio::test]
    async fn test_bolero_verify_requires_target() {
        use dashprove_usl::{parse, typecheck};

        let config = BoleroConfig::default().with_crate_path(PathBuf::from("/test/path"));
        let backend = BoleroBackend::with_config(config);
        let spec = parse("theorem test { true }").expect("parse");
        let typed_spec = typecheck(spec).expect("typecheck");

        let result = backend.verify(&typed_spec).await;
        assert!(result.is_err());
        if let Err(BackendError::Unavailable(msg)) = result {
            assert!(msg.contains("target"));
        } else {
            panic!("Expected Unavailable error");
        }
    }
}
