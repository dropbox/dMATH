//! Proptest backend for property-based testing
//!
//! This backend runs Proptest property-based tests on Rust code to find bugs
//! by generating random inputs that violate specified properties.
//! It wraps the `dashprove-pbt` crate's Proptest support.

use crate::counterexample::StructuredCounterexample;
use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
    VerificationStatus,
};
use async_trait::async_trait;
use dashprove_pbt::{PbtBackend as PbtRunner, PbtConfig, PbtType};
use dashprove_usl::typecheck::TypedSpec;
use std::path::PathBuf;
use std::time::Duration;

/// Configuration for Proptest backend
#[derive(Debug, Clone)]
pub struct ProptestConfig {
    /// Path to the crate to test
    pub crate_path: Option<PathBuf>,
    /// Number of test cases to generate per property
    pub cases: u32,
    /// Maximum shrink iterations when a failure is found
    pub max_shrink_iters: u32,
    /// Seed for reproducibility (None = random)
    pub seed: Option<u64>,
    /// Timeout for the entire test run
    pub timeout: Duration,
    /// Run tests in forked subprocesses
    pub fork: bool,
    /// Enable verbose output
    pub verbose: bool,
}

impl Default for ProptestConfig {
    fn default() -> Self {
        Self {
            crate_path: None,
            cases: 256,
            max_shrink_iters: 1000,
            seed: None,
            timeout: Duration::from_secs(300),
            fork: false,
            verbose: false,
        }
    }
}

impl ProptestConfig {
    /// Set the crate path
    pub fn with_crate_path(mut self, path: PathBuf) -> Self {
        self.crate_path = Some(path);
        self
    }

    /// Set number of test cases
    pub fn with_cases(mut self, cases: u32) -> Self {
        self.cases = cases;
        self
    }

    /// Set maximum shrink iterations
    pub fn with_max_shrink_iters(mut self, iters: u32) -> Self {
        self.max_shrink_iters = iters;
        self
    }

    /// Set seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Enable fork mode
    pub fn with_fork(mut self, fork: bool) -> Self {
        self.fork = fork;
        self
    }

    /// Enable verbose output
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Convert to underlying PbtConfig
    fn to_pbt_config(&self) -> PbtConfig {
        let mut config = PbtConfig::default()
            .with_cases(self.cases)
            .with_max_shrink_iters(self.max_shrink_iters)
            .with_timeout(self.timeout)
            .with_fork(self.fork)
            .with_verbose(self.verbose);

        if let Some(seed) = self.seed {
            config = config.with_seed(seed);
        }

        config
    }
}

/// Proptest verification backend for property-based testing
///
/// Proptest generates random inputs based on strategies and checks that
/// properties hold for all generated inputs. When a failure is found,
/// it automatically shrinks the input to find the minimal failing case.
///
/// # Usage
///
/// Proptest tests are defined in the target crate using the `proptest!` macro:
///
/// ```rust,ignore
/// use proptest::prelude::*;
///
/// proptest! {
///     #[test]
///     fn test_addition_commutative(a in any::<i32>(), b in any::<i32>()) {
///         prop_assert_eq!(a.wrapping_add(b), b.wrapping_add(a));
///     }
/// }
/// ```
pub struct ProptestBackend {
    config: ProptestConfig,
}

impl ProptestBackend {
    /// Create a new Proptest backend with default configuration
    pub fn new() -> Self {
        Self {
            config: ProptestConfig::default(),
        }
    }

    /// Create a new Proptest backend with custom configuration
    pub fn with_config(config: ProptestConfig) -> Self {
        Self { config }
    }

    /// Run Proptest tests on a crate
    pub async fn run_tests(
        &self,
        crate_path: &std::path::Path,
    ) -> Result<BackendResult, BackendError> {
        let pbt_config = self.config.to_pbt_config();
        let runner = PbtRunner::new(PbtType::Proptest, pbt_config);

        let result = runner.run_on_crate(crate_path).await.map_err(|e| {
            BackendError::VerificationFailed(format!("Proptest execution failed: {}", e))
        })?;

        // Convert PbtResult to BackendResult
        let status = if result.passed {
            VerificationStatus::Proven
        } else if !result.failures.is_empty() {
            VerificationStatus::Disproven
        } else {
            VerificationStatus::Unknown {
                reason: "Tests failed but no specific failures identified".to_string(),
            }
        };

        // Build diagnostics
        let mut diagnostics = Vec::new();

        // Summary
        let summary = if result.passed {
            format!(
                "Proptest: All {} tests passed ({} cases generated)",
                result.tests_run, result.cases_generated
            )
        } else {
            format!(
                "Proptest: {} failures in {} tests ({} cases generated)",
                result.failures.len(),
                result.tests_run,
                result.cases_generated
            )
        };
        diagnostics.push(summary);

        // Add failure details
        for failure in &result.failures {
            diagnostics.push(format!("  FAILED: {}", failure.test_name));
            if !failure.failing_input.is_empty() {
                diagnostics.push(format!(
                    "    Minimal input: {} (after {} shrinks)",
                    failure.failing_input, failure.shrink_steps
                ));
            }
            if let Some(ref seed) = failure.seed {
                diagnostics.push(format!("    Reproduce with seed: {}", seed));
            }
        }

        // Build counterexample if failure found
        let counterexample = if !result.failures.is_empty() {
            let ce = result
                .failures
                .iter()
                .map(|f| {
                    format!(
                        "Test: {}\nInput: {}\nError: {}",
                        f.test_name, f.failing_input, f.error_message
                    )
                })
                .collect::<Vec<_>>()
                .join("\n\n");
            Some(StructuredCounterexample::from_raw(ce))
        } else {
            None
        };

        Ok(BackendResult {
            backend: BackendId::Proptest,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: result.duration,
        })
    }
}

impl Default for ProptestBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for ProptestBackend {
    fn id(&self) -> BackendId {
        BackendId::Proptest
    }

    fn supports(&self) -> Vec<PropertyType> {
        // Proptest verifies properties through property-based testing
        vec![PropertyType::Invariant, PropertyType::PropertyBased]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let crate_path = self.config.crate_path.clone().ok_or_else(|| {
            BackendError::Unavailable(
                "Proptest backend requires crate_path pointing to a Rust crate with proptest tests"
                    .to_string(),
            )
        })?;

        self.run_tests(&crate_path).await
    }

    async fn health_check(&self) -> HealthStatus {
        // Check if cargo is available (proptest is a crate dependency)
        match which::which("cargo") {
            Ok(_) => HealthStatus::Healthy,
            Err(_) => HealthStatus::Unavailable {
                reason: "cargo not found. Install Rust toolchain.".to_string(),
            },
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ===== ProptestConfig defaults =====

    #[kani::proof]
    fn verify_config_defaults_cases() {
        let config = ProptestConfig::default();
        assert!(config.cases == 256);
    }

    #[kani::proof]
    fn verify_config_defaults_shrink_iters() {
        let config = ProptestConfig::default();
        assert!(config.max_shrink_iters == 1000);
    }

    #[kani::proof]
    fn verify_config_defaults_timeout() {
        let config = ProptestConfig::default();
        assert!(config.timeout == Duration::from_secs(300));
    }

    #[kani::proof]
    fn verify_config_defaults_options() {
        let config = ProptestConfig::default();
        assert!(config.crate_path.is_none());
        assert!(config.seed.is_none());
        assert!(!config.fork);
        assert!(!config.verbose);
    }

    // ===== ProptestConfig builders =====

    #[kani::proof]
    fn verify_config_with_cases() {
        let config = ProptestConfig::default().with_cases(500);
        assert!(config.cases == 500);
        // Other defaults preserved
        assert!(config.max_shrink_iters == 1000);
    }

    #[kani::proof]
    fn verify_config_with_max_shrink_iters() {
        let config = ProptestConfig::default().with_max_shrink_iters(2000);
        assert!(config.max_shrink_iters == 2000);
        // Other defaults preserved
        assert!(config.cases == 256);
    }

    #[kani::proof]
    fn verify_config_with_timeout() {
        let config = ProptestConfig::default().with_timeout(Duration::from_secs(60));
        assert!(config.timeout == Duration::from_secs(60));
    }

    #[kani::proof]
    fn verify_config_with_seed() {
        let config = ProptestConfig::default().with_seed(42);
        assert!(config.seed == Some(42));
    }

    #[kani::proof]
    fn verify_config_with_fork() {
        let config = ProptestConfig::default().with_fork(true);
        assert!(config.fork);
    }

    #[kani::proof]
    fn verify_config_with_verbose() {
        let config = ProptestConfig::default().with_verbose(true);
        assert!(config.verbose);
    }

    #[kani::proof]
    fn verify_config_with_crate_path() {
        let config = ProptestConfig::default().with_crate_path(PathBuf::from("/test"));
        assert!(config.crate_path.is_some());
    }

    // ===== Backend construction =====

    #[kani::proof]
    fn verify_backend_new_uses_defaults() {
        let backend = ProptestBackend::new();
        assert!(backend.config.cases == 256);
        assert!(backend.config.max_shrink_iters == 1000);
    }

    #[kani::proof]
    fn verify_backend_default_matches_new() {
        let b1 = ProptestBackend::new();
        let b2 = ProptestBackend::default();
        assert!(b1.config.cases == b2.config.cases);
        assert!(b1.config.max_shrink_iters == b2.config.max_shrink_iters);
        assert!(b1.config.timeout == b2.config.timeout);
    }

    #[kani::proof]
    fn verify_backend_with_config_preserves_values() {
        let config = ProptestConfig {
            crate_path: Some(PathBuf::from("/test")),
            cases: 512,
            max_shrink_iters: 500,
            seed: Some(123),
            timeout: Duration::from_secs(60),
            fork: true,
            verbose: true,
        };
        let backend = ProptestBackend::with_config(config);
        assert!(backend.config.cases == 512);
        assert!(backend.config.max_shrink_iters == 500);
        assert!(backend.config.seed == Some(123));
        assert!(backend.config.fork);
        assert!(backend.config.verbose);
    }

    // ===== ID and supports =====

    #[kani::proof]
    fn verify_backend_id() {
        let backend = ProptestBackend::new();
        assert!(matches!(backend.id(), BackendId::Proptest));
    }

    #[kani::proof]
    fn verify_supports_includes_invariant_and_pbt() {
        let backend = ProptestBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::Invariant));
        assert!(supported.contains(&PropertyType::PropertyBased));
        assert!(supported.len() == 2);
    }

    // ===== PbtConfig conversion =====

    #[kani::proof]
    fn verify_to_pbt_config_cases() {
        let config = ProptestConfig::default().with_cases(100);
        let pbt_config = config.to_pbt_config();
        assert!(pbt_config.cases() == 100);
    }

    #[kani::proof]
    fn verify_to_pbt_config_preserves_defaults() {
        let config = ProptestConfig::default();
        let pbt_config = config.to_pbt_config();
        assert!(pbt_config.cases() == 256);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proptest_config_default() {
        let config = ProptestConfig::default();
        assert!(config.crate_path.is_none());
        assert_eq!(config.cases, 256);
        assert_eq!(config.max_shrink_iters, 1000);
        assert!(config.seed.is_none());
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert!(!config.fork);
        assert!(!config.verbose);
    }

    #[test]
    fn test_proptest_config_builder() {
        let config = ProptestConfig::default()
            .with_crate_path(PathBuf::from("/test/path"))
            .with_cases(1000)
            .with_max_shrink_iters(500)
            .with_seed(12345)
            .with_timeout(Duration::from_secs(120))
            .with_fork(true)
            .with_verbose(true);

        assert_eq!(config.crate_path, Some(PathBuf::from("/test/path")));
        assert_eq!(config.cases, 1000);
        assert_eq!(config.max_shrink_iters, 500);
        assert_eq!(config.seed, Some(12345));
        assert_eq!(config.timeout, Duration::from_secs(120));
        assert!(config.fork);
        assert!(config.verbose);
    }

    #[test]
    fn test_proptest_backend_id() {
        let backend = ProptestBackend::new();
        assert_eq!(backend.id(), BackendId::Proptest);
    }

    #[test]
    fn test_proptest_supports() {
        let backend = ProptestBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::Invariant));
        assert!(supported.contains(&PropertyType::PropertyBased));
    }

    #[tokio::test]
    async fn test_proptest_health_check() {
        let backend = ProptestBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => {
                // Expected - cargo is available
            }
            HealthStatus::Unavailable { reason } => {
                assert!(reason.contains("cargo"));
            }
            HealthStatus::Degraded { .. } => {
                // Also acceptable
            }
        }
    }

    #[tokio::test]
    async fn test_proptest_verify_requires_crate_path() {
        use dashprove_usl::{parse, typecheck};

        let backend = ProptestBackend::new();
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
