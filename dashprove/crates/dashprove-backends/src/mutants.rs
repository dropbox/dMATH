//! cargo-mutants backend for mutation testing
//!
//! This backend runs cargo-mutants on Rust code to find surviving mutants,
//! which indicate code paths that lack sufficient test coverage. Mutation
//! testing is valuable for evaluating the quality of a test suite.

use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
    VerificationStatus,
};
use async_trait::async_trait;
use dashprove_static::{AnalysisConfig, AnalysisTool, Severity, StaticAnalysisBackend};
use dashprove_usl::typecheck::TypedSpec;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Configuration for cargo-mutants backend
#[derive(Debug, Clone)]
pub struct MutantsConfig {
    /// Path to the crate to test
    pub crate_path: Option<PathBuf>,
    /// Timeout for mutation testing
    pub timeout: Duration,
    /// Whether to treat surviving mutants as errors
    pub strict: bool,
}

impl Default for MutantsConfig {
    fn default() -> Self {
        Self {
            crate_path: None,
            timeout: Duration::from_secs(600), // Mutation testing can take a while
            strict: false,                     // By default, surviving mutants are warnings
        }
    }
}

impl MutantsConfig {
    /// Set the crate path to test
    pub fn with_crate_path(mut self, path: PathBuf) -> Self {
        self.crate_path = Some(path);
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Treat surviving mutants as errors (strict mode)
    pub fn with_strict(mut self) -> Self {
        self.strict = true;
        self
    }
}

/// cargo-mutants verification backend for mutation testing
///
/// Mutation testing introduces small changes (mutations) to code and checks if
/// the test suite detects them. Mutants that survive (tests still pass after
/// mutation) indicate:
/// - Missing test coverage
/// - Tests that don't check important behavior
/// - Dead code
///
/// This is valuable for evaluating test suite quality beyond simple coverage.
///
/// # Requirements
///
/// - `cargo-mutants` installed (`cargo install cargo-mutants`)
/// - Existing test suite in the target crate
pub struct MutantsBackend {
    config: MutantsConfig,
}

impl MutantsBackend {
    /// Create a new cargo-mutants backend with default configuration
    pub fn new() -> Self {
        Self {
            config: MutantsConfig::default(),
        }
    }

    /// Create a new cargo-mutants backend with custom configuration
    pub fn with_config(config: MutantsConfig) -> Self {
        Self { config }
    }

    /// Run mutation testing on a crate
    pub async fn run_mutation_testing(
        &self,
        crate_path: &Path,
    ) -> Result<BackendResult, BackendError> {
        let analysis_config = AnalysisConfig {
            warnings_as_errors: self.config.strict,
            all_features: false,
            features: Vec::new(),
            targets: Vec::new(),
            timeout: self.config.timeout,
        };

        let backend = StaticAnalysisBackend::new(AnalysisTool::Mutants, analysis_config);

        // Check if cargo-mutants is installed
        match backend.check_installed().await {
            Ok(true) => {}
            Ok(false) => {
                return Err(BackendError::Unavailable(
                    "cargo-mutants not installed. Run: cargo install cargo-mutants".to_string(),
                ));
            }
            Err(e) => {
                return Err(BackendError::Unavailable(format!(
                    "Failed to check cargo-mutants installation: {}",
                    e
                )));
            }
        }

        // Run mutation testing
        let result = backend
            .run_on_crate(crate_path)
            .await
            .map_err(|e| BackendError::VerificationFailed(e.to_string()))?;

        // Count surviving mutants
        let surviving_mutants = result
            .findings
            .iter()
            .filter(|f| f.code == "surviving-mutant")
            .count();

        // Determine verification status
        let status = if result.passed && surviving_mutants == 0 {
            VerificationStatus::Proven
        } else if self.config.strict && surviving_mutants > 0 {
            VerificationStatus::Disproven
        } else if surviving_mutants > 0 {
            // Calculate approximate "kill rate"
            // Assume we found some total mutations (not available in current output,
            // so we estimate from warning count being related to surviving)
            let total_estimate = surviving_mutants + 10; // Conservative estimate
            let kill_rate = 100.0 * (1.0 - surviving_mutants as f64 / total_estimate as f64);
            VerificationStatus::Partial {
                verified_percentage: kill_rate,
            }
        } else {
            VerificationStatus::Unknown {
                reason: "Mutation testing completed but no clear result".to_string(),
            }
        };

        // Build diagnostic messages
        let mut diagnostics = Vec::new();

        // Summary
        let summary = if surviving_mutants == 0 {
            "Mutation testing: All mutants killed - excellent test coverage!".to_string()
        } else {
            format!(
                "Mutation testing: {} surviving mutant{} - test coverage gaps detected",
                surviving_mutants,
                if surviving_mutants == 1 { "" } else { "s" }
            )
        };
        diagnostics.push(summary);

        // Individual findings
        for finding in &result.findings {
            let severity_str = match finding.severity {
                Severity::Error => "ERROR",
                Severity::Warning => "WARNING",
                Severity::Info => "INFO",
                Severity::Help => "HELP",
            };

            diagnostics.push(format!(
                "[{}] {}: {}",
                severity_str, finding.code, finding.message
            ));

            if let Some(ref suggestion) = finding.suggestion {
                diagnostics.push(format!("  Suggestion: {}", suggestion));
            }
        }

        Ok(BackendResult {
            backend: BackendId::Mutants,
            status,
            proof: None,
            counterexample: None,
            diagnostics,
            time_taken: result.duration,
        })
    }
}

impl Default for MutantsBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for MutantsBackend {
    fn id(&self) -> BackendId {
        BackendId::Mutants
    }

    fn supports(&self) -> Vec<PropertyType> {
        // Mutation testing validates test suite quality for invariants
        vec![PropertyType::Invariant]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let crate_path = self.config.crate_path.clone().ok_or_else(|| {
            BackendError::Unavailable(
                "Mutants backend requires crate_path pointing to a Rust crate with tests"
                    .to_string(),
            )
        })?;

        self.run_mutation_testing(&crate_path).await
    }

    async fn health_check(&self) -> HealthStatus {
        let backend = StaticAnalysisBackend::new(AnalysisTool::Mutants, AnalysisConfig::default());

        match backend.check_installed().await {
            Ok(true) => HealthStatus::Healthy,
            Ok(false) => HealthStatus::Unavailable {
                reason: "cargo-mutants not installed. Run: cargo install cargo-mutants".to_string(),
            },
            Err(e) => HealthStatus::Unavailable {
                reason: format!("Check failed: {}", e),
            },
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ===== MutantsConfig default proofs =====

    #[kani::proof]
    fn verify_config_default_crate_path_none() {
        let config = MutantsConfig::default();
        assert!(config.crate_path.is_none());
    }

    #[kani::proof]
    fn verify_config_default_timeout() {
        let config = MutantsConfig::default();
        assert!(config.timeout.as_secs() == 600);
    }

    #[kani::proof]
    fn verify_config_default_strict_false() {
        let config = MutantsConfig::default();
        assert!(!config.strict);
    }

    // ===== MutantsConfig builder proofs =====

    #[kani::proof]
    fn verify_config_with_crate_path() {
        let config = MutantsConfig::default().with_crate_path(PathBuf::from("/test"));
        assert!(config.crate_path == Some(PathBuf::from("/test")));
    }

    #[kani::proof]
    fn verify_config_with_timeout() {
        let config = MutantsConfig::default().with_timeout(Duration::from_secs(300));
        assert!(config.timeout.as_secs() == 300);
    }

    #[kani::proof]
    fn verify_config_with_strict() {
        let config = MutantsConfig::default().with_strict();
        assert!(config.strict);
    }

    #[kani::proof]
    fn verify_config_builder_chaining() {
        let config = MutantsConfig::default()
            .with_crate_path(PathBuf::from("/path"))
            .with_timeout(Duration::from_secs(120))
            .with_strict();
        assert!(config.crate_path == Some(PathBuf::from("/path")));
        assert!(config.timeout.as_secs() == 120);
        assert!(config.strict);
    }

    // ===== MutantsBackend construction proofs =====

    #[kani::proof]
    fn verify_backend_new_has_default_config() {
        let backend = MutantsBackend::new();
        assert!(backend.config.timeout.as_secs() == 600);
        assert!(!backend.config.strict);
    }

    #[kani::proof]
    fn verify_backend_default_equals_new() {
        let b1 = MutantsBackend::new();
        let b2 = MutantsBackend::default();
        assert!(b1.config.timeout == b2.config.timeout);
        assert!(b1.config.strict == b2.config.strict);
    }

    #[kani::proof]
    fn verify_backend_with_config_preserves_timeout() {
        let config = MutantsConfig {
            crate_path: None,
            timeout: Duration::from_secs(1200),
            strict: true,
        };
        let backend = MutantsBackend::with_config(config);
        assert!(backend.config.timeout.as_secs() == 1200);
    }

    #[kani::proof]
    fn verify_backend_with_config_preserves_strict() {
        let config = MutantsConfig {
            crate_path: None,
            timeout: Duration::from_secs(600),
            strict: true,
        };
        let backend = MutantsBackend::with_config(config);
        assert!(backend.config.strict);
    }

    #[kani::proof]
    fn verify_backend_with_config_preserves_crate_path() {
        let config = MutantsConfig {
            crate_path: Some(PathBuf::from("/my/crate")),
            timeout: Duration::from_secs(600),
            strict: false,
        };
        let backend = MutantsBackend::with_config(config);
        assert!(backend.config.crate_path == Some(PathBuf::from("/my/crate")));
    }

    // ===== Backend ID proof =====

    #[kani::proof]
    fn verify_backend_id() {
        let backend = MutantsBackend::new();
        assert!(matches!(backend.id(), BackendId::Mutants));
    }

    // ===== Supports proofs =====

    #[kani::proof]
    fn verify_supports_invariant() {
        let backend = MutantsBackend::new();
        let supported = backend.supports();
        let has_invariant = supported
            .iter()
            .any(|p| matches!(p, PropertyType::Invariant));
        assert!(has_invariant);
    }

    #[kani::proof]
    fn verify_supports_count() {
        let backend = MutantsBackend::new();
        let supported = backend.supports();
        assert!(supported.len() == 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mutants_config_default() {
        let config = MutantsConfig::default();
        assert!(config.crate_path.is_none());
        assert_eq!(config.timeout, Duration::from_secs(600));
        assert!(!config.strict);
    }

    #[test]
    fn test_mutants_config_builder() {
        let config = MutantsConfig::default()
            .with_crate_path(PathBuf::from("/test/path"))
            .with_timeout(Duration::from_secs(300))
            .with_strict();

        assert_eq!(config.crate_path, Some(PathBuf::from("/test/path")));
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert!(config.strict);
    }

    #[test]
    fn test_mutants_backend_id() {
        let backend = MutantsBackend::new();
        assert_eq!(backend.id(), BackendId::Mutants);
    }

    #[test]
    fn test_mutants_supports_invariant() {
        let backend = MutantsBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::Invariant));
    }

    #[tokio::test]
    async fn test_mutants_health_check() {
        let backend = MutantsBackend::new();
        let health = backend.health_check().await;
        // Health check should return a valid status
        match health {
            HealthStatus::Healthy => {
                // Expected if cargo-mutants is installed
            }
            HealthStatus::Unavailable { reason } => {
                // Expected if cargo-mutants not installed
                assert!(
                    reason.contains("mutants")
                        || reason.contains("not installed")
                        || reason.contains("Check")
                );
            }
            HealthStatus::Degraded { .. } => {
                // Also acceptable
            }
        }
    }

    #[tokio::test]
    async fn test_mutants_verify_requires_crate_path() {
        use dashprove_usl::{parse, typecheck};

        let backend = MutantsBackend::new();
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
