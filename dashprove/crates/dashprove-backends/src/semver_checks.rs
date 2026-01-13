//! SemverChecks backend for API compatibility verification
//!
//! This backend runs cargo-semver-checks to detect breaking API changes between
//! versions. It wraps the `dashprove-static` crate's SemverChecks support.

use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
    VerificationStatus,
};
use async_trait::async_trait;
use dashprove_static::{AnalysisConfig, AnalysisTool, Severity, StaticAnalysisBackend};
use dashprove_usl::typecheck::TypedSpec;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Configuration for SemverChecks backend
#[derive(Debug, Clone)]
pub struct SemverChecksConfig {
    /// Path to the crate to analyze
    pub crate_path: Option<PathBuf>,
    /// Baseline version to compare against (defaults to crates.io version)
    pub baseline: Option<String>,
    /// Timeout for analysis
    pub timeout: Duration,
    /// Enable all features when checking
    pub all_features: bool,
}

impl Default for SemverChecksConfig {
    fn default() -> Self {
        Self {
            crate_path: None,
            baseline: None,
            timeout: Duration::from_secs(300),
            all_features: false,
        }
    }
}

impl SemverChecksConfig {
    /// Set the crate path to analyze
    pub fn with_crate_path(mut self, path: PathBuf) -> Self {
        self.crate_path = Some(path);
        self
    }

    /// Set baseline version
    pub fn with_baseline(mut self, version: String) -> Self {
        self.baseline = Some(version);
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
}

/// SemverChecks verification backend for API compatibility
///
/// cargo-semver-checks detects API changes that would be semver-incompatible:
/// - Removed public items
/// - Changed function signatures
/// - Changed type definitions
/// - Changed trait implementations
///
/// # Requirements
///
/// Install cargo-semver-checks:
/// ```bash
/// cargo install cargo-semver-checks
/// ```
pub struct SemverChecksBackend {
    config: SemverChecksConfig,
}

impl SemverChecksBackend {
    /// Create a new SemverChecks backend with default configuration
    pub fn new() -> Self {
        Self {
            config: SemverChecksConfig::default(),
        }
    }

    /// Create a new SemverChecks backend with custom configuration
    pub fn with_config(config: SemverChecksConfig) -> Self {
        Self { config }
    }

    /// Run SemverChecks on a crate path
    pub async fn analyze_crate(&self, crate_path: &Path) -> Result<BackendResult, BackendError> {
        let analysis_config = AnalysisConfig {
            warnings_as_errors: true, // Breaking changes are errors
            all_features: self.config.all_features,
            features: Vec::new(),
            targets: Vec::new(),
            timeout: self.config.timeout,
        };

        let backend = StaticAnalysisBackend::new(AnalysisTool::SemverChecks, analysis_config);

        // Check if tool is installed
        match backend.check_installed().await {
            Ok(true) => {}
            Ok(false) => {
                return Err(BackendError::Unavailable(
                    "cargo-semver-checks not installed. Run: cargo install cargo-semver-checks"
                        .to_string(),
                ));
            }
            Err(e) => {
                return Err(BackendError::Unavailable(format!(
                    "Failed to check cargo-semver-checks installation: {}",
                    e
                )));
            }
        }

        // Run analysis
        let result = backend
            .run_on_crate(crate_path)
            .await
            .map_err(|e| BackendError::VerificationFailed(e.to_string()))?;

        // Convert to BackendResult
        let status = if result.passed {
            VerificationStatus::Proven
        } else if result.error_count > 0 {
            VerificationStatus::Disproven
        } else {
            VerificationStatus::Partial {
                verified_percentage: 100.0 * (1.0 - result.warning_count as f64 / 10.0),
            }
        };

        // Build diagnostic messages
        let mut diagnostics = Vec::new();

        // Summary
        let summary = if result.passed {
            "SemverChecks: No breaking API changes detected".to_string()
        } else {
            format!(
                "SemverChecks: {} breaking changes found",
                result.error_count
            )
        };
        diagnostics.push(summary);

        for finding in &result.findings {
            let location = match (&finding.file, finding.line) {
                (Some(file), Some(line)) => format!("{}:{}", file, line),
                (Some(file), None) => file.clone(),
                _ => "unknown".to_string(),
            };

            let severity_str = match finding.severity {
                Severity::Error => "BREAKING",
                Severity::Warning => "WARNING",
                Severity::Info => "INFO",
                Severity::Help => "HELP",
            };

            diagnostics.push(format!(
                "[{}] {}: {} at {}",
                severity_str, finding.code, finding.message, location
            ));
        }

        Ok(BackendResult {
            backend: BackendId::SemverChecks,
            status,
            proof: None,
            counterexample: None,
            diagnostics,
            time_taken: result.duration,
        })
    }
}

impl Default for SemverChecksBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for SemverChecksBackend {
    fn id(&self) -> BackendId {
        BackendId::SemverChecks
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::ApiCompatibility]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let crate_path = self.config.crate_path.clone().ok_or_else(|| {
            BackendError::Unavailable(
                "SemverChecks backend requires crate_path pointing to a Rust crate".to_string(),
            )
        })?;

        self.analyze_crate(&crate_path).await
    }

    async fn health_check(&self) -> HealthStatus {
        let backend =
            StaticAnalysisBackend::new(AnalysisTool::SemverChecks, AnalysisConfig::default());

        match backend.check_installed().await {
            Ok(true) => HealthStatus::Healthy,
            Ok(false) => HealthStatus::Unavailable {
                reason: "cargo-semver-checks not installed".to_string(),
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

    // ===== SemverChecksConfig defaults =====

    #[kani::proof]
    fn verify_config_defaults_timeout() {
        let config = SemverChecksConfig::default();
        assert!(config.timeout == Duration::from_secs(300));
    }

    #[kani::proof]
    fn verify_config_defaults_options() {
        let config = SemverChecksConfig::default();
        assert!(config.crate_path.is_none());
        assert!(config.baseline.is_none());
        assert!(!config.all_features);
    }

    // ===== SemverChecksConfig builders =====

    #[kani::proof]
    fn verify_config_with_crate_path() {
        let config = SemverChecksConfig::default().with_crate_path(PathBuf::from("/test"));
        assert!(config.crate_path.is_some());
    }

    #[kani::proof]
    fn verify_config_with_baseline() {
        let config = SemverChecksConfig::default().with_baseline("1.0.0".to_string());
        assert!(config.baseline == Some("1.0.0".to_string()));
    }

    #[kani::proof]
    fn verify_config_with_timeout() {
        let config = SemverChecksConfig::default().with_timeout(Duration::from_secs(60));
        assert!(config.timeout == Duration::from_secs(60));
    }

    #[kani::proof]
    fn verify_config_with_all_features() {
        let config = SemverChecksConfig::default().with_all_features();
        assert!(config.all_features);
    }

    // ===== Backend construction =====

    #[kani::proof]
    fn verify_backend_new_uses_defaults() {
        let backend = SemverChecksBackend::new();
        assert!(backend.config.timeout == Duration::from_secs(300));
        assert!(!backend.config.all_features);
    }

    #[kani::proof]
    fn verify_backend_default_matches_new() {
        let b1 = SemverChecksBackend::new();
        let b2 = SemverChecksBackend::default();
        assert!(b1.config.timeout == b2.config.timeout);
        assert!(b1.config.all_features == b2.config.all_features);
    }

    #[kani::proof]
    fn verify_backend_with_config_preserves_values() {
        let config = SemverChecksConfig {
            crate_path: Some(PathBuf::from("/test")),
            baseline: Some("2.0.0".to_string()),
            timeout: Duration::from_secs(60),
            all_features: true,
        };
        let backend = SemverChecksBackend::with_config(config);
        assert!(backend.config.timeout == Duration::from_secs(60));
        assert!(backend.config.baseline == Some("2.0.0".to_string()));
        assert!(backend.config.all_features);
    }

    // ===== ID and supports =====

    #[kani::proof]
    fn verify_backend_id() {
        let backend = SemverChecksBackend::new();
        assert!(matches!(backend.id(), BackendId::SemverChecks));
    }

    #[kani::proof]
    fn verify_supports_api_compatibility() {
        let backend = SemverChecksBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::ApiCompatibility));
        assert!(supported.len() == 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semver_checks_config_default() {
        let config = SemverChecksConfig::default();
        assert!(config.crate_path.is_none());
        assert!(config.baseline.is_none());
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert!(!config.all_features);
    }

    #[test]
    fn test_semver_checks_config_builder() {
        let config = SemverChecksConfig::default()
            .with_crate_path(PathBuf::from("/test/path"))
            .with_baseline("1.0.0".to_string())
            .with_timeout(Duration::from_secs(60))
            .with_all_features();

        assert_eq!(config.crate_path, Some(PathBuf::from("/test/path")));
        assert_eq!(config.baseline, Some("1.0.0".to_string()));
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert!(config.all_features);
    }

    #[test]
    fn test_semver_checks_backend_id() {
        let backend = SemverChecksBackend::new();
        assert_eq!(backend.id(), BackendId::SemverChecks);
    }

    #[test]
    fn test_semver_checks_supports_api_compatibility() {
        let backend = SemverChecksBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::ApiCompatibility));
    }

    #[tokio::test]
    async fn test_semver_checks_health_check() {
        let backend = SemverChecksBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => {}
            HealthStatus::Unavailable { reason } => {
                assert!(
                    reason.contains("semver-checks")
                        || reason.contains("SemverChecks")
                        || reason.contains("Check")
                );
            }
            HealthStatus::Degraded { .. } => {}
        }
    }

    #[tokio::test]
    async fn test_semver_checks_verify_requires_crate_path() {
        use dashprove_usl::{parse, typecheck};

        let backend = SemverChecksBackend::new();
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
