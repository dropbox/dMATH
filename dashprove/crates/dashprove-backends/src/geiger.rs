//! Geiger backend for unsafe code auditing
//!
//! This backend runs cargo-geiger to audit unsafe code usage in a Rust crate
//! and its dependencies. It wraps the `dashprove-static` crate's Geiger support.

// =============================================
// Kani Proofs for Geiger Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- GeigerConfig Default Tests ----

    /// Verify GeigerConfig::default crate_path is None
    #[kani::proof]
    fn proof_geiger_config_default_crate_path_none() {
        let config = GeigerConfig::default();
        kani::assert(
            config.crate_path.is_none(),
            "Default crate_path should be None",
        );
    }

    /// Verify GeigerConfig::default timeout is 300 seconds
    #[kani::proof]
    fn proof_geiger_config_default_timeout() {
        let config = GeigerConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "Default timeout should be 300 seconds",
        );
    }

    /// Verify GeigerConfig::default include_dev_deps is false
    #[kani::proof]
    fn proof_geiger_config_default_dev_deps_false() {
        let config = GeigerConfig::default();
        kani::assert(
            !config.include_dev_deps,
            "Default include_dev_deps should be false",
        );
    }

    /// Verify GeigerConfig::default include_build_deps is false
    #[kani::proof]
    fn proof_geiger_config_default_build_deps_false() {
        let config = GeigerConfig::default();
        kani::assert(
            !config.include_build_deps,
            "Default include_build_deps should be false",
        );
    }

    /// Verify GeigerConfig::default unsafe_threshold is None
    #[kani::proof]
    fn proof_geiger_config_default_threshold_none() {
        let config = GeigerConfig::default();
        kani::assert(
            config.unsafe_threshold.is_none(),
            "Default unsafe_threshold should be None",
        );
    }

    // ---- GeigerConfig Builder Tests ----

    /// Verify with_crate_path sets the crate_path
    #[kani::proof]
    fn proof_geiger_config_with_crate_path() {
        let config = GeigerConfig::default().with_crate_path(PathBuf::from("/test"));
        kani::assert(
            config.crate_path.is_some(),
            "with_crate_path should set crate_path",
        );
    }

    /// Verify with_timeout sets the timeout
    #[kani::proof]
    fn proof_geiger_config_with_timeout() {
        let config = GeigerConfig::default().with_timeout(Duration::from_secs(60));
        kani::assert(
            config.timeout == Duration::from_secs(60),
            "with_timeout should set timeout to 60 seconds",
        );
    }

    /// Verify with_dev_deps enables dev deps
    #[kani::proof]
    fn proof_geiger_config_with_dev_deps() {
        let config = GeigerConfig::default().with_dev_deps();
        kani::assert(
            config.include_dev_deps,
            "with_dev_deps should enable include_dev_deps",
        );
    }

    /// Verify with_build_deps enables build deps
    #[kani::proof]
    fn proof_geiger_config_with_build_deps() {
        let config = GeigerConfig::default().with_build_deps();
        kani::assert(
            config.include_build_deps,
            "with_build_deps should enable include_build_deps",
        );
    }

    /// Verify with_unsafe_threshold sets threshold
    #[kani::proof]
    fn proof_geiger_config_with_unsafe_threshold() {
        let config = GeigerConfig::default().with_unsafe_threshold(10.0);
        kani::assert(
            config.unsafe_threshold == Some(10.0),
            "with_unsafe_threshold should set threshold to 10.0",
        );
    }

    /// Verify builder chain preserves timeout
    #[kani::proof]
    fn proof_geiger_config_builder_chain_timeout() {
        let config = GeigerConfig::default()
            .with_timeout(Duration::from_secs(120))
            .with_dev_deps()
            .with_build_deps();
        kani::assert(
            config.timeout == Duration::from_secs(120),
            "Builder chain should preserve timeout",
        );
    }

    /// Verify builder chain enables both deps
    #[kani::proof]
    fn proof_geiger_config_builder_chain_deps() {
        let config = GeigerConfig::default().with_dev_deps().with_build_deps();
        kani::assert(
            config.include_dev_deps && config.include_build_deps,
            "Builder chain should enable both deps",
        );
    }

    // ---- GeigerBackend Construction Tests ----

    /// Verify GeigerBackend::new uses default timeout
    #[kani::proof]
    fn proof_geiger_backend_new_default_timeout() {
        let backend = GeigerBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "New backend should use default timeout",
        );
    }

    /// Verify GeigerBackend::default equals GeigerBackend::new
    #[kani::proof]
    fn proof_geiger_backend_default_equals_new() {
        let default_backend = GeigerBackend::default();
        let new_backend = GeigerBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
    }

    /// Verify GeigerBackend::with_config preserves custom timeout
    #[kani::proof]
    fn proof_geiger_backend_with_config_timeout() {
        let config = GeigerConfig {
            crate_path: None,
            timeout: Duration::from_secs(600),
            include_dev_deps: false,
            include_build_deps: false,
            unsafe_threshold: None,
        };
        let backend = GeigerBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "with_config should preserve timeout",
        );
    }

    /// Verify GeigerBackend::with_config preserves dev_deps setting
    #[kani::proof]
    fn proof_geiger_backend_with_config_dev_deps() {
        let config = GeigerConfig {
            crate_path: None,
            timeout: Duration::from_secs(300),
            include_dev_deps: true,
            include_build_deps: false,
            unsafe_threshold: None,
        };
        let backend = GeigerBackend::with_config(config);
        kani::assert(
            backend.config.include_dev_deps,
            "with_config should preserve include_dev_deps",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify GeigerBackend::id returns Geiger
    #[kani::proof]
    fn proof_geiger_backend_id() {
        let backend = GeigerBackend::new();
        kani::assert(
            backend.id() == BackendId::Geiger,
            "Backend id should be Geiger",
        );
    }

    /// Verify GeigerBackend::supports includes UnsafeAudit
    #[kani::proof]
    fn proof_geiger_backend_supports_unsafe_audit() {
        let backend = GeigerBackend::new();
        let supported = backend.supports();
        let has_unsafe = supported.iter().any(|p| *p == PropertyType::UnsafeAudit);
        kani::assert(has_unsafe, "Should support UnsafeAudit property");
    }

    /// Verify GeigerBackend::supports returns exactly 1 property
    #[kani::proof]
    fn proof_geiger_backend_supports_length() {
        let backend = GeigerBackend::new();
        let supported = backend.supports();
        kani::assert(supported.len() == 1, "Should support exactly 1 property");
    }
}

use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
    VerificationStatus,
};
use async_trait::async_trait;
use dashprove_static::{AnalysisConfig, AnalysisTool, Severity, StaticAnalysisBackend};
use dashprove_usl::typecheck::TypedSpec;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Configuration for Geiger backend
#[derive(Debug, Clone)]
pub struct GeigerConfig {
    /// Path to the crate to analyze
    pub crate_path: Option<PathBuf>,
    /// Timeout for analysis
    pub timeout: Duration,
    /// Include dev dependencies
    pub include_dev_deps: bool,
    /// Include build dependencies
    pub include_build_deps: bool,
    /// Threshold for maximum allowed unsafe code percentage
    pub unsafe_threshold: Option<f64>,
}

impl Default for GeigerConfig {
    fn default() -> Self {
        Self {
            crate_path: None,
            timeout: Duration::from_secs(300),
            include_dev_deps: false,
            include_build_deps: false,
            unsafe_threshold: None,
        }
    }
}

impl GeigerConfig {
    /// Set the crate path to analyze
    pub fn with_crate_path(mut self, path: PathBuf) -> Self {
        self.crate_path = Some(path);
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Include dev dependencies in analysis
    pub fn with_dev_deps(mut self) -> Self {
        self.include_dev_deps = true;
        self
    }

    /// Include build dependencies in analysis
    pub fn with_build_deps(mut self) -> Self {
        self.include_build_deps = true;
        self
    }

    /// Set unsafe code threshold (0.0-100.0 percentage)
    pub fn with_unsafe_threshold(mut self, threshold: f64) -> Self {
        self.unsafe_threshold = Some(threshold);
        self
    }
}

/// Geiger verification backend for unsafe code auditing
///
/// cargo-geiger counts and reports unsafe code usage:
/// - Unsafe blocks
/// - Unsafe functions
/// - Unsafe traits and impls
/// - Unsafe in dependencies
///
/// # Requirements
///
/// Install cargo-geiger:
/// ```bash
/// cargo install cargo-geiger
/// ```
pub struct GeigerBackend {
    config: GeigerConfig,
}

impl GeigerBackend {
    /// Create a new Geiger backend with default configuration
    pub fn new() -> Self {
        Self {
            config: GeigerConfig::default(),
        }
    }

    /// Create a new Geiger backend with custom configuration
    pub fn with_config(config: GeigerConfig) -> Self {
        Self { config }
    }

    /// Run Geiger on a crate path
    pub async fn analyze_crate(&self, crate_path: &Path) -> Result<BackendResult, BackendError> {
        let analysis_config = AnalysisConfig {
            warnings_as_errors: false,
            all_features: false,
            features: Vec::new(),
            targets: Vec::new(),
            timeout: self.config.timeout,
        };

        let backend = StaticAnalysisBackend::new(AnalysisTool::Geiger, analysis_config);

        // Check if tool is installed
        match backend.check_installed().await {
            Ok(true) => {}
            Ok(false) => {
                return Err(BackendError::Unavailable(
                    "cargo-geiger not installed. Run: cargo install cargo-geiger".to_string(),
                ));
            }
            Err(e) => {
                return Err(BackendError::Unavailable(format!(
                    "Failed to check cargo-geiger installation: {}",
                    e
                )));
            }
        }

        // Run analysis
        let result = backend
            .run_on_crate(crate_path)
            .await
            .map_err(|e| BackendError::VerificationFailed(e.to_string()))?;

        // Calculate unsafe percentage and determine status
        let unsafe_count: usize = result
            .findings
            .iter()
            .filter(|f| f.severity == Severity::Warning || f.severity == Severity::Error)
            .count();

        let status = if let Some(threshold) = self.config.unsafe_threshold {
            let unsafe_percentage =
                (unsafe_count as f64 / result.findings.len().max(1) as f64) * 100.0;
            if unsafe_percentage <= threshold {
                VerificationStatus::Proven
            } else {
                VerificationStatus::Partial {
                    verified_percentage: 100.0 - unsafe_percentage,
                }
            }
        } else if result.passed {
            VerificationStatus::Proven
        } else {
            VerificationStatus::Partial {
                verified_percentage: if result.findings.is_empty() {
                    100.0
                } else {
                    let safe_count = result
                        .findings
                        .iter()
                        .filter(|f| f.severity == Severity::Info)
                        .count();
                    (safe_count as f64 / result.findings.len() as f64) * 100.0
                },
            }
        };

        // Build diagnostic messages
        let mut diagnostics = Vec::new();

        // Summary
        let summary = format!(
            "Geiger: {} unsafe code findings in {} crates analyzed",
            unsafe_count,
            result.findings.len()
        );
        diagnostics.push(summary);

        for finding in &result.findings {
            let severity_str = match finding.severity {
                Severity::Error => "UNSAFE",
                Severity::Warning => "UNSAFE",
                Severity::Info => "SAFE",
                Severity::Help => "INFO",
            };

            diagnostics.push(format!(
                "[{}] {}: {}",
                severity_str, finding.code, finding.message
            ));
        }

        Ok(BackendResult {
            backend: BackendId::Geiger,
            status,
            proof: None,
            counterexample: None,
            diagnostics,
            time_taken: result.duration,
        })
    }
}

impl Default for GeigerBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for GeigerBackend {
    fn id(&self) -> BackendId {
        BackendId::Geiger
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::UnsafeAudit]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let crate_path = self.config.crate_path.clone().ok_or_else(|| {
            BackendError::Unavailable(
                "Geiger backend requires crate_path pointing to a Rust crate".to_string(),
            )
        })?;

        self.analyze_crate(&crate_path).await
    }

    async fn health_check(&self) -> HealthStatus {
        let backend = StaticAnalysisBackend::new(AnalysisTool::Geiger, AnalysisConfig::default());

        match backend.check_installed().await {
            Ok(true) => HealthStatus::Healthy,
            Ok(false) => HealthStatus::Unavailable {
                reason: "cargo-geiger not installed".to_string(),
            },
            Err(e) => HealthStatus::Unavailable {
                reason: format!("Check failed: {}", e),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geiger_config_default() {
        let config = GeigerConfig::default();
        assert!(config.crate_path.is_none());
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert!(!config.include_dev_deps);
        assert!(!config.include_build_deps);
        assert!(config.unsafe_threshold.is_none());
    }

    #[test]
    fn test_geiger_config_builder() {
        let config = GeigerConfig::default()
            .with_crate_path(PathBuf::from("/test/path"))
            .with_timeout(Duration::from_secs(60))
            .with_dev_deps()
            .with_build_deps()
            .with_unsafe_threshold(10.0);

        assert_eq!(config.crate_path, Some(PathBuf::from("/test/path")));
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert!(config.include_dev_deps);
        assert!(config.include_build_deps);
        assert_eq!(config.unsafe_threshold, Some(10.0));
    }

    #[test]
    fn test_geiger_backend_id() {
        let backend = GeigerBackend::new();
        assert_eq!(backend.id(), BackendId::Geiger);
    }

    #[test]
    fn test_geiger_supports_unsafe_audit() {
        let backend = GeigerBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::UnsafeAudit));
    }

    #[tokio::test]
    async fn test_geiger_health_check() {
        let backend = GeigerBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => {}
            HealthStatus::Unavailable { reason } => {
                assert!(
                    reason.contains("geiger")
                        || reason.contains("Geiger")
                        || reason.contains("Check")
                );
            }
            HealthStatus::Degraded { .. } => {}
        }
    }

    #[tokio::test]
    async fn test_geiger_verify_requires_crate_path() {
        use dashprove_usl::{parse, typecheck};

        let backend = GeigerBackend::new();
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
