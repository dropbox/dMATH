//! Clippy backend for Rust lint verification
//!
//! This backend runs Clippy on Rust code to identify lints, potential bugs,
//! and style issues. It wraps the `dashprove-static` crate's Clippy support.

// =============================================
// Kani Proofs for Clippy Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- ClippyConfig Default Tests ----

    /// Verify ClippyConfig::default sets expected baseline values
    #[kani::proof]
    fn proof_clippy_config_defaults() {
        let config = ClippyConfig::default();
        kani::assert(
            config.crate_path.is_none(),
            "crate_path should default to None",
        );
        kani::assert(
            !config.warnings_as_errors,
            "warnings_as_errors should default to false",
        );
        kani::assert(!config.all_features, "all_features should default to false");
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "timeout should default to 300 seconds",
        );
    }

    // ---- ClippyConfig Builder Tests ----

    /// Verify builder methods update ClippyConfig fields
    #[kani::proof]
    fn proof_clippy_config_builder_updates() {
        let config = ClippyConfig::default()
            .with_crate_path(PathBuf::from("/tmp/crate"))
            .with_warnings_as_errors()
            .with_all_features()
            .with_timeout(Duration::from_secs(60));

        kani::assert(
            config.crate_path == Some(PathBuf::from("/tmp/crate")),
            "with_crate_path should set crate_path",
        );
        kani::assert(
            config.warnings_as_errors,
            "with_warnings_as_errors should enable warnings_as_errors",
        );
        kani::assert(
            config.all_features,
            "with_all_features should enable all_features",
        );
        kani::assert(
            config.timeout == Duration::from_secs(60),
            "with_timeout should set timeout",
        );
    }

    // ---- ClippyBackend Construction Tests ----

    /// Verify ClippyBackend::new uses default configuration
    #[kani::proof]
    fn proof_clippy_backend_new_defaults() {
        let backend = ClippyBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "new backend should default timeout to 300 seconds",
        );
        kani::assert(
            backend.config.crate_path.is_none(),
            "new backend should have no crate_path",
        );
        kani::assert(
            !backend.config.warnings_as_errors,
            "new backend should not treat warnings as errors by default",
        );
    }

    /// Verify ClippyBackend::default matches ClippyBackend::new
    #[kani::proof]
    fn proof_clippy_backend_default_equals_new() {
        let default_backend = ClippyBackend::default();
        let new_backend = ClippyBackend::new();
        kani::assert(
            default_backend.id() == new_backend.id(),
            "default and new should expose same backend id",
        );
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "default and new should share timeout",
        );
    }

    /// Verify ClippyBackend::with_config preserves custom settings
    #[kani::proof]
    fn proof_clippy_backend_with_config() {
        let config = ClippyConfig {
            crate_path: Some(PathBuf::from("/custom")),
            warnings_as_errors: true,
            all_features: true,
            timeout: Duration::from_secs(45),
        };
        let backend = ClippyBackend::with_config(config);
        kani::assert(
            backend.config.crate_path == Some(PathBuf::from("/custom")),
            "with_config should preserve crate_path",
        );
        kani::assert(
            backend.config.warnings_as_errors,
            "with_config should preserve warnings_as_errors",
        );
        kani::assert(
            backend.config.all_features,
            "with_config should preserve all_features",
        );
        kani::assert(
            backend.config.timeout == Duration::from_secs(45),
            "with_config should preserve timeout",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify id() returns BackendId::Clippy
    #[kani::proof]
    fn proof_clippy_backend_id() {
        let backend = ClippyBackend::new();
        kani::assert(
            backend.id() == BackendId::Clippy,
            "ClippyBackend id should be BackendId::Clippy",
        );
    }

    /// Verify supports() returns invariant property support
    #[kani::proof]
    fn proof_clippy_backend_supports_invariant() {
        let backend = ClippyBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 1 && supported.contains(&PropertyType::Invariant),
            "supports() should only include PropertyType::Invariant",
        );
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

/// Configuration for Clippy backend
#[derive(Debug, Clone)]
pub struct ClippyConfig {
    /// Path to the crate to analyze
    pub crate_path: Option<PathBuf>,
    /// Treat warnings as errors
    pub warnings_as_errors: bool,
    /// Enable all features when checking
    pub all_features: bool,
    /// Timeout for analysis
    pub timeout: Duration,
}

impl Default for ClippyConfig {
    fn default() -> Self {
        Self {
            crate_path: None,
            warnings_as_errors: false,
            all_features: false,
            timeout: Duration::from_secs(300),
        }
    }
}

impl ClippyConfig {
    /// Set the crate path to analyze
    pub fn with_crate_path(mut self, path: PathBuf) -> Self {
        self.crate_path = Some(path);
        self
    }

    /// Treat warnings as errors
    pub fn with_warnings_as_errors(mut self) -> Self {
        self.warnings_as_errors = true;
        self
    }

    /// Enable all features
    pub fn with_all_features(mut self) -> Self {
        self.all_features = true;
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

/// Clippy verification backend
pub struct ClippyBackend {
    config: ClippyConfig,
}

impl ClippyBackend {
    /// Create a new Clippy backend with default configuration
    pub fn new() -> Self {
        Self {
            config: ClippyConfig::default(),
        }
    }

    /// Create a new Clippy backend with custom configuration
    pub fn with_config(config: ClippyConfig) -> Self {
        Self { config }
    }

    /// Run Clippy on a crate path
    pub async fn analyze_crate(&self, crate_path: &Path) -> Result<BackendResult, BackendError> {
        let analysis_config = AnalysisConfig {
            warnings_as_errors: self.config.warnings_as_errors,
            all_features: self.config.all_features,
            features: Vec::new(),
            targets: Vec::new(),
            timeout: self.config.timeout,
        };

        let backend = StaticAnalysisBackend::new(AnalysisTool::Clippy, analysis_config);

        // Check if clippy is installed
        match backend.check_installed().await {
            Ok(true) => {}
            Ok(false) => {
                return Err(BackendError::Unavailable(
                    "Clippy not installed. Run: rustup component add clippy".to_string(),
                ));
            }
            Err(e) => {
                return Err(BackendError::Unavailable(format!(
                    "Failed to check clippy installation: {}",
                    e
                )));
            }
        }

        // Run clippy analysis
        let result = backend
            .run_on_crate(crate_path)
            .await
            .map_err(|e| BackendError::VerificationFailed(e.to_string()))?;

        // Convert analysis result to BackendResult
        let status = if result.passed {
            VerificationStatus::Proven
        } else if result.error_count > 0 {
            VerificationStatus::Disproven
        } else {
            // Warnings only - consider partial success
            VerificationStatus::Partial {
                verified_percentage: 100.0 * (1.0 - result.warning_count as f64 / 100.0),
            }
        };

        // Build diagnostic messages
        let mut diagnostics = Vec::new();
        for finding in &result.findings {
            let location = match (&finding.file, finding.line) {
                (Some(file), Some(line)) => format!("{}:{}", file, line),
                (Some(file), None) => file.clone(),
                _ => "unknown".to_string(),
            };

            let severity_str = match finding.severity {
                Severity::Error => "ERROR",
                Severity::Warning => "WARNING",
                Severity::Info => "INFO",
                Severity::Help => "HELP",
            };

            diagnostics.push(format!(
                "[{}] {} ({}): {} at {}",
                severity_str, finding.code, finding.tool, finding.message, location
            ));
        }

        // Add summary
        let summary = if result.passed {
            format!(
                "Clippy passed: {} warnings, {} errors",
                result.warning_count, result.error_count
            )
        } else {
            format!(
                "Clippy found issues: {} errors, {} warnings",
                result.error_count, result.warning_count
            )
        };
        diagnostics.insert(0, summary);

        Ok(BackendResult {
            backend: BackendId::Clippy,
            status,
            proof: None,
            counterexample: None,
            diagnostics,
            time_taken: result.duration,
        })
    }
}

impl Default for ClippyBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for ClippyBackend {
    fn id(&self) -> BackendId {
        BackendId::Clippy
    }

    fn supports(&self) -> Vec<PropertyType> {
        // Clippy checks for lints/style, which aligns with invariants
        vec![PropertyType::Invariant]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        // Clippy needs a crate path to analyze
        let crate_path = self.config.crate_path.clone().ok_or_else(|| {
            BackendError::Unavailable(
                "Clippy backend requires crate_path pointing to a Rust crate".to_string(),
            )
        })?;

        self.analyze_crate(&crate_path).await
    }

    async fn health_check(&self) -> HealthStatus {
        let backend = StaticAnalysisBackend::new(AnalysisTool::Clippy, AnalysisConfig::default());

        match backend.check_installed().await {
            Ok(true) => HealthStatus::Healthy,
            Ok(false) => HealthStatus::Unavailable {
                reason: "Clippy not installed".to_string(),
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
    fn test_clippy_config_default() {
        let config = ClippyConfig::default();
        assert!(config.crate_path.is_none());
        assert!(!config.warnings_as_errors);
        assert!(!config.all_features);
        assert_eq!(config.timeout, Duration::from_secs(300));
    }

    #[test]
    fn test_clippy_config_builder() {
        let config = ClippyConfig::default()
            .with_crate_path(PathBuf::from("/test/path"))
            .with_warnings_as_errors()
            .with_all_features()
            .with_timeout(Duration::from_secs(60));

        assert_eq!(config.crate_path, Some(PathBuf::from("/test/path")));
        assert!(config.warnings_as_errors);
        assert!(config.all_features);
        assert_eq!(config.timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_clippy_backend_id() {
        let backend = ClippyBackend::new();
        assert_eq!(backend.id(), BackendId::Clippy);
    }

    #[test]
    fn test_clippy_supports_invariant() {
        let backend = ClippyBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::Invariant));
    }

    #[tokio::test]
    async fn test_clippy_health_check() {
        let backend = ClippyBackend::new();
        let health = backend.health_check().await;
        // Should be healthy on systems with clippy installed
        match health {
            HealthStatus::Healthy => {
                // Expected on systems with clippy
            }
            HealthStatus::Unavailable { reason } => {
                // Expected on systems without clippy
                assert!(
                    reason.contains("Clippy")
                        || reason.contains("clippy")
                        || reason.contains("Check")
                );
            }
            HealthStatus::Degraded { .. } => {
                // Also acceptable
            }
        }
    }

    #[tokio::test]
    async fn test_clippy_verify_requires_crate_path() {
        use dashprove_usl::{parse, typecheck};

        let backend = ClippyBackend::new();
        // Create a minimal typed spec
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
