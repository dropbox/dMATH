//! Miri backend for undefined behavior detection in Rust code
//!
//! This backend runs Miri (the Mid-level Intermediate Representation Interpreter)
//! on Rust code to detect undefined behavior. It wraps the `dashprove-miri` crate.

use crate::counterexample::StructuredCounterexample;
use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
    VerificationStatus,
};
use async_trait::async_trait;
use dashprove_miri::{detect_miri, run_miri, MiriConfig, MiriDetection};
use dashprove_usl::typecheck::TypedSpec;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Configuration for Miri backend
#[derive(Debug, Clone)]
pub struct MiriBackendConfig {
    /// Path to the crate to analyze
    pub crate_path: Option<PathBuf>,
    /// Test filter pattern (run only matching tests)
    pub test_filter: Option<String>,
    /// Timeout for analysis
    pub timeout: Duration,
    /// Whether to track raw pointer provenance
    pub track_raw_pointers: bool,
    /// Whether to detect memory leaks
    pub detect_leaks: bool,
    /// Whether to detect data races (requires stacked borrows)
    pub detect_data_races: bool,
}

impl Default for MiriBackendConfig {
    fn default() -> Self {
        Self {
            crate_path: None,
            test_filter: None,
            timeout: Duration::from_secs(600), // Miri is slow, give it time
            track_raw_pointers: true,
            detect_leaks: true,
            detect_data_races: true,
        }
    }
}

impl MiriBackendConfig {
    /// Set the crate path to analyze
    pub fn with_crate_path(mut self, path: PathBuf) -> Self {
        self.crate_path = Some(path);
        self
    }

    /// Set test filter pattern
    pub fn with_test_filter(mut self, filter: String) -> Self {
        self.test_filter = Some(filter);
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Enable/disable raw pointer tracking
    pub fn with_raw_pointer_tracking(mut self, track: bool) -> Self {
        self.track_raw_pointers = track;
        self
    }

    /// Enable/disable leak detection
    pub fn with_leak_detection(mut self, detect: bool) -> Self {
        self.detect_leaks = detect;
        self
    }

    /// Enable/disable data race detection
    pub fn with_data_race_detection(mut self, detect: bool) -> Self {
        self.detect_data_races = detect;
        self
    }

    /// Convert to the underlying dashprove-miri config
    fn to_miri_config(&self) -> MiriConfig {
        use dashprove_miri::MiriFlags;

        MiriConfig {
            timeout: self.timeout,
            flags: MiriFlags {
                track_raw_pointers: self.track_raw_pointers,
                disable_isolation: true, // Allow file system access for tests
                symbolic_alignment: false,
                ..MiriFlags::default()
            },
            ..MiriConfig::default()
        }
    }
}

/// Miri verification backend for undefined behavior detection
pub struct MiriBackend {
    config: MiriBackendConfig,
}

impl MiriBackend {
    /// Create a new Miri backend with default configuration
    pub fn new() -> Self {
        Self {
            config: MiriBackendConfig::default(),
        }
    }

    /// Create a new Miri backend with custom configuration
    pub fn with_config(config: MiriBackendConfig) -> Self {
        Self { config }
    }

    /// Run Miri on a crate path
    pub async fn analyze_crate(&self, crate_path: &Path) -> Result<BackendResult, BackendError> {
        let miri_config = self.config.to_miri_config();

        // Detect Miri availability
        let detection = detect_miri(&miri_config).await;

        match &detection {
            MiriDetection::NotFound(reason) => {
                return Err(BackendError::Unavailable(format!(
                    "Miri not available: {}. Install with: rustup +nightly component add miri",
                    reason
                )));
            }
            MiriDetection::Available { .. } => {}
        }

        // Run Miri
        let test_filter = self.config.test_filter.as_deref();
        let result = run_miri(&miri_config, &detection, crate_path, test_filter)
            .await
            .map_err(|e| BackendError::VerificationFailed(e.to_string()))?;

        // Parse the result
        let status = if result.success() {
            VerificationStatus::Proven
        } else if result.has_errors {
            VerificationStatus::Disproven
        } else {
            VerificationStatus::Unknown {
                reason: "Test run incomplete or unexpected status".to_string(),
            }
        };

        // Build diagnostic messages
        let mut diagnostics = Vec::new();

        // Add summary
        let summary = if result.success() {
            "Miri: No undefined behavior detected".to_string()
        } else {
            "Miri: Undefined behavior detected".to_string()
        };
        diagnostics.push(summary);

        // Add stderr output (contains error details)
        if !result.stderr.is_empty() && result.has_errors {
            // Extract key error messages from stderr
            for line in result.stderr.lines() {
                if line.contains("Undefined Behavior")
                    || line.contains("error[")
                    || line.contains("memory access")
                    || line.contains("out of bounds")
                    || line.contains("dangling")
                    || line.contains("uninitialized")
                    || line.contains("data race")
                    || line.contains("use after free")
                {
                    diagnostics.push(line.to_string());
                }
            }
        }

        // Build counterexample if UB found
        let counterexample = if result.has_errors && !result.stderr.is_empty() {
            Some(StructuredCounterexample::from_raw(result.stderr.clone()))
        } else {
            None
        };

        Ok(BackendResult {
            backend: BackendId::Miri,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: result.duration,
        })
    }
}

impl Default for MiriBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for MiriBackend {
    fn id(&self) -> BackendId {
        BackendId::Miri
    }

    fn supports(&self) -> Vec<PropertyType> {
        // Miri verifies memory safety and undefined behavior
        vec![PropertyType::MemorySafety, PropertyType::UndefinedBehavior]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        // Miri needs a crate path to analyze
        let crate_path = self.config.crate_path.clone().ok_or_else(|| {
            BackendError::Unavailable(
                "Miri backend requires crate_path pointing to a Rust crate".to_string(),
            )
        })?;

        self.analyze_crate(&crate_path).await
    }

    async fn health_check(&self) -> HealthStatus {
        let miri_config = self.config.to_miri_config();
        let detection = detect_miri(&miri_config).await;

        match detection {
            MiriDetection::Available { .. } => HealthStatus::Healthy,
            MiriDetection::NotFound(reason) => HealthStatus::Unavailable { reason },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_miri_config_default() {
        let config = MiriBackendConfig::default();
        assert!(config.crate_path.is_none());
        assert!(config.test_filter.is_none());
        assert_eq!(config.timeout, Duration::from_secs(600));
        assert!(config.track_raw_pointers);
        assert!(config.detect_leaks);
        assert!(config.detect_data_races);
    }

    #[test]
    fn test_miri_config_builder() {
        let config = MiriBackendConfig::default()
            .with_crate_path(PathBuf::from("/test/path"))
            .with_test_filter("test_foo".to_string())
            .with_timeout(Duration::from_secs(120))
            .with_raw_pointer_tracking(false)
            .with_leak_detection(false)
            .with_data_race_detection(false);

        assert_eq!(config.crate_path, Some(PathBuf::from("/test/path")));
        assert_eq!(config.test_filter, Some("test_foo".to_string()));
        assert_eq!(config.timeout, Duration::from_secs(120));
        assert!(!config.track_raw_pointers);
        assert!(!config.detect_leaks);
        assert!(!config.detect_data_races);
    }

    #[test]
    fn test_miri_backend_id() {
        let backend = MiriBackend::new();
        assert_eq!(backend.id(), BackendId::Miri);
    }

    #[test]
    fn test_miri_supports_memory_safety() {
        let backend = MiriBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::MemorySafety));
        assert!(supported.contains(&PropertyType::UndefinedBehavior));
    }

    #[tokio::test]
    async fn test_miri_health_check() {
        let backend = MiriBackend::new();
        let health = backend.health_check().await;
        // Should return a valid health status (available or unavailable)
        match health {
            HealthStatus::Healthy => {
                // Expected on systems with Miri installed
            }
            HealthStatus::Unavailable { reason } => {
                // Expected on systems without Miri
                assert!(
                    reason.contains("Miri")
                        || reason.contains("miri")
                        || reason.contains("rustup")
                        || reason.contains("nightly")
                );
            }
            HealthStatus::Degraded { .. } => {
                // Also acceptable
            }
        }
    }

    #[tokio::test]
    async fn test_miri_verify_requires_crate_path() {
        use dashprove_usl::{parse, typecheck};

        let backend = MiriBackend::new();
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

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ==================== MiriBackendConfig Default Proofs ====================

    #[kani::proof]
    fn proof_config_default_crate_path_none() {
        let config = MiriBackendConfig::default();
        kani::assert(
            config.crate_path.is_none(),
            "Default crate_path should be None",
        );
    }

    #[kani::proof]
    fn proof_config_default_test_filter_none() {
        let config = MiriBackendConfig::default();
        kani::assert(
            config.test_filter.is_none(),
            "Default test_filter should be None",
        );
    }

    #[kani::proof]
    fn proof_config_default_timeout_600s() {
        let config = MiriBackendConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(600),
            "Default timeout should be 600 seconds",
        );
    }

    #[kani::proof]
    fn proof_config_default_track_raw_pointers_true() {
        let config = MiriBackendConfig::default();
        kani::assert(
            config.track_raw_pointers,
            "Default track_raw_pointers should be true",
        );
    }

    #[kani::proof]
    fn proof_config_default_detect_leaks_true() {
        let config = MiriBackendConfig::default();
        kani::assert(config.detect_leaks, "Default detect_leaks should be true");
    }

    #[kani::proof]
    fn proof_config_default_detect_data_races_true() {
        let config = MiriBackendConfig::default();
        kani::assert(
            config.detect_data_races,
            "Default detect_data_races should be true",
        );
    }

    // ==================== MiriBackendConfig Builder Proofs ====================

    #[kani::proof]
    fn proof_config_with_crate_path() {
        let config = MiriBackendConfig::default().with_crate_path(PathBuf::from("/test/path"));
        kani::assert(
            config.crate_path == Some(PathBuf::from("/test/path")),
            "with_crate_path should set path",
        );
    }

    #[kani::proof]
    fn proof_config_with_test_filter() {
        let config = MiriBackendConfig::default().with_test_filter("test_foo".to_string());
        kani::assert(
            config.test_filter == Some("test_foo".to_string()),
            "with_test_filter should set filter",
        );
    }

    #[kani::proof]
    fn proof_config_with_timeout() {
        let config = MiriBackendConfig::default().with_timeout(Duration::from_secs(120));
        kani::assert(
            config.timeout == Duration::from_secs(120),
            "with_timeout should set timeout",
        );
    }

    #[kani::proof]
    fn proof_config_with_raw_pointer_tracking_false() {
        let config = MiriBackendConfig::default().with_raw_pointer_tracking(false);
        kani::assert(
            !config.track_raw_pointers,
            "with_raw_pointer_tracking(false) should disable",
        );
    }

    #[kani::proof]
    fn proof_config_with_leak_detection_false() {
        let config = MiriBackendConfig::default().with_leak_detection(false);
        kani::assert(
            !config.detect_leaks,
            "with_leak_detection(false) should disable",
        );
    }

    #[kani::proof]
    fn proof_config_with_data_race_detection_false() {
        let config = MiriBackendConfig::default().with_data_race_detection(false);
        kani::assert(
            !config.detect_data_races,
            "with_data_race_detection(false) should disable",
        );
    }

    #[kani::proof]
    fn proof_config_builder_chaining() {
        let config = MiriBackendConfig::default()
            .with_crate_path(PathBuf::from("/test"))
            .with_timeout(Duration::from_secs(60))
            .with_raw_pointer_tracking(false);
        kani::assert(config.crate_path.is_some(), "Should chain crate_path");
        kani::assert(
            config.timeout == Duration::from_secs(60),
            "Should chain timeout",
        );
        kani::assert(
            !config.track_raw_pointers,
            "Should chain raw_pointer_tracking",
        );
    }

    // ==================== MiriBackend Construction Proofs ====================

    #[kani::proof]
    fn proof_backend_new_uses_default_config() {
        let backend = MiriBackend::new();
        kani::assert(
            backend.config.crate_path.is_none(),
            "new() should use default config with crate_path=None",
        );
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "new() should use default timeout of 600s",
        );
    }

    #[kani::proof]
    fn proof_backend_default_equals_new() {
        let b1 = MiriBackend::new();
        let b2 = MiriBackend::default();
        kani::assert(
            b1.config.timeout == b2.config.timeout,
            "default() and new() should produce equal timeout",
        );
        kani::assert(
            b1.config.track_raw_pointers == b2.config.track_raw_pointers,
            "default() and new() should produce equal track_raw_pointers",
        );
    }

    #[kani::proof]
    fn proof_backend_with_config_preserves_settings() {
        let config = MiriBackendConfig::default()
            .with_timeout(Duration::from_secs(120))
            .with_leak_detection(false);
        let backend = MiriBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(120),
            "with_config should preserve timeout",
        );
        kani::assert(
            !backend.config.detect_leaks,
            "with_config should preserve detect_leaks",
        );
    }

    // ==================== Backend Trait Implementation Proofs ====================

    #[kani::proof]
    fn proof_backend_id_is_miri() {
        let backend = MiriBackend::new();
        kani::assert(backend.id() == BackendId::Miri, "Backend ID should be Miri");
    }

    #[kani::proof]
    fn proof_supports_contains_memory_safety() {
        let backend = MiriBackend::new();
        let supported = backend.supports();
        let has_memory_safety = supported.iter().any(|p| *p == PropertyType::MemorySafety);
        kani::assert(
            has_memory_safety,
            "Should support MemorySafety property type",
        );
    }

    #[kani::proof]
    fn proof_supports_contains_undefined_behavior() {
        let backend = MiriBackend::new();
        let supported = backend.supports();
        let has_ub = supported
            .iter()
            .any(|p| *p == PropertyType::UndefinedBehavior);
        kani::assert(has_ub, "Should support UndefinedBehavior property type");
    }

    #[kani::proof]
    fn proof_supports_returns_two_types() {
        let backend = MiriBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 2,
            "Should support exactly 2 property types",
        );
    }

    // ==================== to_miri_config Proofs ====================

    #[kani::proof]
    fn proof_to_miri_config_preserves_timeout() {
        let config = MiriBackendConfig::default().with_timeout(Duration::from_secs(300));
        let miri_config = config.to_miri_config();
        kani::assert(
            miri_config.timeout == Duration::from_secs(300),
            "to_miri_config should preserve timeout",
        );
    }

    #[kani::proof]
    fn proof_to_miri_config_sets_track_raw_pointers() {
        let config = MiriBackendConfig::default().with_raw_pointer_tracking(true);
        let miri_config = config.to_miri_config();
        kani::assert(
            miri_config.flags.track_raw_pointers,
            "to_miri_config should set track_raw_pointers flag",
        );
    }

    #[kani::proof]
    fn proof_to_miri_config_disables_isolation() {
        let config = MiriBackendConfig::default();
        let miri_config = config.to_miri_config();
        kani::assert(
            miri_config.flags.disable_isolation,
            "to_miri_config should disable isolation for tests",
        );
    }
}
