//! MemorySanitizer backend for uninitialized memory detection
//!
//! This backend runs MemorySanitizer (MSAN) on Rust code to detect reads of
//! uninitialized memory. It wraps the `dashprove-sanitizers` crate's MSAN support.

use crate::counterexample::StructuredCounterexample;
use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
    VerificationStatus,
};
use async_trait::async_trait;
use dashprove_sanitizers::{FindingSeverity, SanitizerBackend as SanitizerRunner, SanitizerType};
use dashprove_usl::typecheck::TypedSpec;
use std::path::PathBuf;
use std::time::Duration;

/// Configuration for MemorySanitizer backend
#[derive(Debug, Clone)]
pub struct MsanConfig {
    /// Path to the crate to analyze
    pub crate_path: Option<PathBuf>,
    /// Timeout for analysis
    pub timeout: Duration,
    /// Track origins of uninitialized memory
    pub track_origins: bool,
}

impl Default for MsanConfig {
    fn default() -> Self {
        Self {
            crate_path: None,
            timeout: Duration::from_secs(300),
            track_origins: true,
        }
    }
}

impl MsanConfig {
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

    /// Enable/disable origin tracking
    pub fn with_origin_tracking(mut self, track: bool) -> Self {
        self.track_origins = track;
        self
    }
}

/// MemorySanitizer verification backend for uninitialized memory detection
///
/// MSAN detects reads of uninitialized memory at runtime by tracking which memory
/// has been written to. This catches bugs that may cause undefined behavior or
/// information leaks.
///
/// # Requirements
///
/// - Nightly Rust toolchain (`rustup +nightly`)
/// - Linux only (MSAN does not support macOS)
/// - Builds with `RUSTFLAGS="-Z sanitizer=memory"`
///
/// Note: MSAN requires rebuilding the standard library, which significantly
/// increases compilation time.
pub struct MsanBackend {
    config: MsanConfig,
}

impl MsanBackend {
    /// Create a new MSAN backend with default configuration
    pub fn new() -> Self {
        Self {
            config: MsanConfig::default(),
        }
    }

    /// Create a new MSAN backend with custom configuration
    pub fn with_config(config: MsanConfig) -> Self {
        Self { config }
    }

    /// Run MSAN on a crate
    pub async fn run_msan(
        &self,
        crate_path: &std::path::Path,
    ) -> Result<BackendResult, BackendError> {
        let runner = SanitizerRunner::new(SanitizerType::Memory).with_timeout(self.config.timeout);

        let result = runner.run_on_crate(crate_path).await.map_err(|e| match e {
            dashprove_sanitizers::SanitizerError::NotAvailable(msg) => {
                BackendError::Unavailable(format!("MSAN not available: {}", msg))
            }
            dashprove_sanitizers::SanitizerError::NightlyRequired(msg) => {
                BackendError::Unavailable(format!("Nightly toolchain required: {}", msg))
            }
            dashprove_sanitizers::SanitizerError::BuildFailed(msg) => {
                BackendError::VerificationFailed(format!("Build failed: {}", msg))
            }
            dashprove_sanitizers::SanitizerError::TestFailed(msg) => {
                BackendError::VerificationFailed(format!("Test failed: {}", msg))
            }
            dashprove_sanitizers::SanitizerError::Timeout(d) => BackendError::Timeout(d),
            dashprove_sanitizers::SanitizerError::Io(e) => {
                BackendError::VerificationFailed(format!("IO error: {}", e))
            }
        })?;

        // Convert SanitizerResult to BackendResult
        let status = if result.passed {
            VerificationStatus::Proven
        } else if !result.findings.is_empty() {
            VerificationStatus::Disproven
        } else {
            VerificationStatus::Unknown {
                reason: "Tests failed but no specific findings identified".to_string(),
            }
        };

        // Build diagnostics
        let mut diagnostics = Vec::new();

        // Summary
        let summary = if result.passed {
            format!(
                "MSAN: No uninitialized memory reads detected ({} tests run)",
                result.tests_run
            )
        } else {
            format!(
                "MSAN: {} uninitialized memory issues detected ({} tests run)",
                result.findings.len(),
                result.tests_run
            )
        };
        diagnostics.push(summary);

        // Add finding details
        for finding in &result.findings {
            let severity_str = match finding.severity {
                FindingSeverity::Error => "ERROR",
                FindingSeverity::Warning => "WARNING",
            };
            diagnostics.push(format!(
                "  [{}] {}: {}",
                severity_str, finding.issue_type, finding.description
            ));

            if let Some(ref trace) = finding.stack_trace {
                // Add first few lines of stack trace
                for line in trace.lines().take(5) {
                    diagnostics.push(format!("    {}", line));
                }
            }
        }

        // Build counterexample if errors found
        let counterexample = if !result.findings.is_empty() {
            Some(StructuredCounterexample::from_raw(
                result.raw_output.clone(),
            ))
        } else {
            None
        };

        Ok(BackendResult {
            backend: BackendId::MemorySanitizer,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: result.duration,
        })
    }
}

impl Default for MsanBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for MsanBackend {
    fn id(&self) -> BackendId {
        BackendId::MemorySanitizer
    }

    fn supports(&self) -> Vec<PropertyType> {
        // MSAN verifies memory safety (specifically uninitialized reads)
        vec![PropertyType::MemorySafety]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let crate_path = self.config.crate_path.clone().ok_or_else(|| {
            BackendError::Unavailable(
                "MSAN backend requires crate_path pointing to a Rust crate".to_string(),
            )
        })?;

        self.run_msan(&crate_path).await
    }

    async fn health_check(&self) -> HealthStatus {
        // Check if MSAN is available (requires nightly + Linux)
        match SanitizerRunner::check_nightly().await {
            Ok(true) => {
                // Check platform support
                if SanitizerType::Memory.is_available() {
                    HealthStatus::Healthy
                } else {
                    HealthStatus::Unavailable {
                        reason: "MemorySanitizer requires Linux (not available on macOS)"
                            .to_string(),
                    }
                }
            }
            Ok(false) => HealthStatus::Unavailable {
                reason: "Nightly Rust toolchain required. Run: rustup install nightly".to_string(),
            },
            Err(e) => HealthStatus::Unavailable {
                reason: format!("Failed to check nightly: {}", e),
            },
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ===== MsanConfig default proofs =====

    #[kani::proof]
    fn verify_config_default_crate_path_none() {
        let config = MsanConfig::default();
        assert!(config.crate_path.is_none());
    }

    #[kani::proof]
    fn verify_config_default_timeout() {
        let config = MsanConfig::default();
        assert!(config.timeout.as_secs() == 300);
    }

    #[kani::proof]
    fn verify_config_default_track_origins() {
        let config = MsanConfig::default();
        assert!(config.track_origins);
    }

    // ===== MsanConfig builder proofs =====

    #[kani::proof]
    fn verify_config_with_crate_path() {
        let config = MsanConfig::default().with_crate_path(PathBuf::from("/test"));
        assert!(config.crate_path == Some(PathBuf::from("/test")));
    }

    #[kani::proof]
    fn verify_config_with_timeout() {
        let config = MsanConfig::default().with_timeout(Duration::from_secs(120));
        assert!(config.timeout.as_secs() == 120);
    }

    #[kani::proof]
    fn verify_config_with_origin_tracking_false() {
        let config = MsanConfig::default().with_origin_tracking(false);
        assert!(!config.track_origins);
    }

    #[kani::proof]
    fn verify_config_with_origin_tracking_true() {
        let config = MsanConfig::default().with_origin_tracking(true);
        assert!(config.track_origins);
    }

    #[kani::proof]
    fn verify_config_builder_chaining() {
        let config = MsanConfig::default()
            .with_crate_path(PathBuf::from("/path"))
            .with_timeout(Duration::from_secs(60))
            .with_origin_tracking(false);
        assert!(config.crate_path == Some(PathBuf::from("/path")));
        assert!(config.timeout.as_secs() == 60);
        assert!(!config.track_origins);
    }

    // ===== MsanBackend construction proofs =====

    #[kani::proof]
    fn verify_backend_new_has_default_config() {
        let backend = MsanBackend::new();
        assert!(backend.config.timeout.as_secs() == 300);
        assert!(backend.config.track_origins);
    }

    #[kani::proof]
    fn verify_backend_default_equals_new() {
        let b1 = MsanBackend::new();
        let b2 = MsanBackend::default();
        assert!(b1.config.timeout == b2.config.timeout);
        assert!(b1.config.track_origins == b2.config.track_origins);
    }

    #[kani::proof]
    fn verify_backend_with_config_preserves_timeout() {
        let config = MsanConfig {
            crate_path: None,
            timeout: Duration::from_secs(600),
            track_origins: false,
        };
        let backend = MsanBackend::with_config(config);
        assert!(backend.config.timeout.as_secs() == 600);
    }

    #[kani::proof]
    fn verify_backend_with_config_preserves_track_origins() {
        let config = MsanConfig {
            crate_path: None,
            timeout: Duration::from_secs(300),
            track_origins: false,
        };
        let backend = MsanBackend::with_config(config);
        assert!(!backend.config.track_origins);
    }

    #[kani::proof]
    fn verify_backend_with_config_preserves_crate_path() {
        let config = MsanConfig {
            crate_path: Some(PathBuf::from("/my/crate")),
            timeout: Duration::from_secs(300),
            track_origins: true,
        };
        let backend = MsanBackend::with_config(config);
        assert!(backend.config.crate_path == Some(PathBuf::from("/my/crate")));
    }

    // ===== Backend ID proof =====

    #[kani::proof]
    fn verify_backend_id() {
        let backend = MsanBackend::new();
        assert!(matches!(backend.id(), BackendId::MemorySanitizer));
    }

    // ===== Supports proofs =====

    #[kani::proof]
    fn verify_supports_memory_safety() {
        let backend = MsanBackend::new();
        let supported = backend.supports();
        let has_memory_safety = supported
            .iter()
            .any(|p| matches!(p, PropertyType::MemorySafety));
        assert!(has_memory_safety);
    }

    #[kani::proof]
    fn verify_supports_count() {
        let backend = MsanBackend::new();
        let supported = backend.supports();
        assert!(supported.len() == 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_msan_config_default() {
        let config = MsanConfig::default();
        assert!(config.crate_path.is_none());
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert!(config.track_origins);
    }

    #[test]
    fn test_msan_config_builder() {
        let config = MsanConfig::default()
            .with_crate_path(PathBuf::from("/test/path"))
            .with_timeout(Duration::from_secs(120))
            .with_origin_tracking(false);

        assert_eq!(config.crate_path, Some(PathBuf::from("/test/path")));
        assert_eq!(config.timeout, Duration::from_secs(120));
        assert!(!config.track_origins);
    }

    #[test]
    fn test_msan_backend_id() {
        let backend = MsanBackend::new();
        assert_eq!(backend.id(), BackendId::MemorySanitizer);
    }

    #[test]
    fn test_msan_supports_memory_safety() {
        let backend = MsanBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::MemorySafety));
    }

    #[tokio::test]
    async fn test_msan_health_check() {
        let backend = MsanBackend::new();
        let health = backend.health_check().await;
        // Health check should return a valid status
        match health {
            HealthStatus::Healthy => {
                // Expected if nightly is available and on Linux
            }
            HealthStatus::Unavailable { reason } => {
                // Expected if nightly not installed or on macOS
                assert!(
                    reason.contains("nightly")
                        || reason.contains("Nightly")
                        || reason.contains("Linux")
                        || reason.contains("macOS")
                );
            }
            HealthStatus::Degraded { .. } => {
                // Also acceptable
            }
        }
    }

    #[tokio::test]
    async fn test_msan_verify_requires_crate_path() {
        use dashprove_usl::{parse, typecheck};

        let backend = MsanBackend::new();
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
