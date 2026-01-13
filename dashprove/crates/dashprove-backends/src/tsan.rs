//! ThreadSanitizer backend for data race detection
//!
//! This backend runs ThreadSanitizer (TSAN) on Rust code to detect data races
//! and other threading bugs. It wraps the `dashprove-sanitizers` crate's TSAN support.

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

/// Configuration for ThreadSanitizer backend
#[derive(Debug, Clone)]
pub struct TsanConfig {
    /// Path to the crate to analyze
    pub crate_path: Option<PathBuf>,
    /// Timeout for analysis
    pub timeout: Duration,
}

impl Default for TsanConfig {
    fn default() -> Self {
        Self {
            crate_path: None,
            timeout: Duration::from_secs(300),
        }
    }
}

impl TsanConfig {
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
}

/// ThreadSanitizer verification backend for data race detection
///
/// TSAN detects data races and threading bugs at runtime by instrumenting
/// memory accesses and synchronization operations:
/// - Data races between threads
/// - Use of mutexes without proper synchronization
/// - Deadlocks (in some configurations)
///
/// # Requirements
///
/// - Nightly Rust toolchain (`rustup +nightly`)
/// - Linux or macOS
/// - Builds with `RUSTFLAGS="-Z sanitizer=thread"`
pub struct TsanBackend {
    config: TsanConfig,
}

impl TsanBackend {
    /// Create a new TSAN backend with default configuration
    pub fn new() -> Self {
        Self {
            config: TsanConfig::default(),
        }
    }

    /// Create a new TSAN backend with custom configuration
    pub fn with_config(config: TsanConfig) -> Self {
        Self { config }
    }

    /// Run TSAN on a crate
    pub async fn run_tsan(
        &self,
        crate_path: &std::path::Path,
    ) -> Result<BackendResult, BackendError> {
        let runner = SanitizerRunner::new(SanitizerType::Thread).with_timeout(self.config.timeout);

        let result = runner.run_on_crate(crate_path).await.map_err(|e| match e {
            dashprove_sanitizers::SanitizerError::NotAvailable(msg) => {
                BackendError::Unavailable(format!("TSAN not available: {}", msg))
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
                "TSAN: No data races detected ({} tests run)",
                result.tests_run
            )
        } else {
            format!(
                "TSAN: {} data races detected ({} tests run)",
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
            backend: BackendId::ThreadSanitizer,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: result.duration,
        })
    }
}

impl Default for TsanBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for TsanBackend {
    fn id(&self) -> BackendId {
        BackendId::ThreadSanitizer
    }

    fn supports(&self) -> Vec<PropertyType> {
        // TSAN verifies data race freedom
        vec![PropertyType::DataRace]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let crate_path = self.config.crate_path.clone().ok_or_else(|| {
            BackendError::Unavailable(
                "TSAN backend requires crate_path pointing to a Rust crate".to_string(),
            )
        })?;

        self.run_tsan(&crate_path).await
    }

    async fn health_check(&self) -> HealthStatus {
        // Check if TSAN is available (requires nightly)
        match SanitizerRunner::check_nightly().await {
            Ok(true) => {
                // Check platform support
                if SanitizerType::Thread.is_available() {
                    HealthStatus::Healthy
                } else {
                    HealthStatus::Unavailable {
                        reason: "ThreadSanitizer requires Linux or macOS".to_string(),
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

    // ===== TsanConfig defaults =====

    #[kani::proof]
    fn verify_config_defaults_timeout() {
        let config = TsanConfig::default();
        assert!(config.timeout == Duration::from_secs(300));
    }

    #[kani::proof]
    fn verify_config_defaults_crate_path() {
        let config = TsanConfig::default();
        assert!(config.crate_path.is_none());
    }

    // ===== Config builders =====

    #[kani::proof]
    fn verify_config_with_crate_path() {
        let config = TsanConfig::default().with_crate_path(PathBuf::from("/test/path"));
        assert!(config.crate_path == Some(PathBuf::from("/test/path")));
    }

    #[kani::proof]
    fn verify_config_with_timeout() {
        let config = TsanConfig::default().with_timeout(Duration::from_secs(120));
        assert!(config.timeout == Duration::from_secs(120));
    }

    // ===== Backend construction =====

    #[kani::proof]
    fn verify_backend_new_uses_defaults() {
        let backend = TsanBackend::new();
        assert!(backend.config.timeout == Duration::from_secs(300));
        assert!(backend.config.crate_path.is_none());
    }

    #[kani::proof]
    fn verify_backend_default_matches_new() {
        let b1 = TsanBackend::new();
        let b2 = TsanBackend::default();
        assert!(b1.config.timeout == b2.config.timeout);
    }

    #[kani::proof]
    fn verify_backend_with_config_preserves_values() {
        let config = TsanConfig {
            crate_path: Some(PathBuf::from("/test")),
            timeout: Duration::from_secs(60),
        };
        let backend = TsanBackend::with_config(config);
        assert!(backend.config.timeout == Duration::from_secs(60));
        assert!(backend.config.crate_path.is_some());
    }

    // ===== ID and supports =====

    #[kani::proof]
    fn verify_backend_id() {
        let backend = TsanBackend::new();
        assert!(matches!(backend.id(), BackendId::ThreadSanitizer));
    }

    #[kani::proof]
    fn verify_supports_data_race() {
        let backend = TsanBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::DataRace));
        assert!(supported.len() == 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tsan_config_default() {
        let config = TsanConfig::default();
        assert!(config.crate_path.is_none());
        assert_eq!(config.timeout, Duration::from_secs(300));
    }

    #[test]
    fn test_tsan_config_builder() {
        let config = TsanConfig::default()
            .with_crate_path(PathBuf::from("/test/path"))
            .with_timeout(Duration::from_secs(120));

        assert_eq!(config.crate_path, Some(PathBuf::from("/test/path")));
        assert_eq!(config.timeout, Duration::from_secs(120));
    }

    #[test]
    fn test_tsan_backend_id() {
        let backend = TsanBackend::new();
        assert_eq!(backend.id(), BackendId::ThreadSanitizer);
    }

    #[test]
    fn test_tsan_supports_data_race() {
        let backend = TsanBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::DataRace));
    }

    #[tokio::test]
    async fn test_tsan_health_check() {
        let backend = TsanBackend::new();
        let health = backend.health_check().await;
        // Health check should return a valid status
        match health {
            HealthStatus::Healthy => {
                // Expected if nightly is available and platform supports TSAN
            }
            HealthStatus::Unavailable { reason } => {
                // Expected if nightly not installed or wrong platform
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
    async fn test_tsan_verify_requires_crate_path() {
        use dashprove_usl::{parse, typecheck};

        let backend = TsanBackend::new();
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
