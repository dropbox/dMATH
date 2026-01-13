//! LeakSanitizer backend for memory leak detection
//!
//! This backend runs LeakSanitizer (LSAN) on Rust code to detect memory leaks.
//! It wraps the `dashprove-sanitizers` crate's LSAN support.

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

/// Configuration for LeakSanitizer backend
#[derive(Debug, Clone)]
pub struct LsanConfig {
    /// Path to the crate to analyze
    pub crate_path: Option<PathBuf>,
    /// Timeout for analysis
    pub timeout: Duration,
}

impl Default for LsanConfig {
    fn default() -> Self {
        Self {
            crate_path: None,
            timeout: Duration::from_secs(300),
        }
    }
}

impl LsanConfig {
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

/// LeakSanitizer verification backend for memory leak detection
///
/// LSAN detects memory leaks at runtime by tracking memory allocations
/// and reporting any allocations that are not freed when the program exits:
/// - Direct memory leaks (allocated but never freed)
/// - Indirect memory leaks (leaked through lost pointers)
///
/// # Requirements
///
/// - Nightly Rust toolchain (`rustup +nightly`)
/// - Linux or macOS
/// - Builds with `RUSTFLAGS="-Z sanitizer=leak"`
pub struct LsanBackend {
    config: LsanConfig,
}

impl LsanBackend {
    /// Create a new LSAN backend with default configuration
    pub fn new() -> Self {
        Self {
            config: LsanConfig::default(),
        }
    }

    /// Create a new LSAN backend with custom configuration
    pub fn with_config(config: LsanConfig) -> Self {
        Self { config }
    }

    /// Run LSAN on a crate
    pub async fn run_lsan(
        &self,
        crate_path: &std::path::Path,
    ) -> Result<BackendResult, BackendError> {
        let runner = SanitizerRunner::new(SanitizerType::Leak).with_timeout(self.config.timeout);

        let result = runner.run_on_crate(crate_path).await.map_err(|e| match e {
            dashprove_sanitizers::SanitizerError::NotAvailable(msg) => {
                BackendError::Unavailable(format!("LSAN not available: {}", msg))
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
                "LSAN: No memory leaks detected ({} tests run)",
                result.tests_run
            )
        } else {
            format!(
                "LSAN: {} memory leaks detected ({} tests run)",
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
            backend: BackendId::LeakSanitizer,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: result.duration,
        })
    }
}

impl Default for LsanBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for LsanBackend {
    fn id(&self) -> BackendId {
        BackendId::LeakSanitizer
    }

    fn supports(&self) -> Vec<PropertyType> {
        // LSAN verifies memory leak freedom (related to memory safety)
        vec![PropertyType::MemorySafety]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let crate_path = self.config.crate_path.clone().ok_or_else(|| {
            BackendError::Unavailable(
                "LSAN backend requires crate_path pointing to a Rust crate".to_string(),
            )
        })?;

        self.run_lsan(&crate_path).await
    }

    async fn health_check(&self) -> HealthStatus {
        // Check if LSAN is available (requires nightly)
        match SanitizerRunner::check_nightly().await {
            Ok(true) => {
                // Check platform support
                if SanitizerType::Leak.is_available() {
                    HealthStatus::Healthy
                } else {
                    HealthStatus::Unavailable {
                        reason: "LeakSanitizer requires Linux or macOS".to_string(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lsan_config_default() {
        let config = LsanConfig::default();
        assert!(config.crate_path.is_none());
        assert_eq!(config.timeout, Duration::from_secs(300));
    }

    #[test]
    fn test_lsan_config_builder() {
        let config = LsanConfig::default()
            .with_crate_path(PathBuf::from("/test/path"))
            .with_timeout(Duration::from_secs(120));

        assert_eq!(config.crate_path, Some(PathBuf::from("/test/path")));
        assert_eq!(config.timeout, Duration::from_secs(120));
    }

    #[test]
    fn test_lsan_backend_id() {
        let backend = LsanBackend::new();
        assert_eq!(backend.id(), BackendId::LeakSanitizer);
    }

    #[test]
    fn test_lsan_supports_memory_safety() {
        let backend = LsanBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::MemorySafety));
    }

    #[tokio::test]
    async fn test_lsan_health_check() {
        let backend = LsanBackend::new();
        let health = backend.health_check().await;
        // Health check should return a valid status
        match health {
            HealthStatus::Healthy => {
                // Expected if nightly is available and platform supports LSAN
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
    async fn test_lsan_verify_requires_crate_path() {
        use dashprove_usl::{parse, typecheck};

        let backend = LsanBackend::new();
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

// =============================================
// Kani formal verification proofs
// =============================================
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // =============================================
    // LsanConfig default proofs
    // =============================================

    /// Verify LsanConfig default crate_path is None
    #[kani::proof]
    fn proof_lsan_config_default_crate_path_none() {
        let config = LsanConfig::default();
        kani::assert(
            config.crate_path.is_none(),
            "Default crate_path should be None",
        );
    }

    /// Verify LsanConfig default timeout is 300 seconds
    #[kani::proof]
    fn proof_lsan_config_default_timeout() {
        let config = LsanConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "Default timeout should be 300 seconds",
        );
    }

    // =============================================
    // LsanConfig builder proofs
    // =============================================

    /// Verify with_crate_path sets crate_path
    #[kani::proof]
    fn proof_lsan_config_with_crate_path() {
        let config = LsanConfig::default().with_crate_path(PathBuf::from("/test/path"));
        kani::assert(
            config.crate_path == Some(PathBuf::from("/test/path")),
            "crate_path should be set",
        );
    }

    /// Verify with_timeout sets timeout
    #[kani::proof]
    fn proof_lsan_config_with_timeout() {
        let config = LsanConfig::default().with_timeout(Duration::from_secs(120));
        kani::assert(
            config.timeout == Duration::from_secs(120),
            "timeout should be 120 seconds",
        );
    }

    // =============================================
    // LsanBackend constructor proofs
    // =============================================

    /// Verify LsanBackend::new creates backend with default config
    #[kani::proof]
    fn proof_lsan_backend_new_default_timeout() {
        let backend = LsanBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "New backend should have default timeout",
        );
    }

    /// Verify LsanBackend::with_config preserves config
    #[kani::proof]
    fn proof_lsan_backend_with_config() {
        let config = LsanConfig {
            timeout: Duration::from_secs(120),
            crate_path: Some(PathBuf::from("/test")),
        };
        let backend = LsanBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(120),
            "Custom timeout should be preserved",
        );
        kani::assert(
            backend.config.crate_path == Some(PathBuf::from("/test")),
            "Custom crate_path should be preserved",
        );
    }

    /// Verify LsanBackend implements Default
    #[kani::proof]
    fn proof_lsan_backend_default() {
        let backend = LsanBackend::default();
        kani::assert(
            backend.id() == BackendId::LeakSanitizer,
            "Default backend should have correct ID",
        );
    }

    // =============================================
    // LsanBackend trait implementation proofs
    // =============================================

    /// Verify LsanBackend::id returns BackendId::LeakSanitizer
    #[kani::proof]
    fn proof_lsan_backend_id() {
        let backend = LsanBackend::new();
        kani::assert(
            backend.id() == BackendId::LeakSanitizer,
            "Backend ID should be LeakSanitizer",
        );
    }

    /// Verify LsanBackend::supports includes MemorySafety
    #[kani::proof]
    fn proof_lsan_backend_supports_memory_safety() {
        let backend = LsanBackend::new();
        let supported = backend.supports();
        let mut found = false;
        for pt in supported {
            if pt == PropertyType::MemorySafety {
                found = true;
            }
        }
        kani::assert(found, "Should support MemorySafety");
    }

    /// Verify LsanBackend::supports returns exactly 1 type
    #[kani::proof]
    fn proof_lsan_backend_supports_count() {
        let backend = LsanBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 1,
            "Should support exactly 1 property type",
        );
    }

    // =============================================
    // Builder chaining proofs
    // =============================================

    /// Verify builder methods can be chained
    #[kani::proof]
    fn proof_lsan_config_builder_chain() {
        let config = LsanConfig::default()
            .with_crate_path(PathBuf::from("/test"))
            .with_timeout(Duration::from_secs(60));
        kani::assert(config.crate_path.is_some(), "crate_path should be set");
        kani::assert(
            config.timeout == Duration::from_secs(60),
            "timeout should be 60s",
        );
    }

    // =============================================
    // Config custom values proofs
    // =============================================

    /// Verify zero timeout is preserved
    #[kani::proof]
    fn proof_lsan_config_zero_timeout() {
        let config = LsanConfig::default().with_timeout(Duration::ZERO);
        kani::assert(
            config.timeout == Duration::ZERO,
            "Zero timeout should be preserved",
        );
    }

    /// Verify large timeout is preserved
    #[kani::proof]
    fn proof_lsan_config_large_timeout() {
        let config = LsanConfig::default().with_timeout(Duration::from_secs(3600));
        kani::assert(
            config.timeout == Duration::from_secs(3600),
            "Large timeout should be preserved",
        );
    }

    /// Verify custom config with all options
    #[kani::proof]
    fn proof_lsan_config_all_custom() {
        let config = LsanConfig {
            crate_path: Some(PathBuf::from("/custom/path")),
            timeout: Duration::from_secs(600),
        };
        kani::assert(config.crate_path.is_some(), "crate_path should be Some");
        kani::assert(
            config.timeout == Duration::from_secs(600),
            "timeout should be 600s",
        );
    }

    /// Verify backend preserves crate_path None
    #[kani::proof]
    fn proof_lsan_backend_new_crate_path_none() {
        let backend = LsanBackend::new();
        kani::assert(
            backend.config.crate_path.is_none(),
            "New backend should have None crate_path",
        );
    }
}
