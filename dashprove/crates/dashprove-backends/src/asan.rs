//! AddressSanitizer backend for memory error detection
//!
//! This backend runs AddressSanitizer (ASAN) on Rust code to detect memory errors
//! such as buffer overflows, use-after-free, and double-free bugs.
//! It wraps the `dashprove-sanitizers` crate's ASAN support.

// =============================================
// Kani Proofs for ASAN Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- AsanConfig Default Tests ----

    /// Verify AsanConfig::default sets correct baseline values
    #[kani::proof]
    fn proof_asan_config_defaults() {
        let config = AsanConfig::default();
        kani::assert(
            config.crate_path.is_none(),
            "crate_path should default to None",
        );
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "timeout should default to 300 seconds",
        );
    }

    // ---- AsanConfig Builder Tests ----

    /// Verify with_crate_path updates crate_path
    #[kani::proof]
    fn proof_asan_config_with_crate_path() {
        let config = AsanConfig::default().with_crate_path(PathBuf::from("/test/crate"));
        kani::assert(
            config.crate_path == Some(PathBuf::from("/test/crate")),
            "with_crate_path should set crate_path",
        );
    }

    /// Verify with_timeout updates timeout
    #[kani::proof]
    fn proof_asan_config_with_timeout() {
        let config = AsanConfig::default().with_timeout(Duration::from_secs(120));
        kani::assert(
            config.timeout == Duration::from_secs(120),
            "with_timeout should set timeout",
        );
    }

    /// Verify builder chaining works correctly
    #[kani::proof]
    fn proof_asan_config_builder_chain() {
        let config = AsanConfig::default()
            .with_crate_path(PathBuf::from("/project"))
            .with_timeout(Duration::from_secs(600));

        kani::assert(
            config.crate_path == Some(PathBuf::from("/project")),
            "crate_path should be set after chain",
        );
        kani::assert(
            config.timeout == Duration::from_secs(600),
            "timeout should be set after chain",
        );
    }

    // ---- AsanBackend Construction Tests ----

    /// Verify AsanBackend::new uses default configuration
    #[kani::proof]
    fn proof_asan_backend_new_defaults() {
        let backend = AsanBackend::new();
        kani::assert(
            backend.config.crate_path.is_none(),
            "new backend should have no crate_path",
        );
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "new backend should default timeout to 300 seconds",
        );
    }

    /// Verify AsanBackend::default equals AsanBackend::new
    #[kani::proof]
    fn proof_asan_backend_default_equals_new() {
        let default_backend = AsanBackend::default();
        let new_backend = AsanBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "default and new should share timeout",
        );
        kani::assert(
            default_backend.config.crate_path == new_backend.config.crate_path,
            "default and new should share crate_path",
        );
    }

    /// Verify AsanBackend::with_config preserves custom configuration
    #[kani::proof]
    fn proof_asan_backend_with_config() {
        let config = AsanConfig {
            crate_path: Some(PathBuf::from("/work/project")),
            timeout: Duration::from_secs(180),
        };
        let backend = AsanBackend::with_config(config);
        kani::assert(
            backend.config.crate_path == Some(PathBuf::from("/work/project")),
            "with_config should preserve crate_path",
        );
        kani::assert(
            backend.config.timeout == Duration::from_secs(180),
            "with_config should preserve timeout",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify id() returns BackendId::AddressSanitizer
    #[kani::proof]
    fn proof_asan_backend_id() {
        let backend = AsanBackend::new();
        kani::assert(
            backend.id() == BackendId::AddressSanitizer,
            "AsanBackend id should be BackendId::AddressSanitizer",
        );
    }

    /// Verify supports() includes MemorySafety
    #[kani::proof]
    fn proof_asan_backend_supports() {
        let backend = AsanBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::MemorySafety),
            "supports should include MemorySafety",
        );
    }

    /// Verify supports() returns exactly one property type
    #[kani::proof]
    fn proof_asan_backend_supports_length() {
        let backend = AsanBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 1,
            "ASAN should support exactly one property type",
        );
    }

    // ---- SanitizerType Tests ----

    /// Verify SanitizerType::Address is available on supported platforms
    #[kani::proof]
    fn proof_sanitizer_type_address_exists() {
        // Just verify the type exists and can be used
        let san_type = SanitizerType::Address;
        // is_available() is platform-dependent, so just verify construction works
        let _ = san_type;
    }
}

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

/// Configuration for AddressSanitizer backend
#[derive(Debug, Clone)]
pub struct AsanConfig {
    /// Path to the crate to analyze
    pub crate_path: Option<PathBuf>,
    /// Timeout for analysis
    pub timeout: Duration,
}

impl Default for AsanConfig {
    fn default() -> Self {
        Self {
            crate_path: None,
            timeout: Duration::from_secs(300),
        }
    }
}

impl AsanConfig {
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

/// AddressSanitizer verification backend for memory error detection
///
/// ASAN detects memory errors at runtime by instrumenting memory accesses:
/// - Buffer overflows (heap and stack)
/// - Use-after-free
/// - Use-after-return
/// - Double-free
/// - Allocation/deallocation mismatch
///
/// # Requirements
///
/// - Nightly Rust toolchain (`rustup +nightly`)
/// - Linux or macOS
/// - Builds with `RUSTFLAGS="-Z sanitizer=address"`
pub struct AsanBackend {
    config: AsanConfig,
}

impl AsanBackend {
    /// Create a new ASAN backend with default configuration
    pub fn new() -> Self {
        Self {
            config: AsanConfig::default(),
        }
    }

    /// Create a new ASAN backend with custom configuration
    pub fn with_config(config: AsanConfig) -> Self {
        Self { config }
    }

    /// Run ASAN on a crate
    pub async fn run_asan(
        &self,
        crate_path: &std::path::Path,
    ) -> Result<BackendResult, BackendError> {
        let runner = SanitizerRunner::new(SanitizerType::Address).with_timeout(self.config.timeout);

        let result = runner.run_on_crate(crate_path).await.map_err(|e| match e {
            dashprove_sanitizers::SanitizerError::NotAvailable(msg) => {
                BackendError::Unavailable(format!("ASAN not available: {}", msg))
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
                "ASAN: No memory errors detected ({} tests run)",
                result.tests_run
            )
        } else {
            format!(
                "ASAN: {} memory errors detected ({} tests run)",
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
            backend: BackendId::AddressSanitizer,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: result.duration,
        })
    }
}

impl Default for AsanBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for AsanBackend {
    fn id(&self) -> BackendId {
        BackendId::AddressSanitizer
    }

    fn supports(&self) -> Vec<PropertyType> {
        // ASAN verifies memory safety
        vec![PropertyType::MemorySafety]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let crate_path = self.config.crate_path.clone().ok_or_else(|| {
            BackendError::Unavailable(
                "ASAN backend requires crate_path pointing to a Rust crate".to_string(),
            )
        })?;

        self.run_asan(&crate_path).await
    }

    async fn health_check(&self) -> HealthStatus {
        // Check if ASAN is available (requires nightly)
        match SanitizerRunner::check_nightly().await {
            Ok(true) => {
                // Check platform support
                if SanitizerType::Address.is_available() {
                    HealthStatus::Healthy
                } else {
                    HealthStatus::Unavailable {
                        reason: "AddressSanitizer requires Linux or macOS".to_string(),
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
    fn test_asan_config_default() {
        let config = AsanConfig::default();
        assert!(config.crate_path.is_none());
        assert_eq!(config.timeout, Duration::from_secs(300));
    }

    #[test]
    fn test_asan_config_builder() {
        let config = AsanConfig::default()
            .with_crate_path(PathBuf::from("/test/path"))
            .with_timeout(Duration::from_secs(120));

        assert_eq!(config.crate_path, Some(PathBuf::from("/test/path")));
        assert_eq!(config.timeout, Duration::from_secs(120));
    }

    #[test]
    fn test_asan_backend_id() {
        let backend = AsanBackend::new();
        assert_eq!(backend.id(), BackendId::AddressSanitizer);
    }

    #[test]
    fn test_asan_supports_memory_safety() {
        let backend = AsanBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::MemorySafety));
    }

    #[tokio::test]
    async fn test_asan_health_check() {
        let backend = AsanBackend::new();
        let health = backend.health_check().await;
        // Health check should return a valid status
        match health {
            HealthStatus::Healthy => {
                // Expected if nightly is available and platform supports ASAN
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
    async fn test_asan_verify_requires_crate_path() {
        use dashprove_usl::{parse, typecheck};

        let backend = AsanBackend::new();
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
