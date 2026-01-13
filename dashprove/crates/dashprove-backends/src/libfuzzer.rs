//! LibFuzzer backend for coverage-guided fuzzing
//!
//! This backend runs LibFuzzer (via cargo-fuzz) on Rust code for coverage-guided
//! fuzzing. It wraps the `dashprove-fuzz` crate's LibFuzzer support.

use crate::counterexample::StructuredCounterexample;
use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
    VerificationStatus,
};
use async_trait::async_trait;
use dashprove_fuzz::{FuzzBackend as FuzzRunner, FuzzConfig, FuzzerType};
use dashprove_usl::typecheck::TypedSpec;
use std::path::PathBuf;
use std::time::Duration;

/// Configuration for LibFuzzer backend
#[derive(Debug, Clone)]
pub struct LibFuzzerConfig {
    /// Path to the crate to fuzz
    pub crate_path: Option<PathBuf>,
    /// Name of the fuzz target
    pub target: Option<String>,
    /// Timeout for fuzzing
    pub timeout: Duration,
    /// Maximum number of iterations
    pub max_iterations: u64,
    /// Maximum input length
    pub max_len: Option<usize>,
}

impl Default for LibFuzzerConfig {
    fn default() -> Self {
        Self {
            crate_path: None,
            target: None,
            timeout: Duration::from_secs(60),
            max_iterations: 0, // Run until timeout
            max_len: None,
        }
    }
}

impl LibFuzzerConfig {
    /// Set the crate path
    pub fn with_crate_path(mut self, path: PathBuf) -> Self {
        self.crate_path = Some(path);
        self
    }

    /// Set the fuzz target name
    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.target = Some(target.into());
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set maximum iterations
    pub fn with_max_iterations(mut self, iterations: u64) -> Self {
        self.max_iterations = iterations;
        self
    }

    /// Set maximum input length
    pub fn with_max_len(mut self, max_len: usize) -> Self {
        self.max_len = Some(max_len);
        self
    }
}

/// LibFuzzer verification backend for coverage-guided fuzzing
///
/// LibFuzzer is a coverage-guided fuzzer that discovers inputs to trigger
/// crashes and assertion failures in code. It's effective at finding:
/// - Panics from unexpected inputs
/// - Buffer overflows
/// - Integer overflows
/// - Logic errors
///
/// # Requirements
///
/// - `cargo-fuzz` installed (`cargo install cargo-fuzz`)
/// - Nightly Rust toolchain
/// - Fuzz targets defined in `fuzz/fuzz_targets/`
pub struct LibFuzzerBackend {
    config: LibFuzzerConfig,
}

impl LibFuzzerBackend {
    /// Create a new LibFuzzer backend with default configuration
    pub fn new() -> Self {
        Self {
            config: LibFuzzerConfig::default(),
        }
    }

    /// Create a new LibFuzzer backend with custom configuration
    pub fn with_config(config: LibFuzzerConfig) -> Self {
        Self { config }
    }

    /// Run LibFuzzer on a crate target
    pub async fn run_fuzzer(
        &self,
        crate_path: &std::path::Path,
        target: &str,
    ) -> Result<BackendResult, BackendError> {
        let fuzz_config = FuzzConfig::default()
            .with_timeout(self.config.timeout)
            .with_max_iterations(self.config.max_iterations);

        let runner = FuzzRunner::new(FuzzerType::LibFuzzer, fuzz_config);

        // Check if fuzzer is installed first
        match runner.check_installed().await {
            Ok(true) => {}
            Ok(false) => {
                return Err(BackendError::Unavailable(
                    "cargo-fuzz not installed. Run: cargo install cargo-fuzz".to_string(),
                ));
            }
            Err(e) => {
                return Err(BackendError::Unavailable(format!(
                    "Failed to check cargo-fuzz: {}",
                    e
                )));
            }
        }

        let result = runner
            .run_on_target(crate_path, target)
            .await
            .map_err(|e| match e {
                dashprove_fuzz::FuzzError::NotInstalled(name, _) => {
                    BackendError::Unavailable(format!("{} not installed", name))
                }
                dashprove_fuzz::FuzzError::NightlyRequired(name) => {
                    BackendError::Unavailable(format!("Nightly toolchain required for {}", name))
                }
                dashprove_fuzz::FuzzError::TargetNotFound(target) => {
                    BackendError::VerificationFailed(format!("Fuzz target not found: {}", target))
                }
                dashprove_fuzz::FuzzError::BuildFailed(msg) => {
                    BackendError::VerificationFailed(format!("Build failed: {}", msg))
                }
                dashprove_fuzz::FuzzError::FuzzingFailed(msg) => {
                    BackendError::VerificationFailed(format!("Fuzzing failed: {}", msg))
                }
                dashprove_fuzz::FuzzError::Timeout(d) => BackendError::Timeout(d),
                dashprove_fuzz::FuzzError::Io(e) => {
                    BackendError::VerificationFailed(format!("IO error: {}", e))
                }
            })?;

        // Convert FuzzResult to BackendResult
        let status = if result.passed {
            VerificationStatus::Proven
        } else if !result.crashes.is_empty() {
            VerificationStatus::Disproven
        } else {
            VerificationStatus::Unknown {
                reason: "Fuzzing completed but status unclear".to_string(),
            }
        };

        // Build diagnostics
        let mut diagnostics = Vec::new();

        // Summary
        let summary = if result.passed {
            format!(
                "LibFuzzer: No crashes found ({} executions, {:.1} exec/sec)",
                result.executions, result.exec_per_sec
            )
        } else {
            format!(
                "LibFuzzer: {} crashes found ({} executions)",
                result.crashes.len(),
                result.executions
            )
        };
        diagnostics.push(summary);

        // Coverage info
        if result.coverage.regions_covered > 0 {
            diagnostics.push(format!(
                "  Coverage: {} regions covered",
                result.coverage.regions_covered
            ));
        }

        // Add crash details
        for crash in &result.crashes {
            diagnostics.push(format!("  [CRASH] {:?}", crash.crash_type));
            if let Some(ref trace) = crash.stack_trace {
                for line in trace.lines().take(5) {
                    diagnostics.push(format!("    {}", line));
                }
            }
            if let Some(ref artifact) = crash.artifact_path {
                diagnostics.push(format!("  Artifact: {}", artifact.display()));
            }
        }

        // Build counterexample if crashes found
        let counterexample = if !result.crashes.is_empty() {
            let crash_info = result
                .crashes
                .iter()
                .map(|c| {
                    format!(
                        "{:?}: {}",
                        c.crash_type,
                        c.stack_trace.as_deref().unwrap_or("no stack trace")
                    )
                })
                .collect::<Vec<_>>()
                .join("\n---\n");
            Some(StructuredCounterexample::from_raw(crash_info))
        } else {
            None
        };

        Ok(BackendResult {
            backend: BackendId::LibFuzzer,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: result.duration,
        })
    }
}

impl Default for LibFuzzerBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for LibFuzzerBackend {
    fn id(&self) -> BackendId {
        BackendId::LibFuzzer
    }

    fn supports(&self) -> Vec<PropertyType> {
        // LibFuzzer is good at finding panics and memory issues via fuzzing
        vec![PropertyType::PropertyBased]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let crate_path = self.config.crate_path.clone().ok_or_else(|| {
            BackendError::Unavailable(
                "LibFuzzer backend requires crate_path pointing to a Rust crate with fuzz targets"
                    .to_string(),
            )
        })?;

        let target = self.config.target.clone().ok_or_else(|| {
            BackendError::Unavailable(
                "LibFuzzer backend requires a target name specifying the fuzz target".to_string(),
            )
        })?;

        self.run_fuzzer(&crate_path, &target).await
    }

    async fn health_check(&self) -> HealthStatus {
        // Check if cargo-fuzz is installed
        let fuzz_config = FuzzConfig::default();
        let runner = FuzzRunner::new(FuzzerType::LibFuzzer, fuzz_config);

        match runner.check_installed().await {
            Ok(true) => HealthStatus::Healthy,
            Ok(false) => HealthStatus::Unavailable {
                reason: "cargo-fuzz not installed. Run: cargo install cargo-fuzz".to_string(),
            },
            Err(e) => HealthStatus::Unavailable {
                reason: format!("Failed to check cargo-fuzz: {}", e),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_libfuzzer_config_default() {
        let config = LibFuzzerConfig::default();
        assert!(config.crate_path.is_none());
        assert!(config.target.is_none());
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.max_iterations, 0);
    }

    #[test]
    fn test_libfuzzer_config_builder() {
        let config = LibFuzzerConfig::default()
            .with_crate_path(PathBuf::from("/test/path"))
            .with_target("fuzz_parse")
            .with_timeout(Duration::from_secs(120))
            .with_max_iterations(10000)
            .with_max_len(1024);

        assert_eq!(config.crate_path, Some(PathBuf::from("/test/path")));
        assert_eq!(config.target, Some("fuzz_parse".to_string()));
        assert_eq!(config.timeout, Duration::from_secs(120));
        assert_eq!(config.max_iterations, 10000);
        assert_eq!(config.max_len, Some(1024));
    }

    #[test]
    fn test_libfuzzer_backend_id() {
        let backend = LibFuzzerBackend::new();
        assert_eq!(backend.id(), BackendId::LibFuzzer);
    }

    #[test]
    fn test_libfuzzer_supports_property_based() {
        let backend = LibFuzzerBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::PropertyBased));
    }

    #[tokio::test]
    async fn test_libfuzzer_health_check() {
        let backend = LibFuzzerBackend::new();
        let health = backend.health_check().await;
        // Health check should return a valid status
        match health {
            HealthStatus::Healthy => {
                // Expected if cargo-fuzz is installed
            }
            HealthStatus::Unavailable { reason } => {
                // Expected if cargo-fuzz not installed
                assert!(reason.contains("cargo-fuzz") || reason.contains("not installed"));
            }
            HealthStatus::Degraded { .. } => {
                // Also acceptable
            }
        }
    }

    #[tokio::test]
    async fn test_libfuzzer_verify_requires_crate_path() {
        use dashprove_usl::{parse, typecheck};

        let backend = LibFuzzerBackend::new();
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

    #[tokio::test]
    async fn test_libfuzzer_verify_requires_target() {
        use dashprove_usl::{parse, typecheck};

        let config = LibFuzzerConfig::default().with_crate_path(PathBuf::from("/test/path"));
        let backend = LibFuzzerBackend::with_config(config);
        let spec = parse("theorem test { true }").expect("parse");
        let typed_spec = typecheck(spec).expect("typecheck");

        let result = backend.verify(&typed_spec).await;
        assert!(result.is_err());
        if let Err(BackendError::Unavailable(msg)) = result {
            assert!(msg.contains("target"));
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
    // LibFuzzerConfig default proofs
    // =============================================

    /// Verify LibFuzzerConfig default crate_path is None
    #[kani::proof]
    fn proof_libfuzzer_config_default_crate_path_none() {
        let config = LibFuzzerConfig::default();
        kani::assert(
            config.crate_path.is_none(),
            "Default crate_path should be None",
        );
    }

    /// Verify LibFuzzerConfig default target is None
    #[kani::proof]
    fn proof_libfuzzer_config_default_target_none() {
        let config = LibFuzzerConfig::default();
        kani::assert(config.target.is_none(), "Default target should be None");
    }

    /// Verify LibFuzzerConfig default timeout is 60 seconds
    #[kani::proof]
    fn proof_libfuzzer_config_default_timeout() {
        let config = LibFuzzerConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(60),
            "Default timeout should be 60 seconds",
        );
    }

    /// Verify LibFuzzerConfig default max_iterations is 0
    #[kani::proof]
    fn proof_libfuzzer_config_default_max_iterations() {
        let config = LibFuzzerConfig::default();
        kani::assert(
            config.max_iterations == 0,
            "Default max_iterations should be 0",
        );
    }

    /// Verify LibFuzzerConfig default max_len is None
    #[kani::proof]
    fn proof_libfuzzer_config_default_max_len_none() {
        let config = LibFuzzerConfig::default();
        kani::assert(config.max_len.is_none(), "Default max_len should be None");
    }

    // =============================================
    // LibFuzzerConfig builder proofs
    // =============================================

    /// Verify with_crate_path sets crate_path
    #[kani::proof]
    fn proof_libfuzzer_config_with_crate_path() {
        let config = LibFuzzerConfig::default().with_crate_path(PathBuf::from("/test"));
        kani::assert(
            config.crate_path == Some(PathBuf::from("/test")),
            "crate_path should be set",
        );
    }

    /// Verify with_target sets target
    #[kani::proof]
    fn proof_libfuzzer_config_with_target() {
        let config = LibFuzzerConfig::default().with_target("fuzz_test");
        kani::assert(
            config.target == Some("fuzz_test".to_string()),
            "target should be set",
        );
    }

    /// Verify with_timeout sets timeout
    #[kani::proof]
    fn proof_libfuzzer_config_with_timeout() {
        let config = LibFuzzerConfig::default().with_timeout(Duration::from_secs(120));
        kani::assert(
            config.timeout == Duration::from_secs(120),
            "timeout should be set",
        );
    }

    /// Verify with_max_iterations sets max_iterations
    #[kani::proof]
    fn proof_libfuzzer_config_with_max_iterations() {
        let config = LibFuzzerConfig::default().with_max_iterations(10000);
        kani::assert(
            config.max_iterations == 10000,
            "max_iterations should be set",
        );
    }

    /// Verify with_max_len sets max_len
    #[kani::proof]
    fn proof_libfuzzer_config_with_max_len() {
        let config = LibFuzzerConfig::default().with_max_len(1024);
        kani::assert(config.max_len == Some(1024), "max_len should be set");
    }

    // =============================================
    // LibFuzzerBackend constructor proofs
    // =============================================

    /// Verify LibFuzzerBackend::new creates backend with default config
    #[kani::proof]
    fn proof_libfuzzer_backend_new_default_timeout() {
        let backend = LibFuzzerBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(60),
            "New backend should have default timeout",
        );
    }

    /// Verify LibFuzzerBackend::with_config preserves config
    #[kani::proof]
    fn proof_libfuzzer_backend_with_config() {
        let config = LibFuzzerConfig {
            timeout: Duration::from_secs(120),
            max_iterations: 5000,
            ..Default::default()
        };
        let backend = LibFuzzerBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(120),
            "Custom timeout should be preserved",
        );
        kani::assert(
            backend.config.max_iterations == 5000,
            "Custom max_iterations should be preserved",
        );
    }

    /// Verify LibFuzzerBackend implements Default
    #[kani::proof]
    fn proof_libfuzzer_backend_default() {
        let backend = LibFuzzerBackend::default();
        kani::assert(
            backend.id() == BackendId::LibFuzzer,
            "Default backend should have correct ID",
        );
    }

    // =============================================
    // LibFuzzerBackend trait implementation proofs
    // =============================================

    /// Verify LibFuzzerBackend::id returns BackendId::LibFuzzer
    #[kani::proof]
    fn proof_libfuzzer_backend_id() {
        let backend = LibFuzzerBackend::new();
        kani::assert(
            backend.id() == BackendId::LibFuzzer,
            "Backend ID should be LibFuzzer",
        );
    }

    /// Verify LibFuzzerBackend::supports includes PropertyBased
    #[kani::proof]
    fn proof_libfuzzer_backend_supports_property_based() {
        let backend = LibFuzzerBackend::new();
        let supported = backend.supports();
        let mut found = false;
        for pt in supported {
            if pt == PropertyType::PropertyBased {
                found = true;
            }
        }
        kani::assert(found, "Should support PropertyBased");
    }

    /// Verify LibFuzzerBackend::supports returns exactly 1 type
    #[kani::proof]
    fn proof_libfuzzer_backend_supports_count() {
        let backend = LibFuzzerBackend::new();
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
    fn proof_libfuzzer_config_builder_chain() {
        let config = LibFuzzerConfig::default()
            .with_crate_path(PathBuf::from("/test"))
            .with_target("fuzz_parse")
            .with_timeout(Duration::from_secs(180));
        kani::assert(
            config.crate_path == Some(PathBuf::from("/test")),
            "crate_path should be set after chain",
        );
        kani::assert(
            config.target == Some("fuzz_parse".to_string()),
            "target should be set after chain",
        );
        kani::assert(
            config.timeout == Duration::from_secs(180),
            "timeout should be set after chain",
        );
    }

    /// Verify builder with all options set
    #[kani::proof]
    fn proof_libfuzzer_config_builder_all_options() {
        let config = LibFuzzerConfig::default()
            .with_crate_path(PathBuf::from("/path"))
            .with_target("target")
            .with_timeout(Duration::from_secs(300))
            .with_max_iterations(50000)
            .with_max_len(2048);
        kani::assert(config.crate_path.is_some(), "crate_path should be set");
        kani::assert(config.target.is_some(), "target should be set");
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "timeout should be 300s",
        );
        kani::assert(
            config.max_iterations == 50000,
            "max_iterations should be 50000",
        );
        kani::assert(config.max_len == Some(2048), "max_len should be 2048");
    }

    // =============================================
    // Config custom values proofs
    // =============================================

    /// Verify zero timeout is preserved
    #[kani::proof]
    fn proof_libfuzzer_config_zero_timeout() {
        let config = LibFuzzerConfig::default().with_timeout(Duration::ZERO);
        kani::assert(
            config.timeout == Duration::ZERO,
            "Zero timeout should be preserved",
        );
    }

    /// Verify large max_iterations is preserved
    #[kani::proof]
    fn proof_libfuzzer_config_large_iterations() {
        let config = LibFuzzerConfig::default().with_max_iterations(u64::MAX);
        kani::assert(
            config.max_iterations == u64::MAX,
            "Large max_iterations should be preserved",
        );
    }

    /// Verify empty target string is preserved
    #[kani::proof]
    fn proof_libfuzzer_config_empty_target() {
        let config = LibFuzzerConfig::default().with_target("");
        kani::assert(
            config.target == Some(String::new()),
            "Empty target should be preserved",
        );
    }
}
