//! Honggfuzz backend for coverage-guided fuzzing
//!
//! This backend runs Honggfuzz on Rust code via `cargo-hfuzz`.
//! It wraps the `dashprove-fuzz` crate's Honggfuzz support.

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

/// Configuration for Honggfuzz backend
#[derive(Debug, Clone)]
pub struct HonggfuzzConfig {
    /// Path to the crate to fuzz
    pub crate_path: Option<PathBuf>,
    /// Name of the fuzz target
    pub target: Option<String>,
    /// Timeout for fuzzing
    pub timeout: Duration,
    /// Number of parallel threads
    pub threads: usize,
}

impl Default for HonggfuzzConfig {
    fn default() -> Self {
        Self {
            crate_path: None,
            target: None,
            timeout: Duration::from_secs(60),
            threads: 1,
        }
    }
}

impl HonggfuzzConfig {
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

    /// Set number of parallel threads
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }
}

/// Honggfuzz verification backend for coverage-guided fuzzing
///
/// Honggfuzz is a security-oriented fuzzer with hardware feedback support.
/// It's developed by Google and is particularly effective at:
/// - Finding security vulnerabilities
/// - Using hardware counters for coverage feedback
/// - Persistent fuzzing mode for fast iteration
/// - Discovering edge cases through genetic mutations
///
/// # Requirements
///
/// - `honggfuzz` installed (`cargo install honggfuzz`)
/// - Nightly Rust toolchain
/// - Fuzz targets defined for the crate
///
/// # Usage
///
/// ```rust,ignore
/// use dashprove_backends::{HonggfuzzBackend, HonggfuzzConfig};
/// use std::time::Duration;
/// use std::path::PathBuf;
///
/// let config = HonggfuzzConfig::default()
///     .with_crate_path(PathBuf::from("my_crate"))
///     .with_target("fuzz_parse")
///     .with_timeout(Duration::from_secs(300))
///     .with_threads(4);
///
/// let backend = HonggfuzzBackend::with_config(config);
/// ```
pub struct HonggfuzzBackend {
    config: HonggfuzzConfig,
}

impl HonggfuzzBackend {
    /// Create a new Honggfuzz backend with default configuration
    pub fn new() -> Self {
        Self {
            config: HonggfuzzConfig::default(),
        }
    }

    /// Create a new Honggfuzz backend with custom configuration
    pub fn with_config(config: HonggfuzzConfig) -> Self {
        Self { config }
    }

    /// Run Honggfuzz on a crate target
    pub async fn run_fuzzer(
        &self,
        crate_path: &std::path::Path,
        target: &str,
    ) -> Result<BackendResult, BackendError> {
        let fuzz_config = FuzzConfig::default()
            .with_timeout(self.config.timeout)
            .with_jobs(self.config.threads);

        let runner = FuzzRunner::new(FuzzerType::Honggfuzz, fuzz_config);

        // Check if fuzzer is installed first
        match runner.check_installed().await {
            Ok(true) => {}
            Ok(false) => {
                return Err(BackendError::Unavailable(
                    "honggfuzz not installed. Run: cargo install honggfuzz".to_string(),
                ));
            }
            Err(e) => {
                return Err(BackendError::Unavailable(format!(
                    "Failed to check honggfuzz: {}",
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
                "Honggfuzz: No crashes found ({} executions, {:.1} exec/sec)",
                result.executions, result.exec_per_sec
            )
        } else {
            format!(
                "Honggfuzz: {} crashes found ({} executions)",
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

        // Duration and workspace info
        if let Some(ref corpus) = result.corpus_path {
            diagnostics.push(format!("  Workspace: {}", corpus.display()));
        }

        // Add crash details
        for crash in &result.crashes {
            diagnostics.push(format!("  [CRASH] {:?}", crash.crash_type));
            if let Some(ref artifact) = crash.artifact_path {
                diagnostics.push(format!("  Artifact: {}", artifact.display()));
            }
            // Honggfuzz crash inputs are stored as raw bytes
            if !crash.input.is_empty() {
                diagnostics.push(format!("  Input size: {} bytes", crash.input.len()));
            }
        }

        // Build counterexample if crashes found
        let counterexample = if !result.crashes.is_empty() {
            let crash_info = result
                .crashes
                .iter()
                .map(|c| {
                    let input_str = if c.input.is_empty() {
                        "no input".to_string()
                    } else {
                        format!(
                            "{} bytes: {:?}",
                            c.input.len(),
                            &c.input[..c.input.len().min(64)]
                        )
                    };
                    format!("{:?}: {}", c.crash_type, input_str)
                })
                .collect::<Vec<_>>()
                .join("\n---\n");
            Some(StructuredCounterexample::from_raw(crash_info))
        } else {
            None
        };

        Ok(BackendResult {
            backend: BackendId::Honggfuzz,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: result.duration,
        })
    }
}

impl Default for HonggfuzzBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for HonggfuzzBackend {
    fn id(&self) -> BackendId {
        BackendId::Honggfuzz
    }

    fn supports(&self) -> Vec<PropertyType> {
        // Honggfuzz is good at finding crashes and memory issues via fuzzing
        vec![PropertyType::PropertyBased]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let crate_path = self.config.crate_path.clone().ok_or_else(|| {
            BackendError::Unavailable(
                "Honggfuzz backend requires crate_path pointing to a Rust crate".to_string(),
            )
        })?;

        let target = self.config.target.clone().ok_or_else(|| {
            BackendError::Unavailable(
                "Honggfuzz backend requires a target name specifying the fuzz target".to_string(),
            )
        })?;

        self.run_fuzzer(&crate_path, &target).await
    }

    async fn health_check(&self) -> HealthStatus {
        // Check if honggfuzz is installed
        let fuzz_config = FuzzConfig::default();
        let runner = FuzzRunner::new(FuzzerType::Honggfuzz, fuzz_config);

        match runner.check_installed().await {
            Ok(true) => HealthStatus::Healthy,
            Ok(false) => HealthStatus::Unavailable {
                reason: "honggfuzz not installed. Run: cargo install honggfuzz".to_string(),
            },
            Err(e) => HealthStatus::Unavailable {
                reason: format!("Failed to check honggfuzz: {}", e),
            },
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- HonggfuzzConfig Default Tests ----

    /// Verify HonggfuzzConfig::default crate_path is None
    #[kani::proof]
    fn proof_honggfuzz_config_default_crate_path_none() {
        let config = HonggfuzzConfig::default();
        kani::assert(
            config.crate_path.is_none(),
            "Default crate_path should be None",
        );
    }

    /// Verify HonggfuzzConfig::default target is None
    #[kani::proof]
    fn proof_honggfuzz_config_default_target_none() {
        let config = HonggfuzzConfig::default();
        kani::assert(config.target.is_none(), "Default target should be None");
    }

    /// Verify HonggfuzzConfig::default timeout is 60 seconds
    #[kani::proof]
    fn proof_honggfuzz_config_default_timeout() {
        let config = HonggfuzzConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(60),
            "Default timeout should be 60 seconds",
        );
    }

    /// Verify HonggfuzzConfig::default threads is 1
    #[kani::proof]
    fn proof_honggfuzz_config_default_threads() {
        let config = HonggfuzzConfig::default();
        kani::assert(config.threads == 1, "Default threads should be 1");
    }

    // ---- HonggfuzzConfig Builder Tests ----

    /// Verify with_crate_path sets crate_path
    #[kani::proof]
    fn proof_honggfuzz_config_with_crate_path() {
        let config = HonggfuzzConfig::default().with_crate_path(PathBuf::from("/test"));
        kani::assert(
            config.crate_path.is_some(),
            "with_crate_path should set Some",
        );
    }

    /// Verify with_target sets target
    #[kani::proof]
    fn proof_honggfuzz_config_with_target() {
        let config = HonggfuzzConfig::default().with_target("my_target");
        kani::assert(config.target.is_some(), "with_target should set Some");
        let target = config.target.unwrap();
        kani::assert(target == "my_target", "with_target should store value");
    }

    /// Verify with_timeout sets timeout
    #[kani::proof]
    fn proof_honggfuzz_config_with_timeout() {
        let config = HonggfuzzConfig::default().with_timeout(Duration::from_secs(300));
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "with_timeout should set 300 seconds",
        );
    }

    /// Verify with_threads sets threads
    #[kani::proof]
    fn proof_honggfuzz_config_with_threads() {
        let config = HonggfuzzConfig::default().with_threads(8);
        kani::assert(config.threads == 8, "with_threads should set 8");
    }

    /// Verify builder chain preserves earlier values
    #[kani::proof]
    fn proof_honggfuzz_config_builder_chain() {
        let config = HonggfuzzConfig::default()
            .with_timeout(Duration::from_secs(120))
            .with_threads(4);
        kani::assert(
            config.timeout == Duration::from_secs(120),
            "Builder chain should preserve timeout",
        );
        kani::assert(config.threads == 4, "Builder chain should set threads");
    }

    // ---- HonggfuzzBackend Construction Tests ----

    /// Verify HonggfuzzBackend::new creates default config
    #[kani::proof]
    fn proof_honggfuzz_backend_new_default() {
        let backend = HonggfuzzBackend::new();
        kani::assert(
            backend.config.crate_path.is_none(),
            "New backend should have None crate_path",
        );
        kani::assert(
            backend.config.target.is_none(),
            "New backend should have None target",
        );
    }

    /// Verify HonggfuzzBackend::default equals ::new
    #[kani::proof]
    fn proof_honggfuzz_backend_default_equals_new() {
        let default_backend = HonggfuzzBackend::default();
        let new_backend = HonggfuzzBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
        kani::assert(
            default_backend.config.threads == new_backend.config.threads,
            "Default and new should have same threads",
        );
    }

    /// Verify HonggfuzzBackend::with_config stores config
    #[kani::proof]
    fn proof_honggfuzz_backend_with_config() {
        let config = HonggfuzzConfig {
            crate_path: None,
            target: Some("fuzz_target".to_string()),
            timeout: Duration::from_secs(180),
            threads: 2,
        };
        let backend = HonggfuzzBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(180),
            "with_config should store timeout",
        );
        kani::assert(
            backend.config.threads == 2,
            "with_config should store threads",
        );
        kani::assert(
            backend.config.target.is_some(),
            "with_config should store target",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify HonggfuzzBackend::id returns BackendId::Honggfuzz
    #[kani::proof]
    fn proof_honggfuzz_backend_id() {
        let backend = HonggfuzzBackend::new();
        kani::assert(
            backend.id() == BackendId::Honggfuzz,
            "Backend ID should be Honggfuzz",
        );
    }

    /// Verify HonggfuzzBackend::supports includes PropertyBased
    #[kani::proof]
    fn proof_honggfuzz_supports_property_based() {
        let backend = HonggfuzzBackend::new();
        let supported = backend.supports();
        let has_property_based = supported.iter().any(|p| *p == PropertyType::PropertyBased);
        kani::assert(has_property_based, "Should support PropertyBased");
    }

    /// Verify supports returns exactly 1 property type
    #[kani::proof]
    fn proof_honggfuzz_supports_length() {
        let backend = HonggfuzzBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 1,
            "Should support exactly 1 property type",
        );
    }

    // ---- Config Value Range Tests ----

    /// Verify threads is preserved across builder
    #[kani::proof]
    fn proof_honggfuzz_threads_preserved() {
        let threads: usize = kani::any();
        kani::assume(threads > 0 && threads <= 64);
        let config = HonggfuzzConfig::default().with_threads(threads);
        kani::assert(config.threads == threads, "Threads should be preserved");
    }

    /// Verify timeout seconds is preserved
    #[kani::proof]
    fn proof_honggfuzz_timeout_preserved() {
        let secs: u64 = kani::any();
        kani::assume(secs > 0 && secs <= 3600);
        let config = HonggfuzzConfig::default().with_timeout(Duration::from_secs(secs));
        kani::assert(
            config.timeout == Duration::from_secs(secs),
            "Timeout should be preserved",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_honggfuzz_config_default() {
        let config = HonggfuzzConfig::default();
        assert!(config.crate_path.is_none());
        assert!(config.target.is_none());
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.threads, 1);
    }

    #[test]
    fn test_honggfuzz_config_builder() {
        let config = HonggfuzzConfig::default()
            .with_crate_path(PathBuf::from("/test/path"))
            .with_target("fuzz_parse")
            .with_timeout(Duration::from_secs(300))
            .with_threads(8);

        assert_eq!(config.crate_path, Some(PathBuf::from("/test/path")));
        assert_eq!(config.target, Some("fuzz_parse".to_string()));
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert_eq!(config.threads, 8);
    }

    #[test]
    fn test_honggfuzz_backend_id() {
        let backend = HonggfuzzBackend::new();
        assert_eq!(backend.id(), BackendId::Honggfuzz);
    }

    #[test]
    fn test_honggfuzz_supports_property_based() {
        let backend = HonggfuzzBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::PropertyBased));
    }

    #[tokio::test]
    async fn test_honggfuzz_health_check() {
        let backend = HonggfuzzBackend::new();
        let health = backend.health_check().await;
        // Health check should return a valid status
        match health {
            HealthStatus::Healthy => {
                // Expected if honggfuzz is installed
            }
            HealthStatus::Unavailable { reason } => {
                // Expected if honggfuzz not installed
                assert!(reason.contains("honggfuzz") || reason.contains("not installed"));
            }
            HealthStatus::Degraded { .. } => {
                // Also acceptable
            }
        }
    }

    #[tokio::test]
    async fn test_honggfuzz_verify_requires_crate_path() {
        use dashprove_usl::{parse, typecheck};

        let backend = HonggfuzzBackend::new();
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
    async fn test_honggfuzz_verify_requires_target() {
        use dashprove_usl::{parse, typecheck};

        let config = HonggfuzzConfig::default().with_crate_path(PathBuf::from("/test/path"));
        let backend = HonggfuzzBackend::with_config(config);
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
