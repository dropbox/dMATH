//! AFL backend for coverage-guided fuzzing
//!
//! This backend runs American Fuzzy Lop (AFL) on Rust code via `cargo-afl`.
//! It wraps the `dashprove-fuzz` crate's AFL support.

// =============================================
// Kani Proofs for AFL Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- AflConfig Default Tests ----

    /// Verify AflConfig::default sets correct baseline values
    #[kani::proof]
    fn proof_afl_config_defaults() {
        let config = AflConfig::default();
        kani::assert(
            config.crate_path.is_none(),
            "crate_path should default to None",
        );
        kani::assert(config.target.is_none(), "target should default to None");
        kani::assert(
            config.timeout == Duration::from_secs(60),
            "timeout should default to 60 seconds",
        );
        kani::assert(
            config.seed_corpus.is_none(),
            "seed_corpus should default to None",
        );
        kani::assert(config.jobs == 1, "jobs should default to 1");
    }

    // ---- AflConfig Builder Tests ----

    /// Verify with_crate_path updates crate_path
    #[kani::proof]
    fn proof_afl_config_with_crate_path() {
        let config = AflConfig::default().with_crate_path(PathBuf::from("/test"));
        kani::assert(
            config.crate_path == Some(PathBuf::from("/test")),
            "with_crate_path should set crate_path",
        );
    }

    /// Verify with_target updates target
    #[kani::proof]
    fn proof_afl_config_with_target() {
        let config = AflConfig::default().with_target("fuzz_test");
        kani::assert(
            config.target == Some("fuzz_test".to_string()),
            "with_target should set target",
        );
    }

    /// Verify with_timeout updates timeout
    #[kani::proof]
    fn proof_afl_config_with_timeout() {
        let config = AflConfig::default().with_timeout(Duration::from_secs(300));
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "with_timeout should set timeout",
        );
    }

    /// Verify with_seed_corpus updates seed_corpus
    #[kani::proof]
    fn proof_afl_config_with_seed_corpus() {
        let config = AflConfig::default().with_seed_corpus(PathBuf::from("/seeds"));
        kani::assert(
            config.seed_corpus == Some(PathBuf::from("/seeds")),
            "with_seed_corpus should set seed_corpus",
        );
    }

    /// Verify with_jobs updates jobs
    #[kani::proof]
    fn proof_afl_config_with_jobs() {
        let config = AflConfig::default().with_jobs(4);
        kani::assert(config.jobs == 4, "with_jobs should set jobs");
    }

    /// Verify builder chaining works correctly
    #[kani::proof]
    fn proof_afl_config_builder_chain() {
        let config = AflConfig::default()
            .with_crate_path(PathBuf::from("/crate"))
            .with_target("fuzz")
            .with_timeout(Duration::from_secs(120))
            .with_seed_corpus(PathBuf::from("/corpus"))
            .with_jobs(8);

        kani::assert(
            config.crate_path == Some(PathBuf::from("/crate")),
            "crate_path should be set after chain",
        );
        kani::assert(
            config.target == Some("fuzz".to_string()),
            "target should be set after chain",
        );
        kani::assert(
            config.timeout == Duration::from_secs(120),
            "timeout should be set after chain",
        );
        kani::assert(
            config.seed_corpus == Some(PathBuf::from("/corpus")),
            "seed_corpus should be set after chain",
        );
        kani::assert(config.jobs == 8, "jobs should be set after chain");
    }

    // ---- AflBackend Construction Tests ----

    /// Verify AflBackend::new uses default configuration
    #[kani::proof]
    fn proof_afl_backend_new_defaults() {
        let backend = AflBackend::new();
        kani::assert(
            backend.config.crate_path.is_none(),
            "new backend should have no crate_path",
        );
        kani::assert(
            backend.config.target.is_none(),
            "new backend should have no target",
        );
        kani::assert(
            backend.config.timeout == Duration::from_secs(60),
            "new backend should default timeout to 60 seconds",
        );
        kani::assert(
            backend.config.jobs == 1,
            "new backend should default jobs to 1",
        );
    }

    /// Verify AflBackend::default equals AflBackend::new
    #[kani::proof]
    fn proof_afl_backend_default_equals_new() {
        let default_backend = AflBackend::default();
        let new_backend = AflBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "default and new should share timeout",
        );
        kani::assert(
            default_backend.config.jobs == new_backend.config.jobs,
            "default and new should share jobs",
        );
    }

    /// Verify AflBackend::with_config preserves custom configuration
    #[kani::proof]
    fn proof_afl_backend_with_config() {
        let config = AflConfig {
            crate_path: Some(PathBuf::from("/work")),
            target: Some("fuzz_parser".to_string()),
            timeout: Duration::from_secs(600),
            seed_corpus: Some(PathBuf::from("/seeds")),
            jobs: 16,
        };
        let backend = AflBackend::with_config(config);
        kani::assert(
            backend.config.crate_path == Some(PathBuf::from("/work")),
            "with_config should preserve crate_path",
        );
        kani::assert(
            backend.config.target == Some("fuzz_parser".to_string()),
            "with_config should preserve target",
        );
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "with_config should preserve timeout",
        );
        kani::assert(
            backend.config.seed_corpus == Some(PathBuf::from("/seeds")),
            "with_config should preserve seed_corpus",
        );
        kani::assert(
            backend.config.jobs == 16,
            "with_config should preserve jobs",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify id() returns BackendId::AFL
    #[kani::proof]
    fn proof_afl_backend_id() {
        let backend = AflBackend::new();
        kani::assert(
            backend.id() == BackendId::AFL,
            "AflBackend id should be BackendId::AFL",
        );
    }

    /// Verify supports() includes PropertyBased
    #[kani::proof]
    fn proof_afl_backend_supports() {
        let backend = AflBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::PropertyBased),
            "supports should include PropertyBased",
        );
    }

    /// Verify supports() returns exactly one property type
    #[kani::proof]
    fn proof_afl_backend_supports_length() {
        let backend = AflBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 1,
            "AFL should support exactly one property type",
        );
    }
}

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

/// Configuration for AFL backend
#[derive(Debug, Clone)]
pub struct AflConfig {
    /// Path to the crate to fuzz
    pub crate_path: Option<PathBuf>,
    /// Name of the fuzz target
    pub target: Option<String>,
    /// Timeout for fuzzing
    pub timeout: Duration,
    /// Path to seed corpus directory (required by AFL)
    pub seed_corpus: Option<PathBuf>,
    /// Number of parallel jobs
    pub jobs: usize,
}

impl Default for AflConfig {
    fn default() -> Self {
        Self {
            crate_path: None,
            target: None,
            timeout: Duration::from_secs(60),
            seed_corpus: None,
            jobs: 1,
        }
    }
}

impl AflConfig {
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

    /// Set seed corpus directory
    pub fn with_seed_corpus(mut self, path: PathBuf) -> Self {
        self.seed_corpus = Some(path);
        self
    }

    /// Set number of parallel jobs
    pub fn with_jobs(mut self, jobs: usize) -> Self {
        self.jobs = jobs;
        self
    }
}

/// AFL verification backend for coverage-guided fuzzing
///
/// American Fuzzy Lop (AFL) is a security-oriented fuzzer that uses
/// compile-time instrumentation and genetic algorithms to discover
/// interesting inputs. It's particularly effective at:
/// - Finding security vulnerabilities
/// - Discovering edge cases
/// - Coverage-guided exploration
///
/// # Requirements
///
/// - `cargo-afl` installed (`cargo install afl`)
/// - A seed corpus directory with initial inputs
/// - Fuzz targets defined for the crate
///
/// # Usage
///
/// ```rust,ignore
/// use dashprove_backends::{AflBackend, AflConfig};
/// use std::time::Duration;
/// use std::path::PathBuf;
///
/// let config = AflConfig::default()
///     .with_crate_path(PathBuf::from("my_crate"))
///     .with_target("fuzz_parse")
///     .with_seed_corpus(PathBuf::from("my_crate/seeds"))
///     .with_timeout(Duration::from_secs(300));
///
/// let backend = AflBackend::with_config(config);
/// ```
pub struct AflBackend {
    config: AflConfig,
}

impl AflBackend {
    /// Create a new AFL backend with default configuration
    pub fn new() -> Self {
        Self {
            config: AflConfig::default(),
        }
    }

    /// Create a new AFL backend with custom configuration
    pub fn with_config(config: AflConfig) -> Self {
        Self { config }
    }

    /// Run AFL on a crate target
    pub async fn run_fuzzer(
        &self,
        crate_path: &std::path::Path,
        target: &str,
    ) -> Result<BackendResult, BackendError> {
        let mut fuzz_config = FuzzConfig::default().with_timeout(self.config.timeout);

        if let Some(ref corpus) = self.config.seed_corpus {
            fuzz_config = fuzz_config.with_seed_corpus(corpus.clone());
        }

        if self.config.jobs > 1 {
            fuzz_config = fuzz_config.with_jobs(self.config.jobs);
        }

        let runner = FuzzRunner::new(FuzzerType::AFL, fuzz_config);

        // Check if fuzzer is installed first
        match runner.check_installed().await {
            Ok(true) => {}
            Ok(false) => {
                return Err(BackendError::Unavailable(
                    "cargo-afl not installed. Run: cargo install afl".to_string(),
                ));
            }
            Err(e) => {
                return Err(BackendError::Unavailable(format!(
                    "Failed to check cargo-afl: {}",
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
                "AFL: No crashes found ({} executions, {:.1} exec/sec)",
                result.executions, result.exec_per_sec
            )
        } else {
            format!(
                "AFL: {} crashes found ({} executions)",
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
            if let Some(ref artifact) = crash.artifact_path {
                diagnostics.push(format!("  Artifact: {}", artifact.display()));
            }
            // AFL crash inputs are stored as raw bytes
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
            backend: BackendId::AFL,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: result.duration,
        })
    }
}

impl Default for AflBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for AflBackend {
    fn id(&self) -> BackendId {
        BackendId::AFL
    }

    fn supports(&self) -> Vec<PropertyType> {
        // AFL is good at finding crashes and memory issues via fuzzing
        vec![PropertyType::PropertyBased]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let crate_path = self.config.crate_path.clone().ok_or_else(|| {
            BackendError::Unavailable(
                "AFL backend requires crate_path pointing to a Rust crate".to_string(),
            )
        })?;

        let target = self.config.target.clone().ok_or_else(|| {
            BackendError::Unavailable(
                "AFL backend requires a target name specifying the fuzz target".to_string(),
            )
        })?;

        self.run_fuzzer(&crate_path, &target).await
    }

    async fn health_check(&self) -> HealthStatus {
        // Check if cargo-afl is installed
        let fuzz_config = FuzzConfig::default();
        let runner = FuzzRunner::new(FuzzerType::AFL, fuzz_config);

        match runner.check_installed().await {
            Ok(true) => HealthStatus::Healthy,
            Ok(false) => HealthStatus::Unavailable {
                reason: "cargo-afl not installed. Run: cargo install afl".to_string(),
            },
            Err(e) => HealthStatus::Unavailable {
                reason: format!("Failed to check cargo-afl: {}", e),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_afl_config_default() {
        let config = AflConfig::default();
        assert!(config.crate_path.is_none());
        assert!(config.target.is_none());
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert!(config.seed_corpus.is_none());
        assert_eq!(config.jobs, 1);
    }

    #[test]
    fn test_afl_config_builder() {
        let config = AflConfig::default()
            .with_crate_path(PathBuf::from("/test/path"))
            .with_target("fuzz_parse")
            .with_timeout(Duration::from_secs(300))
            .with_seed_corpus(PathBuf::from("/test/seeds"))
            .with_jobs(4);

        assert_eq!(config.crate_path, Some(PathBuf::from("/test/path")));
        assert_eq!(config.target, Some("fuzz_parse".to_string()));
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert_eq!(config.seed_corpus, Some(PathBuf::from("/test/seeds")));
        assert_eq!(config.jobs, 4);
    }

    #[test]
    fn test_afl_backend_id() {
        let backend = AflBackend::new();
        assert_eq!(backend.id(), BackendId::AFL);
    }

    #[test]
    fn test_afl_supports_property_based() {
        let backend = AflBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::PropertyBased));
    }

    #[tokio::test]
    async fn test_afl_health_check() {
        let backend = AflBackend::new();
        let health = backend.health_check().await;
        // Health check should return a valid status
        match health {
            HealthStatus::Healthy => {
                // Expected if cargo-afl is installed
            }
            HealthStatus::Unavailable { reason } => {
                // Expected if cargo-afl not installed
                assert!(reason.contains("cargo-afl") || reason.contains("not installed"));
            }
            HealthStatus::Degraded { .. } => {
                // Also acceptable
            }
        }
    }

    #[tokio::test]
    async fn test_afl_verify_requires_crate_path() {
        use dashprove_usl::{parse, typecheck};

        let backend = AflBackend::new();
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
    async fn test_afl_verify_requires_target() {
        use dashprove_usl::{parse, typecheck};

        let config = AflConfig::default().with_crate_path(PathBuf::from("/test/path"));
        let backend = AflBackend::with_config(config);
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
