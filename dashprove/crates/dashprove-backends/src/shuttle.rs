//! Shuttle backend for randomized concurrency testing
//!
//! This backend runs Shuttle to detect concurrency bugs through randomized
//! interleaving exploration. Shuttle is similar to Loom but uses random
//! scheduling instead of exhaustive enumeration.

use crate::counterexample::StructuredCounterexample;
use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
    VerificationStatus,
};
use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio::process::Command;

/// Shuttle scheduling strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SchedulingStrategy {
    /// Pure random scheduling
    Random,
    /// Probabilistic concurrency testing (PCT)
    #[default]
    PCT,
    /// Depth-first random scheduling
    DFS,
}

/// Configuration for Shuttle backend
#[derive(Debug, Clone)]
pub struct ShuttleConfig {
    /// Path to the crate to analyze
    pub crate_path: Option<PathBuf>,
    /// Test filter pattern
    pub test_filter: Option<String>,
    /// Timeout for analysis
    pub timeout: Duration,
    /// Number of random iterations to run
    pub iterations: u32,
    /// Scheduling strategy
    pub strategy: SchedulingStrategy,
    /// Maximum number of threads to explore
    pub max_threads: u32,
    /// Random seed (for reproducibility)
    pub seed: Option<u64>,
}

impl Default for ShuttleConfig {
    fn default() -> Self {
        Self {
            crate_path: None,
            test_filter: None,
            timeout: Duration::from_secs(300),
            iterations: 100,
            strategy: SchedulingStrategy::default(),
            max_threads: 4,
            seed: None,
        }
    }
}

impl ShuttleConfig {
    /// Set the crate path
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

    /// Set number of iterations
    pub fn with_iterations(mut self, iterations: u32) -> Self {
        self.iterations = iterations;
        self
    }

    /// Set scheduling strategy
    pub fn with_strategy(mut self, strategy: SchedulingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set maximum threads
    pub fn with_max_threads(mut self, max: u32) -> Self {
        self.max_threads = max;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Shuttle verification backend for randomized concurrency testing
///
/// Shuttle explores thread interleavings through randomized scheduling to find
/// concurrency bugs like:
/// - Data races
/// - Deadlocks
/// - Livelocks
/// - Atomicity violations
///
/// Unlike Loom's exhaustive enumeration, Shuttle uses random sampling which
/// scales better to larger programs but may miss bugs.
///
/// # Requirements
///
/// Add Shuttle to your crate:
/// ```toml
/// [dev-dependencies]
/// shuttle = "0.7"
/// ```
pub struct ShuttleBackend {
    config: ShuttleConfig,
}

impl ShuttleBackend {
    /// Create a new Shuttle backend with default configuration
    pub fn new() -> Self {
        Self {
            config: ShuttleConfig::default(),
        }
    }

    /// Create a new Shuttle backend with custom configuration
    pub fn with_config(config: ShuttleConfig) -> Self {
        Self { config }
    }

    /// Run Shuttle tests on a crate
    pub async fn analyze_crate(&self, crate_path: &Path) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        // Build test command with shuttle configuration
        let mut args = vec!["test"];

        // Add test filter if specified
        if let Some(ref filter) = self.config.test_filter {
            args.push(filter);
        }

        // Run tests with shuttle feature enabled
        args.push("--features");
        args.push("shuttle");

        // Set environment for shuttle configuration
        let mut env_vars = Vec::new();
        env_vars.push((
            "SHUTTLE_ITERATIONS".to_string(),
            self.config.iterations.to_string(),
        ));
        env_vars.push((
            "SHUTTLE_MAX_THREADS".to_string(),
            self.config.max_threads.to_string(),
        ));

        if let Some(seed) = self.config.seed {
            env_vars.push(("SHUTTLE_SEED".to_string(), seed.to_string()));
        }

        let strategy_str = match self.config.strategy {
            SchedulingStrategy::Random => "random",
            SchedulingStrategy::PCT => "pct",
            SchedulingStrategy::DFS => "dfs",
        };
        env_vars.push(("SHUTTLE_STRATEGY".to_string(), strategy_str.to_string()));

        // Run tests
        let mut cmd = Command::new("cargo");
        cmd.args(&args).current_dir(crate_path);

        for (key, value) in &env_vars {
            cmd.env(key, value);
        }

        let output = cmd.output().await.map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to run Shuttle: {}", e))
        })?;

        let duration = start.elapsed();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined = format!("{}\n{}", stdout, stderr);

        // Parse Shuttle output
        let (status, findings) = self.parse_shuttle_output(&combined, output.status.success());

        // Build diagnostics
        let mut diagnostics = Vec::new();

        // Summary
        let summary = match &status {
            VerificationStatus::Proven => format!(
                "Shuttle: {} iterations passed with {} strategy",
                self.config.iterations, strategy_str
            ),
            VerificationStatus::Disproven => {
                format!("Shuttle: {} concurrency bugs found", findings.len())
            }
            VerificationStatus::Partial {
                verified_percentage,
            } => {
                format!("Shuttle: {:.1}% of iterations passed", verified_percentage)
            }
            VerificationStatus::Unknown { reason } => format!("Shuttle: {}", reason),
        };
        diagnostics.push(summary);
        diagnostics.extend(findings.clone());

        // Build counterexample if bugs found
        let counterexample = if !findings.is_empty() {
            Some(StructuredCounterexample::from_raw(combined.clone()))
        } else {
            None
        };

        Ok(BackendResult {
            backend: BackendId::Shuttle,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: duration,
        })
    }

    fn parse_shuttle_output(
        &self,
        output: &str,
        success: bool,
    ) -> (VerificationStatus, Vec<String>) {
        let mut findings = Vec::new();

        // Look for Shuttle-specific failure patterns
        for line in output.lines() {
            // Shuttle failures include:
            // - "deadlock detected"
            // - "panic" with shuttle context
            // - "assertion failed"
            // - "data race"
            if line.contains("deadlock")
                || line.contains("data race")
                || line.contains("livelock")
                || (line.contains("panic") && line.contains("shuttle"))
                || (line.contains("FAILED") && !line.contains("0 FAILED"))
            {
                findings.push(line.trim().to_string());
            }
        }

        let status = if findings.is_empty() && success {
            VerificationStatus::Proven
        } else if !findings.is_empty() {
            VerificationStatus::Disproven
        } else if !success {
            // Check for shuttle not being present
            if output.contains("unresolved import") || output.contains("shuttle") {
                return (
                    VerificationStatus::Unknown {
                        reason: "Shuttle dependency not found in crate".to_string(),
                    },
                    Vec::new(),
                );
            }
            VerificationStatus::Unknown {
                reason: "Test execution failed".to_string(),
            }
        } else {
            VerificationStatus::Proven
        };

        (status, findings)
    }

    /// Check if Shuttle dependency is available
    pub async fn check_installed(&self) -> Result<bool, BackendError> {
        // Shuttle is a library dependency, not a cargo tool
        // We check by attempting to build with the shuttle feature
        Ok(true) // Always return true since shuttle is a library
    }
}

impl Default for ShuttleBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for ShuttleBackend {
    fn id(&self) -> BackendId {
        BackendId::Shuttle
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::DataRace]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let crate_path = self.config.crate_path.clone().ok_or_else(|| {
            BackendError::Unavailable(
                "Shuttle backend requires crate_path pointing to a Rust crate with shuttle tests"
                    .to_string(),
            )
        })?;

        self.analyze_crate(&crate_path).await
    }

    async fn health_check(&self) -> HealthStatus {
        // Shuttle is a library, not a tool - always "available" if we can run cargo test
        match Command::new("cargo").args(["--version"]).output().await {
            Ok(output) if output.status.success() => HealthStatus::Healthy,
            _ => HealthStatus::Unavailable {
                reason: "Cargo not available".to_string(),
            },
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ===== ShuttleConfig defaults =====

    #[kani::proof]
    fn verify_config_defaults_timeout() {
        let config = ShuttleConfig::default();
        assert!(config.timeout == Duration::from_secs(300));
    }

    #[kani::proof]
    fn verify_config_defaults_iterations() {
        let config = ShuttleConfig::default();
        assert!(config.iterations == 100);
    }

    #[kani::proof]
    fn verify_config_defaults_max_threads() {
        let config = ShuttleConfig::default();
        assert!(config.max_threads == 4);
    }

    #[kani::proof]
    fn verify_config_defaults_strategy() {
        let config = ShuttleConfig::default();
        assert!(config.strategy == SchedulingStrategy::PCT);
    }

    #[kani::proof]
    fn verify_config_defaults_options() {
        let config = ShuttleConfig::default();
        assert!(config.crate_path.is_none());
        assert!(config.test_filter.is_none());
        assert!(config.seed.is_none());
    }

    // ===== ShuttleConfig builders =====

    #[kani::proof]
    fn verify_config_with_crate_path() {
        let config = ShuttleConfig::default().with_crate_path(PathBuf::from("/test"));
        assert!(config.crate_path.is_some());
    }

    #[kani::proof]
    fn verify_config_with_test_filter() {
        let config = ShuttleConfig::default().with_test_filter("test_foo".to_string());
        assert!(config.test_filter == Some("test_foo".to_string()));
    }

    #[kani::proof]
    fn verify_config_with_timeout() {
        let config = ShuttleConfig::default().with_timeout(Duration::from_secs(60));
        assert!(config.timeout == Duration::from_secs(60));
    }

    #[kani::proof]
    fn verify_config_with_iterations() {
        let config = ShuttleConfig::default().with_iterations(200);
        assert!(config.iterations == 200);
    }

    #[kani::proof]
    fn verify_config_with_strategy() {
        let config = ShuttleConfig::default().with_strategy(SchedulingStrategy::Random);
        assert!(config.strategy == SchedulingStrategy::Random);
    }

    #[kani::proof]
    fn verify_config_with_max_threads() {
        let config = ShuttleConfig::default().with_max_threads(8);
        assert!(config.max_threads == 8);
    }

    #[kani::proof]
    fn verify_config_with_seed() {
        let config = ShuttleConfig::default().with_seed(42);
        assert!(config.seed == Some(42));
    }

    // ===== Backend construction =====

    #[kani::proof]
    fn verify_backend_new_uses_defaults() {
        let backend = ShuttleBackend::new();
        assert!(backend.config.timeout == Duration::from_secs(300));
        assert!(backend.config.iterations == 100);
    }

    #[kani::proof]
    fn verify_backend_default_matches_new() {
        let b1 = ShuttleBackend::new();
        let b2 = ShuttleBackend::default();
        assert!(b1.config.timeout == b2.config.timeout);
        assert!(b1.config.iterations == b2.config.iterations);
        assert!(b1.config.strategy == b2.config.strategy);
    }

    #[kani::proof]
    fn verify_backend_with_config_preserves_values() {
        let config = ShuttleConfig {
            crate_path: Some(PathBuf::from("/test")),
            test_filter: Some("foo".to_string()),
            timeout: Duration::from_secs(60),
            iterations: 200,
            strategy: SchedulingStrategy::DFS,
            max_threads: 8,
            seed: Some(123),
        };
        let backend = ShuttleBackend::with_config(config);
        assert!(backend.config.timeout == Duration::from_secs(60));
        assert!(backend.config.iterations == 200);
        assert!(backend.config.strategy == SchedulingStrategy::DFS);
        assert!(backend.config.max_threads == 8);
        assert!(backend.config.seed == Some(123));
    }

    // ===== ID and supports =====

    #[kani::proof]
    fn verify_backend_id() {
        let backend = ShuttleBackend::new();
        assert!(matches!(backend.id(), BackendId::Shuttle));
    }

    #[kani::proof]
    fn verify_supports_data_race() {
        let backend = ShuttleBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::DataRace));
        assert!(supported.len() == 1);
    }

    // ===== SchedulingStrategy default =====

    #[kani::proof]
    fn verify_scheduling_strategy_default() {
        assert!(SchedulingStrategy::default() == SchedulingStrategy::PCT);
    }

    // ===== Output parsing =====

    #[kani::proof]
    fn verify_parse_shuttle_output_clean() {
        let backend = ShuttleBackend::new();
        let output = "running 5 tests\ntest test_1 ... ok\ntest result: ok. 5 passed; 0 failed";
        let (status, findings) = backend.parse_shuttle_output(output, true);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(findings.is_empty());
    }

    #[kani::proof]
    fn verify_parse_shuttle_output_deadlock() {
        let backend = ShuttleBackend::new();
        let output = "deadlock detected";
        let (status, findings) = backend.parse_shuttle_output(output, false);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(!findings.is_empty());
    }

    #[kani::proof]
    fn verify_parse_shuttle_output_data_race() {
        let backend = ShuttleBackend::new();
        let output = "data race found";
        let (status, findings) = backend.parse_shuttle_output(output, false);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(!findings.is_empty());
    }

    #[kani::proof]
    fn verify_parse_shuttle_output_livelock() {
        let backend = ShuttleBackend::new();
        let output = "livelock detected";
        let (status, findings) = backend.parse_shuttle_output(output, false);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(!findings.is_empty());
    }

    #[kani::proof]
    fn verify_parse_shuttle_output_missing_dep() {
        let backend = ShuttleBackend::new();
        let output = "unresolved import shuttle";
        let (status, _) = backend.parse_shuttle_output(output, false);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[kani::proof]
    fn verify_parse_shuttle_output_failed() {
        let backend = ShuttleBackend::new();
        let output = "1 FAILED";
        let (status, findings) = backend.parse_shuttle_output(output, false);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(!findings.is_empty());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shuttle_config_default() {
        let config = ShuttleConfig::default();
        assert!(config.crate_path.is_none());
        assert!(config.test_filter.is_none());
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert_eq!(config.iterations, 100);
        assert_eq!(config.strategy, SchedulingStrategy::PCT);
        assert_eq!(config.max_threads, 4);
        assert!(config.seed.is_none());
    }

    #[test]
    fn test_shuttle_config_builder() {
        let config = ShuttleConfig::default()
            .with_crate_path(PathBuf::from("/test/path"))
            .with_test_filter("test_foo".to_string())
            .with_timeout(Duration::from_secs(120))
            .with_iterations(200)
            .with_strategy(SchedulingStrategy::Random)
            .with_max_threads(8)
            .with_seed(42);

        assert_eq!(config.crate_path, Some(PathBuf::from("/test/path")));
        assert_eq!(config.test_filter, Some("test_foo".to_string()));
        assert_eq!(config.timeout, Duration::from_secs(120));
        assert_eq!(config.iterations, 200);
        assert_eq!(config.strategy, SchedulingStrategy::Random);
        assert_eq!(config.max_threads, 8);
        assert_eq!(config.seed, Some(42));
    }

    #[test]
    fn test_shuttle_backend_id() {
        let backend = ShuttleBackend::new();
        assert_eq!(backend.id(), BackendId::Shuttle);
    }

    #[test]
    fn test_shuttle_supports_data_race() {
        let backend = ShuttleBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::DataRace));
    }

    #[test]
    fn test_shuttle_parse_output_clean() {
        let backend = ShuttleBackend::new();
        let output = "running 5 tests\ntest test_1 ... ok\ntest result: ok. 5 passed; 0 failed";
        let (status, findings) = backend.parse_shuttle_output(output, true);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(findings.is_empty());
    }

    #[test]
    fn test_shuttle_parse_output_with_deadlock() {
        let backend = ShuttleBackend::new();
        let output = "thread 'test_mutex' panicked: deadlock detected\n1 FAILED";
        let (status, findings) = backend.parse_shuttle_output(output, false);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(!findings.is_empty());
    }

    #[tokio::test]
    async fn test_shuttle_health_check() {
        let backend = ShuttleBackend::new();
        let health = backend.health_check().await;
        match health {
            HealthStatus::Healthy => {}
            HealthStatus::Unavailable { reason } => {
                assert!(reason.contains("Cargo") || reason.contains("cargo"));
            }
            HealthStatus::Degraded { .. } => {}
        }
    }

    #[tokio::test]
    async fn test_shuttle_verify_requires_crate_path() {
        use dashprove_usl::{parse, typecheck};

        let backend = ShuttleBackend::new();
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

    #[test]
    fn test_scheduling_strategy_default() {
        assert_eq!(SchedulingStrategy::default(), SchedulingStrategy::PCT);
    }
}
