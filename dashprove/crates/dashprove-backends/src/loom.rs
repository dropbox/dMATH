//! Loom backend for concurrency verification
//!
//! This backend uses Loom to exhaustively test all possible thread interleavings
//! in concurrent Rust code. Loom is a library for testing concurrent code by
//! exploring all possible interleavings of threads.
//!
//! Note: Unlike external tool backends (Clippy, Miri), Loom requires code to be
//! instrumented at compile time. This backend provides integration with the
//! dashprove-async crate's Loom support for verifying concurrent state machines.

use crate::counterexample::StructuredCounterexample;
use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
    VerificationStatus,
};
use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use std::path::PathBuf;
use std::time::Duration;

/// Configuration for Loom backend
#[derive(Debug, Clone)]
pub struct LoomConfig {
    /// Path to the crate to test with Loom
    pub crate_path: Option<PathBuf>,
    /// Maximum number of interleavings to explore
    pub max_interleavings: usize,
    /// Timeout for Loom exploration
    pub timeout: Duration,
    /// Preemption bound (limits scheduling choices)
    pub preemption_bound: Option<usize>,
    /// Test filter pattern
    pub test_filter: Option<String>,
}

impl Default for LoomConfig {
    fn default() -> Self {
        Self {
            crate_path: None,
            max_interleavings: 100_000,
            timeout: Duration::from_secs(300),
            preemption_bound: Some(3), // Loom default
            test_filter: None,
        }
    }
}

impl LoomConfig {
    /// Set the crate path
    pub fn with_crate_path(mut self, path: PathBuf) -> Self {
        self.crate_path = Some(path);
        self
    }

    /// Set maximum interleavings
    pub fn with_max_interleavings(mut self, max: usize) -> Self {
        self.max_interleavings = max;
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set preemption bound
    pub fn with_preemption_bound(mut self, bound: usize) -> Self {
        self.preemption_bound = Some(bound);
        self
    }

    /// Set test filter
    pub fn with_test_filter(mut self, filter: String) -> Self {
        self.test_filter = Some(filter);
        self
    }
}

/// Loom verification backend for concurrency testing
///
/// Loom works by intercepting synchronization primitives (Mutex, Arc, etc.)
/// and systematically exploring all possible thread interleavings to find
/// concurrency bugs like data races, deadlocks, and atomicity violations.
///
/// # Usage
///
/// Loom requires test code to:
/// 1. Use `loom::sync` instead of `std::sync`
/// 2. Run tests with `loom::model(|| { ... })`
/// 3. Build with the `loom` feature enabled
///
/// This backend helps verify Loom-instrumented code by running tests with
/// the Loom feature enabled.
pub struct LoomBackend {
    config: LoomConfig,
}

impl LoomBackend {
    /// Create a new Loom backend with default configuration
    pub fn new() -> Self {
        Self {
            config: LoomConfig::default(),
        }
    }

    /// Create a new Loom backend with custom configuration
    pub fn with_config(config: LoomConfig) -> Self {
        Self { config }
    }

    /// Run Loom tests on a crate
    ///
    /// This runs `cargo test --features loom` on the specified crate path.
    pub async fn run_loom_tests(
        &self,
        crate_path: &std::path::Path,
    ) -> Result<BackendResult, BackendError> {
        use std::process::Stdio;
        use tokio::process::Command;

        // Verify path exists
        if !crate_path.exists() {
            return Err(BackendError::Unavailable(format!(
                "Crate path does not exist: {}",
                crate_path.display()
            )));
        }

        // Find Cargo.toml
        let cargo_toml = if crate_path.is_file() && crate_path.ends_with("Cargo.toml") {
            crate_path.to_path_buf()
        } else {
            let toml = crate_path.join("Cargo.toml");
            if !toml.exists() {
                return Err(BackendError::Unavailable(format!(
                    "No Cargo.toml found at {}",
                    crate_path.display()
                )));
            }
            toml
        };

        let working_dir = cargo_toml.parent().unwrap_or(crate_path);

        // Build command: cargo test --features loom
        let mut cmd = Command::new("cargo");
        cmd.arg("test")
            .arg("--features")
            .arg("loom")
            .arg("--")
            .arg("--test-threads=1") // Loom needs single-threaded runner
            .current_dir(working_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Add test filter if specified
        if let Some(ref filter) = self.config.test_filter {
            cmd.arg(filter);
        }

        // Set Loom environment variables
        if let Some(bound) = self.config.preemption_bound {
            cmd.env("LOOM_MAX_PREEMPTIONS", bound.to_string());
        }
        cmd.env(
            "LOOM_MAX_BRANCHES",
            self.config.max_interleavings.to_string(),
        );

        let start = std::time::Instant::now();
        let output = tokio::time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| BackendError::VerificationFailed(format!("Failed to run cargo: {}", e)))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let duration = start.elapsed();

        // Parse results
        let status = if output.status.success() {
            VerificationStatus::Proven
        } else if stderr.contains("Loom detected")
            || stderr.contains("data race")
            || stderr.contains("deadlock")
            || stderr.contains("PANIC")
            || stderr.contains("panicked")
        {
            VerificationStatus::Disproven
        } else if stderr.contains("error[E")
            || stderr.contains("could not compile")
            || stderr.contains("no 'loom' feature")
        {
            // Compilation error or missing loom feature
            return Err(BackendError::Unavailable(format!(
                "Loom test compilation failed. Ensure the crate has a 'loom' feature: {}",
                stderr.lines().take(5).collect::<Vec<_>>().join("\n")
            )));
        } else {
            VerificationStatus::Unknown {
                reason: "Test result unclear or unexpected exit status".to_string(),
            }
        };

        // Build diagnostics
        let mut diagnostics = Vec::new();

        // Count tests
        let _test_count = stdout.matches("test result:").count();
        let passed_count = stdout
            .lines()
            .filter(|l| l.contains("test ") && l.contains(" ok"))
            .count();
        let failed_count = stdout
            .lines()
            .filter(|l| l.contains("test ") && l.contains(" FAILED"))
            .count();

        let summary = match status {
            VerificationStatus::Proven => {
                format!(
                    "Loom: All {} tests passed (no concurrency bugs found)",
                    passed_count
                )
            }
            VerificationStatus::Disproven => {
                format!(
                    "Loom: {} tests failed (concurrency bug detected)",
                    failed_count
                )
            }
            _ => format!("Loom: {} passed, {} failed", passed_count, failed_count),
        };
        diagnostics.push(summary);

        // Add failure details
        if matches!(status, VerificationStatus::Disproven) {
            for line in stderr.lines() {
                if line.contains("Loom")
                    || line.contains("data race")
                    || line.contains("deadlock")
                    || line.contains("panicked")
                {
                    diagnostics.push(line.to_string());
                }
            }
        }

        // Build counterexample if bug found
        let counterexample = if matches!(status, VerificationStatus::Disproven) {
            Some(StructuredCounterexample::from_raw(stderr.clone()))
        } else {
            None
        };

        Ok(BackendResult {
            backend: BackendId::Loom,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: duration,
        })
    }
}

impl Default for LoomBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for LoomBackend {
    fn id(&self) -> BackendId {
        BackendId::Loom
    }

    fn supports(&self) -> Vec<PropertyType> {
        // Loom verifies concurrency properties (data races, deadlocks)
        vec![
            PropertyType::Invariant,
            PropertyType::DataRace,
            PropertyType::Temporal,
        ]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let crate_path = self.config.crate_path.clone().ok_or_else(|| {
            BackendError::Unavailable(
                "Loom backend requires crate_path pointing to a Rust crate with loom feature"
                    .to_string(),
            )
        })?;

        self.run_loom_tests(&crate_path).await
    }

    async fn health_check(&self) -> HealthStatus {
        // Check if cargo is available
        match which::which("cargo") {
            Ok(_) => {
                // Cargo exists; Loom is a crate dependency, not a system tool
                HealthStatus::Healthy
            }
            Err(_) => HealthStatus::Unavailable {
                reason: "cargo not found. Install Rust toolchain.".to_string(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loom_config_default() {
        let config = LoomConfig::default();
        assert!(config.crate_path.is_none());
        assert_eq!(config.max_interleavings, 100_000);
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert_eq!(config.preemption_bound, Some(3));
        assert!(config.test_filter.is_none());
    }

    #[test]
    fn test_loom_config_builder() {
        let config = LoomConfig::default()
            .with_crate_path(PathBuf::from("/test/path"))
            .with_max_interleavings(50_000)
            .with_timeout(Duration::from_secs(120))
            .with_preemption_bound(5)
            .with_test_filter("test_concurrent".to_string());

        assert_eq!(config.crate_path, Some(PathBuf::from("/test/path")));
        assert_eq!(config.max_interleavings, 50_000);
        assert_eq!(config.timeout, Duration::from_secs(120));
        assert_eq!(config.preemption_bound, Some(5));
        assert_eq!(config.test_filter, Some("test_concurrent".to_string()));
    }

    #[test]
    fn test_loom_backend_id() {
        let backend = LoomBackend::new();
        assert_eq!(backend.id(), BackendId::Loom);
    }

    #[test]
    fn test_loom_supports_concurrency() {
        let backend = LoomBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::DataRace));
        assert!(supported.contains(&PropertyType::Temporal));
        assert!(supported.contains(&PropertyType::Invariant));
    }

    #[tokio::test]
    async fn test_loom_health_check() {
        let backend = LoomBackend::new();
        let health = backend.health_check().await;
        // Should be healthy if cargo is available
        match health {
            HealthStatus::Healthy => {
                // Expected - cargo is available
            }
            HealthStatus::Unavailable { reason } => {
                // Only acceptable if cargo isn't available
                assert!(reason.contains("cargo"));
            }
            HealthStatus::Degraded { .. } => {
                // Also acceptable
            }
        }
    }

    #[tokio::test]
    async fn test_loom_verify_requires_crate_path() {
        use dashprove_usl::{parse, typecheck};

        let backend = LoomBackend::new();
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
    // LoomConfig default proofs
    // =============================================

    /// Verify LoomConfig default crate_path is None
    #[kani::proof]
    fn proof_loom_config_default_crate_path_none() {
        let config = LoomConfig::default();
        kani::assert(
            config.crate_path.is_none(),
            "Default crate_path should be None",
        );
    }

    /// Verify LoomConfig default max_interleavings is 100_000
    #[kani::proof]
    fn proof_loom_config_default_max_interleavings() {
        let config = LoomConfig::default();
        kani::assert(
            config.max_interleavings == 100_000,
            "Default max_interleavings should be 100_000",
        );
    }

    /// Verify LoomConfig default timeout is 300 seconds
    #[kani::proof]
    fn proof_loom_config_default_timeout() {
        let config = LoomConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "Default timeout should be 300 seconds",
        );
    }

    /// Verify LoomConfig default preemption_bound is Some(3)
    #[kani::proof]
    fn proof_loom_config_default_preemption_bound() {
        let config = LoomConfig::default();
        kani::assert(
            config.preemption_bound == Some(3),
            "Default preemption_bound should be Some(3)",
        );
    }

    /// Verify LoomConfig default test_filter is None
    #[kani::proof]
    fn proof_loom_config_default_test_filter_none() {
        let config = LoomConfig::default();
        kani::assert(
            config.test_filter.is_none(),
            "Default test_filter should be None",
        );
    }

    // =============================================
    // LoomConfig builder proofs
    // =============================================

    /// Verify with_crate_path sets crate_path
    #[kani::proof]
    fn proof_loom_config_with_crate_path() {
        let config = LoomConfig::default().with_crate_path(PathBuf::from("/test/path"));
        kani::assert(
            config.crate_path == Some(PathBuf::from("/test/path")),
            "crate_path should be set",
        );
    }

    /// Verify with_max_interleavings sets max_interleavings
    #[kani::proof]
    fn proof_loom_config_with_max_interleavings() {
        let config = LoomConfig::default().with_max_interleavings(50_000);
        kani::assert(
            config.max_interleavings == 50_000,
            "max_interleavings should be 50_000",
        );
    }

    /// Verify with_timeout sets timeout
    #[kani::proof]
    fn proof_loom_config_with_timeout() {
        let config = LoomConfig::default().with_timeout(Duration::from_secs(120));
        kani::assert(
            config.timeout == Duration::from_secs(120),
            "timeout should be 120 seconds",
        );
    }

    /// Verify with_preemption_bound sets preemption_bound
    #[kani::proof]
    fn proof_loom_config_with_preemption_bound() {
        let config = LoomConfig::default().with_preemption_bound(5);
        kani::assert(
            config.preemption_bound == Some(5),
            "preemption_bound should be Some(5)",
        );
    }

    /// Verify with_test_filter sets test_filter
    #[kani::proof]
    fn proof_loom_config_with_test_filter() {
        let config = LoomConfig::default().with_test_filter("test_concurrent".to_string());
        kani::assert(
            config.test_filter == Some("test_concurrent".to_string()),
            "test_filter should be set",
        );
    }

    // =============================================
    // LoomBackend constructor proofs
    // =============================================

    /// Verify LoomBackend::new creates backend with default config
    #[kani::proof]
    fn proof_loom_backend_new_default_timeout() {
        let backend = LoomBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "New backend should have default timeout",
        );
    }

    /// Verify LoomBackend::with_config preserves config
    #[kani::proof]
    fn proof_loom_backend_with_config() {
        let config = LoomConfig {
            timeout: Duration::from_secs(120),
            max_interleavings: 50_000,
            ..Default::default()
        };
        let backend = LoomBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(120),
            "Custom timeout should be preserved",
        );
        kani::assert(
            backend.config.max_interleavings == 50_000,
            "Custom max_interleavings should be preserved",
        );
    }

    /// Verify LoomBackend implements Default
    #[kani::proof]
    fn proof_loom_backend_default() {
        let backend = LoomBackend::default();
        kani::assert(
            backend.id() == BackendId::Loom,
            "Default backend should have correct ID",
        );
    }

    // =============================================
    // LoomBackend trait implementation proofs
    // =============================================

    /// Verify LoomBackend::id returns BackendId::Loom
    #[kani::proof]
    fn proof_loom_backend_id() {
        let backend = LoomBackend::new();
        kani::assert(backend.id() == BackendId::Loom, "Backend ID should be Loom");
    }

    /// Verify LoomBackend::supports includes Invariant
    #[kani::proof]
    fn proof_loom_backend_supports_invariant() {
        let backend = LoomBackend::new();
        let supported = backend.supports();
        let mut found = false;
        for pt in supported {
            if pt == PropertyType::Invariant {
                found = true;
            }
        }
        kani::assert(found, "Should support Invariant");
    }

    /// Verify LoomBackend::supports includes DataRace
    #[kani::proof]
    fn proof_loom_backend_supports_data_race() {
        let backend = LoomBackend::new();
        let supported = backend.supports();
        let mut found = false;
        for pt in supported {
            if pt == PropertyType::DataRace {
                found = true;
            }
        }
        kani::assert(found, "Should support DataRace");
    }

    /// Verify LoomBackend::supports includes Temporal
    #[kani::proof]
    fn proof_loom_backend_supports_temporal() {
        let backend = LoomBackend::new();
        let supported = backend.supports();
        let mut found = false;
        for pt in supported {
            if pt == PropertyType::Temporal {
                found = true;
            }
        }
        kani::assert(found, "Should support Temporal");
    }

    /// Verify LoomBackend::supports returns exactly 3 types
    #[kani::proof]
    fn proof_loom_backend_supports_count() {
        let backend = LoomBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 3,
            "Should support exactly 3 property types",
        );
    }

    // =============================================
    // Builder chaining proofs
    // =============================================

    /// Verify builder methods can be chained
    #[kani::proof]
    fn proof_loom_config_builder_chain() {
        let config = LoomConfig::default()
            .with_crate_path(PathBuf::from("/test"))
            .with_max_interleavings(25_000)
            .with_timeout(Duration::from_secs(60))
            .with_preemption_bound(2)
            .with_test_filter("test_mutex".to_string());
        kani::assert(config.crate_path.is_some(), "crate_path should be set");
        kani::assert(
            config.max_interleavings == 25_000,
            "max_interleavings should be 25_000",
        );
        kani::assert(
            config.timeout == Duration::from_secs(60),
            "timeout should be 60s",
        );
        kani::assert(
            config.preemption_bound == Some(2),
            "preemption_bound should be Some(2)",
        );
        kani::assert(config.test_filter.is_some(), "test_filter should be set");
    }

    // =============================================
    // Config custom values proofs
    // =============================================

    /// Verify zero preemption_bound can be set
    #[kani::proof]
    fn proof_loom_config_zero_preemption_bound() {
        let config = LoomConfig::default().with_preemption_bound(0);
        kani::assert(
            config.preemption_bound == Some(0),
            "Zero preemption_bound should be preserved",
        );
    }

    /// Verify large max_interleavings is preserved
    #[kani::proof]
    fn proof_loom_config_large_interleavings() {
        let config = LoomConfig::default().with_max_interleavings(1_000_000);
        kani::assert(
            config.max_interleavings == 1_000_000,
            "Large max_interleavings should be preserved",
        );
    }

    /// Verify empty test_filter is preserved
    #[kani::proof]
    fn proof_loom_config_empty_test_filter() {
        let config = LoomConfig::default().with_test_filter(String::new());
        kani::assert(
            config.test_filter == Some(String::new()),
            "Empty test_filter should be preserved",
        );
    }

    /// Verify custom config with all options
    #[kani::proof]
    fn proof_loom_config_all_custom() {
        let config = LoomConfig {
            crate_path: Some(PathBuf::from("/custom")),
            max_interleavings: 200_000,
            timeout: Duration::from_secs(600),
            preemption_bound: Some(10),
            test_filter: Some("custom_test".to_string()),
        };
        kani::assert(config.crate_path.is_some(), "crate_path should be Some");
        kani::assert(
            config.max_interleavings == 200_000,
            "max_interleavings correct",
        );
        kani::assert(
            config.timeout == Duration::from_secs(600),
            "timeout correct",
        );
        kani::assert(
            config.preemption_bound == Some(10),
            "preemption_bound correct",
        );
        kani::assert(config.test_filter.is_some(), "test_filter should be Some");
    }
}
