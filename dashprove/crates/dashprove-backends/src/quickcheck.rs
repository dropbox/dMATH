//! QuickCheck backend for property-based testing
//!
//! This backend runs QuickCheck-style property-based tests to find counterexamples
//! that violate specified properties. QuickCheck generates random inputs and shrinks
//! failing cases to minimal reproducible examples.

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

/// Configuration for QuickCheck backend
#[derive(Debug, Clone)]
pub struct QuickCheckConfig {
    /// Path to the crate to analyze
    pub crate_path: Option<PathBuf>,
    /// Test filter pattern
    pub test_filter: Option<String>,
    /// Timeout for analysis
    pub timeout: Duration,
    /// Number of test cases to generate
    pub tests: u32,
    /// Maximum size of generated values
    pub max_size: u32,
    /// Random seed (for reproducibility)
    pub seed: Option<u64>,
    /// Enable verbose output
    pub verbose: bool,
}

impl Default for QuickCheckConfig {
    fn default() -> Self {
        Self {
            crate_path: None,
            test_filter: None,
            timeout: Duration::from_secs(300),
            tests: 100,
            max_size: 100,
            seed: None,
            verbose: false,
        }
    }
}

impl QuickCheckConfig {
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

    /// Set number of test cases
    pub fn with_tests(mut self, tests: u32) -> Self {
        self.tests = tests;
        self
    }

    /// Set maximum size of generated values
    pub fn with_max_size(mut self, size: u32) -> Self {
        self.max_size = size;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Enable verbose output
    pub fn with_verbose(mut self) -> Self {
        self.verbose = true;
        self
    }
}

/// QuickCheck verification backend for property-based testing
///
/// QuickCheck generates random inputs to test properties and automatically
/// shrinks failing cases to minimal counterexamples. This is effective for
/// finding edge cases that unit tests miss.
///
/// # Requirements
///
/// Add QuickCheck to your crate:
/// ```toml
/// [dev-dependencies]
/// quickcheck = "1.0"
/// quickcheck_macros = "1.0"
/// ```
///
/// Write property tests using `#[quickcheck]` attribute or `quickcheck!` macro.
pub struct QuickCheckBackend {
    config: QuickCheckConfig,
}

impl QuickCheckBackend {
    /// Create a new QuickCheck backend with default configuration
    pub fn new() -> Self {
        Self {
            config: QuickCheckConfig::default(),
        }
    }

    /// Create a new QuickCheck backend with custom configuration
    pub fn with_config(config: QuickCheckConfig) -> Self {
        Self { config }
    }

    /// Run QuickCheck tests on a crate
    pub async fn analyze_crate(&self, crate_path: &Path) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        // Build test command with quickcheck configuration
        let mut args = vec!["test"];

        // Add test filter if specified
        if let Some(ref filter) = self.config.test_filter {
            args.push(filter);
        }

        // Append nocapture for better output if verbose
        if self.config.verbose {
            args.push("--");
            args.push("--nocapture");
        }

        // Set QuickCheck environment variables
        let mut cmd = Command::new("cargo");
        cmd.args(&args).current_dir(crate_path);

        cmd.env("QUICKCHECK_TESTS", self.config.tests.to_string());
        cmd.env("QUICKCHECK_MAX_TESTS", self.config.tests.to_string());
        cmd.env("QUICKCHECK_MAX_SIZE", self.config.max_size.to_string());

        if let Some(seed) = self.config.seed {
            cmd.env("QUICKCHECK_GENERATOR_SEED", seed.to_string());
        }

        let output = cmd.output().await.map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to run QuickCheck tests: {}", e))
        })?;

        let duration = start.elapsed();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined = format!("{}\n{}", stdout, stderr);

        // Parse QuickCheck output
        let (status, findings, shrunk) =
            self.parse_quickcheck_output(&combined, output.status.success());

        // Build diagnostics
        let mut diagnostics = Vec::new();

        // Summary
        let summary = match &status {
            VerificationStatus::Proven => {
                format!("QuickCheck: {} test cases passed", self.config.tests)
            }
            VerificationStatus::Disproven => {
                let shrunk_msg = if shrunk {
                    " (shrunk to minimal counterexample)"
                } else {
                    ""
                };
                format!(
                    "QuickCheck: {} property violations found{}",
                    findings.len(),
                    shrunk_msg
                )
            }
            VerificationStatus::Partial {
                verified_percentage,
            } => {
                format!("QuickCheck: {:.1}% of tests passed", verified_percentage)
            }
            VerificationStatus::Unknown { reason } => format!("QuickCheck: {}", reason),
        };
        diagnostics.push(summary);
        diagnostics.extend(findings.clone());

        // Build counterexample if failures found
        let counterexample = if !findings.is_empty() {
            Some(StructuredCounterexample::from_raw(combined.clone()))
        } else {
            None
        };

        Ok(BackendResult {
            backend: BackendId::QuickCheck,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: duration,
        })
    }

    fn parse_quickcheck_output(
        &self,
        output: &str,
        success: bool,
    ) -> (VerificationStatus, Vec<String>, bool) {
        let mut findings = Vec::new();
        let mut shrunk = false;

        // Look for QuickCheck-specific failure patterns
        for line in output.lines() {
            // QuickCheck failures include:
            // - "[quickcheck] TEST FAILED"
            // - "shrinking"
            // - "thread 'quickcheck_test' panicked"
            // - counterexample values
            if line.contains("[quickcheck]")
                || line.contains("shrink")
                || line.contains("Shrunk")
                || (line.contains("panicked") && line.contains("quickcheck"))
                || line.contains("falsified")
                || line.contains("FAILED")
            {
                findings.push(line.trim().to_string());

                if line.contains("shrink") || line.contains("Shrunk") {
                    shrunk = true;
                }
            }
        }

        let status = if findings.is_empty() && success {
            VerificationStatus::Proven
        } else if !findings.is_empty() {
            VerificationStatus::Disproven
        } else if !success {
            // Check for quickcheck not being present
            if output.contains("unresolved import") && output.contains("quickcheck") {
                return (
                    VerificationStatus::Unknown {
                        reason: "QuickCheck dependency not found in crate".to_string(),
                    },
                    Vec::new(),
                    false,
                );
            }
            VerificationStatus::Unknown {
                reason: "Test execution failed".to_string(),
            }
        } else {
            VerificationStatus::Proven
        };

        (status, findings, shrunk)
    }

    /// Check if QuickCheck tests can run
    pub async fn check_installed(&self) -> Result<bool, BackendError> {
        // QuickCheck is a library dependency, not a cargo tool
        Ok(true)
    }
}

impl Default for QuickCheckBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for QuickCheckBackend {
    fn id(&self) -> BackendId {
        BackendId::QuickCheck
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::PropertyBased]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let crate_path = self.config.crate_path.clone().ok_or_else(|| {
            BackendError::Unavailable(
                "QuickCheck backend requires crate_path pointing to a Rust crate with QuickCheck tests"
                    .to_string(),
            )
        })?;

        self.analyze_crate(&crate_path).await
    }

    async fn health_check(&self) -> HealthStatus {
        // QuickCheck is a library, not a tool
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

    // ===== QuickCheckConfig defaults =====

    #[kani::proof]
    fn verify_config_defaults_timeout() {
        let config = QuickCheckConfig::default();
        assert!(config.timeout == Duration::from_secs(300));
    }

    #[kani::proof]
    fn verify_config_defaults_tests() {
        let config = QuickCheckConfig::default();
        assert!(config.tests == 100);
    }

    #[kani::proof]
    fn verify_config_defaults_max_size() {
        let config = QuickCheckConfig::default();
        assert!(config.max_size == 100);
    }

    #[kani::proof]
    fn verify_config_defaults_options() {
        let config = QuickCheckConfig::default();
        assert!(config.crate_path.is_none());
        assert!(config.test_filter.is_none());
        assert!(config.seed.is_none());
        assert!(!config.verbose);
    }

    // ===== QuickCheckConfig builders =====

    #[kani::proof]
    fn verify_config_with_crate_path() {
        let config = QuickCheckConfig::default().with_crate_path(PathBuf::from("/test"));
        assert!(config.crate_path.is_some());
    }

    #[kani::proof]
    fn verify_config_with_test_filter() {
        let config = QuickCheckConfig::default().with_test_filter("test_foo".to_string());
        assert!(config.test_filter == Some("test_foo".to_string()));
    }

    #[kani::proof]
    fn verify_config_with_timeout() {
        let config = QuickCheckConfig::default().with_timeout(Duration::from_secs(60));
        assert!(config.timeout == Duration::from_secs(60));
    }

    #[kani::proof]
    fn verify_config_with_tests() {
        let config = QuickCheckConfig::default().with_tests(200);
        assert!(config.tests == 200);
    }

    #[kani::proof]
    fn verify_config_with_max_size() {
        let config = QuickCheckConfig::default().with_max_size(50);
        assert!(config.max_size == 50);
    }

    #[kani::proof]
    fn verify_config_with_seed() {
        let config = QuickCheckConfig::default().with_seed(42);
        assert!(config.seed == Some(42));
    }

    #[kani::proof]
    fn verify_config_with_verbose() {
        let config = QuickCheckConfig::default().with_verbose();
        assert!(config.verbose);
    }

    // ===== Backend construction =====

    #[kani::proof]
    fn verify_backend_new_uses_defaults() {
        let backend = QuickCheckBackend::new();
        assert!(backend.config.timeout == Duration::from_secs(300));
        assert!(backend.config.tests == 100);
    }

    #[kani::proof]
    fn verify_backend_default_matches_new() {
        let b1 = QuickCheckBackend::new();
        let b2 = QuickCheckBackend::default();
        assert!(b1.config.timeout == b2.config.timeout);
        assert!(b1.config.tests == b2.config.tests);
        assert!(b1.config.max_size == b2.config.max_size);
    }

    #[kani::proof]
    fn verify_backend_with_config_preserves_values() {
        let config = QuickCheckConfig {
            crate_path: Some(PathBuf::from("/test")),
            test_filter: Some("foo".to_string()),
            timeout: Duration::from_secs(60),
            tests: 200,
            max_size: 50,
            seed: Some(123),
            verbose: true,
        };
        let backend = QuickCheckBackend::with_config(config);
        assert!(backend.config.timeout == Duration::from_secs(60));
        assert!(backend.config.tests == 200);
        assert!(backend.config.max_size == 50);
        assert!(backend.config.seed == Some(123));
        assert!(backend.config.verbose);
    }

    // ===== ID and supports =====

    #[kani::proof]
    fn verify_backend_id() {
        let backend = QuickCheckBackend::new();
        assert!(matches!(backend.id(), BackendId::QuickCheck));
    }

    #[kani::proof]
    fn verify_supports_property_based() {
        let backend = QuickCheckBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::PropertyBased));
        assert!(supported.len() == 1);
    }

    // ===== Output parsing =====

    #[kani::proof]
    fn verify_parse_quickcheck_output_clean() {
        let backend = QuickCheckBackend::new();
        let output = "running 5 tests\ntest qc_test_1 ... ok\ntest result: ok. 5 passed; 0 failed";
        let (status, findings, shrunk) = backend.parse_quickcheck_output(output, true);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(findings.is_empty());
        assert!(!shrunk);
    }

    #[kani::proof]
    fn verify_parse_quickcheck_output_failure() {
        let backend = QuickCheckBackend::new();
        let output = "[quickcheck] TEST FAILED\nshrinking to minimal case";
        let (status, findings, shrunk) = backend.parse_quickcheck_output(output, false);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(!findings.is_empty());
        assert!(shrunk);
    }

    #[kani::proof]
    fn verify_parse_quickcheck_output_shrunk() {
        let backend = QuickCheckBackend::new();
        let output = "Shrunk: (0, \"\")";
        let (status, _, shrunk) = backend.parse_quickcheck_output(output, false);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(shrunk);
    }

    #[kani::proof]
    fn verify_parse_quickcheck_output_falsified() {
        let backend = QuickCheckBackend::new();
        let output = "property falsified after 5 tests";
        let (status, findings, _) = backend.parse_quickcheck_output(output, false);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(!findings.is_empty());
    }

    #[kani::proof]
    fn verify_parse_quickcheck_output_failed() {
        let backend = QuickCheckBackend::new();
        let output = "1 FAILED";
        let (status, findings, _) = backend.parse_quickcheck_output(output, false);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(!findings.is_empty());
    }

    #[kani::proof]
    fn verify_parse_quickcheck_output_missing_dep() {
        let backend = QuickCheckBackend::new();
        let output = "error: unresolved import `quickcheck`";
        let (status, _, _) = backend.parse_quickcheck_output(output, false);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[kani::proof]
    fn verify_parse_quickcheck_output_unknown_failure() {
        let backend = QuickCheckBackend::new();
        let output = "some random failure without quickcheck keywords";
        let (status, _, _) = backend.parse_quickcheck_output(output, false);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quickcheck_config_default() {
        let config = QuickCheckConfig::default();
        assert!(config.crate_path.is_none());
        assert!(config.test_filter.is_none());
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert_eq!(config.tests, 100);
        assert_eq!(config.max_size, 100);
        assert!(config.seed.is_none());
        assert!(!config.verbose);
    }

    #[test]
    fn test_quickcheck_config_builder() {
        let config = QuickCheckConfig::default()
            .with_crate_path(PathBuf::from("/test/path"))
            .with_test_filter("test_foo".to_string())
            .with_timeout(Duration::from_secs(120))
            .with_tests(200)
            .with_max_size(50)
            .with_seed(42)
            .with_verbose();

        assert_eq!(config.crate_path, Some(PathBuf::from("/test/path")));
        assert_eq!(config.test_filter, Some("test_foo".to_string()));
        assert_eq!(config.timeout, Duration::from_secs(120));
        assert_eq!(config.tests, 200);
        assert_eq!(config.max_size, 50);
        assert_eq!(config.seed, Some(42));
        assert!(config.verbose);
    }

    #[test]
    fn test_quickcheck_backend_id() {
        let backend = QuickCheckBackend::new();
        assert_eq!(backend.id(), BackendId::QuickCheck);
    }

    #[test]
    fn test_quickcheck_supports_property_based() {
        let backend = QuickCheckBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::PropertyBased));
    }

    #[test]
    fn test_quickcheck_parse_output_clean() {
        let backend = QuickCheckBackend::new();
        let output = "running 5 tests\ntest qc_test_1 ... ok\ntest result: ok. 5 passed; 0 failed";
        let (status, findings, shrunk) = backend.parse_quickcheck_output(output, true);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(findings.is_empty());
        assert!(!shrunk);
    }

    #[test]
    fn test_quickcheck_parse_output_with_failure() {
        let backend = QuickCheckBackend::new();
        let output =
            "[quickcheck] TEST FAILED\nshrinking to minimal case\nShrunk: (0, \"\")\n1 FAILED";
        let (status, findings, shrunk) = backend.parse_quickcheck_output(output, false);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(!findings.is_empty());
        assert!(shrunk);
    }

    #[tokio::test]
    async fn test_quickcheck_health_check() {
        let backend = QuickCheckBackend::new();
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
    async fn test_quickcheck_verify_requires_crate_path() {
        use dashprove_usl::{parse, typecheck};

        let backend = QuickCheckBackend::new();
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
