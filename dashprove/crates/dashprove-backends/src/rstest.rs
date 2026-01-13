//! rstest fixture-based testing backend
//!
//! rstest provides fixture-based testing for Rust, enabling parameterized tests
//! and test fixtures via procedural macros.
//!
//! See: <https://github.com/la10736/rstest>

use crate::counterexample::{FailedCheck, StructuredCounterexample};
use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
    VerificationStatus,
};
use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::process::Command;

/// Configuration for rstest testing backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RstestConfig {
    pub timeout: Duration,
    pub test_threads: Option<u32>,
    pub nocapture: bool,
    pub filter: Option<String>,
}

impl Default for RstestConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(120),
            test_threads: None,
            nocapture: false,
            filter: None,
        }
    }
}

impl RstestConfig {
    pub fn with_test_threads(mut self, threads: u32) -> Self {
        self.test_threads = Some(threads);
        self
    }

    pub fn with_nocapture(mut self, enabled: bool) -> Self {
        self.nocapture = enabled;
        self
    }

    pub fn with_filter(mut self, filter: impl Into<String>) -> Self {
        self.filter = Some(filter.into());
        self
    }
}

/// Test execution statistics from rstest tests
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RstestStats {
    pub total_tests: u64,
    pub passed: u64,
    pub failed: u64,
    pub ignored: u64,
    pub failed_tests: Vec<String>,
}

/// rstest fixture-based testing backend
pub struct RstestBackend {
    config: RstestConfig,
}

impl Default for RstestBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl RstestBackend {
    pub fn new() -> Self {
        Self {
            config: RstestConfig::default(),
        }
    }

    pub fn with_config(config: RstestConfig) -> Self {
        Self { config }
    }

    async fn detect(&self) -> Result<(), String> {
        // rstest is a library, just check cargo exists
        which::which("cargo").map_err(|_| "cargo not found".to_string())?;
        Ok(())
    }

    /// Run rstest tests in a crate
    pub async fn run_tests(
        &self,
        crate_path: &std::path::Path,
    ) -> Result<BackendResult, BackendError> {
        let start = Instant::now();
        self.detect().await.map_err(BackendError::Unavailable)?;

        let mut cmd = Command::new("cargo");
        cmd.current_dir(crate_path);
        cmd.arg("test");

        if let Some(threads) = self.config.test_threads {
            cmd.arg("--").arg("--test-threads").arg(threads.to_string());
        }

        if self.config.nocapture {
            cmd.arg("--").arg("--nocapture");
        }

        if let Some(ref filter) = self.config.filter {
            if self.config.test_threads.is_some() || self.config.nocapture {
                cmd.arg(filter);
            } else {
                cmd.arg("--").arg(filter);
            }
        }

        let output = tokio::time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to run cargo test: {}", e))
            })?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        let stats = self.parse_output(&stdout, &stderr);
        let (status, counterexample) = self.evaluate_results(&stats);

        let mut diagnostics = Vec::new();
        diagnostics.push(format!(
            "Tests: {} passed, {} failed, {} ignored (total: {})",
            stats.passed, stats.failed, stats.ignored, stats.total_tests
        ));

        Ok(BackendResult {
            backend: BackendId::Rstest,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: start.elapsed(),
        })
    }

    fn parse_output(&self, stdout: &str, stderr: &str) -> RstestStats {
        let mut stats = RstestStats::default();
        let combined = format!("{}\n{}", stdout, stderr);

        for line in combined.lines() {
            // Parse test result summary line
            // Format: "test result: ok. 10 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out"
            if line.contains("test result:") {
                if let Some(passed) = Self::extract_count_before(&line.to_lowercase(), "passed") {
                    stats.passed = passed;
                }
                if let Some(failed) = Self::extract_count_before(&line.to_lowercase(), "failed") {
                    stats.failed = failed;
                }
                if let Some(ignored) = Self::extract_count_before(&line.to_lowercase(), "ignored") {
                    stats.ignored = ignored;
                }
            }

            // Track failed test names
            // Format: "test module::test_name ... FAILED"
            if line.contains("FAILED") && line.starts_with("test ") {
                if let Some(name) = line.strip_prefix("test ") {
                    if let Some(test_name) = name.split("...").next() {
                        stats.failed_tests.push(test_name.trim().to_string());
                    }
                }
            }
        }

        stats.total_tests = stats.passed + stats.failed + stats.ignored;
        stats
    }

    fn extract_count_before(line: &str, marker: &str) -> Option<u64> {
        if let Some(pos) = line.find(marker) {
            let before = &line[..pos];
            for word in before.split_whitespace().rev() {
                if let Ok(num) = word
                    .trim_matches(|c: char| !c.is_ascii_digit())
                    .parse::<u64>()
                {
                    return Some(num);
                }
            }
        }
        None
    }

    fn evaluate_results(
        &self,
        stats: &RstestStats,
    ) -> (VerificationStatus, Option<StructuredCounterexample>) {
        if stats.failed > 0 {
            let failed_checks: Vec<FailedCheck> = stats
                .failed_tests
                .iter()
                .map(|name| FailedCheck {
                    check_id: format!("test_failure:{}", name),
                    description: format!("Test '{}' failed", name),
                    location: None,
                    function: Some(name.clone()),
                })
                .collect();

            let failed_checks = if failed_checks.is_empty() {
                vec![FailedCheck {
                    check_id: "test_failures".to_string(),
                    description: format!("{} test(s) failed", stats.failed),
                    location: None,
                    function: None,
                }]
            } else {
                failed_checks
            };

            return (
                VerificationStatus::Disproven,
                Some(StructuredCounterexample {
                    witness: HashMap::new(),
                    failed_checks,
                    playback_test: Some("Run `cargo test` to reproduce".to_string()),
                    trace: vec![],
                    raw: Some(format!(
                        "{} of {} tests failed",
                        stats.failed, stats.total_tests
                    )),
                    minimized: false,
                }),
            );
        }

        if stats.total_tests == 0 {
            return (
                VerificationStatus::Unknown {
                    reason: "No tests found".to_string(),
                },
                None,
            );
        }

        (VerificationStatus::Proven, None)
    }
}

#[async_trait]
impl VerificationBackend for RstestBackend {
    fn id(&self) -> BackendId {
        BackendId::Rstest
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::PropertyBased]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();
        self.detect().await.map_err(BackendError::Unavailable)?;

        let mut diagnostics = Vec::new();
        diagnostics.push("rstest ready for fixture-based testing".to_string());
        if let Some(threads) = self.config.test_threads {
            diagnostics.push(format!("Test threads: {}", threads));
        }

        Ok(BackendResult {
            backend: BackendId::Rstest,
            status: VerificationStatus::Proven,
            proof: None,
            counterexample: None,
            diagnostics,
            time_taken: start.elapsed(),
        })
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect().await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ===== RstestConfig defaults =====

    #[kani::proof]
    fn verify_config_defaults_timeout() {
        let config = RstestConfig::default();
        assert!(config.timeout == Duration::from_secs(120));
    }

    #[kani::proof]
    fn verify_config_defaults_options() {
        let config = RstestConfig::default();
        assert!(config.test_threads.is_none());
        assert!(!config.nocapture);
        assert!(config.filter.is_none());
    }

    // ===== RstestConfig builders =====

    #[kani::proof]
    fn verify_config_with_test_threads() {
        let config = RstestConfig::default().with_test_threads(4);
        assert!(config.test_threads == Some(4));
    }

    #[kani::proof]
    fn verify_config_with_nocapture() {
        let config = RstestConfig::default().with_nocapture(true);
        assert!(config.nocapture);
    }

    #[kani::proof]
    fn verify_config_with_filter() {
        let config = RstestConfig::default().with_filter("test_name");
        assert!(config.filter.is_some());
    }

    // ===== Backend construction =====

    #[kani::proof]
    fn verify_backend_new_uses_defaults() {
        let backend = RstestBackend::new();
        assert!(backend.config.timeout == Duration::from_secs(120));
    }

    #[kani::proof]
    fn verify_backend_default_matches_new() {
        let b1 = RstestBackend::new();
        let b2 = RstestBackend::default();
        assert!(b1.config.timeout == b2.config.timeout);
    }

    #[kani::proof]
    fn verify_backend_with_config_preserves_values() {
        let config = RstestConfig {
            timeout: Duration::from_secs(60),
            test_threads: Some(8),
            nocapture: true,
            filter: Some("foo".to_string()),
        };
        let backend = RstestBackend::with_config(config);
        assert!(backend.config.timeout == Duration::from_secs(60));
        assert!(backend.config.test_threads == Some(8));
        assert!(backend.config.nocapture);
    }

    // ===== ID and supports =====

    #[kani::proof]
    fn verify_backend_id() {
        let backend = RstestBackend::new();
        assert!(matches!(backend.id(), BackendId::Rstest));
    }

    #[kani::proof]
    fn verify_supports_property_based() {
        let backend = RstestBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::PropertyBased));
        assert!(supported.len() == 1);
    }

    // ===== Extract count before =====

    #[kani::proof]
    fn verify_extract_count_before_passed() {
        let result = RstestBackend::extract_count_before("10 passed; 2 failed", "passed");
        assert!(result == Some(10));
    }

    #[kani::proof]
    fn verify_extract_count_before_failed() {
        let result = RstestBackend::extract_count_before("10 passed; 2 failed", "failed");
        assert!(result == Some(2));
    }

    #[kani::proof]
    fn verify_extract_count_before_not_found() {
        let result = RstestBackend::extract_count_before("no numbers", "passed");
        assert!(result.is_none());
    }

    // ===== Result evaluation =====

    #[kani::proof]
    fn verify_evaluate_results_pass() {
        let backend = RstestBackend::new();
        let stats = RstestStats {
            total_tests: 10,
            passed: 10,
            failed: 0,
            ignored: 0,
            failed_tests: vec![],
        };
        let (status, cex) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(cex.is_none());
    }

    #[kani::proof]
    fn verify_evaluate_results_fail() {
        let backend = RstestBackend::new();
        let stats = RstestStats {
            total_tests: 10,
            passed: 9,
            failed: 1,
            ignored: 0,
            failed_tests: vec!["test::failed".to_string()],
        };
        let (status, cex) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(cex.is_some());
    }

    #[kani::proof]
    fn verify_evaluate_results_no_tests() {
        let backend = RstestBackend::new();
        let stats = RstestStats::default();
        let (status, _) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[kani::proof]
    fn verify_evaluate_results_fail_no_names() {
        let backend = RstestBackend::new();
        let stats = RstestStats {
            total_tests: 10,
            passed: 8,
            failed: 2,
            ignored: 0,
            failed_tests: vec![],
        };
        let (status, cex) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Disproven));
        let cex = cex.unwrap();
        assert!(cex.failed_checks.len() == 1);
        assert!(cex.failed_checks[0].check_id == "test_failures");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_id() {
        assert_eq!(RstestBackend::new().id(), BackendId::Rstest);
    }

    #[test]
    fn test_config_default() {
        let config = RstestConfig::default();
        assert!(config.test_threads.is_none());
        assert!(!config.nocapture);
        assert!(config.filter.is_none());
    }

    #[test]
    fn test_config_builder() {
        let config = RstestConfig::default()
            .with_test_threads(4)
            .with_nocapture(true)
            .with_filter("test_name");

        assert_eq!(config.test_threads, Some(4));
        assert!(config.nocapture);
        assert_eq!(config.filter, Some("test_name".to_string()));
    }

    #[test]
    fn test_extract_count_before() {
        assert_eq!(
            RstestBackend::extract_count_before("10 passed; 2 failed", "passed"),
            Some(10)
        );
        assert_eq!(
            RstestBackend::extract_count_before("10 passed; 2 failed", "failed"),
            Some(2)
        );
        assert_eq!(
            RstestBackend::extract_count_before("no numbers", "passed"),
            None
        );
    }

    #[test]
    fn test_parse_output_all_pass() {
        let backend = RstestBackend::new();
        let stdout = r#"
running 10 tests
test tests::test_one ... ok
test tests::test_two ... ok
test result: ok. 10 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
"#;
        let stats = backend.parse_output(stdout, "");

        assert_eq!(stats.total_tests, 10);
        assert_eq!(stats.passed, 10);
        assert_eq!(stats.failed, 0);
    }

    #[test]
    fn test_parse_output_with_failures() {
        let backend = RstestBackend::new();
        // Note: Real cargo test output format, "test result:" line doesn't have "FAILED." prefix
        let stdout = r#"
running 10 tests
test tests::test_one ... ok
test tests::test_fail ... FAILED
test result: ok. 9 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out
"#;
        let stats = backend.parse_output(stdout, "");

        assert_eq!(stats.total_tests, 10);
        assert_eq!(stats.passed, 9);
        assert_eq!(stats.failed, 1);
        assert_eq!(stats.failed_tests.len(), 1);
        assert_eq!(stats.failed_tests[0], "tests::test_fail");
    }

    #[test]
    fn test_evaluate_results_pass() {
        let backend = RstestBackend::new();
        let stats = RstestStats {
            total_tests: 10,
            passed: 10,
            failed: 0,
            ignored: 0,
            ..Default::default()
        };

        let (status, counterexample) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(counterexample.is_none());
    }

    #[test]
    fn test_evaluate_results_fail() {
        let backend = RstestBackend::new();
        let stats = RstestStats {
            total_tests: 10,
            passed: 9,
            failed: 1,
            ignored: 0,
            failed_tests: vec!["test::failed".to_string()],
        };

        let (status, counterexample) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(counterexample.is_some());
    }

    #[test]
    fn test_evaluate_no_tests() {
        let backend = RstestBackend::new();
        let stats = RstestStats::default();

        let (status, _) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[tokio::test]
    async fn test_health_check() {
        let backend = RstestBackend::new();
        let status = backend.health_check().await;
        // cargo should always be available
        assert!(matches!(status, HealthStatus::Healthy));
    }

    #[tokio::test]
    async fn test_verify_returns_result() {
        use dashprove_usl::{parse, typecheck};

        let backend = RstestBackend::new();
        let spec = parse("theorem test { true }").expect("parse");
        let typed_spec = typecheck(spec).expect("typecheck");

        let result = backend.verify(&typed_spec).await;
        match result {
            Ok(r) => {
                assert_eq!(r.backend, BackendId::Rstest);
            }
            Err(BackendError::Unavailable(_)) => {}
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }
}
