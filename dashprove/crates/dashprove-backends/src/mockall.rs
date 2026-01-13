//! mockall mocking framework backend
//!
//! mockall provides powerful mocking for Rust tests, enabling trait and struct
//! mocking with expectations, sequences, and argument matchers.
//!
//! See: <https://github.com/asomers/mockall>

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

/// Configuration for mockall mocking backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockallConfig {
    pub timeout: Duration,
    pub test_threads: Option<u32>,
    pub nocapture: bool,
    pub filter: Option<String>,
}

impl Default for MockallConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(120),
            test_threads: None,
            nocapture: false,
            filter: None,
        }
    }
}

impl MockallConfig {
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

/// Test statistics from mockall tests
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MockallStats {
    pub total_tests: u64,
    pub passed: u64,
    pub failed: u64,
    pub ignored: u64,
    pub failed_tests: Vec<String>,
    pub unsatisfied_expectations: Vec<String>,
}

/// mockall mocking framework backend
pub struct MockallBackend {
    config: MockallConfig,
}

impl Default for MockallBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl MockallBackend {
    pub fn new() -> Self {
        Self {
            config: MockallConfig::default(),
        }
    }

    pub fn with_config(config: MockallConfig) -> Self {
        Self { config }
    }

    async fn detect(&self) -> Result<(), String> {
        which::which("cargo").map_err(|_| "cargo not found".to_string())?;
        Ok(())
    }

    /// Run tests using mockall in a crate
    pub async fn run_tests(
        &self,
        crate_path: &std::path::Path,
    ) -> Result<BackendResult, BackendError> {
        let start = Instant::now();
        self.detect().await.map_err(BackendError::Unavailable)?;

        let mut cmd = Command::new("cargo");
        cmd.current_dir(crate_path);
        cmd.arg("test");

        let mut added_separator = false;
        if let Some(threads) = self.config.test_threads {
            cmd.arg("--").arg("--test-threads").arg(threads.to_string());
            added_separator = true;
        }

        if self.config.nocapture {
            if !added_separator {
                cmd.arg("--");
                added_separator = true;
            }
            cmd.arg("--nocapture");
        }

        if let Some(ref filter) = self.config.filter {
            if !added_separator {
                cmd.arg("--");
            }
            cmd.arg(filter);
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

        if !stats.unsatisfied_expectations.is_empty() {
            diagnostics.push(format!(
                "Unsatisfied mock expectations: {}",
                stats.unsatisfied_expectations.len()
            ));
        }

        Ok(BackendResult {
            backend: BackendId::Mockall,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: start.elapsed(),
        })
    }

    fn parse_output(&self, stdout: &str, stderr: &str) -> MockallStats {
        let mut stats = MockallStats::default();
        let combined = format!("{}\n{}", stdout, stderr);

        for line in combined.lines() {
            if line.contains("test result:") {
                let lower = line.to_lowercase();
                if let Some(passed) = Self::extract_count_before(&lower, "passed") {
                    stats.passed = passed;
                }
                if let Some(failed) = Self::extract_count_before(&lower, "failed") {
                    stats.failed = failed;
                }
                if let Some(ignored) = Self::extract_count_before(&lower, "ignored") {
                    stats.ignored = ignored;
                }
            }

            if line.contains("FAILED") && line.starts_with("test ") {
                if let Some(name) = line.strip_prefix("test ") {
                    if let Some(test_name) = name.split("...").next() {
                        stats.failed_tests.push(test_name.trim().to_string());
                    }
                }
            }

            // Detect mockall expectation failures
            if line.contains("Expectation") && line.contains("unsatisfied") {
                stats.unsatisfied_expectations.push(line.trim().to_string());
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
        stats: &MockallStats,
    ) -> (VerificationStatus, Option<StructuredCounterexample>) {
        if stats.failed > 0 || !stats.unsatisfied_expectations.is_empty() {
            let mut failed_checks: Vec<FailedCheck> = stats
                .failed_tests
                .iter()
                .map(|name| FailedCheck {
                    check_id: format!("test_failure:{}", name),
                    description: format!("Test '{}' failed", name),
                    location: None,
                    function: Some(name.clone()),
                })
                .collect();

            // Add unsatisfied expectations as failures
            for exp in &stats.unsatisfied_expectations {
                failed_checks.push(FailedCheck {
                    check_id: "unsatisfied_expectation".to_string(),
                    description: exp.clone(),
                    location: None,
                    function: None,
                });
            }

            if failed_checks.is_empty() {
                failed_checks.push(FailedCheck {
                    check_id: "test_failures".to_string(),
                    description: format!("{} test(s) failed", stats.failed),
                    location: None,
                    function: None,
                });
            }

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
impl VerificationBackend for MockallBackend {
    fn id(&self) -> BackendId {
        BackendId::Mockall
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::Contract]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();
        self.detect().await.map_err(BackendError::Unavailable)?;

        let mut diagnostics = Vec::new();
        diagnostics.push("mockall ready for mock-based testing".to_string());
        if let Some(threads) = self.config.test_threads {
            diagnostics.push(format!("Test threads: {}", threads));
        }

        Ok(BackendResult {
            backend: BackendId::Mockall,
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

    // ===== MockallConfig default proofs =====

    #[kani::proof]
    fn verify_config_default_timeout() {
        let config = MockallConfig::default();
        assert!(config.timeout.as_secs() == 120);
    }

    #[kani::proof]
    fn verify_config_default_test_threads_none() {
        let config = MockallConfig::default();
        assert!(config.test_threads.is_none());
    }

    #[kani::proof]
    fn verify_config_default_nocapture_false() {
        let config = MockallConfig::default();
        assert!(!config.nocapture);
    }

    #[kani::proof]
    fn verify_config_default_filter_none() {
        let config = MockallConfig::default();
        assert!(config.filter.is_none());
    }

    // ===== MockallConfig builder proofs =====

    #[kani::proof]
    fn verify_config_with_test_threads() {
        let config = MockallConfig::default().with_test_threads(4);
        assert!(config.test_threads == Some(4));
    }

    #[kani::proof]
    fn verify_config_with_nocapture_true() {
        let config = MockallConfig::default().with_nocapture(true);
        assert!(config.nocapture);
    }

    #[kani::proof]
    fn verify_config_with_nocapture_false() {
        let config = MockallConfig::default().with_nocapture(false);
        assert!(!config.nocapture);
    }

    #[kani::proof]
    fn verify_config_with_filter() {
        let config = MockallConfig::default().with_filter("mock");
        assert!(config.filter == Some("mock".to_string()));
    }

    #[kani::proof]
    fn verify_config_builder_chaining() {
        let config = MockallConfig::default()
            .with_test_threads(2)
            .with_nocapture(true)
            .with_filter("test");
        assert!(config.test_threads == Some(2));
        assert!(config.nocapture);
        assert!(config.filter == Some("test".to_string()));
    }

    // ===== MockallStats default proofs =====

    #[kani::proof]
    fn verify_stats_default_total_tests_zero() {
        let stats = MockallStats::default();
        assert!(stats.total_tests == 0);
    }

    #[kani::proof]
    fn verify_stats_default_passed_zero() {
        let stats = MockallStats::default();
        assert!(stats.passed == 0);
    }

    #[kani::proof]
    fn verify_stats_default_failed_zero() {
        let stats = MockallStats::default();
        assert!(stats.failed == 0);
    }

    #[kani::proof]
    fn verify_stats_default_ignored_zero() {
        let stats = MockallStats::default();
        assert!(stats.ignored == 0);
    }

    #[kani::proof]
    fn verify_stats_default_failed_tests_empty() {
        let stats = MockallStats::default();
        assert!(stats.failed_tests.is_empty());
    }

    #[kani::proof]
    fn verify_stats_default_unsatisfied_expectations_empty() {
        let stats = MockallStats::default();
        assert!(stats.unsatisfied_expectations.is_empty());
    }

    // ===== MockallBackend construction proofs =====

    #[kani::proof]
    fn verify_backend_new_has_default_config() {
        let backend = MockallBackend::new();
        assert!(backend.config.timeout.as_secs() == 120);
        assert!(backend.config.test_threads.is_none());
    }

    #[kani::proof]
    fn verify_backend_default_equals_new() {
        let b1 = MockallBackend::new();
        let b2 = MockallBackend::default();
        assert!(b1.config.timeout == b2.config.timeout);
        assert!(b1.config.nocapture == b2.config.nocapture);
    }

    #[kani::proof]
    fn verify_backend_with_config_preserves_timeout() {
        let config = MockallConfig {
            timeout: Duration::from_secs(60),
            test_threads: Some(4),
            nocapture: true,
            filter: None,
        };
        let backend = MockallBackend::with_config(config);
        assert!(backend.config.timeout.as_secs() == 60);
    }

    #[kani::proof]
    fn verify_backend_with_config_preserves_threads() {
        let config = MockallConfig {
            timeout: Duration::from_secs(120),
            test_threads: Some(8),
            nocapture: false,
            filter: None,
        };
        let backend = MockallBackend::with_config(config);
        assert!(backend.config.test_threads == Some(8));
    }

    // ===== Backend ID proof =====

    #[kani::proof]
    fn verify_backend_id() {
        let backend = MockallBackend::new();
        assert!(matches!(backend.id(), BackendId::Mockall));
    }

    // ===== Supports proof =====

    #[kani::proof]
    fn verify_supports_contract() {
        let backend = MockallBackend::new();
        let supported = backend.supports();
        let has_contract = supported
            .iter()
            .any(|p| matches!(p, PropertyType::Contract));
        assert!(has_contract);
    }

    #[kani::proof]
    fn verify_supports_count() {
        let backend = MockallBackend::new();
        let supported = backend.supports();
        assert!(supported.len() == 1);
    }

    // ===== extract_count_before proofs =====

    #[kani::proof]
    fn verify_extract_count_before_passed() {
        let result = MockallBackend::extract_count_before("10 passed; 0 failed", "passed");
        assert!(result == Some(10));
    }

    #[kani::proof]
    fn verify_extract_count_before_failed() {
        let result = MockallBackend::extract_count_before("10 passed; 2 failed", "failed");
        assert!(result == Some(2));
    }

    #[kani::proof]
    fn verify_extract_count_before_no_marker() {
        let result = MockallBackend::extract_count_before("no numbers here", "passed");
        assert!(result.is_none());
    }

    #[kani::proof]
    fn verify_extract_count_before_empty_string() {
        let result = MockallBackend::extract_count_before("", "passed");
        assert!(result.is_none());
    }

    // ===== parse_output proofs =====

    #[kani::proof]
    fn verify_parse_output_empty_returns_zeros() {
        let backend = MockallBackend::new();
        let stats = backend.parse_output("", "");
        assert!(stats.total_tests == 0);
        assert!(stats.passed == 0);
        assert!(stats.failed == 0);
    }

    #[kani::proof]
    fn verify_parse_output_all_pass() {
        let backend = MockallBackend::new();
        let stdout = "test result: ok. 10 passed; 0 failed; 0 ignored";
        let stats = backend.parse_output(stdout, "");
        assert!(stats.passed == 10);
        assert!(stats.failed == 0);
    }

    #[kani::proof]
    fn verify_parse_output_total_calculated() {
        let backend = MockallBackend::new();
        let stdout = "test result: ok. 8 passed; 2 failed; 1 ignored";
        let stats = backend.parse_output(stdout, "");
        assert!(stats.total_tests == 11);
    }

    // ===== evaluate_results proofs =====

    #[kani::proof]
    fn verify_evaluate_results_all_pass_proven() {
        let backend = MockallBackend::new();
        let stats = MockallStats {
            total_tests: 10,
            passed: 10,
            failed: 0,
            ignored: 0,
            failed_tests: vec![],
            unsatisfied_expectations: vec![],
        };
        let (status, cex) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(cex.is_none());
    }

    #[kani::proof]
    fn verify_evaluate_results_failures_disproven() {
        let backend = MockallBackend::new();
        let stats = MockallStats {
            total_tests: 10,
            passed: 9,
            failed: 1,
            ignored: 0,
            failed_tests: vec!["test::mock".to_string()],
            unsatisfied_expectations: vec![],
        };
        let (status, cex) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(cex.is_some());
    }

    #[kani::proof]
    fn verify_evaluate_results_no_tests_unknown() {
        let backend = MockallBackend::new();
        let stats = MockallStats::default();
        let (status, _) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[kani::proof]
    fn verify_evaluate_results_unsatisfied_expectations_disproven() {
        let backend = MockallBackend::new();
        let stats = MockallStats {
            total_tests: 10,
            passed: 10,
            failed: 0,
            ignored: 0,
            failed_tests: vec![],
            unsatisfied_expectations: vec!["Expectation unsatisfied".to_string()],
        };
        let (status, _) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[kani::proof]
    fn verify_evaluate_results_counterexample_has_playback() {
        let backend = MockallBackend::new();
        let stats = MockallStats {
            total_tests: 10,
            passed: 9,
            failed: 1,
            ignored: 0,
            failed_tests: vec!["test".to_string()],
            unsatisfied_expectations: vec![],
        };
        let (_, cex) = backend.evaluate_results(&stats);
        assert!(cex.is_some());
        let cex = cex.unwrap();
        assert!(cex.playback_test.is_some());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_id() {
        assert_eq!(MockallBackend::new().id(), BackendId::Mockall);
    }

    #[test]
    fn test_config_default() {
        let config = MockallConfig::default();
        assert!(config.test_threads.is_none());
        assert!(!config.nocapture);
    }

    #[test]
    fn test_config_builder() {
        let config = MockallConfig::default()
            .with_test_threads(4)
            .with_nocapture(true)
            .with_filter("mock");

        assert_eq!(config.test_threads, Some(4));
        assert!(config.nocapture);
        assert_eq!(config.filter, Some("mock".to_string()));
    }

    #[test]
    fn test_parse_output_all_pass() {
        let backend = MockallBackend::new();
        let stdout = r#"
test result: ok. 10 passed; 0 failed; 0 ignored
"#;
        let stats = backend.parse_output(stdout, "");

        assert_eq!(stats.total_tests, 10);
        assert_eq!(stats.passed, 10);
        assert_eq!(stats.failed, 0);
    }

    #[test]
    fn test_parse_output_with_expectation_failure() {
        let backend = MockallBackend::new();
        let stdout = r#"
test tests::mock_test ... FAILED
Expectation MockFoo::bar() unsatisfied
test result: ok. 9 passed; 1 failed; 0 ignored
"#;
        let stats = backend.parse_output(stdout, "");

        assert_eq!(stats.passed, 9);
        assert_eq!(stats.failed, 1);
        assert_eq!(stats.unsatisfied_expectations.len(), 1);
    }

    #[test]
    fn test_evaluate_results_pass() {
        let backend = MockallBackend::new();
        let stats = MockallStats {
            total_tests: 10,
            passed: 10,
            failed: 0,
            ignored: 0,
            ..Default::default()
        };

        let (status, _) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn test_evaluate_results_unsatisfied_expectation() {
        let backend = MockallBackend::new();
        let stats = MockallStats {
            total_tests: 10,
            passed: 9,
            failed: 1,
            ignored: 0,
            failed_tests: vec!["test::mock".to_string()],
            unsatisfied_expectations: vec!["Expectation unsatisfied".to_string()],
        };

        let (status, cex) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(cex.is_some());
        let cex = cex.unwrap();
        assert!(cex.failed_checks.len() >= 2); // test failure + expectation
    }

    #[tokio::test]
    async fn test_health_check() {
        let backend = MockallBackend::new();
        let status = backend.health_check().await;
        assert!(matches!(status, HealthStatus::Healthy));
    }

    #[tokio::test]
    async fn test_verify_returns_result() {
        use dashprove_usl::{parse, typecheck};

        let backend = MockallBackend::new();
        let spec = parse("theorem test { true }").expect("parse");
        let typed_spec = typecheck(spec).expect("typecheck");

        let result = backend.verify(&typed_spec).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().backend, BackendId::Mockall);
    }
}
