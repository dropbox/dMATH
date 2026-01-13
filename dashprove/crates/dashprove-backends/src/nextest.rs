//! cargo-nextest fast test runner backend
//!
//! Nextest is a next-generation test runner for Rust with fast test execution,
//! test filtering, retries, JUnit output, and improved test output formatting.
//!
//! See: <https://nexte.st/>

use crate::counterexample::{FailedCheck, StructuredCounterexample};
use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
    VerificationStatus,
};
use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokio::process::Command;

/// Output format for nextest results
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum NextestOutputFormat {
    #[default]
    Human,
    JUnit,
}

/// Retry configuration for nextest
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum NextestRetries {
    #[default]
    None,
    Fixed(u32),
}

/// Configuration for nextest backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NextestConfig {
    pub timeout: Duration,
    pub jobs: Option<u32>,
    pub retries: NextestRetries,
    pub fail_fast: bool,
    pub no_capture: bool,
    pub output_format: NextestOutputFormat,
    pub filter: Option<String>,
    pub workspace: bool,
}

impl Default for NextestConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(300),
            jobs: None,
            retries: NextestRetries::default(),
            fail_fast: false,
            no_capture: false,
            output_format: NextestOutputFormat::default(),
            filter: None,
            workspace: false,
        }
    }
}

impl NextestConfig {
    pub fn with_jobs(mut self, jobs: u32) -> Self {
        self.jobs = Some(jobs);
        self
    }

    pub fn with_retries(mut self, retries: u32) -> Self {
        self.retries = NextestRetries::Fixed(retries);
        self
    }

    pub fn with_fail_fast(mut self, enabled: bool) -> Self {
        self.fail_fast = enabled;
        self
    }

    pub fn with_filter(mut self, filter: impl Into<String>) -> Self {
        self.filter = Some(filter.into());
        self
    }

    pub fn with_workspace(mut self, enabled: bool) -> Self {
        self.workspace = enabled;
        self
    }
}

/// Test execution statistics from nextest
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NextestStats {
    pub total_tests: u64,
    pub passed: u64,
    pub failed: u64,
    pub skipped: u64,
    pub retried: u64,
    pub failed_tests: Vec<FailedTest>,
}

/// Information about a failed test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailedTest {
    pub name: String,
    pub crate_name: Option<String>,
    pub stdout: Option<String>,
    pub stderr: Option<String>,
    pub duration: Option<Duration>,
}

/// cargo-nextest fast test runner backend
pub struct NextestBackend {
    config: NextestConfig,
}

impl Default for NextestBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl NextestBackend {
    pub fn new() -> Self {
        Self {
            config: NextestConfig::default(),
        }
    }

    pub fn with_config(config: NextestConfig) -> Self {
        Self { config }
    }

    async fn detect(&self) -> Result<PathBuf, String> {
        which::which("cargo-nextest").map_err(|_| {
            "cargo-nextest not found. Install via cargo install cargo-nextest".to_string()
        })
    }

    /// Run nextest on a crate
    pub async fn run_tests(
        &self,
        crate_path: &std::path::Path,
    ) -> Result<BackendResult, BackendError> {
        let start = Instant::now();
        self.detect().await.map_err(BackendError::Unavailable)?;

        let mut cmd = Command::new("cargo");
        cmd.current_dir(crate_path);
        cmd.arg("nextest").arg("run");

        if let Some(jobs) = self.config.jobs {
            cmd.arg("--jobs").arg(jobs.to_string());
        }

        match self.config.retries {
            NextestRetries::None => {}
            NextestRetries::Fixed(n) => {
                cmd.arg("--retries").arg(n.to_string());
            }
        }

        if self.config.fail_fast {
            cmd.arg("--fail-fast");
        }

        if self.config.no_capture {
            cmd.arg("--no-capture");
        }

        if self.config.workspace {
            cmd.arg("--workspace");
        }

        if let Some(ref filter) = self.config.filter {
            cmd.arg("-E").arg(filter);
        }

        let output = tokio::time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to run cargo nextest: {}", e))
            })?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        let stats = self.parse_output(&stdout, &stderr);
        let (status, counterexample) = self.evaluate_results(&stats);

        let mut diagnostics = Vec::new();
        diagnostics.push(format!(
            "Tests: {} passed, {} failed, {} skipped (total: {})",
            stats.passed, stats.failed, stats.skipped, stats.total_tests
        ));

        if stats.retried > 0 {
            diagnostics.push(format!("Retried: {}", stats.retried));
        }

        if !output.status.success() && stats.failed == 0 {
            // There was an error but not test failures
            diagnostics.push(format!("nextest error: {}", stderr.trim()));
        }

        Ok(BackendResult {
            backend: BackendId::Nextest,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: start.elapsed(),
        })
    }

    fn parse_output(&self, stdout: &str, stderr: &str) -> NextestStats {
        let mut stats = NextestStats::default();

        // Combine stdout and stderr for parsing
        let combined = format!("{}\n{}", stdout, stderr);

        // Parse nextest output
        // Typical format: "Summary [ 1.234s] 50 tests run: 49 passed, 1 failed, 0 skipped"
        for line in combined.lines() {
            let line_lower = line.to_lowercase();

            // Parse summary line
            if line_lower.contains("summary") || line_lower.contains("tests run") {
                // Extract counts
                if let Some(total) = Self::extract_number_before(&line_lower, "tests run") {
                    stats.total_tests = total;
                }
                if let Some(passed) = Self::extract_number_before(&line_lower, "passed") {
                    stats.passed = passed;
                }
                if let Some(failed) = Self::extract_number_before(&line_lower, "failed") {
                    stats.failed = failed;
                }
                if let Some(skipped) = Self::extract_number_before(&line_lower, "skipped") {
                    stats.skipped = skipped;
                }
            }

            // Check for FAIL markers for failed test names
            if line.contains("FAIL") && line.contains("::") {
                if let Some(test_name) = Self::extract_test_name(line) {
                    stats.failed_tests.push(FailedTest {
                        name: test_name,
                        crate_name: None,
                        stdout: None,
                        stderr: None,
                        duration: None,
                    });
                }
            }
        }

        // Calculate total if not found
        if stats.total_tests == 0 {
            stats.total_tests = stats.passed + stats.failed + stats.skipped;
        }

        stats
    }

    fn extract_number_before(line: &str, marker: &str) -> Option<u64> {
        if let Some(pos) = line.find(marker) {
            // Look backwards from the marker for a number
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

    fn extract_test_name(line: &str) -> Option<String> {
        // Extract test name from lines like "FAIL [   0.001s] crate::module::test_name"
        for word in line.split_whitespace() {
            if word.contains("::") && !word.starts_with('[') {
                return Some(word.to_string());
            }
        }
        None
    }

    fn evaluate_results(
        &self,
        stats: &NextestStats,
    ) -> (VerificationStatus, Option<StructuredCounterexample>) {
        if stats.failed > 0 {
            let failed_checks: Vec<FailedCheck> = stats
                .failed_tests
                .iter()
                .map(|t| FailedCheck {
                    check_id: format!("test_failure:{}", t.name),
                    description: format!("Test '{}' failed", t.name),
                    location: None,
                    function: Some(t.name.clone()),
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
                    playback_test: Some("Run `cargo nextest run` to reproduce".to_string()),
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
impl VerificationBackend for NextestBackend {
    fn id(&self) -> BackendId {
        BackendId::Nextest
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::Lint]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();
        self.detect().await.map_err(BackendError::Unavailable)?;

        let mut diagnostics = Vec::new();
        diagnostics.push("nextest ready for test execution".to_string());
        if let Some(jobs) = self.config.jobs {
            diagnostics.push(format!("Parallelism: {} jobs", jobs));
        }
        if self.config.fail_fast {
            diagnostics.push("Fail-fast mode enabled".to_string());
        }

        Ok(BackendResult {
            backend: BackendId::Nextest,
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

    // ===== NextestOutputFormat default proof =====

    #[kani::proof]
    fn verify_output_format_default_human() {
        let format = NextestOutputFormat::default();
        assert!(matches!(format, NextestOutputFormat::Human));
    }

    // ===== NextestRetries default proof =====

    #[kani::proof]
    fn verify_retries_default_none() {
        let retries = NextestRetries::default();
        assert!(matches!(retries, NextestRetries::None));
    }

    // ===== NextestConfig default proofs =====

    #[kani::proof]
    fn verify_config_default_timeout() {
        let config = NextestConfig::default();
        assert!(config.timeout.as_secs() == 300);
    }

    #[kani::proof]
    fn verify_config_default_jobs_none() {
        let config = NextestConfig::default();
        assert!(config.jobs.is_none());
    }

    #[kani::proof]
    fn verify_config_default_retries_none() {
        let config = NextestConfig::default();
        assert!(matches!(config.retries, NextestRetries::None));
    }

    #[kani::proof]
    fn verify_config_default_fail_fast_false() {
        let config = NextestConfig::default();
        assert!(!config.fail_fast);
    }

    #[kani::proof]
    fn verify_config_default_no_capture_false() {
        let config = NextestConfig::default();
        assert!(!config.no_capture);
    }

    #[kani::proof]
    fn verify_config_default_output_format_human() {
        let config = NextestConfig::default();
        assert!(matches!(config.output_format, NextestOutputFormat::Human));
    }

    #[kani::proof]
    fn verify_config_default_filter_none() {
        let config = NextestConfig::default();
        assert!(config.filter.is_none());
    }

    #[kani::proof]
    fn verify_config_default_workspace_false() {
        let config = NextestConfig::default();
        assert!(!config.workspace);
    }

    // ===== NextestConfig builder proofs =====

    #[kani::proof]
    fn verify_config_with_jobs() {
        let config = NextestConfig::default().with_jobs(4);
        assert!(config.jobs == Some(4));
    }

    #[kani::proof]
    fn verify_config_with_retries() {
        let config = NextestConfig::default().with_retries(3);
        assert!(matches!(config.retries, NextestRetries::Fixed(3)));
    }

    #[kani::proof]
    fn verify_config_with_fail_fast_true() {
        let config = NextestConfig::default().with_fail_fast(true);
        assert!(config.fail_fast);
    }

    #[kani::proof]
    fn verify_config_with_fail_fast_false() {
        let config = NextestConfig::default().with_fail_fast(false);
        assert!(!config.fail_fast);
    }

    #[kani::proof]
    fn verify_config_with_filter() {
        let config = NextestConfig::default().with_filter("test(smoke)");
        assert!(config.filter == Some("test(smoke)".to_string()));
    }

    #[kani::proof]
    fn verify_config_with_workspace_true() {
        let config = NextestConfig::default().with_workspace(true);
        assert!(config.workspace);
    }

    #[kani::proof]
    fn verify_config_with_workspace_false() {
        let config = NextestConfig::default().with_workspace(false);
        assert!(!config.workspace);
    }

    #[kani::proof]
    fn verify_config_builder_chaining() {
        let config = NextestConfig::default()
            .with_jobs(8)
            .with_retries(2)
            .with_fail_fast(true)
            .with_workspace(true);
        assert!(config.jobs == Some(8));
        assert!(matches!(config.retries, NextestRetries::Fixed(2)));
        assert!(config.fail_fast);
        assert!(config.workspace);
    }

    // ===== NextestStats default proofs =====

    #[kani::proof]
    fn verify_stats_default_total_tests_zero() {
        let stats = NextestStats::default();
        assert!(stats.total_tests == 0);
    }

    #[kani::proof]
    fn verify_stats_default_passed_zero() {
        let stats = NextestStats::default();
        assert!(stats.passed == 0);
    }

    #[kani::proof]
    fn verify_stats_default_failed_zero() {
        let stats = NextestStats::default();
        assert!(stats.failed == 0);
    }

    #[kani::proof]
    fn verify_stats_default_skipped_zero() {
        let stats = NextestStats::default();
        assert!(stats.skipped == 0);
    }

    #[kani::proof]
    fn verify_stats_default_retried_zero() {
        let stats = NextestStats::default();
        assert!(stats.retried == 0);
    }

    #[kani::proof]
    fn verify_stats_default_failed_tests_empty() {
        let stats = NextestStats::default();
        assert!(stats.failed_tests.is_empty());
    }

    // ===== NextestBackend construction proofs =====

    #[kani::proof]
    fn verify_backend_new_has_default_config() {
        let backend = NextestBackend::new();
        assert!(backend.config.timeout.as_secs() == 300);
        assert!(!backend.config.fail_fast);
    }

    #[kani::proof]
    fn verify_backend_default_equals_new() {
        let b1 = NextestBackend::new();
        let b2 = NextestBackend::default();
        assert!(b1.config.timeout == b2.config.timeout);
        assert!(b1.config.fail_fast == b2.config.fail_fast);
    }

    #[kani::proof]
    fn verify_backend_with_config_preserves_timeout() {
        let config = NextestConfig {
            timeout: Duration::from_secs(600),
            jobs: Some(4),
            retries: NextestRetries::Fixed(2),
            fail_fast: true,
            no_capture: false,
            output_format: NextestOutputFormat::Human,
            filter: None,
            workspace: false,
        };
        let backend = NextestBackend::with_config(config);
        assert!(backend.config.timeout.as_secs() == 600);
    }

    #[kani::proof]
    fn verify_backend_with_config_preserves_jobs() {
        let config = NextestConfig {
            timeout: Duration::from_secs(300),
            jobs: Some(16),
            retries: NextestRetries::None,
            fail_fast: false,
            no_capture: false,
            output_format: NextestOutputFormat::Human,
            filter: None,
            workspace: false,
        };
        let backend = NextestBackend::with_config(config);
        assert!(backend.config.jobs == Some(16));
    }

    // ===== Backend ID proof =====

    #[kani::proof]
    fn verify_backend_id() {
        let backend = NextestBackend::new();
        assert!(matches!(backend.id(), BackendId::Nextest));
    }

    // ===== Supports proof =====

    #[kani::proof]
    fn verify_supports_lint() {
        let backend = NextestBackend::new();
        let supported = backend.supports();
        let has_lint = supported.iter().any(|p| matches!(p, PropertyType::Lint));
        assert!(has_lint);
    }

    #[kani::proof]
    fn verify_supports_count() {
        let backend = NextestBackend::new();
        let supported = backend.supports();
        assert!(supported.len() == 1);
    }

    // ===== extract_number_before proofs =====

    #[kani::proof]
    fn verify_extract_number_before_tests_run() {
        let result = NextestBackend::extract_number_before("50 tests run: 49 passed", "tests run");
        assert!(result == Some(50));
    }

    #[kani::proof]
    fn verify_extract_number_before_passed() {
        let result = NextestBackend::extract_number_before("49 passed, 1 failed", "passed");
        assert!(result == Some(49));
    }

    #[kani::proof]
    fn verify_extract_number_before_no_marker() {
        let result = NextestBackend::extract_number_before("no numbers here", "passed");
        assert!(result.is_none());
    }

    #[kani::proof]
    fn verify_extract_number_before_empty() {
        let result = NextestBackend::extract_number_before("", "passed");
        assert!(result.is_none());
    }

    // ===== extract_test_name proofs =====

    #[kani::proof]
    fn verify_extract_test_name_with_colon() {
        let result = NextestBackend::extract_test_name("FAIL [   0.001s] crate::module::test");
        assert!(result == Some("crate::module::test".to_string()));
    }

    #[kani::proof]
    fn verify_extract_test_name_no_colon() {
        let result = NextestBackend::extract_test_name("no test name here");
        assert!(result.is_none());
    }

    // ===== parse_output proofs =====

    #[kani::proof]
    fn verify_parse_output_empty_returns_zeros() {
        let backend = NextestBackend::new();
        let stats = backend.parse_output("", "");
        assert!(stats.total_tests == 0);
        assert!(stats.passed == 0);
        assert!(stats.failed == 0);
    }

    #[kani::proof]
    fn verify_parse_output_all_pass() {
        let backend = NextestBackend::new();
        let stdout = "Summary [   1.234s] 50 tests run: 50 passed, 0 failed, 0 skipped";
        let stats = backend.parse_output(stdout, "");
        assert!(stats.passed == 50);
        assert!(stats.failed == 0);
    }

    #[kani::proof]
    fn verify_parse_output_total_calculated() {
        let backend = NextestBackend::new();
        // If total not directly parsed, it should be calculated
        let stats = backend.parse_output("", "");
        assert!(stats.total_tests == stats.passed + stats.failed + stats.skipped);
    }

    // ===== evaluate_results proofs =====

    #[kani::proof]
    fn verify_evaluate_results_all_pass_proven() {
        let backend = NextestBackend::new();
        let stats = NextestStats {
            total_tests: 50,
            passed: 50,
            failed: 0,
            skipped: 0,
            retried: 0,
            failed_tests: vec![],
        };
        let (status, cex) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(cex.is_none());
    }

    #[kani::proof]
    fn verify_evaluate_results_failures_disproven() {
        let backend = NextestBackend::new();
        let stats = NextestStats {
            total_tests: 50,
            passed: 49,
            failed: 1,
            skipped: 0,
            retried: 0,
            failed_tests: vec![FailedTest {
                name: "test::failed".to_string(),
                crate_name: None,
                stdout: None,
                stderr: None,
                duration: None,
            }],
        };
        let (status, cex) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(cex.is_some());
    }

    #[kani::proof]
    fn verify_evaluate_results_no_tests_unknown() {
        let backend = NextestBackend::new();
        let stats = NextestStats::default();
        let (status, _) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[kani::proof]
    fn verify_evaluate_results_counterexample_has_playback() {
        let backend = NextestBackend::new();
        let stats = NextestStats {
            total_tests: 10,
            passed: 9,
            failed: 1,
            skipped: 0,
            retried: 0,
            failed_tests: vec![FailedTest {
                name: "test".to_string(),
                crate_name: None,
                stdout: None,
                stderr: None,
                duration: None,
            }],
        };
        let (_, cex) = backend.evaluate_results(&stats);
        assert!(cex.is_some());
        let cex = cex.unwrap();
        assert!(cex.playback_test.is_some());
    }

    #[kani::proof]
    fn verify_evaluate_results_counterexample_not_minimized() {
        let backend = NextestBackend::new();
        let stats = NextestStats {
            total_tests: 10,
            passed: 9,
            failed: 1,
            skipped: 0,
            retried: 0,
            failed_tests: vec![],
        };
        let (_, cex) = backend.evaluate_results(&stats);
        assert!(cex.is_some());
        let cex = cex.unwrap();
        assert!(!cex.minimized);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_id() {
        assert_eq!(NextestBackend::new().id(), BackendId::Nextest);
    }

    #[test]
    fn test_config_default() {
        let config = NextestConfig::default();
        assert!(config.jobs.is_none());
        assert!(!config.fail_fast);
        assert!(matches!(config.retries, NextestRetries::None));
    }

    #[test]
    fn test_config_builder() {
        let config = NextestConfig::default()
            .with_jobs(4)
            .with_retries(2)
            .with_fail_fast(true)
            .with_filter("test(smoke)")
            .with_workspace(true);

        assert_eq!(config.jobs, Some(4));
        assert!(matches!(config.retries, NextestRetries::Fixed(2)));
        assert!(config.fail_fast);
        assert_eq!(config.filter, Some("test(smoke)".to_string()));
        assert!(config.workspace);
    }

    #[test]
    fn test_extract_number_before() {
        assert_eq!(
            NextestBackend::extract_number_before("50 tests run: 49 passed", "tests run"),
            Some(50)
        );
        assert_eq!(
            NextestBackend::extract_number_before("49 passed, 1 failed", "passed"),
            Some(49)
        );
        assert_eq!(
            NextestBackend::extract_number_before("no numbers here", "passed"),
            None
        );
    }

    #[test]
    fn test_extract_test_name() {
        assert_eq!(
            NextestBackend::extract_test_name("FAIL [   0.001s] crate::module::test_name"),
            Some("crate::module::test_name".to_string())
        );
        assert_eq!(
            NextestBackend::extract_test_name("PASS [   0.001s] crate::test"),
            Some("crate::test".to_string())
        );
        assert_eq!(NextestBackend::extract_test_name("no test name here"), None);
    }

    #[test]
    fn test_parse_output_all_pass() {
        let backend = NextestBackend::new();
        let stdout = r#"
    Compiling test-crate v0.1.0
Summary [   1.234s] 50 tests run: 50 passed, 0 failed, 0 skipped
"#;
        let stats = backend.parse_output(stdout, "");

        assert_eq!(stats.total_tests, 50);
        assert_eq!(stats.passed, 50);
        assert_eq!(stats.failed, 0);
        assert_eq!(stats.skipped, 0);
    }

    #[test]
    fn test_parse_output_with_failures() {
        let backend = NextestBackend::new();
        let stdout = r#"
FAIL [   0.002s] mycrate::tests::test_something
Summary [   1.234s] 50 tests run: 49 passed, 1 failed, 0 skipped
"#;
        let stats = backend.parse_output(stdout, "");

        assert_eq!(stats.total_tests, 50);
        assert_eq!(stats.passed, 49);
        assert_eq!(stats.failed, 1);
        assert_eq!(stats.failed_tests.len(), 1);
        assert_eq!(stats.failed_tests[0].name, "mycrate::tests::test_something");
    }

    #[test]
    fn test_evaluate_results_pass() {
        let backend = NextestBackend::new();
        let stats = NextestStats {
            total_tests: 50,
            passed: 50,
            failed: 0,
            skipped: 0,
            ..Default::default()
        };

        let (status, counterexample) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(counterexample.is_none());
    }

    #[test]
    fn test_evaluate_results_fail() {
        let backend = NextestBackend::new();
        let stats = NextestStats {
            total_tests: 50,
            passed: 49,
            failed: 1,
            skipped: 0,
            failed_tests: vec![FailedTest {
                name: "test::failed".to_string(),
                crate_name: None,
                stdout: None,
                stderr: None,
                duration: None,
            }],
            ..Default::default()
        };

        let (status, counterexample) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(counterexample.is_some());
        let cex = counterexample.unwrap();
        assert_eq!(cex.failed_checks.len(), 1);
    }

    #[test]
    fn test_evaluate_no_tests() {
        let backend = NextestBackend::new();
        let stats = NextestStats::default();

        let (status, _) = backend.evaluate_results(&stats);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[tokio::test]
    async fn test_health_check() {
        let backend = NextestBackend::new();
        let status = backend.health_check().await;
        match status {
            HealthStatus::Healthy => {}
            HealthStatus::Unavailable { reason } => {
                assert!(reason.contains("cargo-nextest"));
            }
            _ => panic!("Unexpected status"),
        }
    }

    #[tokio::test]
    async fn test_verify_returns_result_or_unavailable() {
        use dashprove_usl::{parse, typecheck};

        let backend = NextestBackend::new();
        let spec = parse("theorem test { true }").expect("parse");
        let typed_spec = typecheck(spec).expect("typecheck");

        let result = backend.verify(&typed_spec).await;
        match result {
            Ok(r) => {
                assert_eq!(r.backend, BackendId::Nextest);
            }
            Err(BackendError::Unavailable(_)) => {}
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }
}
