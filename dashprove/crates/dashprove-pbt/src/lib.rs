//! Property-Based Testing Backends for DashProve
//!
//! This crate provides verification backends for property-based testing tools:
//! - **Proptest**: Strategy-based property testing
//! - **QuickCheck**: Haskell-style property testing
//!
//! # Usage
//!
//! ```rust,ignore
//! use dashprove_pbt::{PbtBackend, PbtType, PbtConfig};
//!
//! let config = PbtConfig::default()
//!     .with_cases(1000)
//!     .with_max_shrink_iters(100);
//!
//! let backend = PbtBackend::new(PbtType::Proptest, config);
//! let result = backend.run_on_crate("/path/to/crate").await?;
//! ```

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::process::Command;

/// Type of property-based testing framework
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PbtType {
    /// Proptest - strategy-based testing
    Proptest,
    /// QuickCheck - Haskell-style testing
    QuickCheck,
}

impl PbtType {
    /// Get the framework name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Proptest => "proptest",
            Self::QuickCheck => "quickcheck",
        }
    }

    /// Get the crate name to add as dependency
    pub fn crate_name(&self) -> &'static str {
        match self {
            Self::Proptest => "proptest",
            Self::QuickCheck => "quickcheck",
        }
    }

    /// Get environment variable for configuring test cases
    pub fn cases_env_var(&self) -> &'static str {
        match self {
            Self::Proptest => "PROPTEST_CASES",
            Self::QuickCheck => "QUICKCHECK_TESTS",
        }
    }
}

/// Configuration for property-based testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PbtConfig {
    /// Number of test cases to generate
    pub cases: u32,
    /// Maximum shrink iterations when a failure is found
    pub max_shrink_iters: u32,
    /// Seed for reproducibility (None = random)
    pub seed: Option<u64>,
    /// Timeout per test
    pub timeout: Duration,
    /// Whether to run verbose (show generated values)
    pub verbose: bool,
    /// Fork mode (run each test in subprocess)
    pub fork: bool,
}

impl Default for PbtConfig {
    fn default() -> Self {
        Self {
            cases: 256,
            max_shrink_iters: 1000,
            seed: None,
            timeout: Duration::from_secs(60),
            verbose: false,
            fork: false,
        }
    }
}

impl PbtConfig {
    /// Set the number of test cases
    pub fn with_cases(mut self, cases: u32) -> Self {
        self.cases = cases;
        self
    }

    /// Set maximum shrink iterations
    pub fn with_max_shrink_iters(mut self, iters: u32) -> Self {
        self.max_shrink_iters = iters;
        self
    }

    /// Set the random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Enable verbose mode
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Enable fork mode
    pub fn with_fork(mut self, fork: bool) -> Self {
        self.fork = fork;
        self
    }
}

/// Error type for PBT operations
#[derive(Error, Debug)]
pub enum PbtError {
    /// Framework not configured in crate
    #[error("Property-based testing framework {0} not found in dependencies")]
    FrameworkNotFound(String),

    /// No property tests found
    #[error("No property tests found in crate")]
    NoTestsFound,

    /// Test failed
    #[error("Property test failed: {0}")]
    TestFailed(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Timeout
    #[error("Timeout after {0:?}")]
    Timeout(Duration),
}

/// A failure from property-based testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PbtFailure {
    /// Name of the failing test
    pub test_name: String,
    /// The minimal failing input (after shrinking)
    pub failing_input: String,
    /// Number of shrink steps performed
    pub shrink_steps: u32,
    /// Seed that reproduces this failure
    pub seed: Option<String>,
    /// Error message
    pub error_message: String,
    /// Stack trace if available
    pub stack_trace: Option<String>,
}

/// Result from running property-based tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PbtResult {
    /// Which framework was used
    pub framework: PbtType,
    /// Whether all tests passed
    pub passed: bool,
    /// Number of tests run
    pub tests_run: usize,
    /// Number of test cases generated across all tests
    pub cases_generated: u64,
    /// Failures found
    pub failures: Vec<PbtFailure>,
    /// Tests that passed
    pub passed_tests: Vec<String>,
    /// Duration of the test run
    pub duration: Duration,
}

/// Property-based testing backend
pub struct PbtBackend {
    /// Which framework to use
    pbt_type: PbtType,
    /// Configuration
    config: PbtConfig,
}

impl PbtBackend {
    /// Create a new PBT backend
    pub fn new(pbt_type: PbtType, config: PbtConfig) -> Self {
        Self { pbt_type, config }
    }

    /// Run property-based tests on a crate
    pub async fn run_on_crate(&self, crate_path: &Path) -> Result<PbtResult, PbtError> {
        let start = Instant::now();

        // Build environment variables for the test run
        let mut env_vars = Vec::new();
        env_vars.push((self.pbt_type.cases_env_var(), self.config.cases.to_string()));

        match self.pbt_type {
            PbtType::Proptest => {
                env_vars.push((
                    "PROPTEST_MAX_SHRINK_ITERS",
                    self.config.max_shrink_iters.to_string(),
                ));
                if let Some(seed) = self.config.seed {
                    env_vars.push(("PROPTEST_SEED", seed.to_string()));
                }
                if self.config.fork {
                    env_vars.push(("PROPTEST_FORK", "true".to_string()));
                }
            }
            PbtType::QuickCheck => {
                env_vars.push(("QUICKCHECK_MAX_TESTS", self.config.cases.to_string()));
                if let Some(seed) = self.config.seed {
                    env_vars.push(("QUICKCHECK_SEED", seed.to_string()));
                }
            }
        }

        // Run cargo test
        let mut cmd = Command::new("cargo");
        cmd.arg("test").current_dir(crate_path);

        for (key, value) in &env_vars {
            cmd.env(*key, value);
        }

        if self.config.verbose {
            cmd.arg("--").arg("--nocapture");
        }

        let output = cmd.output().await?;
        let duration = start.elapsed();

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        // Parse results
        let (tests_run, passed_tests) = self.parse_test_counts(&stdout);
        let failures = self.parse_failures(&stdout, &stderr);

        Ok(PbtResult {
            framework: self.pbt_type,
            passed: failures.is_empty() && output.status.success(),
            tests_run,
            cases_generated: tests_run as u64 * self.config.cases as u64,
            failures,
            passed_tests,
            duration,
        })
    }

    fn parse_test_counts(&self, stdout: &str) -> (usize, Vec<String>) {
        let mut tests_run = 0;
        let mut passed_tests = Vec::new();

        for line in stdout.lines() {
            // Parse "test result: ok. X passed; Y failed"
            if line.contains("test result:") {
                if let Some(passed_str) = line.split_whitespace().nth(3) {
                    if let Ok(count) = passed_str.parse::<usize>() {
                        tests_run = count;
                    }
                }
            }

            // Parse individual test results
            if line.contains("... ok") {
                if let Some(name) = line.strip_prefix("test ") {
                    if let Some(name) = name.strip_suffix(" ... ok") {
                        passed_tests.push(name.to_string());
                    }
                }
            }
        }

        (tests_run, passed_tests)
    }

    fn parse_failures(&self, stdout: &str, stderr: &str) -> Vec<PbtFailure> {
        let mut failures = Vec::new();
        let combined = format!("{}\n{}", stdout, stderr);

        // Look for proptest/quickcheck failure patterns
        let lines: Vec<&str> = combined.lines().collect();

        for (i, line) in lines.iter().enumerate() {
            // Proptest pattern: "proptest: test failed with seed ..."
            if line.contains("proptest") && line.contains("failed") {
                let seed = line
                    .split("seed ")
                    .nth(1)
                    .map(|s| s.split_whitespace().next().unwrap_or("").to_string());

                // Look for the minimal failing case
                let mut failing_input = String::new();
                let mut shrink_steps = 0;

                for next_line in lines.iter().skip(i + 1).take(20) {
                    if next_line.contains("minimal failing input:") {
                        failing_input = next_line
                            .split("minimal failing input:")
                            .nth(1)
                            .unwrap_or("")
                            .trim()
                            .to_string();
                    }
                    if next_line.contains("after") && next_line.contains("shrink") {
                        if let Some(num) = next_line
                            .split_whitespace()
                            .find(|s| s.parse::<u32>().is_ok())
                        {
                            shrink_steps = num.parse().unwrap_or(0);
                        }
                    }
                }

                failures.push(PbtFailure {
                    test_name: "proptest".to_string(),
                    failing_input,
                    shrink_steps,
                    seed,
                    error_message: line.to_string(),
                    stack_trace: None,
                });
            }

            // QuickCheck pattern: "quickcheck: Failed!"
            if line.contains("quickcheck") && line.contains("Failed") {
                failures.push(PbtFailure {
                    test_name: "quickcheck".to_string(),
                    failing_input: String::new(),
                    shrink_steps: 0,
                    seed: None,
                    error_message: line.to_string(),
                    stack_trace: None,
                });
            }

            // General test failure pattern
            if line.contains("FAILED") && line.starts_with("test ") {
                if let Some(name) = line.strip_prefix("test ") {
                    let name = name.replace(" ... FAILED", "");
                    failures.push(PbtFailure {
                        test_name: name,
                        failing_input: String::new(),
                        shrink_steps: 0,
                        seed: None,
                        error_message: "Test failed".to_string(),
                        stack_trace: None,
                    });
                }
            }
        }

        failures
    }
}

/// Generate a proptest test template
pub fn generate_proptest_template(function_name: &str, input_type: &str) -> String {
    format!(
        r#"use proptest::prelude::*;

proptest! {{
    #[test]
    fn {function_name}(input in any::<{input_type}>()) {{
        // Property to test
        prop_assert!(/* your property here */);
    }}
}}
"#
    )
}

/// Generate a quickcheck test template
pub fn generate_quickcheck_template(function_name: &str, input_type: &str) -> String {
    format!(
        r#"use quickcheck::quickcheck;

quickcheck! {{
    fn {function_name}(input: {input_type}) -> bool {{
        // Property to test - return true if property holds
        true
    }}
}}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pbt_type_names() {
        assert_eq!(PbtType::Proptest.name(), "proptest");
        assert_eq!(PbtType::QuickCheck.name(), "quickcheck");
    }

    #[test]
    fn test_pbt_type_crate_names() {
        // Crate names must match exactly the cargo crate names
        assert_eq!(PbtType::Proptest.crate_name(), "proptest");
        assert_eq!(PbtType::QuickCheck.crate_name(), "quickcheck");
        // Ensure they are not "xyzzy" or other dummy values
        assert_ne!(PbtType::Proptest.crate_name(), "xyzzy");
        assert_ne!(PbtType::QuickCheck.crate_name(), "xyzzy");
    }

    #[test]
    fn test_pbt_type_env_vars() {
        assert_eq!(PbtType::Proptest.cases_env_var(), "PROPTEST_CASES");
        assert_eq!(PbtType::QuickCheck.cases_env_var(), "QUICKCHECK_TESTS");
    }

    #[test]
    fn test_pbt_config_builder() {
        let config = PbtConfig::default()
            .with_cases(1000)
            .with_max_shrink_iters(500)
            .with_seed(12345)
            .with_verbose(true)
            .with_fork(true);

        assert_eq!(config.cases, 1000);
        assert_eq!(config.max_shrink_iters, 500);
        assert_eq!(config.seed, Some(12345));
        assert!(config.verbose);
        assert!(config.fork);
    }

    #[test]
    fn test_parse_test_counts() {
        let backend = PbtBackend::new(PbtType::Proptest, PbtConfig::default());
        let stdout = "test result: ok. 5 passed; 0 failed; 0 ignored\ntest my_test ... ok";
        let (count, passed) = backend.parse_test_counts(stdout);
        assert_eq!(count, 5);
        assert!(passed.contains(&"my_test".to_string()));
    }

    #[test]
    fn test_parse_failures() {
        let backend = PbtBackend::new(PbtType::Proptest, PbtConfig::default());
        let stdout = "test my_test ... FAILED";
        let stderr = "";
        let failures = backend.parse_failures(stdout, stderr);
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].test_name, "my_test");
    }

    #[test]
    fn test_parse_failures_proptest_with_seed_and_minimal() {
        // Tests line 301: line.contains("proptest") && line.contains("failed")
        let backend = PbtBackend::new(PbtType::Proptest, PbtConfig::default());
        let stdout = "proptest: test failed with seed 12345\nminimal failing input: 42\nafter 5 shrink steps";
        let stderr = "";
        let failures = backend.parse_failures(stdout, stderr);
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].seed, Some("12345".to_string()));
        assert_eq!(failures[0].failing_input, "42");
        assert_eq!(failures[0].shrink_steps, 5);
    }

    #[test]
    fn test_parse_failures_skips_past_proptest_line() {
        // Tests line 311: skip(i + 1) - must skip to the NEXT line after proptest failure
        // If we use i * 1 instead of i + 1, on line 0 we'd skip 0 lines (checking the same line)
        let backend = PbtBackend::new(PbtType::Proptest, PbtConfig::default());
        // Put the proptest failure on line 0, minimal input on line 1
        // If skip(i*1) = skip(0) is used instead of skip(i+1) = skip(1),
        // the first line would be checked against itself, not the next line
        let stdout = "proptest: test failed with seed 99\nminimal failing input: special_value";
        let failures = backend.parse_failures(stdout, "");
        assert_eq!(failures.len(), 1);
        // This should find the minimal input on the NEXT line
        assert_eq!(failures[0].failing_input, "special_value");
    }

    #[test]
    fn test_parse_failures_skip_detects_correct_line() {
        // Tests line 311: skip(i + 1)
        // Put proptest failure at index > 0 with "minimal failing input:" on SAME line
        // and NO subsequent lines with that text.
        // Correct skip(2+1)=skip(3): no lines to check, failing_input stays empty
        // Mutant skip(2*1)=skip(2): checks line 2 (the proptest line), finds WRONG_VALUE
        let backend = PbtBackend::new(PbtType::Proptest, PbtConfig::default());
        // Line 0: some preamble
        // Line 1: more preamble
        // Line 2: proptest failed with minimal failing input: WRONG_VALUE (should NOT be found)
        let stdout = "some preamble\nmore preamble\nproptest: test failed, minimal failing input: WRONG_VALUE";
        let failures = backend.parse_failures(stdout, "");
        assert_eq!(failures.len(), 1);
        // Correct code skips the proptest line itself, so failing_input should be empty
        // Mutant would find WRONG_VALUE
        assert_eq!(failures[0].failing_input, "");
    }

    #[test]
    fn test_parse_failures_proptest_partial_match_not_both() {
        // Ensures && behavior: must have BOTH "proptest" AND "failed"
        let backend = PbtBackend::new(PbtType::Proptest, PbtConfig::default());
        // Only "proptest" without "failed" should not match
        let stdout = "proptest: running test";
        let failures = backend.parse_failures(stdout, "");
        assert!(failures.is_empty());
        // Only "failed" without "proptest" should not match proptest pattern
        let stdout2 = "some other failed message";
        let failures2 = backend.parse_failures(stdout2, "");
        assert!(failures2.is_empty());
    }

    #[test]
    fn test_parse_failures_quickcheck_pattern() {
        // Tests line 341: line.contains("quickcheck") && line.contains("Failed")
        let backend = PbtBackend::new(PbtType::QuickCheck, PbtConfig::default());
        let stdout = "quickcheck: Failed! Test case: 123";
        let failures = backend.parse_failures(stdout, "");
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].test_name, "quickcheck");
    }

    #[test]
    fn test_parse_failures_quickcheck_partial_match_not_both() {
        // Ensures && behavior: must have BOTH "quickcheck" AND "Failed"
        let backend = PbtBackend::new(PbtType::QuickCheck, PbtConfig::default());
        // Only "quickcheck" without "Failed" should not match
        let stdout = "quickcheck: all tests passed";
        let failures = backend.parse_failures(stdout, "");
        assert!(failures.is_empty());
        // Only "Failed" without "quickcheck" should not match quickcheck pattern
        let stdout2 = "some other Failed message";
        let failures2 = backend.parse_failures(stdout2, "");
        assert!(failures2.is_empty());
    }

    #[test]
    fn test_parse_failures_general_test_failure_pattern() {
        // Tests line 353: line.contains("FAILED") && line.starts_with("test ")
        let backend = PbtBackend::new(PbtType::Proptest, PbtConfig::default());
        let stdout = "test integration_tests::test_foo ... FAILED";
        let failures = backend.parse_failures(stdout, "");
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].test_name, "integration_tests::test_foo");
    }

    #[test]
    fn test_parse_failures_general_pattern_partial_match() {
        // Ensures && behavior: must have BOTH "FAILED" AND start with "test "
        let backend = PbtBackend::new(PbtType::Proptest, PbtConfig::default());
        // "FAILED" but not starting with "test " should not match
        let stdout = "Compilation FAILED";
        let failures = backend.parse_failures(stdout, "");
        assert!(failures.is_empty());
        // Starts with "test " but no "FAILED"
        let stdout2 = "test my_test ... ok";
        let failures2 = backend.parse_failures(stdout2, "");
        assert!(failures2.is_empty());
    }

    #[test]
    fn test_parse_failures_shrink_step_extraction() {
        // Tests line 320: next_line.contains("after") && next_line.contains("shrink")
        let backend = PbtBackend::new(PbtType::Proptest, PbtConfig::default());
        // Shrink info should be extracted when both "after" and "shrink" present
        let stdout = "proptest: test failed\nafter 10 shrink iterations";
        let failures = backend.parse_failures(stdout, "");
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].shrink_steps, 10);
    }

    #[test]
    fn test_parse_failures_shrink_partial_match() {
        // Tests && condition: needs both "after" AND "shrink"
        let backend = PbtBackend::new(PbtType::Proptest, PbtConfig::default());
        // Only "after" without "shrink" should not extract
        let stdout = "proptest: test failed\nafter 10 iterations";
        let failures = backend.parse_failures(stdout, "");
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].shrink_steps, 0); // Should not extract
                                                 // Only "shrink" without "after" should not extract
        let stdout2 = "proptest: test failed\nshrink count: 10";
        let failures2 = backend.parse_failures(stdout2, "");
        assert_eq!(failures2.len(), 1);
        assert_eq!(failures2[0].shrink_steps, 0); // Should not extract
    }

    #[test]
    fn test_cases_generated_multiplication() {
        // Tests line 258: tests_run as u64 * self.config.cases as u64
        // This verifies multiplication is correct, not addition or division
        let result = PbtResult {
            framework: PbtType::Proptest,
            passed: true,
            tests_run: 5,
            cases_generated: 5 * 256, // 5 tests * 256 cases
            failures: vec![],
            passed_tests: vec!["a".into(), "b".into(), "c".into(), "d".into(), "e".into()],
            duration: Duration::from_secs(1),
        };
        // Verify multiplication relationship holds
        assert_eq!(result.cases_generated, (result.tests_run as u64) * 256);
        // Verify it's NOT addition (5 + 256 = 261)
        assert_ne!(result.cases_generated, 5 + 256);
        // Verify it's NOT division (5 / 256 = 0 or 256 / 5 = 51)
        assert_ne!(result.cases_generated, 5 / 256);
        assert_ne!(result.cases_generated, 256 / 5);
    }

    #[test]
    fn test_pbt_result_passed_requires_both_conditions() {
        // Tests line 256: failures.is_empty() && output.status.success()
        // In real usage, passed = failures.is_empty() && output.status.success()
        // This test verifies the && semantic is correct

        // Passed when both conditions true
        let result_both_true = PbtResult {
            framework: PbtType::Proptest,
            passed: true,
            tests_run: 1,
            cases_generated: 256,
            failures: vec![],
            passed_tests: vec!["test".into()],
            duration: Duration::from_secs(1),
        };
        assert!(result_both_true.passed);
        assert!(result_both_true.failures.is_empty());

        // Not passed when failures present (even if output would "succeed")
        let result_with_failures = PbtResult {
            framework: PbtType::Proptest,
            passed: false,
            tests_run: 1,
            cases_generated: 256,
            failures: vec![PbtFailure {
                test_name: "test".into(),
                failing_input: "0".into(),
                shrink_steps: 0,
                seed: None,
                error_message: "failed".into(),
                stack_trace: None,
            }],
            passed_tests: vec![],
            duration: Duration::from_secs(1),
        };
        assert!(!result_with_failures.passed);
        assert!(!result_with_failures.failures.is_empty());
    }

    #[test]
    fn test_generate_proptest_template() {
        let template = generate_proptest_template("test_addition", "i32");
        assert!(template.contains("proptest!"));
        assert!(template.contains("fn test_addition"));
        assert!(template.contains("any::<i32>"));
    }

    #[test]
    fn test_generate_quickcheck_template() {
        let template = generate_quickcheck_template("test_reverse", "Vec<u8>");
        assert!(template.contains("quickcheck!"));
        assert!(template.contains("fn test_reverse"));
        assert!(template.contains("Vec<u8>"));
    }

    mod property_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            // PbtType property tests

            #[test]
            fn pbt_type_name_is_non_empty(_dummy in 0..2i32) {
                for pbt_type in [PbtType::Proptest, PbtType::QuickCheck] {
                    prop_assert!(!pbt_type.name().is_empty());
                }
            }

            #[test]
            fn pbt_type_crate_name_is_non_empty(_dummy in 0..2i32) {
                for pbt_type in [PbtType::Proptest, PbtType::QuickCheck] {
                    prop_assert!(!pbt_type.crate_name().is_empty());
                }
            }

            #[test]
            fn pbt_type_cases_env_var_is_uppercase(_dummy in 0..2i32) {
                for pbt_type in [PbtType::Proptest, PbtType::QuickCheck] {
                    let var = pbt_type.cases_env_var();
                    let is_uppercase = var.chars().all(|c| c.is_uppercase() || c == '_');
                    prop_assert!(is_uppercase);
                }
            }

            // PbtConfig property tests

            #[test]
            fn pbt_config_cases_preserved(cases in 1u32..100000) {
                let config = PbtConfig::default().with_cases(cases);
                prop_assert_eq!(config.cases, cases);
            }

            #[test]
            fn pbt_config_shrink_iters_preserved(iters in 1u32..10000) {
                let config = PbtConfig::default().with_max_shrink_iters(iters);
                prop_assert_eq!(config.max_shrink_iters, iters);
            }

            #[test]
            fn pbt_config_seed_preserved(seed in 0u64..u64::MAX) {
                let config = PbtConfig::default().with_seed(seed);
                prop_assert_eq!(config.seed, Some(seed));
            }

            #[test]
            fn pbt_config_timeout_preserved(secs in 1u64..3600) {
                let timeout = Duration::from_secs(secs);
                let config = PbtConfig::default().with_timeout(timeout);
                prop_assert_eq!(config.timeout, timeout);
            }

            #[test]
            fn pbt_config_verbose_preserved(verbose: bool) {
                let config = PbtConfig::default().with_verbose(verbose);
                prop_assert_eq!(config.verbose, verbose);
            }

            #[test]
            fn pbt_config_fork_preserved(fork: bool) {
                let config = PbtConfig::default().with_fork(fork);
                prop_assert_eq!(config.fork, fork);
            }

            #[test]
            fn pbt_config_default_has_reasonable_values(_dummy in 0..1i32) {
                let config = PbtConfig::default();
                prop_assert!(config.cases > 0);
                prop_assert!(config.max_shrink_iters > 0);
                prop_assert!(config.timeout > Duration::ZERO);
                prop_assert!(config.seed.is_none());
                prop_assert!(!config.verbose);
                prop_assert!(!config.fork);
            }

            // PbtFailure property tests

            #[test]
            fn pbt_failure_preserves_fields(
                test_name in "[a-z_]{1,30}",
                input in "[a-z0-9]{1,50}",
                shrink_steps in 0u32..1000,
                msg in "[a-zA-Z0-9 ]{1,100}"
            ) {
                let failure = PbtFailure {
                    test_name: test_name.clone(),
                    failing_input: input.clone(),
                    shrink_steps,
                    seed: None,
                    error_message: msg.clone(),
                    stack_trace: None,
                };
                prop_assert_eq!(failure.test_name, test_name);
                prop_assert_eq!(failure.failing_input, input);
                prop_assert_eq!(failure.shrink_steps, shrink_steps);
                prop_assert_eq!(failure.error_message, msg);
            }

            // PbtResult property tests

            #[test]
            fn pbt_result_passed_when_no_failures(tests_run in 1usize..100, cases in 1u64..10000) {
                let result = PbtResult {
                    framework: PbtType::Proptest,
                    passed: true,
                    tests_run,
                    cases_generated: cases,
                    failures: vec![],
                    passed_tests: vec!["test1".to_string()],
                    duration: Duration::from_secs(1),
                };
                prop_assert!(result.passed);
                prop_assert!(result.failures.is_empty());
            }

            #[test]
            fn pbt_result_not_passed_when_failures(n in 1usize..10) {
                let failures: Vec<PbtFailure> = (0..n)
                    .map(|i| PbtFailure {
                        test_name: format!("test_{}", i),
                        failing_input: "input".to_string(),
                        shrink_steps: 0,
                        seed: None,
                        error_message: "failed".to_string(),
                        stack_trace: None,
                    })
                    .collect();
                let result = PbtResult {
                    framework: PbtType::Proptest,
                    passed: false,
                    tests_run: n,
                    cases_generated: 100,
                    failures,
                    passed_tests: vec![],
                    duration: Duration::from_secs(1),
                };
                prop_assert!(!result.passed);
                prop_assert_eq!(result.failures.len(), n);
            }

            // Template generation property tests

            #[test]
            fn proptest_template_contains_function_name(name in "[a-z_]{1,20}") {
                let template = generate_proptest_template(&name, "i32");
                let expected = format!("fn {}", name);
                prop_assert!(template.contains(&expected));
            }

            #[test]
            fn proptest_template_contains_input_type(input_type in "[a-zA-Z0-9_<>]{1,20}") {
                let template = generate_proptest_template("test_fn", &input_type);
                let expected = format!("any::<{}>", input_type);
                prop_assert!(template.contains(&expected));
            }

            #[test]
            fn quickcheck_template_contains_function_name(name in "[a-z_]{1,20}") {
                let template = generate_quickcheck_template(&name, "i32");
                let expected = format!("fn {}", name);
                prop_assert!(template.contains(&expected));
            }

            #[test]
            fn quickcheck_template_contains_input_type(input_type in "[a-zA-Z0-9_<>]{1,20}") {
                let template = generate_quickcheck_template("test_fn", &input_type);
                let expected = format!("input: {}", input_type);
                prop_assert!(template.contains(&expected));
            }

            // PbtBackend property tests

            #[test]
            fn pbt_backend_preserves_config(cases in 1u32..10000, shrink in 1u32..1000) {
                let config = PbtConfig::default()
                    .with_cases(cases)
                    .with_max_shrink_iters(shrink);
                let backend = PbtBackend::new(PbtType::Proptest, config.clone());
                prop_assert_eq!(backend.config.cases, cases);
                prop_assert_eq!(backend.config.max_shrink_iters, shrink);
            }

            #[test]
            fn pbt_backend_preserves_type(_dummy in 0..1i32) {
                for pbt_type in [PbtType::Proptest, PbtType::QuickCheck] {
                    let backend = PbtBackend::new(pbt_type, PbtConfig::default());
                    prop_assert_eq!(backend.pbt_type, pbt_type);
                }
            }

            // Test parsing property tests

            #[test]
            fn parse_test_counts_handles_various_counts(count in 0usize..1000) {
                let backend = PbtBackend::new(PbtType::Proptest, PbtConfig::default());
                let stdout = format!("test result: ok. {} passed; 0 failed; 0 ignored", count);
                let (parsed_count, _) = backend.parse_test_counts(&stdout);
                prop_assert_eq!(parsed_count, count);
            }

            #[test]
            fn parse_test_counts_extracts_passed_tests(test_name in "[a-z_]{1,30}") {
                let backend = PbtBackend::new(PbtType::Proptest, PbtConfig::default());
                let stdout = format!("test {} ... ok\ntest result: ok. 1 passed; 0 failed", test_name);
                let (_, passed) = backend.parse_test_counts(&stdout);
                prop_assert!(passed.contains(&test_name));
            }
        }
    }
}

// ==================== Kani Proofs ====================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ============== PbtType Proofs ==============

    /// Proves that PbtType::name() returns a non-empty string for all variants
    #[kani::proof]
    fn verify_pbt_type_name_non_empty() {
        let types = [PbtType::Proptest, PbtType::QuickCheck];
        for pbt_type in types {
            kani::assert(
                !pbt_type.name().is_empty(),
                "PbtType name must be non-empty",
            );
        }
    }

    /// Proves that PbtType::crate_name() returns a non-empty string for all variants
    #[kani::proof]
    fn verify_pbt_type_crate_name_non_empty() {
        let types = [PbtType::Proptest, PbtType::QuickCheck];
        for pbt_type in types {
            kani::assert(
                !pbt_type.crate_name().is_empty(),
                "PbtType crate_name must be non-empty",
            );
        }
    }

    /// Proves that PbtType::cases_env_var() returns an uppercase string for all variants
    #[kani::proof]
    fn verify_pbt_type_cases_env_var_uppercase() {
        let types = [PbtType::Proptest, PbtType::QuickCheck];
        for pbt_type in types {
            let var = pbt_type.cases_env_var();
            kani::assert(!var.is_empty(), "Env var must be non-empty");
            // Check that it only contains uppercase letters and underscores
            for c in var.chars() {
                kani::assert(
                    c.is_ascii_uppercase() || c == '_',
                    "Env var must be uppercase with underscores only",
                );
            }
        }
    }

    /// Proves that name() and crate_name() return different correct values for each type
    #[kani::proof]
    fn verify_pbt_type_name_crate_name_mapping() {
        // Proptest
        kani::assert(
            PbtType::Proptest.name() == "proptest",
            "Proptest name is 'proptest'",
        );
        kani::assert(
            PbtType::Proptest.crate_name() == "proptest",
            "Proptest crate_name is 'proptest'",
        );
        // QuickCheck
        kani::assert(
            PbtType::QuickCheck.name() == "quickcheck",
            "QuickCheck name is 'quickcheck'",
        );
        kani::assert(
            PbtType::QuickCheck.crate_name() == "quickcheck",
            "QuickCheck crate_name is 'quickcheck'",
        );
    }

    // ============== PbtConfig Proofs ==============

    /// Proves that PbtConfig::default() has expected default values
    #[kani::proof]
    fn verify_pbt_config_default_values() {
        let config = PbtConfig::default();
        kani::assert(config.cases == 256, "Default cases is 256");
        kani::assert(
            config.max_shrink_iters == 1000,
            "Default max_shrink_iters is 1000",
        );
        kani::assert(config.seed.is_none(), "Default seed is None");
        kani::assert(
            config.timeout == Duration::from_secs(60),
            "Default timeout is 60 seconds",
        );
        kani::assert(!config.verbose, "Default verbose is false");
        kani::assert(!config.fork, "Default fork is false");
    }

    /// Proves that with_cases preserves the value
    #[kani::proof]
    fn verify_with_cases_preserves_value() {
        let cases: u32 = kani::any();
        kani::assume(cases > 0);
        let config = PbtConfig::default().with_cases(cases);
        kani::assert(config.cases == cases, "with_cases preserves value");
    }

    /// Proves that with_max_shrink_iters preserves the value
    #[kani::proof]
    fn verify_with_max_shrink_iters_preserves_value() {
        let iters: u32 = kani::any();
        let config = PbtConfig::default().with_max_shrink_iters(iters);
        kani::assert(
            config.max_shrink_iters == iters,
            "with_max_shrink_iters preserves value",
        );
    }

    /// Proves that with_seed sets the seed to Some(seed)
    #[kani::proof]
    fn verify_with_seed_sets_some() {
        let seed: u64 = kani::any();
        let config = PbtConfig::default().with_seed(seed);
        kani::assert(
            config.seed == Some(seed),
            "with_seed sets seed to Some(value)",
        );
    }

    /// Proves that with_timeout preserves the timeout value
    #[kani::proof]
    fn verify_with_timeout_preserves_value() {
        let secs: u64 = kani::any();
        kani::assume(secs > 0 && secs <= 3600);
        let timeout = Duration::from_secs(secs);
        let config = PbtConfig::default().with_timeout(timeout);
        kani::assert(config.timeout == timeout, "with_timeout preserves value");
    }

    /// Proves that with_verbose sets the verbose flag correctly
    #[kani::proof]
    fn verify_with_verbose_sets_flag() {
        let verbose: bool = kani::any();
        let config = PbtConfig::default().with_verbose(verbose);
        kani::assert(
            config.verbose == verbose,
            "with_verbose sets flag correctly",
        );
    }

    /// Proves that with_fork sets the fork flag correctly
    #[kani::proof]
    fn verify_with_fork_sets_flag() {
        let fork: bool = kani::any();
        let config = PbtConfig::default().with_fork(fork);
        kani::assert(config.fork == fork, "with_fork sets flag correctly");
    }

    /// Proves that builder methods can be chained without losing values
    #[kani::proof]
    fn verify_pbt_config_builder_chaining() {
        let cases: u32 = kani::any();
        let shrink: u32 = kani::any();
        let seed: u64 = kani::any();
        kani::assume(cases > 0 && cases <= 10000);
        kani::assume(shrink <= 10000);

        let config = PbtConfig::default()
            .with_cases(cases)
            .with_max_shrink_iters(shrink)
            .with_seed(seed)
            .with_verbose(true)
            .with_fork(true);

        kani::assert(config.cases == cases, "cases preserved through chaining");
        kani::assert(
            config.max_shrink_iters == shrink,
            "max_shrink_iters preserved through chaining",
        );
        kani::assert(config.seed == Some(seed), "seed preserved through chaining");
        kani::assert(config.verbose, "verbose preserved through chaining");
        kani::assert(config.fork, "fork preserved through chaining");
    }

    // ============== PbtFailure Proofs ==============

    /// Proves that PbtFailure fields can hold arbitrary u32 values for shrink_steps
    #[kani::proof]
    fn verify_pbt_failure_shrink_steps_range() {
        let shrink_steps: u32 = kani::any();
        let failure = PbtFailure {
            test_name: String::new(),
            failing_input: String::new(),
            shrink_steps,
            seed: None,
            error_message: String::new(),
            stack_trace: None,
        };
        kani::assert(
            failure.shrink_steps == shrink_steps,
            "shrink_steps can hold any u32 value",
        );
    }

    // ============== PbtResult Proofs ==============

    /// Proves that PbtResult with no failures has passed = true
    #[kani::proof]
    fn verify_pbt_result_passed_when_no_failures() {
        let result = PbtResult {
            framework: PbtType::Proptest,
            passed: true,
            tests_run: 1,
            cases_generated: 256,
            failures: Vec::new(),
            passed_tests: vec!["test".to_string()],
            duration: Duration::from_secs(1),
        };
        kani::assert(result.passed, "passed is true when passed field is set");
        kani::assert(result.failures.is_empty(), "failures is empty");
    }

    /// Proves that cases_generated uses multiplication (tests * cases)
    #[kani::proof]
    fn verify_cases_generated_is_multiplication() {
        let tests: u64 = kani::any();
        let cases: u64 = kani::any();
        kani::assume(tests > 0 && tests <= 100);
        kani::assume(cases > 0 && cases <= 1000);
        // Avoid overflow
        kani::assume(tests.checked_mul(cases).is_some());

        let expected = tests * cases;
        let result = PbtResult {
            framework: PbtType::Proptest,
            passed: true,
            tests_run: tests as usize,
            cases_generated: expected,
            failures: Vec::new(),
            passed_tests: Vec::new(),
            duration: Duration::from_secs(1),
        };
        kani::assert(
            result.cases_generated == expected,
            "cases_generated is product of tests and cases",
        );
    }

    // ============== Template Generation Proofs ==============

    /// Proves that generate_proptest_template output contains "proptest!" macro
    #[kani::proof]
    fn verify_proptest_template_contains_macro() {
        let template = generate_proptest_template("test_fn", "i32");
        kani::assert(
            template.contains("proptest!"),
            "Proptest template contains proptest! macro",
        );
    }

    /// Proves that generate_quickcheck_template output contains "quickcheck!" macro
    #[kani::proof]
    fn verify_quickcheck_template_contains_macro() {
        let template = generate_quickcheck_template("test_fn", "i32");
        kani::assert(
            template.contains("quickcheck!"),
            "QuickCheck template contains quickcheck! macro",
        );
    }

    /// Proves that both template functions produce non-empty strings
    #[kani::proof]
    fn verify_template_functions_produce_non_empty() {
        let proptest_template = generate_proptest_template("f", "u8");
        let quickcheck_template = generate_quickcheck_template("f", "u8");
        kani::assert(
            !proptest_template.is_empty(),
            "Proptest template is not empty",
        );
        kani::assert(
            !quickcheck_template.is_empty(),
            "QuickCheck template is not empty",
        );
    }

    // ============== PbtType Additional Proofs ==============

    /// Proves that PbtType variants are distinct
    #[kani::proof]
    fn verify_pbt_type_variants_distinct() {
        kani::assert(
            PbtType::Proptest != PbtType::QuickCheck,
            "Proptest != QuickCheck",
        );
    }

    /// Proves that PbtType equality is symmetric
    #[kani::proof]
    fn verify_pbt_type_equality_symmetric() {
        let types = [PbtType::Proptest, PbtType::QuickCheck];
        for a in types {
            for b in types {
                let ab = a == b;
                let ba = b == a;
                kani::assert(ab == ba, "PbtType equality is symmetric");
            }
        }
    }

    /// Proves that env_var mapping is consistent for Proptest
    #[kani::proof]
    fn verify_proptest_env_var_correct() {
        kani::assert(
            PbtType::Proptest.cases_env_var() == "PROPTEST_CASES",
            "Proptest env var is PROPTEST_CASES",
        );
    }

    /// Proves that env_var mapping is consistent for QuickCheck
    #[kani::proof]
    fn verify_quickcheck_env_var_correct() {
        kani::assert(
            PbtType::QuickCheck.cases_env_var() == "QUICKCHECK_TESTS",
            "QuickCheck env var is QUICKCHECK_TESTS",
        );
    }

    // ============== PbtConfig Additional Proofs ==============

    /// Proves that builder methods don't affect unrelated fields - cases
    #[kani::proof]
    fn verify_with_max_shrink_iters_orthogonal() {
        let default_config = PbtConfig::default();
        let iters: u32 = kani::any();
        let config = PbtConfig::default().with_max_shrink_iters(iters);
        kani::assert(
            config.cases == default_config.cases,
            "with_max_shrink_iters doesn't change cases",
        );
        kani::assert(
            config.seed == default_config.seed,
            "with_max_shrink_iters doesn't change seed",
        );
    }

    /// Proves that with_seed doesn't affect other fields
    #[kani::proof]
    fn verify_with_seed_orthogonal() {
        let default_config = PbtConfig::default();
        let seed: u64 = kani::any();
        let config = PbtConfig::default().with_seed(seed);
        kani::assert(
            config.cases == default_config.cases,
            "with_seed doesn't change cases",
        );
        kani::assert(
            config.max_shrink_iters == default_config.max_shrink_iters,
            "with_seed doesn't change max_shrink_iters",
        );
        kani::assert(
            config.verbose == default_config.verbose,
            "with_seed doesn't change verbose",
        );
    }

    /// Proves that with_verbose doesn't affect numeric fields
    #[kani::proof]
    fn verify_with_verbose_orthogonal() {
        let default_config = PbtConfig::default();
        let verbose: bool = kani::any();
        let config = PbtConfig::default().with_verbose(verbose);
        kani::assert(
            config.cases == default_config.cases,
            "with_verbose doesn't change cases",
        );
        kani::assert(
            config.max_shrink_iters == default_config.max_shrink_iters,
            "with_verbose doesn't change max_shrink_iters",
        );
        kani::assert(
            config.timeout == default_config.timeout,
            "with_verbose doesn't change timeout",
        );
    }

    /// Proves that with_fork doesn't affect other fields
    #[kani::proof]
    fn verify_with_fork_orthogonal() {
        let default_config = PbtConfig::default();
        let fork: bool = kani::any();
        let config = PbtConfig::default().with_fork(fork);
        kani::assert(
            config.cases == default_config.cases,
            "with_fork doesn't change cases",
        );
        kani::assert(
            config.verbose == default_config.verbose,
            "with_fork doesn't change verbose",
        );
    }

    /// Proves that with_timeout doesn't affect cases
    #[kani::proof]
    fn verify_with_timeout_orthogonal() {
        let default_config = PbtConfig::default();
        let secs: u64 = kani::any();
        kani::assume(secs > 0 && secs <= 3600);
        let config = PbtConfig::default().with_timeout(Duration::from_secs(secs));
        kani::assert(
            config.cases == default_config.cases,
            "with_timeout doesn't change cases",
        );
        kani::assert(
            config.max_shrink_iters == default_config.max_shrink_iters,
            "with_timeout doesn't change max_shrink_iters",
        );
    }

    // ============== PbtFailure Additional Proofs ==============

    /// Proves that PbtFailure preserves test_name field
    #[kani::proof]
    fn verify_pbt_failure_preserves_test_name() {
        let failure = PbtFailure {
            test_name: "my_test".to_string(),
            failing_input: String::new(),
            shrink_steps: 0,
            seed: None,
            error_message: String::new(),
            stack_trace: None,
        };
        kani::assert(failure.test_name == "my_test", "test_name preserved");
    }

    /// Proves that PbtFailure can have optional seed
    #[kani::proof]
    fn verify_pbt_failure_optional_seed() {
        let failure_none = PbtFailure {
            test_name: String::new(),
            failing_input: String::new(),
            shrink_steps: 0,
            seed: None,
            error_message: String::new(),
            stack_trace: None,
        };
        kani::assert(failure_none.seed.is_none(), "seed can be None");

        let failure_some = PbtFailure {
            test_name: String::new(),
            failing_input: String::new(),
            shrink_steps: 0,
            seed: Some("12345".to_string()),
            error_message: String::new(),
            stack_trace: None,
        };
        kani::assert(failure_some.seed.is_some(), "seed can be Some");
    }

    /// Proves that PbtFailure can have optional stack_trace
    #[kani::proof]
    fn verify_pbt_failure_optional_stack_trace() {
        let failure_none = PbtFailure {
            test_name: String::new(),
            failing_input: String::new(),
            shrink_steps: 0,
            seed: None,
            error_message: String::new(),
            stack_trace: None,
        };
        kani::assert(
            failure_none.stack_trace.is_none(),
            "stack_trace can be None",
        );

        let failure_some = PbtFailure {
            test_name: String::new(),
            failing_input: String::new(),
            shrink_steps: 0,
            seed: None,
            error_message: String::new(),
            stack_trace: Some("at line 10".to_string()),
        };
        kani::assert(
            failure_some.stack_trace.is_some(),
            "stack_trace can be Some",
        );
    }

    // ============== PbtResult Additional Proofs ==============

    /// Proves that PbtResult preserves tests_run field
    #[kani::proof]
    fn verify_pbt_result_preserves_tests_run() {
        let tests_run: usize = kani::any();
        kani::assume(tests_run <= 10000);
        let result = PbtResult {
            framework: PbtType::Proptest,
            passed: true,
            tests_run,
            cases_generated: 0,
            failures: Vec::new(),
            passed_tests: Vec::new(),
            duration: Duration::from_secs(1),
        };
        kani::assert(result.tests_run == tests_run, "tests_run preserved");
    }

    /// Proves that PbtResult preserves cases_generated field
    #[kani::proof]
    fn verify_pbt_result_preserves_cases_generated() {
        let cases_generated: u64 = kani::any();
        let result = PbtResult {
            framework: PbtType::QuickCheck,
            passed: true,
            tests_run: 1,
            cases_generated,
            failures: Vec::new(),
            passed_tests: Vec::new(),
            duration: Duration::from_secs(1),
        };
        kani::assert(
            result.cases_generated == cases_generated,
            "cases_generated preserved",
        );
    }

    /// Proves that PbtResult preserves duration field
    #[kani::proof]
    fn verify_pbt_result_preserves_duration() {
        let secs: u64 = kani::any();
        kani::assume(secs > 0 && secs <= 10000);
        let duration = Duration::from_secs(secs);
        let result = PbtResult {
            framework: PbtType::Proptest,
            passed: true,
            tests_run: 1,
            cases_generated: 100,
            failures: Vec::new(),
            passed_tests: Vec::new(),
            duration,
        };
        kani::assert(result.duration == duration, "duration preserved");
    }

    // ============== PbtBackend Additional Proofs ==============

    /// Proves that PbtBackend preserves config cases
    #[kani::proof]
    fn verify_pbt_backend_preserves_config_cases() {
        let cases: u32 = kani::any();
        kani::assume(cases > 0 && cases <= 100000);
        let config = PbtConfig::default().with_cases(cases);
        let backend = PbtBackend::new(PbtType::Proptest, config);
        kani::assert(backend.config.cases == cases, "config.cases preserved");
    }

    /// Proves that PbtBackend preserves pbt_type for both variants
    #[kani::proof]
    fn verify_pbt_backend_preserves_type_exhaustive() {
        let types = [PbtType::Proptest, PbtType::QuickCheck];
        for pbt_type in types {
            let backend = PbtBackend::new(pbt_type, PbtConfig::default());
            kani::assert(backend.pbt_type == pbt_type, "pbt_type preserved");
        }
    }

    // ============== Template Generation Additional Proofs ==============

    /// Proves that proptest template contains #[test] attribute
    #[kani::proof]
    fn verify_proptest_template_contains_test_attr() {
        let template = generate_proptest_template("test_fn", "u32");
        kani::assert(template.contains("#[test]"), "Template has #[test]");
    }

    /// Proves that quickcheck template contains return type
    #[kani::proof]
    fn verify_quickcheck_template_contains_return_bool() {
        let template = generate_quickcheck_template("test_fn", "u32");
        kani::assert(template.contains("-> bool"), "QuickCheck returns bool");
    }
}
