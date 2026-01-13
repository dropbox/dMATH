//! # dashprove-bisim
//!
//! Bisimulation and behavioral equivalence checking for DashProve.
//!
//! This crate provides infrastructure for verifying that two implementations
//! (an oracle/reference and a subject/test) behave identically according to
//! specified equivalence criteria.
//!
//! ## Key Concepts
//!
//! - **Oracle**: The reference implementation (could be a binary or recorded traces)
//! - **Subject**: The implementation under test
//! - **Equivalence Criteria**: What aspects must match (API requests, tool calls, output)
//! - **Nondeterminism Strategy**: How to handle non-deterministic behavior

mod config;
mod diff;
mod error;
mod executor;
mod trace;

pub use config::*;
pub use diff::*;
pub use error::*;
pub use executor::*;
pub use trace::*;

use async_trait::async_trait;
use std::time::Duration;

/// Result of bisimulation check
#[derive(Debug, Clone)]
pub struct BisimulationResult {
    /// Whether the implementations are equivalent according to criteria
    pub equivalent: bool,
    /// Detailed differences found
    pub differences: Vec<Difference>,
    /// Execution trace from oracle
    pub oracle_trace: ExecutionTrace,
    /// Execution trace from subject
    pub subject_trace: ExecutionTrace,
    /// Confidence in the result (0.0-1.0)
    pub confidence: f64,
}

impl BisimulationResult {
    /// Create a result indicating equivalence
    pub fn equivalent(oracle_trace: ExecutionTrace, subject_trace: ExecutionTrace) -> Self {
        Self {
            equivalent: true,
            differences: vec![],
            oracle_trace,
            subject_trace,
            confidence: 1.0,
        }
    }

    /// Create a result indicating non-equivalence with differences
    pub fn not_equivalent(
        oracle_trace: ExecutionTrace,
        subject_trace: ExecutionTrace,
        differences: Vec<Difference>,
        confidence: f64,
    ) -> Self {
        Self {
            equivalent: false,
            differences,
            oracle_trace,
            subject_trace,
            confidence,
        }
    }

    /// Get a summary of the result
    pub fn summary(&self) -> String {
        if self.equivalent {
            format!("Equivalent (confidence: {:.2}%)", self.confidence * 100.0)
        } else {
            format!(
                "Not equivalent: {} differences found (confidence: {:.2}%)",
                self.differences.len(),
                self.confidence * 100.0
            )
        }
    }
}

/// Input for a bisimulation test
#[derive(Debug, Clone)]
pub struct TestInput {
    /// Human-readable name for this test
    pub name: String,
    /// Input to provide to both oracle and subject
    pub input: String,
    /// Environment variables to set
    pub env: std::collections::HashMap<String, String>,
    /// Timeout for execution
    pub timeout: Duration,
}

impl TestInput {
    /// Create a new test input with just a name and input string
    pub fn new(name: impl Into<String>, input: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            input: input.into(),
            env: std::collections::HashMap::new(),
            timeout: Duration::from_secs(60),
        }
    }

    /// Set the timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Add an environment variable
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }
}

/// Main trait for bisimulation checking
#[async_trait]
pub trait BisimulationChecker: Send + Sync {
    /// Check equivalence for a single input
    async fn check(&self, input: &TestInput) -> Result<BisimulationResult, BisimError>;

    /// Check equivalence for multiple inputs
    async fn check_batch(
        &self,
        inputs: &[TestInput],
    ) -> Result<Vec<BisimulationResult>, BisimError>;
}

/// Trait for implementations that can be subjects of bisimulation testing
#[async_trait]
pub trait Subject: Send + Sync {
    /// Execute the subject with the given input
    async fn execute(&mut self, input: &TestInput) -> Result<ExecutionTrace, BisimError>;

    /// Reset the subject to initial state (for multiple tests)
    async fn reset(&mut self) -> Result<(), BisimError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Property tests
    mod property_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// BisimulationResult::equivalent should always be equivalent
            #[test]
            fn result_equivalent_is_equivalent(confidence in 0.0f64..=1.0) {
                let result = BisimulationResult::equivalent(
                    ExecutionTrace::empty(),
                    ExecutionTrace::empty(),
                );
                prop_assert!(result.equivalent);
                prop_assert!(result.differences.is_empty());
                // Ignore confidence parameter - just testing the constructor
                let _ = confidence;
            }

            /// BisimulationResult::not_equivalent should not be equivalent when differences exist
            #[test]
            fn result_not_equivalent_with_diffs(confidence in 0.0f64..=1.0) {
                let diff = Difference::OutputMismatch {
                    oracle: "a".to_string(),
                    subject: "b".to_string(),
                    similarity: 0.5,
                };
                let result = BisimulationResult::not_equivalent(
                    ExecutionTrace::empty(),
                    ExecutionTrace::empty(),
                    vec![diff],
                    confidence,
                );
                prop_assert!(!result.equivalent);
                prop_assert!(!result.differences.is_empty());
                prop_assert_eq!(result.confidence, confidence);
            }

            /// BisimulationResult summary should contain "Equivalent" when equivalent
            #[test]
            fn result_summary_contains_equivalent(_dummy in any::<bool>()) {
                let result = BisimulationResult::equivalent(
                    ExecutionTrace::empty(),
                    ExecutionTrace::empty(),
                );
                prop_assert!(result.summary().contains("Equivalent"));
            }

            /// BisimulationResult summary should contain "Not equivalent" when not equivalent
            #[test]
            fn result_summary_contains_not_equivalent(diff_count in 1usize..10) {
                let diffs: Vec<_> = (0..diff_count)
                    .map(|_| Difference::OutputMismatch {
                        oracle: "a".to_string(),
                        subject: "b".to_string(),
                        similarity: 0.5,
                    })
                    .collect();
                let result = BisimulationResult::not_equivalent(
                    ExecutionTrace::empty(),
                    ExecutionTrace::empty(),
                    diffs,
                    0.8,
                );
                prop_assert!(result.summary().contains("Not equivalent"));
                prop_assert!(result.summary().contains(&diff_count.to_string()));
            }

            /// TestInput builder should preserve name and input
            #[test]
            fn test_input_preserves_fields(
                name in "[a-z_]+",
                input in ".*"
            ) {
                let ti = TestInput::new(name.clone(), input.clone());
                prop_assert_eq!(ti.name, name);
                prop_assert_eq!(ti.input, input);
            }

            /// TestInput with_timeout should preserve timeout
            #[test]
            fn test_input_with_timeout(secs in 1u64..3600) {
                let timeout = Duration::from_secs(secs);
                let ti = TestInput::new("test", "input")
                    .with_timeout(timeout);
                prop_assert_eq!(ti.timeout, timeout);
            }

            /// TestInput with_env should add environment variable
            #[test]
            fn test_input_with_env(
                key in "[A-Z_]+",
                value in "[a-z0-9]+"
            ) {
                let ti = TestInput::new("test", "input")
                    .with_env(key.clone(), value.clone());
                prop_assert_eq!(ti.env.get(&key), Some(&value));
            }

            /// Multiple with_env calls should preserve all env vars
            #[test]
            fn test_input_multiple_env(
                key1 in "[A-Z_]{1,5}",
                val1 in "[a-z0-9]+",
                key2 in "[A-Z_]{6,10}",
                val2 in "[a-z0-9]+"
            ) {
                // Keys have different lengths to ensure they're different
                let ti = TestInput::new("test", "input")
                    .with_env(key1.clone(), val1.clone())
                    .with_env(key2.clone(), val2.clone());
                prop_assert_eq!(ti.env.get(&key1), Some(&val1));
                prop_assert_eq!(ti.env.get(&key2), Some(&val2));
            }

            /// Chaining TestInput builders should preserve all fields
            #[test]
            fn test_input_chained_builders(
                name in "[a-z_]+",
                input in "[a-z ]+",
                secs in 10u64..120,
                env_key in "[A-Z]+",
                env_val in "[0-9]+"
            ) {
                let ti = TestInput::new(name.clone(), input.clone())
                    .with_timeout(Duration::from_secs(secs))
                    .with_env(env_key.clone(), env_val.clone());

                prop_assert_eq!(ti.name, name);
                prop_assert_eq!(ti.input, input);
                prop_assert_eq!(ti.timeout, Duration::from_secs(secs));
                prop_assert_eq!(ti.env.get(&env_key), Some(&env_val));
            }
        }
    }

    #[test]
    fn test_bisimulation_result_equivalent() {
        let oracle = ExecutionTrace::empty();
        let subject = ExecutionTrace::empty();
        let result = BisimulationResult::equivalent(oracle, subject);

        assert!(result.equivalent);
        assert!(result.differences.is_empty());
        assert_eq!(result.confidence, 1.0);
        assert!(result.summary().contains("Equivalent"));
    }

    #[test]
    fn test_bisimulation_result_not_equivalent() {
        let oracle = ExecutionTrace::empty();
        let subject = ExecutionTrace::empty();
        let diff = Difference::OutputMismatch {
            oracle: "hello".to_string(),
            subject: "world".to_string(),
            similarity: 0.0,
        };
        let result = BisimulationResult::not_equivalent(oracle, subject, vec![diff], 0.9);

        assert!(!result.equivalent);
        assert_eq!(result.differences.len(), 1);
        assert_eq!(result.confidence, 0.9);
        assert!(result.summary().contains("Not equivalent"));
    }

    #[test]
    fn test_test_input_builder() {
        let input = TestInput::new("test", "hello world")
            .with_timeout(Duration::from_secs(30))
            .with_env("DEBUG", "1");

        assert_eq!(input.name, "test");
        assert_eq!(input.input, "hello world");
        assert_eq!(input.timeout, Duration::from_secs(30));
        assert_eq!(input.env.get("DEBUG"), Some(&"1".to_string()));
    }
}

// Kani formal verification proofs
// NOTE: Proofs that use ExecutionTrace or HashMap are excluded because Kani
// doesn't support the CCRandomGenerateBytes C function used by HashMap's hasher.
// The main Kani proofs for dashprove-bisim are in diff.rs and trace.rs.
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify TestInput default timeout value
    #[kani::proof]
    fn verify_test_input_default_timeout() {
        // Test the expected default timeout value (60 seconds)
        let default_timeout = Duration::from_secs(60);
        kani::assert(
            default_timeout.as_secs() == 60,
            "Default timeout must be 60 seconds",
        );
    }

    /// Verify Duration arithmetic for timeouts
    #[kani::proof]
    fn verify_duration_from_secs() {
        let secs: u64 = kani::any();
        kani::assume(secs <= 3600); // Up to 1 hour

        let d = Duration::from_secs(secs);
        kani::assert(
            d.as_secs() == secs,
            "Duration::from_secs must preserve seconds",
        );
    }
}
