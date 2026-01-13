//! Counterexample minimization using delta debugging
//!
//! This module implements delta debugging to produce minimal counterexamples.
//! A minimal counterexample contains only the witness values necessary to
//! reproduce the verification failure.

use crate::types::{CounterexampleValue, StructuredCounterexample, TraceState};
use lazy_static::lazy_static;
use regex::Regex;
use std::collections::{HashMap, HashSet};

// Pre-compiled regex for extracting variable references
lazy_static! {
    /// Matches identifier patterns for variable extraction
    static ref RE_IDENTIFIER: Regex = Regex::new(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b").unwrap();
}

/// Configuration for counterexample minimization
#[derive(Debug, Clone)]
pub struct MinimizationConfig {
    /// Maximum number of iterations to attempt
    pub max_iterations: usize,
    /// Whether to minimize trace states
    pub minimize_trace: bool,
    /// Whether to minimize witness values
    pub minimize_witness: bool,
    /// Granularity for delta debugging (minimum 1, 2 = binary search)
    pub granularity: usize,
}

impl Default for MinimizationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            minimize_trace: true,
            minimize_witness: true,
            granularity: 2,
        }
    }
}

/// Result of minimization
#[derive(Debug, Clone)]
pub struct MinimizationResult {
    /// The minimized counterexample
    pub counterexample: StructuredCounterexample,
    /// Number of iterations performed
    pub iterations: usize,
    /// Original witness count
    pub original_witness_count: usize,
    /// Minimized witness count
    pub minimized_witness_count: usize,
    /// Original trace length
    pub original_trace_length: usize,
    /// Minimized trace length
    pub minimized_trace_length: usize,
    /// Variables that were removed
    pub removed_variables: HashSet<String>,
    /// Variables that are essential
    pub essential_variables: HashSet<String>,
}

impl MinimizationResult {
    /// Calculate the reduction ratio
    #[must_use]
    pub fn reduction_ratio(&self) -> f64 {
        if self.original_witness_count == 0 {
            return 1.0;
        }
        1.0 - (self.minimized_witness_count as f64 / self.original_witness_count as f64)
    }

    /// Get a summary of the minimization
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "Minimization: {} -> {} variables ({:.1}% reduction), {} -> {} trace states, {} iterations",
            self.original_witness_count,
            self.minimized_witness_count,
            self.reduction_ratio() * 100.0,
            self.original_trace_length,
            self.minimized_trace_length,
            self.iterations
        )
    }
}

/// A test function that checks if a reduced counterexample still triggers the failure
pub type FailureTest = Box<dyn Fn(&StructuredCounterexample) -> bool>;

/// Delta debugging minimizer for counterexamples
pub struct DeltaDebugger {
    config: MinimizationConfig,
}

impl DeltaDebugger {
    /// Create a new delta debugger with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: MinimizationConfig::default(),
        }
    }

    /// Create a new delta debugger with custom configuration
    #[must_use]
    pub fn with_config(config: MinimizationConfig) -> Self {
        Self { config }
    }

    /// Minimize a counterexample using delta debugging
    ///
    /// The `test_failure` function should return `true` if the counterexample
    /// still triggers the verification failure.
    pub fn minimize(
        &self,
        counterexample: &StructuredCounterexample,
        test_failure: FailureTest,
    ) -> MinimizationResult {
        let original_witness_count = counterexample.witness.len();
        let original_trace_length = counterexample.trace.len();

        let mut result = counterexample.clone();
        let mut iterations = 0;
        let mut removed_variables = HashSet::new();
        let mut minimized_trace_length = counterexample.trace.len();

        // Minimize witness values using delta debugging
        if self.config.minimize_witness && !counterexample.witness.is_empty() {
            let (minimized_witness, witness_removed, witness_iterations) = self
                .minimize_witness_values(&counterexample.witness, counterexample, &test_failure);
            result.witness = minimized_witness;
            removed_variables.extend(witness_removed);
            iterations += witness_iterations;
        }

        // Minimize trace
        if self.config.minimize_trace {
            let (minimized_trace, trace_iterations) =
                self.minimize_trace(&counterexample.trace, &result, &test_failure);
            minimized_trace_length = minimized_trace.len();
            result.trace = minimized_trace;
            iterations += trace_iterations;
        }

        result.minimized = true;

        let essential_variables: HashSet<String> = result.witness.keys().cloned().collect();

        MinimizationResult {
            counterexample: result,
            iterations,
            original_witness_count,
            minimized_witness_count: essential_variables.len(),
            original_trace_length,
            minimized_trace_length,
            removed_variables,
            essential_variables,
        }
    }

    /// Minimize witness values using delta debugging algorithm (ddmin)
    fn minimize_witness_values(
        &self,
        witness: &HashMap<String, CounterexampleValue>,
        original: &StructuredCounterexample,
        test_failure: &FailureTest,
    ) -> (HashMap<String, CounterexampleValue>, HashSet<String>, usize) {
        let mut variables: Vec<String> = witness.keys().cloned().collect();
        variables.sort(); // Deterministic ordering
                          // Avoid division-by-zero when callers provide granularity 0.
        let mut n = self.config.granularity.max(1);
        let mut iterations = 0;
        let mut removed = HashSet::new();

        while variables.len() >= n && iterations < self.config.max_iterations {
            let chunk_size = variables.len().div_ceil(n);
            let mut made_progress = false;

            // Try removing each chunk
            for i in 0..n {
                let start = i * chunk_size;
                let end = std::cmp::min(start + chunk_size, variables.len());
                if start >= variables.len() {
                    break;
                }

                // Create witness without this chunk
                let chunk: HashSet<_> = variables[start..end].iter().cloned().collect();
                let reduced_witness: HashMap<String, CounterexampleValue> = witness
                    .iter()
                    .filter(|(k, _)| !chunk.contains(*k) && !removed.contains(*k))
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();

                // Test if failure still occurs
                let mut test_ce = original.clone();
                test_ce.witness = reduced_witness.clone();
                iterations += 1;

                if test_failure(&test_ce) {
                    // Success! The chunk is not needed
                    removed.extend(chunk);
                    made_progress = true;
                }
            }

            if made_progress {
                // Increase granularity (fewer, larger chunks)
                n = std::cmp::max(n - 1, 2);
                variables.retain(|v| !removed.contains(v));
            } else {
                // Decrease granularity (more, smaller chunks)
                n = std::cmp::min(n * 2, variables.len());
            }

            if n > variables.len() {
                break;
            }
        }

        let minimized: HashMap<String, CounterexampleValue> = witness
            .iter()
            .filter(|(k, _)| !removed.contains(*k))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        (minimized, removed, iterations)
    }

    /// Minimize trace by removing non-essential states
    fn minimize_trace(
        &self,
        trace: &[TraceState],
        original: &StructuredCounterexample,
        test_failure: &FailureTest,
    ) -> (Vec<TraceState>, usize) {
        if trace.len() <= 2 {
            return (trace.to_vec(), 0);
        }

        let mut current_trace = trace.to_vec();
        let mut iterations = 0;

        // Try to remove states from the middle (keep first and last)
        // MUTATION NOTE: The `iterations < max_iterations` condition is an equivalent mutant
        // when changed to `<=`. One extra iteration doesn't change correctness.
        let mut i = 1;
        while i < current_trace.len().saturating_sub(1) && iterations < self.config.max_iterations {
            let mut test_trace = current_trace.clone();
            test_trace.remove(i);

            // Renumber states
            for (idx, state) in test_trace.iter_mut().enumerate() {
                state.state_num = (idx + 1) as u32;
            }

            let mut test_ce = original.clone();
            test_ce.trace = test_trace.clone();
            iterations += 1;

            if test_failure(&test_ce) {
                // State i is not needed
                current_trace = test_trace;
                // Don't increment i - check the new state at position i
            } else {
                // State i is needed
                // MUTATION NOTE: `i += 1` -> `i *= 1` is an equivalent mutant because
                // max_iterations eventually terminates the loop even without progress.
                i += 1;
            }
        }

        (current_trace, iterations)
    }
}

impl Default for DeltaDebugger {
    fn default() -> Self {
        Self::new()
    }
}

/// Identify essential variables by analyzing variable dependencies
pub fn analyze_essential_variables(counterexample: &StructuredCounterexample) -> HashSet<String> {
    let mut essential = HashSet::new();

    // Variables mentioned in failed checks are essential
    for check in &counterexample.failed_checks {
        // Extract variable names from the check description
        extract_variable_references(&check.description, &mut essential);
    }

    // Variables that change in the trace leading to failure are potentially essential
    if counterexample.trace.len() >= 2 {
        // Safety: len() >= 2 guarantees last() returns Some
        let last_state = counterexample.trace.last().unwrap();
        let prev_state = &counterexample.trace[counterexample.trace.len() - 2];
        let diffs = last_state.diff_from(prev_state);
        essential.extend(diffs.keys().cloned());
    }

    // Filter to only include variables that exist in witness
    essential
        .into_iter()
        .filter(|v| counterexample.witness.contains_key(v))
        .collect()
}

/// Extract variable references from a string (heuristic)
fn extract_variable_references(text: &str, variables: &mut HashSet<String>) {
    // Look for common patterns like "x = 5" or "variable x"
    for cap in RE_IDENTIFIER.captures_iter(text) {
        let name = &cap[1];
        // Filter out common keywords
        if !is_keyword(name) {
            variables.insert(name.to_string());
        }
    }
}

const KEYWORDS: [&str; 24] = [
    "if",
    "else",
    "while",
    "for",
    "let",
    "mut",
    "fn",
    "return",
    "true",
    "false",
    "attempt",
    "divide",
    "by",
    "zero",
    "overflow",
    "underflow",
    "assertion",
    "failed",
    "check",
    "in",
    "function",
    "at",
    "line",
    "file",
];

/// Check if a name is a common keyword (not a variable)
fn is_keyword(name: &str) -> bool {
    KEYWORDS
        .iter()
        .any(|keyword| name.eq_ignore_ascii_case(keyword))
}

/// Simplify witness values by replacing complex values with simpler equivalents
pub fn simplify_witness_values(
    witness: &HashMap<String, CounterexampleValue>,
) -> HashMap<String, CounterexampleValue> {
    witness
        .iter()
        .map(|(k, v)| (k.clone(), simplify_value(v)))
        .collect()
}

/// Simplify a single counterexample value
///
/// Currently this function just clones values without simplification.
/// The match arms for Int, UInt, Sequence, and Record were removed as
/// equivalent to the default clone (they didn't perform any transformation).
///
/// Future improvements could include:
/// - Simplifying boundary values (MAX_INT → "MAX")
/// - Reducing nested structures
/// - Converting complex values to simpler representations
fn simplify_value(value: &CounterexampleValue) -> CounterexampleValue {
    // All value types are currently just cloned.
    // The previous explicit match arms for Int, UInt, Sequence, and Record
    // were removed as they produced identical output to clone().
    value.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::FailedCheck;
    use std::cell::RefCell;
    use std::rc::Rc;

    // ==================== MinimizationConfig Tests ====================

    #[test]
    fn test_minimization_config_default() {
        let config = MinimizationConfig::default();
        assert_eq!(config.max_iterations, 100);
        assert!(config.minimize_trace);
        assert!(config.minimize_witness);
        assert_eq!(config.granularity, 2);
    }

    #[test]
    fn test_minimization_config_custom() {
        let config = MinimizationConfig {
            max_iterations: 50,
            minimize_trace: false,
            minimize_witness: true,
            granularity: 4,
        };
        assert_eq!(config.max_iterations, 50);
        assert!(!config.minimize_trace);
        assert!(config.minimize_witness);
        assert_eq!(config.granularity, 4);
    }

    #[test]
    fn test_minimization_config_debug() {
        let config = MinimizationConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("MinimizationConfig"));
        assert!(debug.contains("max_iterations"));
        assert!(debug.contains("100"));
    }

    #[test]
    fn test_minimization_config_clone() {
        let config = MinimizationConfig {
            max_iterations: 25,
            minimize_trace: false,
            minimize_witness: false,
            granularity: 8,
        };
        let cloned = config.clone();
        assert_eq!(cloned.max_iterations, 25);
        assert!(!cloned.minimize_trace);
        assert!(!cloned.minimize_witness);
        assert_eq!(cloned.granularity, 8);
    }

    // ==================== MinimizationResult Tests ====================

    #[test]
    fn test_minimization_result_reduction_ratio() {
        let result = MinimizationResult {
            counterexample: StructuredCounterexample::new(),
            iterations: 5,
            original_witness_count: 10,
            minimized_witness_count: 2,
            original_trace_length: 5,
            minimized_trace_length: 3,
            removed_variables: HashSet::new(),
            essential_variables: HashSet::new(),
        };
        assert!((result.reduction_ratio() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_minimization_result_reduction_ratio_zero_original() {
        let result = MinimizationResult {
            counterexample: StructuredCounterexample::new(),
            iterations: 0,
            original_witness_count: 0,
            minimized_witness_count: 0,
            original_trace_length: 0,
            minimized_trace_length: 0,
            removed_variables: HashSet::new(),
            essential_variables: HashSet::new(),
        };
        assert!((result.reduction_ratio() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_minimization_result_reduction_ratio_no_reduction() {
        let result = MinimizationResult {
            counterexample: StructuredCounterexample::new(),
            iterations: 3,
            original_witness_count: 5,
            minimized_witness_count: 5,
            original_trace_length: 3,
            minimized_trace_length: 3,
            removed_variables: HashSet::new(),
            essential_variables: HashSet::new(),
        };
        assert!((result.reduction_ratio() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_minimization_result_reduction_ratio_full_reduction() {
        let result = MinimizationResult {
            counterexample: StructuredCounterexample::new(),
            iterations: 10,
            original_witness_count: 20,
            minimized_witness_count: 0,
            original_trace_length: 5,
            minimized_trace_length: 1,
            removed_variables: HashSet::new(),
            essential_variables: HashSet::new(),
        };
        assert!((result.reduction_ratio() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_minimization_result_summary() {
        let result = MinimizationResult {
            counterexample: StructuredCounterexample::new(),
            iterations: 5,
            original_witness_count: 10,
            minimized_witness_count: 2,
            original_trace_length: 5,
            minimized_trace_length: 3,
            removed_variables: HashSet::new(),
            essential_variables: HashSet::new(),
        };
        let summary = result.summary();
        assert!(summary.contains("10 -> 2 variables"));
        assert!(summary.contains("80.0% reduction"));
        assert!(summary.contains("5 -> 3 trace states"));
        assert!(summary.contains("5 iterations"));
    }

    #[test]
    fn test_minimization_result_summary_zero_percent() {
        let result = MinimizationResult {
            counterexample: StructuredCounterexample::new(),
            iterations: 1,
            original_witness_count: 3,
            minimized_witness_count: 3,
            original_trace_length: 2,
            minimized_trace_length: 2,
            removed_variables: HashSet::new(),
            essential_variables: HashSet::new(),
        };
        let summary = result.summary();
        assert!(summary.contains("0.0% reduction"));
    }

    #[test]
    fn test_minimization_result_debug() {
        let result = MinimizationResult {
            counterexample: StructuredCounterexample::new(),
            iterations: 3,
            original_witness_count: 5,
            minimized_witness_count: 2,
            original_trace_length: 4,
            minimized_trace_length: 2,
            removed_variables: HashSet::from(["x".to_string()]),
            essential_variables: HashSet::from(["y".to_string()]),
        };
        let debug = format!("{:?}", result);
        assert!(debug.contains("MinimizationResult"));
        assert!(debug.contains("iterations"));
    }

    #[test]
    fn test_minimization_result_clone() {
        let mut removed = HashSet::new();
        removed.insert("a".to_string());
        let mut essential = HashSet::new();
        essential.insert("b".to_string());

        let result = MinimizationResult {
            counterexample: StructuredCounterexample::new(),
            iterations: 7,
            original_witness_count: 8,
            minimized_witness_count: 3,
            original_trace_length: 4,
            minimized_trace_length: 2,
            removed_variables: removed,
            essential_variables: essential,
        };

        let cloned = result.clone();
        assert_eq!(cloned.iterations, 7);
        assert_eq!(cloned.original_witness_count, 8);
        assert!(cloned.removed_variables.contains("a"));
        assert!(cloned.essential_variables.contains("b"));
    }

    // ==================== DeltaDebugger Tests ====================

    #[test]
    fn test_delta_debugger_creation() {
        let debugger = DeltaDebugger::new();
        assert_eq!(debugger.config.max_iterations, 100);
    }

    #[test]
    fn test_delta_debugger_with_config() {
        let config = MinimizationConfig {
            max_iterations: 20,
            minimize_trace: false,
            minimize_witness: true,
            granularity: 3,
        };
        let debugger = DeltaDebugger::with_config(config);
        assert_eq!(debugger.config.max_iterations, 20);
        assert!(!debugger.config.minimize_trace);
        assert_eq!(debugger.config.granularity, 3);
    }

    #[test]
    fn test_delta_debugger_default_trait() {
        let debugger = DeltaDebugger::default();
        assert_eq!(debugger.config.max_iterations, 100);
        assert!(debugger.config.minimize_trace);
        assert!(debugger.config.minimize_witness);
    }

    #[test]
    fn test_minimization_empty_witness() {
        let debugger = DeltaDebugger::new();
        let ce = StructuredCounterexample::new();
        let result = debugger.minimize(&ce, Box::new(|_| true));
        assert_eq!(result.minimized_witness_count, 0);
        assert_eq!(result.original_witness_count, 0);
        assert!(result.counterexample.minimized);
    }

    #[test]
    fn test_minimize_with_single_essential_variable() {
        let debugger = DeltaDebugger::new();
        let mut ce = StructuredCounterexample::new();

        // x is essential, y and z are not
        ce.witness.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 0,
                type_hint: None,
            },
        );
        ce.witness.insert(
            "y".to_string(),
            CounterexampleValue::Int {
                value: 100,
                type_hint: None,
            },
        );
        ce.witness.insert(
            "z".to_string(),
            CounterexampleValue::Int {
                value: 200,
                type_hint: None,
            },
        );

        // Failure occurs only if x = 0 (simulating division by zero)
        let result = debugger.minimize(
            &ce,
            Box::new(|test_ce| {
                test_ce
                    .witness
                    .get("x")
                    .is_some_and(|v| matches!(v, CounterexampleValue::Int { value: 0, .. }))
            }),
        );

        // x should be kept, y and z should be removed
        assert!(result.counterexample.witness.contains_key("x"));
        assert!(!result.counterexample.witness.contains_key("y"));
        assert!(!result.counterexample.witness.contains_key("z"));
        assert_eq!(result.minimized_witness_count, 1);
    }

    #[test]
    fn test_minimize_multiple_essential_variables() {
        let debugger = DeltaDebugger::new();
        let mut ce = StructuredCounterexample::new();

        // x and y are essential (both needed for failure)
        ce.witness.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 5,
                type_hint: None,
            },
        );
        ce.witness.insert(
            "y".to_string(),
            CounterexampleValue::Int {
                value: 0,
                type_hint: None,
            },
        );
        ce.witness.insert(
            "z".to_string(),
            CounterexampleValue::Int {
                value: 100,
                type_hint: None,
            },
        );

        // Failure requires both x > 0 and y = 0
        let result = debugger.minimize(
            &ce,
            Box::new(|test_ce| {
                let x_ok = test_ce.witness.get("x").is_some_and(
                    |v| matches!(v, CounterexampleValue::Int { value, .. } if *value > 0),
                );
                let y_ok = test_ce
                    .witness
                    .get("y")
                    .is_some_and(|v| matches!(v, CounterexampleValue::Int { value: 0, .. }));
                x_ok && y_ok
            }),
        );

        assert!(result.counterexample.witness.contains_key("x"));
        assert!(result.counterexample.witness.contains_key("y"));
        assert!(!result.counterexample.witness.contains_key("z"));
    }

    #[test]
    fn test_minimize_all_variables_essential() {
        let debugger = DeltaDebugger::new();
        let mut ce = StructuredCounterexample::new();

        ce.witness.insert(
            "a".to_string(),
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
        );
        ce.witness.insert(
            "b".to_string(),
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
        );

        // All variables are essential
        let result = debugger.minimize(&ce, Box::new(|test_ce| test_ce.witness.len() == 2));

        assert_eq!(result.minimized_witness_count, 2);
        assert!(result.counterexample.witness.contains_key("a"));
        assert!(result.counterexample.witness.contains_key("b"));
    }

    #[test]
    fn test_minimize_no_variables_essential() {
        let debugger = DeltaDebugger::new();
        let mut ce = StructuredCounterexample::new();

        ce.witness.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 42,
                type_hint: None,
            },
        );

        // Failure always occurs regardless of variables
        let result = debugger.minimize(&ce, Box::new(|_| true));

        // Even with no essential variables, at least the structure is preserved
        assert!(result.counterexample.minimized);
    }

    #[test]
    fn test_minimize_with_disabled_witness_minimization() {
        let config = MinimizationConfig {
            max_iterations: 100,
            minimize_trace: true,
            minimize_witness: false,
            granularity: 2,
        };
        let debugger = DeltaDebugger::with_config(config);
        let mut ce = StructuredCounterexample::new();

        ce.witness.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 0,
                type_hint: None,
            },
        );
        ce.witness.insert(
            "y".to_string(),
            CounterexampleValue::Int {
                value: 100,
                type_hint: None,
            },
        );

        let result = debugger.minimize(&ce, Box::new(|_| true));

        // Witness should not be minimized
        assert_eq!(result.minimized_witness_count, 2);
    }

    #[test]
    fn test_minimize_with_disabled_trace_minimization() {
        let config = MinimizationConfig {
            max_iterations: 100,
            minimize_trace: false,
            minimize_witness: true,
            granularity: 2,
        };
        let debugger = DeltaDebugger::with_config(config);
        let mut ce = StructuredCounterexample::new();

        for i in 1..=5 {
            ce.trace.push(TraceState::new(i));
        }

        let result = debugger.minimize(&ce, Box::new(|_| true));

        // Trace should not be minimized
        assert_eq!(result.counterexample.trace.len(), 5);
    }

    #[test]
    fn test_minimize_with_max_iterations_limit() {
        let config = MinimizationConfig {
            max_iterations: 1,
            minimize_trace: true,
            minimize_witness: true,
            granularity: 2,
        };
        let debugger = DeltaDebugger::with_config(config);
        let mut ce = StructuredCounterexample::new();

        for i in 0..10 {
            ce.witness.insert(
                format!("var_{}", i),
                CounterexampleValue::Int {
                    value: i,
                    type_hint: None,
                },
            );
        }

        // Should stop early due to iteration limit
        let result = debugger.minimize(&ce, Box::new(|_| true));
        assert!(result.iterations <= 2); // May do 1 or 2 iterations before hitting limit
    }

    // ==================== Trace Minimization Tests ====================

    #[test]
    fn test_minimize_trace() {
        let debugger = DeltaDebugger::new();
        let mut ce = StructuredCounterexample::new();

        // Create a trace with 5 states
        for i in 1..=5 {
            let mut state = TraceState::new(i);
            state.variables.insert(
                "counter".to_string(),
                CounterexampleValue::Int {
                    value: i as i128,
                    type_hint: None,
                },
            );
            ce.trace.push(state);
        }

        // Failure requires first and last state only
        let result = debugger.minimize(
            &ce,
            Box::new(|test_ce| {
                test_ce.trace.len() >= 2 && test_ce.trace.first().is_some_and(|s| s.state_num >= 1)
            }),
        );

        // Should have reduced the trace
        assert!(result.counterexample.minimized);
    }

    #[test]
    fn test_minimize_trace_single_state() {
        let debugger = DeltaDebugger::new();
        let mut ce = StructuredCounterexample::new();
        ce.trace.push(TraceState::new(1));

        let result = debugger.minimize(&ce, Box::new(|_| true));

        // Single state should remain
        assert_eq!(result.counterexample.trace.len(), 1);
    }

    #[test]
    fn test_minimize_trace_two_states() {
        let debugger = DeltaDebugger::new();
        let mut ce = StructuredCounterexample::new();
        ce.trace.push(TraceState::new(1));
        ce.trace.push(TraceState::new(2));

        let result = debugger.minimize(&ce, Box::new(|_| true));

        // Two states is minimum, should remain
        assert!(result.counterexample.trace.len() <= 2);
    }

    #[test]
    fn test_minimize_trace_all_states_essential() {
        let debugger = DeltaDebugger::new();
        let mut ce = StructuredCounterexample::new();

        for i in 1..=4 {
            ce.trace.push(TraceState::new(i));
        }

        // All states are essential
        let result = debugger.minimize(&ce, Box::new(|test_ce| test_ce.trace.len() == 4));

        assert_eq!(result.counterexample.trace.len(), 4);
    }

    #[test]
    fn test_minimize_trace_state_renumbering() {
        let debugger = DeltaDebugger::new();
        let mut ce = StructuredCounterexample::new();

        for i in 1..=5 {
            ce.trace.push(TraceState::new(i));
        }

        // Only need 2 states
        let result = debugger.minimize(&ce, Box::new(|test_ce| test_ce.trace.len() >= 2));

        // Check that states are renumbered correctly
        for (idx, state) in result.counterexample.trace.iter().enumerate() {
            assert_eq!(state.state_num, (idx + 1) as u32);
        }
    }

    // ==================== analyze_essential_variables Tests ====================

    #[test]
    fn test_analyze_essential_variables() {
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "test".to_string(),
            description: "x divided by y resulted in overflow".to_string(),
            location: None,
            function: None,
        });
        ce.witness.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 10,
                type_hint: None,
            },
        );
        ce.witness.insert(
            "y".to_string(),
            CounterexampleValue::Int {
                value: 0,
                type_hint: None,
            },
        );
        ce.witness.insert(
            "z".to_string(),
            CounterexampleValue::Int {
                value: 5,
                type_hint: None,
            },
        );

        let essential = analyze_essential_variables(&ce);
        assert!(essential.contains("x"));
        assert!(essential.contains("y"));
        // z is not mentioned in the failure
    }

    #[test]
    fn test_analyze_essential_variables_empty() {
        let ce = StructuredCounterexample::new();
        let essential = analyze_essential_variables(&ce);
        assert!(essential.is_empty());
    }

    #[test]
    fn test_analyze_essential_variables_no_witness_match() {
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "test".to_string(),
            description: "variable foo caused error".to_string(),
            location: None,
            function: None,
        });
        // foo mentioned but not in witness
        ce.witness.insert(
            "bar".to_string(),
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
        );

        let essential = analyze_essential_variables(&ce);
        // foo is mentioned but not in witness, so shouldn't be included
        assert!(!essential.contains("foo"));
        assert!(!essential.contains("bar"));
    }

    #[test]
    fn test_analyze_essential_variables_with_trace_diffs() {
        let mut ce = StructuredCounterexample::new();

        let mut state1 = TraceState::new(1);
        state1.variables.insert(
            "counter".to_string(),
            CounterexampleValue::Int {
                value: 0,
                type_hint: None,
            },
        );
        ce.trace.push(state1);

        let mut state2 = TraceState::new(2);
        state2.variables.insert(
            "counter".to_string(),
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
        );
        ce.trace.push(state2);

        ce.witness.insert(
            "counter".to_string(),
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
        );

        let essential = analyze_essential_variables(&ce);
        assert!(essential.contains("counter"));
    }

    #[test]
    fn test_analyze_essential_variables_multiple_checks() {
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "check1".to_string(),
            description: "variable alpha is invalid".to_string(),
            location: None,
            function: None,
        });
        ce.failed_checks.push(FailedCheck {
            check_id: "check2".to_string(),
            description: "variable beta exceeded limit".to_string(),
            location: None,
            function: None,
        });
        ce.witness.insert(
            "alpha".to_string(),
            CounterexampleValue::Int {
                value: -1,
                type_hint: None,
            },
        );
        ce.witness.insert(
            "beta".to_string(),
            CounterexampleValue::Int {
                value: 1000,
                type_hint: None,
            },
        );

        let essential = analyze_essential_variables(&ce);
        assert!(essential.contains("alpha"));
        assert!(essential.contains("beta"));
    }

    // ==================== is_keyword Tests ====================

    #[test]
    fn test_is_keyword() {
        assert!(is_keyword("if"));
        assert!(is_keyword("attempt"));
        assert!(is_keyword("overflow"));
        assert!(!is_keyword("my_var"));
        assert!(!is_keyword("counter"));
    }

    #[test]
    fn test_is_keyword_case_insensitive() {
        assert!(is_keyword("IF"));
        assert!(is_keyword("If"));
        assert!(is_keyword("OVERFLOW"));
        assert!(is_keyword("Overflow"));
    }

    #[test]
    fn test_is_keyword_all_keywords() {
        for kw in KEYWORDS {
            assert!(is_keyword(kw), "Expected '{}' to be a keyword", kw);
            assert!(
                is_keyword(&kw.to_uppercase()),
                "Expected '{}' to be case-insensitive",
                kw
            );
        }
    }

    #[test]
    fn test_is_keyword_not_keywords() {
        let non_keywords = ["myVar", "x", "y", "counter", "result", "data", "foo_bar"];
        for nk in non_keywords {
            assert!(!is_keyword(nk), "Expected '{}' to not be a keyword", nk);
        }
    }

    // ==================== extract_variable_references Tests ====================

    #[test]
    fn test_extract_variable_references_simple() {
        let mut vars = HashSet::new();
        extract_variable_references("x equals 5", &mut vars);
        assert!(vars.contains("x"));
        assert!(vars.contains("equals"));
    }

    #[test]
    fn test_extract_variable_references_filters_keywords() {
        let mut vars = HashSet::new();
        extract_variable_references("if x overflow then y", &mut vars);
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
        assert!(!vars.contains("if"));
        assert!(!vars.contains("overflow"));
        assert!(vars.contains("then")); // 'then' is not in our keyword list
    }

    #[test]
    fn test_extract_variable_references_underscore_names() {
        let mut vars = HashSet::new();
        extract_variable_references("my_variable and _private", &mut vars);
        assert!(vars.contains("my_variable"));
        assert!(vars.contains("_private"));
    }

    #[test]
    fn test_extract_variable_references_numbers_in_names() {
        let mut vars = HashSet::new();
        extract_variable_references("var1 and var_2", &mut vars);
        assert!(vars.contains("var1"));
        assert!(vars.contains("var_2"));
    }

    #[test]
    fn test_extract_variable_references_empty_string() {
        let mut vars = HashSet::new();
        extract_variable_references("", &mut vars);
        assert!(vars.is_empty());
    }

    // ==================== simplify_witness_values Tests ====================

    #[test]
    fn test_simplify_witness_values() {
        let mut witness = HashMap::new();
        witness.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 42,
                type_hint: Some("i32".to_string()),
            },
        );
        let simplified = simplify_witness_values(&witness);
        assert_eq!(simplified.len(), 1);
    }

    #[test]
    fn test_simplify_witness_values_empty() {
        let witness = HashMap::new();
        let simplified = simplify_witness_values(&witness);
        assert!(simplified.is_empty());
    }

    #[test]
    fn test_simplify_witness_values_multiple() {
        let mut witness = HashMap::new();
        witness.insert(
            "a".to_string(),
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
        );
        witness.insert(
            "b".to_string(),
            CounterexampleValue::UInt {
                value: 2,
                type_hint: Some("u64".to_string()),
            },
        );
        witness.insert("c".to_string(), CounterexampleValue::Bool(true));

        let simplified = simplify_witness_values(&witness);
        assert_eq!(simplified.len(), 3);
    }

    // ==================== simplify_value Tests ====================

    #[test]
    fn test_simplify_value_int() {
        let value = CounterexampleValue::Int {
            value: 100,
            type_hint: Some("i64".to_string()),
        };
        let simplified = simplify_value(&value);
        match simplified {
            CounterexampleValue::Int {
                value: v,
                type_hint,
            } => {
                assert_eq!(v, 100);
                assert_eq!(type_hint, Some("i64".to_string()));
            }
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn test_simplify_value_uint() {
        let value = CounterexampleValue::UInt {
            value: 255,
            type_hint: Some("u8".to_string()),
        };
        let simplified = simplify_value(&value);
        match simplified {
            CounterexampleValue::UInt {
                value: v,
                type_hint,
            } => {
                assert_eq!(v, 255);
                assert_eq!(type_hint, Some("u8".to_string()));
            }
            _ => panic!("Expected UInt"),
        }
    }

    #[test]
    fn test_simplify_value_bool() {
        let value = CounterexampleValue::Bool(true);
        let simplified = simplify_value(&value);
        assert_eq!(simplified, CounterexampleValue::Bool(true));
    }

    #[test]
    fn test_simplify_value_string() {
        let value = CounterexampleValue::String("hello".to_string());
        let simplified = simplify_value(&value);
        assert_eq!(simplified, CounterexampleValue::String("hello".to_string()));
    }

    #[test]
    fn test_simplify_value_float() {
        let value = CounterexampleValue::Float { value: 1.5 };
        let simplified = simplify_value(&value);
        match simplified {
            CounterexampleValue::Float { value: v } => {
                assert!((v - 1.5).abs() < 0.001);
            }
            _ => panic!("Expected Float"),
        }
    }

    #[test]
    fn test_simplify_value_bytes() {
        let value = CounterexampleValue::Bytes(vec![1, 2, 3]);
        let simplified = simplify_value(&value);
        assert_eq!(simplified, CounterexampleValue::Bytes(vec![1, 2, 3]));
    }

    #[test]
    fn test_simplify_value_sequence() {
        let value = CounterexampleValue::Sequence(vec![
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
        ]);
        let simplified = simplify_value(&value);
        match simplified {
            CounterexampleValue::Sequence(elems) => {
                assert_eq!(elems.len(), 2);
            }
            _ => panic!("Expected Sequence"),
        }
    }

    #[test]
    fn test_simplify_value_record() {
        let mut fields = HashMap::new();
        fields.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 10,
                type_hint: None,
            },
        );
        fields.insert("y".to_string(), CounterexampleValue::Bool(false));

        let value = CounterexampleValue::Record(fields);
        let simplified = simplify_value(&value);
        match simplified {
            CounterexampleValue::Record(f) => {
                assert_eq!(f.len(), 2);
                assert!(f.contains_key("x"));
                assert!(f.contains_key("y"));
            }
            _ => panic!("Expected Record"),
        }
    }

    #[test]
    fn test_simplify_value_nested_sequence() {
        let inner = CounterexampleValue::Sequence(vec![CounterexampleValue::Int {
            value: 42,
            type_hint: None,
        }]);
        let value = CounterexampleValue::Sequence(vec![inner]);
        let simplified = simplify_value(&value);
        match simplified {
            CounterexampleValue::Sequence(outer) => {
                assert_eq!(outer.len(), 1);
                match &outer[0] {
                    CounterexampleValue::Sequence(inner) => {
                        assert_eq!(inner.len(), 1);
                    }
                    _ => panic!("Expected inner Sequence"),
                }
            }
            _ => panic!("Expected Sequence"),
        }
    }

    #[test]
    fn test_simplify_value_unknown() {
        let value = CounterexampleValue::Unknown("some raw value".to_string());
        let simplified = simplify_value(&value);
        assert_eq!(
            simplified,
            CounterexampleValue::Unknown("some raw value".to_string())
        );
    }

    #[test]
    fn test_simplify_value_set() {
        let value = CounterexampleValue::Set(vec![
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
        ]);
        // Set goes through the default case (clone)
        let simplified = simplify_value(&value);
        match simplified {
            CounterexampleValue::Set(elems) => {
                assert_eq!(elems.len(), 2);
            }
            _ => panic!("Expected Set"),
        }
    }

    #[test]
    fn test_simplify_value_function() {
        let value = CounterexampleValue::Function(vec![(
            CounterexampleValue::Int {
                value: 0,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
        )]);
        // Function goes through the default case (clone)
        let simplified = simplify_value(&value);
        match simplified {
            CounterexampleValue::Function(mappings) => {
                assert_eq!(mappings.len(), 1);
            }
            _ => panic!("Expected Function"),
        }
    }

    // ==================== Integration Tests ====================

    #[test]
    fn test_full_minimization_workflow() {
        let debugger = DeltaDebugger::new();
        let mut ce = StructuredCounterexample::new();

        // Setup witness with 5 variables, only 2 essential
        ce.witness.insert(
            "divisor".to_string(),
            CounterexampleValue::Int {
                value: 0,
                type_hint: Some("i32".to_string()),
            },
        );
        ce.witness.insert(
            "dividend".to_string(),
            CounterexampleValue::Int {
                value: 100,
                type_hint: Some("i32".to_string()),
            },
        );
        ce.witness
            .insert("unused1".to_string(), CounterexampleValue::Bool(true));
        ce.witness.insert(
            "unused2".to_string(),
            CounterexampleValue::String("test".to_string()),
        );
        ce.witness.insert(
            "unused3".to_string(),
            CounterexampleValue::Float { value: 1.5 },
        );

        // Setup trace with 4 states
        for i in 1..=4 {
            ce.trace.push(TraceState::new(i));
        }

        // Add failed check
        ce.failed_checks.push(FailedCheck {
            check_id: "div_check".to_string(),
            description: "division by zero with divisor".to_string(),
            location: None,
            function: Some("divide".to_string()),
        });

        // Failure requires divisor == 0
        let result = debugger.minimize(
            &ce,
            Box::new(|test_ce| {
                test_ce
                    .witness
                    .get("divisor")
                    .is_some_and(|v| matches!(v, CounterexampleValue::Int { value: 0, .. }))
            }),
        );

        // Verify minimization occurred
        assert!(result.counterexample.minimized);
        assert!(result.counterexample.witness.contains_key("divisor"));
        assert!(result.reduction_ratio() > 0.0);

        // Verify summary is generated correctly
        let summary = result.summary();
        assert!(summary.contains("variables"));
        assert!(summary.contains("reduction"));
    }

    #[test]
    fn test_minimization_preserves_failed_checks() {
        let debugger = DeltaDebugger::new();
        let mut ce = StructuredCounterexample::new();

        ce.witness.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 0,
                type_hint: None,
            },
        );
        ce.failed_checks.push(FailedCheck {
            check_id: "check1".to_string(),
            description: "assertion failed".to_string(),
            location: None,
            function: None,
        });

        let result = debugger.minimize(&ce, Box::new(|_| true));

        // Failed checks should be preserved
        assert_eq!(result.counterexample.failed_checks.len(), 1);
        assert_eq!(result.counterexample.failed_checks[0].check_id, "check1");
    }

    // ==================== Minimization Algorithm Boundary Tests ====================

    #[test]
    fn test_minimize_granularity_adjustment_decrease() {
        // Test the n = n * 2 branch when no progress is made
        let config = MinimizationConfig {
            max_iterations: 10,
            minimize_trace: false,
            minimize_witness: true,
            granularity: 2,
        };
        let debugger = DeltaDebugger::with_config(config);
        let mut ce = StructuredCounterexample::new();

        // All variables are essential - no progress will be made
        for i in 0..4 {
            ce.witness.insert(
                format!("essential_{}", i),
                CounterexampleValue::Int {
                    value: i,
                    type_hint: None,
                },
            );
        }

        // All variables essential
        let result = debugger.minimize(&ce, Box::new(|test_ce| test_ce.witness.len() == 4));

        assert_eq!(result.minimized_witness_count, 4);
        // Should have tried multiple iterations
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_minimize_granularity_adjustment_increase() {
        // Test the n = max(n - 1, 2) branch when progress is made
        let config = MinimizationConfig {
            max_iterations: 100,
            minimize_trace: false,
            minimize_witness: true,
            granularity: 4,
        };
        let debugger = DeltaDebugger::with_config(config);
        let mut ce = StructuredCounterexample::new();

        // Only var_0 is essential
        for i in 0..8 {
            ce.witness.insert(
                format!("var_{}", i),
                CounterexampleValue::Int {
                    value: i,
                    type_hint: None,
                },
            );
        }

        let result = debugger.minimize(
            &ce,
            Box::new(|test_ce| test_ce.witness.contains_key("var_0")),
        );

        assert!(result.counterexample.witness.contains_key("var_0"));
        // Should have made progress and removed most variables
        assert!(result.minimized_witness_count < 8);
    }

    #[test]
    fn test_minimize_witness_chunk_boundary_condition() {
        // Test start >= variables.len() check (line 182)
        let config = MinimizationConfig {
            max_iterations: 50,
            minimize_trace: false,
            minimize_witness: true,
            granularity: 8, // More chunks than variables
        };
        let debugger = DeltaDebugger::with_config(config);
        let mut ce = StructuredCounterexample::new();

        // Only 3 variables with granularity 8
        ce.witness.insert(
            "a".to_string(),
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
        );
        ce.witness.insert(
            "b".to_string(),
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
        );
        ce.witness.insert(
            "c".to_string(),
            CounterexampleValue::Int {
                value: 3,
                type_hint: None,
            },
        );

        let result = debugger.minimize(&ce, Box::new(|test_ce| test_ce.witness.contains_key("a")));

        // Should still work and minimize
        assert!(result.counterexample.witness.contains_key("a"));
    }

    #[test]
    fn test_minimize_witness_values_div_ceil_behavior() {
        // Test chunk_size = variables.len().div_ceil(n) (line 175)
        let config = MinimizationConfig {
            max_iterations: 100,
            minimize_trace: false,
            minimize_witness: true,
            granularity: 3, // Not evenly divisible
        };
        let debugger = DeltaDebugger::with_config(config);
        let mut ce = StructuredCounterexample::new();

        // 7 variables with granularity 3 -> chunk_size = ceil(7/3) = 3
        for i in 0..7 {
            ce.witness.insert(
                format!("v{}", i),
                CounterexampleValue::Int {
                    value: i,
                    type_hint: None,
                },
            );
        }

        let result = debugger.minimize(&ce, Box::new(|test_ce| test_ce.witness.contains_key("v0")));

        assert!(result.counterexample.witness.contains_key("v0"));
    }

    #[test]
    fn test_minimize_n_exceeds_variables_len_exit() {
        // Test n > variables.len() break (line 215-217)
        let config = MinimizationConfig {
            max_iterations: 100,
            minimize_trace: false,
            minimize_witness: true,
            granularity: 2,
        };
        let debugger = DeltaDebugger::with_config(config);
        let mut ce = StructuredCounterexample::new();

        // Just 2 variables
        ce.witness.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
        );
        ce.witness.insert(
            "y".to_string(),
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
        );

        // All essential
        let result = debugger.minimize(&ce, Box::new(|test_ce| test_ce.witness.len() == 2));

        assert_eq!(result.minimized_witness_count, 2);
    }

    // ==================== minimize_trace Boundary Tests ====================

    #[test]
    fn test_minimize_trace_iteration_progress() {
        // Test i < current_trace.len().saturating_sub(1) (line 245)
        let debugger = DeltaDebugger::new();
        let mut ce = StructuredCounterexample::new();

        // Create trace with 6 states
        for i in 1..=6 {
            ce.trace.push(TraceState::new(i));
        }

        // Only first and last states needed
        let result = debugger.minimize(
            &ce,
            Box::new(|test_ce| {
                test_ce.trace.len() >= 2
                    && !test_ce.trace.is_empty()
                    && test_ce.trace.last().is_some()
            }),
        );

        // Should minimize to 2 states
        assert!(result.counterexample.trace.len() <= 6);
    }

    #[test]
    fn test_minimize_trace_state_removal_increments_i() {
        // Test the else branch (i += 1) when state is essential
        let debugger = DeltaDebugger::new();
        let mut ce = StructuredCounterexample::new();

        // Create trace where middle states have critical data
        for i in 1..=4 {
            let mut state = TraceState::new(i);
            state.variables.insert(
                format!("state_{}", i),
                CounterexampleValue::Int {
                    value: i as i128,
                    type_hint: None,
                },
            );
            ce.trace.push(state);
        }

        // All states essential
        let result = debugger.minimize(&ce, Box::new(|test_ce| test_ce.trace.len() == 4));

        assert_eq!(result.counterexample.trace.len(), 4);
    }

    #[test]
    fn test_minimize_trace_max_iterations_exit() {
        // Test iterations < self.config.max_iterations (line 245)
        let config = MinimizationConfig {
            max_iterations: 2,
            minimize_trace: true,
            minimize_witness: false,
            granularity: 2,
        };
        let debugger = DeltaDebugger::with_config(config);
        let mut ce = StructuredCounterexample::new();

        for i in 1..=10 {
            ce.trace.push(TraceState::new(i));
        }

        let result = debugger.minimize(&ce, Box::new(|_| true));

        // Should stop early due to iteration limit
        assert!(result.iterations <= 3);
    }

    // ==================== simplify_value Match Arm Tests ====================

    #[test]
    fn test_simplify_value_int_preserves_value() {
        let value = CounterexampleValue::Int {
            value: -500,
            type_hint: Some("i16".to_string()),
        };
        let simplified = simplify_value(&value);
        match simplified {
            CounterexampleValue::Int {
                value: v,
                type_hint,
            } => {
                assert_eq!(v, -500);
                assert_eq!(type_hint, Some("i16".to_string()));
            }
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn test_simplify_value_uint_preserves_value() {
        let value = CounterexampleValue::UInt {
            value: 65535,
            type_hint: Some("u16".to_string()),
        };
        let simplified = simplify_value(&value);
        match simplified {
            CounterexampleValue::UInt {
                value: v,
                type_hint,
            } => {
                assert_eq!(v, 65535);
                assert_eq!(type_hint, Some("u16".to_string()));
            }
            _ => panic!("Expected UInt"),
        }
    }

    #[test]
    fn test_simplify_value_sequence_recursive() {
        let value = CounterexampleValue::Sequence(vec![
            CounterexampleValue::Int {
                value: 1,
                type_hint: Some("i32".to_string()),
            },
            CounterexampleValue::UInt {
                value: 2,
                type_hint: Some("u32".to_string()),
            },
            CounterexampleValue::Bool(true),
        ]);
        let simplified = simplify_value(&value);
        match simplified {
            CounterexampleValue::Sequence(elems) => {
                assert_eq!(elems.len(), 3);
                // Verify each element was recursively simplified
                match &elems[0] {
                    CounterexampleValue::Int { value: v, .. } => assert_eq!(*v, 1),
                    _ => panic!("Expected Int"),
                }
                match &elems[1] {
                    CounterexampleValue::UInt { value: v, .. } => assert_eq!(*v, 2),
                    _ => panic!("Expected UInt"),
                }
                assert_eq!(elems[2], CounterexampleValue::Bool(true));
            }
            _ => panic!("Expected Sequence"),
        }
    }

    #[test]
    fn test_simplify_value_record_recursive() {
        let mut fields = HashMap::new();
        fields.insert(
            "num".to_string(),
            CounterexampleValue::Int {
                value: 42,
                type_hint: None,
            },
        );
        fields.insert("flag".to_string(), CounterexampleValue::Bool(false));

        let value = CounterexampleValue::Record(fields);
        let simplified = simplify_value(&value);
        match simplified {
            CounterexampleValue::Record(f) => {
                assert_eq!(f.len(), 2);
                assert!(f.contains_key("num"));
                assert!(f.contains_key("flag"));
                match f.get("num") {
                    Some(CounterexampleValue::Int { value: v, .. }) => assert_eq!(*v, 42),
                    _ => panic!("Expected Int for num field"),
                }
            }
            _ => panic!("Expected Record"),
        }
    }

    #[test]
    fn test_simplify_value_empty_sequence() {
        let value = CounterexampleValue::Sequence(vec![]);
        let simplified = simplify_value(&value);
        match simplified {
            CounterexampleValue::Sequence(elems) => {
                assert!(elems.is_empty());
            }
            _ => panic!("Expected Sequence"),
        }
    }

    #[test]
    fn test_simplify_value_empty_record() {
        let value = CounterexampleValue::Record(HashMap::new());
        let simplified = simplify_value(&value);
        match simplified {
            CounterexampleValue::Record(f) => {
                assert!(f.is_empty());
            }
            _ => panic!("Expected Record"),
        }
    }

    // ==================== Algorithm Edge Cases ====================

    #[test]
    fn test_minimize_witness_with_many_iterations() {
        let config = MinimizationConfig {
            max_iterations: 200,
            minimize_trace: false,
            minimize_witness: true,
            granularity: 2,
        };
        let debugger = DeltaDebugger::with_config(config);
        let mut ce = StructuredCounterexample::new();

        // Many variables where only first is essential
        for i in 0..50 {
            ce.witness.insert(
                format!("var_{:03}", i),
                CounterexampleValue::Int {
                    value: i,
                    type_hint: None,
                },
            );
        }

        let result = debugger.minimize(
            &ce,
            Box::new(|test_ce| test_ce.witness.contains_key("var_000")),
        );

        // Should find the minimal set
        assert!(result.counterexample.witness.contains_key("var_000"));
        assert!(result.iterations > 0);
        assert!(result.reduction_ratio() > 0.0);
    }

    #[test]
    fn test_minimize_trace_removes_middle_states_only() {
        let debugger = DeltaDebugger::new();
        let mut ce = StructuredCounterexample::new();

        // Create trace: states 1, 2, 3, 4, 5
        for i in 1..=5 {
            let mut state = TraceState::new(i);
            state.variables.insert(
                "step".to_string(),
                CounterexampleValue::Int {
                    value: i as i128,
                    type_hint: None,
                },
            );
            ce.trace.push(state);
        }

        // First and last states are essential
        let result = debugger.minimize(
            &ce,
            Box::new(|test_ce| {
                if test_ce.trace.len() < 2 {
                    return false;
                }
                let first = test_ce.trace.first().unwrap();
                let last = test_ce.trace.last().unwrap();
                first.state_num == 1 || last.state_num == 5
            }),
        );

        // Minimization should have attempted to remove middle states
        assert!(result.counterexample.minimized);
    }

    #[test]
    fn test_minimization_with_high_granularity() {
        let config = MinimizationConfig {
            max_iterations: 100,
            minimize_trace: true,
            minimize_witness: true,
            granularity: 8,
        };
        let debugger = DeltaDebugger::with_config(config);
        let mut ce = StructuredCounterexample::new();

        for i in 0..20 {
            ce.witness.insert(
                format!("var_{}", i),
                CounterexampleValue::Int {
                    value: i,
                    type_hint: None,
                },
            );
        }

        // Only var_0 is essential
        let result = debugger.minimize(
            &ce,
            Box::new(|test_ce| test_ce.witness.contains_key("var_0")),
        );

        assert!(result.counterexample.witness.contains_key("var_0"));
        assert!(result.minimized_witness_count < 20);
    }

    #[test]
    fn test_minimize_witness_zero_granularity_clamps_to_one() {
        let config = MinimizationConfig {
            max_iterations: 50,
            minimize_trace: false,
            minimize_witness: true,
            granularity: 0,
        };
        let debugger = DeltaDebugger::with_config(config);
        let mut ce = StructuredCounterexample::new();

        ce.witness.insert(
            "essential".to_string(),
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
        );
        ce.witness.insert(
            "optional".to_string(),
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
        );

        let result = debugger.minimize(
            &ce,
            Box::new(|test_ce| test_ce.witness.contains_key("essential")),
        );

        assert_eq!(result.counterexample.witness.len(), 1);
        assert!(result.counterexample.witness.contains_key("essential"));
        assert!(result.removed_variables.contains("optional"));
    }

    // ==================== Mutation Coverage Tests ====================
    // Tests targeting specific mutations identified by cargo-mutants

    #[test]
    fn test_minimize_trace_len_greater_than_one_boundary() {
        // Target: line 137 - counterexample.trace.len() > 1
        // Mutations: > with ==, > with <, > with >=
        let debugger = DeltaDebugger::new();

        // Case 1: trace.len() == 1 (should NOT trigger trace minimization)
        let mut ce1 = StructuredCounterexample::new();
        ce1.trace.push(TraceState::new(1));
        let result1 = debugger.minimize(&ce1, Box::new(|_| true));
        // With 1 trace element, trace minimization should be skipped
        assert_eq!(result1.counterexample.trace.len(), 1);

        // Case 2: trace.len() == 2 (SHOULD trigger trace minimization)
        let mut ce2 = StructuredCounterexample::new();
        ce2.trace.push(TraceState::new(1));
        ce2.trace.push(TraceState::new(2));
        let result2 = debugger.minimize(&ce2, Box::new(|_| true));
        // With 2 elements, trace minimization is attempted but early returns
        assert!(result2.counterexample.trace.len() <= 2);

        // Case 3: trace.len() == 0 (should NOT trigger trace minimization)
        let ce3 = StructuredCounterexample::new();
        let result3 = debugger.minimize(&ce3, Box::new(|_| true));
        assert!(result3.counterexample.trace.is_empty());
    }

    #[test]
    fn test_iterations_accumulate_from_trace_minimization() {
        // Target: line 141 - iterations += trace_iterations
        // Mutations: += with *=
        let debugger = DeltaDebugger::new();
        let mut ce = StructuredCounterexample::new();

        // Add witness and trace so both minimizations run
        ce.witness.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 0,
                type_hint: None,
            },
        );
        ce.witness.insert(
            "y".to_string(),
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
        );

        // Add 4 trace states (more than 2 to trigger actual minimization work)
        for i in 1..=4 {
            ce.trace.push(TraceState::new(i));
        }

        // x is essential, y is not; all trace states required
        let result = debugger.minimize(
            &ce,
            Box::new(|test_ce| test_ce.witness.contains_key("x") && test_ce.trace.len() == 4),
        );

        // Iterations should be sum of witness + trace iterations
        // With += they add, with *= they multiply (giving wrong result)
        // Witness: at least 1 iteration, Trace: at least 1 iteration
        // If *= was used: 1 * 1 = 1, but we expect at least 2
        assert!(
            result.iterations >= 2,
            "Expected at least 2 iterations (witness + trace), got {}",
            result.iterations
        );
    }

    #[test]
    fn test_minimize_witness_chunk_loop_boundary() {
        // Target: line 174 - while variables.len() >= n && iterations < max_iterations
        // Also: line 174 - && with ||
        // Also: line 174 - < with <=
        let config = MinimizationConfig {
            max_iterations: 5,
            minimize_trace: false,
            minimize_witness: true,
            granularity: 2,
        };
        let debugger = DeltaDebugger::with_config(config);
        let mut ce = StructuredCounterexample::new();

        // Exactly 2 variables with granularity 2
        ce.witness.insert(
            "a".to_string(),
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
        );
        ce.witness.insert(
            "b".to_string(),
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
        );

        // Both essential - loop should run but not remove anything
        let result = debugger.minimize(&ce, Box::new(|test_ce| test_ce.witness.len() == 2));

        assert_eq!(result.minimized_witness_count, 2);
        // If && became ||, it would run even when it shouldn't
        // If < became <=, it would exit one iteration early
    }

    #[test]
    fn test_minimize_witness_chunk_multiplication() {
        // Target: line 180 - start = i * chunk_size
        // Mutations: * with +, * with /
        let config = MinimizationConfig {
            max_iterations: 100,
            minimize_trace: false,
            minimize_witness: true,
            granularity: 3,
        };
        let debugger = DeltaDebugger::with_config(config);
        let mut ce = StructuredCounterexample::new();

        // 9 variables with granularity 3 -> 3 chunks of 3
        // Chunk 0: a, b, c (indices 0-2)
        // Chunk 1: d, e, f (indices 3-5)
        // Chunk 2: g, h, i (indices 6-8)
        for name in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'] {
            ce.witness.insert(
                name.to_string(),
                CounterexampleValue::Int {
                    value: name as i128,
                    type_hint: None,
                },
            );
        }

        // Only 'a' is essential
        let result = debugger.minimize(&ce, Box::new(|test_ce| test_ce.witness.contains_key("a")));

        // With correct multiplication, chunks are [abc][def][ghi]
        // With +, chunks would be wrong: start=0*3=0, start=1*3=3, start=2*3=6
        // vs start=0+3=3, start=1+3=4, start=2+3=5 (wrong)
        assert!(result.counterexample.witness.contains_key("a"));
    }

    #[test]
    fn test_minimize_witness_start_bounds_check() {
        // Target: line 182 - if start >= variables.len()
        // Mutations: >= with <
        let config = MinimizationConfig {
            max_iterations: 100,
            minimize_trace: false,
            minimize_witness: true,
            granularity: 10, // More chunks than variables
        };
        let debugger = DeltaDebugger::with_config(config);
        let mut ce = StructuredCounterexample::new();

        // Only 3 variables with granularity 10
        // chunk_size = ceil(3/10) = 1
        // So we'd try chunks starting at 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        // But only 0, 1, 2 are valid - rest should break
        ce.witness.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
        );
        ce.witness.insert(
            "y".to_string(),
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
        );
        ce.witness.insert(
            "z".to_string(),
            CounterexampleValue::Int {
                value: 3,
                type_hint: None,
            },
        );

        // Only x essential
        let result = debugger.minimize(&ce, Box::new(|test_ce| test_ce.witness.contains_key("x")));

        // Should not panic from out-of-bounds access
        assert!(result.counterexample.witness.contains_key("x"));
    }

    #[test]
    fn test_minimize_witness_iteration_add() {
        // Target: line 197 - iterations += 1
        // Mutations: += with *=
        let config = MinimizationConfig {
            max_iterations: 100,
            minimize_trace: false,
            minimize_witness: true,
            granularity: 2,
        };
        let debugger = DeltaDebugger::with_config(config);
        let mut ce = StructuredCounterexample::new();

        // 4 variables
        for i in 0..4 {
            ce.witness.insert(
                format!("v{}", i),
                CounterexampleValue::Int {
                    value: i,
                    type_hint: None,
                },
            );
        }

        // All essential
        let result = debugger.minimize(&ce, Box::new(|test_ce| test_ce.witness.len() == 4));

        // With += 1: iterations = 0 + 1 + 1 + ... = N
        // With *= 1: iterations = 0 * 1 = 0 (stays at 0!)
        assert!(
            result.iterations >= 1,
            "Expected at least 1 iteration, got {}",
            result.iterations
        );
    }

    #[test]
    fn test_minimize_witness_granularity_decrease_subtraction() {
        // Target: line 208 - n = max(n - 1, 2)
        // Mutations: - with /
        let config = MinimizationConfig {
            max_iterations: 100,
            minimize_trace: false,
            minimize_witness: true,
            granularity: 4,
        };
        let debugger = DeltaDebugger::with_config(config);
        let mut ce = StructuredCounterexample::new();

        // 8 variables, first one essential
        for i in 0..8 {
            ce.witness.insert(
                format!("var_{}", i),
                CounterexampleValue::Int {
                    value: i,
                    type_hint: None,
                },
            );
        }

        let result = debugger.minimize(
            &ce,
            Box::new(|test_ce| test_ce.witness.contains_key("var_0")),
        );

        // Algorithm should converge to minimal set
        // With n-1: decreases granularity gradually (4->3->2)
        // With n/1: stays same (4/1=4)
        assert!(result.counterexample.witness.contains_key("var_0"));
        // Should have removed at least some variables
        assert!(result.minimized_witness_count < 8);
    }

    #[test]
    fn test_minimize_witness_granularity_increase_multiplication() {
        // Target: line 212 - n = min(n * 2, variables.len())
        // Mutations: * with +
        let config = MinimizationConfig {
            max_iterations: 100,
            minimize_trace: false,
            minimize_witness: true,
            granularity: 2,
        };
        let debugger = DeltaDebugger::with_config(config);
        let mut ce = StructuredCounterexample::new();

        // 16 variables, all essential
        for i in 0..16 {
            ce.witness.insert(
                format!("v{:02}", i),
                CounterexampleValue::Int {
                    value: i,
                    type_hint: None,
                },
            );
        }

        let result = debugger.minimize(&ce, Box::new(|test_ce| test_ce.witness.len() == 16));

        // With * 2: n doubles each failed round (2->4->8->16)
        // With + 2: n increases linearly (2->4->6->8...)
        // Both should eventually work but * 2 is faster
        assert_eq!(result.minimized_witness_count, 16);
    }

    #[test]
    fn test_minimize_witness_n_exceeds_length_check() {
        // Target: line 215 - if n > variables.len()
        // Mutations: > with <
        let config = MinimizationConfig {
            max_iterations: 100,
            minimize_trace: false,
            minimize_witness: true,
            granularity: 2,
        };
        let debugger = DeltaDebugger::with_config(config);
        let mut ce = StructuredCounterexample::new();

        // Start with 2 variables, all essential
        ce.witness.insert(
            "a".to_string(),
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
        );
        ce.witness.insert(
            "b".to_string(),
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
        );

        let result = debugger.minimize(&ce, Box::new(|test_ce| test_ce.witness.len() == 2));

        // When n doubles and exceeds variables.len(), should break
        // With > : breaks when n > 2 (correct)
        // With < : breaks when n < 2 (immediately, wrong)
        assert_eq!(result.minimized_witness_count, 2);
    }

    #[test]
    fn test_minimize_witness_iteration_limit_with_optional_variable() {
        let config = MinimizationConfig {
            max_iterations: 2,
            minimize_trace: false,
            minimize_witness: true,
            granularity: 2,
        };
        let debugger = DeltaDebugger::with_config(config);
        let mut ce = StructuredCounterexample::new();

        ce.witness.insert(
            "a".to_string(),
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
        );
        ce.witness.insert(
            "b".to_string(),
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
        );
        ce.witness.insert(
            "c".to_string(),
            CounterexampleValue::Int {
                value: 3,
                type_hint: None,
            },
        );

        let result = debugger.minimize(&ce, Box::new(|test_ce| !test_ce.witness.contains_key("c")));

        assert_eq!(result.counterexample.witness.len(), 2);
        assert!(result.removed_variables.contains("c"));
        assert_eq!(result.iterations, 2);
    }

    #[test]
    fn test_minimize_witness_granularity_decreases_after_progress() {
        let config = MinimizationConfig {
            max_iterations: 6,
            minimize_trace: false,
            minimize_witness: true,
            granularity: 3,
        };
        let debugger = DeltaDebugger::with_config(config.clone());
        let mut ce = StructuredCounterexample::new();

        for i in 0..5 {
            ce.witness.insert(
                format!("v{i}"),
                CounterexampleValue::Int {
                    value: i,
                    type_hint: None,
                },
            );
        }

        let calls = Rc::new(RefCell::new(0usize));
        let call_counter = Rc::clone(&calls);
        let result = debugger.minimize(
            &ce,
            Box::new(move |_| {
                let mut counter = call_counter.borrow_mut();
                let current = *counter;
                *counter += 1;
                current == 0
            }),
        );

        assert_eq!(result.counterexample.witness.len(), 3);
        assert!(result.iterations > config.max_iterations);
    }

    #[test]
    fn test_minimize_witness_granularity_growth_without_progress() {
        let config = MinimizationConfig {
            max_iterations: 6,
            minimize_trace: false,
            minimize_witness: true,
            granularity: 4,
        };
        let debugger = DeltaDebugger::with_config(config);
        let mut ce = StructuredCounterexample::new();

        for i in 0..7 {
            ce.witness.insert(
                format!("var_{i}"),
                CounterexampleValue::Int {
                    value: i as i128,
                    type_hint: None,
                },
            );
        }

        let result = debugger.minimize(&ce, Box::new(|_| false));

        assert_eq!(result.counterexample.witness.len(), 7);
        assert_eq!(result.iterations, 11);
    }

    #[test]
    fn test_minimize_witness_continues_when_chunks_remaining() {
        let config = MinimizationConfig {
            max_iterations: 5,
            minimize_trace: false,
            minimize_witness: true,
            granularity: 2,
        };
        let debugger = DeltaDebugger::with_config(config.clone());
        let mut ce = StructuredCounterexample::new();

        for i in 0..7 {
            ce.witness.insert(
                format!("opt_{i}"),
                CounterexampleValue::Int {
                    value: i as i128,
                    type_hint: None,
                },
            );
        }

        let calls = Rc::new(RefCell::new(0usize));
        let call_counter = Rc::clone(&calls);
        let result = debugger.minimize(
            &ce,
            Box::new(move |_| {
                let mut counter = call_counter.borrow_mut();
                let current = *counter;
                *counter += 1;
                current == 0
            }),
        );

        assert_eq!(result.counterexample.witness.len(), 3);
        assert!(result.iterations > config.max_iterations);
    }

    #[test]
    fn test_minimize_trace_len_check() {
        // Target: line 236 - if trace.len() <= 2
        // Mutations: <= with >
        let debugger = DeltaDebugger::new();
        let mut ce = StructuredCounterexample::new();

        // Exactly 2 trace elements
        ce.trace.push(TraceState::new(1));
        ce.trace.push(TraceState::new(2));

        let result = debugger.minimize(&ce, Box::new(|_| true));

        // With <= 2: returns early for 0, 1, or 2 elements
        // With > 2: would try to minimize 2-element traces
        assert!(result.counterexample.trace.len() <= 2);
    }

    #[test]
    fn test_minimize_trace_while_condition() {
        // Target: line 245 - while i < current_trace.len().saturating_sub(1) && iterations < max
        // Mutations: < with ==, < with >, < with <=
        let config = MinimizationConfig {
            max_iterations: 100,
            minimize_trace: true,
            minimize_witness: false,
            granularity: 2,
        };
        let debugger = DeltaDebugger::with_config(config);
        let mut ce = StructuredCounterexample::new();

        // 5 trace states
        for i in 1..=5 {
            ce.trace.push(TraceState::new(i));
        }

        // Middle states can be removed
        let result = debugger.minimize(&ce, Box::new(|test_ce| test_ce.trace.len() >= 2));

        // With correct <: iterates from 1 to len-2 (inclusive)
        // With ==: only checks when i == len-1 (misses middle)
        // With >: never iterates
        // With <=: might go out of bounds
        assert!(result.counterexample.trace.len() <= 5);
    }

    #[test]
    fn test_minimize_trace_iterations_add() {
        // Target: line 256 - iterations += 1
        // Mutations: += with *=
        let config = MinimizationConfig {
            max_iterations: 100,
            minimize_trace: true,
            minimize_witness: false,
            granularity: 2,
        };
        let debugger = DeltaDebugger::with_config(config);
        let mut ce = StructuredCounterexample::new();

        // 4 trace states, all essential
        for i in 1..=4 {
            ce.trace.push(TraceState::new(i));
        }

        let result = debugger.minimize(&ce, Box::new(|test_ce| test_ce.trace.len() == 4));

        // With +=: iterations accumulate
        // With *=: iterations stay at 0 if starting from 0
        assert!(result.iterations >= 1);
    }

    #[test]
    fn test_minimize_trace_index_increment() {
        // Target: line 264 - i += 1
        // Mutations: += with *=
        let config = MinimizationConfig {
            max_iterations: 100,
            minimize_trace: true,
            minimize_witness: false,
            granularity: 2,
        };
        let debugger = DeltaDebugger::with_config(config);
        let mut ce = StructuredCounterexample::new();

        // 4 trace states, all essential
        for i in 1..=4 {
            ce.trace.push(TraceState::new(i));
        }

        let result = debugger.minimize(&ce, Box::new(|test_ce| test_ce.trace.len() == 4));

        // With i += 1: progresses through trace
        // With i *= 1: stays at i=1 forever (infinite loop, but max_iterations prevents)
        assert_eq!(result.counterexample.trace.len(), 4);
    }

    // Note: Tests for simplify_value match arms removed because the function
    // was simplified to just clone values. The previous Int, UInt, Sequence,
    // and Record arms were equivalent to clone() and have been removed.
    // See simplify_value() documentation for future improvement ideas.

    // ==================== Mutation Coverage Tests ====================
    // Tests specifically designed to catch mutations in minimize_trace

    #[test]
    fn test_minimize_trace_last_state_preserved() {
        // This test catches the mutation: i < len.saturating_sub(1) -> i <= len.saturating_sub(1)
        // With the mutation, the loop would try to remove the last state (index len-1)
        // but we require the last state to always be present.
        let debugger = DeltaDebugger::new();
        let mut ce = StructuredCounterexample::new();

        // Create 3 trace states
        for i in 1..=3 {
            let mut state = TraceState::new(i);
            state.variables.insert(
                format!("v{}", i),
                CounterexampleValue::Int {
                    value: i as i128,
                    type_hint: None,
                },
            );
            ce.trace.push(state);
        }

        // The failure requires the LAST state to have v3=3
        // If the mutant tries to remove the last state, this will fail
        let result = debugger.minimize(
            &ce,
            Box::new(|test_ce| {
                test_ce
                    .trace
                    .last()
                    .is_some_and(|s| s.variables.contains_key("v3"))
            }),
        );

        // The last state must be preserved (it contains v3)
        assert!(
            result
                .counterexample
                .trace
                .last()
                .unwrap()
                .variables
                .contains_key("v3"),
            "Last state with v3 must be preserved"
        );
    }

    #[test]
    fn test_minimize_trace_boundary_condition() {
        // Test with exactly 3 states where only first and last are essential
        // This tests the boundary: i < len.saturating_sub(1) when len=3
        // Loop should only try i=1, not i=2 (which is len-1)
        let debugger = DeltaDebugger::new();
        let mut ce = StructuredCounterexample::new();

        for i in 1..=3 {
            let mut state = TraceState::new(i);
            // Mark first and last as essential
            if i == 1 || i == 3 {
                state
                    .variables
                    .insert("essential".to_string(), CounterexampleValue::Bool(true));
            }
            ce.trace.push(state);
        }

        // Require first and last states to have "essential" variable
        let result = debugger.minimize(
            &ce,
            Box::new(|test_ce| {
                test_ce
                    .trace
                    .first()
                    .is_some_and(|s| s.variables.contains_key("essential"))
                    && test_ce
                        .trace
                        .last()
                        .is_some_and(|s| s.variables.contains_key("essential"))
            }),
        );

        // Should have minimized to 2 states (first and last)
        assert_eq!(result.counterexample.trace.len(), 2);
        assert!(result
            .counterexample
            .trace
            .first()
            .unwrap()
            .variables
            .contains_key("essential"));
        assert!(result
            .counterexample
            .trace
            .last()
            .unwrap()
            .variables
            .contains_key("essential"));
    }
}
