//! Invariant compilation and checking
//!
//! This module provides functionality to define, compile, and check
//! invariants on execution traces. It supports both simple predicate-based
//! invariants and more complex temporal invariants.

#![allow(clippy::type_complexity)]

use serde::{Deserialize, Serialize};

use dashprove_async::ExecutionTrace;

/// An invariant that can be checked on traces
#[derive(Clone)]
pub struct Invariant {
    /// Invariant name
    pub name: String,
    /// Description
    pub description: String,
    /// The checker function
    checker: InvariantChecker,
}

/// Types of invariant checkers
#[derive(Clone)]
enum InvariantChecker {
    /// Simple predicate on state
    Predicate(std::sync::Arc<dyn Fn(&serde_json::Value) -> bool + Send + Sync>),
    /// Transition invariant (checks from_state, action, to_state)
    Transition(
        std::sync::Arc<dyn Fn(&serde_json::Value, &str, &serde_json::Value) -> bool + Send + Sync>,
    ),
    /// Implication: if condition, then property
    Implication {
        condition: std::sync::Arc<dyn Fn(&serde_json::Value) -> bool + Send + Sync>,
        property: std::sync::Arc<dyn Fn(&serde_json::Value) -> bool + Send + Sync>,
    },
}

impl Invariant {
    /// Create a simple state invariant
    pub fn state(
        name: impl Into<String>,
        predicate: impl Fn(&serde_json::Value) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            checker: InvariantChecker::Predicate(std::sync::Arc::new(predicate)),
        }
    }

    /// Create a transition invariant
    pub fn transition(
        name: impl Into<String>,
        predicate: impl Fn(&serde_json::Value, &str, &serde_json::Value) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            checker: InvariantChecker::Transition(std::sync::Arc::new(predicate)),
        }
    }

    /// Create an implication invariant (if P then Q)
    pub fn implication(
        name: impl Into<String>,
        condition: impl Fn(&serde_json::Value) -> bool + Send + Sync + 'static,
        property: impl Fn(&serde_json::Value) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            checker: InvariantChecker::Implication {
                condition: std::sync::Arc::new(condition),
                property: std::sync::Arc::new(property),
            },
        }
    }

    /// Set description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Check the invariant on a state
    pub fn check_state(&self, state: &serde_json::Value) -> bool {
        match &self.checker {
            InvariantChecker::Predicate(p) => p(state),
            InvariantChecker::Implication {
                condition,
                property,
            } => {
                // P => Q is equivalent to !P || Q
                !condition(state) || property(state)
            }
            InvariantChecker::Transition(_) => true, // State invariants don't apply to transitions
        }
    }

    /// Check the invariant on a transition
    pub fn check_transition(
        &self,
        from_state: &serde_json::Value,
        action: &str,
        to_state: &serde_json::Value,
    ) -> bool {
        match &self.checker {
            InvariantChecker::Transition(p) => p(from_state, action, to_state),
            _ => self.check_state(to_state), // Non-transition invariants check the to_state
        }
    }
}

/// Result of checking an invariant on a trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvariantCheckResult {
    /// Invariant name
    pub invariant_name: String,
    /// Whether the invariant held throughout the trace
    pub satisfied: bool,
    /// State index where first violation occurred (if any)
    pub first_violation_index: Option<usize>,
    /// Total number of violations
    pub violation_count: usize,
    /// States where violations occurred
    pub violation_states: Vec<serde_json::Value>,
}

impl InvariantCheckResult {
    /// Create a satisfied result
    pub fn satisfied(name: impl Into<String>) -> Self {
        Self {
            invariant_name: name.into(),
            satisfied: true,
            first_violation_index: None,
            violation_count: 0,
            violation_states: vec![],
        }
    }

    /// Create a violated result
    pub fn violated(
        name: impl Into<String>,
        first_index: usize,
        violation_states: Vec<serde_json::Value>,
    ) -> Self {
        Self {
            invariant_name: name.into(),
            satisfied: false,
            first_violation_index: Some(first_index),
            violation_count: violation_states.len(),
            violation_states,
        }
    }
}

/// Check an invariant on an execution trace
pub fn check_invariant(trace: &ExecutionTrace, invariant: &Invariant) -> InvariantCheckResult {
    let mut violations = vec![];
    let mut first_violation_index = None;

    // Check initial state
    if !invariant.check_state(&trace.initial_state) {
        violations.push(trace.initial_state.clone());
        first_violation_index = Some(0);
    }

    // Check each transition
    let mut current_state = &trace.initial_state;
    for (i, transition) in trace.transitions.iter().enumerate() {
        let state_index = i + 1;

        if !invariant.check_transition(current_state, &transition.event, &transition.to_state) {
            violations.push(transition.to_state.clone());
            if first_violation_index.is_none() {
                first_violation_index = Some(state_index);
            }
        }

        current_state = &transition.to_state;
    }

    if violations.is_empty() {
        InvariantCheckResult::satisfied(&invariant.name)
    } else {
        InvariantCheckResult::violated(
            &invariant.name,
            first_violation_index.unwrap_or(0),
            violations,
        )
    }
}

/// Check multiple invariants on a trace
pub fn check_invariants(
    trace: &ExecutionTrace,
    invariants: &[Invariant],
) -> Vec<InvariantCheckResult> {
    invariants
        .iter()
        .map(|inv| check_invariant(trace, inv))
        .collect()
}

/// Trace invariant checker with multiple invariants
pub struct TraceInvariantChecker {
    invariants: Vec<Invariant>,
}

impl TraceInvariantChecker {
    /// Create a new checker
    pub fn new() -> Self {
        Self { invariants: vec![] }
    }

    /// Add an invariant
    pub fn add(&mut self, invariant: Invariant) {
        self.invariants.push(invariant);
    }

    /// Add a simple state invariant
    pub fn add_state(
        &mut self,
        name: impl Into<String>,
        predicate: impl Fn(&serde_json::Value) -> bool + Send + Sync + 'static,
    ) {
        self.add(Invariant::state(name, predicate));
    }

    /// Check all invariants on a trace
    pub fn check(&self, trace: &ExecutionTrace) -> TraceCheckResult {
        let results = check_invariants(trace, &self.invariants);
        let all_satisfied = results.iter().all(|r| r.satisfied);

        TraceCheckResult {
            passed: all_satisfied,
            results,
        }
    }

    /// Get number of invariants
    pub fn len(&self) -> usize {
        self.invariants.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.invariants.is_empty()
    }
}

impl Default for TraceInvariantChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of checking all invariants on a trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceCheckResult {
    /// Whether all invariants passed
    pub passed: bool,
    /// Individual results
    pub results: Vec<InvariantCheckResult>,
}

impl TraceCheckResult {
    /// Get violated invariants
    pub fn violated(&self) -> Vec<&InvariantCheckResult> {
        self.results.iter().filter(|r| !r.satisfied).collect()
    }

    /// Get satisfied invariants
    pub fn satisfied(&self) -> Vec<&InvariantCheckResult> {
        self.results.iter().filter(|r| r.satisfied).collect()
    }
}

/// Common invariant patterns
pub mod patterns {
    use super::*;

    /// Create an invariant that a field is always positive
    pub fn field_positive(field_name: &str) -> Invariant {
        let field = field_name.to_string();
        Invariant::state(format!("{}_positive", field), move |state| {
            state
                .get(&field)
                .and_then(|v| v.as_i64())
                .is_some_and(|n| n >= 0)
        })
    }

    /// Create an invariant that a field is always in range [min, max]
    pub fn field_in_range(field_name: &str, min: i64, max: i64) -> Invariant {
        let field = field_name.to_string();
        Invariant::state(format!("{}_in_range", field), move |state| {
            state
                .get(&field)
                .and_then(|v| v.as_i64())
                .is_some_and(|n| n >= min && n <= max)
        })
    }

    /// Create an invariant that a field is always non-empty
    pub fn field_non_empty(field_name: &str) -> Invariant {
        let field = field_name.to_string();
        Invariant::state(format!("{}_non_empty", field), move |state| {
            state.get(&field).is_some_and(|v| match v {
                serde_json::Value::String(s) => !s.is_empty(),
                serde_json::Value::Array(a) => !a.is_empty(),
                serde_json::Value::Object(o) => !o.is_empty(),
                _ => true,
            })
        })
    }

    /// Create an invariant that two fields are mutually exclusive
    pub fn mutually_exclusive(field_a: &str, field_b: &str) -> Invariant {
        let a = field_a.to_string();
        let b = field_b.to_string();
        Invariant::state(format!("{}_or_{}_exclusive", a, b), move |state| {
            let a_true = state.get(&a).and_then(|v| v.as_bool()).unwrap_or(false);
            let b_true = state.get(&b).and_then(|v| v.as_bool()).unwrap_or(false);
            !(a_true && b_true)
        })
    }

    /// Create an invariant that a field is monotonically increasing
    pub fn monotonic_increasing(field_name: &str) -> Invariant {
        let field = field_name.to_string();
        Invariant::transition(
            format!("{}_monotonic_increasing", field),
            move |from, _action, to| {
                let from_val = from.get(&field).and_then(|v| v.as_i64()).unwrap_or(0);
                let to_val = to.get(&field).and_then(|v| v.as_i64()).unwrap_or(0);
                to_val >= from_val
            },
        )
    }

    /// Create an invariant that changes are bounded
    pub fn bounded_change(field_name: &str, max_change: i64) -> Invariant {
        let field = field_name.to_string();
        Invariant::transition(
            format!("{}_bounded_change", field),
            move |from, _action, to| {
                let from_val = from.get(&field).and_then(|v| v.as_i64()).unwrap_or(0);
                let to_val = to.get(&field).and_then(|v| v.as_i64()).unwrap_or(0);
                (to_val - from_val).abs() <= max_change
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dashprove_async::StateTransition;

    fn make_trace(states: Vec<serde_json::Value>) -> ExecutionTrace {
        let initial = states.first().cloned().unwrap_or(serde_json::json!({}));
        let mut trace = ExecutionTrace::new(initial);

        for window in states.windows(2) {
            let from = window[0].clone();
            let to = window[1].clone();
            trace.add_transition(StateTransition::new(from, "step".to_string(), to));
        }

        trace
    }

    #[test]
    fn test_state_invariant_satisfied() {
        let trace = make_trace(vec![
            serde_json::json!({"value": 1}),
            serde_json::json!({"value": 2}),
            serde_json::json!({"value": 3}),
        ]);

        let inv = Invariant::state("positive", |s| {
            s.get("value")
                .and_then(|v| v.as_i64())
                .is_some_and(|n| n > 0)
        });

        let result = check_invariant(&trace, &inv);
        assert!(result.satisfied);
        assert_eq!(result.violation_count, 0);
    }

    #[test]
    fn test_state_invariant_violated() {
        let trace = make_trace(vec![
            serde_json::json!({"value": 1}),
            serde_json::json!({"value": -1}),
            serde_json::json!({"value": 2}),
        ]);

        let inv = Invariant::state("positive", |s| {
            s.get("value")
                .and_then(|v| v.as_i64())
                .is_some_and(|n| n > 0)
        });

        let result = check_invariant(&trace, &inv);
        assert!(!result.satisfied);
        assert_eq!(result.first_violation_index, Some(1));
        assert_eq!(result.violation_count, 1);
    }

    #[test]
    fn test_transition_invariant() {
        let trace = make_trace(vec![
            serde_json::json!({"value": 0}),
            serde_json::json!({"value": 1}),
            serde_json::json!({"value": 2}),
        ]);

        let inv = Invariant::transition("monotonic", |from, _action, to| {
            let from_v = from.get("value").and_then(|v| v.as_i64()).unwrap_or(0);
            let to_v = to.get("value").and_then(|v| v.as_i64()).unwrap_or(0);
            to_v >= from_v
        });

        let result = check_invariant(&trace, &inv);
        assert!(result.satisfied);
    }

    #[test]
    fn test_implication_invariant() {
        let trace = make_trace(vec![
            serde_json::json!({"locked": false, "owner": null}),
            serde_json::json!({"locked": true, "owner": "user1"}),
            serde_json::json!({"locked": false, "owner": null}),
        ]);

        let inv = Invariant::implication(
            "lock_implies_owner",
            |s| s.get("locked").and_then(|v| v.as_bool()).unwrap_or(false),
            |s| s.get("owner").and_then(|v| v.as_str()).is_some(),
        );

        let result = check_invariant(&trace, &inv);
        assert!(result.satisfied);
    }

    #[test]
    fn test_trace_invariant_checker() {
        let trace = make_trace(vec![
            serde_json::json!({"x": 1, "y": 1}),
            serde_json::json!({"x": 2, "y": 2}),
        ]);

        let mut checker = TraceInvariantChecker::new();
        checker.add_state("x_positive", |s| {
            s.get("x").and_then(|v| v.as_i64()).is_some_and(|n| n > 0)
        });
        checker.add_state("y_positive", |s| {
            s.get("y").and_then(|v| v.as_i64()).is_some_and(|n| n > 0)
        });

        let result = checker.check(&trace);
        assert!(result.passed);
        assert_eq!(result.satisfied().len(), 2);
    }

    #[test]
    fn test_pattern_field_positive() {
        let trace = make_trace(vec![
            serde_json::json!({"balance": 100}),
            serde_json::json!({"balance": 50}),
            serde_json::json!({"balance": 0}),
        ]);

        let inv = patterns::field_positive("balance");
        let result = check_invariant(&trace, &inv);
        assert!(result.satisfied);
    }

    #[test]
    fn test_pattern_field_in_range() {
        let trace = make_trace(vec![
            serde_json::json!({"temp": 20}),
            serde_json::json!({"temp": 25}),
            serde_json::json!({"temp": 30}),
        ]);

        let inv = patterns::field_in_range("temp", 0, 100);
        let result = check_invariant(&trace, &inv);
        assert!(result.satisfied);
    }

    #[test]
    fn test_pattern_monotonic_increasing() {
        let trace = make_trace(vec![
            serde_json::json!({"counter": 0}),
            serde_json::json!({"counter": 1}),
            serde_json::json!({"counter": 1}),
            serde_json::json!({"counter": 2}),
        ]);

        let inv = patterns::monotonic_increasing("counter");
        let result = check_invariant(&trace, &inv);
        assert!(result.satisfied);

        // Now test violation
        let trace2 = make_trace(vec![
            serde_json::json!({"counter": 2}),
            serde_json::json!({"counter": 1}),
        ]);

        let result2 = check_invariant(&trace2, &inv);
        assert!(!result2.satisfied);
    }

    #[test]
    fn test_pattern_bounded_change() {
        let trace = make_trace(vec![
            serde_json::json!({"value": 0}),
            serde_json::json!({"value": 5}),
            serde_json::json!({"value": 8}),
        ]);

        let inv = patterns::bounded_change("value", 5);
        let result = check_invariant(&trace, &inv);
        assert!(result.satisfied);

        // Now test violation
        let trace2 = make_trace(vec![
            serde_json::json!({"value": 0}),
            serde_json::json!({"value": 10}),
        ]);

        let result2 = check_invariant(&trace2, &inv);
        assert!(!result2.satisfied);
    }

    #[test]
    fn test_pattern_mutually_exclusive() {
        let trace = make_trace(vec![
            serde_json::json!({"reading": false, "writing": false}),
            serde_json::json!({"reading": true, "writing": false}),
            serde_json::json!({"reading": false, "writing": true}),
        ]);

        let inv = patterns::mutually_exclusive("reading", "writing");
        let result = check_invariant(&trace, &inv);
        assert!(result.satisfied);
    }

    #[test]
    fn test_invariant_description() {
        let inv = Invariant::state("test", |_| true).with_description("This is a test invariant");

        assert_eq!(inv.description, "This is a test invariant");
    }

    // Mutation-killing tests for TraceInvariantChecker::len
    #[test]
    fn test_trace_invariant_checker_len() {
        let mut checker = TraceInvariantChecker::new();
        assert_eq!(checker.len(), 0);
        assert!(checker.is_empty());

        checker.add_state("inv1", |_| true);
        assert_eq!(checker.len(), 1);
        assert!(!checker.is_empty());

        checker.add_state("inv2", |_| true);
        assert_eq!(checker.len(), 2);
        assert!(!checker.is_empty());
    }

    // Mutation-killing tests for TraceCheckResult::violated
    #[test]
    fn test_trace_check_result_violated() {
        let trace = make_trace(vec![
            serde_json::json!({"x": 1, "y": -1}),
            serde_json::json!({"x": 2, "y": 2}),
        ]);

        let mut checker = TraceInvariantChecker::new();
        // First invariant always passes
        checker.add_state("x_positive", |s| {
            s.get("x").and_then(|v| v.as_i64()).is_some_and(|n| n > 0)
        });
        // Second invariant fails on first state
        checker.add_state("y_positive", |s| {
            s.get("y").and_then(|v| v.as_i64()).is_some_and(|n| n > 0)
        });

        let result = checker.check(&trace);
        assert!(!result.passed);

        // Verify violated() returns non-empty vec with the right invariant
        let violated = result.violated();
        assert_eq!(violated.len(), 1);
        assert_eq!(violated[0].invariant_name, "y_positive");
        assert!(!violated[0].satisfied);
    }

    // Mutation-killing tests for patterns::field_in_range && condition
    #[test]
    fn test_pattern_field_in_range_boundary() {
        // Test that both min and max bounds are enforced (kills && -> || mutation)
        let inv = patterns::field_in_range("val", 10, 20);

        // Value below min - should fail
        let trace_below = make_trace(vec![serde_json::json!({"val": 5})]);
        let result_below = check_invariant(&trace_below, &inv);
        assert!(!result_below.satisfied);

        // Value at min - should pass
        let trace_at_min = make_trace(vec![serde_json::json!({"val": 10})]);
        let result_at_min = check_invariant(&trace_at_min, &inv);
        assert!(result_at_min.satisfied);

        // Value above max - should fail
        let trace_above = make_trace(vec![serde_json::json!({"val": 25})]);
        let result_above = check_invariant(&trace_above, &inv);
        assert!(!result_above.satisfied);

        // Value at max - should pass
        let trace_at_max = make_trace(vec![serde_json::json!({"val": 20})]);
        let result_at_max = check_invariant(&trace_at_max, &inv);
        assert!(result_at_max.satisfied);
    }

    // Mutation-killing tests for patterns::field_non_empty match arms
    #[test]
    fn test_pattern_field_non_empty_string() {
        let inv = patterns::field_non_empty("name");

        // Empty string should fail
        let trace_empty = make_trace(vec![serde_json::json!({"name": ""})]);
        let result_empty = check_invariant(&trace_empty, &inv);
        assert!(!result_empty.satisfied);

        // Non-empty string should pass
        let trace_nonempty = make_trace(vec![serde_json::json!({"name": "test"})]);
        let result_nonempty = check_invariant(&trace_nonempty, &inv);
        assert!(result_nonempty.satisfied);
    }

    #[test]
    fn test_pattern_field_non_empty_array() {
        let inv = patterns::field_non_empty("items");

        // Empty array should fail
        let trace_empty = make_trace(vec![serde_json::json!({"items": []})]);
        let result_empty = check_invariant(&trace_empty, &inv);
        assert!(!result_empty.satisfied);

        // Non-empty array should pass
        let trace_nonempty = make_trace(vec![serde_json::json!({"items": [1, 2, 3]})]);
        let result_nonempty = check_invariant(&trace_nonempty, &inv);
        assert!(result_nonempty.satisfied);
    }

    #[test]
    fn test_pattern_field_non_empty_object() {
        let inv = patterns::field_non_empty("data");

        // Empty object should fail
        let trace_empty = make_trace(vec![serde_json::json!({"data": {}})]);
        let result_empty = check_invariant(&trace_empty, &inv);
        assert!(!result_empty.satisfied);

        // Non-empty object should pass
        let trace_nonempty = make_trace(vec![serde_json::json!({"data": {"key": "value"}})]);
        let result_nonempty = check_invariant(&trace_nonempty, &inv);
        assert!(result_nonempty.satisfied);
    }

    #[test]
    fn test_pattern_field_non_empty_other_types() {
        // Non-container types should pass (numbers, booleans, null)
        let inv = patterns::field_non_empty("val");

        let trace_num = make_trace(vec![serde_json::json!({"val": 42})]);
        let result_num = check_invariant(&trace_num, &inv);
        assert!(result_num.satisfied);

        let trace_bool = make_trace(vec![serde_json::json!({"val": true})]);
        let result_bool = check_invariant(&trace_bool, &inv);
        assert!(result_bool.satisfied);
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify InvariantCheckResult::satisfied creates a satisfied result
    #[kani::proof]
    fn kani_invariant_check_result_satisfied() {
        let result = InvariantCheckResult::satisfied("test");
        assert!(result.satisfied);
        assert!(result.first_violation_index.is_none());
        assert_eq!(result.violation_count, 0);
        assert!(result.violation_states.is_empty());
    }

    /// Verify InvariantCheckResult::violated creates a violated result
    #[kani::proof]
    fn kani_invariant_check_result_violated() {
        let result = InvariantCheckResult::violated("test", 5, vec![]);
        assert!(!result.satisfied);
        assert_eq!(result.first_violation_index, Some(5));
    }

    /// Verify TraceInvariantChecker default is empty
    #[kani::proof]
    fn kani_trace_checker_default_empty() {
        let checker = TraceInvariantChecker::default();
        assert!(checker.is_empty());
        assert_eq!(checker.len(), 0);
    }

    /// Verify TraceInvariantChecker::new creates empty checker
    #[kani::proof]
    fn kani_trace_checker_new_empty() {
        let checker = TraceInvariantChecker::new();
        assert!(checker.is_empty());
    }

    /// Verify TraceCheckResult::violated returns non-satisfied results
    #[kani::proof]
    fn kani_trace_check_result_satisfied_filter() {
        let result = TraceCheckResult {
            passed: true,
            results: vec![InvariantCheckResult::satisfied("ok")],
        };
        let satisfied = result.satisfied();
        assert_eq!(satisfied.len(), 1);
        let violated = result.violated();
        assert!(violated.is_empty());
    }

    /// Verify TraceCheckResult with mixed results
    #[kani::proof]
    fn kani_trace_check_result_mixed() {
        let result = TraceCheckResult {
            passed: false,
            results: vec![
                InvariantCheckResult::satisfied("ok"),
                InvariantCheckResult::violated("bad", 0, vec![]),
            ],
        };
        let satisfied = result.satisfied();
        let violated = result.violated();
        assert_eq!(satisfied.len(), 1);
        assert_eq!(violated.len(), 1);
    }

    /// Verify violation_count matches violation_states length
    #[kani::proof]
    fn kani_violation_count_matches_states() {
        let states = vec![serde_json::json!({}), serde_json::json!({})];
        let result = InvariantCheckResult::violated("test", 0, states.clone());
        assert_eq!(result.violation_count, states.len());
    }

    /// Verify invariant_name is preserved in satisfied result
    #[kani::proof]
    fn kani_invariant_name_preserved_satisfied() {
        let name = "my_invariant";
        let result = InvariantCheckResult::satisfied(name);
        assert_eq!(result.invariant_name, name);
    }

    /// Verify invariant_name is preserved in violated result
    #[kani::proof]
    fn kani_invariant_name_preserved_violated() {
        let name = "my_invariant";
        let result = InvariantCheckResult::violated(name, 0, vec![]);
        assert_eq!(result.invariant_name, name);
    }
}
