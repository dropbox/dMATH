//! Liveness property checking for traces
//!
//! This module provides functionality to check liveness properties
//! on execution traces. Liveness properties assert that "something good
//! eventually happens" (e.g., every request eventually gets a response).

#![allow(clippy::type_complexity)]

use serde::{Deserialize, Serialize};

use dashprove_async::ExecutionTrace;

/// A liveness property that can be checked on traces
pub struct LivenessProperty {
    /// Property name
    pub name: String,
    /// Description
    pub description: String,
    /// The predicate to check (returns true when property is satisfied)
    pub predicate: Box<dyn Fn(&serde_json::Value) -> bool + Send + Sync>,
    /// The trigger condition (when this becomes true, predicate must eventually be true)
    pub trigger: Option<Box<dyn Fn(&serde_json::Value) -> bool + Send + Sync>>,
    /// Maximum steps allowed between trigger and satisfaction
    pub deadline: Option<usize>,
}

impl LivenessProperty {
    /// Create a new liveness property (eventually P)
    pub fn eventually(
        name: impl Into<String>,
        predicate: impl Fn(&serde_json::Value) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            predicate: Box::new(predicate),
            trigger: None,
            deadline: None,
        }
    }

    /// Create a leads-to property (P leads to Q)
    pub fn leads_to(
        name: impl Into<String>,
        trigger: impl Fn(&serde_json::Value) -> bool + Send + Sync + 'static,
        response: impl Fn(&serde_json::Value) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            predicate: Box::new(response),
            trigger: Some(Box::new(trigger)),
            deadline: None,
        }
    }

    /// Set description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set deadline (bounded liveness)
    pub fn with_deadline(mut self, steps: usize) -> Self {
        self.deadline = Some(steps);
        self
    }

    /// Check if the predicate is satisfied in the given state
    pub fn is_satisfied(&self, state: &serde_json::Value) -> bool {
        (self.predicate)(state)
    }

    /// Check if the trigger condition is met in the given state
    pub fn is_triggered(&self, state: &serde_json::Value) -> bool {
        self.trigger.as_ref().is_none_or(|t| t(state))
    }
}

/// Result of checking a liveness property
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LivenessResult {
    /// Property name
    pub property_name: String,
    /// Whether the property was satisfied
    pub satisfied: bool,
    /// State index where trigger occurred (for leads-to properties)
    pub trigger_index: Option<usize>,
    /// State index where property was satisfied (if satisfied)
    pub satisfaction_index: Option<usize>,
    /// Number of steps between trigger and satisfaction
    pub steps_to_satisfaction: Option<usize>,
    /// If unsatisfied, the last state of the trace
    pub final_state: Option<serde_json::Value>,
    /// Error message if property was violated
    pub error_message: Option<String>,
}

impl LivenessResult {
    /// Create a satisfied result
    pub fn satisfied(property_name: impl Into<String>, satisfaction_index: usize) -> Self {
        Self {
            property_name: property_name.into(),
            satisfied: true,
            trigger_index: None,
            satisfaction_index: Some(satisfaction_index),
            steps_to_satisfaction: Some(satisfaction_index),
            final_state: None,
            error_message: None,
        }
    }

    /// Create an unsatisfied result
    pub fn unsatisfied(property_name: impl Into<String>, final_state: serde_json::Value) -> Self {
        Self {
            property_name: property_name.into(),
            satisfied: false,
            trigger_index: None,
            satisfaction_index: None,
            steps_to_satisfaction: None,
            final_state: Some(final_state),
            error_message: Some("Property not satisfied by end of trace".to_string()),
        }
    }

    /// Create a deadline exceeded result
    pub fn deadline_exceeded(
        property_name: impl Into<String>,
        trigger_index: usize,
        deadline: usize,
    ) -> Self {
        Self {
            property_name: property_name.into(),
            satisfied: false,
            trigger_index: Some(trigger_index),
            satisfaction_index: None,
            steps_to_satisfaction: None,
            final_state: None,
            error_message: Some(format!(
                "Deadline of {} steps exceeded after trigger at index {}",
                deadline, trigger_index
            )),
        }
    }

    /// Set trigger index
    pub fn with_trigger_index(mut self, index: usize) -> Self {
        self.trigger_index = Some(index);
        self
    }
}

/// Check a liveness property on an execution trace
pub fn check_liveness(trace: &ExecutionTrace, property: &LivenessProperty) -> LivenessResult {
    // Collect all states in the trace
    let mut states: Vec<&serde_json::Value> = vec![&trace.initial_state];
    for transition in &trace.transitions {
        states.push(&transition.to_state);
    }

    // For simple eventually properties (no trigger)
    if property.trigger.is_none() {
        for (i, state) in states.iter().enumerate() {
            if property.is_satisfied(state) {
                return LivenessResult::satisfied(&property.name, i);
            }
        }
        return LivenessResult::unsatisfied(&property.name, trace.final_state.clone());
    }

    // For leads-to properties
    let mut trigger_index: Option<usize> = None;

    for (i, state) in states.iter().enumerate() {
        // Check if trigger condition is met
        if trigger_index.is_none() && property.is_triggered(state) {
            trigger_index = Some(i);
        }

        // If triggered, check for satisfaction
        if let Some(ti) = trigger_index {
            if property.is_satisfied(state) {
                let steps = i - ti;
                return LivenessResult {
                    property_name: property.name.clone(),
                    satisfied: true,
                    trigger_index: Some(ti),
                    satisfaction_index: Some(i),
                    steps_to_satisfaction: Some(steps),
                    final_state: None,
                    error_message: None,
                };
            }

            // Check deadline
            if let Some(deadline) = property.deadline {
                if i - ti >= deadline {
                    return LivenessResult::deadline_exceeded(&property.name, ti, deadline);
                }
            }
        }
    }

    // If triggered but not satisfied
    if let Some(ti) = trigger_index {
        let mut result = LivenessResult::unsatisfied(&property.name, trace.final_state.clone());
        result.trigger_index = Some(ti);
        result.error_message = Some(format!(
            "Trigger at index {} but property never satisfied",
            ti
        ));
        return result;
    }

    // Never triggered - property vacuously satisfied
    LivenessResult {
        property_name: property.name.clone(),
        satisfied: true,
        trigger_index: None,
        satisfaction_index: None,
        steps_to_satisfaction: None,
        final_state: None,
        error_message: None,
    }
}

/// Check multiple liveness properties on a trace
pub fn check_liveness_properties(
    trace: &ExecutionTrace,
    properties: &[LivenessProperty],
) -> Vec<LivenessResult> {
    properties
        .iter()
        .map(|p| check_liveness(trace, p))
        .collect()
}

/// Fairness condition types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FairnessType {
    /// Weak fairness: if action is continuously enabled, it eventually occurs
    Weak,
    /// Strong fairness: if action is infinitely often enabled, it eventually occurs
    Strong,
}

/// A fairness condition for checking progress
pub struct FairnessCondition {
    /// Action name
    pub action: String,
    /// Enablement predicate
    pub enabled: Box<dyn Fn(&serde_json::Value) -> bool + Send + Sync>,
    /// Type of fairness
    pub fairness_type: FairnessType,
}

impl FairnessCondition {
    /// Create a weak fairness condition
    pub fn weak(
        action: impl Into<String>,
        enabled: impl Fn(&serde_json::Value) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self {
            action: action.into(),
            enabled: Box::new(enabled),
            fairness_type: FairnessType::Weak,
        }
    }

    /// Create a strong fairness condition
    pub fn strong(
        action: impl Into<String>,
        enabled: impl Fn(&serde_json::Value) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self {
            action: action.into(),
            enabled: Box::new(enabled),
            fairness_type: FairnessType::Strong,
        }
    }
}

/// Check if a trace satisfies weak fairness for an action
pub fn check_weak_fairness(trace: &ExecutionTrace, condition: &FairnessCondition) -> bool {
    // Weak fairness: if enabled continuously from some point, action must occur

    let mut states: Vec<&serde_json::Value> = vec![&trace.initial_state];
    for transition in &trace.transitions {
        states.push(&transition.to_state);
    }

    // Find runs where action is continuously enabled
    let mut enabled_from: Option<usize> = None;

    for (i, state) in states.iter().enumerate() {
        let is_enabled = (condition.enabled)(state);

        if is_enabled {
            if enabled_from.is_none() {
                enabled_from = Some(i);
            }
        } else {
            // Action became disabled - check if it occurred
            if let Some(start) = enabled_from {
                // Check if action occurred between start and i
                let action_occurred = trace
                    .transitions
                    .iter()
                    .skip(start)
                    .take(i - start)
                    .any(|t| t.event == condition.action);

                if !action_occurred && i > start {
                    // Was enabled for multiple steps but never occurred
                    // This might be a fairness violation depending on interpretation
                }
            }
            enabled_from = None;
        }
    }

    // If still enabled at end and never occurred, might be violation
    if let Some(start) = enabled_from {
        let action_occurred = trace
            .transitions
            .iter()
            .skip(start)
            .any(|t| t.event == condition.action);

        if !action_occurred && states.len() - start > 1 {
            return false; // Continuously enabled but never occurred
        }
    }

    true
}

/// Progress property checker
pub struct ProgressChecker {
    liveness_properties: Vec<LivenessProperty>,
    fairness_conditions: Vec<FairnessCondition>,
}

impl ProgressChecker {
    /// Create a new progress checker
    pub fn new() -> Self {
        Self {
            liveness_properties: vec![],
            fairness_conditions: vec![],
        }
    }

    /// Add a liveness property
    pub fn add_liveness(&mut self, property: LivenessProperty) {
        self.liveness_properties.push(property);
    }

    /// Add a fairness condition
    pub fn add_fairness(&mut self, condition: FairnessCondition) {
        self.fairness_conditions.push(condition);
    }

    /// Check all properties on a trace
    pub fn check(&self, trace: &ExecutionTrace) -> ProgressCheckResult {
        let liveness_results = check_liveness_properties(trace, &self.liveness_properties);

        let fairness_results: Vec<_> = self
            .fairness_conditions
            .iter()
            .map(|c| {
                let satisfied = check_weak_fairness(trace, c);
                (c.action.clone(), satisfied)
            })
            .collect();

        let all_liveness_satisfied = liveness_results.iter().all(|r| r.satisfied);
        let all_fairness_satisfied = fairness_results.iter().all(|(_, s)| *s);

        ProgressCheckResult {
            passed: all_liveness_satisfied && all_fairness_satisfied,
            liveness_results,
            fairness_results,
        }
    }
}

impl Default for ProgressChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of checking progress properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressCheckResult {
    /// Whether all properties passed
    pub passed: bool,
    /// Results for each liveness property
    pub liveness_results: Vec<LivenessResult>,
    /// Results for each fairness condition (action name, satisfied)
    pub fairness_results: Vec<(String, bool)>,
}

impl ProgressCheckResult {
    /// Get violated liveness properties
    pub fn violated_liveness(&self) -> Vec<&LivenessResult> {
        self.liveness_results
            .iter()
            .filter(|r| !r.satisfied)
            .collect()
    }

    /// Get violated fairness conditions
    pub fn violated_fairness(&self) -> Vec<&str> {
        self.fairness_results
            .iter()
            .filter(|(_, s)| !*s)
            .map(|(a, _)| a.as_str())
            .collect()
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
    fn test_eventually_satisfied() {
        let trace = make_trace(vec![
            serde_json::json!({"done": false}),
            serde_json::json!({"done": false}),
            serde_json::json!({"done": true}),
        ]);

        let prop = LivenessProperty::eventually("done", |s| {
            s.get("done").and_then(|v| v.as_bool()).unwrap_or(false)
        });

        let result = check_liveness(&trace, &prop);
        assert!(result.satisfied);
        assert_eq!(result.satisfaction_index, Some(2));
    }

    #[test]
    fn test_eventually_unsatisfied() {
        let trace = make_trace(vec![
            serde_json::json!({"done": false}),
            serde_json::json!({"done": false}),
            serde_json::json!({"done": false}),
        ]);

        let prop = LivenessProperty::eventually("done", |s| {
            s.get("done").and_then(|v| v.as_bool()).unwrap_or(false)
        });

        let result = check_liveness(&trace, &prop);
        assert!(!result.satisfied);
    }

    #[test]
    fn test_leads_to_satisfied() {
        let trace = make_trace(vec![
            serde_json::json!({"requested": false, "responded": false}),
            serde_json::json!({"requested": true, "responded": false}),
            serde_json::json!({"requested": true, "responded": false}),
            serde_json::json!({"requested": true, "responded": true}),
        ]);

        let prop = LivenessProperty::leads_to(
            "request_response",
            |s| {
                s.get("requested")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
            },
            |s| {
                s.get("responded")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
            },
        );

        let result = check_liveness(&trace, &prop);
        assert!(result.satisfied);
        assert_eq!(result.trigger_index, Some(1));
        assert_eq!(result.satisfaction_index, Some(3));
        assert_eq!(result.steps_to_satisfaction, Some(2));
    }

    #[test]
    fn test_leads_to_deadline_exceeded() {
        let trace = make_trace(vec![
            serde_json::json!({"requested": false, "responded": false}),
            serde_json::json!({"requested": true, "responded": false}),
            serde_json::json!({"requested": true, "responded": false}),
            serde_json::json!({"requested": true, "responded": false}),
            serde_json::json!({"requested": true, "responded": false}),
        ]);

        let prop = LivenessProperty::leads_to(
            "request_response",
            |s| {
                s.get("requested")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
            },
            |s| {
                s.get("responded")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
            },
        )
        .with_deadline(2);

        let result = check_liveness(&trace, &prop);
        assert!(!result.satisfied);
        assert!(result
            .error_message
            .as_ref()
            .is_some_and(|m| m.contains("Deadline")));
    }

    #[test]
    fn test_leads_to_not_triggered() {
        let trace = make_trace(vec![
            serde_json::json!({"requested": false, "responded": false}),
            serde_json::json!({"requested": false, "responded": false}),
        ]);

        let prop = LivenessProperty::leads_to(
            "request_response",
            |s| {
                s.get("requested")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
            },
            |s| {
                s.get("responded")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
            },
        );

        let result = check_liveness(&trace, &prop);
        // Vacuously satisfied since trigger never occurred
        assert!(result.satisfied);
        assert!(result.trigger_index.is_none());
    }

    #[test]
    fn test_liveness_property_builder() {
        let prop = LivenessProperty::eventually("test", |_| true)
            .with_description("Test property")
            .with_deadline(10);

        assert_eq!(prop.name, "test");
        assert_eq!(prop.description, "Test property");
        assert_eq!(prop.deadline, Some(10));
    }

    #[test]
    fn test_progress_checker() {
        let trace = make_trace(vec![
            serde_json::json!({"value": 0}),
            serde_json::json!({"value": 1}),
            serde_json::json!({"value": 2}),
        ]);

        let mut checker = ProgressChecker::new();
        checker.add_liveness(LivenessProperty::eventually("positive", |s| {
            s.get("value")
                .and_then(|v| v.as_i64())
                .is_some_and(|n| n > 0)
        }));

        let result = checker.check(&trace);
        assert!(result.passed);
        assert!(result.violated_liveness().is_empty());
    }

    #[test]
    fn test_multiple_liveness_properties() {
        let trace = make_trace(vec![
            serde_json::json!({"a": false, "b": false}),
            serde_json::json!({"a": true, "b": false}),
            serde_json::json!({"a": true, "b": true}),
        ]);

        let props = vec![
            LivenessProperty::eventually("a_true", |s| {
                s.get("a").and_then(|v| v.as_bool()).unwrap_or(false)
            }),
            LivenessProperty::eventually("b_true", |s| {
                s.get("b").and_then(|v| v.as_bool()).unwrap_or(false)
            }),
        ];

        let results = check_liveness_properties(&trace, &props);
        assert!(results.iter().all(|r| r.satisfied));
    }

    #[test]
    fn test_fairness_condition() {
        let weak = FairnessCondition::weak("action", |_| true);
        assert_eq!(weak.fairness_type, FairnessType::Weak);

        let strong = FairnessCondition::strong("action", |_| true);
        assert_eq!(strong.fairness_type, FairnessType::Strong);
    }

    // Mutation-killing tests for check_liveness >= condition (line 197)
    #[test]
    fn test_liveness_deadline_exactly_reached() {
        // When i - ti == deadline, should trigger deadline_exceeded
        let trace = make_trace(vec![
            serde_json::json!({"requested": false, "responded": false}),
            serde_json::json!({"requested": true, "responded": false}), // trigger at index 1
            serde_json::json!({"requested": true, "responded": false}), // index 2 (1 step)
            serde_json::json!({"requested": true, "responded": false}), // index 3 (2 steps = deadline)
        ]);

        let prop = LivenessProperty::leads_to(
            "request_response",
            |s| {
                s.get("requested")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
            },
            |s| {
                s.get("responded")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
            },
        )
        .with_deadline(2);

        let result = check_liveness(&trace, &prop);
        assert!(!result.satisfied);
    }

    // Mutation-killing tests for check_liveness subtraction (i - ti)
    #[test]
    fn test_liveness_steps_calculation() {
        let trace = make_trace(vec![
            serde_json::json!({"requested": false, "responded": false}), // index 0
            serde_json::json!({"requested": true, "responded": false}),  // index 1 - trigger
            serde_json::json!({"requested": true, "responded": false}),  // index 2
            serde_json::json!({"requested": true, "responded": true}),   // index 3 - satisfied
        ]);

        let prop = LivenessProperty::leads_to(
            "request_response",
            |s| {
                s.get("requested")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
            },
            |s| {
                s.get("responded")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
            },
        );

        let result = check_liveness(&trace, &prop);
        assert!(result.satisfied);
        assert_eq!(result.trigger_index, Some(1));
        assert_eq!(result.satisfaction_index, Some(3));
        // Steps = 3 - 1 = 2
        assert_eq!(result.steps_to_satisfaction, Some(2));
    }

    // Mutation-killing tests for check_liveness_properties returning vec![]
    #[test]
    fn test_check_liveness_properties_not_empty() {
        let trace = make_trace(vec![
            serde_json::json!({"done": true}),
            serde_json::json!({"done": true}),
        ]);

        let props = vec![LivenessProperty::eventually("done", |s| {
            s.get("done").and_then(|v| v.as_bool()).unwrap_or(false)
        })];

        let results = check_liveness_properties(&trace, &props);
        assert_eq!(results.len(), 1);
        assert!(results[0].satisfied);
        assert_eq!(results[0].property_name, "done");
    }

    // Mutation-killing tests for check_weak_fairness (lines 287-336)
    #[test]
    fn test_weak_fairness_action_occurs() {
        // Action is continuously enabled and occurs - should pass
        let mut trace = ExecutionTrace::new(serde_json::json!({"enabled": true}));
        trace.add_transition(StateTransition::new(
            serde_json::json!({"enabled": true}),
            "action".to_string(),
            serde_json::json!({"enabled": true}),
        ));
        trace.add_transition(StateTransition::new(
            serde_json::json!({"enabled": true}),
            "other".to_string(),
            serde_json::json!({"enabled": false}),
        ));

        let condition = FairnessCondition::weak("action", |s| {
            s.get("enabled").and_then(|v| v.as_bool()).unwrap_or(false)
        });

        assert!(check_weak_fairness(&trace, &condition));
    }

    #[test]
    fn test_weak_fairness_action_never_occurs_violation() {
        // Action continuously enabled at end but never occurs - should fail
        let mut trace = ExecutionTrace::new(serde_json::json!({"enabled": true}));
        trace.add_transition(StateTransition::new(
            serde_json::json!({"enabled": true}),
            "other1".to_string(),
            serde_json::json!({"enabled": true}),
        ));
        trace.add_transition(StateTransition::new(
            serde_json::json!({"enabled": true}),
            "other2".to_string(),
            serde_json::json!({"enabled": true}),
        ));

        let condition = FairnessCondition::weak("action", |s| {
            s.get("enabled").and_then(|v| v.as_bool()).unwrap_or(false)
        });

        // Should fail because continuously enabled but action never occurred
        assert!(!check_weak_fairness(&trace, &condition));
    }

    #[test]
    fn test_weak_fairness_disabled_at_end() {
        // Action is disabled before trace ends - passes even without occurrence
        let mut trace = ExecutionTrace::new(serde_json::json!({"enabled": true}));
        trace.add_transition(StateTransition::new(
            serde_json::json!({"enabled": true}),
            "other".to_string(),
            serde_json::json!({"enabled": false}),
        ));

        let condition = FairnessCondition::weak("action", |s| {
            s.get("enabled").and_then(|v| v.as_bool()).unwrap_or(false)
        });

        assert!(check_weak_fairness(&trace, &condition));
    }

    // Mutation-killing tests for ProgressChecker::add_liveness and add_fairness
    #[test]
    fn test_progress_checker_add_liveness() {
        let mut checker = ProgressChecker::new();
        checker.add_liveness(LivenessProperty::eventually("test", |_| true));

        let trace = make_trace(vec![serde_json::json!({})]);
        let result = checker.check(&trace);

        // Should have exactly one liveness result
        assert_eq!(result.liveness_results.len(), 1);
        assert!(result.liveness_results[0].satisfied);
    }

    #[test]
    fn test_progress_checker_add_fairness() {
        let mut checker = ProgressChecker::new();
        checker.add_fairness(FairnessCondition::weak("action", |_| false));

        let trace = make_trace(vec![serde_json::json!({})]);
        let result = checker.check(&trace);

        // Should have exactly one fairness result
        assert_eq!(result.fairness_results.len(), 1);
    }

    // Mutation-killing tests for ProgressChecker::check && condition (line 380)
    #[test]
    fn test_progress_checker_both_conditions_required() {
        let mut trace = ExecutionTrace::new(serde_json::json!({"enabled": true}));
        trace.add_transition(StateTransition::new(
            serde_json::json!({"enabled": true}),
            "other".to_string(),
            serde_json::json!({"enabled": true}),
        ));

        let mut checker = ProgressChecker::new();
        // Liveness passes
        checker.add_liveness(LivenessProperty::eventually("always_true", |_| true));
        // Fairness fails - enabled but action never occurs
        checker.add_fairness(FairnessCondition::weak("action", |s| {
            s.get("enabled").and_then(|v| v.as_bool()).unwrap_or(false)
        }));

        let result = checker.check(&trace);
        // Should fail because fairness fails even though liveness passes
        assert!(!result.passed);
    }

    // Mutation-killing tests for ProgressCheckResult::violated_liveness (lines 407-411)
    #[test]
    fn test_progress_check_result_violated_liveness() {
        let trace = make_trace(vec![
            serde_json::json!({"done": false}),
            serde_json::json!({"done": false}),
        ]);

        let mut checker = ProgressChecker::new();
        checker.add_liveness(LivenessProperty::eventually("done", |s| {
            s.get("done").and_then(|v| v.as_bool()).unwrap_or(false)
        }));

        let result = checker.check(&trace);
        assert!(!result.passed);

        let violated = result.violated_liveness();
        assert_eq!(violated.len(), 1);
        assert_eq!(violated[0].property_name, "done");
    }

    // Mutation-killing tests for ProgressCheckResult::violated_fairness (lines 415-420)
    #[test]
    fn test_progress_check_result_violated_fairness() {
        let mut trace = ExecutionTrace::new(serde_json::json!({"enabled": true}));
        trace.add_transition(StateTransition::new(
            serde_json::json!({"enabled": true}),
            "other".to_string(),
            serde_json::json!({"enabled": true}),
        ));

        let mut checker = ProgressChecker::new();
        checker.add_fairness(FairnessCondition::weak("missing_action", |s| {
            s.get("enabled").and_then(|v| v.as_bool()).unwrap_or(false)
        }));

        let result = checker.check(&trace);

        let violated = result.violated_fairness();
        assert_eq!(violated.len(), 1);
        assert_eq!(violated[0], "missing_action");
    }

    #[test]
    fn test_progress_check_result_no_violated_fairness() {
        let trace = make_trace(vec![serde_json::json!({"enabled": false})]);

        let mut checker = ProgressChecker::new();
        checker.add_fairness(FairnessCondition::weak("action", |s| {
            s.get("enabled").and_then(|v| v.as_bool()).unwrap_or(false)
        }));

        let result = checker.check(&trace);

        // Fairness should pass because action is never enabled
        let violated = result.violated_fairness();
        assert!(violated.is_empty());
    }

    // Test LivenessResult::with_trigger_index
    #[test]
    fn test_liveness_result_with_trigger_index() {
        let result = LivenessResult::satisfied("test", 5).with_trigger_index(2);
        assert_eq!(result.trigger_index, Some(2));
        assert_eq!(result.satisfaction_index, Some(5));
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify LivenessResult::satisfied creates satisfied result
    #[kani::proof]
    fn kani_liveness_result_satisfied() {
        let result = LivenessResult::satisfied("test", 5);
        assert!(result.satisfied);
        assert_eq!(result.satisfaction_index, Some(5));
        assert!(result.final_state.is_none());
        assert!(result.error_message.is_none());
    }

    /// Verify LivenessResult::unsatisfied creates unsatisfied result
    #[kani::proof]
    fn kani_liveness_result_unsatisfied() {
        let result = LivenessResult::unsatisfied("test", serde_json::json!({}));
        assert!(!result.satisfied);
        assert!(result.satisfaction_index.is_none());
        assert!(result.final_state.is_some());
        assert!(result.error_message.is_some());
    }

    /// Verify LivenessResult::deadline_exceeded creates unsatisfied result
    #[kani::proof]
    fn kani_liveness_result_deadline_exceeded() {
        let result = LivenessResult::deadline_exceeded("test", 2, 5);
        assert!(!result.satisfied);
        assert_eq!(result.trigger_index, Some(2));
        assert!(result.error_message.is_some());
    }

    /// Verify with_trigger_index preserves other fields
    #[kani::proof]
    fn kani_with_trigger_index_preserves() {
        let result = LivenessResult::satisfied("test", 10).with_trigger_index(3);
        assert!(result.satisfied);
        assert_eq!(result.satisfaction_index, Some(10));
        assert_eq!(result.trigger_index, Some(3));
    }

    /// Verify FairnessType variants are distinct
    #[kani::proof]
    fn kani_fairness_type_variants() {
        let weak = FairnessType::Weak;
        let strong = FairnessType::Strong;
        assert!(weak != strong);
    }

    /// Verify ProgressChecker::new creates empty checker
    #[kani::proof]
    fn kani_progress_checker_new_empty() {
        let checker = ProgressChecker::new();
        let result = checker.check(&ExecutionTrace::new(serde_json::json!({})));
        assert!(result.liveness_results.is_empty());
        assert!(result.fairness_results.is_empty());
    }

    /// Verify ProgressChecker::default is same as new
    #[kani::proof]
    fn kani_progress_checker_default() {
        let checker = ProgressChecker::default();
        let result = checker.check(&ExecutionTrace::new(serde_json::json!({})));
        assert!(result.passed);
    }

    /// Verify ProgressCheckResult with empty results passes
    #[kani::proof]
    fn kani_progress_check_result_empty_passes() {
        let result = ProgressCheckResult {
            passed: true,
            liveness_results: vec![],
            fairness_results: vec![],
        };
        assert!(result.passed);
        assert!(result.violated_liveness().is_empty());
        assert!(result.violated_fairness().is_empty());
    }

    /// Verify violated_liveness filters correctly
    #[kani::proof]
    fn kani_violated_liveness_filter() {
        let result = ProgressCheckResult {
            passed: false,
            liveness_results: vec![
                LivenessResult::satisfied("ok", 0),
                LivenessResult::unsatisfied("bad", serde_json::json!({})),
            ],
            fairness_results: vec![],
        };
        let violated = result.violated_liveness();
        assert_eq!(violated.len(), 1);
        assert_eq!(violated[0].property_name, "bad");
    }

    /// Verify violated_fairness filters correctly
    #[kani::proof]
    fn kani_violated_fairness_filter() {
        let result = ProgressCheckResult {
            passed: false,
            liveness_results: vec![],
            fairness_results: vec![("ok".to_string(), true), ("bad".to_string(), false)],
        };
        let violated = result.violated_fairness();
        assert_eq!(violated.len(), 1);
        assert_eq!(violated[0], "bad");
    }

    /// Verify property name preserved in satisfied result
    #[kani::proof]
    fn kani_property_name_preserved_satisfied() {
        let result = LivenessResult::satisfied("my_prop", 0);
        assert_eq!(result.property_name, "my_prop");
    }

    /// Verify property name preserved in unsatisfied result
    #[kani::proof]
    fn kani_property_name_preserved_unsatisfied() {
        let result = LivenessResult::unsatisfied("my_prop", serde_json::json!({}));
        assert_eq!(result.property_name, "my_prop");
    }

    /// Verify steps_to_satisfaction is set correctly for satisfied
    #[kani::proof]
    fn kani_steps_to_satisfaction() {
        let idx: usize = 7;
        let result = LivenessResult::satisfied("test", idx);
        assert_eq!(result.steps_to_satisfaction, Some(idx));
    }
}
