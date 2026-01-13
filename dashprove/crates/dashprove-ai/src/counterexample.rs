//! Counterexample explanation
//!
//! When verification fails with a counterexample, this module provides
//! human-readable explanations of why the property doesn't hold.

use dashprove_backends::traits::BackendId;
use dashprove_usl::ast::Property;
use serde::{Deserialize, Serialize};

/// Type of explanation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExplanationKind {
    /// Direct variable assignment that violates property
    VariableAssignment,
    /// Sequence of states leading to violation
    StateTrace,
    /// Missing case in proof
    MissingCase,
    /// Precondition not satisfied
    PreconditionViolation,
    /// Postcondition not established
    PostconditionViolation,
    /// General explanation
    General,
}

/// A parsed variable binding from a counterexample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Binding {
    /// Variable name
    pub name: String,
    /// Variable value as a string representation
    pub value: String,
    /// Optional type annotation for the variable
    pub ty: Option<String>,
}

/// A step in a trace counterexample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceStep {
    /// Step index in the trace (0-based)
    pub step_number: usize,
    /// Name of the action taken at this step (for TLA+ state machines)
    pub action: Option<String>,
    /// Variable bindings at this step
    pub state: Vec<Binding>,
}

/// Human-readable explanation of a counterexample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterexampleExplanation {
    /// Type of explanation
    pub kind: ExplanationKind,
    /// Short summary
    pub summary: String,
    /// Detailed explanation
    pub details: String,
    /// Variable bindings that violate the property
    pub bindings: Vec<Binding>,
    /// Trace steps (for temporal properties)
    pub trace: Vec<TraceStep>,
    /// Suggestions to fix the property
    pub suggestions: Vec<String>,
}

/// Explain a counterexample in human-readable form
pub fn explain_counterexample(
    property: &Property,
    counterexample: &str,
    backend: &BackendId,
) -> CounterexampleExplanation {
    match backend {
        BackendId::Lean4 => explain_lean_counterexample(property, counterexample),
        BackendId::TlaPlus => explain_tlaplus_counterexample(property, counterexample),
        BackendId::Kani => explain_kani_counterexample(property, counterexample),
        BackendId::Alloy => explain_alloy_counterexample(property, counterexample),
        _ => generic_explanation(counterexample),
    }
}

/// Explain LEAN 4 counterexample/error
fn explain_lean_counterexample(
    property: &Property,
    counterexample: &str,
) -> CounterexampleExplanation {
    let kind = if counterexample.contains("unsolved goals") {
        ExplanationKind::MissingCase
    } else {
        ExplanationKind::General
    };

    let property_name = match property {
        Property::Theorem(t) => &t.name,
        Property::Invariant(i) => &i.name,
        Property::Refinement(r) => &r.name,
        Property::DistributedInvariant(d) => &d.name,
        _ => "property",
    };

    let summary = if counterexample.contains("unsolved goals") {
        format!(
            "Proof of '{}' is incomplete - there are unsolved goals",
            property_name
        )
    } else if counterexample.contains("type mismatch") {
        format!(
            "Type error in proof of '{}' - expected and actual types don't match",
            property_name
        )
    } else if counterexample.contains("unknown identifier") {
        format!(
            "Unknown identifier referenced in proof of '{}'",
            property_name
        )
    } else {
        format!("Verification of '{}' failed", property_name)
    };

    let mut suggestions = Vec::new();
    if counterexample.contains("unsolved goals") {
        suggestions.push("Try adding more tactics to complete the proof".to_string());
        suggestions
            .push("Consider using 'sorry' temporarily to identify remaining goals".to_string());
    }
    if counterexample.contains("type mismatch") {
        suggestions.push("Check that function arguments have the correct types".to_string());
        suggestions.push("Use explicit type annotations to clarify intent".to_string());
    }

    CounterexampleExplanation {
        kind,
        summary,
        details: counterexample.to_string(),
        bindings: vec![],
        trace: vec![],
        suggestions,
    }
}

/// Explain TLA+ counterexample (trace)
fn explain_tlaplus_counterexample(
    property: &Property,
    counterexample: &str,
) -> CounterexampleExplanation {
    let property_name = match property {
        Property::Temporal(t) => &t.name,
        Property::Invariant(i) => &i.name,
        Property::DistributedInvariant(d) => &d.name,
        Property::DistributedTemporal(d) => &d.name,
        _ => "property",
    };

    // Parse TLA+ trace format
    let trace = parse_tlaplus_trace(counterexample);
    let bindings = if let Some(first_step) = trace.first() {
        first_step.state.clone()
    } else {
        vec![]
    };

    let summary = if trace.is_empty() {
        format!(
            "Property '{}' was violated but no trace available",
            property_name
        )
    } else if trace.len() == 1 {
        format!("Property '{}' violated in initial state", property_name)
    } else {
        format!(
            "Property '{}' violated after {} steps",
            property_name,
            trace.len() - 1
        )
    };

    let details = if trace.len() > 1 {
        let last_action = trace.last().and_then(|s| s.action.as_ref());
        if let Some(action) = last_action {
            format!(
                "The violation occurs after action '{}'. Check the invariant preservation for this action.",
                action
            )
        } else {
            "The trace shows a sequence of states leading to the violation.".to_string()
        }
    } else {
        "Check the initial state satisfies the property.".to_string()
    };

    let mut suggestions = Vec::new();
    suggestions.push("Review the action that leads to the violating state".to_string());
    if trace.len() == 1 {
        suggestions
            .push("The initial state violates the property - check Init predicate".to_string());
    }

    CounterexampleExplanation {
        kind: ExplanationKind::StateTrace,
        summary,
        details,
        bindings,
        trace,
        suggestions,
    }
}

/// Parse TLA+ counterexample trace
fn parse_tlaplus_trace(counterexample: &str) -> Vec<TraceStep> {
    let mut trace = Vec::new();
    let mut current_step: Option<TraceStep> = None;

    for line in counterexample.lines() {
        let line = line.trim();

        // Detect state headers like "State 1:" or "Initial state:"
        if line.starts_with("State ") || line.contains("Initial state") {
            if let Some(step) = current_step.take() {
                trace.push(step);
            }
            let step_number = trace.len();
            current_step = Some(TraceStep {
                step_number,
                action: None,
                state: vec![],
            });
        }
        // Detect action lines
        else if line.starts_with("/\\") || line.contains("_action") {
            if let Some(ref mut step) = current_step {
                // Extract action name
                let action = line
                    .trim_start_matches("/\\ ")
                    .split_whitespace()
                    .next()
                    .unwrap_or("")
                    .to_string();
                if !action.is_empty() {
                    step.action = Some(action);
                }
            }
        }
        // Parse variable assignments
        else if line.contains(" = ") || line.contains(" |-> ") {
            if let Some(ref mut step) = current_step {
                let parts: Vec<&str> = if line.contains(" |-> ") {
                    line.splitn(2, " |-> ").collect()
                } else {
                    line.splitn(2, " = ").collect()
                };
                if parts.len() == 2 {
                    step.state.push(Binding {
                        name: parts[0].trim().to_string(),
                        value: parts[1].trim().to_string(),
                        ty: None,
                    });
                }
            }
        }
    }

    // Don't forget the last step
    if let Some(step) = current_step {
        trace.push(step);
    }

    trace
}

/// Explain Kani counterexample
fn explain_kani_counterexample(
    property: &Property,
    counterexample: &str,
) -> CounterexampleExplanation {
    let property_name = match property {
        Property::Contract(c) => c.type_path.join("::"),
        _ => "contract".to_string(),
    };

    let kind = if counterexample.contains("assertion failed") {
        ExplanationKind::PostconditionViolation
    } else if counterexample.contains("assume") {
        ExplanationKind::PreconditionViolation
    } else {
        ExplanationKind::General
    };

    let summary = match kind {
        ExplanationKind::PostconditionViolation => {
            format!("Postcondition of '{}' not satisfied", property_name)
        }
        ExplanationKind::PreconditionViolation => {
            format!("Precondition of '{}' may not hold", property_name)
        }
        _ => format!("Contract '{}' verification failed", property_name),
    };

    // Parse Kani's concrete values
    let bindings = parse_kani_values(counterexample);

    let mut suggestions = Vec::new();
    if kind == ExplanationKind::PostconditionViolation {
        suggestions
            .push("Check the postcondition logic matches implementation behavior".to_string());
        suggestions.push("Consider if the postcondition is too strong".to_string());
    }
    if kind == ExplanationKind::PreconditionViolation {
        suggestions.push("Strengthen the precondition to exclude this case".to_string());
    }

    CounterexampleExplanation {
        kind,
        summary,
        details: counterexample.to_string(),
        bindings,
        trace: vec![],
        suggestions,
    }
}

/// Parse Kani counterexample values
fn parse_kani_values(counterexample: &str) -> Vec<Binding> {
    let mut bindings = Vec::new();

    for line in counterexample.lines() {
        let line = line.trim();
        // Kani outputs concrete values in various formats
        if line.contains("concrete value:") || line.contains(" = ") {
            let parts: Vec<&str> = line.splitn(2, " = ").collect();
            if parts.len() == 2 {
                bindings.push(Binding {
                    name: parts[0].trim().to_string(),
                    value: parts[1].trim().to_string(),
                    ty: None,
                });
            }
        }
    }

    bindings
}

/// Explain Alloy counterexample
fn explain_alloy_counterexample(
    property: &Property,
    counterexample: &str,
) -> CounterexampleExplanation {
    let property_name = match property {
        Property::Invariant(i) => &i.name,
        Property::Theorem(t) => &t.name,
        Property::DistributedInvariant(d) => &d.name,
        _ => "property",
    };

    let summary = format!(
        "Counterexample found for '{}' within bounded scope",
        property_name
    );

    // Parse Alloy's instance format
    let bindings = parse_alloy_instance(counterexample);

    let suggestions = vec![
        "Review the counterexample instance to understand why the property fails".to_string(),
        "Consider if the property needs additional constraints".to_string(),
        "Increase scope if you want to check larger instances".to_string(),
    ];

    CounterexampleExplanation {
        kind: ExplanationKind::VariableAssignment,
        summary,
        details: counterexample.to_string(),
        bindings,
        trace: vec![],
        suggestions,
    }
}

/// Parse Alloy instance format
fn parse_alloy_instance(counterexample: &str) -> Vec<Binding> {
    let mut bindings = Vec::new();

    for line in counterexample.lines() {
        let line = line.trim();
        // Alloy outputs tuples and relations
        if line.contains("=") && !line.starts_with("--") {
            let parts: Vec<&str> = line.splitn(2, "=").collect();
            if parts.len() == 2 {
                bindings.push(Binding {
                    name: parts[0].trim().to_string(),
                    value: parts[1].trim().to_string(),
                    ty: None,
                });
            }
        }
    }

    bindings
}

/// Generic explanation when backend-specific parsing isn't available
fn generic_explanation(counterexample: &str) -> CounterexampleExplanation {
    CounterexampleExplanation {
        kind: ExplanationKind::General,
        summary: "Verification failed with counterexample".to_string(),
        details: counterexample.to_string(),
        bindings: vec![],
        trace: vec![],
        suggestions: vec![
            "Review the counterexample output for specific values".to_string(),
            "Check if the property specification is correct".to_string(),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dashprove_usl::ast::{Invariant, Temporal, TemporalExpr, Theorem};

    #[test]
    fn test_lean_unsolved_goals_explanation() {
        let prop = Property::Theorem(Theorem {
            name: "test_theorem".to_string(),
            body: dashprove_usl::ast::Expr::Bool(true),
        });
        let error = "unsolved goals\n⊢ P ∧ Q";
        let explanation = explain_counterexample(&prop, error, &BackendId::Lean4);

        assert_eq!(explanation.kind, ExplanationKind::MissingCase);
        assert!(explanation.summary.contains("incomplete"));
        assert!(!explanation.suggestions.is_empty());
    }

    #[test]
    fn test_tlaplus_trace_parsing() {
        let trace_output = r#"
State 1:
  x = 0
  y = 1
State 2:
  /\ Next_action
  x = 1
  y = 2
"#;
        let trace = parse_tlaplus_trace(trace_output);

        assert_eq!(trace.len(), 2);
        assert_eq!(trace[0].step_number, 0);
        assert_eq!(trace[0].state.len(), 2);
        assert!(trace[1].action.is_some());
    }

    #[test]
    fn test_tlaplus_explanation() {
        let prop = Property::Temporal(Temporal {
            name: "safety".to_string(),
            body: TemporalExpr::Always(Box::new(TemporalExpr::Atom(
                dashprove_usl::ast::Expr::Bool(true),
            ))),
            fairness: vec![],
        });
        let ce = "State 1:\n  x = 0\nState 2:\n  x = 1";
        let explanation = explain_counterexample(&prop, ce, &BackendId::TlaPlus);

        assert_eq!(explanation.kind, ExplanationKind::StateTrace);
        assert!(explanation.summary.contains("violated"));
    }

    #[test]
    fn test_kani_postcondition_explanation() {
        let prop = Property::Contract(dashprove_usl::ast::Contract {
            type_path: vec!["add".to_string()],
            params: vec![],
            return_type: None,
            requires: vec![],
            ensures: vec![],
            ensures_err: vec![],
            assigns: vec![],
            allocates: vec![],
            frees: vec![],
            terminates: None,
            decreases: None,
            behaviors: vec![],
            complete_behaviors: false,
            disjoint_behaviors: false,
        });
        let error = "assertion failed: result > 0";
        let explanation = explain_counterexample(&prop, error, &BackendId::Kani);

        assert_eq!(explanation.kind, ExplanationKind::PostconditionViolation);
        assert!(explanation.summary.contains("Postcondition"));
    }

    #[test]
    fn test_alloy_explanation() {
        let prop = Property::Invariant(Invariant {
            name: "no_cycles".to_string(),
            body: dashprove_usl::ast::Expr::Bool(true),
        });
        let ce = "sig Node = {Node0, Node1}\nedges = {Node0->Node1, Node1->Node0}";
        let explanation = explain_counterexample(&prop, ce, &BackendId::Alloy);

        assert_eq!(explanation.kind, ExplanationKind::VariableAssignment);
        assert!(explanation.summary.contains("Counterexample"));
    }

    #[test]
    fn test_generic_explanation() {
        let explanation = generic_explanation("some error output");

        assert_eq!(explanation.kind, ExplanationKind::General);
        assert!(!explanation.suggestions.is_empty());
    }

    // ========== Mutation-killing tests ==========

    // Tests for property name extraction in explain_lean_counterexample (lines 93-96)
    #[test]
    fn test_lean_theorem_name_used_in_summary() {
        let prop = Property::Theorem(Theorem {
            name: "unique_theorem_name".to_string(),
            body: dashprove_usl::ast::Expr::Bool(true),
        });
        let error = "general error";
        let explanation = explain_counterexample(&prop, error, &BackendId::Lean4);
        assert!(
            explanation.summary.contains("unique_theorem_name"),
            "Theorem name should appear in summary"
        );
    }

    #[test]
    fn test_lean_invariant_name_used_in_summary() {
        let prop = Property::Invariant(Invariant {
            name: "unique_invariant_name".to_string(),
            body: dashprove_usl::ast::Expr::Bool(true),
        });
        let error = "general error";
        let explanation = explain_counterexample(&prop, error, &BackendId::Lean4);
        assert!(
            explanation.summary.contains("unique_invariant_name"),
            "Invariant name should appear in summary"
        );
    }

    #[test]
    fn test_lean_refinement_name_used_in_summary() {
        let prop = Property::Refinement(dashprove_usl::ast::Refinement {
            name: "unique_refinement_name".to_string(),
            refines: "AbstractSpec".to_string(),
            mappings: vec![],
            invariants: vec![],
            abstraction: dashprove_usl::ast::Expr::Bool(true),
            simulation: dashprove_usl::ast::Expr::Bool(true),
            actions: vec![],
        });
        let error = "general error";
        let explanation = explain_counterexample(&prop, error, &BackendId::Lean4);
        assert!(
            explanation.summary.contains("unique_refinement_name"),
            "Refinement name should appear in summary"
        );
    }

    // Tests for property name extraction in explain_tlaplus_counterexample (lines 145-147)
    #[test]
    fn test_tlaplus_temporal_name_used_in_summary() {
        let prop = Property::Temporal(Temporal {
            name: "unique_temporal_name".to_string(),
            body: TemporalExpr::Always(Box::new(TemporalExpr::Atom(
                dashprove_usl::ast::Expr::Bool(true),
            ))),
            fairness: vec![],
        });
        let ce = "State 1:\n  x = 0";
        let explanation = explain_counterexample(&prop, ce, &BackendId::TlaPlus);
        assert!(
            explanation.summary.contains("unique_temporal_name"),
            "Temporal property name should appear in summary"
        );
    }

    #[test]
    fn test_tlaplus_invariant_name_used_in_summary() {
        let prop = Property::Invariant(Invariant {
            name: "unique_tla_invariant".to_string(),
            body: dashprove_usl::ast::Expr::Bool(true),
        });
        let ce = "State 1:\n  x = 0";
        let explanation = explain_counterexample(&prop, ce, &BackendId::TlaPlus);
        assert!(
            explanation.summary.contains("unique_tla_invariant"),
            "Invariant name should appear in summary"
        );
    }

    // Tests for trace.len() comparisons (lines 163, 173, 189)
    #[test]
    fn test_tlaplus_single_state_summary_says_initial_state() {
        // len() == 1 should give "violated in initial state" message
        let prop = Property::Temporal(Temporal {
            name: "test".to_string(),
            body: TemporalExpr::Always(Box::new(TemporalExpr::Atom(
                dashprove_usl::ast::Expr::Bool(true),
            ))),
            fairness: vec![],
        });
        let ce = "State 1:\n  x = 0";
        let explanation = explain_counterexample(&prop, ce, &BackendId::TlaPlus);
        assert!(
            explanation.summary.contains("initial state"),
            "Single state trace should mention initial state violation"
        );
    }

    #[test]
    fn test_tlaplus_multi_state_shows_step_count() {
        // trace.len() > 1 should show step count
        let prop = Property::Temporal(Temporal {
            name: "test".to_string(),
            body: TemporalExpr::Always(Box::new(TemporalExpr::Atom(
                dashprove_usl::ast::Expr::Bool(true),
            ))),
            fairness: vec![],
        });
        let ce = "State 1:\n  x = 0\nState 2:\n  x = 1\nState 3:\n  x = 2";
        let explanation = explain_counterexample(&prop, ce, &BackendId::TlaPlus);
        // 3 states means 2 steps
        assert!(
            explanation.summary.contains("2 steps"),
            "Multi-state trace should show step count"
        );
    }

    #[test]
    fn test_tlaplus_empty_trace_summary() {
        // empty trace: no states parsed
        let prop = Property::Temporal(Temporal {
            name: "test".to_string(),
            body: TemporalExpr::Always(Box::new(TemporalExpr::Atom(
                dashprove_usl::ast::Expr::Bool(true),
            ))),
            fairness: vec![],
        });
        let ce = "Some unrecognized output without state markers";
        let explanation = explain_counterexample(&prop, ce, &BackendId::TlaPlus);
        assert!(
            explanation.summary.contains("no trace available"),
            "Empty trace should mention no trace available"
        );
    }

    #[test]
    fn test_tlaplus_single_state_suggestion_mentions_init() {
        // trace.len() == 1 should suggest checking Init predicate
        let prop = Property::Temporal(Temporal {
            name: "test".to_string(),
            body: TemporalExpr::Always(Box::new(TemporalExpr::Atom(
                dashprove_usl::ast::Expr::Bool(true),
            ))),
            fairness: vec![],
        });
        let ce = "State 1:\n  x = 0";
        let explanation = explain_counterexample(&prop, ce, &BackendId::TlaPlus);
        let has_init_suggestion = explanation.suggestions.iter().any(|s| s.contains("Init"));
        assert!(
            has_init_suggestion,
            "Single state trace should suggest checking Init predicate"
        );
    }

    // Tests for parse_tlaplus_trace || vs && (line 225)
    #[test]
    fn test_tlaplus_parse_state_with_only_state_keyword() {
        // "State 1:" pattern - requires || to work correctly
        let trace_output = "State 1:\n  x = 0";
        let trace = parse_tlaplus_trace(trace_output);
        assert_eq!(trace.len(), 1, "Should parse State N: pattern");
    }

    #[test]
    fn test_tlaplus_parse_initial_state_keyword() {
        // "Initial state" pattern - tests the || part
        let trace_output = "Initial state:\n  x = 0";
        let trace = parse_tlaplus_trace(trace_output);
        assert_eq!(trace.len(), 1, "Should parse Initial state pattern");
    }

    #[test]
    fn test_tlaplus_parse_action_with_underscore_action_suffix() {
        // Tests the || condition in line 225: line without /\ but with _action
        let trace_output = "State 1:\n  x = 0\nState 2:\n  Next_action\n  x = 1";
        let trace = parse_tlaplus_trace(trace_output);
        assert_eq!(trace.len(), 2, "Should parse two states");
        // The second state should have an action detected via _action
        assert!(
            trace[1].action.is_some(),
            "Should detect action via _action suffix"
        );
    }

    // Tests for Kani property type extraction (line 272)
    #[test]
    fn test_kani_contract_name_used_in_summary() {
        let prop = Property::Contract(dashprove_usl::ast::Contract {
            type_path: vec!["module".to_string(), "unique_func".to_string()],
            params: vec![],
            return_type: None,
            requires: vec![],
            ensures: vec![],
            ensures_err: vec![],
            assigns: vec![],
            allocates: vec![],
            frees: vec![],
            terminates: None,
            decreases: None,
            behaviors: vec![],
            complete_behaviors: false,
            disjoint_behaviors: false,
        });
        let error = "general error";
        let explanation = explain_counterexample(&prop, error, &BackendId::Kani);
        assert!(
            explanation.summary.contains("module::unique_func"),
            "Contract type_path should appear in summary joined with ::"
        );
    }

    // Tests for Kani ExplanationKind::PreconditionViolation match arm (line 288)
    #[test]
    fn test_kani_precondition_violation_summary() {
        let prop = Property::Contract(dashprove_usl::ast::Contract {
            type_path: vec!["func".to_string()],
            params: vec![],
            return_type: None,
            requires: vec![],
            ensures: vec![],
            ensures_err: vec![],
            assigns: vec![],
            allocates: vec![],
            frees: vec![],
            terminates: None,
            decreases: None,
            behaviors: vec![],
            complete_behaviors: false,
            disjoint_behaviors: false,
        });
        let error = "assume failed"; // triggers PreconditionViolation
        let explanation = explain_counterexample(&prop, error, &BackendId::Kani);
        assert_eq!(explanation.kind, ExplanationKind::PreconditionViolation);
        assert!(
            explanation.summary.contains("Precondition"),
            "Summary should mention Precondition for assume failures"
        );
    }

    // Tests for Kani kind == comparisons for suggestions (lines 298, 303)
    #[test]
    fn test_kani_postcondition_has_postcondition_suggestions() {
        let prop = Property::Contract(dashprove_usl::ast::Contract {
            type_path: vec!["func".to_string()],
            params: vec![],
            return_type: None,
            requires: vec![],
            ensures: vec![],
            ensures_err: vec![],
            assigns: vec![],
            allocates: vec![],
            frees: vec![],
            terminates: None,
            decreases: None,
            behaviors: vec![],
            complete_behaviors: false,
            disjoint_behaviors: false,
        });
        let error = "assertion failed: result > 0";
        let explanation = explain_counterexample(&prop, error, &BackendId::Kani);
        let has_postcond_suggestion = explanation
            .suggestions
            .iter()
            .any(|s| s.contains("postcondition"));
        assert!(
            has_postcond_suggestion,
            "Postcondition violations should have postcondition-specific suggestions"
        );
    }

    #[test]
    fn test_kani_precondition_has_precondition_suggestions() {
        let prop = Property::Contract(dashprove_usl::ast::Contract {
            type_path: vec!["func".to_string()],
            params: vec![],
            return_type: None,
            requires: vec![],
            ensures: vec![],
            ensures_err: vec![],
            assigns: vec![],
            allocates: vec![],
            frees: vec![],
            terminates: None,
            decreases: None,
            behaviors: vec![],
            complete_behaviors: false,
            disjoint_behaviors: false,
        });
        let error = "assume failed";
        let explanation = explain_counterexample(&prop, error, &BackendId::Kani);
        let has_precond_suggestion = explanation
            .suggestions
            .iter()
            .any(|s| s.contains("precondition"));
        assert!(
            has_precond_suggestion,
            "Precondition violations should have precondition-specific suggestions"
        );
    }

    // Tests for parse_kani_values (lines 319, 324, 326)
    #[test]
    fn test_parse_kani_values_returns_bindings() {
        let output = "x = 42\ny = 100";
        let bindings = parse_kani_values(output);
        assert!(!bindings.is_empty(), "Should parse bindings from = format");
        assert_eq!(bindings.len(), 2);
    }

    #[test]
    fn test_parse_kani_values_concrete_value_format() {
        let output = "concrete value: x = 42";
        let bindings = parse_kani_values(output);
        assert!(!bindings.is_empty(), "Should parse concrete value format");
    }

    #[test]
    fn test_parse_kani_values_only_concrete_value_without_equals() {
        // Tests the || condition: line contains "concrete value:" but no " = "
        let output = "concrete value: something";
        let bindings = parse_kani_values(output);
        // This line has "concrete value:" but splitn(2, " = ") won't find 2 parts
        assert!(
            bindings.is_empty(),
            "Without = sign, should not create binding"
        );
    }

    // Tests for Alloy property type extraction (lines 345-347)
    #[test]
    fn test_alloy_invariant_name_used_in_summary() {
        let prop = Property::Invariant(Invariant {
            name: "unique_alloy_inv".to_string(),
            body: dashprove_usl::ast::Expr::Bool(true),
        });
        let ce = "sig A = {A0}";
        let explanation = explain_counterexample(&prop, ce, &BackendId::Alloy);
        assert!(
            explanation.summary.contains("unique_alloy_inv"),
            "Invariant name should appear in Alloy explanation summary"
        );
    }

    #[test]
    fn test_alloy_theorem_name_used_in_summary() {
        let prop = Property::Theorem(Theorem {
            name: "unique_alloy_thm".to_string(),
            body: dashprove_usl::ast::Expr::Bool(true),
        });
        let ce = "sig A = {A0}";
        let explanation = explain_counterexample(&prop, ce, &BackendId::Alloy);
        assert!(
            explanation.summary.contains("unique_alloy_thm"),
            "Theorem name should appear in Alloy explanation summary"
        );
    }

    // Tests for parse_alloy_instance (lines 376, 381, 383)
    #[test]
    fn test_parse_alloy_instance_returns_bindings() {
        let output = "sig Node = {Node0}\nedges = {Node0->Node1}";
        let bindings = parse_alloy_instance(output);
        assert!(!bindings.is_empty(), "Should parse Alloy instance bindings");
        assert_eq!(bindings.len(), 2);
    }

    #[test]
    fn test_parse_alloy_instance_skips_comments() {
        // Tests the && !line.starts_with("--") condition
        let output = "sig Node = {Node0}\n-- this = comment\nedges = {}";
        let bindings = parse_alloy_instance(output);
        // Should have 2 bindings, not 3 (comment line skipped)
        assert_eq!(
            bindings.len(),
            2,
            "Should skip comment lines starting with --"
        );
    }

    #[test]
    fn test_parse_alloy_instance_requires_equals() {
        // Tests that lines without = are skipped
        let output = "some text without equals\nsig Node = {Node0}";
        let bindings = parse_alloy_instance(output);
        assert_eq!(bindings.len(), 1, "Should only parse lines with =");
    }

    // Test for multi-step trace details (line 173: trace.len() > 1)
    #[test]
    fn test_tlaplus_multi_step_details_mention_action() {
        let prop = Property::Temporal(Temporal {
            name: "test".to_string(),
            body: TemporalExpr::Always(Box::new(TemporalExpr::Atom(
                dashprove_usl::ast::Expr::Bool(true),
            ))),
            fairness: vec![],
        });
        // Multi-step trace with action
        let ce = "State 1:\n  x = 0\nState 2:\n  /\\ MyAction\n  x = 1";
        let explanation = explain_counterexample(&prop, ce, &BackendId::TlaPlus);
        // With more than 1 step and an action, details should mention the action
        assert!(
            explanation.details.contains("MyAction") || explanation.details.contains("sequence"),
            "Multi-step trace details should mention action or sequence"
        );
    }

    // Test for trace.len() == 1 details (line 183: else branch)
    #[test]
    fn test_tlaplus_single_step_details_mention_initial_state() {
        let prop = Property::Temporal(Temporal {
            name: "test".to_string(),
            body: TemporalExpr::Always(Box::new(TemporalExpr::Atom(
                dashprove_usl::ast::Expr::Bool(true),
            ))),
            fairness: vec![],
        });
        let ce = "State 1:\n  x = 0";
        let explanation = explain_counterexample(&prop, ce, &BackendId::TlaPlus);
        assert!(
            explanation.details.contains("initial state"),
            "Single step trace details should mention initial state"
        );
    }
}

// ========== Kani proof harnesses ==========

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify ExplanationKind VariableAssignment variant
    #[kani::proof]
    fn verify_explanation_kind_variable_assignment() {
        let kind = ExplanationKind::VariableAssignment;
        kani::assert(
            matches!(kind, ExplanationKind::VariableAssignment),
            "VariableAssignment should match",
        );
    }

    /// Verify ExplanationKind StateTrace variant
    #[kani::proof]
    fn verify_explanation_kind_state_trace() {
        let kind = ExplanationKind::StateTrace;
        kani::assert(
            matches!(kind, ExplanationKind::StateTrace),
            "StateTrace should match",
        );
    }

    /// Verify ExplanationKind MissingCase variant
    #[kani::proof]
    fn verify_explanation_kind_missing_case() {
        let kind = ExplanationKind::MissingCase;
        kani::assert(
            matches!(kind, ExplanationKind::MissingCase),
            "MissingCase should match",
        );
    }

    /// Verify ExplanationKind PreconditionViolation variant
    #[kani::proof]
    fn verify_explanation_kind_precondition() {
        let kind = ExplanationKind::PreconditionViolation;
        kani::assert(
            matches!(kind, ExplanationKind::PreconditionViolation),
            "PreconditionViolation should match",
        );
    }

    /// Verify ExplanationKind PostconditionViolation variant
    #[kani::proof]
    fn verify_explanation_kind_postcondition() {
        let kind = ExplanationKind::PostconditionViolation;
        kani::assert(
            matches!(kind, ExplanationKind::PostconditionViolation),
            "PostconditionViolation should match",
        );
    }

    /// Verify ExplanationKind General variant
    #[kani::proof]
    fn verify_explanation_kind_general() {
        let kind = ExplanationKind::General;
        kani::assert(
            matches!(kind, ExplanationKind::General),
            "General should match",
        );
    }

    /// Verify Binding creation stores name correctly
    #[kani::proof]
    fn verify_binding_stores_name() {
        let binding = Binding {
            name: String::from("x"),
            value: String::from("42"),
            ty: None,
        };
        kani::assert(!binding.name.is_empty(), "name should be stored");
    }

    /// Verify Binding creation stores value correctly
    #[kani::proof]
    fn verify_binding_stores_value() {
        let binding = Binding {
            name: String::from("x"),
            value: String::from("42"),
            ty: None,
        };
        kani::assert(!binding.value.is_empty(), "value should be stored");
    }

    /// Verify Binding with type annotation
    #[kani::proof]
    fn verify_binding_with_type() {
        let binding = Binding {
            name: String::from("x"),
            value: String::from("42"),
            ty: Some(String::from("i32")),
        };
        kani::assert(binding.ty.is_some(), "ty should be Some");
    }

    /// Verify TraceStep stores step number
    #[kani::proof]
    fn verify_trace_step_stores_number() {
        let step = TraceStep {
            step_number: 5,
            action: None,
            state: vec![],
        };
        kani::assert(step.step_number == 5, "step_number should be preserved");
    }

    /// Verify TraceStep with action
    #[kani::proof]
    fn verify_trace_step_with_action() {
        let step = TraceStep {
            step_number: 0,
            action: Some(String::from("Next")),
            state: vec![],
        };
        kani::assert(step.action.is_some(), "action should be Some");
    }

    /// Verify TraceStep with empty state
    #[kani::proof]
    fn verify_trace_step_empty_state() {
        let step = TraceStep {
            step_number: 0,
            action: None,
            state: vec![],
        };
        kani::assert(step.state.is_empty(), "state should be empty");
    }

    /// Verify CounterexampleExplanation stores kind
    #[kani::proof]
    fn verify_explanation_stores_kind() {
        let explanation = CounterexampleExplanation {
            kind: ExplanationKind::General,
            summary: String::from("test"),
            details: String::from("details"),
            bindings: vec![],
            trace: vec![],
            suggestions: vec![],
        };
        kani::assert(
            matches!(explanation.kind, ExplanationKind::General),
            "kind should be preserved",
        );
    }

    /// Verify CounterexampleExplanation stores summary
    #[kani::proof]
    fn verify_explanation_stores_summary() {
        let explanation = CounterexampleExplanation {
            kind: ExplanationKind::General,
            summary: String::from("test summary"),
            details: String::from("details"),
            bindings: vec![],
            trace: vec![],
            suggestions: vec![],
        };
        kani::assert(!explanation.summary.is_empty(), "summary should be stored");
    }

    /// Verify CounterexampleExplanation empty bindings
    #[kani::proof]
    fn verify_explanation_empty_bindings() {
        let explanation = CounterexampleExplanation {
            kind: ExplanationKind::General,
            summary: String::from("test"),
            details: String::from("details"),
            bindings: vec![],
            trace: vec![],
            suggestions: vec![],
        };
        kani::assert(explanation.bindings.is_empty(), "bindings should be empty");
    }

    /// Verify CounterexampleExplanation empty trace
    #[kani::proof]
    fn verify_explanation_empty_trace() {
        let explanation = CounterexampleExplanation {
            kind: ExplanationKind::General,
            summary: String::from("test"),
            details: String::from("details"),
            bindings: vec![],
            trace: vec![],
            suggestions: vec![],
        };
        kani::assert(explanation.trace.is_empty(), "trace should be empty");
    }
}
