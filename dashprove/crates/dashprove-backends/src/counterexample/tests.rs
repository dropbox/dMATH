//! Tests for counterexample types and analysis

use super::*;
use crate::traits::{html_download_buttons, html_download_buttons_with_container};
use std::collections::HashMap;

#[test]
fn download_buttons_use_unique_function_names() {
    let first = html_download_buttons("graph TB; A-->B;", None);
    let second = html_download_buttons("graph TB; C-->D;", Some("digraph { A -> B }"));

    let extract_id = |html: &str, marker: &str| -> String {
        html.split(marker)
            .nth(1)
            .map(|s| {
                s.chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect::<String>()
            })
            .expect("marker not found")
    };

    let first_mermaid = extract_id(&first, "downloadMermaid_");
    let first_svg = extract_id(&first, "downloadSvg_");
    let second_mermaid = extract_id(&second, "downloadMermaid_");
    let second_svg = extract_id(&second, "downloadSvg_");
    let second_dot = extract_id(&second, "downloadDot_");

    // All function IDs within same call should match
    assert!(!first_mermaid.is_empty());
    assert!(!first_svg.is_empty());
    assert_eq!(first_mermaid, first_svg);

    assert!(!second_mermaid.is_empty());
    assert!(!second_svg.is_empty());
    assert_eq!(second_mermaid, second_svg);
    assert_eq!(second_mermaid, second_dot);

    // Different calls should have different IDs
    assert_ne!(first_mermaid, second_mermaid);
}

#[test]
fn download_buttons_include_svg_button() {
    let html = html_download_buttons("graph TB; A-->B;", None);
    assert!(html.contains("Download SVG"));
    assert!(html.contains("downloadSvg_"));
    assert!(html.contains("XMLSerializer"));
    assert!(html.contains("image/svg+xml"));
}

#[test]
fn download_buttons_with_container_uses_custom_selector() {
    let html = html_download_buttons_with_container("graph TB; A-->B;", None, Some("my-container"));
    assert!(html.contains("#my-container svg"));
}

#[test]
fn trace_state_diff_detects_changes() {
    let mut state1 = TraceState::new(1);
    state1.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );
    state1.variables.insert(
        "y".to_string(),
        CounterexampleValue::Int {
            value: 10,
            type_hint: None,
        },
    );

    let mut state2 = TraceState::new(2);
    state2.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 2,
            type_hint: None,
        },
    ); // changed
    state2.variables.insert(
        "y".to_string(),
        CounterexampleValue::Int {
            value: 10,
            type_hint: None,
        },
    ); // unchanged
    state2.variables.insert(
        "z".to_string(),
        CounterexampleValue::Int {
            value: 5,
            type_hint: None,
        },
    ); // new

    let diffs = state2.diff_from(&state1);

    // x changed from 1 to 2
    assert!(diffs.contains_key("x"));
    if let Some((Some(old), new)) = diffs.get("x") {
        assert_eq!(
            *old,
            CounterexampleValue::Int {
                value: 1,
                type_hint: None
            }
        );
        assert_eq!(
            *new,
            CounterexampleValue::Int {
                value: 2,
                type_hint: None
            }
        );
    } else {
        panic!("Expected x to have old and new values");
    }

    // y unchanged - should NOT be in diffs
    assert!(!diffs.contains_key("y"));

    // z is new
    assert!(diffs.contains_key("z"));
    if let Some((old, new)) = diffs.get("z") {
        assert!(old.is_none());
        assert_eq!(
            *new,
            CounterexampleValue::Int {
                value: 5,
                type_hint: None
            }
        );
    } else {
        panic!("Expected z to be new");
    }
}

#[test]
fn trace_state_display() {
    let mut state = TraceState::new(1);
    state.action = Some("Initial predicate".to_string());
    state.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 42,
            type_hint: None,
        },
    );
    state
        .variables
        .insert("active".to_string(), CounterexampleValue::Bool(true));

    let display = format!("{}", state);
    assert!(display.contains("State 1"));
    assert!(display.contains("Initial predicate"));
    assert!(display.contains("x = 42"));
    assert!(display.contains("active = true"));
}

#[test]
fn structured_counterexample_format_detailed() {
    let mut ce = StructuredCounterexample::new();

    // Add failed check
    ce.failed_checks.push(FailedCheck {
        check_id: "inv1".to_string(),
        description: "Invariant violated".to_string(),
        location: Some(SourceLocation {
            file: "spec.tla".to_string(),
            line: 10,
            column: None,
        }),
        function: None,
    });

    // Add witness
    ce.witness.insert(
        "n".to_string(),
        CounterexampleValue::Int {
            value: 5,
            type_hint: None,
        },
    );

    // Add trace
    let mut state1 = TraceState::new(1);
    state1.action = Some("Init".to_string());
    state1.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 0,
            type_hint: None,
        },
    );
    ce.trace.push(state1);

    let mut state2 = TraceState::new(2);
    state2.action = Some("Next".to_string());
    state2.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );
    ce.trace.push(state2);

    let detailed = ce.format_detailed();

    // Check sections exist
    assert!(detailed.contains("=== Failed Checks ==="));
    assert!(detailed.contains("Invariant violated"));
    assert!(detailed.contains("spec.tla:10"));

    assert!(detailed.contains("=== Witness Values ==="));
    assert!(detailed.contains("n = 5"));

    assert!(detailed.contains("=== Counterexample Trace (2 states) ==="));
    assert!(detailed.contains("State 1"));
    assert!(detailed.contains("State 2"));
    assert!(detailed.contains("Init"));
    assert!(detailed.contains("Next"));
}

#[test]
fn structured_counterexample_trace_diffs() {
    let mut ce = StructuredCounterexample::new();

    let mut state1 = TraceState::new(1);
    state1.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 0,
            type_hint: None,
        },
    );
    state1.variables.insert(
        "y".to_string(),
        CounterexampleValue::Int {
            value: 0,
            type_hint: None,
        },
    );
    ce.trace.push(state1);

    let mut state2 = TraceState::new(2);
    state2.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );
    state2.variables.insert(
        "y".to_string(),
        CounterexampleValue::Int {
            value: 0,
            type_hint: None,
        },
    );
    ce.trace.push(state2);

    let mut state3 = TraceState::new(3);
    state3.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );
    state3.variables.insert(
        "y".to_string(),
        CounterexampleValue::Int {
            value: 5,
            type_hint: None,
        },
    );
    ce.trace.push(state3);

    let diffs = ce.trace_diffs();

    // Should have 2 diffs (state 2 vs 1, state 3 vs 2)
    assert_eq!(diffs.len(), 2);

    // State 2: x changed
    let (state_num, changes) = &diffs[0];
    assert_eq!(*state_num, 2);
    assert!(changes.contains_key("x"));
    assert!(!changes.contains_key("y"));

    // State 3: y changed
    let (state_num, changes) = &diffs[1];
    assert_eq!(*state_num, 3);
    assert!(!changes.contains_key("x"));
    assert!(changes.contains_key("y"));
}

#[test]
fn format_trace_with_diffs_shows_changes() {
    let mut ce = StructuredCounterexample::new();

    let mut state1 = TraceState::new(1);
    state1.action = Some("Init".to_string());
    state1.variables.insert(
        "counter".to_string(),
        CounterexampleValue::Int {
            value: 0,
            type_hint: None,
        },
    );
    ce.trace.push(state1);

    let mut state2 = TraceState::new(2);
    state2.action = Some("Increment".to_string());
    state2.variables.insert(
        "counter".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );
    ce.trace.push(state2);

    let formatted = ce.format_trace_with_diffs();

    // Initial state shows all variables
    assert!(formatted.contains("Initial state:"));
    assert!(formatted.contains("counter = 0"));

    // Second state shows changes
    assert!(formatted.contains("Changes:"));
    assert!(formatted.contains("counter : 0 -> 1"));
}

// ==================== JSON Serialization Tests ====================

#[test]
fn counterexample_json_roundtrip() {
    let mut ce = StructuredCounterexample::new();
    ce.witness.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 42,
            type_hint: Some("i32".to_string()),
        },
    );
    ce.witness
        .insert("flag".to_string(), CounterexampleValue::Bool(true));
    ce.failed_checks.push(FailedCheck {
        check_id: "check1".to_string(),
        description: "assertion failed".to_string(),
        location: Some(SourceLocation {
            file: "test.rs".to_string(),
            line: 10,
            column: Some(5),
        }),
        function: Some("my_func".to_string()),
    });

    // Serialize
    let json = ce.to_json().expect("serialize failed");
    assert!(json.contains("42"));
    assert!(json.contains("assertion failed"));

    // Deserialize
    let restored = StructuredCounterexample::from_json(&json).expect("deserialize failed");
    assert_eq!(restored.witness.get("x"), ce.witness.get("x"));
    assert_eq!(restored.failed_checks.len(), 1);
    assert_eq!(restored.failed_checks[0].check_id, "check1");
}

#[test]
fn counterexample_json_pretty() {
    let mut ce = StructuredCounterexample::new();
    ce.witness.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );

    let json = ce.to_json_pretty().expect("pretty serialize failed");
    // Pretty JSON should have newlines and indentation
    assert!(json.contains('\n'));
    assert!(json.contains("  "));
}

#[test]
fn counterexample_to_json_value() {
    let mut ce = StructuredCounterexample::new();
    ce.witness.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 5,
            type_hint: None,
        },
    );

    let value = ce.to_json_value().expect("to_value failed");
    assert!(value.is_object());
    assert!(value.get("witness").is_some());
}

// ==================== Counterexample Comparison Tests ====================

#[test]
fn counterexample_diff_equivalent() {
    let mut ce1 = StructuredCounterexample::new();
    ce1.witness.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );

    let mut ce2 = StructuredCounterexample::new();
    ce2.witness.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );

    let diff = ce1.diff(&ce2);
    assert!(diff.is_equivalent());
    assert_eq!(diff.summary(), "Counterexamples are equivalent");
    assert!(ce1.is_equivalent_to(&ce2));
}

#[test]
fn counterexample_diff_value_differences() {
    let mut ce1 = StructuredCounterexample::new();
    ce1.witness.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );

    let mut ce2 = StructuredCounterexample::new();
    ce2.witness.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 2,
            type_hint: None,
        },
    );

    let diff = ce1.diff(&ce2);
    assert!(!diff.is_equivalent());
    assert!(diff.value_differences.contains_key("x"));
    assert!(diff.summary().contains("1 variables differ"));
}

#[test]
fn counterexample_diff_only_in_one() {
    let mut ce1 = StructuredCounterexample::new();
    ce1.witness.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );
    ce1.witness.insert(
        "y".to_string(),
        CounterexampleValue::Int {
            value: 2,
            type_hint: None,
        },
    );

    let mut ce2 = StructuredCounterexample::new();
    ce2.witness.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );
    ce2.witness.insert(
        "z".to_string(),
        CounterexampleValue::Int {
            value: 3,
            type_hint: None,
        },
    );

    let diff = ce1.diff(&ce2);
    assert!(!diff.is_equivalent());
    assert!(diff.only_in_first.contains_key("y"));
    assert!(diff.only_in_second.contains_key("z"));
    assert!(diff.summary().contains("only in first"));
    assert!(diff.summary().contains("only in second"));
}

#[test]
fn counterexample_diff_trace_length() {
    let mut ce1 = StructuredCounterexample::new();
    ce1.trace.push(TraceState::new(1));
    ce1.trace.push(TraceState::new(2));

    let mut ce2 = StructuredCounterexample::new();
    ce2.trace.push(TraceState::new(1));

    let diff = ce1.diff(&ce2);
    assert!(!diff.is_equivalent());
    assert_eq!(diff.trace_length_diff, Some((2, 1)));
    assert!(diff.summary().contains("trace lengths differ"));
}

#[test]
fn counterexample_diff_failed_checks() {
    let mut ce1 = StructuredCounterexample::new();
    ce1.failed_checks.push(FailedCheck {
        check_id: "check1".to_string(),
        description: "first".to_string(),
        location: None,
        function: None,
    });

    let mut ce2 = StructuredCounterexample::new();
    ce2.failed_checks.push(FailedCheck {
        check_id: "check2".to_string(),
        description: "second".to_string(),
        location: None,
        function: None,
    });

    let diff = ce1.diff(&ce2);
    assert!(!diff.is_equivalent());
    assert_eq!(diff.checks_only_in_first.len(), 1);
    assert_eq!(diff.checks_only_in_second.len(), 1);
    assert!(diff.summary().contains("failed checks differ"));
}

// ==================== Trace Filtering Tests ====================

fn create_test_trace() -> StructuredCounterexample {
    let mut ce = StructuredCounterexample::new();

    let mut state1 = TraceState::new(1);
    state1.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 0,
            type_hint: None,
        },
    );
    state1.variables.insert(
        "y".to_string(),
        CounterexampleValue::Int {
            value: 0,
            type_hint: None,
        },
    );
    state1.variables.insert(
        "debug".to_string(),
        CounterexampleValue::String("init".to_string()),
    );
    ce.trace.push(state1);

    let mut state2 = TraceState::new(2);
    state2.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );
    state2.variables.insert(
        "y".to_string(),
        CounterexampleValue::Int {
            value: 0,
            type_hint: None,
        },
    );
    state2.variables.insert(
        "debug".to_string(),
        CounterexampleValue::String("step1".to_string()),
    );
    ce.trace.push(state2);

    let mut state3 = TraceState::new(3);
    state3.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );
    state3.variables.insert(
        "y".to_string(),
        CounterexampleValue::Int {
            value: 5,
            type_hint: None,
        },
    );
    state3.variables.insert(
        "debug".to_string(),
        CounterexampleValue::String("step2".to_string()),
    );
    ce.trace.push(state3);

    ce
}

#[test]
fn filter_trace_include_only() {
    let ce = create_test_trace();
    let options = TraceFilterOptions::only(vec!["x".to_string()]);

    let filtered = ce.filter_trace(&options);
    assert_eq!(filtered.len(), 3);

    // Each state should only have x
    for state in &filtered {
        assert!(state.variables.contains_key("x"));
        assert!(!state.variables.contains_key("y"));
        assert!(!state.variables.contains_key("debug"));
    }
}

#[test]
fn filter_trace_exclude() {
    let ce = create_test_trace();
    let options = TraceFilterOptions::excluding(vec!["debug".to_string()]);

    let filtered = ce.filter_trace(&options);
    assert_eq!(filtered.len(), 3);

    // No state should have debug
    for state in &filtered {
        assert!(state.variables.contains_key("x"));
        assert!(state.variables.contains_key("y"));
        assert!(!state.variables.contains_key("debug"));
    }
}

#[test]
fn filter_trace_changed_variables() {
    let ce = create_test_trace();
    let options = TraceFilterOptions::changed(vec!["y".to_string()]);

    let filtered = ce.filter_trace(&options);
    // Should include: state 1 (initial), state 3 (y changed)
    // State 2 excluded because y didn't change
    assert_eq!(filtered.len(), 2);
    assert_eq!(filtered[0].state_num, 1);
    assert_eq!(filtered[1].state_num, 3);
}

#[test]
fn filter_trace_skip_unchanged() {
    let mut ce = StructuredCounterexample::new();

    let mut state1 = TraceState::new(1);
    state1.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 0,
            type_hint: None,
        },
    );
    ce.trace.push(state1);

    let mut state2 = TraceState::new(2);
    state2.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 0,
            type_hint: None,
        },
    ); // no change
    ce.trace.push(state2);

    let mut state3 = TraceState::new(3);
    state3.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    ); // changed
    ce.trace.push(state3);

    let options = TraceFilterOptions {
        skip_unchanged_states: true,
        ..Default::default()
    };

    let filtered = ce.filter_trace(&options);
    // Should include: state 1 (initial), state 3 (x changed)
    // State 2 excluded because nothing changed
    assert_eq!(filtered.len(), 2);
    assert_eq!(filtered[0].state_num, 1);
    assert_eq!(filtered[1].state_num, 3);
}

#[test]
fn filter_trace_max_states() {
    let ce = create_test_trace();
    let options = TraceFilterOptions {
        max_states: Some(2),
        ..Default::default()
    };

    let filtered = ce.filter_trace(&options);
    assert_eq!(filtered.len(), 2);
}

#[test]
fn with_filtered_trace_preserves_other_fields() {
    let mut ce = create_test_trace();
    ce.witness.insert(
        "n".to_string(),
        CounterexampleValue::Int {
            value: 10,
            type_hint: None,
        },
    );
    ce.failed_checks.push(FailedCheck {
        check_id: "inv".to_string(),
        description: "failed".to_string(),
        location: None,
        function: None,
    });

    let options = TraceFilterOptions::only(vec!["x".to_string()]);
    let filtered = ce.with_filtered_trace(&options);

    // Other fields preserved
    assert_eq!(filtered.witness.len(), 1);
    assert_eq!(filtered.failed_checks.len(), 1);

    // Trace is filtered
    for state in &filtered.trace {
        assert!(!state.variables.contains_key("y"));
    }
}

#[test]
fn trace_variables_returns_all() {
    let ce = create_test_trace();
    let vars = ce.trace_variables();
    assert_eq!(vars, vec!["debug", "x", "y"]);
}

#[test]
fn changing_variables_detects_changes() {
    let ce = create_test_trace();
    let changing = ce.changing_variables();
    // x changes in state 2, y changes in state 3, debug changes in states 2 and 3
    assert!(changing.contains(&"x".to_string()));
    assert!(changing.contains(&"y".to_string()));
    assert!(changing.contains(&"debug".to_string()));
}

#[test]
fn changing_variables_excludes_constants() {
    let mut ce = StructuredCounterexample::new();

    let mut state1 = TraceState::new(1);
    state1.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 0,
            type_hint: None,
        },
    );
    state1.variables.insert(
        "const".to_string(),
        CounterexampleValue::Int {
            value: 42,
            type_hint: None,
        },
    );
    ce.trace.push(state1);

    let mut state2 = TraceState::new(2);
    state2.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );
    state2.variables.insert(
        "const".to_string(),
        CounterexampleValue::Int {
            value: 42,
            type_hint: None,
        },
    ); // same
    ce.trace.push(state2);

    let changing = ce.changing_variables();
    assert!(changing.contains(&"x".to_string()));
    assert!(!changing.contains(&"const".to_string()));
}

// ==================== State-Level Comparison Tests ====================

#[test]
fn trace_diff_identical_traces() {
    let mut ce1 = StructuredCounterexample::new();
    let mut state = TraceState::new(1);
    state.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );
    ce1.trace.push(state);

    let ce2 = ce1.clone();

    let trace_diff = ce1.diff_traces(&ce2);
    assert!(trace_diff.is_equivalent());
    assert_eq!(trace_diff.identical_states, vec![1]);
    assert!(trace_diff.state_diffs.is_empty());
}

#[test]
fn trace_diff_different_values() {
    let mut ce1 = StructuredCounterexample::new();
    let mut state1 = TraceState::new(1);
    state1.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );
    ce1.trace.push(state1);

    let mut ce2 = StructuredCounterexample::new();
    let mut state2 = TraceState::new(1);
    state2.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 2,
            type_hint: None,
        },
    );
    ce2.trace.push(state2);

    let trace_diff = ce1.diff_traces(&ce2);
    assert!(!trace_diff.is_equivalent());
    assert!(trace_diff.identical_states.is_empty());
    assert!(trace_diff.state_diffs.contains_key(&1));
    assert!(trace_diff
        .state_diffs
        .get(&1)
        .unwrap()
        .value_diffs
        .contains_key("x"));
}

#[test]
fn trace_diff_mismatched_states() {
    let mut ce1 = StructuredCounterexample::new();
    ce1.trace.push(TraceState::new(1));
    ce1.trace.push(TraceState::new(2));
    ce1.trace.push(TraceState::new(3));

    let mut ce2 = StructuredCounterexample::new();
    ce2.trace.push(TraceState::new(1));
    ce2.trace.push(TraceState::new(4)); // different state

    let trace_diff = ce1.diff_traces(&ce2);
    assert!(!trace_diff.is_equivalent());
    assert_eq!(trace_diff.states_only_in_first, vec![2, 3]);
    assert_eq!(trace_diff.states_only_in_second, vec![4]);
    assert!(trace_diff.summary().contains("states only in first"));
}

#[test]
fn diff_detailed_includes_trace_diff() {
    let mut ce1 = StructuredCounterexample::new();
    let mut state1 = TraceState::new(1);
    state1.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );
    ce1.trace.push(state1);

    let mut ce2 = StructuredCounterexample::new();
    let mut state2 = TraceState::new(1);
    state2.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 2,
            type_hint: None,
        },
    );
    ce2.trace.push(state2);

    let diff = ce1.diff_detailed(&ce2);
    assert!(diff.trace_diff.is_some());
    assert!(!diff.trace_diff.unwrap().is_equivalent());
}

#[test]
fn align_traces_different_lengths() {
    let mut ce1 = StructuredCounterexample::new();
    ce1.trace.push(TraceState::new(1));
    ce1.trace.push(TraceState::new(2));

    let mut ce2 = StructuredCounterexample::new();
    ce2.trace.push(TraceState::new(1));
    ce2.trace.push(TraceState::new(3));
    ce2.trace.push(TraceState::new(4));

    let aligned = ce1.align_traces(&ce2);

    // Should have states 1, 2, 3, 4
    assert_eq!(aligned.len(), 4);

    // State 1: both have it
    assert_eq!(aligned[0].0, 1);
    assert!(aligned[0].1.is_some());
    assert!(aligned[0].2.is_some());

    // State 2: only in first
    assert_eq!(aligned[1].0, 2);
    assert!(aligned[1].1.is_some());
    assert!(aligned[1].2.is_none());

    // State 3: only in second
    assert_eq!(aligned[2].0, 3);
    assert!(aligned[2].1.is_none());
    assert!(aligned[2].2.is_some());
}

#[test]
fn format_aligned_traces_shows_differences() {
    let mut ce1 = StructuredCounterexample::new();
    let mut state1 = TraceState::new(1);
    state1.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );
    ce1.trace.push(state1);

    let mut ce2 = StructuredCounterexample::new();
    let mut state2 = TraceState::new(1);
    state2.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 2,
            type_hint: None,
        },
    );
    ce2.trace.push(state2);

    let formatted = ce1.format_aligned_traces(&ce2);
    assert!(formatted.contains("=== State 1 ==="));
    assert!(formatted.contains("Value differences"));
    assert!(formatted.contains("x :"));
}

// ==================== Trace Interleaving Tests ====================

#[test]
fn detect_interleaving_from_actions() {
    let mut ce = StructuredCounterexample::new();

    let mut s1 = TraceState::new(1);
    s1.action = Some("WorkerA: init".to_string());
    ce.trace.push(s1);

    let mut s2 = TraceState::new(2);
    s2.action = Some("WorkerB: receive".to_string());
    ce.trace.push(s2);

    let mut s3 = TraceState::new(3);
    s3.action = Some("WorkerA: compute".to_string());
    ce.trace.push(s3);

    let mut s4 = TraceState::new(4);
    s4.action = Some("WorkerB: commit".to_string());
    ce.trace.push(s4);

    let interleaving = ce.detect_interleavings();
    assert_eq!(interleaving.lanes.len(), 2);
    assert!(interleaving
        .lanes
        .iter()
        .any(|l| l.actor == "WorkerA" && l.states.len() == 2));
    assert!(interleaving
        .lanes
        .iter()
        .any(|l| l.actor == "WorkerB" && l.states.len() == 2));
    assert_eq!(interleaving.unassigned_states.len(), 0);
    assert_eq!(interleaving.lane_switches(), 3);
    assert!((interleaving.coverage() - 1.0).abs() < f64::EPSILON);
}

#[test]
fn detect_interleaving_from_variable_prefixes() {
    let mut ce = StructuredCounterexample::new();

    let mut s1 = TraceState::new(1);
    s1.variables.insert(
        "proc1.x".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );
    ce.trace.push(s1);

    let mut s2 = TraceState::new(2);
    s2.variables.insert(
        "proc2.y".to_string(),
        CounterexampleValue::Int {
            value: 2,
            type_hint: None,
        },
    );
    ce.trace.push(s2);

    let mut s3 = TraceState::new(3);
    s3.variables.insert(
        "proc1.z".to_string(),
        CounterexampleValue::Int {
            value: 3,
            type_hint: None,
        },
    );
    ce.trace.push(s3);

    let interleaving = ce.detect_interleavings();
    assert_eq!(interleaving.lanes.len(), 2);
    assert!(interleaving
        .lanes
        .iter()
        .any(|l| l.actor == "proc1" && l.states.len() == 2));
    assert!(interleaving
        .lanes
        .iter()
        .any(|l| l.actor == "proc2" && l.states.len() == 1));
    assert!(interleaving.unassigned_states.is_empty());
}

#[test]
fn mermaid_sequence_diagram_includes_participants() {
    let mut ce = StructuredCounterexample::new();

    let mut s1 = TraceState::new(1);
    s1.action = Some("A: start".to_string());
    ce.trace.push(s1);

    let mut s2 = TraceState::new(2);
    s2.action = Some("B: continue".to_string());
    ce.trace.push(s2);

    let interleaving = ce.detect_interleavings();
    let diagram = interleaving.to_mermaid_sequence_diagram();

    assert!(diagram.contains("participant A as A"));
    assert!(diagram.contains("participant B as B"));
    assert!(diagram.contains("A->>A: [1] A: start"));
    assert!(diagram.contains("B->>B: [2] B: continue"));
}

#[test]
fn trace_interleaving_to_mermaid_basic() {
    let mut ce = StructuredCounterexample::new();

    let mut s1 = TraceState::new(1);
    s1.action = Some("A: start".to_string());
    s1.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );
    ce.trace.push(s1);

    let mut s2 = TraceState::new(2);
    s2.action = Some("B: process".to_string());
    s2.variables.insert(
        "y".to_string(),
        CounterexampleValue::Int {
            value: 2,
            type_hint: None,
        },
    );
    ce.trace.push(s2);

    let mut s3 = TraceState::new(3);
    s3.action = Some("A: finish".to_string());
    s3.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 3,
            type_hint: None,
        },
    );
    ce.trace.push(s3);

    let interleaving = ce.detect_interleavings();
    let mermaid = interleaving.to_mermaid();

    // Check flowchart header
    assert!(mermaid.contains("flowchart TB"));
    // Check subgraphs for lanes
    assert!(mermaid.contains("subgraph A[\"A\"]"));
    assert!(mermaid.contains("subgraph B[\"B\"]"));
    // Check node creation
    assert!(mermaid.contains("A_0[\""));
    assert!(mermaid.contains("A_1[\""));
    assert!(mermaid.contains("B_0[\""));
    // Check edge with label
    assert!(mermaid.contains("A_0 -->"));
    // Check style definitions
    assert!(mermaid.contains("style A fill:"));
    assert!(mermaid.contains("style B fill:"));
}

#[test]
fn trace_interleaving_to_mermaid_empty() {
    let interleaving = TraceInterleaving {
        lanes: vec![],
        unassigned_states: vec![],
        assignments: vec![],
        original_trace: vec![],
    };

    let mermaid = interleaving.to_mermaid();
    assert!(mermaid.contains("flowchart TB"));
    assert!(mermaid.contains("Empty trace"));
}

#[test]
fn trace_interleaving_to_mermaid_with_unassigned() {
    let mut ce = StructuredCounterexample::new();

    let mut s1 = TraceState::new(1);
    s1.action = Some("A: start".to_string());
    ce.trace.push(s1);

    let mut s2 = TraceState::new(2);
    // No clear actor prefix - will be unassigned
    s2.action = Some("unknown action".to_string());
    ce.trace.push(s2);

    let interleaving = ce.detect_interleavings();
    let mermaid = interleaving.to_mermaid();

    // Should have unassigned subgraph
    if !interleaving.unassigned_states.is_empty() {
        assert!(mermaid.contains("subgraph Unassigned[\"Unassigned States\"]"));
        assert!(mermaid.contains("style Unassigned fill:#f5f5f5"));
    }
}

#[test]
fn trace_interleaving_to_html_basic() {
    let mut ce = StructuredCounterexample::new();

    let mut s1 = TraceState::new(1);
    s1.action = Some("A: start".to_string());
    ce.trace.push(s1);

    let mut s2 = TraceState::new(2);
    s2.action = Some("B: process".to_string());
    ce.trace.push(s2);

    let interleaving = ce.detect_interleavings();
    let html = interleaving.to_html(Some("Test Interleaving"));

    // Check HTML structure
    assert!(html.contains("<!DOCTYPE html>"));
    assert!(html.contains("<title>Test Interleaving</title>"));
    assert!(html.contains("mermaid@10"));
    // Check summary stats
    assert!(html.contains("<strong>Lanes:</strong>"));
    assert!(html.contains("<strong>Total States:</strong>"));
    assert!(html.contains("<strong>Coverage:</strong>"));
    assert!(html.contains("<strong>Lane Switches:</strong>"));
    // Check lanes table
    assert!(html.contains("<table class=\"lanes-table\">"));
    // Check both diagram types
    assert!(html.contains("flowchart TB"));
    assert!(html.contains("sequenceDiagram"));
    // Check tabs for switching views
    assert!(html.contains("showDiagram('flowchart')"));
    assert!(html.contains("showDiagram('sequence')"));
    assert_eq!(html.matches("Download Mermaid").count(), 2);
    assert!(html.contains("download-buttons"));
}

#[test]
fn trace_interleaving_to_html_default_title() {
    let interleaving = TraceInterleaving {
        lanes: vec![],
        unassigned_states: vec![],
        assignments: vec![],
        original_trace: vec![],
    };

    let html = interleaving.to_html(None);
    assert!(html.contains("<title>Trace Interleaving Visualization</title>"));
}

#[test]
fn trace_interleaving_to_html_with_multiple_lanes() {
    let mut ce = StructuredCounterexample::new();

    let mut s1 = TraceState::new(1);
    s1.action = Some("A: init".to_string());
    ce.trace.push(s1);

    let mut s2 = TraceState::new(2);
    s2.action = Some("B: process".to_string());
    ce.trace.push(s2);

    let mut s3 = TraceState::new(3);
    s3.action = Some("C: finalize".to_string());
    ce.trace.push(s3);

    let interleaving = ce.detect_interleavings();
    let html = interleaving.to_html(None);

    // Should list all three actors in the table
    assert!(html.contains("<strong>A</strong>"));
    assert!(html.contains("<strong>B</strong>"));
    assert!(html.contains("<strong>C</strong>"));
    // Should have legend items
    assert!(html.contains("Lane 1"));
    assert!(html.contains("Lane 2"));
    assert!(html.contains("Lane 3"));
}

// ==================== Regex Filtering Tests ====================

#[test]
fn filter_trace_regex_include() {
    let ce = create_test_trace();
    // Include only variables starting with 'x' or 'y'
    let options = TraceFilterOptions::matching("^[xy]$");
    let filtered = ce.filter_trace(&options);

    for state in &filtered {
        assert!(state.variables.contains_key("x") || state.variables.contains_key("y"));
        assert!(!state.variables.contains_key("debug"));
    }
}

#[test]
fn filter_trace_regex_exclude() {
    let ce = create_test_trace();
    // Exclude variables containing 'debug'
    let options = TraceFilterOptions::excluding_pattern("debug");
    let filtered = ce.filter_trace(&options);

    for state in &filtered {
        assert!(!state.variables.contains_key("debug"));
        assert!(state.variables.contains_key("x"));
        assert!(state.variables.contains_key("y"));
    }
}

// ==================== Minimization Tests ====================

#[test]
fn minimize_removes_constants() {
    let mut ce = StructuredCounterexample::new();

    let mut state1 = TraceState::new(1);
    state1.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 0,
            type_hint: None,
        },
    );
    state1.variables.insert(
        "const".to_string(),
        CounterexampleValue::Int {
            value: 42,
            type_hint: None,
        },
    );
    ce.trace.push(state1);

    let mut state2 = TraceState::new(2);
    state2.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );
    state2.variables.insert(
        "const".to_string(),
        CounterexampleValue::Int {
            value: 42,
            type_hint: None,
        },
    );
    ce.trace.push(state2);

    let minimized = ce.minimize();
    assert!(minimized.minimized);

    // const should be removed as it doesn't change
    for state in &minimized.trace {
        assert!(!state.variables.contains_key("const"));
        assert!(state.variables.contains_key("x"));
    }
}

#[test]
fn minimize_removes_unchanged_states() {
    let mut ce = StructuredCounterexample::new();

    let mut state1 = TraceState::new(1);
    state1.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 0,
            type_hint: None,
        },
    );
    ce.trace.push(state1);

    let mut state2 = TraceState::new(2);
    state2.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 0,
            type_hint: None,
        },
    ); // no change
    ce.trace.push(state2);

    let mut state3 = TraceState::new(3);
    state3.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    ); // change
    ce.trace.push(state3);

    let minimized = ce.minimize();

    // State 2 should be removed
    assert_eq!(minimized.trace.len(), 2);
    assert_eq!(minimized.trace[0].state_num, 1);
    assert_eq!(minimized.trace[1].state_num, 3);
}

#[test]
fn minimize_trace_length_keeps_first_last() {
    let mut ce = StructuredCounterexample::new();
    for i in 1..=10 {
        let mut state = TraceState::new(i);
        state.variables.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: i as i128,
                type_hint: None,
            },
        );
        ce.trace.push(state);
    }

    let minimized = ce.minimize_trace_length(3);
    assert_eq!(minimized.trace.len(), 3);
    assert_eq!(minimized.trace[0].state_num, 1); // first
    assert_eq!(minimized.trace[2].state_num, 10); // last
}

#[test]
fn keep_only_variables() {
    let mut ce = StructuredCounterexample::new();
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

    let mut state = TraceState::new(1);
    state.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );
    state.variables.insert(
        "y".to_string(),
        CounterexampleValue::Int {
            value: 2,
            type_hint: None,
        },
    );
    state.variables.insert(
        "z".to_string(),
        CounterexampleValue::Int {
            value: 3,
            type_hint: None,
        },
    );
    ce.trace.push(state);

    let filtered = ce.keep_only_variables(&["x".to_string(), "y".to_string()]);

    assert!(filtered.witness.contains_key("x"));
    assert!(filtered.witness.contains_key("y"));
    assert!(!filtered.witness.contains_key("z"));

    assert!(filtered.trace[0].variables.contains_key("x"));
    assert!(filtered.trace[0].variables.contains_key("y"));
    assert!(!filtered.trace[0].variables.contains_key("z"));
}

#[test]
fn remove_variables() {
    let mut ce = StructuredCounterexample::new();
    ce.witness.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );
    ce.witness.insert(
        "debug_internal".to_string(),
        CounterexampleValue::Int {
            value: 999,
            type_hint: None,
        },
    );

    let mut state = TraceState::new(1);
    state.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );
    state.variables.insert(
        "debug_internal".to_string(),
        CounterexampleValue::Int {
            value: 999,
            type_hint: None,
        },
    );
    ce.trace.push(state);

    let filtered = ce.remove_variables(&["debug_internal".to_string()]);

    assert!(filtered.witness.contains_key("x"));
    assert!(!filtered.witness.contains_key("debug_internal"));
    assert!(!filtered.trace[0].variables.contains_key("debug_internal"));
}

#[test]
fn counterexample_statistics() {
    let ce = create_test_trace();
    let stats = ce.statistics();

    assert_eq!(stats.num_trace_states, 3);
    assert_eq!(stats.num_trace_variables, 3); // x, y, debug
    assert!(stats.num_changing_variables > 0);
    assert!(!stats.is_minimized);

    let display = format!("{}", stats);
    assert!(display.contains("Counterexample Statistics"));
    assert!(display.contains("Trace states: 3"));
}

// ==================== Semantic Equivalence Tests ====================

#[test]
fn semantic_equal_sets_order_independent() {
    // {1, 2, 3} should equal {3, 2, 1}
    let set1 = CounterexampleValue::Set(vec![
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
        CounterexampleValue::Int {
            value: 2,
            type_hint: None,
        },
        CounterexampleValue::Int {
            value: 3,
            type_hint: None,
        },
    ]);
    let set2 = CounterexampleValue::Set(vec![
        CounterexampleValue::Int {
            value: 3,
            type_hint: None,
        },
        CounterexampleValue::Int {
            value: 2,
            type_hint: None,
        },
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    ]);

    assert!(set1.semantically_equal(&set2));
    assert!(set2.semantically_equal(&set1));
    // But PartialEq should not match (order-dependent)
    assert_ne!(set1, set2);
}

#[test]
fn semantic_equal_sets_different_elements() {
    let set1 = CounterexampleValue::Set(vec![
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
        CounterexampleValue::Int {
            value: 2,
            type_hint: None,
        },
    ]);
    let set2 = CounterexampleValue::Set(vec![
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
        CounterexampleValue::Int {
            value: 3,
            type_hint: None,
        },
    ]);

    assert!(!set1.semantically_equal(&set2));
}

#[test]
fn semantic_equal_nested_sets() {
    // {{1, 2}, {3}} should equal {{3}, {2, 1}}
    let set1 = CounterexampleValue::Set(vec![
        CounterexampleValue::Set(vec![
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
        ]),
        CounterexampleValue::Set(vec![CounterexampleValue::Int {
            value: 3,
            type_hint: None,
        }]),
    ]);
    let set2 = CounterexampleValue::Set(vec![
        CounterexampleValue::Set(vec![CounterexampleValue::Int {
            value: 3,
            type_hint: None,
        }]),
        CounterexampleValue::Set(vec![
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
        ]),
    ]);

    assert!(set1.semantically_equal(&set2));
}

#[test]
fn semantic_equal_sequences_order_matters() {
    // <<1, 2>> should NOT equal <<2, 1>>
    let seq1 = CounterexampleValue::Sequence(vec![
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
        CounterexampleValue::Int {
            value: 2,
            type_hint: None,
        },
    ]);
    let seq2 = CounterexampleValue::Sequence(vec![
        CounterexampleValue::Int {
            value: 2,
            type_hint: None,
        },
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    ]);

    assert!(!seq1.semantically_equal(&seq2));

    // But <<1, 2>> should equal <<1, 2>>
    let seq3 = CounterexampleValue::Sequence(vec![
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
        CounterexampleValue::Int {
            value: 2,
            type_hint: None,
        },
    ]);
    assert!(seq1.semantically_equal(&seq3));
}

#[test]
fn semantic_equal_records() {
    // Records with same fields in different order should match
    let mut r1 = HashMap::new();
    r1.insert(
        "a".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );
    r1.insert(
        "b".to_string(),
        CounterexampleValue::Int {
            value: 2,
            type_hint: None,
        },
    );

    let mut r2 = HashMap::new();
    r2.insert(
        "b".to_string(),
        CounterexampleValue::Int {
            value: 2,
            type_hint: None,
        },
    );
    r2.insert(
        "a".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );

    let rec1 = CounterexampleValue::Record(r1);
    let rec2 = CounterexampleValue::Record(r2);

    assert!(rec1.semantically_equal(&rec2));
}

#[test]
fn semantic_equal_functions() {
    // Functions with same mappings in different order should match
    let func1 = CounterexampleValue::Function(vec![
        (
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
            CounterexampleValue::String("a".to_string()),
        ),
        (
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
            CounterexampleValue::String("b".to_string()),
        ),
    ]);
    let func2 = CounterexampleValue::Function(vec![
        (
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
            CounterexampleValue::String("b".to_string()),
        ),
        (
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
            CounterexampleValue::String("a".to_string()),
        ),
    ]);

    assert!(func1.semantically_equal(&func2));
}

#[test]
fn normalize_value_sorts_sets() {
    let set = CounterexampleValue::Set(vec![
        CounterexampleValue::Int {
            value: 3,
            type_hint: None,
        },
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
        CounterexampleValue::Int {
            value: 2,
            type_hint: None,
        },
    ]);

    let normalized = set.normalize();
    if let CounterexampleValue::Set(elems) = normalized {
        // Should be sorted by display representation: 1, 2, 3
        assert_eq!(elems.len(), 3);
        assert_eq!(
            elems[0],
            CounterexampleValue::Int {
                value: 1,
                type_hint: None
            }
        );
        assert_eq!(
            elems[1],
            CounterexampleValue::Int {
                value: 2,
                type_hint: None
            }
        );
        assert_eq!(
            elems[2],
            CounterexampleValue::Int {
                value: 3,
                type_hint: None
            }
        );
    } else {
        panic!("Expected Set");
    }
}

#[test]
fn semantic_diff_with_sets() {
    let mut ce1 = StructuredCounterexample::new();
    ce1.witness.insert(
        "s".to_string(),
        CounterexampleValue::Set(vec![
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
        ]),
    );

    let mut ce2 = StructuredCounterexample::new();
    ce2.witness.insert(
        "s".to_string(),
        CounterexampleValue::Set(vec![
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
        ]),
    );

    // Regular diff should see these as different
    let regular_diff = ce1.diff(&ce2);
    assert!(!regular_diff.is_equivalent());
    assert!(regular_diff.value_differences.contains_key("s"));

    // Semantic diff should see these as equivalent
    let semantic_diff = ce1.diff_semantic(&ce2);
    assert!(semantic_diff.is_equivalent());
    assert!(ce1.is_semantically_equivalent_to(&ce2));
}

// ==================== Trace Compression Tests ====================

#[test]
fn compress_trace_empty() {
    let ce = StructuredCounterexample::new();
    let compressed = ce.compress_trace();

    assert_eq!(compressed.original_length, 0);
    assert_eq!(compressed.compressed_length, 0);
    assert!(compressed.segments.is_empty());
    assert_eq!(compressed.compression_ratio(), 0.0);
}

#[test]
fn compress_trace_no_repetition() {
    let mut ce = StructuredCounterexample::new();

    // Create trace with no repetition
    for i in 1..=3 {
        let mut state = TraceState::new(i);
        state.variables.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: i as i128,
                type_hint: None,
            },
        );
        ce.trace.push(state);
    }

    let compressed = ce.compress_trace();

    // Should have 3 single segments
    assert_eq!(compressed.original_length, 3);
    assert_eq!(compressed.segments.len(), 3);
    for segment in &compressed.segments {
        assert!(matches!(segment, TraceSegment::Single(_)));
    }
}

#[test]
fn compress_trace_simple_repetition() {
    let mut ce = StructuredCounterexample::new();

    // Create trace with pattern: A, A, A, A (same state repeated 4 times)
    for i in 1..=4 {
        let mut state = TraceState::new(i);
        state.action = Some("Step".to_string());
        state.variables.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
        );
        ce.trace.push(state);
    }

    let compressed = ce.compress_trace();

    // Should have 1 repeated segment
    assert_eq!(compressed.original_length, 4);
    assert_eq!(compressed.segments.len(), 1);

    if let TraceSegment::Repeated { pattern, count } = &compressed.segments[0] {
        assert_eq!(pattern.len(), 1);
        assert_eq!(*count, 4);
    } else {
        panic!("Expected Repeated segment");
    }
}

#[test]
fn compress_trace_pattern_repetition() {
    let mut ce = StructuredCounterexample::new();

    // Create trace with pattern: A, B, A, B (pattern of 2 repeated 2 times)
    for i in 0..4 {
        let mut state = TraceState::new(i as u32 + 1);
        let val = if i % 2 == 0 { 0 } else { 1 };
        state.variables.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: val,
                type_hint: None,
            },
        );
        ce.trace.push(state);
    }

    let compressed = ce.compress_trace();

    // Should detect the A, B pattern repeated twice
    assert_eq!(compressed.original_length, 4);
    assert!(compressed.segments.len() <= 2); // Either 1 Repeated(2) or some singles

    // Verify total states preserved
    assert_eq!(compressed.total_states(), 4);
}

#[test]
fn compress_trace_expand_roundtrip() {
    let mut ce = StructuredCounterexample::new();

    // Create trace: A, B, A, B, C
    for i in 0..5 {
        let mut state = TraceState::new(i as u32 + 1);
        let val = if i < 4 { i % 2 } else { 2 };
        state.variables.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: val as i128,
                type_hint: None,
            },
        );
        ce.trace.push(state);
    }

    let compressed = ce.compress_trace();
    let expanded = compressed.expand();

    // Expanded should match original in variable values (state_num may differ due to renumbering)
    assert_eq!(expanded.len(), ce.trace.len());
    for (exp, orig) in expanded.iter().zip(ce.trace.iter()) {
        assert_eq!(exp.variables, orig.variables);
    }
}

#[test]
fn detect_cycle_simple() {
    let mut ce = StructuredCounterexample::new();

    // Create trace: A, B, A (cycle at state 0, length 2)
    let mut state1 = TraceState::new(1);
    state1.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 0,
            type_hint: None,
        },
    );
    ce.trace.push(state1);

    let mut state2 = TraceState::new(2);
    state2.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );
    ce.trace.push(state2);

    let mut state3 = TraceState::new(3);
    state3.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 0,
            type_hint: None,
        },
    );
    ce.trace.push(state3);

    let cycle = ce.detect_cycle();
    assert!(cycle.is_some());

    let (start, len) = cycle.unwrap();
    assert_eq!(start, 0); // Cycle starts at first state
    assert_eq!(len, 2); // Cycle length is 2 (A -> B -> A)
}

#[test]
fn detect_cycle_none() {
    let mut ce = StructuredCounterexample::new();

    // Create trace with no cycles: A, B, C (all different)
    for i in 0..3 {
        let mut state = TraceState::new(i as u32 + 1);
        state.variables.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: i,
                type_hint: None,
            },
        );
        ce.trace.push(state);
    }

    let cycle = ce.detect_cycle();
    assert!(cycle.is_none());
}

#[test]
fn compressed_trace_display() {
    let mut ce = StructuredCounterexample::new();

    // Create trace with repetition
    for i in 0..6 {
        let mut state = TraceState::new(i as u32 + 1);
        state.action = Some("Step".to_string());
        state.variables.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: (i % 2) as i128,
                type_hint: None,
            },
        );
        ce.trace.push(state);
    }

    let compressed = ce.compress_trace();
    let display = format!("{}", compressed);

    assert!(display.contains("Compressed Trace"));
    assert!(display.contains("6 states"));
    assert!(display.contains("compression"));
}

#[test]
fn format_trace_with_cycles_shows_cycle_info() {
    let mut ce = StructuredCounterexample::new();

    // Create cyclic trace
    for i in 0..4 {
        let mut state = TraceState::new(i as u32 + 1);
        state.variables.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: (i % 2) as i128,
                type_hint: None,
            },
        );
        ce.trace.push(state);
    }

    let formatted = ce.format_trace_with_cycles();
    // Should detect the cycle
    assert!(formatted.contains("Cycle") || formatted.contains("compression"));
}

#[test]
fn trace_segment_total_states() {
    let state = TraceState::new(1);
    let single = TraceSegment::Single(state.clone());
    assert_eq!(single.total_states(), 1);

    let repeated = TraceSegment::Repeated {
        pattern: vec![state.clone(), state.clone()],
        count: 3,
    };
    assert_eq!(repeated.total_states(), 6); // 2 * 3
}

// DOT export tests

#[test]
fn dot_export_empty_counterexample() {
    let ce = StructuredCounterexample::new();
    let dot = ce.to_dot();
    assert!(dot.starts_with("digraph Counterexample {"));
    assert!(dot.ends_with("}\n"));
}

#[test]
fn dot_export_with_trace() {
    let mut ce = StructuredCounterexample::new();

    let mut state1 = TraceState::new(1);
    state1.action = Some("Init".to_string());
    state1.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 0,
            type_hint: None,
        },
    );
    ce.trace.push(state1);

    let mut state2 = TraceState::new(2);
    state2.action = Some("Increment".to_string());
    state2.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );
    ce.trace.push(state2);

    let dot = ce.to_dot();
    assert!(dot.contains("digraph Counterexample"));
    assert!(dot.contains("s0"));
    assert!(dot.contains("s1"));
    assert!(dot.contains("s0 -> s1"));
    assert!(dot.contains("Increment"));
}

#[test]
fn dot_export_with_witness() {
    let mut ce = StructuredCounterexample::new();
    ce.witness.insert(
        "n".to_string(),
        CounterexampleValue::Int {
            value: 42,
            type_hint: None,
        },
    );

    let dot = ce.to_dot();
    assert!(dot.contains("witness"));
    assert!(dot.contains("n: 42"));
}

#[test]
fn dot_export_with_failed_checks() {
    let mut ce = StructuredCounterexample::new();
    ce.failed_checks.push(FailedCheck {
        check_id: "assert.1".to_string(),
        description: "Division by zero".to_string(),
        location: None,
        function: None,
    });

    let mut state1 = TraceState::new(1);
    state1.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 0,
            type_hint: None,
        },
    );
    ce.trace.push(state1);

    let dot = ce.to_dot();
    assert!(dot.contains("failures"));
    assert!(dot.contains("Division by zero"));
    assert!(dot.contains("s0 -> failures"));
}

#[test]
fn compressed_trace_dot_export() {
    let mut ce = StructuredCounterexample::new();

    // Create a trace with repeated pattern
    for i in 1..=6 {
        let mut state = TraceState::new(i);
        state.action = Some(format!("Step{}", (i - 1) % 2 + 1));
        state.variables.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: (i % 2) as i128,
                type_hint: None,
            },
        );
        ce.trace.push(state);
    }

    let compressed = ce.compress_trace();
    let dot = compressed.to_dot();
    assert!(dot.contains("digraph CompressedTrace"));
}

#[test]
fn compressed_trace_mermaid_basic() {
    let mut ce = StructuredCounterexample::new();

    // Create a trace with a single state
    let mut state = TraceState::new(1);
    state.action = Some("Init".to_string());
    state.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 0,
            type_hint: None,
        },
    );
    ce.trace.push(state);

    let compressed = ce.compress_trace();
    let mermaid = compressed.to_mermaid();

    assert!(mermaid.contains("flowchart TB"));
    assert!(mermaid.contains("s0[\""));
    assert!(mermaid.contains("S1"));
}

#[test]
fn compressed_trace_mermaid_with_repetition() {
    let mut ce = StructuredCounterexample::new();

    // Create a trace with repeated pattern (same action repeating)
    for i in 1..=6 {
        let mut state = TraceState::new(i);
        state.action = Some(format!("Step{}", (i - 1) % 2 + 1));
        state.variables.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: (i % 2) as i128,
                type_hint: None,
            },
        );
        ce.trace.push(state);
    }

    let compressed = ce.compress_trace();
    let mermaid = compressed.to_mermaid();

    assert!(mermaid.contains("flowchart TB"));
    // Should contain some indication of the compression
    assert!(mermaid.contains("%% Compressed Trace"));
}

#[test]
fn compressed_trace_mermaid_empty() {
    let ce = StructuredCounterexample::new();
    let compressed = ce.compress_trace();
    let mermaid = compressed.to_mermaid();

    assert!(mermaid.contains("flowchart TB"));
    assert!(mermaid.contains("Empty trace"));
}

#[test]
fn compressed_trace_html_basic() {
    let mut ce = StructuredCounterexample::new();

    let mut state = TraceState::new(1);
    state.action = Some("Init".to_string());
    ce.trace.push(state);

    let compressed = ce.compress_trace();
    let html = compressed.to_html(Some("Test Compressed"));

    assert!(html.contains("<!DOCTYPE html>"));
    assert!(html.contains("<title>Test Compressed</title>"));
    assert!(html.contains("mermaid@10"));
    assert!(html.contains("<strong>Original States:</strong>"));
    assert!(html.contains("<strong>Segments:</strong>"));
    assert!(html.contains("<strong>Compression:</strong>"));
    assert!(html.contains("Download Mermaid"));
    assert!(html.contains("Download DOT"));
}

#[test]
fn compressed_trace_html_default_title() {
    let ce = StructuredCounterexample::new();
    let compressed = ce.compress_trace();
    let html = compressed.to_html(None);

    assert!(html.contains("<title>Compressed Trace Visualization</title>"));
}

// Clustering tests

#[test]
fn clustering_empty_list() {
    let clusters = CounterexampleClusters::from_counterexamples(vec![], 0.7);
    assert_eq!(clusters.num_clusters(), 0);
    assert_eq!(clusters.total_counterexamples(), 0);
}

#[test]
fn clustering_single_counterexample() {
    let mut ce = StructuredCounterexample::new();
    ce.failed_checks.push(FailedCheck {
        check_id: "check.1".to_string(),
        description: "Test failure".to_string(),
        location: None,
        function: None,
    });

    let clusters = CounterexampleClusters::from_counterexamples(vec![ce], 0.7);
    assert_eq!(clusters.num_clusters(), 1);
    assert_eq!(clusters.total_counterexamples(), 1);
}

#[test]
fn clustering_similar_counterexamples() {
    // Create two similar counterexamples with same failed check
    let mut ce1 = StructuredCounterexample::new();
    ce1.failed_checks.push(FailedCheck {
        check_id: "overflow.1".to_string(),
        description: "Integer overflow".to_string(),
        location: None,
        function: None,
    });
    let mut state1 = TraceState::new(1);
    state1.action = Some("Init".to_string());
    ce1.trace.push(state1);

    let mut ce2 = StructuredCounterexample::new();
    ce2.failed_checks.push(FailedCheck {
        check_id: "overflow.1".to_string(),
        description: "Integer overflow".to_string(),
        location: None,
        function: None,
    });
    let mut state2 = TraceState::new(1);
    state2.action = Some("Init".to_string());
    ce2.trace.push(state2);

    let clusters = CounterexampleClusters::from_counterexamples(vec![ce1, ce2], 0.7);
    // Should cluster together since they have same failed check
    assert_eq!(clusters.num_clusters(), 1);
    assert_eq!(clusters.total_counterexamples(), 2);
    assert_eq!(clusters.largest_cluster().unwrap().size(), 2);
}

#[test]
fn clustering_different_counterexamples() {
    let mut ce1 = StructuredCounterexample::new();
    ce1.failed_checks.push(FailedCheck {
        check_id: "overflow.1".to_string(),
        description: "Integer overflow".to_string(),
        location: None,
        function: None,
    });

    let mut ce2 = StructuredCounterexample::new();
    ce2.failed_checks.push(FailedCheck {
        check_id: "null.1".to_string(),
        description: "Null pointer".to_string(),
        location: None,
        function: None,
    });

    // Use high threshold to ensure they don't cluster
    let clusters = CounterexampleClusters::from_counterexamples(vec![ce1, ce2], 0.9);
    // Different failure types should be separate clusters
    assert_eq!(clusters.num_clusters(), 2);
}

#[test]
fn cluster_similarity_identical() {
    let mut ce = StructuredCounterexample::new();
    ce.failed_checks.push(FailedCheck {
        check_id: "check.1".to_string(),
        description: "Test".to_string(),
        location: None,
        function: None,
    });

    let cluster = CounterexampleCluster::new(ce.clone(), 0.5);
    let sim = cluster.similarity(&ce);
    assert!(
        sim > 0.99,
        "Identical counterexamples should have ~1.0 similarity"
    );
}

#[test]
fn cluster_summary_format() {
    let mut ce1 = StructuredCounterexample::new();
    ce1.failed_checks.push(FailedCheck {
        check_id: "test.1".to_string(),
        description: "Test failure".to_string(),
        location: None,
        function: None,
    });

    let ce2 = ce1.clone();

    let clusters = CounterexampleClusters::from_counterexamples(vec![ce1, ce2], 0.5);
    let summary = clusters.summary();
    assert!(summary.contains("Clustered 2 counterexamples"));
    assert!(summary.contains("1 clusters"));
    assert!(summary.contains("2 members"));
}

#[test]
fn clusters_to_mermaid_empty() {
    let clusters = CounterexampleClusters::from_counterexamples(vec![], 0.7);
    let mermaid = clusters.to_mermaid();

    assert!(mermaid.contains("flowchart TB"));
    assert!(mermaid.contains("No clusters"));
}

#[test]
fn clusters_to_mermaid_with_clusters() {
    let mut ce1 = StructuredCounterexample::new();
    ce1.failed_checks.push(FailedCheck {
        check_id: "overflow.1".to_string(),
        description: "Integer overflow".to_string(),
        location: None,
        function: None,
    });

    let mut ce2 = StructuredCounterexample::new();
    ce2.failed_checks.push(FailedCheck {
        check_id: "null.1".to_string(),
        description: "Null pointer".to_string(),
        location: None,
        function: None,
    });

    let clusters = CounterexampleClusters::from_counterexamples(vec![ce1, ce2], 0.9);
    let mermaid = clusters.to_mermaid();

    // Should be a pie chart
    assert!(mermaid.contains("pie showData"));
    assert!(mermaid.contains("Cluster 1"));
    assert!(mermaid.contains("Cluster 2"));
}

#[test]
fn clusters_to_mermaid_flowchart() {
    let mut ce = StructuredCounterexample::new();
    ce.failed_checks.push(FailedCheck {
        check_id: "test.1".to_string(),
        description: "Test failure".to_string(),
        location: None,
        function: None,
    });

    let clusters = CounterexampleClusters::from_counterexamples(vec![ce], 0.7);
    let mermaid = clusters.to_mermaid_flowchart();

    assert!(mermaid.contains("flowchart TB"));
    assert!(mermaid.contains("root{"));
    assert!(mermaid.contains("c0["));
    assert!(mermaid.contains("root --> c0"));
}

#[test]
fn clusters_to_html_basic() {
    let mut ce = StructuredCounterexample::new();
    ce.failed_checks.push(FailedCheck {
        check_id: "test.1".to_string(),
        description: "Test failure".to_string(),
        location: None,
        function: None,
    });

    let clusters = CounterexampleClusters::from_counterexamples(vec![ce], 0.7);
    let html = clusters.to_html(Some("Test Clusters"));

    assert!(html.contains("<!DOCTYPE html>"));
    assert!(html.contains("<title>Test Clusters</title>"));
    assert!(html.contains("mermaid@10"));
    assert!(html.contains("<strong>Total Counterexamples:</strong>"));
    assert!(html.contains("<strong>Clusters:</strong>"));
    assert!(html.contains("<strong>Similarity Threshold:</strong>"));
    assert!(html.contains("showDiagram('pie')"));
    assert!(html.contains("showDiagram('flowchart')"));
    assert_eq!(html.matches("Download Mermaid").count(), 2);
    assert!(html.contains("download-buttons"));
}

#[test]
fn clusters_to_html_default_title() {
    let clusters = CounterexampleClusters::from_counterexamples(vec![], 0.7);
    let html = clusters.to_html(None);

    assert!(html.contains("<title>Counterexample Clusters Visualization</title>"));
}

// ==================== Trace Abstraction Tests ====================

#[test]
fn abstract_trace_no_abstraction_for_small_traces() {
    let mut ce = StructuredCounterexample::new();
    ce.trace.push(TraceState::new(1));
    ce.trace.push(TraceState::new(2));

    let abstracted = ce.abstract_trace(3);
    // With min_group_size=3 and only 2 states, no abstraction happens
    assert_eq!(abstracted.segments.len(), 2);
    assert!(matches!(
        abstracted.segments[0],
        TraceAbstractionSegment::Concrete(_)
    ));
}

#[test]
fn abstract_trace_groups_similar_states() {
    let mut ce = StructuredCounterexample::new();

    // Create 5 states with similar action pattern "Increment"
    for i in 1..=5 {
        let mut state = TraceState::new(i);
        state.action = Some(format!("Increment{}", i));
        state.variables.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: i as i128,
                type_hint: None,
            },
        );
        ce.trace.push(state);
    }

    let abstracted = ce.abstract_trace(3);
    // Should abstract all 5 into one segment
    assert!(abstracted.segments.len() < 5);
    assert!(abstracted.compression_ratio > 0.0);
}

#[test]
fn abstract_trace_preserves_different_patterns() {
    let mut ce = StructuredCounterexample::new();

    // Create 3 "Init" states
    for i in 1..=3 {
        let mut state = TraceState::new(i);
        state.action = Some("Init".to_string());
        state.variables.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 0,
                type_hint: None,
            },
        );
        ce.trace.push(state);
    }

    // Then 3 "Step" states
    for i in 4..=6 {
        let mut state = TraceState::new(i as u32);
        state.action = Some("Step".to_string());
        state.variables.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: i as i128,
                type_hint: None,
            },
        );
        ce.trace.push(state);
    }

    let abstracted = ce.abstract_trace(2);
    // Should have 2 segments (Init group and Step group)
    assert_eq!(abstracted.segments.len(), 2);
}

#[test]
fn abstracted_value_int_range() {
    let values: Vec<CounterexampleValue> = vec![
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
        CounterexampleValue::Int {
            value: 5,
            type_hint: None,
        },
        CounterexampleValue::Int {
            value: 3,
            type_hint: None,
        },
    ];
    let refs: Vec<_> = values.iter().collect();
    let abstracted = StructuredCounterexample::abstract_values(&refs);

    match abstracted {
        AbstractedValue::IntRange { min, max } => {
            assert_eq!(min, 1);
            assert_eq!(max, 5);
        }
        _ => panic!("Expected IntRange"),
    }
}

#[test]
fn abstracted_value_concrete_for_identical() {
    let values: Vec<CounterexampleValue> = vec![
        CounterexampleValue::Bool(true),
        CounterexampleValue::Bool(true),
        CounterexampleValue::Bool(true),
    ];
    let refs: Vec<_> = values.iter().collect();
    let abstracted = StructuredCounterexample::abstract_values(&refs);

    assert!(matches!(
        abstracted,
        AbstractedValue::Concrete(CounterexampleValue::Bool(true))
    ));
}

#[test]
fn abstracted_trace_display() {
    let mut ce = StructuredCounterexample::new();
    for i in 1..=4 {
        let mut state = TraceState::new(i);
        state.action = Some("Loop".to_string());
        state.variables.insert(
            "counter".to_string(),
            CounterexampleValue::Int {
                value: i as i128,
                type_hint: None,
            },
        );
        ce.trace.push(state);
    }

    let abstracted = ce.abstract_trace(2);
    let display = format!("{}", abstracted);
    assert!(display.contains("Abstracted Trace"));
    assert!(display.contains("compression"));
}

// ==================== Configurable Actor Pattern Tests ====================

#[test]
fn detect_interleavings_with_custom_action_pattern() {
    let mut ce = StructuredCounterexample::new();

    // Custom pattern: "Actor<name>:step"
    let mut s1 = TraceState::new(1);
    s1.action = Some("Actor<Alice>:step".to_string());
    ce.trace.push(s1);

    let mut s2 = TraceState::new(2);
    s2.action = Some("Actor<Bob>:step".to_string());
    ce.trace.push(s2);

    let mut s3 = TraceState::new(3);
    s3.action = Some("Actor<Alice>:finish".to_string());
    ce.trace.push(s3);

    // Custom pattern to extract actor name from angle brackets
    let config = ActorPatternConfig::custom(vec![r"Actor<([^>]+)>".to_string()], vec![]);

    let interleaving = ce.detect_interleavings_with_config(&config);
    assert_eq!(interleaving.lanes.len(), 2);
    assert!(interleaving
        .lanes
        .iter()
        .any(|l| l.actor == "Alice" && l.states.len() == 2));
    assert!(interleaving
        .lanes
        .iter()
        .any(|l| l.actor == "Bob" && l.states.len() == 1));
}

#[test]
fn detect_interleavings_with_custom_variable_pattern() {
    let mut ce = StructuredCounterexample::new();

    let mut s1 = TraceState::new(1);
    s1.variables.insert(
        "state[Worker1]".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );
    ce.trace.push(s1);

    let mut s2 = TraceState::new(2);
    s2.variables.insert(
        "state[Worker2]".to_string(),
        CounterexampleValue::Int {
            value: 2,
            type_hint: None,
        },
    );
    ce.trace.push(s2);

    // Custom pattern to extract from bracket notation
    let config = ActorPatternConfig::custom(vec![], vec![r"\[([^\]]+)\]".to_string()]);

    let interleaving = ce.detect_interleavings_with_config(&config);
    assert_eq!(interleaving.lanes.len(), 2);
    assert!(interleaving.lanes.iter().any(|l| l.actor == "Worker1"));
    assert!(interleaving.lanes.iter().any(|l| l.actor == "Worker2"));
}

#[test]
fn detect_interleavings_combined_patterns() {
    let mut ce = StructuredCounterexample::new();

    // Default pattern format
    let mut s1 = TraceState::new(1);
    s1.action = Some("ProcessA: init".to_string());
    ce.trace.push(s1);

    // Custom pattern format
    let mut s2 = TraceState::new(2);
    s2.action = Some("~ProcessB~start".to_string());
    ce.trace.push(s2);

    // Combined: use both custom and default patterns
    let config = ActorPatternConfig::combined(vec![r"~([A-Za-z0-9]+)~".to_string()], vec![]);

    let interleaving = ce.detect_interleavings_with_config(&config);
    assert_eq!(interleaving.lanes.len(), 2);
    assert!(interleaving.lanes.iter().any(|l| l.actor == "ProcessA"));
    assert!(interleaving.lanes.iter().any(|l| l.actor == "ProcessB"));
}

#[test]
fn actor_pattern_config_default_behavior() {
    // Verify default patterns still work
    let mut ce = StructuredCounterexample::new();

    let mut s1 = TraceState::new(1);
    s1.action = Some("Thread1: execute".to_string());
    ce.trace.push(s1);

    let interleaving = ce.detect_interleavings();
    assert_eq!(interleaving.lanes.len(), 1);
    assert_eq!(interleaving.lanes[0].actor, "Thread1");
}

#[test]
fn actor_pattern_config_custom_only_ignores_defaults() {
    let mut ce = StructuredCounterexample::new();

    // This uses default pattern format, but we'll use custom-only config
    let mut s1 = TraceState::new(1);
    s1.action = Some("Thread1: execute".to_string());
    ce.trace.push(s1);

    // Custom pattern that won't match
    let config = ActorPatternConfig::custom(vec![r"CUSTOM\[([^\]]+)\]".to_string()], vec![]);

    let interleaving = ce.detect_interleavings_with_config(&config);
    // Should have no lanes since default patterns are disabled
    assert_eq!(interleaving.lanes.len(), 0);
    assert_eq!(interleaving.unassigned_states.len(), 1);
}

// ==================== AbstractedTrace DOT Export Tests ====================

#[test]
fn abstracted_trace_to_dot_basic() {
    let mut ce = StructuredCounterexample::new();

    // Create a trace with a repeated pattern
    for i in 1..=6 {
        let mut state = TraceState::new(i);
        state.action = Some("Step".to_string());
        state.variables.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: i as i128,
                type_hint: None,
            },
        );
        ce.trace.push(state);
    }

    let abstracted = ce.abstract_trace(3);
    let dot = abstracted.to_dot();

    assert!(dot.starts_with("digraph AbstractedTrace {"));
    assert!(dot.contains("rankdir=TB"));
    assert!(dot.ends_with("}\n"));
}

#[test]
fn abstracted_trace_to_dot_with_mixed_segments() {
    let mut ce = StructuredCounterexample::new();

    // Single state
    let mut s1 = TraceState::new(1);
    s1.action = Some("Init".to_string());
    s1.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 0,
            type_hint: None,
        },
    );
    ce.trace.push(s1);

    // Repeated pattern (4 states with same action)
    for i in 2..=5 {
        let mut state = TraceState::new(i);
        state.action = Some("Loop".to_string());
        state.variables.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: i as i128,
                type_hint: None,
            },
        );
        ce.trace.push(state);
    }

    // Final state
    let mut s_final = TraceState::new(6);
    s_final.action = Some("Done".to_string());
    ce.trace.push(s_final);

    let abstracted = ce.abstract_trace(3);
    let dot = abstracted.to_dot();

    assert!(dot.contains("digraph AbstractedTrace"));
    // Should have edges between nodes
    assert!(dot.contains("->"));
}

// ==================== Multi-Trace Alignment Tests ====================

#[test]
fn align_multiple_traces_basic() {
    let mut ce1 = StructuredCounterexample::new();
    let mut s1 = TraceState::new(1);
    s1.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );
    ce1.trace.push(s1);

    let mut ce2 = StructuredCounterexample::new();
    let mut s2 = TraceState::new(1);
    s2.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 2,
            type_hint: None,
        },
    );
    ce2.trace.push(s2);

    let alignment = align_multiple_traces(&[&ce1, &ce2], None);

    assert_eq!(alignment.trace_count(), 2);
    assert_eq!(alignment.rows.len(), 1);
    assert!(alignment.rows[0].states[0].is_some());
    assert!(alignment.rows[0].states[1].is_some());
}

#[test]
fn align_multiple_traces_with_divergence() {
    let mut ce1 = StructuredCounterexample::new();
    let mut s1 = TraceState::new(1);
    s1.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 10,
            type_hint: None,
        },
    );
    ce1.trace.push(s1);

    let mut ce2 = StructuredCounterexample::new();
    let mut s2 = TraceState::new(1);
    s2.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 20,
            type_hint: None,
        },
    );
    ce2.trace.push(s2);

    let alignment = align_multiple_traces(&[&ce1, &ce2], None);

    assert_eq!(alignment.divergence_points.len(), 1);
    assert_eq!(alignment.divergence_points[0].variable, "x");
    assert_eq!(alignment.first_divergence(), Some(1));
}

#[test]
fn align_multiple_traces_different_lengths() {
    let mut ce1 = StructuredCounterexample::new();
    ce1.trace.push(TraceState::new(1));
    ce1.trace.push(TraceState::new(2));
    ce1.trace.push(TraceState::new(3));

    let mut ce2 = StructuredCounterexample::new();
    ce2.trace.push(TraceState::new(1));
    ce2.trace.push(TraceState::new(2));

    let alignment = align_multiple_traces(&[&ce1, &ce2], None);

    assert_eq!(alignment.rows.len(), 3);
    // State 3 should only be in ce1
    assert!(alignment.rows[2].states[0].is_some());
    assert!(alignment.rows[2].states[1].is_none());
}

#[test]
fn align_multiple_traces_three_traces() {
    let mut ce1 = StructuredCounterexample::new();
    let mut ce2 = StructuredCounterexample::new();
    let mut ce3 = StructuredCounterexample::new();

    for ce in [&mut ce1, &mut ce2, &mut ce3] {
        let mut s = TraceState::new(1);
        s.variables.insert(
            "y".to_string(),
            CounterexampleValue::Int {
                value: 5,
                type_hint: None,
            },
        );
        ce.trace.push(s);
    }

    let alignment = align_multiple_traces(
        &[&ce1, &ce2, &ce3],
        Some(vec![
            "Alpha".to_string(),
            "Beta".to_string(),
            "Gamma".to_string(),
        ]),
    );

    assert_eq!(alignment.trace_count(), 3);
    assert_eq!(alignment.trace_labels, vec!["Alpha", "Beta", "Gamma"]);
    // No divergence since all have same value
    assert!(alignment.divergence_points.is_empty());
}

#[test]
fn align_multiple_traces_format_table() {
    let mut ce1 = StructuredCounterexample::new();
    let mut s1 = TraceState::new(1);
    s1.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );
    ce1.trace.push(s1);

    let mut ce2 = StructuredCounterexample::new();
    let mut s2 = TraceState::new(1);
    s2.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 2,
            type_hint: None,
        },
    );
    ce2.trace.push(s2);

    let alignment = align_multiple_traces(&[&ce1, &ce2], None);
    let table = alignment.format_table();

    assert!(table.contains("Multi-Trace Alignment"));
    assert!(table.contains("State 1"));
    assert!(table.contains("x [differs]"));
    assert!(table.contains("Divergence Summary"));
}

// ==================== Pattern Suggestion Tests ====================

#[test]
fn suggest_patterns_empty_trace() {
    let ce = StructuredCounterexample::new();
    let suggestions = ce.suggest_patterns();
    assert!(suggestions.is_empty());
}

#[test]
fn suggest_patterns_invariant_variable() {
    let mut ce = StructuredCounterexample::new();

    for i in 1..=5 {
        let mut state = TraceState::new(i);
        // x changes, y stays the same
        state.variables.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: i as i128,
                type_hint: None,
            },
        );
        state.variables.insert(
            "constant".to_string(),
            CounterexampleValue::Int {
                value: 42,
                type_hint: None,
            },
        );
        ce.trace.push(state);
    }

    let suggestions = ce.suggest_patterns();
    let invariant_suggestions: Vec<_> = suggestions
        .iter()
        .filter(|s| s.kind == SuggestionKind::InvariantVariable)
        .collect();

    assert!(!invariant_suggestions.is_empty());
    assert!(invariant_suggestions[0].description.contains("constant"));
}

#[test]
fn suggest_patterns_monotonic_increasing() {
    let mut ce = StructuredCounterexample::new();

    for i in 1..=5 {
        let mut state = TraceState::new(i);
        state.variables.insert(
            "counter".to_string(),
            CounterexampleValue::Int {
                value: i as i128 * 10,
                type_hint: None,
            },
        );
        ce.trace.push(state);
    }

    let suggestions = ce.suggest_patterns();
    let monotonic_suggestions: Vec<_> = suggestions
        .iter()
        .filter(|s| s.kind == SuggestionKind::MonotonicVariable)
        .collect();

    assert!(!monotonic_suggestions.is_empty());
    assert!(monotonic_suggestions[0]
        .description
        .contains("strictly increasing"));
}

#[test]
fn suggest_patterns_monotonic_decreasing() {
    let mut ce = StructuredCounterexample::new();

    for i in 1..=5 {
        let mut state = TraceState::new(i);
        state.variables.insert(
            "countdown".to_string(),
            CounterexampleValue::Int {
                value: 100 - (i as i128 * 10),
                type_hint: None,
            },
        );
        ce.trace.push(state);
    }

    let suggestions = ce.suggest_patterns();
    let monotonic_suggestions: Vec<_> = suggestions
        .iter()
        .filter(|s| s.kind == SuggestionKind::MonotonicVariable)
        .collect();

    assert!(!monotonic_suggestions.is_empty());
    assert!(monotonic_suggestions[0]
        .description
        .contains("strictly decreasing"));
}

#[test]
fn suggest_patterns_repeating() {
    let mut ce = StructuredCounterexample::new();

    // Create a trace with a clear repeating pattern
    for i in 1..=12 {
        let mut state = TraceState::new(i);
        state.action = Some("Step".to_string());
        // Value cycles between 0, 1, 2
        state.variables.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: ((i - 1) % 3) as i128,
                type_hint: None,
            },
        );
        ce.trace.push(state);
    }

    let suggestions = ce.suggest_patterns();
    let repeating_suggestions: Vec<_> = suggestions
        .iter()
        .filter(|s| s.kind == SuggestionKind::RepeatingPattern)
        .collect();

    // Detect repeating pattern suggestions
    assert!(!repeating_suggestions.is_empty());
}

#[test]
fn suggest_patterns_interleaving() {
    let mut ce = StructuredCounterexample::new();

    for i in 1..=6 {
        let mut state = TraceState::new(i);
        // Alternate between two actors
        state.action = Some(format!("Thread{}: work", if i % 2 == 1 { 1 } else { 2 }));
        ce.trace.push(state);
    }

    let suggestions = ce.suggest_patterns();
    let interleaving_suggestions: Vec<_> = suggestions
        .iter()
        .filter(|s| s.kind == SuggestionKind::InterleavingActors)
        .collect();

    assert!(!interleaving_suggestions.is_empty());
    assert!(interleaving_suggestions[0].description.contains("Thread1"));
    assert!(interleaving_suggestions[0].description.contains("Thread2"));
}

#[test]
fn suggest_patterns_sorted_by_confidence() {
    let mut ce = StructuredCounterexample::new();

    for i in 1..=10 {
        let mut state = TraceState::new(i);
        state.action = Some(format!("Actor{}: step", if i % 2 == 1 { "A" } else { "B" }));
        state.variables.insert(
            "increasing".to_string(),
            CounterexampleValue::Int {
                value: i as i128,
                type_hint: None,
            },
        );
        state.variables.insert(
            "constant".to_string(),
            CounterexampleValue::Int {
                value: 99,
                type_hint: None,
            },
        );
        ce.trace.push(state);
    }

    let suggestions = ce.suggest_patterns();

    // Verify sorted by severity descending (primary), then confidence (secondary)
    for i in 1..suggestions.len() {
        let prev = &suggestions[i - 1];
        let curr = &suggestions[i];
        // Either severity is higher, or equal severity with higher confidence
        assert!(
            prev.severity > curr.severity
                || (prev.severity == curr.severity && prev.confidence >= curr.confidence),
            "Suggestions not properly sorted: {:?} should come before {:?}",
            prev.severity,
            curr.severity
        );
    }
}

#[test]
fn test_suggestion_severity_from_confidence() {
    assert_eq!(
        SuggestionSeverity::from_confidence(0.95),
        SuggestionSeverity::Critical
    );
    assert_eq!(
        SuggestionSeverity::from_confidence(0.75),
        SuggestionSeverity::High
    );
    assert_eq!(
        SuggestionSeverity::from_confidence(0.55),
        SuggestionSeverity::Medium
    );
    assert_eq!(
        SuggestionSeverity::from_confidence(0.3),
        SuggestionSeverity::Low
    );
}

#[test]
fn test_suggestion_severity_ordering() {
    // Test that Ord is correctly implemented
    assert!(SuggestionSeverity::Critical > SuggestionSeverity::High);
    assert!(SuggestionSeverity::High > SuggestionSeverity::Medium);
    assert!(SuggestionSeverity::Medium > SuggestionSeverity::Low);
}

#[test]
fn test_suggestion_severity_display() {
    assert_eq!(format!("{}", SuggestionSeverity::Critical), "Critical");
    assert_eq!(format!("{}", SuggestionSeverity::High), "High");
    assert_eq!(format!("{}", SuggestionSeverity::Medium), "Medium");
    assert_eq!(format!("{}", SuggestionSeverity::Low), "Low");
}

#[test]
fn test_trace_suggestion_new_calculates_severity() {
    // Interleaving actors with high confidence -> Critical (weight 1.2 * 0.9 = 1.08 -> 0.9)
    let sugg = TraceSuggestion::new(
        SuggestionKind::InterleavingActors,
        "test".to_string(),
        0.9,
        "action".to_string(),
    );
    assert_eq!(sugg.severity, SuggestionSeverity::Critical);

    // Invariant variable with medium confidence -> Low (weight 0.7 * 0.6 = 0.42)
    let sugg2 = TraceSuggestion::new(
        SuggestionKind::InvariantVariable,
        "test".to_string(),
        0.6,
        "action".to_string(),
    );
    assert_eq!(sugg2.severity, SuggestionSeverity::Low);
}

#[test]
fn test_trace_suggestion_format_summary() {
    let sugg = TraceSuggestion::new(
        SuggestionKind::MonotonicVariable,
        "Variable x is increasing".to_string(),
        0.9,
        "Use for ordering".to_string(),
    );

    let summary = sugg.format_summary();
    assert!(summary.contains("Monotonic Variable"));
    assert!(summary.contains("Variable x is increasing"));
}

#[test]
fn test_trace_suggestion_format_detailed() {
    let sugg = TraceSuggestion::new(
        SuggestionKind::RepeatingPattern,
        "50% repetition".to_string(),
        0.5,
        "Use compress_trace()".to_string(),
    );

    let detailed = sugg.format_detailed();
    assert!(detailed.contains("Repeating Pattern"));
    assert!(detailed.contains("50%"));
    assert!(detailed.contains("Action: Use compress_trace()"));
}

#[test]
fn test_multi_trace_alignment_to_dot() {
    // Create two counterexamples with slightly different traces
    let mut ce1 = StructuredCounterexample::new();
    let mut state1 = TraceState::new(1);
    state1.action = Some("Init".to_string());
    state1.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 0,
            type_hint: None,
        },
    );
    ce1.trace.push(state1);

    let mut state2 = TraceState::new(2);
    state2.action = Some("Step".to_string());
    state2.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );
    ce1.trace.push(state2);

    let mut ce2 = StructuredCounterexample::new();
    let mut state1_b = TraceState::new(1);
    state1_b.action = Some("Init".to_string());
    state1_b.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 0,
            type_hint: None,
        },
    );
    ce2.trace.push(state1_b);

    let mut state2_b = TraceState::new(2);
    state2_b.action = Some("Step".to_string());
    // Different value - divergence point
    state2_b.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 2,
            type_hint: None,
        },
    );
    ce2.trace.push(state2_b);

    let alignment = align_multiple_traces(
        &[&ce1, &ce2],
        Some(vec!["Run A".to_string(), "Run B".to_string()]),
    );

    let dot = alignment.to_dot();

    // Verify DOT structure
    assert!(dot.starts_with("digraph MultiTraceAlignment"));
    assert!(dot.contains("cluster_trace_0"));
    assert!(dot.contains("cluster_trace_1"));
    assert!(dot.contains("Run A"));
    assert!(dot.contains("Run B"));
    assert!(dot.contains("t0_s1")); // Trace 0, State 1
    assert!(dot.contains("t1_s1")); // Trace 1, State 1
    assert!(dot.contains("State 1"));
    assert!(dot.contains("State 2"));
    // Should have divergence indicator
    assert!(dot.contains("style=dotted"));
    assert!(dot.contains("color=red"));
}

#[test]
fn test_multi_trace_alignment_to_dot_no_divergence() {
    // Create two identical traces
    let mut ce1 = StructuredCounterexample::new();
    let mut state1 = TraceState::new(1);
    state1.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 10,
            type_hint: None,
        },
    );
    ce1.trace.push(state1);

    let mut ce2 = StructuredCounterexample::new();
    let mut state1_b = TraceState::new(1);
    state1_b.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 10,
            type_hint: None,
        },
    );
    ce2.trace.push(state1_b);

    let alignment = align_multiple_traces(&[&ce1, &ce2], None);

    let dot = alignment.to_dot();

    // Verify basic structure
    assert!(dot.contains("digraph"));
    assert!(dot.contains("Trace 1")); // Default label
    assert!(dot.contains("Trace 2")); // Default label
                                      // No divergence indicators (no dotted red edges)
    assert!(!dot.contains("// Divergence points"));
}

#[test]
fn test_multi_trace_alignment_to_dot_three_traces() {
    // Test with three traces
    let mut ce1 = StructuredCounterexample::new();
    let mut ce2 = StructuredCounterexample::new();
    let mut ce3 = StructuredCounterexample::new();

    for i in 1..=3 {
        let mut s1 = TraceState::new(i);
        s1.variables.insert(
            "v".to_string(),
            CounterexampleValue::Int {
                value: (i * 10) as i128,
                type_hint: None,
            },
        );
        ce1.trace.push(s1);

        let mut s2 = TraceState::new(i);
        s2.variables.insert(
            "v".to_string(),
            CounterexampleValue::Int {
                value: (i * 20) as i128,
                type_hint: None,
            },
        );
        ce2.trace.push(s2);

        let mut s3 = TraceState::new(i);
        s3.variables.insert(
            "v".to_string(),
            CounterexampleValue::Int {
                value: (i * 30) as i128,
                type_hint: None,
            },
        );
        ce3.trace.push(s3);
    }

    let alignment = align_multiple_traces(
        &[&ce1, &ce2, &ce3],
        Some(vec![
            "Alpha".to_string(),
            "Beta".to_string(),
            "Gamma".to_string(),
        ]),
    );

    let dot = alignment.to_dot();

    // Should have three clusters
    assert!(dot.contains("cluster_trace_0"));
    assert!(dot.contains("cluster_trace_1"));
    assert!(dot.contains("cluster_trace_2"));
    assert!(dot.contains("Alpha"));
    assert!(dot.contains("Beta"));
    assert!(dot.contains("Gamma"));
    // Each trace should have 3 states
    assert!(dot.contains("t0_s1"));
    assert!(dot.contains("t0_s2"));
    assert!(dot.contains("t0_s3"));
}

#[test]
fn test_trace_diff_to_dot_with_differences() {
    // Create two traces with differences
    let mut trace1 = Vec::new();
    let mut s1 = TraceState::new(1);
    s1.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 0,
            type_hint: None,
        },
    );
    trace1.push(s1);

    let mut s2 = TraceState::new(2);
    s2.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 5,
            type_hint: None,
        },
    );
    trace1.push(s2);

    let mut trace2 = Vec::new();
    let mut s1b = TraceState::new(1);
    s1b.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 0,
            type_hint: None,
        },
    );
    trace2.push(s1b);

    let mut s2b = TraceState::new(2);
    // Different value
    s2b.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 10,
            type_hint: None,
        },
    );
    trace2.push(s2b);

    // Build the diff manually
    let mut diff = TraceDiff::default();
    diff.identical_states.push(1);

    let mut state_diff = StateLevelDiff::default();
    state_diff.value_diffs.insert(
        "x".to_string(),
        (
            CounterexampleValue::Int {
                value: 5,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 10,
                type_hint: None,
            },
        ),
    );
    diff.state_diffs.insert(2, state_diff);

    let dot = diff.to_dot(&trace1, &trace2, "Baseline", "Modified");

    // Verify structure
    assert!(dot.starts_with("digraph TraceDiff"));
    assert!(dot.contains("Baseline"));
    assert!(dot.contains("Modified"));
    assert!(dot.contains("cluster_trace1"));
    assert!(dot.contains("cluster_trace2"));
    // State 1 should be identical (gray dashed edge)
    assert!(dot.contains("style=dashed, color=gray"));
    // State 2 should show difference (red edge)
    assert!(dot.contains("color=red"));
    assert!(dot.contains("1 diff"));
}

#[test]
fn test_trace_diff_to_dot_unique_states() {
    // Trace 1 has state 3, trace 2 doesn't
    let mut trace1 = Vec::new();
    let mut s1 = TraceState::new(1);
    s1.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );
    trace1.push(s1);

    let mut s3 = TraceState::new(3);
    s3.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 3,
            type_hint: None,
        },
    );
    trace1.push(s3);

    let mut trace2 = Vec::new();
    let mut s1b = TraceState::new(1);
    s1b.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );
    trace2.push(s1b);

    // Build diff
    let mut diff = TraceDiff::default();
    diff.identical_states.push(1);
    diff.states_only_in_first.push(3);

    let dot = diff.to_dot(&trace1, &trace2, "Full", "Partial");

    // Should have state 3 only in first trace
    assert!(dot.contains("t1_s3"));
    // Should not have t2_s3
    assert!(!dot.contains("t2_s3"));
    // Legend should exist
    assert!(dot.contains("cluster_legend"));
    assert!(dot.contains("Legend"));
}

#[test]
fn test_trace_diff_to_dot_identical() {
    // Two identical traces
    let mut trace1 = Vec::new();
    let mut s1 = TraceState::new(1);
    s1.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 42,
            type_hint: None,
        },
    );
    trace1.push(s1);

    let mut trace2 = Vec::new();
    let mut s1b = TraceState::new(1);
    s1b.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 42,
            type_hint: None,
        },
    );
    trace2.push(s1b);

    let mut diff = TraceDiff::default();
    diff.identical_states.push(1);

    let dot = diff.to_dot(&trace1, &trace2, "A", "B");

    // Both traces have state 1
    assert!(dot.contains("t1_s1"));
    assert!(dot.contains("t2_s1"));
    // Summary should say equivalent
    assert!(dot.contains("Traces are equivalent"));
}

#[test]
fn test_trace_diff_to_mermaid_with_differences() {
    // Create two traces with some differences
    let mut trace1 = Vec::new();
    let mut s1 = TraceState::new(1);
    s1.action = Some("Init".to_string());
    s1.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 10,
            type_hint: None,
        },
    );
    trace1.push(s1);

    let mut s2 = TraceState::new(2);
    s2.action = Some("Step".to_string());
    s2.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 20,
            type_hint: None,
        },
    );
    trace1.push(s2);

    let mut trace2 = Vec::new();
    let mut s1b = TraceState::new(1);
    s1b.action = Some("Init".to_string());
    s1b.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 10,
            type_hint: None,
        },
    );
    trace2.push(s1b);

    let mut s2b = TraceState::new(2);
    s2b.action = Some("Step".to_string());
    s2b.variables.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 30,
            type_hint: None,
        },
    ); // Different!
    trace2.push(s2b);

    let mut diff = TraceDiff::default();
    diff.identical_states.push(1);
    let mut state_diff = StateLevelDiff::default();
    state_diff.value_diffs.insert(
        "x".to_string(),
        (
            CounterexampleValue::Int {
                value: 20,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 30,
                type_hint: None,
            },
        ),
    );
    diff.state_diffs.insert(2, state_diff);

    let mermaid = diff.to_mermaid(&trace1, &trace2, "Baseline", "Modified");

    // Should have flowchart declaration
    assert!(mermaid.contains("flowchart TB"));
    // Should have subgraphs for both traces
    assert!(mermaid.contains("subgraph T1[\"Baseline\"]"));
    assert!(mermaid.contains("subgraph T2[\"Modified\"]"));
    // Should have state nodes
    assert!(mermaid.contains("t1_s1"));
    assert!(mermaid.contains("t2_s2"));
    // Should have style classes
    assert!(mermaid.contains("classDef identical"));
    assert!(mermaid.contains("classDef different"));
    // Should show diff link
    assert!(mermaid.contains("diff(s)"));
}

#[test]
fn test_trace_diff_to_mermaid_unique_states() {
    // Trace 1 has states 1,2,3; Trace 2 has states 1,2
    let mut trace1 = Vec::new();
    for i in 1..=3 {
        let mut s = TraceState::new(i);
        s.variables.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: i as i128,
                type_hint: None,
            },
        );
        trace1.push(s);
    }

    let mut trace2 = Vec::new();
    for i in 1..=2 {
        let mut s = TraceState::new(i);
        s.variables.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: i as i128,
                type_hint: None,
            },
        );
        trace2.push(s);
    }

    let mut diff = TraceDiff::default();
    diff.identical_states.push(1);
    diff.identical_states.push(2);
    diff.states_only_in_first.push(3);

    let mermaid = diff.to_mermaid(&trace1, &trace2, "Full", "Partial");

    // Should have state 3 only in first
    assert!(mermaid.contains("t1_s3"));
    // State 3 should use unique style
    assert!(mermaid.contains(":::unique") || mermaid.contains("classDef unique"));
}

#[test]
fn test_trace_diff_to_html_basic() {
    // Simple trace diff to HTML
    let mut trace1 = Vec::new();
    let mut s1 = TraceState::new(1);
    s1.variables.insert(
        "val".to_string(),
        CounterexampleValue::Int {
            value: 100,
            type_hint: None,
        },
    );
    trace1.push(s1);

    let mut trace2 = Vec::new();
    let mut s1b = TraceState::new(1);
    s1b.variables.insert(
        "val".to_string(),
        CounterexampleValue::Int {
            value: 200,
            type_hint: None,
        },
    );
    trace2.push(s1b);

    let mut diff = TraceDiff::default();
    let mut state_diff = StateLevelDiff::default();
    state_diff.value_diffs.insert(
        "val".to_string(),
        (
            CounterexampleValue::Int {
                value: 100,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 200,
                type_hint: None,
            },
        ),
    );
    diff.state_diffs.insert(1, state_diff);

    let html = diff.to_html(&trace1, &trace2, "Before", "After", Some("Test Diff"));

    // Should be valid HTML
    assert!(html.contains("<!DOCTYPE html>"));
    assert!(html.contains("<title>Test Diff</title>"));
    // Should include mermaid library
    assert!(html.contains("mermaid"));
    // Should have summary section
    assert!(html.contains("Summary"));
    // Should show stats
    assert!(html.contains("identical states"));
    assert!(html.contains("states with differences"));
    // Should have legend section
    assert!(html.contains("legend-item"));
    // Should have the trace labels
    assert!(html.contains("Before"));
    assert!(html.contains("After"));
    assert!(html.contains("Download Mermaid"));
    assert!(html.contains("Download DOT"));
}

#[test]
fn test_trace_diff_to_html_default_title() {
    let trace1 = vec![TraceState::new(1)];
    let trace2 = vec![TraceState::new(1)];

    let mut diff = TraceDiff::default();
    diff.identical_states.push(1);

    let html = diff.to_html(&trace1, &trace2, "A", "B", None);

    // Should use default title
    assert!(html.contains("<title>Trace Diff Visualization</title>"));
}

#[test]
fn test_trace_diff_mermaid_escapes_quotes() {
    // Test that labels with quotes are properly escaped
    let mut trace1 = Vec::new();
    let mut s1 = TraceState::new(1);
    s1.action = Some("Action with \"quotes\"".to_string());
    trace1.push(s1);

    let trace2 = vec![TraceState::new(1)];

    let diff = TraceDiff::default();
    let mermaid = diff.to_mermaid(&trace1, &trace2, "Label \"quoted\"", "Normal");

    // Quotes should be replaced
    assert!(!mermaid.contains("Label \"quoted\""));
    assert!(mermaid.contains("Label 'quoted'"));
}

#[test]
fn test_structured_counterexample_to_mermaid_basic() {
    let mut ce = StructuredCounterexample::default();
    ce.witness.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 42,
            type_hint: None,
        },
    );

    let mut s1 = TraceState::new(1);
    s1.action = Some("Init".to_string());
    s1.variables.insert(
        "count".to_string(),
        CounterexampleValue::Int {
            value: 0,
            type_hint: None,
        },
    );
    ce.trace.push(s1);

    let mut s2 = TraceState::new(2);
    s2.action = Some("Increment".to_string());
    s2.variables.insert(
        "count".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );
    ce.trace.push(s2);

    let mermaid = ce.to_mermaid();

    // Should have flowchart
    assert!(mermaid.contains("flowchart TB"));
    // Should have witness node
    assert!(mermaid.contains("witness"));
    // Should have trace states
    assert!(mermaid.contains("s0"));
    assert!(mermaid.contains("s1"));
    // Should have style definitions
    assert!(mermaid.contains("classDef witness"));
}

#[test]
fn test_structured_counterexample_to_mermaid_with_failures() {
    let mut ce = StructuredCounterexample::default();

    let mut s1 = TraceState::new(1);
    s1.variables
        .insert("valid".to_string(), CounterexampleValue::Bool(false));
    ce.trace.push(s1);

    ce.failed_checks.push(FailedCheck {
        check_id: "assertion.1".to_string(),
        description: "Assertion failed: valid".to_string(),
        location: Some(SourceLocation {
            file: "test.rs".to_string(),
            line: 10,
            column: Some(5),
        }),
        function: Some("test_func".to_string()),
    });

    let mermaid = ce.to_mermaid();

    // Should have failures node
    assert!(mermaid.contains("failures"));
    assert!(mermaid.contains("Failed Checks"));
    // Should have error style
    assert!(mermaid.contains("classDef error"));
    // Should connect to failures
    assert!(mermaid.contains("==> failures"));
}

#[test]
fn test_structured_counterexample_to_html_basic() {
    let mut ce = StructuredCounterexample::default();
    ce.witness.insert(
        "n".to_string(),
        CounterexampleValue::Int {
            value: -1,
            type_hint: None,
        },
    );

    let s1 = TraceState::new(1);
    ce.trace.push(s1);

    let html = ce.to_html(Some("Test Counterexample"));

    // Should be valid HTML
    assert!(html.contains("<!DOCTYPE html>"));
    assert!(html.contains("<title>Test Counterexample</title>"));
    // Should include mermaid library
    assert!(html.contains("mermaid"));
    // Should have summary with stats
    assert!(html.contains("trace states"));
    assert!(html.contains("witness values"));
    assert!(html.contains("failed checks"));
    // Should have witness table
    assert!(html.contains("witness-table"));
    // Should have footer
    assert!(html.contains("DashProve"));
    assert!(html.contains("Download Mermaid"));
    assert!(html.contains("Download DOT"));
}

#[test]
fn test_structured_counterexample_to_html_with_failures() {
    let mut ce = StructuredCounterexample::default();
    ce.trace.push(TraceState::new(1));
    ce.failed_checks.push(FailedCheck {
        check_id: "property.1".to_string(),
        description: "Property violated".to_string(),
        location: Some(SourceLocation {
            file: "main.rs".to_string(),
            line: 42,
            column: None,
        }),
        function: None,
    });

    let html = ce.to_html(None);

    // Should use default title
    assert!(html.contains("Counterexample Visualization"));
    // Should show failures
    assert!(html.contains("Failed Checks"));
    assert!(html.contains("Property violated"));
    assert!(html.contains("main.rs"));
}

#[test]
fn test_structured_counterexample_to_html_default_title() {
    let ce = StructuredCounterexample::default();
    let html = ce.to_html(None);

    assert!(html.contains("<title>Counterexample Visualization</title>"));
}

#[test]
fn test_abstracted_trace_to_mermaid_basic() {
    let trace = AbstractedTrace {
        segments: vec![
            TraceAbstractionSegment::Concrete(TraceState {
                state_num: 0,
                action: Some("Init".to_string()),
                variables: [(
                    "x".to_string(),
                    CounterexampleValue::Int {
                        value: 0,
                        type_hint: None,
                    },
                )]
                .into_iter()
                .collect(),
            }),
            TraceAbstractionSegment::Abstracted(AbstractedState {
                description: "Loop iterations".to_string(),
                count: 5,
                variables: [(
                    "x".to_string(),
                    AbstractedValue::IntRange { min: 1, max: 5 },
                )]
                .into_iter()
                .collect(),
                original_indices: vec![1, 2, 3, 4, 5],
                common_action: Some("Increment".to_string()),
            }),
            TraceAbstractionSegment::Concrete(TraceState {
                state_num: 6,
                action: Some("Done".to_string()),
                variables: [(
                    "x".to_string(),
                    CounterexampleValue::Int {
                        value: 6,
                        type_hint: None,
                    },
                )]
                .into_iter()
                .collect(),
            }),
        ],
        original_length: 7,
        compression_ratio: 0.57,
    };

    let mermaid = trace.to_mermaid();

    assert!(mermaid.contains("flowchart TB"));
    assert!(mermaid.contains("s0["));
    assert!(mermaid.contains("s1([")); // Stadium shape for abstracted
    assert!(mermaid.contains("s2["));
    assert!(mermaid.contains("classDef abstracted"));
    assert!(mermaid.contains("Loop iterations"));
    assert!(mermaid.contains("5 states"));
}

#[test]
fn test_abstracted_trace_to_mermaid_no_abstraction() {
    let trace = AbstractedTrace {
        segments: vec![TraceAbstractionSegment::Concrete(TraceState {
            state_num: 0,
            action: None,
            variables: HashMap::new(),
        })],
        original_length: 1,
        compression_ratio: 0.0,
    };

    let mermaid = trace.to_mermaid();

    assert!(mermaid.contains("flowchart TB"));
    assert!(mermaid.contains("s0["));
    assert!(!mermaid.contains("s1"));
}

#[test]
fn test_abstracted_trace_to_html_basic() {
    let trace = AbstractedTrace {
        segments: vec![
            TraceAbstractionSegment::Concrete(TraceState {
                state_num: 0,
                action: None,
                variables: [(
                    "x".to_string(),
                    CounterexampleValue::Int {
                        value: 0,
                        type_hint: None,
                    },
                )]
                .into_iter()
                .collect(),
            }),
            TraceAbstractionSegment::Abstracted(AbstractedState {
                description: "Repeated pattern".to_string(),
                count: 3,
                variables: HashMap::new(),
                original_indices: vec![1, 2, 3],
                common_action: None,
            }),
        ],
        original_length: 4,
        compression_ratio: 0.5,
    };

    let html = trace.to_html(Some("Test Abstracted Trace"));

    assert!(html.contains("<title>Test Abstracted Trace</title>"));
    assert!(html.contains("<!DOCTYPE html>"));
    assert!(html.contains("4</strong> original states"));
    assert!(html.contains("2</strong> segments"));
    assert!(html.contains("1</strong> abstracted"));
    assert!(html.contains("50.0%</strong> compression"));
    assert!(html.contains("mermaid.initialize"));
    assert!(html.contains("Repeated pattern"));
    assert!(html.contains("Download Mermaid"));
    assert!(html.contains("Download DOT"));
}

#[test]
fn test_abstracted_trace_to_html_default_title() {
    let trace = AbstractedTrace {
        segments: vec![],
        original_length: 0,
        compression_ratio: 0.0,
    };

    let html = trace.to_html(None);

    assert!(html.contains("<title>Abstracted Trace Visualization</title>"));
}

#[test]
fn test_multi_trace_alignment_to_mermaid_basic() {
    let alignment = MultiTraceAlignment {
        trace_labels: vec!["Trace A".to_string(), "Trace B".to_string()],
        rows: vec![
            MultiTraceAlignmentRow {
                state_num: 0,
                states: vec![
                    Some(TraceState {
                        state_num: 0,
                        action: None,
                        variables: [(
                            "x".to_string(),
                            CounterexampleValue::Int {
                                value: 0,
                                type_hint: None,
                            },
                        )]
                        .into_iter()
                        .collect(),
                    }),
                    Some(TraceState {
                        state_num: 0,
                        action: None,
                        variables: [(
                            "x".to_string(),
                            CounterexampleValue::Int {
                                value: 0,
                                type_hint: None,
                            },
                        )]
                        .into_iter()
                        .collect(),
                    }),
                ],
            },
            MultiTraceAlignmentRow {
                state_num: 1,
                states: vec![
                    Some(TraceState {
                        state_num: 1,
                        action: Some("Step".to_string()),
                        variables: [(
                            "x".to_string(),
                            CounterexampleValue::Int {
                                value: 1,
                                type_hint: None,
                            },
                        )]
                        .into_iter()
                        .collect(),
                    }),
                    Some(TraceState {
                        state_num: 1,
                        action: Some("Step".to_string()),
                        variables: [(
                            "x".to_string(),
                            CounterexampleValue::Int {
                                value: 2,
                                type_hint: None,
                            },
                        )]
                        .into_iter()
                        .collect(),
                    }),
                ],
            },
        ],
        divergence_points: vec![DivergencePoint {
            state_num: 1,
            variable: "x".to_string(),
            values: vec![
                Some(CounterexampleValue::Int {
                    value: 1,
                    type_hint: None,
                }),
                Some(CounterexampleValue::Int {
                    value: 2,
                    type_hint: None,
                }),
            ],
        }],
    };

    let mermaid = alignment.to_mermaid();

    assert!(mermaid.contains("flowchart TB"));
    assert!(mermaid.contains("subgraph T0[\"Trace A\"]"));
    assert!(mermaid.contains("subgraph T1[\"Trace B\"]"));
    assert!(mermaid.contains("t0_s0"));
    assert!(mermaid.contains("t1_s0"));
    assert!(mermaid.contains("divergent")); // Style for divergent states
    assert!(mermaid.contains("Divergence connections"));
}

#[test]
fn test_multi_trace_alignment_to_mermaid_no_divergence() {
    let alignment = MultiTraceAlignment {
        trace_labels: vec!["A".to_string(), "B".to_string()],
        rows: vec![MultiTraceAlignmentRow {
            state_num: 0,
            states: vec![
                Some(TraceState {
                    state_num: 0,
                    action: None,
                    variables: HashMap::new(),
                }),
                Some(TraceState {
                    state_num: 0,
                    action: None,
                    variables: HashMap::new(),
                }),
            ],
        }],
        divergence_points: vec![],
    };

    let mermaid = alignment.to_mermaid();

    assert!(mermaid.contains("flowchart TB"));
    assert!(!mermaid.contains("Divergence connections"));
}

#[test]
fn test_multi_trace_alignment_to_html_basic() {
    let alignment = MultiTraceAlignment {
        trace_labels: vec!["Baseline".to_string(), "Modified".to_string()],
        rows: vec![MultiTraceAlignmentRow {
            state_num: 0,
            states: vec![
                Some(TraceState {
                    state_num: 0,
                    action: None,
                    variables: [(
                        "x".to_string(),
                        CounterexampleValue::Int {
                            value: 0,
                            type_hint: None,
                        },
                    )]
                    .into_iter()
                    .collect(),
                }),
                Some(TraceState {
                    state_num: 0,
                    action: None,
                    variables: [(
                        "x".to_string(),
                        CounterexampleValue::Int {
                            value: 5,
                            type_hint: None,
                        },
                    )]
                    .into_iter()
                    .collect(),
                }),
            ],
        }],
        divergence_points: vec![DivergencePoint {
            state_num: 0,
            variable: "x".to_string(),
            values: vec![
                Some(CounterexampleValue::Int {
                    value: 0,
                    type_hint: None,
                }),
                Some(CounterexampleValue::Int {
                    value: 5,
                    type_hint: None,
                }),
            ],
        }],
    };

    let html = alignment.to_html(Some("Test Multi-Trace Alignment"));

    assert!(html.contains("<title>Test Multi-Trace Alignment</title>"));
    assert!(html.contains("<!DOCTYPE html>"));
    assert!(html.contains("2</strong> traces"));
    assert!(html.contains("1</strong> states"));
    assert!(html.contains("1</strong> divergence points"));
    assert!(html.contains("Baseline"));
    assert!(html.contains("Modified"));
    assert!(html.contains("mermaid.initialize"));
    assert!(html.contains("divergence-table"));
    assert!(html.contains("Download Mermaid"));
    assert!(html.contains("Download DOT"));
}

#[test]
fn test_multi_trace_alignment_to_html_no_divergence() {
    let alignment = MultiTraceAlignment {
        trace_labels: vec!["A".to_string()],
        rows: vec![],
        divergence_points: vec![],
    };

    let html = alignment.to_html(None);

    assert!(html.contains("<title>Multi-Trace Alignment Visualization</title>"));
    assert!(html.contains("No divergence points - all traces are identical."));
}

#[test]
fn test_multi_trace_alignment_to_html_default_title() {
    let alignment = MultiTraceAlignment {
        trace_labels: vec![],
        rows: vec![],
        divergence_points: vec![],
    };

    let html = alignment.to_html(None);

    assert!(html.contains("<title>Multi-Trace Alignment Visualization</title>"));
}

// ============================================================================
// Cross-Category Consistency Tests for Counterexample Helpers
// ============================================================================
// These tests verify that all counterexample builders follow consistent patterns:
// 1. Return None when no error/issue is detected
// 2. Return Some with non-empty failed_checks when error is detected
// 3. Check IDs follow naming conventions (backend_name_category_failure)
// 4. Descriptions contain useful diagnostic information

mod cross_category_consistency {
    use super::*;
    use crate::counterexample::data_quality::{
        build_deepchecks_counterexample, build_evidently_counterexample,
        build_great_expectations_counterexample, build_whylogs_counterexample,
    };
    use crate::counterexample::fairness::{
        build_aequitas_counterexample, build_aif360_counterexample, build_fairlearn_counterexample,
    };
    use crate::counterexample::formal_verification::{
        build_bmc_counterexample, build_model_checker_counterexample,
        build_static_analysis_counterexample, build_symbolic_execution_counterexample,
    };
    use crate::counterexample::guardrails::{
        build_guardrails_ai_counterexample, build_guidance_counterexample,
        build_nemo_guardrails_counterexample,
    };
    use crate::counterexample::interpretability::{
        build_alibi_counterexample, build_captum_counterexample, build_interpretml_counterexample,
        build_lime_counterexample, build_shap_counterexample,
    };
    use crate::counterexample::llm_eval::build_llm_eval_counterexample;
    use crate::counterexample::model_optimization::{
        build_compiler_counterexample, build_inference_counterexample,
        build_quantization_counterexample,
    };
    use crate::counterexample::nn::build_nn_counterexample;
    use crate::counterexample::robustness::{
        build_adversarial_attack_counterexample, build_nn_verification_counterexample,
        build_text_attack_counterexample,
    };
    use serde_json::json;

    // Helper to verify counterexample structure
    fn verify_counterexample_structure(
        cex: &StructuredCounterexample,
        expected_check_id_substring: &str,
    ) {
        // Must have at least one failed check
        assert!(
            !cex.failed_checks.is_empty(),
            "Counterexample should have at least one failed check"
        );

        // Check ID should contain backend identifier
        assert!(
            cex.failed_checks[0]
                .check_id
                .contains(expected_check_id_substring),
            "Check ID '{}' should contain '{}'",
            cex.failed_checks[0].check_id,
            expected_check_id_substring
        );

        // Description should not be empty
        assert!(
            !cex.failed_checks[0].description.is_empty(),
            "Failed check description should not be empty"
        );
    }

    // ========================================================================
    // Formal Verification Helpers Consistency
    // ========================================================================

    #[test]
    fn bmc_helper_returns_none_on_success() {
        assert!(build_bmc_counterexample("VERIFICATION SUCCESSFUL", "", "TestBMC", None).is_none());
    }

    #[test]
    fn bmc_helper_returns_structured_counterexample_on_failure() {
        let cex =
            build_bmc_counterexample("VERIFICATION FAILED\nbuffer overflow", "", "TestBMC", None)
                .expect("Should return counterexample on failure");
        verify_counterexample_structure(&cex, "testbmc");
    }

    #[test]
    fn model_checker_helper_returns_none_on_success() {
        assert!(
            build_model_checker_counterexample("All properties verified", "", "TestMC", None)
                .is_none()
        );
    }

    #[test]
    fn model_checker_helper_returns_structured_counterexample_on_failure() {
        let cex =
            build_model_checker_counterexample("error: deadlock detected", "", "TestMC", None)
                .expect("Should return counterexample on deadlock");
        verify_counterexample_structure(&cex, "testmc");
    }

    #[test]
    fn symbolic_exec_helper_returns_none_on_no_errors() {
        assert!(build_symbolic_execution_counterexample("", "", "TestSE", &[]).is_none());
    }

    #[test]
    fn symbolic_exec_helper_returns_structured_counterexample_on_errors() {
        let errors = vec!["memory error".to_string()];
        let cex = build_symbolic_execution_counterexample("", "", "TestSE", &errors)
            .expect("Should return counterexample on errors");
        verify_counterexample_structure(&cex, "testse");
    }

    #[test]
    fn static_analysis_helper_returns_none_on_no_issues() {
        assert!(build_static_analysis_counterexample(&[], "TestSA").is_none());
    }

    #[test]
    fn static_analysis_helper_returns_structured_counterexample_on_issues() {
        let issues = vec![json!({"bug_type": "NULL_DEREFERENCE", "file": "test.c", "line": 10})];
        let cex = build_static_analysis_counterexample(&issues, "TestSA")
            .expect("Should return counterexample on issues");
        verify_counterexample_structure(&cex, "testsa");
    }

    // ========================================================================
    // Model Optimization Helpers Consistency
    // ========================================================================

    #[test]
    fn quantization_helper_returns_none_on_acceptable_metrics() {
        let result = json!({
            "output_max_diff": 0.05,
            "output_mse": 0.005
        });
        assert!(build_quantization_counterexample(&result, "TestQuant").is_none());
    }

    #[test]
    fn quantization_helper_returns_counterexample_on_exceeding_thresholds() {
        let result = json!({
            "output_max_diff": 0.5,
            "output_mse": 0.05,
            "compression_ratio": 4.0
        });
        let cex = build_quantization_counterexample(&result, "TestQuant")
            .expect("Should return counterexample when thresholds exceeded");
        verify_counterexample_structure(&cex, "testquant");
    }

    #[test]
    fn inference_helper_returns_none_on_consistent_outputs() {
        let result = json!({
            "consistent_outputs": true,
            "max_output_diff": 1e-6
        });
        assert!(build_inference_counterexample(&result, "TestInf").is_none());
    }

    #[test]
    fn inference_helper_returns_counterexample_on_inconsistency() {
        let result = json!({
            "consistent_outputs": false,
            "max_output_diff": 0.1
        });
        let cex = build_inference_counterexample(&result, "TestInf")
            .expect("Should return counterexample on inconsistency");
        verify_counterexample_structure(&cex, "testinf");
    }

    #[test]
    fn compiler_helper_returns_none_on_successful_compilation() {
        // Uses compilation_success, output_correct, and numerical_diff fields
        let result = json!({
            "compilation_success": true,
            "output_correct": true,
            "numerical_diff": 0.0
        });
        assert!(build_compiler_counterexample(&result, "TestComp").is_none());
    }

    #[test]
    fn compiler_helper_returns_counterexample_on_failure() {
        // compilation_success=false or output_correct=false triggers counterexample
        let result = json!({
            "compilation_success": false,
            "output_correct": false,
            "error_message": "unsupported operation"
        });
        let cex = build_compiler_counterexample(&result, "TestComp")
            .expect("Should return counterexample on compile error");
        verify_counterexample_structure(&cex, "testcomp");
    }

    // ========================================================================
    // Robustness Helpers Consistency
    // ========================================================================

    #[test]
    fn adversarial_attack_helper_returns_none_on_robust() {
        // Uses attack_success_rate field (< 0.01 = robust)
        let result = json!({
            "attack_success_rate": 0.0
        });
        assert!(build_adversarial_attack_counterexample(&result, "TestAdv").is_none());
    }

    #[test]
    fn adversarial_attack_helper_returns_counterexample_on_attack_success() {
        // Uses attack_success_rate field (>= 0.01 = attack successful)
        let result = json!({
            "attack_success_rate": 0.8,
            "attack_method": "PGD",
            "epsilon": 0.03
        });
        let cex = build_adversarial_attack_counterexample(&result, "TestAdv")
            .expect("Should return counterexample on attack success");
        verify_counterexample_structure(&cex, "testadv");
    }

    #[test]
    fn nn_verification_helper_returns_none_on_verified() {
        // Uses verified field
        let result = json!({
            "verified": true
        });
        assert!(build_nn_verification_counterexample(&result, "TestNN").is_none());
    }

    #[test]
    fn nn_verification_helper_returns_counterexample_on_violation() {
        // Uses verified field (false = violation)
        let result = json!({
            "verified": false,
            "bounds": {"lower": -0.5, "upper": 1.5}
        });
        let cex = build_nn_verification_counterexample(&result, "TestNN")
            .expect("Should return counterexample on violation");
        verify_counterexample_structure(&cex, "testnn");
    }

    #[test]
    fn text_attack_helper_returns_none_on_robust() {
        // Uses attack_success_rate field (< 0.01 = robust)
        let result = json!({
            "attack_success_rate": 0.0
        });
        assert!(build_text_attack_counterexample(&result, "TestText").is_none());
    }

    #[test]
    fn text_attack_helper_returns_counterexample_on_attack_success() {
        // Uses attack_success_rate field (>= 0.01 = attack successful)
        let result = json!({
            "attack_success_rate": 0.75,
            "original_text": "good movie",
            "adversarial_text": "g00d m0vie"
        });
        let cex = build_text_attack_counterexample(&result, "TestText")
            .expect("Should return counterexample on text attack success");
        verify_counterexample_structure(&cex, "testtext");
    }

    // ========================================================================
    // Data Quality Helpers Consistency
    // ========================================================================

    #[test]
    fn great_expectations_helper_returns_none_on_pass() {
        // Uses success_rate field (1.0 = pass)
        let result = json!({
            "success_rate": 1.0
        });
        assert!(build_great_expectations_counterexample(&result).is_none());
    }

    #[test]
    fn great_expectations_helper_returns_counterexample_on_failure() {
        // Uses success_rate field (< 1.0 = has failures)
        let result = json!({
            "success_rate": 0.7,
            "expectations_passed": 7,
            "expectations_failed": 3,
            "failed_expectations": ["null_check", "range_check", "type_check"]
        });
        let cex = build_great_expectations_counterexample(&result)
            .expect("Should return counterexample on expectation failure");
        verify_counterexample_structure(&cex, "great_expectations");
    }

    #[test]
    fn whylogs_helper_returns_none_on_passing_constraints() {
        // Uses success_rate field (1.0 = pass)
        let result = json!({
            "success_rate": 1.0,
            "constraints_passed": 15,
            "constraints_failed": 0
        });
        assert!(build_whylogs_counterexample(&result).is_none());
    }

    #[test]
    fn whylogs_helper_returns_counterexample_on_failed_constraints() {
        // Uses success_rate field (< 1.0 = has failures)
        let result = json!({
            "success_rate": 0.9,
            "constraints_passed": 9,
            "constraints_failed": 1,
            "constraint_results": [
                {"feature": "age", "constraint": "non_negative", "passed": true},
                {"feature": "income", "constraint": "range", "passed": false}
            ]
        });
        let cex = build_whylogs_counterexample(&result)
            .expect("Should return counterexample on constraint failure");
        verify_counterexample_structure(&cex, "whylogs");
    }

    #[test]
    fn evidently_helper_returns_none_when_no_drift() {
        // Uses drift_detected/drift_score
        let result = json!({
            "drift_detected": false,
            "drift_score": 0.02,
            "drift_threshold": 0.1
        });
        assert!(build_evidently_counterexample(&result).is_none());
    }

    #[test]
    fn evidently_helper_returns_counterexample_when_drift_detected() {
        // Uses drift_detected or drift_score > threshold
        let result = json!({
            "drift_detected": true,
            "drift_score": 0.25,
            "drift_threshold": 0.1,
            "drifted_features": ["feature_a", "feature_b"],
            "total_features": 10,
            "report_type": "data_drift"
        });
        let cex = build_evidently_counterexample(&result)
            .expect("Should return counterexample when drift is detected");
        verify_counterexample_structure(&cex, "evidently");
    }

    #[test]
    fn deepchecks_helper_returns_none_on_pass() {
        // Uses success_rate field (1.0 = pass)
        let result = json!({
            "success_rate": 1.0
        });
        assert!(build_deepchecks_counterexample(&result).is_none());
    }

    #[test]
    fn deepchecks_helper_returns_counterexample_on_failure() {
        // Uses success_rate field (< 1.0 = has failures)
        let result = json!({
            "success_rate": 0.6,
            "checks_passed": 6,
            "checks_failed": 4,
            "failed_checks": ["DataDrift", "FeatureCorrelation"]
        });
        let cex = build_deepchecks_counterexample(&result)
            .expect("Should return counterexample on check failure");
        verify_counterexample_structure(&cex, "deepchecks");
    }

    // ========================================================================
    // Fairness Helpers Consistency
    // ========================================================================

    #[test]
    fn aequitas_helper_returns_none_on_fair_audit() {
        // Uses is_fair flag
        let result = json!({
            "is_fair": true,
            "min_disparity_ratio": 0.95,
            "avg_disparity_ratio": 0.97,
            "disparity_tolerance": 0.9
        });
        assert!(build_aequitas_counterexample(&result).is_none());
    }

    #[test]
    fn aequitas_helper_returns_counterexample_on_bias_detected() {
        // Uses is_fair flag with disparity metrics
        let result = json!({
            "is_fair": false,
            "min_disparity_ratio": 0.6,
            "avg_disparity_ratio": 0.7,
            "disparity_tolerance": 0.8,
            "metrics_by_group": {
                "group_a": {"ppr_disparity": 0.6},
                "group_b": {"ppr_disparity": 0.95}
            }
        });
        let cex = build_aequitas_counterexample(&result)
            .expect("Should return counterexample when disparity falls below tolerance");
        verify_counterexample_structure(&cex, "aequitas");
    }

    #[test]
    fn fairlearn_helper_returns_none_on_fair() {
        // Uses is_fair field
        let result = json!({
            "is_fair": true
        });
        assert!(build_fairlearn_counterexample(&result).is_none());
    }

    #[test]
    fn fairlearn_helper_returns_counterexample_on_unfair() {
        // Uses is_fair field
        let result = json!({
            "is_fair": false,
            "primary_metric_value": 0.3,
            "demographic_parity_difference": 0.25
        });
        let cex = build_fairlearn_counterexample(&result)
            .expect("Should return counterexample on unfairness");
        verify_counterexample_structure(&cex, "fairlearn");
    }

    #[test]
    fn aif360_helper_returns_none_on_fair() {
        // Uses is_fair field
        let result = json!({
            "is_fair": true
        });
        assert!(build_aif360_counterexample(&result).is_none());
    }

    #[test]
    fn aif360_helper_returns_counterexample_on_bias() {
        // Uses is_fair field
        let result = json!({
            "is_fair": false,
            "primary_metric_value": 0.75,
            "statistical_parity_difference": -0.15
        });
        let cex = build_aif360_counterexample(&result)
            .expect("Should return counterexample on bias detected");
        verify_counterexample_structure(&cex, "aif360");
    }

    // ========================================================================
    // Guardrails Helpers Consistency
    // ========================================================================

    #[test]
    fn guidance_helper_returns_none_on_passing_generation() {
        // Uses pass_rate field (>= threshold = pass)
        let result = json!({
            "pass_rate": 0.92,
            "pass_threshold": 0.85,
            "passed": 23,
            "failed": 2,
            "generation_mode": "tool"
        });
        assert!(build_guidance_counterexample(&result).is_none());
    }

    #[test]
    fn guidance_helper_returns_counterexample_on_low_pass_rate() {
        // Uses pass_rate field (< threshold = fail)
        let result = json!({
            "pass_rate": 0.4,
            "pass_threshold": 0.9,
            "errors": ["Missing schema field", "Invalid JSON output"]
        });
        let cex = build_guidance_counterexample(&result)
            .expect("Should return counterexample on validation failure");
        verify_counterexample_structure(&cex, "guidance");
    }

    #[test]
    fn guardrails_ai_helper_returns_none_on_pass() {
        // Uses pass_rate field (>= threshold = pass)
        let result = json!({
            "pass_rate": 1.0,
            "pass_threshold": 0.8
        });
        assert!(build_guardrails_ai_counterexample(&result).is_none());
    }

    #[test]
    fn guardrails_ai_helper_returns_counterexample_on_failure() {
        // Uses pass_rate field (< threshold = fail)
        let result = json!({
            "pass_rate": 0.5,
            "pass_threshold": 0.8,
            "total_validations": 10,
            "failed_validations": 5
        });
        let cex = build_guardrails_ai_counterexample(&result)
            .expect("Should return counterexample on guard failure");
        verify_counterexample_structure(&cex, "guardrails_ai");
    }

    #[test]
    fn nemo_guardrails_helper_returns_none_on_pass() {
        // Uses pass_rate field (>= threshold = pass)
        let result = json!({
            "pass_rate": 1.0,
            "pass_threshold": 0.85
        });
        assert!(build_nemo_guardrails_counterexample(&result).is_none());
    }

    #[test]
    fn nemo_guardrails_helper_returns_counterexample_on_blocked() {
        // Uses pass_rate field (< threshold = fail)
        let result = json!({
            "pass_rate": 0.6,
            "pass_threshold": 0.85,
            "blocked_count": 4,
            "total_count": 10
        });
        let cex = build_nemo_guardrails_counterexample(&result)
            .expect("Should return counterexample on rail violation");
        verify_counterexample_structure(&cex, "nemo_guardrails");
    }

    // ========================================================================
    // Interpretability Helpers Consistency
    // ========================================================================

    #[test]
    fn shap_helper_returns_none_on_expected_behavior() {
        // Uses mean_abs_shap >= importance_threshold to pass
        let result = json!({
            "mean_abs_shap": 0.5,
            "importance_threshold": 0.3
        });
        assert!(build_shap_counterexample(&result).is_none());
    }

    #[test]
    fn shap_helper_returns_counterexample_on_anomaly() {
        // Low mean_abs_shap or high stability_gap triggers counterexample
        let result = json!({
            "mean_abs_shap": 0.1,
            "importance_threshold": 0.3,
            "top_features": ["feature_a", "feature_b"]
        });
        let cex =
            build_shap_counterexample(&result).expect("Should return counterexample on anomaly");
        verify_counterexample_structure(&cex, "shap");
    }

    #[test]
    fn lime_helper_returns_none_on_expected_behavior() {
        // Uses fidelity >= threshold to pass
        let result = json!({
            "fidelity": 0.9,
            "threshold": 0.8,
            "coverage": 0.7
        });
        assert!(build_lime_counterexample(&result).is_none());
    }

    #[test]
    fn lime_helper_returns_counterexample_on_unexpected_explanation() {
        // Low fidelity or low coverage triggers counterexample
        let result = json!({
            "fidelity": 0.5,
            "threshold": 0.8,
            "coverage": 0.4
        });
        let cex = build_lime_counterexample(&result)
            .expect("Should return counterexample on unexpected explanation");
        verify_counterexample_structure(&cex, "lime");
    }

    #[test]
    fn captum_helper_returns_none_on_strong_attribution() {
        // Attribution exceeds threshold and stability gap is acceptable
        let result = json!({
            "attribution_mean": 0.6,
            "attribution_threshold": 0.1,
            "stability_gap": 0.15,
            "method": "integrated_gradients",
            "model_type": "classifier"
        });
        assert!(build_captum_counterexample(&result).is_none());
    }

    #[test]
    fn captum_helper_returns_counterexample_on_low_attribution() {
        // Attribution falls below threshold
        let result = json!({
            "attribution_mean": 0.02,
            "attribution_threshold": 0.1,
            "stability_gap": 0.5,
            "top_feature": 3,
            "method": "integrated_gradients",
            "model_type": "classifier"
        });
        let cex = build_captum_counterexample(&result)
            .expect("Should return counterexample when attributions are weak");
        verify_counterexample_structure(&cex, "captum");
    }

    #[test]
    fn interpretml_helper_returns_none_on_high_importance_and_fidelity() {
        // Global importance above threshold and local fidelity acceptable
        let result = json!({
            "global_importance_mean": 0.4,
            "threshold": 0.3,
            "local_fidelity": 0.9,
            "mode": "global",
            "explainer": "ebm",
            "model_type": "classifier"
        });
        assert!(build_interpretml_counterexample(&result).is_none());
    }

    #[test]
    fn interpretml_helper_returns_counterexample_on_low_fidelity() {
        // Global importance or fidelity below thresholds triggers counterexample
        let result = json!({
            "global_importance_mean": 0.1,
            "threshold": 0.3,
            "local_fidelity": 0.5,
            "mode": "local",
            "explainer": "lime",
            "model_type": "regressor"
        });
        let cex = build_interpretml_counterexample(&result)
            .expect("Should return counterexample when explanations are weak");
        verify_counterexample_structure(&cex, "interpretml");
    }

    #[test]
    fn alibi_helper_returns_none_when_coverage_and_fidelity_sufficient() {
        // Coverage above threshold and fidelity above threshold
        let result = json!({
            "explanation_coverage": 0.85,
            "coverage_threshold": 0.8,
            "fidelity": 0.9,
            "precision_threshold": 0.8,
            "method": "anchors",
            "model_type": "classifier"
        });
        assert!(build_alibi_counterexample(&result).is_none());
    }

    #[test]
    fn alibi_helper_returns_counterexample_on_low_coverage() {
        // Coverage below threshold triggers counterexample
        let result = json!({
            "explanation_coverage": 0.5,
            "coverage_threshold": 0.7,
            "fidelity": 0.9,
            "method": "anchors",
            "model_type": "classifier"
        });
        let cex = build_alibi_counterexample(&result)
            .expect("Should return counterexample on inadequate coverage");
        verify_counterexample_structure(&cex, "alibi");
    }

    // ========================================================================
    // LLM Eval Helper Consistency
    // ========================================================================

    #[test]
    fn llm_eval_helper_returns_none_on_pass() {
        // Uses pass_rate >= pass_threshold to pass
        let result = json!({
            "pass_rate": 0.95,
            "pass_threshold": 0.8
        });
        assert!(build_llm_eval_counterexample("TestEval", &result).is_none());
    }

    #[test]
    fn llm_eval_helper_returns_counterexample_on_failure() {
        // Uses pass_rate < pass_threshold to fail
        let result = json!({
            "pass_rate": 0.5,
            "pass_threshold": 0.8,
            "failed_metrics": ["relevance", "coherence"]
        });
        let cex = build_llm_eval_counterexample("TestEval", &result)
            .expect("Should return counterexample on eval failure");
        verify_counterexample_structure(&cex, "testeval");
    }

    // ========================================================================
    // NN Counterexample Helper Consistency
    // ========================================================================

    #[test]
    fn nn_counterexample_helper_returns_none_on_verified() {
        // Must have counterexample field to produce output
        let result = json!({
            "verified": true,
            "property": "robustness"
        });
        // Third arg is verification_rate (1.0 = fully verified)
        assert!(build_nn_counterexample("TestNNGeneral", &result, 1.0).is_none());
    }

    #[test]
    fn nn_counterexample_helper_returns_counterexample_on_violation() {
        // Must have counterexample field with original_input to produce output
        let result = json!({
            "counterexample": {
                "original_input": [0.1, 0.2, 0.3],
                "counterexample": [0.15, 0.25, 0.35],
                "original_output": 1,
                "adversarial_output": 0
            }
        });
        // Third arg is verification_rate (0.0 = violation found)
        let cex = build_nn_counterexample("TestNNGeneral", &result, 0.0)
            .expect("Should return counterexample on violation");
        verify_counterexample_structure(&cex, "testnngeneral");
    }

    // ========================================================================
    // Cross-Category Pattern Verification
    // ========================================================================

    #[test]
    fn all_helpers_produce_lowercase_backend_names_in_check_ids() {
        // BMC
        let cex = build_bmc_counterexample("VERIFICATION FAILED", "", "MyBackend", None).unwrap();
        assert!(
            cex.failed_checks[0].check_id.contains("mybackend"),
            "Check ID should use lowercase backend name"
        );

        // Model checker
        let cex = build_model_checker_counterexample("error: assertion failed", "", "MyMC", None)
            .unwrap();
        assert!(cex.failed_checks[0].check_id.contains("mymc"));

        // Symbolic execution
        let cex = build_symbolic_execution_counterexample("", "", "MySE", &["error".to_string()])
            .unwrap();
        assert!(cex.failed_checks[0].check_id.contains("myse"));

        // Static analysis
        let cex = build_static_analysis_counterexample(
            &[json!({"bug_type": "test", "file": "x", "line": 1})],
            "MySA",
        )
        .unwrap();
        assert!(cex.failed_checks[0].check_id.contains("mysa"));
    }

    #[test]
    fn all_helpers_handle_special_characters_in_backend_names() {
        // Backend name with space
        let cex = build_bmc_counterexample("VERIFICATION FAILED", "", "My Backend", None).unwrap();
        assert!(
            !cex.failed_checks[0].check_id.contains(' '),
            "Check ID should not contain spaces"
        );

        // Backend name with mixed case
        let cex = build_model_checker_counterexample("error: deadlock", "", "MyModelChecker", None)
            .unwrap();
        assert!(cex.failed_checks[0].check_id.contains("mymodelchecker"));
    }

    #[test]
    fn counterexample_descriptions_contain_backend_name() {
        let cex = build_bmc_counterexample("VERIFICATION FAILED", "", "CBMC", None).unwrap();
        assert!(
            cex.failed_checks[0].description.contains("CBMC"),
            "Description should mention backend name"
        );

        let cex = build_model_checker_counterexample("error: deadlock", "", "SPIN", None).unwrap();
        assert!(cex.failed_checks[0].description.contains("SPIN"));

        let cex = build_symbolic_execution_counterexample("", "", "KLEE", &["error".to_string()])
            .unwrap();
        assert!(cex.failed_checks[0].description.contains("KLEE"));
    }
}

// ==================== Mutation-Killing Tests for types.rs ====================

mod mutation_killing_tests {
    use super::*;

    // Tests for CounterexampleValue::semantically_equal Float comparison
    #[test]
    fn semantically_equal_floats_within_epsilon() {
        let v1 = CounterexampleValue::Float { value: 1.0 };
        let v2 = CounterexampleValue::Float {
            value: 1.0 + f64::EPSILON / 2.0,
        };
        assert!(
            v1.semantically_equal(&v2),
            "Floats within epsilon should be equal"
        );
    }

    #[test]
    fn semantically_equal_floats_beyond_epsilon() {
        let v1 = CounterexampleValue::Float { value: 1.0 };
        let v2 = CounterexampleValue::Float {
            value: 1.0 + f64::EPSILON * 2.0,
        };
        assert!(
            !v1.semantically_equal(&v2),
            "Floats beyond epsilon should not be equal"
        );
    }

    #[test]
    fn semantically_equal_floats_subtraction_order() {
        // Test that v1 - v2 works correctly (catches - with + mutation)
        let v1 = CounterexampleValue::Float { value: 2.0 };
        let v2 = CounterexampleValue::Float { value: 1.0 };
        assert!(
            !v1.semantically_equal(&v2),
            "Different floats should not be equal"
        );
    }

    #[test]
    fn semantically_equal_floats_subtraction_vs_division() {
        // (v1 - v2).abs() should differ from (v1 / v2).abs()
        // For identical values: |1.0 - 1.0| = 0.0 < epsilon (equal)
        //                      |1.0 / 1.0| = 1.0 >= epsilon (would be not equal if / was used)
        let v1 = CounterexampleValue::Float { value: 1.0 };
        let v2 = CounterexampleValue::Float { value: 1.0 };
        assert!(
            v1.semantically_equal(&v2),
            "Identical floats should be equal (catches - vs / mutation)"
        );
    }

    #[test]
    fn semantically_equal_floats_comparison_operators() {
        // Test boundary cases for < vs <= vs == vs >
        let v1 = CounterexampleValue::Float { value: 1.0 };
        let v2 = CounterexampleValue::Float { value: 1.0 };
        // abs(0.0) = 0.0, and 0.0 < EPSILON should be true
        assert!(v1.semantically_equal(&v2), "Same float should be equal");

        // Exact epsilon difference should NOT be equal (< vs <=)
        let v3 = CounterexampleValue::Float { value: 1.0 };
        let v4 = CounterexampleValue::Float {
            value: 1.0 + f64::EPSILON,
        };
        assert!(
            !v3.semantically_equal(&v4),
            "Float at exact epsilon should not be equal"
        );
    }

    #[test]
    fn semantically_equal_uint_requires_both_value_and_type() {
        // Tests && vs || mutation for UInt
        let v1 = CounterexampleValue::UInt {
            value: 42,
            type_hint: Some("u32".to_string()),
        };
        let v2 = CounterexampleValue::UInt {
            value: 42,
            type_hint: Some("u64".to_string()),
        };
        assert!(
            !v1.semantically_equal(&v2),
            "Same value but different types should not be equal"
        );

        let v3 = CounterexampleValue::UInt {
            value: 42,
            type_hint: Some("u32".to_string()),
        };
        let v4 = CounterexampleValue::UInt {
            value: 99,
            type_hint: Some("u32".to_string()),
        };
        assert!(
            !v3.semantically_equal(&v4),
            "Different values same type should not be equal"
        );

        let v5 = CounterexampleValue::UInt {
            value: 42,
            type_hint: Some("u32".to_string()),
        };
        let v6 = CounterexampleValue::UInt {
            value: 42,
            type_hint: Some("u32".to_string()),
        };
        assert!(
            v5.semantically_equal(&v6),
            "Same value and type should be equal"
        );
    }

    #[test]
    fn semantically_equal_function_requires_both_key_and_value_match() {
        // Tests && vs || mutation for Function
        let f1 = CounterexampleValue::Function(vec![(
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
            CounterexampleValue::String("a".to_string()),
        )]);
        let f2 = CounterexampleValue::Function(vec![(
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
            CounterexampleValue::String("b".to_string()),
        )]);
        assert!(
            !f1.semantically_equal(&f2),
            "Same key different value should not be equal"
        );

        let f3 = CounterexampleValue::Function(vec![(
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
            CounterexampleValue::String("a".to_string()),
        )]);
        let f4 = CounterexampleValue::Function(vec![(
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
            CounterexampleValue::String("a".to_string()),
        )]);
        assert!(
            !f3.semantically_equal(&f4),
            "Different key same value should not be equal"
        );

        let f5 = CounterexampleValue::Function(vec![(
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
            CounterexampleValue::String("a".to_string()),
        )]);
        let f6 = CounterexampleValue::Function(vec![(
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
            CounterexampleValue::String("a".to_string()),
        )]);
        assert!(
            f5.semantically_equal(&f6),
            "Same key and value should be equal"
        );
    }

    // Tests for CounterexampleValue::normalize
    #[test]
    fn normalize_sequence_recursively_normalizes() {
        let seq = CounterexampleValue::Sequence(vec![
            CounterexampleValue::Set(vec![
                CounterexampleValue::Int {
                    value: 2,
                    type_hint: None,
                },
                CounterexampleValue::Int {
                    value: 1,
                    type_hint: None,
                },
            ]),
            CounterexampleValue::Int {
                value: 5,
                type_hint: None,
            },
        ]);

        let normalized = seq.normalize();

        // The outer structure should still be a sequence
        if let CounterexampleValue::Sequence(elems) = &normalized {
            assert_eq!(elems.len(), 2);
            // The inner set should have been normalized (sorted)
            if let CounterexampleValue::Set(inner_set) = &elems[0] {
                // Sorted by string representation: "1" < "2"
                assert_eq!(
                    inner_set[0],
                    CounterexampleValue::Int {
                        value: 1,
                        type_hint: None
                    }
                );
                assert_eq!(
                    inner_set[1],
                    CounterexampleValue::Int {
                        value: 2,
                        type_hint: None
                    }
                );
            } else {
                panic!("First element should be a set");
            }
        } else {
            panic!("Should still be a sequence");
        }
    }

    #[test]
    fn normalize_record_recursively_normalizes() {
        let mut inner = HashMap::new();
        inner.insert(
            "z".to_string(),
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
        );

        let mut outer = HashMap::new();
        outer.insert("nested".to_string(), CounterexampleValue::Record(inner));

        let rec = CounterexampleValue::Record(outer);
        let normalized = rec.normalize();

        if let CounterexampleValue::Record(fields) = normalized {
            assert!(fields.contains_key("nested"));
            if let Some(CounterexampleValue::Record(inner_fields)) = fields.get("nested") {
                assert!(inner_fields.contains_key("z"));
            } else {
                panic!("Nested should be a record");
            }
        } else {
            panic!("Should be a record");
        }
    }

    #[test]
    fn normalize_function_recursively_normalizes_and_sorts() {
        let func = CounterexampleValue::Function(vec![
            (
                CounterexampleValue::Int {
                    value: 2,
                    type_hint: None,
                },
                CounterexampleValue::Set(vec![
                    CounterexampleValue::Int {
                        value: 9,
                        type_hint: None,
                    },
                    CounterexampleValue::Int {
                        value: 1,
                        type_hint: None,
                    },
                ]),
            ),
            (
                CounterexampleValue::Int {
                    value: 1,
                    type_hint: None,
                },
                CounterexampleValue::String("first".to_string()),
            ),
        ]);

        let normalized = func.normalize();

        if let CounterexampleValue::Function(mappings) = normalized {
            assert_eq!(mappings.len(), 2);
            // Should be sorted by key: 1 < 2
            assert_eq!(
                mappings[0].0,
                CounterexampleValue::Int {
                    value: 1,
                    type_hint: None
                }
            );
            assert_eq!(
                mappings[1].0,
                CounterexampleValue::Int {
                    value: 2,
                    type_hint: None
                }
            );
            // The set value should be normalized (sorted)
            if let CounterexampleValue::Set(inner) = &mappings[1].1 {
                assert_eq!(
                    inner[0],
                    CounterexampleValue::Int {
                        value: 1,
                        type_hint: None
                    }
                );
            } else {
                panic!("Value should be a set");
            }
        } else {
            panic!("Should be a function");
        }
    }

    // Tests for StructuredCounterexample::summary
    #[test]
    fn summary_with_trace_only() {
        let mut ce = StructuredCounterexample::new();
        ce.trace.push(TraceState::new(1));
        ce.trace.push(TraceState::new(2));
        ce.trace.push(TraceState::new(3));

        let summary = ce.summary();
        assert!(
            summary.contains("Trace: 3 states"),
            "Summary should mention trace"
        );
    }

    #[test]
    fn summary_empty_uses_raw_fallback() {
        let mut ce = StructuredCounterexample::new();
        ce.raw = Some("This is raw output\nSecond line".to_string());

        let summary = ce.summary();
        assert_eq!(
            summary, "This is raw output",
            "Should return first line of raw"
        );
    }

    #[test]
    fn summary_truly_empty() {
        let ce = StructuredCounterexample::new();
        let summary = ce.summary();
        assert_eq!(summary, "Unknown counterexample");
    }

    // Tests for StructuredCounterexample Display
    #[test]
    fn display_uses_summary() {
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "test".to_string(),
            description: "Test failed".to_string(),
            location: None,
            function: None,
        });

        let display = format!("{}", ce);
        let summary = ce.summary();
        assert_eq!(display, summary, "Display should match summary");
        assert!(display.contains("Test failed"));
    }

    #[test]
    fn display_non_empty_vs_default() {
        // Ensure Display doesn't just return Ok(Default::default())
        let mut ce = StructuredCounterexample::new();
        ce.witness.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 42,
                type_hint: None,
            },
        );

        let display = format!("{}", ce);
        assert!(
            !display.is_empty(),
            "Display should not be empty for non-empty counterexample"
        );
        assert!(
            display.contains("42") || display.contains("x"),
            "Display should contain witness info"
        );
    }

    // Test has_structured_data to ensure the ! operator matters
    #[test]
    fn has_structured_data_with_trace() {
        let mut ce = StructuredCounterexample::new();
        assert!(
            !ce.has_structured_data(),
            "Empty counterexample has no content"
        );

        ce.trace.push(TraceState::new(1));
        assert!(
            ce.has_structured_data(),
            "Counterexample with trace should have content"
        );
    }

    #[test]
    fn has_structured_data_with_witness() {
        let mut ce = StructuredCounterexample::new();
        ce.witness.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
        );
        assert!(
            ce.has_structured_data(),
            "Counterexample with witness should have content"
        );
    }

    #[test]
    fn has_structured_data_with_failed_checks() {
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "test".to_string(),
            description: "Failed".to_string(),
            location: None,
            function: None,
        });
        assert!(
            ce.has_structured_data(),
            "Counterexample with failed checks should have content"
        );
    }

    #[test]
    fn has_structured_data_with_playback() {
        let mut ce = StructuredCounterexample::new();
        ce.playback_test = Some("test code".to_string());
        assert!(
            ce.has_structured_data(),
            "Counterexample with playback should have content"
        );
    }

    // Tests for remaining missed mutants
    #[test]
    fn semantically_equal_sets_require_both_directions() {
        // Tests && vs || mutation at line 106
        // s1 = {1, 2}, s2 = {1, 2, 3}
        // s1 is subset of s2 (all of s1 in s2: true)
        // but s2 is NOT subset of s1 (all of s2 in s1: false)
        // So if && becomes ||, they would incorrectly be considered equal
        let s1 = CounterexampleValue::Set(vec![
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
        ]);
        let s2 = CounterexampleValue::Set(vec![
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 3,
                type_hint: None,
            },
        ]);
        // Length check should catch this, but let's also test same-length case
        let s3 = CounterexampleValue::Set(vec![
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
        ]);
        let s4 = CounterexampleValue::Set(vec![
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 3,
                type_hint: None,
            },
        ]);
        // s3 and s4 have same length but different elements
        // all of s3 in s4: 1 is in s4, 2 is NOT in s4 -> false
        // all of s4 in s3: 1 is in s3, 3 is NOT in s3 -> false
        // Both conditions are false, so && and || both return false
        // We need a case where one is true and one is false

        // Let's try: s5 = {1, 2}, s6 = {2, 1} (should be equal, order doesn't matter)
        let s5 = CounterexampleValue::Set(vec![
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
        ]);
        let s6 = CounterexampleValue::Set(vec![
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
        ]);
        assert!(
            s5.semantically_equal(&s6),
            "Sets with same elements in different order should be equal"
        );

        // Different length sets
        assert!(
            !s1.semantically_equal(&s2),
            "Sets with different lengths should not be equal"
        );

        // Same length but different elements
        assert!(
            !s3.semantically_equal(&s4),
            "Sets with different elements should not be equal"
        );

        // Critical test for && vs || mutation:
        // s7 = [1, 1], s8 = [1, 2] (same length due to duplicates)
        // all of s7 in s8: 1 is in s8 (yes for both), -> true
        // all of s8 in s7: 1 is in s7 (yes), 2 is NOT in s7 -> false
        // With &&: false (correct)
        // With ||: true (wrong - mutation not caught without this test)
        let s7 = CounterexampleValue::Set(vec![
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
        ]);
        let s8 = CounterexampleValue::Set(vec![
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
        ]);
        assert!(
            !s7.semantically_equal(&s8),
            "Set with duplicates [1,1] should not equal [1,2] (catches && vs || mutation)"
        );
    }

    #[test]
    fn normalize_record_with_nested_set_normalizes_inner() {
        // Tests that deleting the Record match arm would break normalization
        // If Record arm is deleted, it falls through to _ => self.clone()
        // which means nested Sets wouldn't get sorted
        let mut fields = HashMap::new();
        fields.insert(
            "inner_set".to_string(),
            CounterexampleValue::Set(vec![
                CounterexampleValue::Int {
                    value: 9,
                    type_hint: None,
                },
                CounterexampleValue::Int {
                    value: 1,
                    type_hint: None,
                },
                CounterexampleValue::Int {
                    value: 5,
                    type_hint: None,
                },
            ]),
        );

        let rec = CounterexampleValue::Record(fields);
        let normalized = rec.normalize();

        if let CounterexampleValue::Record(norm_fields) = normalized {
            if let Some(CounterexampleValue::Set(inner_set)) = norm_fields.get("inner_set") {
                // Should be sorted: 1, 5, 9
                assert_eq!(inner_set.len(), 3);
                assert_eq!(
                    inner_set[0],
                    CounterexampleValue::Int {
                        value: 1,
                        type_hint: None
                    }
                );
                assert_eq!(
                    inner_set[1],
                    CounterexampleValue::Int {
                        value: 5,
                        type_hint: None
                    }
                );
                assert_eq!(
                    inner_set[2],
                    CounterexampleValue::Int {
                        value: 9,
                        type_hint: None
                    }
                );
            } else {
                panic!("inner_set should be a Set");
            }
        } else {
            panic!("Should be a Record");
        }
    }
}
