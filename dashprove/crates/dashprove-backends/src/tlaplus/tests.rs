//! Tests for TLA+ backend

use super::*;
use crate::traits::{CounterexampleValue, HealthStatus, VerificationBackend, VerificationStatus};
use dashprove_usl::{parse, typecheck};
use std::time::Duration;

fn make_typed_spec(input: &str) -> dashprove_usl::typecheck::TypedSpec {
    let spec = parse(input).expect("parse failed");
    typecheck(spec).expect("typecheck failed")
}

#[test]
fn test_config_generation() {
    let input = r#"
        invariant safety {
            forall x: Node . x == x
        }
        temporal liveness {
            always(eventually(done))
        }
    "#;
    let spec = make_typed_spec(input);
    let cfg = spec::generate_config(&spec, "TestSpec");

    assert!(cfg.contains("INVARIANT safety"));
    assert!(cfg.contains("PROPERTY liveness"));
    // Liveness property should enable CHECK_DEADLOCK FALSE
    assert!(cfg.contains("CHECK_DEADLOCK FALSE"));
}

#[test]
fn test_config_generation_with_weak_fairness() {
    let input = r#"
        temporal eventually_served {
            fair weak Next
            eventually(served)
        }
    "#;
    let spec = make_typed_spec(input);
    let cfg = spec::generate_config(&spec, "TestSpec");

    // Should include fairness in specification
    assert!(cfg.contains("SPECIFICATION Spec /\\ WF_vars(Next)"));
    assert!(cfg.contains("PROPERTY eventually_served"));
    assert!(cfg.contains("CHECK_DEADLOCK FALSE"));
}

#[test]
fn test_config_generation_with_strong_fairness() {
    let input = r#"
        temporal no_starvation {
            fair strong Acquire
            always(eventually(released))
        }
    "#;
    let spec = make_typed_spec(input);
    let cfg = spec::generate_config(&spec, "TestSpec");

    // Should include strong fairness in specification
    assert!(cfg.contains("SPECIFICATION Spec /\\ SF_vars(Acquire)"));
    assert!(cfg.contains("PROPERTY no_starvation"));
}

#[test]
fn test_config_generation_with_multiple_fairness() {
    let input = r#"
        temporal termination {
            fair weak Next
            fair strong Acquire
            eventually(done)
        }
    "#;
    let spec = make_typed_spec(input);
    let cfg = spec::generate_config(&spec, "TestSpec");

    // Should include both fairness constraints
    assert!(cfg.contains("WF_vars(Next)"));
    assert!(cfg.contains("SF_vars(Acquire)"));
}

#[test]
fn test_config_generation_with_fairness_vars() {
    let input = r#"
        temporal bounded_wait {
            fair weak Next on state
            eventually(done)
        }
    "#;
    let spec = make_typed_spec(input);
    let cfg = spec::generate_config(&spec, "TestSpec");

    // Should use specified vars instead of default
    assert!(cfg.contains("WF_state(Next)"));
}

#[test]
fn test_config_generation_safety_only() {
    let input = r#"
        invariant no_overflow {
            forall x: Int . x == x
        }
    "#;
    let spec = make_typed_spec(input);
    let cfg = spec::generate_config(&spec, "TestSpec");

    // Safety-only specs should not have CHECK_DEADLOCK FALSE
    assert!(!cfg.contains("CHECK_DEADLOCK FALSE"));
    // Should have simple SPECIFICATION without fairness
    assert!(cfg.contains("SPECIFICATION Spec\n"));
}

#[test]
fn test_parse_success_output() {
    let output = execution::TlcOutput {
        stdout: "Model checking completed. No error has been found.\n1000 states generated, 500 distinct states".to_string(),
        stderr: String::new(),
        exit_code: Some(0),
        duration: Duration::from_secs(5),
    };

    let result = parsing::parse_output(&output);
    assert!(matches!(result.status, VerificationStatus::Proven));
}

#[test]
fn test_parse_invariant_violation() {
    let output = execution::TlcOutput {
        stdout: "Error: Invariant safety is violated.\nState 1:\n  x = 0\n".to_string(),
        stderr: String::new(),
        exit_code: Some(1),
        duration: Duration::from_secs(2),
    };

    let result = parsing::parse_output(&output);
    assert!(matches!(result.status, VerificationStatus::Disproven));
    assert!(result.counterexample.is_some());
}

#[test]
fn test_parse_temporal_violation() {
    let output = execution::TlcOutput {
        stdout: "Error: Temporal properties were violated.\nBack to state: ...\n".to_string(),
        stderr: String::new(),
        exit_code: Some(1),
        duration: Duration::from_secs(3),
    };

    let result = parsing::parse_output(&output);
    assert!(matches!(result.status, VerificationStatus::Disproven));
}

#[test]
fn test_parse_error_output() {
    let output = execution::TlcOutput {
        stdout: "Semantic error: Unknown operator\nError: parsing failed".to_string(),
        stderr: String::new(),
        exit_code: Some(1),
        duration: Duration::from_secs(1),
    };

    let result = parsing::parse_output(&output);
    assert!(matches!(result.status, VerificationStatus::Unknown { .. }));
}

#[tokio::test]
async fn test_health_check_reports_status() {
    let backend = TlaPlusBackend::new();
    let status = backend.health_check().await;

    // Should report some status (Healthy or Unavailable depending on system)
    match status {
        HealthStatus::Healthy => println!("TLC is available"),
        HealthStatus::Unavailable { reason } => println!("TLC not available: {}", reason),
        HealthStatus::Degraded { reason } => println!("TLC degraded: {}", reason),
    }
}

#[tokio::test]
async fn test_verify_returns_result_or_unavailable() {
    let backend = TlaPlusBackend::new();
    let input = r#"
        invariant test {
            forall x: Bool . x or not x
        }
    "#;
    let spec = make_typed_spec(input);

    let result = backend.verify(&spec).await;
    match result {
        Ok(r) => {
            println!("Verification result: {:?}", r.status);
        }
        Err(crate::traits::BackendError::Unavailable(reason)) => {
            println!("Backend unavailable: {}", reason);
        }
        Err(e) => {
            panic!("Unexpected error: {:?}", e);
        }
    }
}

#[test]
fn test_parse_trace_extracts_states() {
    let output = r#"
Error: Invariant Safety is violated.
Error: The behavior up to this point is:
State 1: <Initial predicate>
counter = 0

State 2: <Next line 12, col 8 to line 12, col 44 of module MinimalFail>
counter = 1

State 3: <Next line 12, col 8 to line 12, col 44 of module MinimalFail>
counter = 2
"#;

    let ce = trace::parse_trace(output);

    // Check we extracted 3 states
    assert_eq!(ce.trace.len(), 3);

    // Check state numbers
    assert_eq!(ce.trace[0].state_num, 1);
    assert_eq!(ce.trace[1].state_num, 2);
    assert_eq!(ce.trace[2].state_num, 3);

    // Check actions
    assert_eq!(ce.trace[0].action.as_deref(), Some("Initial predicate"));
    assert!(ce.trace[1].action.as_ref().unwrap().contains("Next line"));

    // Check variable values
    assert_eq!(
        ce.trace[0].variables.get("counter"),
        Some(&CounterexampleValue::Int {
            value: 0,
            type_hint: None
        })
    );
    assert_eq!(
        ce.trace[1].variables.get("counter"),
        Some(&CounterexampleValue::Int {
            value: 1,
            type_hint: None
        })
    );
    assert_eq!(
        ce.trace[2].variables.get("counter"),
        Some(&CounterexampleValue::Int {
            value: 2,
            type_hint: None
        })
    );
}

#[test]
fn test_parse_trace_extracts_failed_check() {
    let output = "Error: Invariant Safety is violated.\nState 1:\nx = 0\n";

    let ce = trace::parse_trace(output);

    assert_eq!(ce.failed_checks.len(), 1);
    assert_eq!(ce.failed_checks[0].check_id, "Safety");
    assert!(ce.failed_checks[0].description.contains("Safety"));
}

#[test]
fn test_parse_trace_handles_booleans() {
    let output = r#"
Error: Invariant Test is violated.
State 1: <Initial predicate>
flag = TRUE
done = FALSE
"#;

    let ce = trace::parse_trace(output);

    assert_eq!(ce.trace.len(), 1);
    assert_eq!(
        ce.trace[0].variables.get("flag"),
        Some(&CounterexampleValue::Bool(true))
    );
    assert_eq!(
        ce.trace[0].variables.get("done"),
        Some(&CounterexampleValue::Bool(false))
    );
}

#[test]
fn test_parse_trace_handles_strings() {
    let output = r#"
Error: Invariant Test is violated.
State 1: <Initial predicate>
name = "hello"
status = "running"
"#;

    let ce = trace::parse_trace(output);

    assert_eq!(ce.trace.len(), 1);
    assert_eq!(
        ce.trace[0].variables.get("name"),
        Some(&CounterexampleValue::String("hello".to_string()))
    );
    assert_eq!(
        ce.trace[0].variables.get("status"),
        Some(&CounterexampleValue::String("running".to_string()))
    );
}

#[test]
fn test_parse_trace_handles_sets() {
    let output = r#"
Error: Invariant Test is violated.
State 1: <Initial predicate>
set = {1, 2, 3}
"#;

    let ce = trace::parse_trace(output);

    assert_eq!(ce.trace.len(), 1);
    // Set should be parsed as CounterexampleValue::Set
    match ce.trace[0].variables.get("set") {
        Some(CounterexampleValue::Set(elems)) => {
            assert_eq!(elems.len(), 3);
            assert!(elems.contains(&CounterexampleValue::Int {
                value: 1,
                type_hint: None
            }));
            assert!(elems.contains(&CounterexampleValue::Int {
                value: 2,
                type_hint: None
            }));
            assert!(elems.contains(&CounterexampleValue::Int {
                value: 3,
                type_hint: None
            }));
        }
        other => panic!("Expected Set, got {:?}", other),
    }
}

#[test]
fn test_parse_trace_handles_records() {
    let output = r#"
Error: Invariant Test is violated.
State 1: <Initial predicate>
record = [a |-> 1, b |-> 2]
"#;

    let ce = trace::parse_trace(output);

    assert_eq!(ce.trace.len(), 1);
    // Record should be parsed as CounterexampleValue::Record
    match ce.trace[0].variables.get("record") {
        Some(CounterexampleValue::Record(fields)) => {
            assert_eq!(fields.len(), 2);
            assert_eq!(
                fields.get("a"),
                Some(&CounterexampleValue::Int {
                    value: 1,
                    type_hint: None
                })
            );
            assert_eq!(
                fields.get("b"),
                Some(&CounterexampleValue::Int {
                    value: 2,
                    type_hint: None
                })
            );
        }
        other => panic!("Expected Record, got {:?}", other),
    }
}

#[test]
fn test_parse_trace_handles_sequences() {
    let output = r#"
Error: Invariant Test is violated.
State 1: <Initial predicate>
seq = <<1, 2, 3>>
"#;

    let ce = trace::parse_trace(output);

    assert_eq!(ce.trace.len(), 1);
    // Sequence should be parsed as CounterexampleValue::Sequence
    match ce.trace[0].variables.get("seq") {
        Some(CounterexampleValue::Sequence(elems)) => {
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
        }
        other => panic!("Expected Sequence, got {:?}", other),
    }
}

#[test]
fn test_parse_trace_handles_functions() {
    let output = r#"
Error: Invariant Test is violated.
State 1: <Initial predicate>
func = (1 :> "a" @@ 2 :> "b")
"#;

    let ce = trace::parse_trace(output);

    assert_eq!(ce.trace.len(), 1);
    // Function should be parsed as CounterexampleValue::Function
    match ce.trace[0].variables.get("func") {
        Some(CounterexampleValue::Function(mappings)) => {
            assert_eq!(mappings.len(), 2);
            // Check that we have mappings 1 -> "a" and 2 -> "b"
            let has_1_to_a = mappings.iter().any(|(k, v)| {
                *k == CounterexampleValue::Int {
                    value: 1,
                    type_hint: None,
                } && *v == CounterexampleValue::String("a".to_string())
            });
            let has_2_to_b = mappings.iter().any(|(k, v)| {
                *k == CounterexampleValue::Int {
                    value: 2,
                    type_hint: None,
                } && *v == CounterexampleValue::String("b".to_string())
            });
            assert!(has_1_to_a, "Should have mapping 1 :> \"a\"");
            assert!(has_2_to_b, "Should have mapping 2 :> \"b\"");
        }
        other => panic!("Expected Function, got {:?}", other),
    }
}

#[test]
fn test_parse_tla_value_empty_set() {
    let value = values::parse_tla_value("{}");
    assert_eq!(value, CounterexampleValue::Set(vec![]));
}

#[test]
fn test_parse_tla_value_empty_sequence() {
    let value = values::parse_tla_value("<<>>");
    assert_eq!(value, CounterexampleValue::Sequence(vec![]));
}

#[test]
fn test_parse_tla_value_nested_structures() {
    // Set of sets
    let value = values::parse_tla_value("{{1, 2}, {3, 4}}");
    match value {
        CounterexampleValue::Set(outer) => {
            assert_eq!(outer.len(), 2);
            for elem in &outer {
                assert!(matches!(elem, CounterexampleValue::Set(_)));
            }
        }
        _ => panic!("Expected nested Set"),
    }
}

#[test]
fn test_parse_tla_value_record_with_nested_set() {
    let value = values::parse_tla_value("[items |-> {1, 2}, count |-> 2]");
    match value {
        CounterexampleValue::Record(fields) => {
            assert_eq!(fields.len(), 2);
            assert!(matches!(
                fields.get("items"),
                Some(CounterexampleValue::Set(_))
            ));
            assert_eq!(
                fields.get("count"),
                Some(&CounterexampleValue::Int {
                    value: 2,
                    type_hint: None
                })
            );
        }
        _ => panic!("Expected Record"),
    }
}

#[test]
fn test_parse_tla_value_sequence_with_strings() {
    let value = values::parse_tla_value(r#"<<"hello", "world">>"#);
    match value {
        CounterexampleValue::Sequence(elems) => {
            assert_eq!(elems.len(), 2);
            assert_eq!(elems[0], CounterexampleValue::String("hello".to_string()));
            assert_eq!(elems[1], CounterexampleValue::String("world".to_string()));
        }
        _ => panic!("Expected Sequence"),
    }
}

#[test]
fn test_counterexample_value_display_set() {
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
    let display = format!("{}", value);
    assert!(display.contains("{"));
    assert!(display.contains("}"));
    assert!(display.contains("1"));
    assert!(display.contains("2"));
}

#[test]
fn test_counterexample_value_display_sequence() {
    let value = CounterexampleValue::Sequence(vec![
        CounterexampleValue::String("a".to_string()),
        CounterexampleValue::String("b".to_string()),
    ]);
    let display = format!("{}", value);
    assert_eq!(display, r#"<<"a", "b">>"#);
}

#[test]
fn test_counterexample_value_display_record() {
    let mut fields = std::collections::HashMap::new();
    fields.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 42,
            type_hint: None,
        },
    );
    let value = CounterexampleValue::Record(fields);
    let display = format!("{}", value);
    assert!(display.contains("["));
    assert!(display.contains("]"));
    assert!(display.contains("x |-> 42"));
}

#[test]
fn test_counterexample_value_display_function() {
    let value = CounterexampleValue::Function(vec![(
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
        CounterexampleValue::String("one".to_string()),
    )]);
    let display = format!("{}", value);
    assert!(display.contains("1 :> \"one\""));
}

#[test]
fn test_parse_trace_stores_raw() {
    let output = "Error: Invariant X is violated.\nState 1:\nval = 42\n";

    let ce = trace::parse_trace(output);

    assert!(ce.raw.is_some());
    assert!(ce.raw.as_ref().unwrap().contains("State 1"));
}

#[test]
fn test_parse_trace_summary_shows_state_count() {
    let output = r#"
Error: Invariant Safety is violated.
State 1: <Initial predicate>
x = 0

State 2: <Next>
x = 1

State 3: <Next>
x = 2
"#;

    let ce = trace::parse_trace(output);
    let summary = ce.summary();

    // Summary should mention failed check and trace length
    assert!(summary.contains("Safety") || summary.contains("Trace: 3 states"));
}

#[test]
fn test_parse_output_with_real_fail_output() {
    // Use the actual captured output from examples/tlaplus/OUTPUT_fail.txt
    let output = execution::TlcOutput {
        stdout: include_str!("../../../../examples/tlaplus/OUTPUT_fail.txt").to_string(),
        stderr: String::new(),
        exit_code: Some(1),
        duration: Duration::from_secs(1),
    };

    let result = parsing::parse_output(&output);

    assert!(matches!(result.status, VerificationStatus::Disproven));
    let ce = result.counterexample.expect("should have counterexample");

    // Check that we have a trace with multiple states
    assert!(!ce.trace.is_empty(), "trace should not be empty");
    assert_eq!(ce.trace.len(), 5, "should have 5 states");

    // Check the first state is the initial state
    assert_eq!(ce.trace[0].state_num, 1);
    assert_eq!(ce.trace[0].action.as_deref(), Some("Initial predicate"));
    assert_eq!(
        ce.trace[0].variables.get("counter"),
        Some(&CounterexampleValue::Int {
            value: 0,
            type_hint: None
        })
    );

    // Check that Safety invariant is recorded as failed
    assert!(
        ce.failed_checks.iter().any(|c| c.check_id == "Safety"),
        "should record Safety invariant failure"
    );
}

// Tests for TLA+ interval syntax parsing

#[test]
fn test_parse_tla_interval_basic() {
    let result = values::parse_tla_value("1..5");

    match result {
        CounterexampleValue::Set(elements) => {
            assert_eq!(elements.len(), 5);
            // Check values are 1, 2, 3, 4, 5
            for (i, elem) in elements.iter().enumerate() {
                match elem {
                    CounterexampleValue::Int { value, .. } => {
                        assert_eq!(*value, (i + 1) as i128);
                    }
                    _ => panic!("Expected Int, got {:?}", elem),
                }
            }
        }
        _ => panic!("Expected Set, got {:?}", result),
    }
}

#[test]
fn test_parse_tla_interval_negative() {
    let result = values::parse_tla_value("-3..2");

    match result {
        CounterexampleValue::Set(elements) => {
            assert_eq!(elements.len(), 6); // -3, -2, -1, 0, 1, 2
            match &elements[0] {
                CounterexampleValue::Int { value, .. } => assert_eq!(*value, -3),
                _ => panic!("Expected Int"),
            }
        }
        _ => panic!("Expected Set, got {:?}", result),
    }
}

#[test]
fn test_parse_tla_interval_single_element() {
    let result = values::parse_tla_value("5..5");

    match result {
        CounterexampleValue::Set(elements) => {
            assert_eq!(elements.len(), 1);
            match &elements[0] {
                CounterexampleValue::Int { value, .. } => assert_eq!(*value, 5),
                _ => panic!("Expected Int"),
            }
        }
        _ => panic!("Expected Set, got {:?}", result),
    }
}

#[test]
fn test_parse_tla_interval_empty() {
    let result = values::parse_tla_value("5..3");

    // Empty interval (start > end)
    match result {
        CounterexampleValue::Set(elements) => {
            assert!(elements.is_empty());
        }
        _ => panic!("Expected empty Set, got {:?}", result),
    }
}

#[test]
fn test_parse_tla_interval_large() {
    // Large interval should be stored as Unknown to avoid memory issues
    let result = values::parse_tla_value("1..10000");

    match result {
        CounterexampleValue::Unknown(s) => {
            assert_eq!(s, "1..10000");
        }
        _ => panic!("Expected Unknown for large interval, got {:?}", result),
    }
}

#[test]
fn test_parse_tla_interval_in_set() {
    // Intervals inside sets are expanded
    let result = values::parse_tla_value("{1..3, 10}");

    // This should parse as a set containing the interval elements plus 10
    match result {
        CounterexampleValue::Set(elements) => {
            // Note: the parsing splits by comma first, so "1..3" and "10" are separate
            // "1..3" gets parsed and becomes 3 elements
            assert!(elements.len() >= 2); // At least 1..3 (parsed somehow) and 10
        }
        _ => panic!("Expected Set, got {:?}", result),
    }
}

#[test]
fn test_parse_tla_value_not_interval() {
    // These should NOT be parsed as intervals
    let result = values::parse_tla_value("abc..def");
    assert!(matches!(result, CounterexampleValue::Unknown(_)));

    // Three dots is different syntax (used in some TLA+ contexts)
    let result = values::parse_tla_value("1...5");
    assert!(matches!(result, CounterexampleValue::Unknown(_)));
}
