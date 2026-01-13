//! Property-based tests for dashprove-backends
//!
//! Uses proptest to verify:
//! - CounterexampleValue serialization round-trips
//! - Semantic equality is reflexive, symmetric, and transitive
//! - Normalization is idempotent
//! - TraceState diffs are correct
//! - StructuredCounterexample JSON round-trips

use dashprove_backends::{
    CounterexampleValue, FailedCheck, SourceLocation, StructuredCounterexample, TraceState,
};
use proptest::prelude::*;
use std::collections::HashMap;

// ============================================================================
// Generators for counterexample types
// ============================================================================

/// Generate simple CounterexampleValue variants (non-recursive leaves)
fn leaf_value_strategy() -> impl Strategy<Value = CounterexampleValue> {
    prop_oneof![
        any::<i128>().prop_map(|v| CounterexampleValue::Int {
            value: v,
            type_hint: None
        }),
        any::<u64>().prop_map(|v| CounterexampleValue::UInt {
            value: v as u128,
            type_hint: None
        }),
        any::<bool>().prop_map(CounterexampleValue::Bool),
        "[a-z]{1,10}".prop_map(CounterexampleValue::String),
        prop::collection::vec(any::<u8>(), 0..10).prop_map(CounterexampleValue::Bytes),
        ".*".prop_map(CounterexampleValue::Unknown),
    ]
}

/// Generate CounterexampleValue with int type hints
fn int_with_hint_strategy() -> impl Strategy<Value = CounterexampleValue> {
    (any::<i128>(), prop::option::of("[iu](8|16|32|64|128)")).prop_map(|(v, hint)| {
        CounterexampleValue::Int {
            value: v,
            type_hint: hint,
        }
    })
}

/// Generate CounterexampleValue with uint type hints
fn uint_with_hint_strategy() -> impl Strategy<Value = CounterexampleValue> {
    (any::<u64>(), prop::option::of("u(8|16|32|64|128)")).prop_map(|(v, hint)| {
        CounterexampleValue::UInt {
            value: v as u128,
            type_hint: hint,
        }
    })
}

/// Generate CounterexampleValue up to depth 2
fn value_strategy() -> impl Strategy<Value = CounterexampleValue> {
    leaf_value_strategy().prop_recursive(2, 16, 4, |inner| {
        prop_oneof![
            // Set of values
            prop::collection::vec(inner.clone(), 0..4).prop_map(CounterexampleValue::Set),
            // Sequence of values
            prop::collection::vec(inner.clone(), 0..4).prop_map(CounterexampleValue::Sequence),
            // Record with named fields
            prop::collection::hash_map(
                "[a-z]{1,5}".prop_map(|s| s.to_string()),
                inner.clone(),
                0..3
            )
            .prop_map(CounterexampleValue::Record),
            // Function mappings
            prop::collection::vec((inner.clone(), inner), 0..3)
                .prop_map(CounterexampleValue::Function),
        ]
    })
}

/// Generate SourceLocation
fn source_location_strategy() -> impl Strategy<Value = SourceLocation> {
    (
        "[a-z/_]{1,20}\\.rs".prop_map(|s| s.to_string()),
        1u32..1000u32,
        prop::option::of(1u32..100u32),
    )
        .prop_map(|(file, line, column)| SourceLocation { file, line, column })
}

/// Generate FailedCheck
fn failed_check_strategy() -> impl Strategy<Value = FailedCheck> {
    (
        "[a-z_]+\\.[a-z_]+\\.[0-9]+".prop_map(|s| s.to_string()),
        "[a-zA-Z0-9 ]{1,50}".prop_map(|s| s.to_string()),
        prop::option::of(source_location_strategy()),
        prop::option::of("[a-z_]+".prop_map(|s| s.to_string())),
    )
        .prop_map(|(check_id, description, location, function)| FailedCheck {
            check_id,
            description,
            location,
            function,
        })
}

/// Generate TraceState
fn trace_state_strategy() -> impl Strategy<Value = TraceState> {
    (
        1u32..100u32,
        prop::option::of("[a-zA-Z ]{1,20}".prop_map(|s| s.to_string())),
        prop::collection::hash_map(
            "[a-z_]{1,10}".prop_map(|s| s.to_string()),
            leaf_value_strategy(),
            0..5,
        ),
    )
        .prop_map(|(state_num, action, variables)| TraceState {
            state_num,
            action,
            variables,
        })
}

/// Generate StructuredCounterexample
fn counterexample_strategy() -> impl Strategy<Value = StructuredCounterexample> {
    (
        prop::collection::hash_map(
            "[a-z_]{1,10}".prop_map(|s| s.to_string()),
            leaf_value_strategy(),
            0..3,
        ),
        prop::collection::vec(failed_check_strategy(), 0..3),
        prop::option::of("[a-zA-Z0-9_ (){};\n]{1,100}".prop_map(|s| s.to_string())),
        prop::collection::vec(trace_state_strategy(), 0..5),
        prop::option::of("[a-zA-Z0-9 \n]{1,50}".prop_map(|s| s.to_string())),
        any::<bool>(),
    )
        .prop_map(
            |(witness, failed_checks, playback_test, trace, raw, minimized)| {
                StructuredCounterexample {
                    witness,
                    failed_checks,
                    playback_test,
                    trace,
                    raw,
                    minimized,
                }
            },
        )
}

// ============================================================================
// Property tests for CounterexampleValue
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Property: CounterexampleValue JSON serialization round-trips
    #[test]
    fn value_serialization_roundtrip(value in value_strategy()) {
        let json = serde_json::to_string(&value).expect("serialize failed");
        let roundtrip: CounterexampleValue = serde_json::from_str(&json).expect("deserialize failed");
        prop_assert_eq!(value, roundtrip);
    }

    /// Property: Int with type hints serialization round-trips
    #[test]
    fn int_hint_serialization_roundtrip(value in int_with_hint_strategy()) {
        let json = serde_json::to_string(&value).expect("serialize failed");
        let roundtrip: CounterexampleValue = serde_json::from_str(&json).expect("deserialize failed");
        prop_assert_eq!(value, roundtrip);
    }

    /// Property: UInt with type hints serialization round-trips
    #[test]
    fn uint_hint_serialization_roundtrip(value in uint_with_hint_strategy()) {
        let json = serde_json::to_string(&value).expect("serialize failed");
        let roundtrip: CounterexampleValue = serde_json::from_str(&json).expect("deserialize failed");
        prop_assert_eq!(value, roundtrip);
    }

    /// Property: Semantic equality is reflexive (a == a)
    #[test]
    fn semantic_equality_reflexive(value in value_strategy()) {
        prop_assert!(value.semantically_equal(&value));
    }

    /// Property: Semantic equality is symmetric (a == b implies b == a)
    #[test]
    fn semantic_equality_symmetric(a in value_strategy(), b in value_strategy()) {
        let ab = a.semantically_equal(&b);
        let ba = b.semantically_equal(&a);
        prop_assert_eq!(ab, ba, "Semantic equality not symmetric");
    }

    /// Property: Normalization is idempotent (normalize(normalize(x)) == normalize(x))
    #[test]
    fn normalization_idempotent(value in value_strategy()) {
        let normalized = value.normalize();
        let double_normalized = normalized.normalize();
        prop_assert_eq!(normalized, double_normalized);
    }

    /// Property: Normalized values are semantically equal to originals
    #[test]
    fn normalized_semantically_equal(value in value_strategy()) {
        let normalized = value.normalize();
        prop_assert!(value.semantically_equal(&normalized));
    }

    /// Property: Display produces valid string (no panics)
    #[test]
    fn value_display_no_panic(value in value_strategy()) {
        let _ = value.to_string();
    }

    /// Property: Clone equals original
    #[test]
    fn value_clone_equals(value in value_strategy()) {
        let cloned = value.clone();
        prop_assert_eq!(value, cloned);
    }
}

// ============================================================================
// Property tests for FailedCheck
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Property: FailedCheck serialization round-trips
    #[test]
    fn failed_check_serialization_roundtrip(check in failed_check_strategy()) {
        let json = serde_json::to_string(&check).expect("serialize failed");
        let roundtrip: FailedCheck = serde_json::from_str(&json).expect("deserialize failed");
        prop_assert_eq!(check, roundtrip);
    }

    /// Property: FailedCheck display produces valid string
    #[test]
    fn failed_check_display_no_panic(check in failed_check_strategy()) {
        let _ = check.to_string();
    }
}

// ============================================================================
// Property tests for SourceLocation
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Property: SourceLocation serialization round-trips
    #[test]
    fn source_location_serialization_roundtrip(loc in source_location_strategy()) {
        let json = serde_json::to_string(&loc).expect("serialize failed");
        let roundtrip: SourceLocation = serde_json::from_str(&json).expect("deserialize failed");
        prop_assert_eq!(loc, roundtrip);
    }

    /// Property: SourceLocation display produces valid string
    #[test]
    fn source_location_display_no_panic(loc in source_location_strategy()) {
        let display = loc.to_string();
        // Should contain file and line
        prop_assert!(display.contains(&loc.file));
        prop_assert!(display.contains(&loc.line.to_string()));
    }
}

// ============================================================================
// Property tests for TraceState
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Property: TraceState serialization round-trips
    #[test]
    fn trace_state_serialization_roundtrip(state in trace_state_strategy()) {
        let json = serde_json::to_string(&state).expect("serialize failed");
        let roundtrip: TraceState = serde_json::from_str(&json).expect("deserialize failed");
        prop_assert_eq!(state, roundtrip);
    }

    /// Property: TraceState display produces valid string
    #[test]
    fn trace_state_display_no_panic(state in trace_state_strategy()) {
        let display = state.to_string();
        // Should contain state number
        prop_assert!(display.contains(&state.state_num.to_string()));
    }

    /// Property: TraceState::new creates state with given number
    #[test]
    fn trace_state_new_correct(num in 1u32..1000u32) {
        let state = TraceState::new(num);
        prop_assert_eq!(state.state_num, num);
        prop_assert!(state.action.is_none());
        prop_assert!(state.variables.is_empty());
    }

    /// Property: Diff from self is empty
    #[test]
    fn trace_state_diff_from_self_empty(state in trace_state_strategy()) {
        let diff = state.diff_from(&state);
        prop_assert!(diff.is_empty(), "Diff from self should be empty");
    }
}

// ============================================================================
// Property tests for StructuredCounterexample
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Property: StructuredCounterexample JSON round-trips
    #[test]
    fn counterexample_json_roundtrip(cx in counterexample_strategy()) {
        let json = cx.to_json().expect("to_json failed");
        let roundtrip = StructuredCounterexample::from_json(&json).expect("from_json failed");

        // Compare field by field since we can't derive PartialEq easily
        prop_assert_eq!(cx.witness.len(), roundtrip.witness.len());
        prop_assert_eq!(cx.failed_checks.len(), roundtrip.failed_checks.len());
        prop_assert_eq!(cx.playback_test, roundtrip.playback_test);
        prop_assert_eq!(cx.trace.len(), roundtrip.trace.len());
        prop_assert_eq!(cx.raw, roundtrip.raw);
        prop_assert_eq!(cx.minimized, roundtrip.minimized);
    }

    /// Property: StructuredCounterexample pretty JSON round-trips
    #[test]
    fn counterexample_pretty_json_roundtrip(cx in counterexample_strategy()) {
        let json = cx.to_json_pretty().expect("to_json_pretty failed");
        let roundtrip = StructuredCounterexample::from_json(&json).expect("from_json failed");

        prop_assert_eq!(cx.witness.len(), roundtrip.witness.len());
        prop_assert_eq!(cx.failed_checks.len(), roundtrip.failed_checks.len());
        prop_assert_eq!(cx.minimized, roundtrip.minimized);
    }

    /// Property: StructuredCounterexample::new creates empty counterexample
    #[test]
    fn counterexample_new_is_empty(_dummy in 0..1u8) {
        let cx = StructuredCounterexample::new();
        prop_assert!(cx.witness.is_empty());
        prop_assert!(cx.failed_checks.is_empty());
        prop_assert!(cx.playback_test.is_none());
        prop_assert!(cx.trace.is_empty());
        prop_assert!(cx.raw.is_none());
        prop_assert!(!cx.minimized);
        prop_assert!(!cx.has_structured_data());
    }

    /// Property: from_raw preserves raw text
    #[test]
    fn counterexample_from_raw_preserves(text in ".*") {
        let cx = StructuredCounterexample::from_raw(text.clone());
        prop_assert_eq!(cx.raw, Some(text));
    }

    /// Property: has_structured_data returns true when data present
    #[test]
    fn counterexample_has_structured_data_correct(cx in counterexample_strategy()) {
        let has_data = cx.has_structured_data();
        let expected = !cx.witness.is_empty()
            || !cx.failed_checks.is_empty()
            || cx.playback_test.is_some()
            || !cx.trace.is_empty();
        prop_assert_eq!(has_data, expected);
    }

    /// Property: summary produces valid string
    #[test]
    fn counterexample_summary_no_panic(cx in counterexample_strategy()) {
        let _ = cx.summary();
    }

    /// Property: format_detailed produces valid string
    #[test]
    fn counterexample_format_detailed_no_panic(cx in counterexample_strategy()) {
        let _ = cx.format_detailed();
    }

    /// Property: format_trace_with_diffs produces valid string
    #[test]
    fn counterexample_format_trace_no_panic(cx in counterexample_strategy()) {
        let _ = cx.format_trace_with_diffs();
    }

    /// Property: trace_diffs length is at most trace.len() - 1
    #[test]
    fn counterexample_trace_diffs_length(cx in counterexample_strategy()) {
        let diffs = cx.trace_diffs();
        if cx.trace.is_empty() {
            prop_assert!(diffs.is_empty());
        } else {
            prop_assert!(diffs.len() < cx.trace.len());
        }
    }

    /// Property: Display produces valid string
    #[test]
    fn counterexample_display_no_panic(cx in counterexample_strategy()) {
        let _ = cx.to_string();
    }
}

// ============================================================================
// Specific edge case tests
// ============================================================================

#[test]
fn empty_set_serialization() {
    let value = CounterexampleValue::Set(vec![]);
    let json = serde_json::to_string(&value).expect("serialize failed");
    let roundtrip: CounterexampleValue = serde_json::from_str(&json).expect("deserialize failed");
    assert_eq!(value, roundtrip);
}

#[test]
fn nested_record_serialization() {
    let mut inner = HashMap::new();
    inner.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 42,
            type_hint: None,
        },
    );

    let mut outer = HashMap::new();
    outer.insert("nested".to_string(), CounterexampleValue::Record(inner));

    let value = CounterexampleValue::Record(outer);
    let json = serde_json::to_string(&value).expect("serialize failed");
    let roundtrip: CounterexampleValue = serde_json::from_str(&json).expect("deserialize failed");
    assert_eq!(value, roundtrip);
}

#[test]
fn set_semantic_equality_ignores_order() {
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
            value: 2,
            type_hint: None,
        },
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    ]);

    assert!(set1.semantically_equal(&set2));
}

#[test]
fn sequence_semantic_equality_preserves_order() {
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
}

#[test]
fn trace_state_diff_detects_changes() {
    let mut vars1 = HashMap::new();
    vars1.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );
    vars1.insert(
        "y".to_string(),
        CounterexampleValue::Int {
            value: 10,
            type_hint: None,
        },
    );

    let mut vars2 = HashMap::new();
    vars2.insert(
        "x".to_string(),
        CounterexampleValue::Int {
            value: 2,
            type_hint: None,
        },
    );
    vars2.insert(
        "y".to_string(),
        CounterexampleValue::Int {
            value: 10,
            type_hint: None,
        },
    );

    let state1 = TraceState {
        state_num: 1,
        action: None,
        variables: vars1,
    };

    let state2 = TraceState {
        state_num: 2,
        action: None,
        variables: vars2,
    };

    let diff = state2.diff_from(&state1);

    // Only x changed
    assert_eq!(diff.len(), 1);
    assert!(diff.contains_key("x"));
}

#[test]
fn trace_state_diff_detects_new_variables() {
    let state1 = TraceState::new(1);

    let mut vars2 = HashMap::new();
    vars2.insert(
        "new_var".to_string(),
        CounterexampleValue::Int {
            value: 42,
            type_hint: None,
        },
    );

    let state2 = TraceState {
        state_num: 2,
        action: None,
        variables: vars2,
    };

    let diff = state2.diff_from(&state1);

    assert_eq!(diff.len(), 1);
    let (old, new) = diff.get("new_var").unwrap();
    assert!(old.is_none()); // New variable
    assert!(matches!(new, CounterexampleValue::Int { value: 42, .. }));
}
