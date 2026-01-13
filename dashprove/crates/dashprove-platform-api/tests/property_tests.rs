//! Property-based tests for dashprove-platform-api
//!
//! Tests invariants and properties that should hold for all API configurations.

use dashprove_platform_api::{
    ApiChecker, ApiState, BuiltinCatalog, ConstraintKind, CudaCatalog, MetalCatalog, PlatformApi,
    PosixCatalog, StateTransition, TemporalRelation,
};
use proptest::prelude::*;

// Strategy for generating valid state names
fn state_name_strategy() -> impl Strategy<Value = String> {
    prop::string::string_regex("[A-Z][a-zA-Z0-9]{0,15}")
        .unwrap()
        .prop_filter("non-empty", |s| !s.is_empty())
}

// Strategy for generating method names
fn method_name_strategy() -> impl Strategy<Value = String> {
    prop::string::string_regex("[a-z][a-zA-Z0-9_]{0,20}")
        .unwrap()
        .prop_filter("non-empty", |s| !s.is_empty())
}

// Strategy for generating API states
#[allow(dead_code)]
fn api_state_strategy() -> impl Strategy<Value = ApiState> {
    (
        state_name_strategy(),
        prop::option::of(any::<String>().prop_map(|s| s.chars().take(50).collect::<String>())),
        any::<bool>(),
        any::<bool>(),
    )
        .prop_map(|(name, desc, is_error, is_terminal)| {
            let mut state = ApiState::new(name);
            if let Some(d) = desc {
                state = state.with_description(d);
            }
            if is_error {
                state = state.as_error();
            }
            if is_terminal {
                state = state.as_terminal();
            }
            state
        })
}

proptest! {
    // Property: ApiState builder pattern is idempotent for terminal/error flags
    #[test]
    fn api_state_terminal_flag_idempotent(name in state_name_strategy()) {
        let state1 = ApiState::new(&name).as_terminal();
        let state2 = ApiState::new(&name).as_terminal().as_terminal();
        prop_assert_eq!(state1.is_terminal, state2.is_terminal);
        prop_assert!(state1.is_terminal);
    }

    #[test]
    fn api_state_error_flag_idempotent(name in state_name_strategy()) {
        let state1 = ApiState::new(&name).as_error();
        let state2 = ApiState::new(&name).as_error().as_error();
        prop_assert_eq!(state1.is_error, state2.is_error);
        prop_assert!(state1.is_error);
    }

    // Property: Adding states with same name overwrites
    #[test]
    fn adding_duplicate_state_overwrites(
        platform in "[A-Z][a-z]+",
        obj in "[A-Z][a-zA-Z]+",
        state_name in state_name_strategy()
    ) {
        let mut api = PlatformApi::new(&platform, &obj);
        api.add_state(ApiState::new(&state_name));
        api.add_state(ApiState::new(&state_name).as_terminal());

        // Should only have one state with that name
        prop_assert_eq!(api.states.len(), 1);
        // The second addition should win
        prop_assert!(api.states.get(&state_name).unwrap().is_terminal);
    }

    // Property: State machine validation catches missing initial state
    #[test]
    fn validation_requires_initial_state(
        platform in "[A-Z][a-z]+",
        obj in "[A-Z][a-zA-Z]+",
        state_name in state_name_strategy()
    ) {
        let mut api = PlatformApi::new(&platform, &obj);
        api.add_state(ApiState::new(&state_name));
        // No initial state set
        let result = api.validate();
        prop_assert!(result.is_err());
        let errors = result.unwrap_err();
        prop_assert!(errors.iter().any(|e| e.contains("initial state")));
    }

    // Property: Valid API with initial state passes validation
    #[test]
    fn valid_api_passes_validation(
        platform in "[A-Z][a-z]+",
        obj in "[A-Z][a-zA-Z]+",
        state_name in state_name_strategy()
    ) {
        let mut api = PlatformApi::new(&platform, &obj);
        api.add_state(ApiState::new(&state_name));
        api.set_initial_state(&state_name);
        let result = api.validate();
        prop_assert!(result.is_ok());
    }

    // Property: Empty call sequence always passes (no violations)
    #[test]
    fn empty_sequence_always_passes(
        platform in "[A-Z][a-z]+",
        obj in "[A-Z][a-zA-Z]+"
    ) {
        let mut api = PlatformApi::new(&platform, &obj);
        api.add_state(ApiState::new("Init"));
        api.set_initial_state("Init");
        let checker = ApiChecker::new(&api);
        let result = checker.check_sequence(&[] as &[&str]);
        prop_assert!(result.passed);
        prop_assert!(result.violations.is_empty());
    }

    // Property: Valid single transition sequence passes
    #[test]
    fn valid_single_transition_passes(
        platform in "[A-Z][a-z]+",
        obj in "[A-Z][a-zA-Z]+",
        method in method_name_strategy(),
        state1 in state_name_strategy(),
        state2 in state_name_strategy()
    ) {
        prop_assume!(state1 != state2);

        let mut api = PlatformApi::new(&platform, &obj);
        api.add_state(ApiState::new(&state1));
        api.add_state(ApiState::new(&state2));
        api.set_initial_state(&state1);
        api.add_transition(StateTransition::new(&method, vec![&state1], &state2));

        let checker = ApiChecker::new(&api);
        let result = checker.check_sequence(&[method.as_str()]);
        prop_assert!(result.passed, "Expected valid transition to pass");
    }

    // Property: Invalid transition from wrong state fails
    #[test]
    fn invalid_transition_from_wrong_state_fails(
        platform in "[A-Z][a-z]+",
        obj in "[A-Z][a-zA-Z]+",
        method in method_name_strategy(),
        state1 in state_name_strategy(),
        state2 in state_name_strategy(),
        state3 in state_name_strategy()
    ) {
        prop_assume!(state1 != state2 && state2 != state3 && state1 != state3);

        let mut api = PlatformApi::new(&platform, &obj);
        api.add_state(ApiState::new(&state1));
        api.add_state(ApiState::new(&state2));
        api.add_state(ApiState::new(&state3));
        api.set_initial_state(&state1);
        // Transition only valid from state2, not state1
        api.add_transition(StateTransition::new(&method, vec![&state2], &state3));

        let checker = ApiChecker::new(&api);
        let result = checker.check_sequence(&[method.as_str()]);
        prop_assert!(!result.passed, "Expected invalid transition to fail");
    }

    // Property: StateTransition builder preserves all fields
    #[test]
    fn state_transition_builder_preserves_fields(
        method in method_name_strategy(),
        from in state_name_strategy(),
        to in state_name_strategy(),
        desc in "[a-zA-Z ]{0,30}",
        pre in "[a-zA-Z ]{0,20}",
        post in "[a-zA-Z ]{0,20}"
    ) {
        let transition = StateTransition::new(&method, vec![&from], &to)
            .with_description(&desc)
            .with_precondition(&pre)
            .with_postcondition(&post);

        prop_assert_eq!(&transition.method, &method);
        prop_assert_eq!(transition.from_states, vec![from]);
        prop_assert_eq!(&transition.to_state, &to);
        prop_assert_eq!(transition.description, Some(desc));
        prop_assert_eq!(transition.preconditions, vec![pre]);
        prop_assert_eq!(transition.postconditions, vec![post]);
    }

    // Property: Multiple transitions from same state are all valid
    #[test]
    fn multiple_transitions_from_same_state(
        platform in "[A-Z][a-z]+",
        obj in "[A-Z][a-zA-Z]+",
        method1 in method_name_strategy(),
        method2 in method_name_strategy(),
        from_state in state_name_strategy(),
        to_state1 in state_name_strategy(),
        to_state2 in state_name_strategy()
    ) {
        prop_assume!(method1 != method2);
        prop_assume!(from_state != to_state1 && from_state != to_state2);

        let mut api = PlatformApi::new(&platform, &obj);
        api.add_state(ApiState::new(&from_state));
        api.add_state(ApiState::new(&to_state1));
        api.add_state(ApiState::new(&to_state2));
        api.set_initial_state(&from_state);
        api.add_transition(StateTransition::new(&method1, vec![&from_state], &to_state1));
        api.add_transition(StateTransition::new(&method2, vec![&from_state], &to_state2));

        let checker = ApiChecker::new(&api);

        // Both should be individually valid
        let result1 = checker.check_sequence(&[method1.as_str()]);
        let result2 = checker.check_sequence(&[method2.as_str()]);

        prop_assert!(result1.passed, "First transition should be valid");
        prop_assert!(result2.passed, "Second transition should be valid");
    }
}

// Non-proptest tests for built-in catalogs

#[test]
fn metal_catalog_all_apis_valid() {
    for mut api in MetalCatalog::apis() {
        assert!(
            api.validate().is_ok(),
            "Metal API {} should be valid",
            api.api_object
        );
    }
}

#[test]
fn cuda_catalog_all_apis_valid() {
    for mut api in CudaCatalog::apis() {
        assert!(
            api.validate().is_ok(),
            "CUDA API {} should be valid",
            api.api_object
        );
    }
}

#[test]
fn posix_catalog_all_apis_valid() {
    for mut api in PosixCatalog::apis() {
        assert!(
            api.validate().is_ok(),
            "POSIX API {} should be valid",
            api.api_object
        );
    }
}

#[test]
fn check_result_severity_counting() {
    let mut api = PlatformApi::new("Test", "Object");
    api.add_state(ApiState::new("Init"));
    api.add_state(ApiState::new("Final").as_terminal());
    api.set_initial_state("Init");
    api.add_transition(StateTransition::new("go", vec!["Init"], "Final"));

    let checker = ApiChecker::new(&api);

    // Valid sequence should have no violations
    let result = checker.check_sequence(&["go"]);
    let counts = result.count_by_severity();
    assert_eq!(counts.values().sum::<usize>(), 0);
}

#[test]
fn constraint_kinds_are_distinguishable() {
    let temporal_before = ConstraintKind::Temporal(TemporalRelation::Before);
    let temporal_after = ConstraintKind::Temporal(TemporalRelation::After);
    let forbidden = ConstraintKind::Forbidden {
        state: "ErrorState".to_string(),
    };
    let at_most_once = ConstraintKind::AtMostOnce;
    let paired = ConstraintKind::Paired;

    // All constraint kinds should be different from each other
    assert_ne!(
        format!("{:?}", temporal_before),
        format!("{:?}", temporal_after)
    );
    assert_ne!(format!("{:?}", temporal_before), format!("{:?}", forbidden));
    assert_ne!(format!("{:?}", at_most_once), format!("{:?}", paired));
}
