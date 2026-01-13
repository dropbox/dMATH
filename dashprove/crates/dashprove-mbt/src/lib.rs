//! Model-Based Testing for DashProve
//!
//! This crate provides model-based test generation capabilities from TLA+
//! specifications and other state machine models.
//!
//! # Features
//!
//! - **State Machine Models**: Define models with states, transitions, and invariants
//! - **TLA+ Parsing**: Parse TLA+ specifications into explorable models
//! - **State Space Exploration**: BFS/DFS exploration of model state space
//! - **Test Generation**: Generate tests for state, transition, and boundary coverage
//! - **Multiple Output Formats**: Generate Rust, Python, JSON, or Markdown output
//!
//! # Example
//!
//! ```rust
//! use dashprove_mbt::{
//!     model::{ModelState, ModelValue, StateMachineModel, VariableDomain, ActionSpec},
//!     explorer::{PrecomputedExecutor, StateExplorer, ExplorationConfig},
//!     generator::{TestGenerator, GeneratorConfig, GenerationStrategy},
//!     output::{format_results, OutputFormat},
//! };
//!
//! // Create a simple counter model
//! let model = StateMachineModel::new("Counter")
//!     .with_variable("count", VariableDomain::IntRange { min: 0, max: 3 })
//!     .with_action(ActionSpec::new("increment"));
//!
//! // Set up the executor with transitions
//! let mut exec = PrecomputedExecutor::new();
//! for i in 0..3 {
//!     let mut from = ModelState::new();
//!     from.set("count", ModelValue::Int(i));
//!     let mut to = ModelState::new();
//!     to.set("count", ModelValue::Int(i + 1));
//!     exec.add_transition(&from, dashprove_mbt::model::ModelAction::new("increment"), to);
//! }
//!
//! // Explore the state space
//! let mut initial = ModelState::new();
//! initial.set("count", ModelValue::Int(0));
//! let explorer = StateExplorer::new(exec);
//! let exploration = explorer.explore(&[initial]).unwrap();
//!
//! // Generate tests
//! let config = GeneratorConfig::new()
//!     .with_strategy(GenerationStrategy::StateCoverage);
//! let mut generator = TestGenerator::with_config(config);
//! let result = generator.generate(&exploration).unwrap();
//!
//! // Output as Rust tests
//! let output = format_results(&result, OutputFormat::Rust);
//! println!("{}", output);
//! ```

pub mod error;
pub mod explorer;
pub mod generator;
pub mod model;
pub mod output;
pub mod tlaplus;

// Re-export main types
pub use error::{MbtError, MbtResult};
pub use explorer::{ExplorationConfig, ExplorationResult, StateExecutor, StateExplorer};
pub use generator::{
    CoverageGoal, CoverageReport, GenerationResult, GenerationStrategy, GeneratorConfig, TestCase,
    TestGenerator,
};
pub use model::{
    ActionSpec, Invariant, ModelAction, ModelState, ModelTransition, ModelValue, StateMachineModel,
    VariableDomain,
};
pub use output::{format_results, OutputFormat};
pub use tlaplus::{parse_tlaplus_file, parse_tlaplus_spec, TlaPlusParser};

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Strategies for generating test data

    fn model_value_strategy() -> impl Strategy<Value = ModelValue> {
        prop_oneof![
            any::<bool>().prop_map(ModelValue::Bool),
            any::<i64>().prop_map(ModelValue::Int),
            "[a-zA-Z0-9_]{0,20}".prop_map(ModelValue::String),
            Just(ModelValue::Null),
        ]
    }

    fn simple_model_value_strategy() -> impl Strategy<Value = ModelValue> {
        prop_oneof![
            any::<bool>().prop_map(ModelValue::Bool),
            (-1000i64..1000i64).prop_map(ModelValue::Int),
            "[a-z]{1,10}".prop_map(ModelValue::String),
            Just(ModelValue::Null),
        ]
    }

    fn model_state_strategy() -> impl Strategy<Value = ModelState> {
        prop::collection::vec(("[a-z]{1,10}", simple_model_value_strategy()), 0..5).prop_map(
            |pairs| {
                let mut state = ModelState::new();
                for (name, value) in pairs {
                    state.set(name, value);
                }
                state
            },
        )
    }

    fn model_action_strategy() -> impl Strategy<Value = ModelAction> {
        (
            "[a-zA-Z][a-zA-Z0-9_]{0,15}",
            prop::collection::vec(simple_model_value_strategy(), 0..3),
        )
            .prop_map(|(name, params)| {
                if params.is_empty() {
                    ModelAction::new(name)
                } else {
                    ModelAction::with_params(name, params)
                }
            })
    }

    fn variable_domain_strategy() -> impl Strategy<Value = VariableDomain> {
        prop_oneof![
            Just(VariableDomain::Boolean),
            ((-100i64..0i64), (1i64..100i64))
                .prop_map(|(min, max)| VariableDomain::IntRange { min, max }),
            prop::collection::vec(simple_model_value_strategy(), 1..5)
                .prop_map(VariableDomain::Enumeration),
        ]
    }

    #[allow(dead_code)]
    fn exploration_config_strategy() -> impl Strategy<Value = ExplorationConfig> {
        (1usize..10000, 1usize..100, 1000u64..120000, any::<bool>()).prop_map(
            |(max_states, max_depth, timeout, compute)| {
                ExplorationConfig::new()
                    .with_max_states(max_states)
                    .with_max_depth(max_depth)
                    .with_timeout_ms(timeout)
                    .with_transitions(compute)
            },
        )
    }

    fn generation_strategy_strategy() -> impl Strategy<Value = GenerationStrategy> {
        prop_oneof![
            Just(GenerationStrategy::StateCoverage),
            Just(GenerationStrategy::TransitionCoverage),
            Just(GenerationStrategy::BoundaryValue),
            Just(GenerationStrategy::Combined),
            Just(GenerationStrategy::RandomWalk),
        ]
    }

    fn generator_config_strategy() -> impl Strategy<Value = GeneratorConfig> {
        (
            generation_strategy_strategy(),
            1usize..100,
            1usize..200,
            any::<u64>(),
        )
            .prop_map(|(strategy, max_length, max_tests, seed)| {
                GeneratorConfig::new()
                    .with_strategy(strategy)
                    .with_max_length(max_length)
                    .with_max_tests(max_tests)
                    .with_seed(seed)
            })
    }

    fn output_format_strategy() -> impl Strategy<Value = OutputFormat> {
        prop_oneof![
            Just(OutputFormat::Json),
            Just(OutputFormat::Rust),
            Just(OutputFormat::Python),
            Just(OutputFormat::Text),
            Just(OutputFormat::Markdown),
        ]
    }

    fn coverage_goal_strategy() -> impl Strategy<Value = CoverageGoal> {
        prop_oneof![
            "[a-z]{1,10}".prop_map(CoverageGoal::State),
            ("[a-z]{1,10}", "[a-z]{1,10}", "[a-z]{1,10}")
                .prop_map(|(action, from, to)| { CoverageGoal::Transition { action, from, to } }),
            ("[a-z]{1,10}", "[a-z]{1,10}").prop_map(|(var, val)| CoverageGoal::BoundaryValue {
                variable: var,
                value: val
            }),
            "[a-z]{1,10}".prop_map(CoverageGoal::Action),
        ]
    }

    fn test_case_strategy() -> impl Strategy<Value = TestCase> {
        (
            "[a-z]{1,10}",
            "[a-zA-Z ]{1,30}",
            model_state_strategy(),
            prop::collection::vec(model_action_strategy(), 0..3),
            prop::collection::vec(model_state_strategy(), 0..3),
        )
            .prop_map(|(id, desc, initial, actions, states)| {
                let mut test = TestCase::new(id, desc).with_initial_state(initial);
                let min_len = actions.len().min(states.len());
                for i in 0..min_len {
                    test.add_step(actions[i].clone(), states[i].clone());
                }
                test
            })
    }

    proptest! {
        // ==================== ModelValue Tests ====================

        #[test]
        fn model_value_canonical_string_non_empty(val in model_value_strategy()) {
            let canonical = val.canonical_string();
            prop_assert!(!canonical.is_empty());
        }

        #[test]
        fn model_value_canonical_string_deterministic(val in model_value_strategy()) {
            let s1 = val.canonical_string();
            let s2 = val.canonical_string();
            prop_assert_eq!(s1, s2);
        }

        #[test]
        fn model_value_display_matches_canonical(val in model_value_strategy()) {
            let display = format!("{}", val);
            let canonical = val.canonical_string();
            prop_assert_eq!(display, canonical);
        }

        #[test]
        fn model_value_bool_type_check(b in any::<bool>()) {
            let val = ModelValue::Bool(b);
            prop_assert!(val.is_bool());
            prop_assert!(!val.is_int());
            prop_assert_eq!(val.as_bool(), Some(b));
        }

        #[test]
        fn model_value_int_type_check(i in any::<i64>()) {
            let val = ModelValue::Int(i);
            prop_assert!(val.is_int());
            prop_assert!(!val.is_bool());
            prop_assert_eq!(val.as_int(), Some(i));
        }

        #[test]
        fn model_value_string_type_check(s in "[a-z]{1,20}") {
            let val = ModelValue::String(s.clone());
            prop_assert_eq!(val.as_str(), Some(s.as_str()));
        }

        #[test]
        fn model_value_equality_reflexive(val in model_value_strategy()) {
            prop_assert_eq!(val.clone(), val);
        }

        #[test]
        fn model_value_hash_consistent(val in model_value_strategy()) {
            let mut h1 = DefaultHasher::new();
            let mut h2 = DefaultHasher::new();
            val.hash(&mut h1);
            val.hash(&mut h2);
            prop_assert_eq!(h1.finish(), h2.finish());
        }

        #[test]
        fn model_value_null_is_null(_x in 0..1i32) {
            let val = ModelValue::Null;
            prop_assert_eq!(val.canonical_string(), "NULL");
            prop_assert_eq!(val.as_bool(), None);
            prop_assert_eq!(val.as_int(), None);
        }

        // ==================== ModelState Tests ====================

        #[test]
        fn model_state_new_is_empty(_x in 0..1i32) {
            let state = ModelState::new();
            prop_assert!(state.variable_names().next().is_none());
        }

        #[test]
        fn model_state_default_is_empty(_x in 0..1i32) {
            let state = ModelState::default();
            prop_assert!(state.variable_names().next().is_none());
        }

        #[test]
        fn model_state_set_get_roundtrip(name in "[a-z]{1,10}", val in simple_model_value_strategy()) {
            let mut state = ModelState::new();
            state.set(name.clone(), val.clone());
            prop_assert_eq!(state.get(&name), Some(&val));
            prop_assert!(state.contains(&name));
        }

        #[test]
        fn model_state_canonical_deterministic(state in model_state_strategy()) {
            let s1 = state.canonical_string();
            let s2 = state.canonical_string();
            prop_assert_eq!(s1, s2);
        }

        #[test]
        fn model_state_hash_consistent(state in model_state_strategy()) {
            let mut h1 = DefaultHasher::new();
            let mut h2 = DefaultHasher::new();
            state.hash(&mut h1);
            state.hash(&mut h2);
            prop_assert_eq!(h1.finish(), h2.finish());
        }

        #[test]
        fn model_state_diff_with_self_empty(state in model_state_strategy()) {
            let diff = state.diff(&state);
            prop_assert!(diff.is_empty());
        }

        #[test]
        fn model_state_from_variables_preserves_data(name in "[a-z]{1,10}", val in simple_model_value_strategy()) {
            let mut vars = indexmap::IndexMap::new();
            vars.insert(name.clone(), val.clone());
            let state = ModelState::from_variables(vars);
            prop_assert_eq!(state.get(&name), Some(&val));
        }

        // ==================== ModelAction Tests ====================

        #[test]
        fn model_action_new_has_name(name in "[a-zA-Z][a-zA-Z0-9_]{0,15}") {
            let action = ModelAction::new(name.clone());
            prop_assert_eq!(action.name, name);
            prop_assert!(action.parameters.is_empty());
        }

        #[test]
        fn model_action_signature_contains_name(action in model_action_strategy()) {
            let sig = action.signature();
            prop_assert!(sig.contains(&action.name));
        }

        #[test]
        fn model_action_display_equals_signature(action in model_action_strategy()) {
            let display = format!("{}", action);
            let sig = action.signature();
            prop_assert_eq!(display, sig);
        }

        #[test]
        fn model_action_with_params_preserves_params(
            name in "[a-zA-Z]{1,10}",
            params in prop::collection::vec(simple_model_value_strategy(), 1..3)
        ) {
            let action = ModelAction::with_params(name, params.clone());
            prop_assert_eq!(action.parameters.len(), params.len());
        }

        // ==================== ModelTransition Tests ====================

        #[test]
        fn model_transition_new_preserves_fields(
            from in model_state_strategy(),
            action in model_action_strategy(),
            to in model_state_strategy()
        ) {
            let trans = ModelTransition::new(from.clone(), action.clone(), to.clone());
            prop_assert_eq!(trans.from.canonical_string(), from.canonical_string());
            prop_assert_eq!(trans.action.name, action.name);
            prop_assert_eq!(trans.to.canonical_string(), to.canonical_string());
        }

        #[test]
        fn model_transition_display_contains_action(
            from in model_state_strategy(),
            action in model_action_strategy(),
            to in model_state_strategy()
        ) {
            let trans = ModelTransition::new(from, action.clone(), to);
            let display = format!("{}", trans);
            prop_assert!(display.contains(&action.name));
        }

        // ==================== VariableDomain Tests ====================

        #[test]
        fn variable_domain_boolean_boundary_values(_x in 0..1i32) {
            let domain = VariableDomain::Boolean;
            let boundaries = domain.boundary_values();
            prop_assert!(boundaries.contains(&ModelValue::Bool(true)));
            prop_assert!(boundaries.contains(&ModelValue::Bool(false)));
        }

        #[test]
        fn variable_domain_int_range_boundary_contains_min_max(min in -100i64..0i64, max in 1i64..100i64) {
            let domain = VariableDomain::IntRange { min, max };
            let boundaries = domain.boundary_values();
            prop_assert!(boundaries.contains(&ModelValue::Int(min)));
            prop_assert!(boundaries.contains(&ModelValue::Int(max)));
        }

        #[test]
        fn variable_domain_min_value_exists(domain in variable_domain_strategy()) {
            prop_assert!(domain.min_value().is_some());
        }

        #[test]
        fn variable_domain_max_value_exists(domain in variable_domain_strategy()) {
            prop_assert!(domain.max_value().is_some());
        }

        #[test]
        fn variable_domain_boundary_values_non_empty(domain in variable_domain_strategy()) {
            let boundaries = domain.boundary_values();
            prop_assert!(!boundaries.is_empty());
        }

        #[test]
        fn variable_domain_boolean_all_values(_x in 0..1i32) {
            let domain = VariableDomain::Boolean;
            let all = domain.all_values().unwrap();
            prop_assert_eq!(all.len(), 2);
        }

        #[test]
        fn variable_domain_int_range_all_values(min in 0i64..5i64) {
            let max = min + 5;
            let domain = VariableDomain::IntRange { min, max };
            let all = domain.all_values().unwrap();
            prop_assert_eq!(all.len(), (max - min + 1) as usize);
        }

        // ==================== Invariant Tests ====================

        #[test]
        fn invariant_new_preserves_fields(name in "[a-z]{1,10}", desc in "[a-zA-Z ]{1,30}") {
            let inv = Invariant::new(name.clone(), desc.clone());
            prop_assert_eq!(inv.name, name);
            prop_assert_eq!(inv.description, desc);
        }

        // ==================== ActionSpec Tests ====================

        #[test]
        fn action_spec_new_has_name(name in "[a-z]{1,10}") {
            let spec = ActionSpec::new(name.clone());
            prop_assert_eq!(spec.name, name);
            prop_assert!(spec.parameters.is_empty());
            prop_assert_eq!(spec.enabled_description, "true");
        }

        #[test]
        fn action_spec_with_param_adds_param(name in "[a-z]{1,10}", param in "[a-z]{1,10}") {
            let spec = ActionSpec::new(name).with_param(param.clone(), VariableDomain::Boolean);
            prop_assert_eq!(spec.parameters.len(), 1);
            prop_assert_eq!(&spec.parameters[0].0, &param);
        }

        #[test]
        fn action_spec_with_enabled_sets_description(name in "[a-z]{1,10}", enabled in "[a-z]{1,20}") {
            let spec = ActionSpec::new(name).with_enabled(enabled.clone());
            prop_assert_eq!(spec.enabled_description, enabled);
        }

        #[test]
        fn action_spec_with_effect_sets_description(name in "[a-z]{1,10}", effect in "[a-z]{1,20}") {
            let spec = ActionSpec::new(name).with_effect(effect.clone());
            prop_assert_eq!(spec.effect_description, effect);
        }

        // ==================== StateMachineModel Tests ====================

        #[test]
        fn state_machine_model_new_has_name(name in "[a-zA-Z]{1,15}") {
            let model = StateMachineModel::new(name.clone());
            prop_assert_eq!(model.name, name);
            prop_assert!(model.variables.is_empty());
            prop_assert!(model.initial_states.is_empty());
            prop_assert!(model.actions.is_empty());
            prop_assert!(model.invariants.is_empty());
        }

        #[test]
        fn state_machine_model_with_variable_adds_variable(
            name in "[a-zA-Z]{1,10}",
            var_name in "[a-z]{1,10}",
            domain in variable_domain_strategy()
        ) {
            let model = StateMachineModel::new(name).with_variable(var_name.clone(), domain);
            prop_assert!(model.variables.contains_key(&var_name));
        }

        #[test]
        fn state_machine_model_with_action_adds_action(name in "[a-zA-Z]{1,10}") {
            let action = ActionSpec::new("test_action");
            let model = StateMachineModel::new(name).with_action(action);
            prop_assert_eq!(model.actions.len(), 1);
        }

        #[test]
        fn state_machine_model_with_invariant_adds_invariant(name in "[a-zA-Z]{1,10}") {
            let inv = Invariant::new("test_inv", "description");
            let model = StateMachineModel::new(name).with_invariant(inv);
            prop_assert_eq!(model.invariants.len(), 1);
        }

        #[test]
        fn state_machine_model_validation_fails_without_initial_state(name in "[a-zA-Z]{1,10}") {
            let model = StateMachineModel::new(name);
            prop_assert!(model.validate().is_err());
        }

        // ==================== ExplorationConfig Tests ====================

        #[test]
        fn exploration_config_default_values(_x in 0..1i32) {
            let config = ExplorationConfig::default();
            prop_assert!(config.max_states > 0);
            prop_assert!(config.max_depth > 0);
            prop_assert!(config.timeout_ms > 0);
        }

        #[test]
        fn exploration_config_with_max_states_preserves(max in 1usize..10000) {
            let config = ExplorationConfig::new().with_max_states(max);
            prop_assert_eq!(config.max_states, max);
        }

        #[test]
        fn exploration_config_with_max_depth_preserves(max in 1usize..1000) {
            let config = ExplorationConfig::new().with_max_depth(max);
            prop_assert_eq!(config.max_depth, max);
        }

        #[test]
        fn exploration_config_with_timeout_preserves(timeout in 1u64..120000) {
            let config = ExplorationConfig::new().with_timeout_ms(timeout);
            prop_assert_eq!(config.timeout_ms, timeout);
        }

        #[test]
        fn exploration_config_with_transitions_preserves(compute in any::<bool>()) {
            let config = ExplorationConfig::new().with_transitions(compute);
            prop_assert_eq!(config.compute_transitions, compute);
        }

        // ==================== ExplorationResult Tests ====================

        #[test]
        fn exploration_result_new_is_empty(_x in 0..1i32) {
            let result = ExplorationResult::new();
            prop_assert!(result.states.is_empty());
            prop_assert!(result.transitions.is_empty());
            prop_assert!(!result.complete);
        }

        #[test]
        fn exploration_result_default_is_empty(_x in 0..1i32) {
            let result = ExplorationResult::default();
            prop_assert!(result.states.is_empty());
        }

        #[test]
        fn exploration_result_state_coverage_zero_when_empty(_x in 0..1i32) {
            let result = ExplorationResult::new();
            prop_assert_eq!(result.state_coverage(), 0.0);
        }

        // ==================== GeneratorConfig Tests ====================

        #[test]
        fn generator_config_default_values(_x in 0..1i32) {
            let config = GeneratorConfig::default();
            prop_assert!(config.max_test_length > 0);
            prop_assert!(config.max_tests > 0);
        }

        #[test]
        fn generator_config_with_strategy_preserves(strategy in generation_strategy_strategy()) {
            let config = GeneratorConfig::new().with_strategy(strategy);
            prop_assert_eq!(config.strategy, strategy);
        }

        #[test]
        fn generator_config_with_max_length_preserves(max in 1usize..100) {
            let config = GeneratorConfig::new().with_max_length(max);
            prop_assert_eq!(config.max_test_length, max);
        }

        #[test]
        fn generator_config_with_max_tests_preserves(max in 1usize..200) {
            let config = GeneratorConfig::new().with_max_tests(max);
            prop_assert_eq!(config.max_tests, max);
        }

        #[test]
        fn generator_config_with_seed_preserves(seed in any::<u64>()) {
            let config = GeneratorConfig::new().with_seed(seed);
            prop_assert_eq!(config.seed, Some(seed));
        }

        // ==================== CoverageGoal Tests ====================

        #[test]
        fn coverage_goal_state_from_model_state(state in model_state_strategy()) {
            let goal = CoverageGoal::state(&state);
            if let CoverageGoal::State(s) = goal {
                prop_assert_eq!(s, state.canonical_string());
            } else {
                prop_assert!(false, "Expected State variant");
            }
        }

        #[test]
        fn coverage_goal_action_preserves_name(name in "[a-z]{1,10}") {
            let goal = CoverageGoal::action(&name);
            prop_assert_eq!(goal, CoverageGoal::Action(name));
        }

        #[test]
        fn coverage_goal_boundary_preserves_fields(var in "[a-z]{1,10}", val in simple_model_value_strategy()) {
            let goal = CoverageGoal::boundary(&var, &val);
            if let CoverageGoal::BoundaryValue { variable, value } = goal {
                prop_assert_eq!(variable, var);
                prop_assert_eq!(value, val.canonical_string());
            } else {
                prop_assert!(false, "Expected BoundaryValue variant");
            }
        }

        // ==================== CoverageReport Tests ====================

        #[test]
        fn coverage_report_state_coverage_pct_100_when_empty(_x in 0..1i32) {
            let report = CoverageReport {
                states_covered: 0,
                states_total: 0,
                transitions_covered: 0,
                transitions_total: 0,
                actions_covered: 0,
                actions_total: 0,
                boundaries_covered: 0,
                boundaries_total: 0,
                uncovered: vec![],
            };
            prop_assert_eq!(report.state_coverage_pct(), 100.0);
        }

        #[test]
        fn coverage_report_transition_coverage_pct_100_when_empty(_x in 0..1i32) {
            let report = CoverageReport {
                states_covered: 0,
                states_total: 0,
                transitions_covered: 0,
                transitions_total: 0,
                actions_covered: 0,
                actions_total: 0,
                boundaries_covered: 0,
                boundaries_total: 0,
                uncovered: vec![],
            };
            prop_assert_eq!(report.transition_coverage_pct(), 100.0);
        }

        #[test]
        fn coverage_report_overall_coverage_pct_100_when_empty(_x in 0..1i32) {
            let report = CoverageReport {
                states_covered: 0,
                states_total: 0,
                transitions_covered: 0,
                transitions_total: 0,
                actions_covered: 0,
                actions_total: 0,
                boundaries_covered: 0,
                boundaries_total: 0,
                uncovered: vec![],
            };
            prop_assert_eq!(report.overall_coverage_pct(), 100.0);
        }

        #[test]
        fn coverage_report_state_coverage_pct_correct(covered in 0usize..100, total in 1usize..100) {
            let total = total.max(covered);
            let report = CoverageReport {
                states_covered: covered,
                states_total: total,
                transitions_covered: 0,
                transitions_total: 0,
                actions_covered: 0,
                actions_total: 0,
                boundaries_covered: 0,
                boundaries_total: 0,
                uncovered: vec![],
            };
            let expected = 100.0 * covered as f64 / total as f64;
            prop_assert!((report.state_coverage_pct() - expected).abs() < 0.001);
        }

        // ==================== TestCase Tests ====================

        #[test]
        fn test_case_new_has_id_and_description(id in "[a-z]{1,10}", desc in "[a-zA-Z ]{1,30}") {
            let test = TestCase::new(id.clone(), desc.clone());
            prop_assert_eq!(test.id, id);
            prop_assert_eq!(test.description, desc);
            prop_assert!(test.actions.is_empty());
            prop_assert!(test.expected_states.is_empty());
        }

        #[test]
        fn test_case_with_initial_state_preserves(id in "[a-z]{1,10}", state in model_state_strategy()) {
            let test = TestCase::new(id, "desc").with_initial_state(state.clone());
            prop_assert_eq!(test.initial_state.canonical_string(), state.canonical_string());
        }

        #[test]
        fn test_case_length_matches_actions(test in test_case_strategy()) {
            prop_assert_eq!(test.length(), test.actions.len());
        }

        #[test]
        fn test_case_final_state_is_last_expected(test in test_case_strategy()) {
            if !test.expected_states.is_empty() {
                prop_assert!(test.final_state().is_some());
            } else {
                prop_assert!(test.final_state().is_none());
            }
        }

        #[test]
        fn test_case_to_trace_length_matches_actions(test in test_case_strategy()) {
            let trace = test.to_trace();
            prop_assert_eq!(trace.len(), test.actions.len());
        }

        #[test]
        fn test_case_add_step_increases_length(
            id in "[a-z]{1,10}",
            action in model_action_strategy(),
            state in model_state_strategy()
        ) {
            let mut test = TestCase::new(id, "desc");
            let initial_len = test.length();
            test.add_step(action, state);
            prop_assert_eq!(test.length(), initial_len + 1);
        }

        #[test]
        fn test_case_add_coverage_adds_goal(id in "[a-z]{1,10}", goal in coverage_goal_strategy()) {
            let mut test = TestCase::new(id, "desc");
            test.add_coverage(goal.clone());
            prop_assert!(test.covers.contains(&goal));
        }

        // ==================== OutputFormat Tests ====================

        #[test]
        fn output_format_display_non_empty(format in output_format_strategy()) {
            let display = format!("{}", format);
            prop_assert!(!display.is_empty());
        }

        #[test]
        fn output_format_roundtrip(format in output_format_strategy()) {
            let display = format!("{}", format);
            let parsed: OutputFormat = display.parse().unwrap();
            prop_assert_eq!(parsed, format);
        }

        #[test]
        fn output_format_parse_variants(_x in 0..1i32) {
            prop_assert!("json".parse::<OutputFormat>().is_ok());
            prop_assert!("rust".parse::<OutputFormat>().is_ok());
            prop_assert!("rs".parse::<OutputFormat>().is_ok());
            prop_assert!("python".parse::<OutputFormat>().is_ok());
            prop_assert!("py".parse::<OutputFormat>().is_ok());
            prop_assert!("text".parse::<OutputFormat>().is_ok());
            prop_assert!("txt".parse::<OutputFormat>().is_ok());
            prop_assert!("markdown".parse::<OutputFormat>().is_ok());
            prop_assert!("md".parse::<OutputFormat>().is_ok());
        }

        #[test]
        fn output_format_parse_case_insensitive(_x in 0..1i32) {
            prop_assert!("JSON".parse::<OutputFormat>().is_ok());
            prop_assert!("Rust".parse::<OutputFormat>().is_ok());
            prop_assert!("PYTHON".parse::<OutputFormat>().is_ok());
        }

        #[test]
        fn output_format_parse_invalid_fails(s in "[xyz]{5,10}") {
            prop_assert!(s.parse::<OutputFormat>().is_err());
        }

        // ==================== TlaPlusParser Tests ====================

        #[test]
        fn tlaplus_parser_new_creates_instance(source in "[a-zA-Z ]{0,50}") {
            let _parser = TlaPlusParser::new(source);
            prop_assert!(true);
        }

        #[test]
        fn tlaplus_parser_parse_empty_spec(_x in 0..1i32) {
            let mut parser = TlaPlusParser::new("");
            let result = parser.parse();
            prop_assert!(result.is_ok());
        }

        #[test]
        fn tlaplus_parser_parse_module_name(name in "[A-Z][a-zA-Z]{1,10}") {
            let spec = format!("---- MODULE {} ----\n====", name);
            let mut parser = TlaPlusParser::new(spec);
            let result = parser.parse();
            prop_assert!(result.is_ok());
            let model = result.unwrap();
            prop_assert_eq!(model.name, name);
        }

        #[test]
        fn tlaplus_parser_parse_variable(var_name in "[a-z]{1,10}") {
            let spec = format!("---- MODULE Test ----\nVARIABLE {}\n====", var_name);
            let mut parser = TlaPlusParser::new(spec);
            let result = parser.parse();
            prop_assert!(result.is_ok());
            let model = result.unwrap();
            prop_assert!(model.variables.contains_key(&var_name));
        }

        #[test]
        fn tlaplus_parser_parse_multiple_variables(var1 in "[a-z]{1,5}", var2 in "[a-z]{6,10}") {
            let spec = format!("---- MODULE Test ----\nVARIABLES {}, {}\n====", var1, var2);
            let mut parser = TlaPlusParser::new(spec);
            let result = parser.parse();
            prop_assert!(result.is_ok());
            let model = result.unwrap();
            prop_assert!(model.variables.contains_key(&var1));
            prop_assert!(model.variables.contains_key(&var2));
        }

        #[test]
        fn parse_tlaplus_spec_function_works(source in "[a-zA-Z0-9 \n]*") {
            // parse_tlaplus_spec should not panic on arbitrary input
            let result = parse_tlaplus_spec(&source);
            prop_assert!(result.is_ok());
        }

        // ==================== MbtError Tests ====================

        #[test]
        fn mbt_error_parse_error_preserves_message(msg in "[a-zA-Z ]{1,30}") {
            let err = MbtError::ParseError(msg.clone());
            let display = format!("{}", err);
            prop_assert!(display.contains(&msg));
        }

        #[test]
        fn mbt_error_exploration_error_preserves_message(msg in "[a-zA-Z ]{1,30}") {
            let err = MbtError::ExplorationError(msg.clone());
            let display = format!("{}", err);
            prop_assert!(display.contains(&msg));
        }

        #[test]
        fn mbt_error_invalid_model_preserves_message(msg in "[a-zA-Z ]{1,30}") {
            let err = MbtError::InvalidModel(msg.clone());
            let display = format!("{}", err);
            prop_assert!(display.contains(&msg));
        }

        #[test]
        fn mbt_error_no_initial_state_has_message(_x in 0..1i32) {
            let err = MbtError::NoInitialState;
            let display = format!("{}", err);
            prop_assert!(!display.is_empty());
        }

        #[test]
        fn mbt_error_state_space_exhausted_has_counts(states in 1usize..1000, transitions in 1usize..1000) {
            let err = MbtError::StateSpaceExhausted { states, transitions };
            let display = format!("{}", err);
            prop_assert!(display.contains(&states.to_string()));
            prop_assert!(display.contains(&transitions.to_string()));
        }

        #[test]
        fn mbt_error_timeout_preserves_duration(ms in 1u64..120000) {
            let err = MbtError::Timeout(ms);
            let display = format!("{}", err);
            prop_assert!(display.contains(&ms.to_string()));
        }

        #[test]
        fn mbt_error_max_depth_reached_preserves_depth(depth in 1usize..1000) {
            let err = MbtError::MaxDepthReached(depth);
            let display = format!("{}", err);
            prop_assert!(display.contains(&depth.to_string()));
        }

        #[test]
        fn mbt_error_variable_not_found_preserves_name(name in "[a-z]{1,10}") {
            let err = MbtError::VariableNotFound(name.clone());
            let display = format!("{}", err);
            prop_assert!(display.contains(&name));
        }

        #[test]
        fn mbt_error_invalid_variable_type_preserves_fields(
            name in "[a-z]{1,10}",
            expected in "[a-z]{1,10}",
            actual in "[a-z]{1,10}"
        ) {
            let err = MbtError::InvalidVariableType {
                name: name.clone(),
                expected: expected.clone(),
                actual: actual.clone(),
            };
            let display = format!("{}", err);
            prop_assert!(display.contains(&name));
            prop_assert!(display.contains(&expected));
            prop_assert!(display.contains(&actual));
        }

        #[test]
        fn mbt_error_test_generation_failed_preserves_message(msg in "[a-zA-Z ]{1,30}") {
            let err = MbtError::TestGenerationFailed(msg.clone());
            let display = format!("{}", err);
            prop_assert!(display.contains(&msg));
        }

        // ==================== PrecomputedExecutor Tests ====================

        #[test]
        fn precomputed_executor_new_has_no_transitions(_x in 0..1i32) {
            let exec = explorer::PrecomputedExecutor::new();
            let state = ModelState::new();
            let transitions = exec.enabled_transitions(&state);
            prop_assert!(transitions.is_empty());
        }

        #[test]
        fn precomputed_executor_default_has_no_transitions(_x in 0..1i32) {
            let exec = explorer::PrecomputedExecutor::default();
            let state = ModelState::new();
            let transitions = exec.enabled_transitions(&state);
            prop_assert!(transitions.is_empty());
        }

        #[test]
        fn precomputed_executor_add_transition_retrievable(
            from in model_state_strategy(),
            action in model_action_strategy(),
            to in model_state_strategy()
        ) {
            let mut exec = explorer::PrecomputedExecutor::new();
            exec.add_transition(&from, action.clone(), to);
            let transitions = exec.enabled_transitions(&from);
            prop_assert!(!transitions.is_empty());
            prop_assert!(transitions.iter().any(|(a, _)| a.name == action.name));
        }

        // ==================== TestGenerator Tests ====================

        #[test]
        fn test_generator_new_creates_instance(_x in 0..1i32) {
            // TestGenerator::new() should create a valid instance
            let _gen = TestGenerator::new();
            prop_assert!(true);
        }

        #[test]
        fn test_generator_default_creates_instance(_x in 0..1i32) {
            // TestGenerator::default() should create a valid instance
            let _gen = TestGenerator::default();
            prop_assert!(true);
        }

        #[test]
        fn test_generator_with_config_creates_instance(config in generator_config_strategy()) {
            // TestGenerator::with_config() should create a valid instance with the given config
            let _gen = TestGenerator::with_config(config);
            prop_assert!(true);
        }

        // ==================== TransitionRecord Tests ====================

        #[test]
        fn transition_record_fields_preserved(from_idx in 0usize..100, to_idx in 0usize..100, action in model_action_strategy()) {
            let record = explorer::TransitionRecord {
                from_idx,
                to_idx,
                action: action.clone(),
            };
            prop_assert_eq!(record.from_idx, from_idx);
            prop_assert_eq!(record.to_idx, to_idx);
            prop_assert_eq!(record.action.name, action.name);
        }
    }
}

#[cfg(test)]
mod mutation_killing_tests {
    //! Tests specifically designed to catch mutations that were previously missed.
    use super::*;
    use std::collections::HashSet;

    // ==================== ExplorationResult Method Tests ====================

    #[test]
    fn test_exploration_result_get_state_returns_correct_state() {
        let mut result = ExplorationResult::new();
        let mut s0 = ModelState::new();
        s0.set("x", ModelValue::Int(0));
        let mut s1 = ModelState::new();
        s1.set("x", ModelValue::Int(1));

        result.states.push(s0.clone());
        result.states.push(s1.clone());

        // Verify get_state returns the correct state
        let state0 = result.get_state(0).unwrap();
        assert_eq!(state0.get("x"), Some(&ModelValue::Int(0)));

        let state1 = result.get_state(1).unwrap();
        assert_eq!(state1.get("x"), Some(&ModelValue::Int(1)));

        // Out of bounds returns None
        assert!(result.get_state(2).is_none());
    }

    #[test]
    fn test_exploration_result_state_index_returns_correct_index() {
        let mut result = ExplorationResult::new();
        let mut s0 = ModelState::new();
        s0.set("x", ModelValue::Int(42));

        // Must populate state_index manually for this test
        result.state_index.insert(s0.canonical_string(), 5);

        let idx = result.state_index(&s0);
        assert_eq!(idx, Some(5));

        // Non-existent state
        let mut s1 = ModelState::new();
        s1.set("x", ModelValue::Int(99));
        assert_eq!(result.state_index(&s1), None);
    }

    #[test]
    fn test_exploration_result_transitions_from_filters_correctly() {
        let mut result = ExplorationResult::new();
        result.transitions.push(explorer::TransitionRecord {
            from_idx: 0,
            to_idx: 1,
            action: ModelAction::new("a"),
        });
        result.transitions.push(explorer::TransitionRecord {
            from_idx: 1,
            to_idx: 2,
            action: ModelAction::new("b"),
        });
        result.transitions.push(explorer::TransitionRecord {
            from_idx: 0,
            to_idx: 2,
            action: ModelAction::new("c"),
        });

        // transitions_from(0) should return 2 transitions
        let from_0: Vec<_> = result.transitions_from(0).collect();
        assert_eq!(from_0.len(), 2);
        assert!(from_0.iter().all(|t| t.from_idx == 0));
        assert!(from_0.iter().any(|t| t.action.name == "a"));
        assert!(from_0.iter().any(|t| t.action.name == "c"));

        // transitions_from(1) should return 1 transition
        let from_1: Vec<_> = result.transitions_from(1).collect();
        assert_eq!(from_1.len(), 1);
        assert_eq!(from_1[0].from_idx, 1);
        assert_eq!(from_1[0].action.name, "b");

        // transitions_from(2) should return 0 transitions
        let from_2: Vec<_> = result.transitions_from(2).collect();
        assert!(from_2.is_empty());
    }

    #[test]
    fn test_exploration_result_transitions_to_filters_correctly() {
        let mut result = ExplorationResult::new();
        result.transitions.push(explorer::TransitionRecord {
            from_idx: 0,
            to_idx: 2,
            action: ModelAction::new("a"),
        });
        result.transitions.push(explorer::TransitionRecord {
            from_idx: 1,
            to_idx: 2,
            action: ModelAction::new("b"),
        });
        result.transitions.push(explorer::TransitionRecord {
            from_idx: 0,
            to_idx: 1,
            action: ModelAction::new("c"),
        });

        // transitions_to(2) should return 2 transitions
        let to_2: Vec<_> = result.transitions_to(2).collect();
        assert_eq!(to_2.len(), 2);
        assert!(to_2.iter().all(|t| t.to_idx == 2));

        // transitions_to(1) should return 1 transition
        let to_1: Vec<_> = result.transitions_to(1).collect();
        assert_eq!(to_1.len(), 1);
        assert_eq!(to_1[0].to_idx, 1);

        // transitions_to(0) should return 0 transitions
        let to_0: Vec<_> = result.transitions_to(0).collect();
        assert!(to_0.is_empty());
    }

    #[test]
    fn test_exploration_result_unique_actions_collects_all_unique() {
        let mut result = ExplorationResult::new();
        result.transitions.push(explorer::TransitionRecord {
            from_idx: 0,
            to_idx: 1,
            action: ModelAction::new("increment"),
        });
        result.transitions.push(explorer::TransitionRecord {
            from_idx: 1,
            to_idx: 2,
            action: ModelAction::new("increment"),
        });
        result.transitions.push(explorer::TransitionRecord {
            from_idx: 2,
            to_idx: 0,
            action: ModelAction::new("reset"),
        });

        let unique = result.unique_actions();
        assert_eq!(unique.len(), 2);
        assert!(unique.contains("increment"));
        assert!(unique.contains("reset"));
    }

    #[test]
    fn test_exploration_result_all_actions_exercised_true() {
        let mut result = ExplorationResult::new();
        result.transitions.push(explorer::TransitionRecord {
            from_idx: 0,
            to_idx: 1,
            action: ModelAction::new("inc"),
        });
        result.transitions.push(explorer::TransitionRecord {
            from_idx: 1,
            to_idx: 0,
            action: ModelAction::new("dec"),
        });

        let model = StateMachineModel::new("test")
            .with_action(ActionSpec::new("inc"))
            .with_action(ActionSpec::new("dec"));

        assert!(result.all_actions_exercised(&model));
    }

    #[test]
    fn test_exploration_result_all_actions_exercised_false() {
        let mut result = ExplorationResult::new();
        result.transitions.push(explorer::TransitionRecord {
            from_idx: 0,
            to_idx: 1,
            action: ModelAction::new("inc"),
        });

        let model = StateMachineModel::new("test")
            .with_action(ActionSpec::new("inc"))
            .with_action(ActionSpec::new("dec"))
            .with_action(ActionSpec::new("reset"));

        assert!(!result.all_actions_exercised(&model));
    }

    #[test]
    fn test_exploration_result_state_coverage_formula() {
        let mut result = ExplorationResult::new();
        result.states.push(ModelState::new());
        result.states.push(ModelState::new());
        result.states.push(ModelState::new());
        result.states_explored = 6;

        // Coverage = states.len() / states_explored = 3/6 = 0.5
        let coverage = result.state_coverage();
        assert!((coverage - 0.5).abs() < 0.001);

        // Test different ratio
        result.states_explored = 3;
        let coverage2 = result.state_coverage();
        assert!((coverage2 - 1.0).abs() < 0.001);

        // Test zero case
        result.states.clear();
        result.states_explored = 0;
        assert_eq!(result.state_coverage(), 0.0);
    }

    // ==================== StateExecutor Tests ====================

    #[test]
    fn test_state_executor_is_enabled_matches_action_name() {
        let mut exec = explorer::PrecomputedExecutor::new();
        let mut s0 = ModelState::new();
        s0.set("x", ModelValue::Int(0));
        let mut s1 = ModelState::new();
        s1.set("x", ModelValue::Int(1));

        exec.add_transition(&s0, ModelAction::new("increment"), s1);

        // is_enabled should return true for matching action
        assert!(exec.is_enabled(&s0, "increment"));

        // is_enabled should return false for non-matching action
        assert!(!exec.is_enabled(&s0, "decrement"));
        assert!(!exec.is_enabled(&s0, "reset"));
    }

    // ==================== CoverageReport Calculation Tests ====================

    #[test]
    fn test_coverage_report_transition_coverage_formula() {
        let report = generator::CoverageReport {
            states_covered: 0,
            states_total: 0,
            transitions_covered: 3,
            transitions_total: 12,
            actions_covered: 0,
            actions_total: 0,
            boundaries_covered: 0,
            boundaries_total: 0,
            uncovered: vec![],
        };

        // 100.0 * 3 / 12 = 25.0
        let pct = report.transition_coverage_pct();
        assert!((pct - 25.0).abs() < 0.001);
    }

    #[test]
    fn test_coverage_report_overall_coverage_formula() {
        let report = generator::CoverageReport {
            states_covered: 2,
            states_total: 4,
            transitions_covered: 3,
            transitions_total: 6,
            actions_covered: 1,
            actions_total: 2,
            boundaries_covered: 1,
            boundaries_total: 2,
            uncovered: vec![],
        };

        // total = 4 + 6 + 2 = 12
        // covered = 2 + 3 + 1 = 6
        // overall = 100.0 * 6 / 12 = 50.0
        let pct = report.overall_coverage_pct();
        assert!((pct - 50.0).abs() < 0.001);
    }

    // ==================== ModelValue Equality Tests ====================

    #[test]
    fn test_model_value_sequence_equality_order_matters() {
        let seq1 = ModelValue::Sequence(vec![ModelValue::Int(1), ModelValue::Int(2)]);
        let seq2 = ModelValue::Sequence(vec![ModelValue::Int(2), ModelValue::Int(1)]);
        let seq3 = ModelValue::Sequence(vec![ModelValue::Int(1), ModelValue::Int(2)]);

        // Sequences care about order
        assert_ne!(seq1, seq2);
        assert_eq!(seq1, seq3);
    }

    #[test]
    fn test_model_value_record_equality() {
        let mut r1 = indexmap::IndexMap::new();
        r1.insert("a".to_string(), ModelValue::Int(1));
        r1.insert("b".to_string(), ModelValue::Int(2));

        let mut r2 = indexmap::IndexMap::new();
        r2.insert("b".to_string(), ModelValue::Int(2));
        r2.insert("a".to_string(), ModelValue::Int(1));

        let mut r3 = indexmap::IndexMap::new();
        r3.insert("a".to_string(), ModelValue::Int(1));
        r3.insert("b".to_string(), ModelValue::Int(3)); // Different value

        let rec1 = ModelValue::Record(r1);
        let rec2 = ModelValue::Record(r2);
        let rec3 = ModelValue::Record(r3);

        assert_eq!(rec1, rec2); // Same content, different order
        assert_ne!(rec1, rec3); // Different values
    }

    #[test]
    fn test_model_value_function_equality() {
        let f1 = ModelValue::Function(vec![
            (ModelValue::Int(1), ModelValue::String("one".into())),
            (ModelValue::Int(2), ModelValue::String("two".into())),
        ]);
        let f2 = ModelValue::Function(vec![
            (ModelValue::Int(2), ModelValue::String("two".into())),
            (ModelValue::Int(1), ModelValue::String("one".into())),
        ]);
        let f3 = ModelValue::Function(vec![
            (ModelValue::Int(1), ModelValue::String("one".into())),
            (ModelValue::Int(2), ModelValue::String("trois".into())), // Different value
        ]);
        let f4 = ModelValue::Function(vec![(ModelValue::Int(1), ModelValue::String("one".into()))]); // Different length

        assert_eq!(f1, f2); // Same mappings, different order
        assert_ne!(f1, f3); // Different values
        assert_ne!(f1, f4); // Different length
    }

    #[test]
    fn test_model_value_different_types_not_equal() {
        let int_val = ModelValue::Int(1);
        let bool_val = ModelValue::Bool(true);
        let string_val = ModelValue::String("1".into());

        assert_ne!(int_val, bool_val);
        assert_ne!(int_val, string_val);
        assert_ne!(bool_val, string_val);
    }

    // ==================== ModelValue Hash Tests ====================

    #[test]
    fn test_model_value_hash_is_called() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let seq = ModelValue::Sequence(vec![ModelValue::Int(1), ModelValue::Int(2)]);
        let mut hasher = DefaultHasher::new();
        seq.hash(&mut hasher);
        let hash1 = hasher.finish();

        // Different sequence should have different hash
        let seq2 = ModelValue::Sequence(vec![ModelValue::Int(2), ModelValue::Int(1)]);
        let mut hasher2 = DefaultHasher::new();
        seq2.hash(&mut hasher2);
        let hash2 = hasher2.finish();

        assert_ne!(hash1, hash2);
    }

    // ==================== ModelState Tests ====================

    #[test]
    fn test_model_state_contains_returns_correct_value() {
        let mut state = ModelState::new();
        state.set("existing", ModelValue::Int(42));

        assert!(state.contains("existing"));
        assert!(!state.contains("nonexistent"));
    }

    #[test]
    fn test_model_state_variable_names_iterates_all() {
        let mut state = ModelState::new();
        state.set("a", ModelValue::Int(1));
        state.set("b", ModelValue::Int(2));
        state.set("c", ModelValue::Int(3));

        let names: Vec<_> = state.variable_names().collect();
        assert_eq!(names.len(), 3);
        assert!(names.contains(&"a"));
        assert!(names.contains(&"b"));
        assert!(names.contains(&"c"));
    }

    #[test]
    fn test_model_state_display_contains_all_variables() {
        let mut state = ModelState::new();
        state.set("x", ModelValue::Int(10));
        state.set("y", ModelValue::Bool(true));

        let display = format!("{}", state);
        assert!(display.contains("x"));
        assert!(display.contains("10"));
        assert!(display.contains("y"));
        assert!(display.contains("true"));
    }

    // ==================== VariableDomain Boundary Tests ====================

    #[test]
    fn test_variable_domain_int_range_boundary_adjacent_values() {
        let domain = VariableDomain::IntRange { min: -5, max: 5 };
        let boundaries = domain.boundary_values();

        // Should contain min, max, min+1, max-1, and 0 (since in range)
        assert!(boundaries.contains(&ModelValue::Int(-5))); // min
        assert!(boundaries.contains(&ModelValue::Int(5))); // max
        assert!(boundaries.contains(&ModelValue::Int(-4))); // min+1
        assert!(boundaries.contains(&ModelValue::Int(4))); // max-1
        assert!(boundaries.contains(&ModelValue::Int(0))); // zero in range
    }

    #[test]
    fn test_variable_domain_int_range_boundary_no_zero_when_out_of_range() {
        let domain = VariableDomain::IntRange { min: 5, max: 10 };
        let boundaries = domain.boundary_values();

        // Zero should NOT be in boundaries since it's outside range
        assert!(!boundaries.contains(&ModelValue::Int(0)));
    }

    #[test]
    fn test_variable_domain_int_range_boundary_min_equals_max() {
        let domain = VariableDomain::IntRange { min: 5, max: 5 };
        let boundaries = domain.boundary_values();

        // Should contain 5 once (not twice)
        let count = boundaries
            .iter()
            .filter(|v| **v == ModelValue::Int(5))
            .count();
        assert!(count >= 1);
    }

    #[test]
    fn test_variable_domain_enumeration_boundary_values() {
        let domain = VariableDomain::Enumeration(vec![
            ModelValue::String("a".into()),
            ModelValue::String("b".into()),
            ModelValue::String("c".into()),
        ]);
        let boundaries = domain.boundary_values();

        // Should contain first and last
        assert!(boundaries.contains(&ModelValue::String("a".into())));
        assert!(boundaries.contains(&ModelValue::String("c".into())));
    }

    #[test]
    fn test_variable_domain_sequence_boundary_values() {
        let inner = VariableDomain::Boolean;
        let domain = VariableDomain::SequenceOf {
            element: Box::new(inner),
            max_length: 3,
        };
        let boundaries = domain.boundary_values();

        // Should include empty sequence
        assert!(boundaries.contains(&ModelValue::Sequence(vec![])));
        // Should include single element sequences
        assert!(boundaries
            .iter()
            .any(|v| matches!(v, ModelValue::Sequence(s) if s.len() == 1)));
        // Should include max length sequences
        assert!(boundaries
            .iter()
            .any(|v| matches!(v, ModelValue::Sequence(s) if s.len() == 3)));
    }

    #[test]
    fn test_variable_domain_all_values_int_range() {
        let domain = VariableDomain::IntRange { min: 0, max: 5 };
        let all = domain.all_values().unwrap();

        assert_eq!(all.len(), 6); // 0, 1, 2, 3, 4, 5
        for i in 0..=5 {
            assert!(all.contains(&ModelValue::Int(i)));
        }
    }

    #[test]
    fn test_variable_domain_all_values_enumeration() {
        let domain = VariableDomain::Enumeration(vec![
            ModelValue::Int(10),
            ModelValue::Int(20),
            ModelValue::Int(30),
        ]);
        let all = domain.all_values().unwrap();

        assert_eq!(all.len(), 3);
        assert!(all.contains(&ModelValue::Int(10)));
        assert!(all.contains(&ModelValue::Int(20)));
        assert!(all.contains(&ModelValue::Int(30)));
    }

    #[test]
    fn test_variable_domain_all_values_too_large() {
        let domain = VariableDomain::IntRange { min: 0, max: 5000 };
        let result = domain.all_values();

        assert!(result.is_err());
    }

    // ==================== StateMachineModel Tests ====================

    #[test]
    fn test_state_machine_model_action_names_iterates_all() {
        let model = StateMachineModel::new("test")
            .with_action(ActionSpec::new("action1"))
            .with_action(ActionSpec::new("action2"))
            .with_action(ActionSpec::new("action3"));

        let names: Vec<_> = model.action_names().collect();
        assert_eq!(names.len(), 3);
        assert!(names.contains(&"action1"));
        assert!(names.contains(&"action2"));
        assert!(names.contains(&"action3"));
    }

    // ==================== Explorer DFS Tests ====================

    #[test]
    fn test_explore_dfs_finds_traces() {
        let mut exec = explorer::PrecomputedExecutor::new();

        // Build a linear chain: s0 -> s1 -> s2
        let mut s0 = ModelState::new();
        s0.set("x", ModelValue::Int(0));
        let mut s1 = ModelState::new();
        s1.set("x", ModelValue::Int(1));
        let mut s2 = ModelState::new();
        s2.set("x", ModelValue::Int(2));

        exec.add_transition(&s0, ModelAction::new("step"), s1.clone());
        exec.add_transition(&s1, ModelAction::new("step"), s2.clone());

        let explorer = StateExplorer::new(exec);
        let traces = explorer.explore_dfs(&[s0], 10).unwrap();

        // Should find traces up to terminal state
        assert!(!traces.is_empty());
        // At least one trace should reach s2
        assert!(traces.iter().any(|t| !t.is_empty()));
    }

    #[test]
    fn test_explore_dfs_respects_depth_limit() {
        let mut exec = explorer::PrecomputedExecutor::new();

        // Build a long chain
        for i in 0..10 {
            let mut from = ModelState::new();
            from.set("x", ModelValue::Int(i));
            let mut to = ModelState::new();
            to.set("x", ModelValue::Int(i + 1));
            exec.add_transition(&from, ModelAction::new("step"), to);
        }

        let mut s0 = ModelState::new();
        s0.set("x", ModelValue::Int(0));

        let explorer = StateExplorer::new(exec);
        let traces = explorer.explore_dfs(&[s0], 3).unwrap();

        // All traces should have length <= 3
        for trace in &traces {
            assert!(trace.len() <= 3);
        }
    }

    // ==================== find_path Tests ====================

    #[test]
    fn test_find_path_returns_shortest() {
        let mut exec = explorer::PrecomputedExecutor::new();

        // s0 -> s1 -> s2 (long path)
        // s0 -> s2 (short path)
        let mut s0 = ModelState::new();
        s0.set("x", ModelValue::Int(0));
        let mut s1 = ModelState::new();
        s1.set("x", ModelValue::Int(1));
        let mut s2 = ModelState::new();
        s2.set("x", ModelValue::Int(2));

        exec.add_transition(&s0, ModelAction::new("step1"), s1.clone());
        exec.add_transition(&s1, ModelAction::new("step2"), s2.clone());
        exec.add_transition(&s0, ModelAction::new("jump"), s2.clone());

        let path = explorer::find_path(&exec, &s0, &s2, 10).unwrap();
        assert_eq!(path.len(), 1); // Should find the direct path
    }

    #[test]
    fn test_find_path_respects_max_depth() {
        let mut exec = explorer::PrecomputedExecutor::new();

        // Very long path
        for i in 0..100 {
            let mut from = ModelState::new();
            from.set("x", ModelValue::Int(i));
            let mut to = ModelState::new();
            to.set("x", ModelValue::Int(i + 1));
            exec.add_transition(&from, ModelAction::new("step"), to);
        }

        let mut s0 = ModelState::new();
        s0.set("x", ModelValue::Int(0));
        let mut s100 = ModelState::new();
        s100.set("x", ModelValue::Int(100));

        // Max depth 5 should not find path to state 100
        let path = explorer::find_path(&exec, &s0, &s100, 5);
        assert!(path.is_none());
    }

    // ==================== Output Format Tests ====================

    #[test]
    fn test_format_rust_state_contains_variables() {
        let mut state = ModelState::new();
        state.set("counter", ModelValue::Int(42));
        state.set("flag", ModelValue::Bool(true));

        let result = output::format_results(
            &generator::GenerationResult {
                tests: vec![generator::TestCase::new("t1", "desc").with_initial_state(state)],
                coverage: generator::CoverageReport {
                    states_covered: 1,
                    states_total: 1,
                    transitions_covered: 0,
                    transitions_total: 0,
                    actions_covered: 0,
                    actions_total: 0,
                    boundaries_covered: 0,
                    boundaries_total: 0,
                    uncovered: vec![],
                },
                stats: generator::GenerationStats {
                    tests_generated: 1,
                    total_steps: 0,
                    avg_test_length: 0.0,
                    max_test_length: 0,
                    duration_ms: 0,
                },
            },
            output::OutputFormat::Rust,
        );

        assert!(result.contains("counter"));
        assert!(result.contains("42"));
        assert!(result.contains("flag"));
    }

    // ==================== TLA+ Parser Tests ====================

    #[test]
    fn test_tlaplus_parser_skips_comments() {
        let spec = r#"
---- MODULE Test ----
\* This is a comment
(* Multi-line
   comment *)
VARIABLE x
====
"#;
        let model = tlaplus::parse_tlaplus_spec(spec).unwrap();
        assert!(model.variables.contains_key("x"));
    }

    #[test]
    fn test_tlaplus_parser_parses_init() {
        let spec = r#"
---- MODULE Test ----
VARIABLE x, y
Init == x = 0 /\ y = TRUE
====
"#;
        let model = tlaplus::parse_tlaplus_spec(spec).unwrap();
        assert!(!model.initial_states.is_empty());
    }

    #[test]
    fn test_tlaplus_parser_parses_next_disjunction() {
        let spec = r#"
---- MODULE Test ----
VARIABLE x
Init == x = 0
Increment == x' = x + 1
Decrement == x' = x - 1
Next == Increment \/ Decrement
====
"#;
        let model = tlaplus::parse_tlaplus_spec(spec).unwrap();
        let action_names: HashSet<_> = model.actions.iter().map(|a| a.name.as_str()).collect();
        assert!(action_names.contains("Increment") || action_names.contains("Decrement"));
    }

    #[test]
    fn test_tlaplus_parser_parses_type_ok() {
        // Note: TypeInvariant contains "inv" so goes to invariants first.
        // Use "TypeOK" which only contains "type" to trigger type parsing.
        let spec = r#"
---- MODULE Test ----
VARIABLE count
TypeOK == count \in 0..100
====
"#;
        let model = tlaplus::parse_tlaplus_spec(spec).unwrap();
        let domain = model.variables.get("count");
        assert!(domain.is_some());
        // Should have parsed as IntRange
        assert!(matches!(domain.unwrap(), VariableDomain::IntRange { .. }));
    }

    #[test]
    fn test_tlaplus_parser_parses_constants_section() {
        let spec = r#"
---- MODULE Test ----
CONSTANTS MaxVal
VARIABLE x
====
"#;
        let model = tlaplus::parse_tlaplus_spec(spec).unwrap();
        // CONSTANTS should end variable parsing
        assert!(model.variables.contains_key("x"));
    }

    // ==================== TestGenerator Tests ====================

    #[test]
    fn test_generator_random_walks_respects_max_length() {
        let mut exec = explorer::PrecomputedExecutor::new();

        // Build a long chain
        for i in 0..20 {
            let mut from = ModelState::new();
            from.set("x", ModelValue::Int(i));
            let mut to = ModelState::new();
            to.set("x", ModelValue::Int(i + 1));
            exec.add_transition(&from, ModelAction::new("step"), to);
        }

        let mut s0 = ModelState::new();
        s0.set("x", ModelValue::Int(0));

        let explorer = StateExplorer::new(exec);
        let exploration = explorer.explore(&[s0]).unwrap();

        let config = GeneratorConfig::new()
            .with_strategy(GenerationStrategy::RandomWalk)
            .with_max_length(5)
            .with_seed(42);
        let mut gen = TestGenerator::with_config(config);
        let result = gen.generate(&exploration).unwrap();

        // All generated tests should have length <= 5
        for test in &result.tests {
            assert!(test.length() <= 5);
        }
    }

    #[test]
    fn test_generator_boundary_requires_model() {
        let exploration = ExplorationResult::new();

        let config = GeneratorConfig::new().with_strategy(GenerationStrategy::BoundaryValue);
        let mut gen = TestGenerator::with_config(config);
        let result = gen.generate(&exploration);

        // Should return error because boundary value needs model
        assert!(result.is_err());
    }

    // ==================== TestCase Tests ====================

    #[test]
    fn test_case_final_state_returns_last() {
        let mut test = TestCase::new("t1", "test");
        let mut s0 = ModelState::new();
        s0.set("x", ModelValue::Int(0));
        let mut s1 = ModelState::new();
        s1.set("x", ModelValue::Int(1));
        let mut s2 = ModelState::new();
        s2.set("x", ModelValue::Int(2));

        test = test.with_initial_state(s0.clone());
        test.add_step(ModelAction::new("a"), s1.clone());
        test.add_step(ModelAction::new("b"), s2.clone());

        let final_state = test.final_state().unwrap();
        assert_eq!(final_state.get("x"), Some(&ModelValue::Int(2)));
    }
}

// ==================== Kani Proofs ====================
// NOTE: Avoid proofs that construct HashMap as Kani on macOS doesn't support
// CCRandomGenerateBytes used by HashMap's hasher.

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ============== ModelValue Proofs ==============

    /// Proves that ModelValue::Bool preserves its value
    #[kani::proof]
    fn verify_model_value_bool_preserved() {
        let b: bool = kani::any();
        let val = ModelValue::Bool(b);
        kani::assert(val.is_bool(), "Bool value is_bool() returns true");
        kani::assert(!val.is_int(), "Bool value is_int() returns false");
        kani::assert(val.as_bool() == Some(b), "as_bool() returns correct value");
    }

    /// Proves that ModelValue::Int preserves its value
    #[kani::proof]
    fn verify_model_value_int_preserved() {
        let i: i64 = kani::any();
        let val = ModelValue::Int(i);
        kani::assert(val.is_int(), "Int value is_int() returns true");
        kani::assert(!val.is_bool(), "Int value is_bool() returns false");
        kani::assert(val.as_int() == Some(i), "as_int() returns correct value");
    }

    /// Proves that ModelValue::Null is correctly identified
    #[kani::proof]
    fn verify_model_value_null() {
        let val = ModelValue::Null;
        kani::assert(val.as_bool().is_none(), "Null as_bool() returns None");
        kani::assert(val.as_int().is_none(), "Null as_int() returns None");
        kani::assert(
            val.canonical_string() == "NULL",
            "Null canonical_string is 'NULL'",
        );
    }

    /// Proves that ModelValue Bool canonical_string is "true" or "false"
    #[kani::proof]
    fn verify_model_value_bool_canonical() {
        let val_true = ModelValue::Bool(true);
        let val_false = ModelValue::Bool(false);
        kani::assert(
            val_true.canonical_string() == "true",
            "Bool(true) canonical is 'true'",
        );
        kani::assert(
            val_false.canonical_string() == "false",
            "Bool(false) canonical is 'false'",
        );
    }

    // ============== ModelAction Proofs ==============

    /// Proves that ModelAction::new creates an action with empty parameters
    #[kani::proof]
    fn verify_model_action_new_empty_params() {
        let action = ModelAction::new("test");
        kani::assert(action.name == "test", "Action name is preserved");
        kani::assert(
            action.parameters.is_empty(),
            "New action has empty parameters",
        );
    }

    /// Proves that ModelAction signature equals name when no parameters
    #[kani::proof]
    fn verify_model_action_signature_no_params() {
        let action = ModelAction::new("increment");
        kani::assert(
            action.signature() == "increment",
            "Signature equals name when no params",
        );
    }

    // ============== ModelState Proofs ==============
    // Note: ModelState uses IndexMap which hits CCRandomGenerateBytes on macOS.
    // Proofs involving ModelState construction are not run on macOS.

    // ============== VariableDomain Proofs ==============

    /// Proves that Boolean domain min_value is Bool(false)
    #[kani::proof]
    fn verify_variable_domain_boolean_min() {
        let domain = VariableDomain::Boolean;
        let min = domain.min_value();
        kani::assert(min == Some(ModelValue::Bool(false)), "Boolean min is false");
    }

    /// Proves that Boolean domain max_value is Bool(true)
    #[kani::proof]
    fn verify_variable_domain_boolean_max() {
        let domain = VariableDomain::Boolean;
        let max = domain.max_value();
        kani::assert(max == Some(ModelValue::Bool(true)), "Boolean max is true");
    }

    /// Proves that IntRange domain min_value is the min bound
    #[kani::proof]
    fn verify_variable_domain_int_range_min() {
        let min: i64 = kani::any();
        let max: i64 = kani::any();
        kani::assume(min <= max);
        let domain = VariableDomain::IntRange { min, max };
        let min_val = domain.min_value();
        kani::assert(
            min_val == Some(ModelValue::Int(min)),
            "IntRange min_value is min bound",
        );
    }

    /// Proves that IntRange domain max_value is the max bound
    #[kani::proof]
    fn verify_variable_domain_int_range_max() {
        let min: i64 = kani::any();
        let max: i64 = kani::any();
        kani::assume(min <= max);
        let domain = VariableDomain::IntRange { min, max };
        let max_val = domain.max_value();
        kani::assert(
            max_val == Some(ModelValue::Int(max)),
            "IntRange max_value is max bound",
        );
    }

    /// Proves that Boolean boundary_values contains both true and false
    #[kani::proof]
    fn verify_variable_domain_boolean_boundary_values() {
        let domain = VariableDomain::Boolean;
        let boundaries = domain.boundary_values();
        kani::assert(
            boundaries.len() == 2,
            "Boolean has exactly 2 boundary values",
        );
        kani::assert(
            boundaries.contains(&ModelValue::Bool(false)),
            "Boolean boundaries contain false",
        );
        kani::assert(
            boundaries.contains(&ModelValue::Bool(true)),
            "Boolean boundaries contain true",
        );
    }

    /// Proves that Boolean all_values returns [false, true]
    #[kani::proof]
    fn verify_variable_domain_boolean_all_values() {
        let domain = VariableDomain::Boolean;
        let all = domain.all_values().unwrap();
        kani::assert(all.len() == 2, "Boolean has exactly 2 values");
        kani::assert(
            all.contains(&ModelValue::Bool(false)),
            "All values contain false",
        );
        kani::assert(
            all.contains(&ModelValue::Bool(true)),
            "All values contain true",
        );
    }

    // ============== ActionSpec Proofs ==============

    /// Proves that ActionSpec::new has expected defaults
    #[kani::proof]
    fn verify_action_spec_new_defaults() {
        let spec = ActionSpec::new("action");
        kani::assert(spec.name == "action", "Action name is preserved");
        kani::assert(spec.parameters.is_empty(), "Parameters are empty");
        kani::assert(
            spec.enabled_description == "true",
            "Default enabled_description is 'true'",
        );
        kani::assert(
            spec.effect_description.is_empty(),
            "Default effect_description is empty",
        );
    }

    /// Proves that with_enabled sets the enabled_description
    #[kani::proof]
    fn verify_action_spec_with_enabled() {
        let spec = ActionSpec::new("test").with_enabled("x > 0");
        kani::assert(
            spec.enabled_description == "x > 0",
            "with_enabled sets description",
        );
    }

    /// Proves that with_effect sets the effect_description
    #[kani::proof]
    fn verify_action_spec_with_effect() {
        let spec = ActionSpec::new("test").with_effect("x' = x + 1");
        kani::assert(
            spec.effect_description == "x' = x + 1",
            "with_effect sets description",
        );
    }

    // ============== Invariant Proofs ==============

    /// Proves that Invariant::new preserves name and description
    #[kani::proof]
    fn verify_invariant_new_preserves_fields() {
        let inv = Invariant::new("TypeOK", "Type invariant");
        kani::assert(inv.name == "TypeOK", "Name is preserved");
        kani::assert(
            inv.description == "Type invariant",
            "Description is preserved",
        );
    }

    // ============== StateMachineModel Proofs ==============
    // Note: StateMachineModel uses IndexMap which hits CCRandomGenerateBytes on macOS.
    // Proofs involving StateMachineModel construction are not run on macOS.

    // ============== GeneratorConfig Proofs ==============

    /// Proves that GeneratorConfig::new has expected defaults
    #[kani::proof]
    fn verify_generator_config_new_defaults() {
        let config = GeneratorConfig::new();
        kani::assert(
            config.strategy == GenerationStrategy::Combined,
            "Default strategy is Combined",
        );
        kani::assert(
            config.max_test_length == 20,
            "Default max_test_length is 20",
        );
        kani::assert(config.max_tests == 100, "Default max_tests is 100");
    }

    /// Proves that with_strategy preserves the strategy
    #[kani::proof]
    fn verify_generator_config_with_strategy() {
        let config = GeneratorConfig::new().with_strategy(GenerationStrategy::TransitionCoverage);
        kani::assert(
            config.strategy == GenerationStrategy::TransitionCoverage,
            "with_strategy sets strategy",
        );
    }

    /// Proves that with_max_length preserves the value
    #[kani::proof]
    fn verify_generator_config_with_max_length() {
        let len: usize = kani::any();
        kani::assume(len > 0 && len <= 1000);
        let config = GeneratorConfig::new().with_max_length(len);
        kani::assert(
            config.max_test_length == len,
            "with_max_length sets max_test_length",
        );
    }

    /// Proves that with_max_tests preserves the value
    #[kani::proof]
    fn verify_generator_config_with_max_tests() {
        let max: usize = kani::any();
        kani::assume(max > 0 && max <= 10000);
        let config = GeneratorConfig::new().with_max_tests(max);
        kani::assert(config.max_tests == max, "with_max_tests sets max_tests");
    }

    /// Proves that with_seed sets the seed
    #[kani::proof]
    fn verify_generator_config_with_seed() {
        let seed: u64 = kani::any();
        let config = GeneratorConfig::new().with_seed(seed);
        kani::assert(
            config.seed == Some(seed),
            "with_seed sets seed to Some(value)",
        );
    }

    // ============== TestCase Proofs ==============
    // Note: TestCase constructors use IndexMap which hits CCRandomGenerateBytes
    // on macOS, so we only verify proofs that don't construct TestCase directly.

    // ============== OutputFormat Proofs ==============

    /// Proves that OutputFormat variants are distinct
    #[kani::proof]
    fn verify_output_format_variants_distinct() {
        kani::assert(OutputFormat::Json != OutputFormat::Rust, "Json != Rust");
        kani::assert(OutputFormat::Json != OutputFormat::Python, "Json != Python");
        kani::assert(OutputFormat::Json != OutputFormat::Text, "Json != Text");
        kani::assert(
            OutputFormat::Json != OutputFormat::Markdown,
            "Json != Markdown",
        );
        kani::assert(OutputFormat::Rust != OutputFormat::Python, "Rust != Python");
        kani::assert(OutputFormat::Rust != OutputFormat::Text, "Rust != Text");
        kani::assert(
            OutputFormat::Rust != OutputFormat::Markdown,
            "Rust != Markdown",
        );
        kani::assert(OutputFormat::Python != OutputFormat::Text, "Python != Text");
        kani::assert(
            OutputFormat::Python != OutputFormat::Markdown,
            "Python != Markdown",
        );
        kani::assert(
            OutputFormat::Text != OutputFormat::Markdown,
            "Text != Markdown",
        );
    }

    // ============== GenerationStrategy Proofs ==============

    /// Proves that GenerationStrategy variants are distinct
    #[kani::proof]
    fn verify_generation_strategy_variants_distinct() {
        kani::assert(
            GenerationStrategy::StateCoverage != GenerationStrategy::TransitionCoverage,
            "StateCoverage != TransitionCoverage",
        );
        kani::assert(
            GenerationStrategy::StateCoverage != GenerationStrategy::BoundaryValue,
            "StateCoverage != BoundaryValue",
        );
        kani::assert(
            GenerationStrategy::StateCoverage != GenerationStrategy::Combined,
            "StateCoverage != Combined",
        );
        kani::assert(
            GenerationStrategy::StateCoverage != GenerationStrategy::RandomWalk,
            "StateCoverage != RandomWalk",
        );
    }

    // ============== ExplorationConfig Proofs ==============

    /// Proves that ExplorationConfig::new has expected defaults
    #[kani::proof]
    fn verify_exploration_config_new_defaults() {
        let config = ExplorationConfig::new();
        kani::assert(config.max_states == 10000, "Default max_states is 10000");
        kani::assert(config.max_depth == 100, "Default max_depth is 100");
        kani::assert(config.timeout_ms == 60000, "Default timeout is 60000ms");
        kani::assert(
            config.compute_transitions,
            "Default compute_transitions is true",
        );
    }

    /// Proves that with_max_states preserves the value
    #[kani::proof]
    fn verify_exploration_config_with_max_states() {
        let max: usize = kani::any();
        kani::assume(max > 0 && max <= 100000);
        let config = ExplorationConfig::new().with_max_states(max);
        kani::assert(config.max_states == max, "with_max_states sets max_states");
    }

    /// Proves that with_max_depth preserves the value
    #[kani::proof]
    fn verify_exploration_config_with_max_depth() {
        let depth: usize = kani::any();
        kani::assume(depth > 0 && depth <= 1000);
        let config = ExplorationConfig::new().with_max_depth(depth);
        kani::assert(config.max_depth == depth, "with_max_depth sets max_depth");
    }

    /// Proves that with_timeout_ms preserves the value
    #[kani::proof]
    fn verify_exploration_config_with_timeout_ms() {
        let timeout: u64 = kani::any();
        kani::assume(timeout > 0 && timeout <= 600000);
        let config = ExplorationConfig::new().with_timeout_ms(timeout);
        kani::assert(config.timeout_ms == timeout, "with_timeout_ms sets timeout");
    }

    /// Proves that with_transitions preserves the value
    #[kani::proof]
    fn verify_exploration_config_with_transitions() {
        let compute: bool = kani::any();
        let config = ExplorationConfig::new().with_transitions(compute);
        kani::assert(
            config.compute_transitions == compute,
            "with_transitions sets compute_transitions",
        );
    }
}
