//! Test case generation from state machine models
//!
//! This module provides various test generation strategies:
//! - State coverage: Generate tests to visit all states
//! - Transition coverage: Generate tests to exercise all transitions
//! - Boundary value: Generate tests at domain boundaries
//! - Random walk: Generate random exploration tests

use std::collections::{HashMap, HashSet};

use rand::prelude::*;
use serde::{Deserialize, Serialize};

use crate::error::{MbtError, MbtResult};
use crate::explorer::{ExplorationResult, TransitionRecord};
use crate::model::{ModelAction, ModelState, ModelTransition, ModelValue, StateMachineModel};

/// A generated test case
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    /// Test case identifier
    pub id: String,
    /// Description of what this test covers
    pub description: String,
    /// Initial state
    pub initial_state: ModelState,
    /// Sequence of actions to execute
    pub actions: Vec<ModelAction>,
    /// Expected states after each action
    pub expected_states: Vec<ModelState>,
    /// Coverage goals this test satisfies
    pub covers: Vec<CoverageGoal>,
}

impl TestCase {
    /// Create a new test case
    #[must_use]
    pub fn new(id: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            initial_state: ModelState::new(),
            actions: Vec::new(),
            expected_states: Vec::new(),
            covers: Vec::new(),
        }
    }

    /// Set initial state
    #[must_use]
    pub fn with_initial_state(mut self, state: ModelState) -> Self {
        self.initial_state = state;
        self
    }

    /// Add an action and expected resulting state
    pub fn add_step(&mut self, action: ModelAction, expected_state: ModelState) {
        self.actions.push(action);
        self.expected_states.push(expected_state);
    }

    /// Add a coverage goal
    pub fn add_coverage(&mut self, goal: CoverageGoal) {
        self.covers.push(goal);
    }

    /// Get the final expected state
    #[must_use]
    pub fn final_state(&self) -> Option<&ModelState> {
        self.expected_states.last()
    }

    /// Get test length (number of actions)
    #[must_use]
    pub fn length(&self) -> usize {
        self.actions.len()
    }

    /// Convert to a trace of transitions
    #[must_use]
    pub fn to_trace(&self) -> Vec<ModelTransition> {
        let mut trace = Vec::new();
        let mut current = self.initial_state.clone();

        for (action, next_state) in self.actions.iter().zip(self.expected_states.iter()) {
            trace.push(ModelTransition::new(
                current.clone(),
                action.clone(),
                next_state.clone(),
            ));
            current = next_state.clone();
        }

        trace
    }
}

/// A coverage goal that a test can satisfy
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CoverageGoal {
    /// Visit a specific state
    State(String),
    /// Exercise a specific transition
    Transition {
        action: String,
        from: String,
        to: String,
    },
    /// Test a boundary value for a variable
    BoundaryValue { variable: String, value: String },
    /// Execute a specific action
    Action(String),
    /// Reach a specific path
    Path(Vec<String>),
}

impl CoverageGoal {
    /// Create a state coverage goal
    #[must_use]
    pub fn state(state: &ModelState) -> Self {
        Self::State(state.canonical_string())
    }

    /// Create a transition coverage goal
    #[must_use]
    pub fn transition(action: &str, from: &ModelState, to: &ModelState) -> Self {
        Self::Transition {
            action: action.to_string(),
            from: from.canonical_string(),
            to: to.canonical_string(),
        }
    }

    /// Create a boundary value goal
    #[must_use]
    pub fn boundary(variable: &str, value: &ModelValue) -> Self {
        Self::BoundaryValue {
            variable: variable.to_string(),
            value: value.canonical_string(),
        }
    }

    /// Create an action coverage goal
    #[must_use]
    pub fn action(action: &str) -> Self {
        Self::Action(action.to_string())
    }
}

/// Test generation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GenerationStrategy {
    /// Generate tests to cover all states
    StateCoverage,
    /// Generate tests to cover all transitions
    TransitionCoverage,
    /// Generate tests at domain boundaries
    BoundaryValue,
    /// Combine all strategies
    Combined,
    /// Generate random walks
    RandomWalk,
}

/// Configuration for test generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorConfig {
    /// Generation strategy
    pub strategy: GenerationStrategy,
    /// Maximum test length
    pub max_test_length: usize,
    /// Maximum number of tests to generate
    pub max_tests: usize,
    /// Seed for random generation
    pub seed: Option<u64>,
    /// Include state assertions in tests
    pub include_assertions: bool,
    /// Prefix for test IDs
    pub test_id_prefix: String,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            strategy: GenerationStrategy::Combined,
            max_test_length: 20,
            max_tests: 100,
            seed: None,
            include_assertions: true,
            test_id_prefix: "test".into(),
        }
    }
}

impl GeneratorConfig {
    /// Create a new configuration
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set strategy
    #[must_use]
    pub fn with_strategy(mut self, strategy: GenerationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set max test length
    #[must_use]
    pub fn with_max_length(mut self, max: usize) -> Self {
        self.max_test_length = max;
        self
    }

    /// Set max tests
    #[must_use]
    pub fn with_max_tests(mut self, max: usize) -> Self {
        self.max_tests = max;
        self
    }

    /// Set seed
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Result of test generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResult {
    /// Generated test cases
    pub tests: Vec<TestCase>,
    /// Coverage achieved
    pub coverage: CoverageReport,
    /// Generation statistics
    pub stats: GenerationStats,
}

/// Coverage report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageReport {
    /// States covered
    pub states_covered: usize,
    /// Total states
    pub states_total: usize,
    /// Transitions covered
    pub transitions_covered: usize,
    /// Total transitions
    pub transitions_total: usize,
    /// Actions covered
    pub actions_covered: usize,
    /// Total actions
    pub actions_total: usize,
    /// Boundary values covered
    pub boundaries_covered: usize,
    /// Total boundary values
    pub boundaries_total: usize,
    /// Uncovered goals
    pub uncovered: Vec<CoverageGoal>,
}

impl CoverageReport {
    /// Calculate state coverage percentage
    #[must_use]
    pub fn state_coverage_pct(&self) -> f64 {
        if self.states_total == 0 {
            100.0
        } else {
            100.0 * self.states_covered as f64 / self.states_total as f64
        }
    }

    /// Calculate transition coverage percentage
    #[must_use]
    pub fn transition_coverage_pct(&self) -> f64 {
        if self.transitions_total == 0 {
            100.0
        } else {
            100.0 * self.transitions_covered as f64 / self.transitions_total as f64
        }
    }

    /// Calculate overall coverage
    #[must_use]
    pub fn overall_coverage_pct(&self) -> f64 {
        let total = self.states_total + self.transitions_total + self.boundaries_total;
        if total == 0 {
            100.0
        } else {
            let covered = self.states_covered + self.transitions_covered + self.boundaries_covered;
            100.0 * covered as f64 / total as f64
        }
    }
}

/// Generation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationStats {
    /// Number of tests generated
    pub tests_generated: usize,
    /// Total test steps
    pub total_steps: usize,
    /// Average test length
    pub avg_test_length: f64,
    /// Maximum test length
    pub max_test_length: usize,
    /// Generation duration in milliseconds
    pub duration_ms: u64,
}

/// Test generator from exploration results
pub struct TestGenerator {
    /// Configuration
    config: GeneratorConfig,
    /// Random number generator
    rng: StdRng,
}

impl TestGenerator {
    /// Create a new generator with default config
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: GeneratorConfig::default(),
            rng: StdRng::from_entropy(),
        }
    }

    /// Create with custom configuration
    #[must_use]
    pub fn with_config(config: GeneratorConfig) -> Self {
        let rng = if let Some(seed) = config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };
        Self { config, rng }
    }

    /// Generate tests from exploration result
    pub fn generate(&mut self, exploration: &ExplorationResult) -> MbtResult<GenerationResult> {
        let start = std::time::Instant::now();

        let tests = match self.config.strategy {
            GenerationStrategy::StateCoverage => self.generate_state_coverage(exploration)?,
            GenerationStrategy::TransitionCoverage => {
                self.generate_transition_coverage(exploration)?
            }
            GenerationStrategy::BoundaryValue => {
                // Boundary values need the model, not just exploration
                return Err(MbtError::TestGenerationFailed(
                    "Boundary value generation requires model".into(),
                ));
            }
            GenerationStrategy::RandomWalk => self.generate_random_walks(exploration)?,
            GenerationStrategy::Combined => {
                let mut all_tests = self.generate_state_coverage(exploration)?;
                all_tests.extend(self.generate_transition_coverage(exploration)?);
                all_tests.extend(self.generate_random_walks(exploration)?);
                self.deduplicate_tests(all_tests)
            }
        };

        let coverage = self.compute_coverage(&tests, exploration);
        let stats = GenerationStats {
            tests_generated: tests.len(),
            total_steps: tests.iter().map(|t| t.length()).sum(),
            avg_test_length: if tests.is_empty() {
                0.0
            } else {
                tests.iter().map(|t| t.length()).sum::<usize>() as f64 / tests.len() as f64
            },
            max_test_length: tests.iter().map(|t| t.length()).max().unwrap_or(0),
            duration_ms: start.elapsed().as_millis() as u64,
        };

        Ok(GenerationResult {
            tests,
            coverage,
            stats,
        })
    }

    /// Generate tests for state coverage
    fn generate_state_coverage(
        &mut self,
        exploration: &ExplorationResult,
    ) -> MbtResult<Vec<TestCase>> {
        let mut tests = Vec::new();
        let mut covered_states: HashSet<String> = HashSet::new();

        // Find paths to each state from initial states
        for (state_idx, state) in exploration.states.iter().enumerate() {
            let canonical = state.canonical_string();
            if covered_states.contains(&canonical) {
                continue;
            }

            // Find shortest path to this state
            if let Some(path) = self.find_path_to_state(exploration, state_idx) {
                let mut test = TestCase::new(
                    format!("{}_{}", self.config.test_id_prefix, tests.len()),
                    format!("Reach state {}", canonical),
                );

                if let Some(initial_idx) = path.first().map(|t| t.from_idx) {
                    if let Some(initial) = exploration.get_state(initial_idx) {
                        test = test.with_initial_state(initial.clone());
                    }
                }

                for trans in &path {
                    if let (Some(to_state), Some(_from_state)) = (
                        exploration.get_state(trans.to_idx),
                        exploration.get_state(trans.from_idx),
                    ) {
                        test.add_step(trans.action.clone(), to_state.clone());
                        covered_states.insert(to_state.canonical_string());
                    }
                }

                test.add_coverage(CoverageGoal::state(state));
                tests.push(test);

                if tests.len() >= self.config.max_tests {
                    break;
                }
            }
        }

        Ok(tests)
    }

    /// Generate tests for transition coverage
    fn generate_transition_coverage(
        &mut self,
        exploration: &ExplorationResult,
    ) -> MbtResult<Vec<TestCase>> {
        let mut tests = Vec::new();
        let mut covered_transitions: HashSet<(usize, usize, String)> = HashSet::new();

        for trans in &exploration.transitions {
            let key = (trans.from_idx, trans.to_idx, trans.action.name.clone());
            if covered_transitions.contains(&key) {
                continue;
            }

            // Find path to source state, then add the transition
            if let Some(mut path) = self.find_path_to_state(exploration, trans.from_idx) {
                path.push(trans.clone());

                let mut test = TestCase::new(
                    format!("{}_{}", self.config.test_id_prefix, tests.len()),
                    format!("Execute transition {}", trans.action.signature()),
                );

                if let Some(initial_idx) = path.first().map(|t| t.from_idx) {
                    if let Some(initial) = exploration.get_state(initial_idx) {
                        test = test.with_initial_state(initial.clone());
                    }
                }

                for t in &path {
                    if let Some(to_state) = exploration.get_state(t.to_idx) {
                        test.add_step(t.action.clone(), to_state.clone());
                        covered_transitions.insert((t.from_idx, t.to_idx, t.action.name.clone()));
                    }
                }

                if let (Some(from), Some(to)) = (
                    exploration.get_state(trans.from_idx),
                    exploration.get_state(trans.to_idx),
                ) {
                    test.add_coverage(CoverageGoal::transition(&trans.action.name, from, to));
                }
                tests.push(test);

                if tests.len() >= self.config.max_tests {
                    break;
                }
            }
        }

        Ok(tests)
    }

    /// Generate random walk tests
    fn generate_random_walks(
        &mut self,
        exploration: &ExplorationResult,
    ) -> MbtResult<Vec<TestCase>> {
        let mut tests = Vec::new();

        if exploration.states.is_empty() {
            return Ok(tests);
        }

        // Build adjacency map
        let mut adjacency: HashMap<usize, Vec<&TransitionRecord>> = HashMap::new();
        for trans in &exploration.transitions {
            adjacency.entry(trans.from_idx).or_default().push(trans);
        }

        let num_walks = self.config.max_tests.min(10);
        for walk_idx in 0..num_walks {
            let initial_idx = self.rng.gen_range(0..exploration.states.len());
            let initial = exploration.get_state(initial_idx).unwrap().clone();

            let mut test = TestCase::new(
                format!("{}_random_{}", self.config.test_id_prefix, walk_idx),
                "Random walk test",
            )
            .with_initial_state(initial);

            let mut current_idx = initial_idx;
            for _ in 0..self.config.max_test_length {
                if let Some(outgoing) = adjacency.get(&current_idx) {
                    if outgoing.is_empty() {
                        break;
                    }
                    let trans = outgoing[self.rng.gen_range(0..outgoing.len())];
                    if let Some(to_state) = exploration.get_state(trans.to_idx) {
                        test.add_step(trans.action.clone(), to_state.clone());
                        current_idx = trans.to_idx;
                    }
                } else {
                    break;
                }
            }

            if test.length() > 0 {
                tests.push(test);
            }
        }

        Ok(tests)
    }

    /// Find shortest path to a state (BFS from initial states)
    fn find_path_to_state(
        &self,
        exploration: &ExplorationResult,
        target_idx: usize,
    ) -> Option<Vec<TransitionRecord>> {
        // Initial states have index 0 by convention (first in exploration)
        if target_idx == 0 {
            return Some(vec![]);
        }

        // Build reverse adjacency for BFS
        let mut predecessors: HashMap<usize, Vec<&TransitionRecord>> = HashMap::new();
        for trans in &exploration.transitions {
            predecessors.entry(trans.to_idx).or_default().push(trans);
        }

        // BFS from target back to initial state
        let mut visited: HashSet<usize> = HashSet::new();
        let mut queue: std::collections::VecDeque<(usize, Vec<TransitionRecord>)> =
            std::collections::VecDeque::new();

        queue.push_back((target_idx, vec![]));
        visited.insert(target_idx);

        while let Some((current, path)) = queue.pop_front() {
            if current == 0 {
                // Reached initial state, reverse path
                let result: Vec<_> = path.into_iter().rev().collect();
                return Some(result);
            }

            if let Some(preds) = predecessors.get(&current) {
                for trans in preds {
                    if visited.insert(trans.from_idx) {
                        let mut new_path = path.clone();
                        new_path.push((*trans).clone());
                        queue.push_back((trans.from_idx, new_path));
                    }
                }
            }
        }

        None
    }

    /// Deduplicate tests by coverage
    fn deduplicate_tests(&self, tests: Vec<TestCase>) -> Vec<TestCase> {
        let mut seen_coverage: HashSet<Vec<CoverageGoal>> = HashSet::new();
        let mut result = Vec::new();

        for test in tests {
            let mut sorted_covers = test.covers.clone();
            sorted_covers.sort_by_key(|c| format!("{c:?}"));

            if seen_coverage.insert(sorted_covers) {
                result.push(test);
            }
        }

        result
    }

    /// Compute coverage achieved by tests
    fn compute_coverage(
        &self,
        tests: &[TestCase],
        exploration: &ExplorationResult,
    ) -> CoverageReport {
        let mut covered_states: HashSet<String> = HashSet::new();
        let mut covered_transitions: HashSet<(String, String, String)> = HashSet::new();
        let mut covered_actions: HashSet<String> = HashSet::new();

        for test in tests {
            covered_states.insert(test.initial_state.canonical_string());
            for (action, state) in test.actions.iter().zip(test.expected_states.iter()) {
                covered_states.insert(state.canonical_string());
                covered_actions.insert(action.name.clone());
            }

            for goal in &test.covers {
                if let CoverageGoal::Transition { action, from, to } = goal {
                    covered_transitions.insert((action.clone(), from.clone(), to.clone()));
                }
            }
        }

        let all_actions: HashSet<String> = exploration
            .transitions
            .iter()
            .map(|t| t.action.name.clone())
            .collect();

        let uncovered: Vec<CoverageGoal> = exploration
            .states
            .iter()
            .filter(|s| !covered_states.contains(&s.canonical_string()))
            .map(CoverageGoal::state)
            .collect();

        CoverageReport {
            states_covered: covered_states.len(),
            states_total: exploration.states.len(),
            transitions_covered: covered_transitions.len(),
            transitions_total: exploration.transitions.len(),
            actions_covered: covered_actions.len(),
            actions_total: all_actions.len(),
            boundaries_covered: 0,
            boundaries_total: 0,
            uncovered,
        }
    }
}

impl Default for TestGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate tests with boundary values from a model
pub fn generate_boundary_tests(model: &StateMachineModel) -> MbtResult<Vec<TestCase>> {
    let mut tests = Vec::new();

    for (var_name, domain) in &model.variables {
        let boundaries = domain.boundary_values();

        for (idx, value) in boundaries.iter().enumerate() {
            let mut initial = ModelState::new();

            // Set all variables to their minimum values
            for (name, dom) in &model.variables {
                if name == var_name {
                    initial.set(name, value.clone());
                } else if let Some(min_val) = dom.min_value() {
                    initial.set(name, min_val);
                }
            }

            let test = TestCase::new(
                format!("boundary_{}_{}", var_name, idx),
                format!("Boundary test: {} = {}", var_name, value),
            )
            .with_initial_state(initial);

            tests.push(test);
        }
    }

    Ok(tests)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::explorer::{PrecomputedExecutor, StateExplorer};
    use crate::model::VariableDomain;

    fn create_simple_exploration() -> ExplorationResult {
        let mut exec = PrecomputedExecutor::new();

        // Simple: s0 -> s1 -> s2
        let mut s0 = ModelState::new();
        s0.set("x", ModelValue::Int(0));
        let mut s1 = ModelState::new();
        s1.set("x", ModelValue::Int(1));
        let mut s2 = ModelState::new();
        s2.set("x", ModelValue::Int(2));

        exec.add_transition(&s0, ModelAction::new("inc"), s1.clone());
        exec.add_transition(&s1, ModelAction::new("inc"), s2.clone());

        let explorer = StateExplorer::new(exec);
        explorer.explore(&[s0]).unwrap()
    }

    #[test]
    fn test_state_coverage_generation() {
        let exploration = create_simple_exploration();
        let config = GeneratorConfig::new()
            .with_strategy(GenerationStrategy::StateCoverage)
            .with_seed(42);
        let mut gen = TestGenerator::with_config(config);

        let result = gen.generate(&exploration).unwrap();
        assert!(!result.tests.is_empty());
        assert!(result.coverage.state_coverage_pct() > 0.0);
    }

    #[test]
    fn test_transition_coverage_generation() {
        let exploration = create_simple_exploration();
        let config = GeneratorConfig::new()
            .with_strategy(GenerationStrategy::TransitionCoverage)
            .with_seed(42);
        let mut gen = TestGenerator::with_config(config);

        let result = gen.generate(&exploration).unwrap();
        assert!(!result.tests.is_empty());
    }

    #[test]
    fn test_boundary_generation() {
        let model = StateMachineModel::new("test")
            .with_variable("count", VariableDomain::IntRange { min: 0, max: 10 });

        let tests = generate_boundary_tests(&model).unwrap();
        assert!(tests.len() >= 4); // min, max, min+1, max-1, possibly 0
    }
}
