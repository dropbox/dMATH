//! State space exploration for model-based testing
//!
//! This module provides functionality to explore the state space of a model
//! and discover all reachable states and transitions.

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::error::MbtResult;
use crate::model::{ModelAction, ModelState, ModelTransition, StateMachineModel};

/// Configuration for state space exploration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationConfig {
    /// Maximum number of states to explore
    pub max_states: usize,
    /// Maximum depth for BFS exploration
    pub max_depth: usize,
    /// Timeout in milliseconds
    pub timeout_ms: u64,
    /// Whether to compute full transition graph
    pub compute_transitions: bool,
    /// Enable verbose output
    pub verbose: bool,
}

impl Default for ExplorationConfig {
    fn default() -> Self {
        Self {
            max_states: 10000,
            max_depth: 100,
            timeout_ms: 60000,
            compute_transitions: true,
            verbose: false,
        }
    }
}

impl ExplorationConfig {
    /// Create a new configuration
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum states
    #[must_use]
    pub fn with_max_states(mut self, max: usize) -> Self {
        self.max_states = max;
        self
    }

    /// Set maximum depth
    #[must_use]
    pub fn with_max_depth(mut self, max: usize) -> Self {
        self.max_depth = max;
        self
    }

    /// Set timeout
    #[must_use]
    pub fn with_timeout_ms(mut self, timeout: u64) -> Self {
        self.timeout_ms = timeout;
        self
    }

    /// Enable/disable transition computation
    #[must_use]
    pub fn with_transitions(mut self, compute: bool) -> Self {
        self.compute_transitions = compute;
        self
    }
}

/// Result of state space exploration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationResult {
    /// All discovered states
    pub states: Vec<ModelState>,
    /// State index by canonical string
    #[serde(skip)]
    pub state_index: HashMap<String, usize>,
    /// All discovered transitions (if computed)
    pub transitions: Vec<TransitionRecord>,
    /// Actions that were never enabled
    pub dead_actions: Vec<String>,
    /// Number of states explored
    pub states_explored: usize,
    /// Number of transitions explored
    pub transitions_explored: usize,
    /// Maximum depth reached
    pub max_depth_reached: usize,
    /// Exploration duration in milliseconds
    pub duration_ms: u64,
    /// Whether exploration was complete (no timeout/limits hit)
    pub complete: bool,
    /// Reason for incomplete exploration (if any)
    pub incomplete_reason: Option<String>,
}

/// A recorded transition with indices into the state list
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionRecord {
    /// Source state index
    pub from_idx: usize,
    /// Target state index
    pub to_idx: usize,
    /// Action that triggered the transition
    pub action: ModelAction,
}

impl ExplorationResult {
    /// Create a new empty result
    #[must_use]
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            state_index: HashMap::new(),
            transitions: Vec::new(),
            dead_actions: Vec::new(),
            states_explored: 0,
            transitions_explored: 0,
            max_depth_reached: 0,
            duration_ms: 0,
            complete: false,
            incomplete_reason: None,
        }
    }

    /// Get state by index
    #[must_use]
    pub fn get_state(&self, idx: usize) -> Option<&ModelState> {
        self.states.get(idx)
    }

    /// Get index of a state
    #[must_use]
    pub fn state_index(&self, state: &ModelState) -> Option<usize> {
        self.state_index.get(&state.canonical_string()).copied()
    }

    /// Get all transitions from a state
    pub fn transitions_from(&self, state_idx: usize) -> impl Iterator<Item = &TransitionRecord> {
        self.transitions
            .iter()
            .filter(move |t| t.from_idx == state_idx)
    }

    /// Get all transitions to a state
    pub fn transitions_to(&self, state_idx: usize) -> impl Iterator<Item = &TransitionRecord> {
        self.transitions
            .iter()
            .filter(move |t| t.to_idx == state_idx)
    }

    /// Get unique action names
    #[must_use]
    pub fn unique_actions(&self) -> HashSet<String> {
        self.transitions
            .iter()
            .map(|t| t.action.name.clone())
            .collect()
    }

    /// Check if all actions were exercised
    #[must_use]
    pub fn all_actions_exercised(&self, model: &StateMachineModel) -> bool {
        let exercised = self.unique_actions();
        model.action_names().all(|a| exercised.contains(a))
    }

    /// Get state coverage as fraction
    #[must_use]
    pub fn state_coverage(&self) -> f64 {
        if self.states_explored == 0 {
            0.0
        } else {
            self.states.len() as f64 / self.states_explored as f64
        }
    }
}

impl Default for ExplorationResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for state machine executors that can compute next states
pub trait StateExecutor: Send + Sync {
    /// Get all enabled actions and their resulting states from a given state
    fn enabled_transitions(&self, state: &ModelState) -> Vec<(ModelAction, ModelState)>;

    /// Check if an action is enabled in a state
    fn is_enabled(&self, state: &ModelState, action: &str) -> bool {
        self.enabled_transitions(state)
            .iter()
            .any(|(a, _)| a.name == action)
    }
}

/// A simple executor that uses pre-computed transitions
pub struct PrecomputedExecutor {
    /// Map from state canonical string to list of (action, next_state)
    transitions: HashMap<String, Vec<(ModelAction, ModelState)>>,
}

impl PrecomputedExecutor {
    /// Create a new precomputed executor
    #[must_use]
    pub fn new() -> Self {
        Self {
            transitions: HashMap::new(),
        }
    }

    /// Add a transition
    pub fn add_transition(&mut self, from: &ModelState, action: ModelAction, to: ModelState) {
        let key = from.canonical_string();
        self.transitions.entry(key).or_default().push((action, to));
    }
}

impl Default for PrecomputedExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl StateExecutor for PrecomputedExecutor {
    fn enabled_transitions(&self, state: &ModelState) -> Vec<(ModelAction, ModelState)> {
        self.transitions
            .get(&state.canonical_string())
            .cloned()
            .unwrap_or_default()
    }
}

/// State space explorer using BFS
pub struct StateExplorer<E: StateExecutor> {
    /// Executor for computing transitions
    executor: E,
    /// Configuration
    config: ExplorationConfig,
}

impl<E: StateExecutor> StateExplorer<E> {
    /// Create a new explorer with the given executor
    #[must_use]
    pub fn new(executor: E) -> Self {
        Self {
            executor,
            config: ExplorationConfig::default(),
        }
    }

    /// Create with custom configuration
    #[must_use]
    pub fn with_config(executor: E, config: ExplorationConfig) -> Self {
        Self { executor, config }
    }

    /// Explore the state space starting from initial states
    pub fn explore(&self, initial_states: &[ModelState]) -> MbtResult<ExplorationResult> {
        let start_time = Instant::now();
        let timeout = Duration::from_millis(self.config.timeout_ms);

        let mut result = ExplorationResult::new();
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(ModelState, usize)> = VecDeque::new();

        // Add initial states
        for state in initial_states {
            let canonical = state.canonical_string();
            if visited.insert(canonical.clone()) {
                let idx = result.states.len();
                result.state_index.insert(canonical, idx);
                result.states.push(state.clone());
                queue.push_back((state.clone(), 0));
            }
        }

        // BFS exploration
        while let Some((current_state, depth)) = queue.pop_front() {
            // Check timeout
            if start_time.elapsed() > timeout {
                result.incomplete_reason = Some("Timeout".into());
                break;
            }

            // Check state limit
            if result.states.len() >= self.config.max_states {
                result.incomplete_reason = Some("Max states reached".into());
                break;
            }

            result.states_explored += 1;
            result.max_depth_reached = result.max_depth_reached.max(depth);

            // Check depth limit
            if depth >= self.config.max_depth {
                continue;
            }

            let current_idx = result
                .state_index
                .get(&current_state.canonical_string())
                .copied()
                .unwrap_or(0);

            // Get enabled transitions
            let transitions = self.executor.enabled_transitions(&current_state);

            for (action, next_state) in transitions {
                result.transitions_explored += 1;

                let canonical = next_state.canonical_string();
                let next_idx = if let Some(&idx) = result.state_index.get(&canonical) {
                    idx
                } else {
                    let idx = result.states.len();
                    result.state_index.insert(canonical.clone(), idx);
                    result.states.push(next_state.clone());
                    queue.push_back((next_state, depth + 1));
                    idx
                };

                if self.config.compute_transitions {
                    result.transitions.push(TransitionRecord {
                        from_idx: current_idx,
                        to_idx: next_idx,
                        action,
                    });
                }
            }
        }

        result.complete = result.incomplete_reason.is_none() && queue.is_empty();
        result.duration_ms = start_time.elapsed().as_millis() as u64;

        Ok(result)
    }

    /// Explore with depth-first search (useful for finding long traces)
    pub fn explore_dfs(
        &self,
        initial_states: &[ModelState],
        target_depth: usize,
    ) -> MbtResult<Vec<Vec<ModelTransition>>> {
        let mut traces = Vec::new();
        let mut visited: HashSet<String> = HashSet::new();

        for initial in initial_states {
            let mut stack: Vec<(ModelState, Vec<ModelTransition>)> =
                vec![(initial.clone(), vec![])];

            while let Some((current, path)) = stack.pop() {
                let canonical = current.canonical_string();

                if path.len() >= target_depth {
                    if !path.is_empty() {
                        traces.push(path);
                    }
                    continue;
                }

                if !visited.insert(canonical.clone()) {
                    continue;
                }

                let transitions = self.executor.enabled_transitions(&current);

                if transitions.is_empty() && !path.is_empty() {
                    // Terminal state reached
                    traces.push(path);
                } else {
                    for (action, next_state) in transitions {
                        let mut new_path = path.clone();
                        new_path.push(ModelTransition::new(
                            current.clone(),
                            action,
                            next_state.clone(),
                        ));
                        stack.push((next_state, new_path));
                    }
                }
            }
        }

        Ok(traces)
    }
}

/// Find the shortest path between two states
pub fn find_path<E: StateExecutor>(
    executor: &E,
    from: &ModelState,
    to: &ModelState,
    max_depth: usize,
) -> Option<Vec<ModelTransition>> {
    let to_canonical = to.canonical_string();
    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<(ModelState, Vec<ModelTransition>)> = VecDeque::new();

    queue.push_back((from.clone(), vec![]));
    visited.insert(from.canonical_string());

    while let Some((current, path)) = queue.pop_front() {
        if path.len() >= max_depth {
            continue;
        }

        for (action, next_state) in executor.enabled_transitions(&current) {
            let canonical = next_state.canonical_string();

            if canonical == to_canonical {
                let mut result = path;
                result.push(ModelTransition::new(current, action, next_state));
                return Some(result);
            }

            if visited.insert(canonical) {
                let mut new_path = path.clone();
                new_path.push(ModelTransition::new(
                    current.clone(),
                    action,
                    next_state.clone(),
                ));
                queue.push_back((next_state, new_path));
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelValue;

    fn create_counter_executor() -> PrecomputedExecutor {
        let mut exec = PrecomputedExecutor::new();

        // Simple counter: 0 -> 1 -> 2 (cycle back to 0)
        for i in 0..3 {
            let mut from = ModelState::new();
            from.set("count", ModelValue::Int(i));

            let mut to = ModelState::new();
            to.set("count", ModelValue::Int((i + 1) % 3));

            exec.add_transition(&from, ModelAction::new("increment"), to);
        }

        exec
    }

    #[test]
    fn test_bfs_exploration() {
        let exec = create_counter_executor();
        let explorer = StateExplorer::new(exec);

        let mut initial = ModelState::new();
        initial.set("count", ModelValue::Int(0));

        let result = explorer.explore(&[initial]).unwrap();

        assert_eq!(result.states.len(), 3);
        assert_eq!(result.transitions.len(), 3);
        assert!(result.complete);
    }

    #[test]
    fn test_find_path() {
        let exec = create_counter_executor();

        let mut from = ModelState::new();
        from.set("count", ModelValue::Int(0));

        let mut to = ModelState::new();
        to.set("count", ModelValue::Int(2));

        let path = find_path(&exec, &from, &to, 10).unwrap();
        assert_eq!(path.len(), 2);
    }

    #[test]
    fn test_exploration_limits() {
        let exec = create_counter_executor();
        let config = ExplorationConfig::new().with_max_states(2);
        let explorer = StateExplorer::with_config(exec, config);

        let mut initial = ModelState::new();
        initial.set("count", ModelValue::Int(0));

        let result = explorer.explore(&[initial]).unwrap();

        assert_eq!(result.states.len(), 2);
        assert!(!result.complete);
    }
}
