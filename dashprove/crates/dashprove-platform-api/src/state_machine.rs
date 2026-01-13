//! State machine representation for platform APIs

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use std::collections::HashMap;

/// Unique identifier for a state in the state machine
pub type StateId = NodeIndex;

/// Unique identifier for a transition in the state machine
pub type TransitionId = petgraph::graph::EdgeIndex;

/// A finite state machine representing valid API call sequences
///
/// Note: This type does not implement Serialize/Deserialize because it's
/// rebuilt from PlatformApi on demand. The PlatformApi is the source of truth.
#[derive(Debug, Clone)]
pub struct StateMachine {
    /// The underlying directed graph
    graph: DiGraph<String, String>,
    /// Map from state names to node indices
    state_indices: HashMap<String, StateId>,
    /// The initial state
    initial_state: Option<StateId>,
    /// Terminal (accepting) states
    terminal_states: Vec<StateId>,
}

impl Default for StateMachine {
    fn default() -> Self {
        Self::new()
    }
}

impl StateMachine {
    /// Create a new empty state machine
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            state_indices: HashMap::new(),
            initial_state: None,
            terminal_states: Vec::new(),
        }
    }

    /// Add a state to the state machine
    pub fn add_state(&mut self, name: &str) -> StateId {
        if let Some(&idx) = self.state_indices.get(name) {
            return idx;
        }
        let idx = self.graph.add_node(name.to_string());
        self.state_indices.insert(name.to_string(), idx);
        idx
    }

    /// Set the initial state
    pub fn set_initial_state(&mut self, name: &str) -> Option<StateId> {
        let idx = self.state_indices.get(name).copied()?;
        self.initial_state = Some(idx);
        Some(idx)
    }

    /// Mark a state as terminal (accepting)
    pub fn mark_terminal(&mut self, name: &str) -> Option<StateId> {
        let idx = self.state_indices.get(name).copied()?;
        if !self.terminal_states.contains(&idx) {
            self.terminal_states.push(idx);
        }
        Some(idx)
    }

    /// Add a transition between states
    pub fn add_transition(&mut self, from: &str, to: &str, label: &str) -> Option<TransitionId> {
        let from_idx = self.state_indices.get(from).copied()?;
        let to_idx = self.state_indices.get(to).copied()?;
        Some(self.graph.add_edge(from_idx, to_idx, label.to_string()))
    }

    /// Get the initial state
    pub fn initial_state(&self) -> Option<StateId> {
        self.initial_state
    }

    /// Check if a state is terminal
    pub fn is_terminal(&self, state: StateId) -> bool {
        self.terminal_states.contains(&state)
    }

    /// Get all valid transitions from a given state
    pub fn transitions_from(&self, state: StateId) -> Vec<(String, StateId)> {
        self.graph
            .edges(state)
            .map(|edge| (edge.weight().clone(), edge.target()))
            .collect()
    }

    /// Get the state index for a state name
    pub fn state_id(&self, name: &str) -> Option<StateId> {
        self.state_indices.get(name).copied()
    }

    /// Get the state name for a state index
    pub fn state_name(&self, id: StateId) -> Option<&str> {
        self.graph.node_weight(id).map(|s| s.as_str())
    }

    /// Check if a transition is valid from the current state
    pub fn is_valid_transition(&self, from: StateId, action: &str) -> Option<StateId> {
        for edge in self.graph.edges(from) {
            if edge.weight() == action {
                return Some(edge.target());
            }
        }
        None
    }

    /// Execute a sequence of actions and return the final state (or error)
    pub fn execute_sequence(&self, actions: &[&str]) -> Result<StateId, (usize, String)> {
        let mut current = self
            .initial_state
            .ok_or((0, "No initial state".to_string()))?;

        for (idx, action) in actions.iter().enumerate() {
            match self.is_valid_transition(current, action) {
                Some(next) => current = next,
                None => {
                    let state_name = self.state_name(current).unwrap_or("unknown");
                    return Err((
                        idx,
                        format!(
                            "Invalid action '{}' from state '{}' at position {}",
                            action, state_name, idx
                        ),
                    ));
                }
            }
        }

        Ok(current)
    }

    /// Get all reachable states from the initial state
    pub fn reachable_states(&self) -> Vec<StateId> {
        let Some(initial) = self.initial_state else {
            return Vec::new();
        };

        let mut visited = vec![false; self.graph.node_count()];
        let mut stack = vec![initial];
        let mut result = Vec::new();

        while let Some(state) = stack.pop() {
            if visited[state.index()] {
                continue;
            }
            visited[state.index()] = true;
            result.push(state);

            for edge in self.graph.edges(state) {
                stack.push(edge.target());
            }
        }

        result
    }

    /// Check if all states are reachable from the initial state
    pub fn is_fully_reachable(&self) -> bool {
        let reachable = self.reachable_states();
        reachable.len() == self.graph.node_count()
    }

    /// Get all state names
    pub fn states(&self) -> Vec<&str> {
        self.state_indices.keys().map(|s| s.as_str()).collect()
    }

    /// Get number of states
    pub fn state_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get number of transitions
    pub fn transition_count(&self) -> usize {
        self.graph.edge_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_state_machine() {
        let mut sm = StateMachine::new();
        sm.add_state("Init");
        sm.add_state("Running");
        sm.add_state("Done");

        sm.set_initial_state("Init");
        sm.mark_terminal("Done");

        sm.add_transition("Init", "Running", "start");
        sm.add_transition("Running", "Done", "finish");

        assert_eq!(sm.state_count(), 3);
        assert_eq!(sm.transition_count(), 2);
    }

    #[test]
    fn test_execute_valid_sequence() {
        let mut sm = StateMachine::new();
        sm.add_state("Init");
        sm.add_state("Running");
        sm.add_state("Done");

        sm.set_initial_state("Init");
        sm.add_transition("Init", "Running", "start");
        sm.add_transition("Running", "Done", "finish");

        let result = sm.execute_sequence(&["start", "finish"]);
        assert!(result.is_ok());
        let final_state = result.unwrap();
        assert_eq!(sm.state_name(final_state), Some("Done"));
    }

    #[test]
    fn test_execute_invalid_sequence() {
        let mut sm = StateMachine::new();
        sm.add_state("Init");
        sm.add_state("Running");

        sm.set_initial_state("Init");
        sm.add_transition("Init", "Running", "start");

        let result = sm.execute_sequence(&["finish"]); // Invalid - can't finish from Init
        assert!(result.is_err());
        let (idx, _msg) = result.unwrap_err();
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_reachability() {
        let mut sm = StateMachine::new();
        sm.add_state("Init");
        sm.add_state("Reachable");
        sm.add_state("Unreachable");

        sm.set_initial_state("Init");
        sm.add_transition("Init", "Reachable", "go");
        // No transition to "Unreachable"

        assert!(!sm.is_fully_reachable());
        assert_eq!(sm.reachable_states().len(), 2);
    }

    #[test]
    fn test_mark_terminal_returns_some_for_valid_state() {
        let mut sm = StateMachine::new();
        sm.add_state("Init");
        sm.add_state("Done");

        let result = sm.mark_terminal("Done");
        assert!(
            result.is_some(),
            "mark_terminal should return Some for valid state"
        );

        // Verify the state is actually terminal
        let done_id = result.unwrap();
        assert!(sm.is_terminal(done_id), "State should be marked terminal");
    }

    #[test]
    fn test_mark_terminal_returns_none_for_invalid_state() {
        let mut sm = StateMachine::new();
        sm.add_state("Init");

        let result = sm.mark_terminal("NonExistent");
        assert!(
            result.is_none(),
            "mark_terminal should return None for invalid state"
        );
    }

    #[test]
    fn test_mark_terminal_does_not_duplicate() {
        let mut sm = StateMachine::new();
        sm.add_state("Done");

        // Mark terminal twice
        sm.mark_terminal("Done");
        sm.mark_terminal("Done");

        // Should only be in terminal_states once
        let done_id = sm.state_id("Done").unwrap();
        let terminal_count = sm
            .terminal_states
            .iter()
            .filter(|&&id| id == done_id)
            .count();
        assert_eq!(terminal_count, 1, "Terminal state should not be duplicated");
    }

    #[test]
    fn test_initial_state_returns_some_when_set() {
        let mut sm = StateMachine::new();
        sm.add_state("Init");
        sm.set_initial_state("Init");

        let result = sm.initial_state();
        assert!(
            result.is_some(),
            "initial_state should return Some when set"
        );
    }

    #[test]
    fn test_initial_state_returns_none_when_not_set() {
        let sm = StateMachine::new();
        let result = sm.initial_state();
        assert!(
            result.is_none(),
            "initial_state should return None when not set"
        );
    }

    #[test]
    fn test_is_terminal_true_for_terminal_states() {
        let mut sm = StateMachine::new();
        sm.add_state("Init");
        sm.add_state("Done");
        sm.mark_terminal("Done");

        let done_id = sm.state_id("Done").unwrap();
        assert!(
            sm.is_terminal(done_id),
            "is_terminal should return true for terminal state"
        );
    }

    #[test]
    fn test_is_terminal_false_for_non_terminal_states() {
        let mut sm = StateMachine::new();
        sm.add_state("Init");
        sm.add_state("Done");
        sm.mark_terminal("Done");

        let init_id = sm.state_id("Init").unwrap();
        assert!(
            !sm.is_terminal(init_id),
            "is_terminal should return false for non-terminal state"
        );
    }

    #[test]
    fn test_transitions_from_returns_correct_transitions() {
        let mut sm = StateMachine::new();
        sm.add_state("A");
        sm.add_state("B");
        sm.add_state("C");
        sm.add_transition("A", "B", "go_b");
        sm.add_transition("A", "C", "go_c");

        let a_id = sm.state_id("A").unwrap();
        let transitions = sm.transitions_from(a_id);

        assert_eq!(transitions.len(), 2, "Should have 2 transitions from A");
        let labels: Vec<_> = transitions.iter().map(|(l, _)| l.as_str()).collect();
        assert!(labels.contains(&"go_b"), "Should have go_b transition");
        assert!(labels.contains(&"go_c"), "Should have go_c transition");
    }

    #[test]
    fn test_transitions_from_returns_empty_for_terminal() {
        let mut sm = StateMachine::new();
        sm.add_state("Init");
        sm.add_state("Done");
        sm.set_initial_state("Init");
        sm.mark_terminal("Done");
        sm.add_transition("Init", "Done", "finish");

        let done_id = sm.state_id("Done").unwrap();
        let transitions = sm.transitions_from(done_id);

        assert!(
            transitions.is_empty(),
            "Terminal state should have no transitions"
        );
    }

    #[test]
    fn test_is_fully_reachable_true_when_all_reachable() {
        let mut sm = StateMachine::new();
        sm.add_state("A");
        sm.add_state("B");
        sm.add_state("C");
        sm.set_initial_state("A");
        sm.add_transition("A", "B", "go");
        sm.add_transition("B", "C", "go");

        assert!(sm.is_fully_reachable(), "All states should be reachable");
    }

    #[test]
    fn test_is_fully_reachable_false_when_unreachable() {
        let mut sm = StateMachine::new();
        sm.add_state("A");
        sm.add_state("B");
        sm.add_state("Isolated");
        sm.set_initial_state("A");
        sm.add_transition("A", "B", "go");
        // No transition to Isolated

        assert!(
            !sm.is_fully_reachable(),
            "Isolated state makes it not fully reachable"
        );
    }

    #[test]
    fn test_states_returns_all_state_names() {
        let mut sm = StateMachine::new();
        sm.add_state("First");
        sm.add_state("Second");
        sm.add_state("Third");

        let states = sm.states();
        assert_eq!(states.len(), 3, "Should return all 3 states");
        assert!(states.contains(&"First"), "Should contain First");
        assert!(states.contains(&"Second"), "Should contain Second");
        assert!(states.contains(&"Third"), "Should contain Third");
    }

    #[test]
    fn test_states_returns_empty_for_empty_machine() {
        let sm = StateMachine::new();
        let states = sm.states();
        assert!(
            states.is_empty(),
            "Empty machine should return empty states"
        );
    }
}

// =============================================================================
// Kani Proof Harnesses
// =============================================================================
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Prove that a new state machine starts empty
    #[kani::proof]
    fn verify_state_machine_new_is_empty() {
        let sm = StateMachine::new();
        kani::assert(sm.state_count() == 0, "New machine has no states");
        kani::assert(sm.transition_count() == 0, "New machine has no transitions");
        kani::assert(
            sm.initial_state().is_none(),
            "New machine has no initial state",
        );
    }

    /// Prove that adding the same state twice reuses the existing node id
    #[kani::proof]
    fn verify_add_state_idempotent() {
        let mut sm = StateMachine::new();
        let first = sm.add_state("Init");
        let second = sm.add_state("Init");
        kani::assert(first == second, "add_state must return stable ids");
        kani::assert(sm.state_count() == 1, "Duplicate states are not added");
    }

    /// Prove that executing without an initial state returns an error at index 0
    #[kani::proof]
    fn verify_execute_sequence_requires_initial_state() {
        let sm = StateMachine::new();
        let result = sm.execute_sequence(&["start"]);
        kani::assert(
            result.is_err(),
            "execute_sequence must fail without initial",
        );
        if let Err((idx, _)) = result {
            kani::assert(idx == 0, "Error index should be 0 when no initial state");
        } else {
            kani::assert(false, "Expected an error result");
        }
    }

    /// Prove that reachability is empty when no initial state is set
    #[kani::proof]
    fn verify_reachable_states_empty_without_initial() {
        let sm = StateMachine::new();
        let reachable = sm.reachable_states();
        kani::assert(
            reachable.is_empty(),
            "No initial state yields no reachable states",
        );
    }

    /// Prove that a single-state machine is fully reachable when initial is set
    #[kani::proof]
    fn verify_is_fully_reachable_single_state() {
        let mut sm = StateMachine::new();
        sm.add_state("Only");
        sm.set_initial_state("Only");
        kani::assert(
            sm.is_fully_reachable(),
            "Single-state machine should be fully reachable",
        );
    }
}
