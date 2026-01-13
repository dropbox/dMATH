//! Platform API definitions and state machine types

use crate::constraint::{ApiConstraint, ConstraintKind, TemporalRelation};
use crate::state_machine::StateMachine;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A state in a platform API state machine
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ApiState {
    /// Name of the state
    pub name: String,
    /// Human-readable description
    pub description: Option<String>,
    /// Whether this is an error state
    pub is_error: bool,
    /// Whether this is a terminal/final state
    pub is_terminal: bool,
}

impl ApiState {
    /// Create a new API state
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            is_error: false,
            is_terminal: false,
        }
    }

    /// Set the description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Mark as error state
    pub fn as_error(mut self) -> Self {
        self.is_error = true;
        self
    }

    /// Mark as terminal state
    pub fn as_terminal(mut self) -> Self {
        self.is_terminal = true;
        self
    }
}

/// A transition between states triggered by an API call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition {
    /// Name of the API method/function
    pub method: String,
    /// States from which this transition is valid
    pub from_states: Vec<String>,
    /// Target state after the transition
    pub to_state: String,
    /// Human-readable description
    pub description: Option<String>,
    /// Preconditions that must hold
    pub preconditions: Vec<String>,
    /// Postconditions that will hold after
    pub postconditions: Vec<String>,
}

impl StateTransition {
    /// Create a new state transition
    pub fn new(
        method: impl Into<String>,
        from_states: Vec<impl Into<String>>,
        to_state: impl Into<String>,
    ) -> Self {
        Self {
            method: method.into(),
            from_states: from_states.into_iter().map(Into::into).collect(),
            to_state: to_state.into(),
            description: None,
            preconditions: Vec::new(),
            postconditions: Vec::new(),
        }
    }

    /// Set description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add a precondition
    pub fn with_precondition(mut self, pre: impl Into<String>) -> Self {
        self.preconditions.push(pre.into());
        self
    }

    /// Add a postcondition
    pub fn with_postcondition(mut self, post: impl Into<String>) -> Self {
        self.postconditions.push(post.into());
        self
    }
}

/// A complete platform API specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformApi {
    /// Name of the platform (e.g., "Metal", "CUDA", "Vulkan")
    pub platform: String,
    /// Name of the API object (e.g., "MTLCommandBuffer", "cudaStream_t")
    pub api_object: String,
    /// Human-readable description
    pub description: Option<String>,
    /// States in this API
    pub states: HashMap<String, ApiState>,
    /// Transitions between states
    pub transitions: Vec<StateTransition>,
    /// Constraints that must be satisfied
    pub constraints: Vec<ApiConstraint>,
    /// The initial state name
    pub initial_state: Option<String>,
    /// Cached state machine (built on demand)
    #[serde(skip)]
    state_machine: Option<StateMachine>,
}

impl PlatformApi {
    /// Create a new platform API specification
    pub fn new(platform: impl Into<String>, api_object: impl Into<String>) -> Self {
        Self {
            platform: platform.into(),
            api_object: api_object.into(),
            description: None,
            states: HashMap::new(),
            transitions: Vec::new(),
            constraints: Vec::new(),
            initial_state: None,
            state_machine: None,
        }
    }

    /// Set description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add a state to the API
    pub fn add_state(&mut self, state: ApiState) {
        self.states.insert(state.name.clone(), state);
        self.state_machine = None; // Invalidate cache
    }

    /// Set the initial state
    pub fn set_initial_state(&mut self, state: impl Into<String>) {
        self.initial_state = Some(state.into());
        self.state_machine = None;
    }

    /// Add a transition to the API
    pub fn add_transition(&mut self, transition: StateTransition) {
        self.transitions.push(transition);
        self.state_machine = None;
    }

    /// Add a constraint
    pub fn add_constraint(&mut self, constraint: ApiConstraint) {
        self.constraints.push(constraint);
    }

    /// Add a "must call before" constraint
    pub fn must_call_before(&mut self, first: &str, second: &str, message: &str) {
        self.add_constraint(ApiConstraint {
            kind: ConstraintKind::Temporal(TemporalRelation::Before),
            method_a: first.to_string(),
            method_b: Some(second.to_string()),
            message: message.to_string(),
            severity: crate::constraint::Severity::Critical,
        });
    }

    /// Add a "must call after" constraint
    pub fn must_call_after(&mut self, first: &str, second: &str, message: &str) {
        self.add_constraint(ApiConstraint {
            kind: ConstraintKind::Temporal(TemporalRelation::After),
            method_a: first.to_string(),
            method_b: Some(second.to_string()),
            message: message.to_string(),
            severity: crate::constraint::Severity::Critical,
        });
    }

    /// Add a "never call" constraint (method is forbidden in certain states)
    pub fn never_call_from(&mut self, method: &str, states: &[&str], message: &str) {
        for state in states {
            self.add_constraint(ApiConstraint {
                kind: ConstraintKind::Forbidden {
                    state: state.to_string(),
                },
                method_a: method.to_string(),
                method_b: None,
                message: message.to_string(),
                severity: crate::constraint::Severity::Critical,
            });
        }
    }

    /// Build and cache the state machine
    pub fn state_machine(&mut self) -> &StateMachine {
        if self.state_machine.is_some() {
            return self.state_machine.as_ref().unwrap();
        }

        let mut sm = StateMachine::new();

        // Add all states
        for state in self.states.values() {
            sm.add_state(&state.name);
            if state.is_terminal {
                sm.mark_terminal(&state.name);
            }
        }

        // Set initial state
        if let Some(ref initial) = self.initial_state {
            sm.set_initial_state(initial);
        }

        // Add all transitions
        for trans in &self.transitions {
            for from in &trans.from_states {
                sm.add_transition(from, &trans.to_state, &trans.method);
            }
        }

        self.state_machine = Some(sm);
        self.state_machine.as_ref().unwrap()
    }

    /// Validate the API specification
    pub fn validate(&mut self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // Check initial state exists
        if let Some(ref initial) = self.initial_state {
            if !self.states.contains_key(initial) {
                errors.push(format!("Initial state '{}' not defined", initial));
            }
        } else {
            errors.push("No initial state defined".to_string());
        }

        // Check all transition states exist
        for trans in &self.transitions {
            for from in &trans.from_states {
                if !self.states.contains_key(from) {
                    errors.push(format!(
                        "Transition '{}' references undefined source state '{}'",
                        trans.method, from
                    ));
                }
            }
            if !self.states.contains_key(&trans.to_state) {
                errors.push(format!(
                    "Transition '{}' references undefined target state '{}'",
                    trans.method, trans.to_state
                ));
            }
        }

        // Check state machine reachability (excluding terminal states which may be
        // reached through external events like resource deallocation)
        // Build set of terminal state names first (owned to avoid borrow issues)
        let terminal_states: std::collections::HashSet<String> = self
            .states
            .values()
            .filter(|s| s.is_terminal)
            .map(|s| s.name.clone())
            .collect();

        let sm = self.state_machine();
        let reachable: Vec<_> = sm.reachable_states();
        let all_states: Vec<_> = sm.states();
        let unreachable: Vec<_> = all_states
            .iter()
            .filter(|s| {
                // Skip terminal states - they may be reached externally
                let s_str: &str = s;
                if terminal_states.contains(s_str) {
                    return false;
                }
                sm.state_id(s)
                    .map(|id| !reachable.contains(&id))
                    .unwrap_or(true)
            })
            .collect();
        if !unreachable.is_empty() {
            errors.push(format!(
                "Unreachable non-terminal states: {:?}",
                unreachable
            ));
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Get all states
    pub fn get_states(&self) -> impl Iterator<Item = &ApiState> {
        self.states.values()
    }

    /// Get all transitions
    pub fn get_transitions(&self) -> impl Iterator<Item = &StateTransition> {
        self.transitions.iter()
    }

    /// Get all constraints
    pub fn get_constraints(&self) -> impl Iterator<Item = &ApiConstraint> {
        self.constraints.iter()
    }

    /// Get the full API name (platform::object)
    pub fn full_name(&self) -> String {
        format!("{}::{}", self.platform, self.api_object)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_api() {
        let mut api = PlatformApi::new("Metal", "MTLCommandBuffer");
        api.add_state(ApiState::new("Created"));
        api.add_state(ApiState::new("Committed").as_terminal());
        api.set_initial_state("Created");
        api.add_transition(StateTransition::new("commit", vec!["Created"], "Committed"));

        assert_eq!(api.full_name(), "Metal::MTLCommandBuffer");
        assert!(api.validate().is_ok());
    }

    #[test]
    fn test_must_call_after_adds_constraint() {
        let mut api = PlatformApi::new("Test", "Object");
        api.add_state(ApiState::new("Init"));
        api.add_state(ApiState::new("Done").as_terminal());
        api.set_initial_state("Init");

        // Before adding must_call_after
        let count_before = api.constraints.len();

        api.must_call_after("methodA", "methodB", "A must be called after B");

        // After adding must_call_after, should have one more constraint
        assert_eq!(api.constraints.len(), count_before + 1);

        // Verify the constraint has correct values
        let constraint = api.constraints.last().unwrap();
        assert_eq!(constraint.method_a, "methodA");
        assert_eq!(constraint.method_b, Some("methodB".to_string()));
        assert!(matches!(
            constraint.kind,
            ConstraintKind::Temporal(TemporalRelation::After)
        ));
    }

    #[test]
    fn test_never_call_from_adds_constraints_for_each_state() {
        let mut api = PlatformApi::new("Test", "Object");
        api.add_state(ApiState::new("Init"));
        api.add_state(ApiState::new("Running"));
        api.add_state(ApiState::new("Done").as_terminal());
        api.set_initial_state("Init");

        // Before adding never_call_from
        let count_before = api.constraints.len();

        // Add for two states
        api.never_call_from(
            "dangerousMethod",
            &["Running", "Done"],
            "Cannot call from these",
        );

        // Should add 2 constraints (one per state)
        assert_eq!(api.constraints.len(), count_before + 2);

        // Verify both constraints are present
        let running_constraint = api.constraints.iter().any(|c| {
            matches!(&c.kind, ConstraintKind::Forbidden { state } if state == "Running")
                && c.method_a == "dangerousMethod"
        });
        let done_constraint = api.constraints.iter().any(|c| {
            matches!(&c.kind, ConstraintKind::Forbidden { state } if state == "Done")
                && c.method_a == "dangerousMethod"
        });
        assert!(
            running_constraint,
            "Should have constraint for Running state"
        );
        assert!(done_constraint, "Should have constraint for Done state");
    }

    #[test]
    fn test_get_states_returns_all_states() {
        let mut api = PlatformApi::new("Test", "Object");
        api.add_state(ApiState::new("State1"));
        api.add_state(ApiState::new("State2"));
        api.add_state(ApiState::new("State3").as_terminal());
        api.set_initial_state("State1");

        let states: Vec<_> = api.get_states().collect();
        assert_eq!(states.len(), 3);

        let names: Vec<_> = states.iter().map(|s| &s.name).collect();
        assert!(names.contains(&&"State1".to_string()));
        assert!(names.contains(&&"State2".to_string()));
        assert!(names.contains(&&"State3".to_string()));
    }

    #[test]
    fn test_get_states_not_empty_iterator() {
        let mut api = PlatformApi::new("Test", "Object");
        api.add_state(ApiState::new("OnlyState"));
        api.set_initial_state("OnlyState");

        // Verify it returns non-empty iterator
        let states: Vec<_> = api.get_states().collect();
        assert!(
            !states.is_empty(),
            "get_states() should return non-empty iterator"
        );
        assert_eq!(states[0].name, "OnlyState");
    }

    #[test]
    fn test_invalid_api_missing_initial() {
        let mut api = PlatformApi::new("Test", "Object");
        api.add_state(ApiState::new("State1"));
        // No initial state set

        let result = api.validate();
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| e.contains("initial state")));
    }

    #[test]
    fn test_invalid_api_undefined_state() {
        let mut api = PlatformApi::new("Test", "Object");
        api.add_state(ApiState::new("State1"));
        api.set_initial_state("State1");
        // Transition to undefined state
        api.add_transition(StateTransition::new("go", vec!["State1"], "Undefined"));

        let result = api.validate();
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| e.contains("undefined target state")));
    }

    #[test]
    fn test_state_machine_execution() {
        let mut api = PlatformApi::new("Metal", "MTLCommandBuffer");
        api.add_state(ApiState::new("Created"));
        api.add_state(ApiState::new("Encoding"));
        api.add_state(ApiState::new("Committed").as_terminal());
        api.set_initial_state("Created");
        api.add_transition(StateTransition::new(
            "beginEncoding",
            vec!["Created"],
            "Encoding",
        ));
        api.add_transition(StateTransition::new(
            "commit",
            vec!["Created", "Encoding"],
            "Committed",
        ));

        let sm = api.state_machine();
        let result = sm.execute_sequence(&["beginEncoding", "commit"]);
        assert!(result.is_ok());
    }
}
