//! State machine model types for model-based testing
//!
//! This module defines the core types for representing state machine models
//! that can be explored for test generation.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};

use crate::error::{MbtError, MbtResult};

/// A value in the model state space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelValue {
    /// Boolean value
    Bool(bool),
    /// Integer value
    Int(i64),
    /// String value
    String(String),
    /// Set of values
    Set(Vec<ModelValue>),
    /// Sequence/list of values
    Sequence(Vec<ModelValue>),
    /// Record with named fields
    Record(IndexMap<String, ModelValue>),
    /// Function mapping
    Function(Vec<(ModelValue, ModelValue)>),
    /// Null/undefined value
    Null,
}

impl PartialEq for ModelValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ModelValue::Bool(a), ModelValue::Bool(b)) => a == b,
            (ModelValue::Int(a), ModelValue::Int(b)) => a == b,
            (ModelValue::String(a), ModelValue::String(b)) => a == b,
            (ModelValue::Sequence(a), ModelValue::Sequence(b)) => a == b,
            (ModelValue::Record(a), ModelValue::Record(b)) => a == b,
            (ModelValue::Null, ModelValue::Null) => true,
            // Sets: compare as sorted for equality
            (ModelValue::Set(a), ModelValue::Set(b)) => {
                if a.len() != b.len() {
                    return false;
                }
                let mut a_sorted: Vec<_> = a.iter().map(|v| v.canonical_string()).collect();
                let mut b_sorted: Vec<_> = b.iter().map(|v| v.canonical_string()).collect();
                a_sorted.sort();
                b_sorted.sort();
                a_sorted == b_sorted
            }
            (ModelValue::Function(a), ModelValue::Function(b)) => {
                if a.len() != b.len() {
                    return false;
                }
                // Compare sorted by key
                let mut a_sorted: Vec<_> = a.iter().collect();
                let mut b_sorted: Vec<_> = b.iter().collect();
                a_sorted.sort_by(|x, y| x.0.canonical_string().cmp(&y.0.canonical_string()));
                b_sorted.sort_by(|x, y| x.0.canonical_string().cmp(&y.0.canonical_string()));
                a_sorted
                    .iter()
                    .zip(b_sorted.iter())
                    .all(|(x, y)| x.0 == y.0 && x.1 == y.1)
            }
            _ => false,
        }
    }
}

impl Eq for ModelValue {}

impl Hash for ModelValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            ModelValue::Bool(b) => b.hash(state),
            ModelValue::Int(i) => i.hash(state),
            ModelValue::String(s) => s.hash(state),
            ModelValue::Sequence(v) => {
                for item in v {
                    item.hash(state);
                }
            }
            ModelValue::Set(v) => {
                // Hash sorted elements for consistent hashing
                let mut sorted: Vec<_> = v.iter().map(|x| x.canonical_string()).collect();
                sorted.sort();
                for s in sorted {
                    s.hash(state);
                }
            }
            ModelValue::Record(r) => {
                // Hash sorted fields
                let mut keys: Vec<_> = r.keys().collect();
                keys.sort();
                for k in keys {
                    k.hash(state);
                    r.get(k).unwrap().hash(state);
                }
            }
            ModelValue::Function(f) => {
                let mut sorted: Vec<_> = f.iter().collect();
                sorted.sort_by(|a, b| a.0.canonical_string().cmp(&b.0.canonical_string()));
                for (k, v) in sorted {
                    k.hash(state);
                    v.hash(state);
                }
            }
            ModelValue::Null => {}
        }
    }
}

impl ModelValue {
    /// Get a canonical string representation for comparison/hashing
    #[must_use]
    pub fn canonical_string(&self) -> String {
        match self {
            ModelValue::Bool(b) => b.to_string(),
            ModelValue::Int(i) => i.to_string(),
            ModelValue::String(s) => format!("\"{s}\""),
            ModelValue::Sequence(v) => {
                let items: Vec<_> = v.iter().map(|x| x.canonical_string()).collect();
                format!("<<{}>>", items.join(", "))
            }
            ModelValue::Set(v) => {
                let mut items: Vec<_> = v.iter().map(|x| x.canonical_string()).collect();
                items.sort();
                format!("{{{}}}", items.join(", "))
            }
            ModelValue::Record(r) => {
                let mut items: Vec<_> = r
                    .iter()
                    .map(|(k, v)| format!("{k} |-> {}", v.canonical_string()))
                    .collect();
                items.sort();
                format!("[{}]", items.join(", "))
            }
            ModelValue::Function(f) => {
                let mut items: Vec<_> = f
                    .iter()
                    .map(|(k, v)| format!("{} :> {}", k.canonical_string(), v.canonical_string()))
                    .collect();
                items.sort();
                format!("({})", items.join(" @@ "))
            }
            ModelValue::Null => "NULL".to_string(),
        }
    }

    /// Check if this value is a boolean
    #[must_use]
    pub fn is_bool(&self) -> bool {
        matches!(self, ModelValue::Bool(_))
    }

    /// Check if this value is an integer
    #[must_use]
    pub fn is_int(&self) -> bool {
        matches!(self, ModelValue::Int(_))
    }

    /// Get as boolean
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ModelValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Get as integer
    #[must_use]
    pub fn as_int(&self) -> Option<i64> {
        match self {
            ModelValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Get as string
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            ModelValue::String(s) => Some(s),
            _ => None,
        }
    }
}

impl fmt::Display for ModelValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.canonical_string())
    }
}

/// A state in the model, mapping variable names to values
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelState {
    /// Variable assignments
    pub variables: IndexMap<String, ModelValue>,
}

impl Hash for ModelState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.canonical_string().hash(state);
    }
}

impl ModelState {
    /// Create a new empty state
    #[must_use]
    pub fn new() -> Self {
        Self {
            variables: IndexMap::new(),
        }
    }

    /// Create from a map of variables
    #[must_use]
    pub fn from_variables(variables: IndexMap<String, ModelValue>) -> Self {
        Self { variables }
    }

    /// Get a variable value
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&ModelValue> {
        self.variables.get(name)
    }

    /// Set a variable value
    pub fn set(&mut self, name: impl Into<String>, value: ModelValue) {
        self.variables.insert(name.into(), value);
    }

    /// Check if a variable exists
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.variables.contains_key(name)
    }

    /// Get all variable names
    pub fn variable_names(&self) -> impl Iterator<Item = &str> {
        self.variables.keys().map(|s| s.as_str())
    }

    /// Compute diff from another state
    #[must_use]
    pub fn diff(&self, other: &ModelState) -> HashMap<String, (Option<ModelValue>, ModelValue)> {
        let mut changes = HashMap::new();

        for (name, new_value) in &self.variables {
            match other.variables.get(name) {
                Some(old_value) if old_value != new_value => {
                    changes.insert(name.clone(), (Some(old_value.clone()), new_value.clone()));
                }
                None => {
                    changes.insert(name.clone(), (None, new_value.clone()));
                }
                _ => {}
            }
        }

        changes
    }

    /// Get a canonical string for this state (for hashing/comparison)
    #[must_use]
    pub fn canonical_string(&self) -> String {
        let mut items: Vec<_> = self
            .variables
            .iter()
            .map(|(k, v)| format!("{k}={}", v.canonical_string()))
            .collect();
        items.sort();
        items.join(";")
    }
}

impl Default for ModelState {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ModelState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "State {{")?;
        for (name, value) in &self.variables {
            writeln!(f, "  {name} = {value}")?;
        }
        write!(f, "}}")
    }
}

/// An action/transition in the model
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelAction {
    /// Action name
    pub name: String,
    /// Action parameters (if any)
    pub parameters: Vec<ModelValue>,
}

impl ModelAction {
    /// Create a new action without parameters
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            parameters: Vec::new(),
        }
    }

    /// Create a new action with parameters
    #[must_use]
    pub fn with_params(name: impl Into<String>, parameters: Vec<ModelValue>) -> Self {
        Self {
            name: name.into(),
            parameters,
        }
    }

    /// Get action signature (name with parameter types)
    #[must_use]
    pub fn signature(&self) -> String {
        if self.parameters.is_empty() {
            self.name.clone()
        } else {
            let params: Vec<_> = self
                .parameters
                .iter()
                .map(|p| p.canonical_string())
                .collect();
            format!("{}({})", self.name, params.join(", "))
        }
    }
}

impl fmt::Display for ModelAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.signature())
    }
}

/// A transition in the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTransition {
    /// Source state
    pub from: ModelState,
    /// Action that triggers the transition
    pub action: ModelAction,
    /// Target state
    pub to: ModelState,
}

impl ModelTransition {
    /// Create a new transition
    #[must_use]
    pub fn new(from: ModelState, action: ModelAction, to: ModelState) -> Self {
        Self { from, action, to }
    }
}

impl fmt::Display for ModelTransition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} --[{}]--> {}",
            self.from.canonical_string(),
            self.action,
            self.to.canonical_string()
        )
    }
}

/// Variable domain specification for test generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VariableDomain {
    /// Boolean domain
    Boolean,
    /// Integer range
    IntRange { min: i64, max: i64 },
    /// Enumeration of specific values
    Enumeration(Vec<ModelValue>),
    /// Set of values from another domain
    SetOf(Box<VariableDomain>),
    /// Sequence with bounded length
    SequenceOf {
        element: Box<VariableDomain>,
        max_length: usize,
    },
    /// Record with field domains
    RecordOf(IndexMap<String, VariableDomain>),
}

impl VariableDomain {
    /// Get the minimum value for this domain (for boundary testing)
    #[must_use]
    pub fn min_value(&self) -> Option<ModelValue> {
        match self {
            VariableDomain::Boolean => Some(ModelValue::Bool(false)),
            VariableDomain::IntRange { min, .. } => Some(ModelValue::Int(*min)),
            VariableDomain::Enumeration(values) => values.first().cloned(),
            VariableDomain::SetOf(_) => Some(ModelValue::Set(Vec::new())),
            VariableDomain::SequenceOf { .. } => Some(ModelValue::Sequence(Vec::new())),
            VariableDomain::RecordOf(fields) => {
                let mut record = IndexMap::new();
                for (name, domain) in fields {
                    if let Some(val) = domain.min_value() {
                        record.insert(name.clone(), val);
                    }
                }
                Some(ModelValue::Record(record))
            }
        }
    }

    /// Get the maximum value for this domain (for boundary testing)
    #[must_use]
    pub fn max_value(&self) -> Option<ModelValue> {
        match self {
            VariableDomain::Boolean => Some(ModelValue::Bool(true)),
            VariableDomain::IntRange { max, .. } => Some(ModelValue::Int(*max)),
            VariableDomain::Enumeration(values) => values.last().cloned(),
            VariableDomain::SetOf(inner) => {
                // Max is a set with one element at max value
                inner.max_value().map(|v| ModelValue::Set(vec![v]))
            }
            VariableDomain::SequenceOf {
                element,
                max_length,
            } => element
                .max_value()
                .map(|v| ModelValue::Sequence(vec![v; *max_length])),
            VariableDomain::RecordOf(fields) => {
                let mut record = IndexMap::new();
                for (name, domain) in fields {
                    if let Some(val) = domain.max_value() {
                        record.insert(name.clone(), val);
                    }
                }
                Some(ModelValue::Record(record))
            }
        }
    }

    /// Get boundary values for this domain
    #[must_use]
    pub fn boundary_values(&self) -> Vec<ModelValue> {
        let mut values = Vec::new();

        match self {
            VariableDomain::Boolean => {
                values.push(ModelValue::Bool(false));
                values.push(ModelValue::Bool(true));
            }
            VariableDomain::IntRange { min, max } => {
                values.push(ModelValue::Int(*min));
                if *min != *max {
                    values.push(ModelValue::Int(*max));
                }
                // Adjacent to boundaries
                if *min < i64::MAX {
                    values.push(ModelValue::Int(*min + 1));
                }
                if *max > i64::MIN {
                    values.push(ModelValue::Int(*max - 1));
                }
                // Zero if in range
                if *min <= 0 && 0 <= *max && *min != 0 && *max != 0 {
                    values.push(ModelValue::Int(0));
                }
            }
            VariableDomain::Enumeration(vals) => {
                if let Some(first) = vals.first() {
                    values.push(first.clone());
                }
                if vals.len() > 1 {
                    if let Some(last) = vals.last() {
                        values.push(last.clone());
                    }
                }
            }
            VariableDomain::SetOf(inner) => {
                values.push(ModelValue::Set(Vec::new())); // Empty set
                for bv in inner.boundary_values() {
                    values.push(ModelValue::Set(vec![bv])); // Singleton
                }
            }
            VariableDomain::SequenceOf {
                element,
                max_length,
            } => {
                values.push(ModelValue::Sequence(Vec::new())); // Empty
                for bv in element.boundary_values() {
                    values.push(ModelValue::Sequence(vec![bv.clone()])); // Single element
                    if *max_length > 1 {
                        values.push(ModelValue::Sequence(vec![bv; *max_length]));
                        // Full length
                    }
                }
            }
            VariableDomain::RecordOf(_fields) => {
                // Generate boundary combinations
                if let (Some(min), Some(max)) = (self.min_value(), self.max_value()) {
                    values.push(min);
                    values.push(max);
                }
            }
        }

        values
    }

    /// Generate all values in domain (for small domains)
    pub fn all_values(&self) -> MbtResult<Vec<ModelValue>> {
        match self {
            VariableDomain::Boolean => Ok(vec![ModelValue::Bool(false), ModelValue::Bool(true)]),
            VariableDomain::IntRange { min, max } => {
                let count = (*max - *min + 1) as usize;
                if count > 1000 {
                    return Err(MbtError::InvalidModel(format!(
                        "Integer range too large: {count} values"
                    )));
                }
                Ok((*min..=*max).map(ModelValue::Int).collect())
            }
            VariableDomain::Enumeration(values) => Ok(values.clone()),
            _ => Err(MbtError::InvalidModel(
                "Cannot enumerate all values for complex domain".into(),
            )),
        }
    }
}

/// A state machine model for test generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateMachineModel {
    /// Model name
    pub name: String,
    /// Variable domains
    pub variables: IndexMap<String, VariableDomain>,
    /// Initial states
    pub initial_states: Vec<ModelState>,
    /// Actions with their enabled predicates and effects
    pub actions: Vec<ActionSpec>,
    /// Invariants (predicates that must hold in all states)
    pub invariants: Vec<Invariant>,
}

/// Specification of an action in the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionSpec {
    /// Action name
    pub name: String,
    /// Parameters with their domains
    pub parameters: Vec<(String, VariableDomain)>,
    /// Whether this action is enabled (evaluated at runtime)
    /// This is a simplified representation - real evaluation happens in the explorer
    pub enabled_description: String,
    /// Effect description
    pub effect_description: String,
}

impl ActionSpec {
    /// Create a new action spec
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            parameters: Vec::new(),
            enabled_description: "true".into(),
            effect_description: String::new(),
        }
    }

    /// Add a parameter
    #[must_use]
    pub fn with_param(mut self, name: impl Into<String>, domain: VariableDomain) -> Self {
        self.parameters.push((name.into(), domain));
        self
    }

    /// Set enabled predicate description
    #[must_use]
    pub fn with_enabled(mut self, description: impl Into<String>) -> Self {
        self.enabled_description = description.into();
        self
    }

    /// Set effect description
    #[must_use]
    pub fn with_effect(mut self, description: impl Into<String>) -> Self {
        self.effect_description = description.into();
        self
    }
}

/// An invariant that must hold in all reachable states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Invariant {
    /// Invariant name
    pub name: String,
    /// Description of the invariant
    pub description: String,
}

impl Invariant {
    /// Create a new invariant
    #[must_use]
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
        }
    }
}

impl StateMachineModel {
    /// Create a new empty model
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            variables: IndexMap::new(),
            initial_states: Vec::new(),
            actions: Vec::new(),
            invariants: Vec::new(),
        }
    }

    /// Add a variable with its domain
    #[must_use]
    pub fn with_variable(mut self, name: impl Into<String>, domain: VariableDomain) -> Self {
        self.variables.insert(name.into(), domain);
        self
    }

    /// Add an initial state
    #[must_use]
    pub fn with_initial_state(mut self, state: ModelState) -> Self {
        self.initial_states.push(state);
        self
    }

    /// Add an action specification
    #[must_use]
    pub fn with_action(mut self, action: ActionSpec) -> Self {
        self.actions.push(action);
        self
    }

    /// Add an invariant
    #[must_use]
    pub fn with_invariant(mut self, invariant: Invariant) -> Self {
        self.invariants.push(invariant);
        self
    }

    /// Get action names
    pub fn action_names(&self) -> impl Iterator<Item = &str> {
        self.actions.iter().map(|a| a.name.as_str())
    }

    /// Validate the model structure
    pub fn validate(&self) -> MbtResult<()> {
        if self.initial_states.is_empty() {
            return Err(MbtError::NoInitialState);
        }

        // Check that initial states have all required variables
        for state in &self.initial_states {
            for var_name in self.variables.keys() {
                if !state.contains(var_name) {
                    return Err(MbtError::InvalidModel(format!(
                        "Initial state missing variable '{var_name}'"
                    )));
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_value_equality() {
        let set1 = ModelValue::Set(vec![ModelValue::Int(1), ModelValue::Int(2)]);
        let set2 = ModelValue::Set(vec![ModelValue::Int(2), ModelValue::Int(1)]);
        assert_eq!(set1, set2);

        let seq1 = ModelValue::Sequence(vec![ModelValue::Int(1), ModelValue::Int(2)]);
        let seq2 = ModelValue::Sequence(vec![ModelValue::Int(2), ModelValue::Int(1)]);
        assert_ne!(seq1, seq2);
    }

    #[test]
    fn test_model_state_diff() {
        let mut state1 = ModelState::new();
        state1.set("x", ModelValue::Int(1));
        state1.set("y", ModelValue::Int(2));

        let mut state2 = ModelState::new();
        state2.set("x", ModelValue::Int(1));
        state2.set("y", ModelValue::Int(3));
        state2.set("z", ModelValue::Int(4));

        let diff = state2.diff(&state1);
        assert_eq!(diff.len(), 2);
        assert!(diff.contains_key("y"));
        assert!(diff.contains_key("z"));
    }

    #[test]
    fn test_boundary_values() {
        let domain = VariableDomain::IntRange { min: 0, max: 10 };
        let boundaries = domain.boundary_values();
        assert!(boundaries.contains(&ModelValue::Int(0)));
        assert!(boundaries.contains(&ModelValue::Int(10)));
        assert!(boundaries.contains(&ModelValue::Int(1)));
        assert!(boundaries.contains(&ModelValue::Int(9)));
    }

    #[test]
    fn test_model_validation() {
        let model = StateMachineModel::new("test")
            .with_variable("x", VariableDomain::IntRange { min: 0, max: 10 });

        assert!(model.validate().is_err()); // No initial state

        let mut initial = ModelState::new();
        initial.set("x", ModelValue::Int(0));
        let model = model.with_initial_state(initial);
        assert!(model.validate().is_ok());
    }
}
