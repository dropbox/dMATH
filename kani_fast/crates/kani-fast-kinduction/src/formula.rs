//! Formula representation for k-induction
//!
//! This module defines the transition system and property representations
//! used in k-induction verification.

use once_cell::sync::OnceCell;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::fmt::Write as _;

/// A transition system for k-induction verification
///
/// A transition system consists of:
/// - State variables with their types
/// - Initial state formula (I)
/// - Transition relation (T)
/// - Property to verify (P)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionSystem {
    /// State variables with their SMT-LIB2 types
    pub variables: Vec<StateVariable>,

    /// Initial state formula in SMT-LIB2 format
    /// Uses unprimed variable names (e.g., x, y, z)
    pub init: StateFormula,

    /// Transition relation in SMT-LIB2 format
    /// Uses primed names for next-state variables (e.g., x', y', z')
    pub transition: StateFormula,

    /// Properties to verify
    pub properties: Vec<Property>,

    /// Optional auxiliary invariants to strengthen induction
    pub invariants: Vec<StateFormula>,
}

impl TransitionSystem {
    /// Create a new empty transition system
    pub fn new() -> Self {
        Self {
            variables: Vec::new(),
            init: StateFormula::true_formula(),
            transition: StateFormula::true_formula(),
            properties: Vec::new(),
            invariants: Vec::new(),
        }
    }

    /// Add a state variable
    pub fn add_variable(&mut self, name: impl Into<String>, smt_type: SmtType) {
        self.variables.push(StateVariable::new(name, smt_type));
    }

    /// Set the initial state formula
    pub fn set_init(&mut self, formula: StateFormula) {
        self.init = formula;
    }

    /// Set the transition relation
    pub fn set_transition(&mut self, formula: StateFormula) {
        self.transition = formula;
    }

    /// Add a property to verify
    pub fn add_property(&mut self, property: Property) {
        self.properties.push(property);
    }

    /// Add an auxiliary invariant
    pub fn add_invariant(&mut self, invariant: StateFormula) {
        self.invariants.push(invariant);
    }

    /// Generate SMT-LIB2 declarations for step i
    pub fn declare_step(&self, step: u32) -> String {
        let mut decls = String::new();
        for var in &self.variables {
            let type_str = var.smt_type.to_smt_string();
            let _ = writeln!(decls, "(declare-const {}_{step} {type_str})", var.name);
        }
        decls
    }

    /// Instantiate a formula at a given step
    /// Replaces variable references with step-indexed versions
    pub fn instantiate(&self, formula: &StateFormula, step: u32) -> String {
        let mut result = formula.smt_formula.clone();
        for var in &self.variables {
            // Replace unprimed variable with step-indexed version
            let replacement = format!("{}_{}", var.name, step);
            result = var
                .unprimed_regex()
                .replace_all(&result, replacement.as_str())
                .to_string();
        }
        result
    }

    /// Instantiate a transition (relates step to step+1)
    pub fn instantiate_transition(&self, step: u32) -> String {
        let mut result = self.transition.smt_formula.clone();
        for var in &self.variables {
            // IMPORTANT: Replace primed variables FIRST, before unprimed
            // This avoids the issue of unprimed replacement affecting primed vars

            // Replace primed variable (next state) with step+1 indexed version
            // Pattern: match var name followed by apostrophe, with word boundary before
            // Note: ' is not a word char, so \b only needed before the name
            let primed_replacement = format!("{}_{}", var.name, step + 1);
            result = var
                .primed_regex()
                .replace_all(&result, primed_replacement.as_str())
                .to_string();
        }

        // Second pass: replace unprimed variables
        for var in &self.variables {
            // Replace unprimed variable (current state) with step indexed version
            // Now that primed vars are replaced, we can safely match unprimed
            let unprimed_replacement = format!("{}_{}", var.name, step);
            result = var
                .unprimed_regex()
                .replace_all(&result, unprimed_replacement.as_str())
                .to_string();
        }
        result
    }
}

impl Default for TransitionSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// A state variable in the transition system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateVariable {
    /// Variable name
    pub name: String,
    /// SMT type
    pub smt_type: SmtType,
    /// Cached regex for unprimed occurrences (not serialized)
    #[serde(skip, default = "StateVariable::empty_regex_cell")]
    unprimed_regex: OnceCell<Regex>,
    /// Cached regex for primed occurrences (not serialized)
    #[serde(skip, default = "StateVariable::empty_regex_cell")]
    primed_regex: OnceCell<Regex>,
}

impl StateVariable {
    pub fn new(name: impl Into<String>, smt_type: SmtType) -> Self {
        Self {
            name: name.into(),
            smt_type,
            unprimed_regex: OnceCell::new(),
            primed_regex: OnceCell::new(),
        }
    }

    fn unprimed_regex(&self) -> &Regex {
        self.unprimed_regex.get_or_init(|| {
            Regex::new(&format!(r"\b{}\b", regex::escape(&self.name))).expect("valid regex")
        })
    }

    fn primed_regex(&self) -> &Regex {
        self.primed_regex.get_or_init(|| {
            Regex::new(&format!(r"\b{}'", regex::escape(&self.name))).expect("valid regex")
        })
    }

    fn empty_regex_cell() -> OnceCell<Regex> {
        OnceCell::new()
    }
}

/// SMT-LIB2 types for state variables
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SmtType {
    Bool,
    Int,
    Real,
    BitVec(u32),
    Array {
        index: Box<SmtType>,
        element: Box<SmtType>,
    },
}

impl SmtType {
    /// Convert to SMT-LIB2 type string
    pub fn to_smt_string(&self) -> String {
        match self {
            SmtType::Bool => "Bool".to_string(),
            SmtType::Int => "Int".to_string(),
            SmtType::Real => "Real".to_string(),
            SmtType::BitVec(width) => format!("(_ BitVec {width})"),
            SmtType::Array { index, element } => {
                format!(
                    "(Array {} {})",
                    index.to_smt_string(),
                    element.to_smt_string()
                )
            }
        }
    }
}

/// A formula over state variables
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateFormula {
    /// The formula in SMT-LIB2 format
    pub smt_formula: String,
    /// Human-readable description
    pub description: Option<String>,
}

impl StateFormula {
    /// Create a formula from an SMT-LIB2 string
    pub fn new(smt_formula: impl Into<String>) -> Self {
        Self {
            smt_formula: smt_formula.into(),
            description: None,
        }
    }

    /// Create a formula with description
    pub fn with_description(
        smt_formula: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            smt_formula: smt_formula.into(),
            description: Some(description.into()),
        }
    }

    /// Create a "true" formula
    pub fn true_formula() -> Self {
        Self::new("true")
    }

    /// Create a "false" formula
    pub fn false_formula() -> Self {
        Self::new("false")
    }

    /// Negate this formula
    pub fn negate(&self) -> Self {
        Self {
            smt_formula: format!("(not {})", self.smt_formula),
            description: self.description.as_ref().map(|d| format!("not ({d})")),
        }
    }

    /// Conjunction with another formula
    pub fn and(&self, other: &Self) -> Self {
        Self {
            smt_formula: format!("(and {} {})", self.smt_formula, other.smt_formula),
            description: match (&self.description, &other.description) {
                (Some(a), Some(b)) => Some(format!("({a}) and ({b})")),
                _ => None,
            },
        }
    }

    /// Disjunction with another formula
    pub fn or(&self, other: &Self) -> Self {
        Self {
            smt_formula: format!("(or {} {})", self.smt_formula, other.smt_formula),
            description: match (&self.description, &other.description) {
                (Some(a), Some(b)) => Some(format!("({a}) or ({b})")),
                _ => None,
            },
        }
    }

    /// Implication
    pub fn implies(&self, other: &Self) -> Self {
        Self {
            smt_formula: format!("(=> {} {})", self.smt_formula, other.smt_formula),
            description: match (&self.description, &other.description) {
                (Some(a), Some(b)) => Some(format!("({a}) => ({b})")),
                _ => None,
            },
        }
    }
}

/// A property to verify
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Property {
    /// Unique identifier for this property
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// The property formula
    pub formula: StateFormula,
    /// Property type
    pub property_type: PropertyType,
}

impl Property {
    /// Create a safety property (invariant)
    pub fn safety(id: impl Into<String>, name: impl Into<String>, formula: StateFormula) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            formula,
            property_type: PropertyType::Safety,
        }
    }

    /// Create a reachability property
    pub fn reachability(
        id: impl Into<String>,
        name: impl Into<String>,
        formula: StateFormula,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            formula,
            property_type: PropertyType::Reachability,
        }
    }
}

/// Types of properties
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PropertyType {
    /// Safety property: should always hold (G P)
    Safety,
    /// Reachability: should eventually be reachable (F P)
    Reachability,
    /// Liveness: should hold infinitely often (GF P)
    Liveness,
}

/// Builder for creating transition systems
#[derive(Debug, Default)]
pub struct TransitionSystemBuilder {
    system: TransitionSystem,
}

impl TransitionSystemBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn variable(mut self, name: impl Into<String>, smt_type: SmtType) -> Self {
        self.system.add_variable(name, smt_type);
        self
    }

    pub fn init(mut self, formula: impl Into<String>) -> Self {
        self.system.set_init(StateFormula::new(formula));
        self
    }

    pub fn transition(mut self, formula: impl Into<String>) -> Self {
        self.system.set_transition(StateFormula::new(formula));
        self
    }

    pub fn property(
        mut self,
        id: impl Into<String>,
        name: impl Into<String>,
        formula: impl Into<String>,
    ) -> Self {
        self.system
            .add_property(Property::safety(id, name, StateFormula::new(formula)));
        self
    }

    pub fn invariant(mut self, formula: impl Into<String>) -> Self {
        self.system.add_invariant(StateFormula::new(formula));
        self
    }

    pub fn build(self) -> TransitionSystem {
        self.system
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ======== TransitionSystem tests ========

    #[test]
    fn test_transition_system_builder() {
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .variable("y", SmtType::Bool)
            .init("(and (= x 0) y)")
            .transition("(and (= x' (+ x 1)) (= y' y))")
            .property("p1", "x_positive", "(>= x 0)")
            .build();

        assert_eq!(ts.variables.len(), 2);
        assert_eq!(ts.properties.len(), 1);
        assert_eq!(ts.properties[0].id, "p1");
    }

    #[test]
    fn test_transition_system_new() {
        let ts = TransitionSystem::new();
        assert!(ts.variables.is_empty());
        assert!(ts.properties.is_empty());
        assert!(ts.invariants.is_empty());
        assert_eq!(ts.init.smt_formula, "true");
        assert_eq!(ts.transition.smt_formula, "true");
    }

    #[test]
    fn test_transition_system_default() {
        let ts = TransitionSystem::default();
        assert!(ts.variables.is_empty());
    }

    #[test]
    fn test_transition_system_add_variable() {
        let mut ts = TransitionSystem::new();
        ts.add_variable("x", SmtType::Int);
        ts.add_variable("flag", SmtType::Bool);

        assert_eq!(ts.variables.len(), 2);
        assert_eq!(ts.variables[0].name, "x");
        assert_eq!(ts.variables[0].smt_type, SmtType::Int);
        assert_eq!(ts.variables[1].name, "flag");
        assert_eq!(ts.variables[1].smt_type, SmtType::Bool);
    }

    #[test]
    fn test_transition_system_set_init() {
        let mut ts = TransitionSystem::new();
        ts.set_init(StateFormula::new("(= x 0)"));
        assert_eq!(ts.init.smt_formula, "(= x 0)");
    }

    #[test]
    fn test_transition_system_set_transition() {
        let mut ts = TransitionSystem::new();
        ts.set_transition(StateFormula::new("(= x' (+ x 1))"));
        assert_eq!(ts.transition.smt_formula, "(= x' (+ x 1))");
    }

    #[test]
    fn test_transition_system_add_property() {
        let mut ts = TransitionSystem::new();
        let prop = Property::safety("p1", "test", StateFormula::new("true"));
        ts.add_property(prop);

        assert_eq!(ts.properties.len(), 1);
        assert_eq!(ts.properties[0].id, "p1");
    }

    #[test]
    fn test_transition_system_add_invariant() {
        let mut ts = TransitionSystem::new();
        ts.add_invariant(StateFormula::new("(>= x 0)"));
        ts.add_invariant(StateFormula::new("(< y 100)"));

        assert_eq!(ts.invariants.len(), 2);
        assert_eq!(ts.invariants[0].smt_formula, "(>= x 0)");
        assert_eq!(ts.invariants[1].smt_formula, "(< y 100)");
    }

    #[test]
    fn test_transition_system_declare_step() {
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .variable("y", SmtType::Bool)
            .build();

        let decls = ts.declare_step(3);
        assert!(decls.contains("(declare-const x_3 Int)"));
        assert!(decls.contains("(declare-const y_3 Bool)"));
    }

    // ======== SmtType tests ========

    #[test]
    fn test_smt_type_string() {
        assert_eq!(SmtType::Bool.to_smt_string(), "Bool");
        assert_eq!(SmtType::Int.to_smt_string(), "Int");
        assert_eq!(SmtType::BitVec(32).to_smt_string(), "(_ BitVec 32)");
        assert_eq!(
            SmtType::Array {
                index: Box::new(SmtType::Int),
                element: Box::new(SmtType::Bool)
            }
            .to_smt_string(),
            "(Array Int Bool)"
        );
    }

    #[test]
    fn test_smt_type_real() {
        assert_eq!(SmtType::Real.to_smt_string(), "Real");
    }

    #[test]
    fn test_smt_type_bitvec_various_widths() {
        assert_eq!(SmtType::BitVec(8).to_smt_string(), "(_ BitVec 8)");
        assert_eq!(SmtType::BitVec(16).to_smt_string(), "(_ BitVec 16)");
        assert_eq!(SmtType::BitVec(64).to_smt_string(), "(_ BitVec 64)");
        assert_eq!(SmtType::BitVec(1).to_smt_string(), "(_ BitVec 1)");
    }

    #[test]
    fn test_smt_type_nested_array() {
        let nested = SmtType::Array {
            index: Box::new(SmtType::Int),
            element: Box::new(SmtType::Array {
                index: Box::new(SmtType::Int),
                element: Box::new(SmtType::Bool),
            }),
        };
        assert_eq!(nested.to_smt_string(), "(Array Int (Array Int Bool))");
    }

    #[test]
    fn test_smt_type_equality() {
        assert_eq!(SmtType::Int, SmtType::Int);
        assert_eq!(SmtType::Bool, SmtType::Bool);
        assert_eq!(SmtType::BitVec(32), SmtType::BitVec(32));
        assert_ne!(SmtType::Int, SmtType::Bool);
        assert_ne!(SmtType::BitVec(32), SmtType::BitVec(64));
    }

    #[test]
    fn test_smt_type_clone() {
        let orig = SmtType::Array {
            index: Box::new(SmtType::Int),
            element: Box::new(SmtType::Bool),
        };
        let cloned = orig.clone();
        assert_eq!(orig.to_smt_string(), cloned.to_smt_string());
    }

    #[test]
    fn test_smt_type_serialization() {
        let smt_type = SmtType::BitVec(32);
        let json = serde_json::to_string(&smt_type);
        assert!(json.is_ok());

        let deserialized: SmtType = serde_json::from_str(&json.unwrap()).unwrap();
        assert_eq!(deserialized, SmtType::BitVec(32));
    }

    // ======== StateFormula tests ========

    #[test]
    fn test_formula_operations() {
        let p = StateFormula::new("P");
        let q = StateFormula::new("Q");

        assert_eq!(p.negate().smt_formula, "(not P)");
        assert_eq!(p.and(&q).smt_formula, "(and P Q)");
        assert_eq!(p.or(&q).smt_formula, "(or P Q)");
        assert_eq!(p.implies(&q).smt_formula, "(=> P Q)");
    }

    #[test]
    fn test_formula_new() {
        let f = StateFormula::new("(>= x 0)");
        assert_eq!(f.smt_formula, "(>= x 0)");
        assert!(f.description.is_none());
    }

    #[test]
    fn test_formula_with_description() {
        let f = StateFormula::with_description("(>= x 0)", "x is non-negative");
        assert_eq!(f.smt_formula, "(>= x 0)");
        assert_eq!(f.description, Some("x is non-negative".to_string()));
    }

    #[test]
    fn test_formula_true() {
        let f = StateFormula::true_formula();
        assert_eq!(f.smt_formula, "true");
    }

    #[test]
    fn test_formula_false() {
        let f = StateFormula::false_formula();
        assert_eq!(f.smt_formula, "false");
    }

    #[test]
    fn test_formula_negate_with_description() {
        let f = StateFormula::with_description("P", "property P");
        let negated = f.negate();
        assert_eq!(negated.smt_formula, "(not P)");
        assert_eq!(negated.description, Some("not (property P)".to_string()));
    }

    #[test]
    fn test_formula_and_with_description() {
        let p = StateFormula::with_description("P", "prop P");
        let q = StateFormula::with_description("Q", "prop Q");
        let conj = p.and(&q);
        assert_eq!(conj.smt_formula, "(and P Q)");
        assert_eq!(conj.description, Some("(prop P) and (prop Q)".to_string()));
    }

    #[test]
    fn test_formula_or_with_description() {
        let p = StateFormula::with_description("P", "prop P");
        let q = StateFormula::with_description("Q", "prop Q");
        let disj = p.or(&q);
        assert_eq!(disj.smt_formula, "(or P Q)");
        assert_eq!(disj.description, Some("(prop P) or (prop Q)".to_string()));
    }

    #[test]
    fn test_formula_implies_with_description() {
        let p = StateFormula::with_description("P", "antecedent");
        let q = StateFormula::with_description("Q", "consequent");
        let impl_f = p.implies(&q);
        assert_eq!(impl_f.smt_formula, "(=> P Q)");
        assert_eq!(
            impl_f.description,
            Some("(antecedent) => (consequent)".to_string())
        );
    }

    #[test]
    fn test_formula_operations_no_description() {
        let p = StateFormula::new("P");
        let q = StateFormula::new("Q");

        assert!(p.and(&q).description.is_none());
        assert!(p.or(&q).description.is_none());
        assert!(p.implies(&q).description.is_none());
    }

    #[test]
    fn test_formula_clone() {
        let f = StateFormula::with_description("(= x 0)", "initial");
        let cloned = f.clone();
        assert_eq!(cloned.smt_formula, "(= x 0)");
        assert_eq!(cloned.description, Some("initial".to_string()));
    }

    #[test]
    fn test_formula_serialization() {
        let f = StateFormula::with_description("(>= x 0)", "non-negative");
        let json = serde_json::to_string(&f).unwrap();
        let parsed: StateFormula = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.smt_formula, "(>= x 0)");
        assert_eq!(parsed.description, Some("non-negative".to_string()));
    }

    // ======== Property tests ========

    #[test]
    fn test_property_safety() {
        let prop = Property::safety("p1", "x_positive", StateFormula::new("(> x 0)"));
        assert_eq!(prop.id, "p1");
        assert_eq!(prop.name, "x_positive");
        assert_eq!(prop.formula.smt_formula, "(> x 0)");
        assert_eq!(prop.property_type, PropertyType::Safety);
    }

    #[test]
    fn test_property_reachability() {
        let prop = Property::reachability("r1", "goal_reached", StateFormula::new("goal"));
        assert_eq!(prop.id, "r1");
        assert_eq!(prop.name, "goal_reached");
        assert_eq!(prop.property_type, PropertyType::Reachability);
    }

    #[test]
    fn test_property_type_equality() {
        assert_eq!(PropertyType::Safety, PropertyType::Safety);
        assert_eq!(PropertyType::Reachability, PropertyType::Reachability);
        assert_eq!(PropertyType::Liveness, PropertyType::Liveness);
        assert_ne!(PropertyType::Safety, PropertyType::Reachability);
    }

    #[test]
    fn test_property_serialization() {
        let prop = Property::safety("p1", "test", StateFormula::new("true"));
        let json = serde_json::to_string(&prop).unwrap();
        let parsed: Property = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, "p1");
        assert_eq!(parsed.property_type, PropertyType::Safety);
    }

    // ======== StateVariable tests ========

    #[test]
    fn test_state_variable_clone() {
        let var = StateVariable::new("counter", SmtType::Int);
        let cloned = var.clone();
        assert_eq!(cloned.name, "counter");
        assert_eq!(cloned.smt_type, SmtType::Int);
    }

    #[test]
    fn test_state_variable_serialization() {
        let var = StateVariable::new("flag", SmtType::Bool);
        let json = serde_json::to_string(&var).unwrap();
        let parsed: StateVariable = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "flag");
        assert_eq!(parsed.smt_type, SmtType::Bool);
    }

    #[test]
    fn test_state_variable_regex_rebuilds_after_deserialize() {
        let var = StateVariable::new("idx", SmtType::Int);
        let json = serde_json::to_string(&var).unwrap();
        let parsed: StateVariable = serde_json::from_str(&json).unwrap();

        assert!(parsed.unprimed_regex().is_match("idx"));
        assert!(parsed.primed_regex().is_match("idx'"));
    }

    // ======== Instantiation tests ========

    #[test]
    fn test_step_instantiation() {
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .build();

        let formula = StateFormula::new("(>= x 0)");
        let instantiated = ts.instantiate(&formula, 5);
        assert_eq!(instantiated, "(>= x_5 0)");
    }

    #[test]
    fn test_transition_instantiation_primed_vars() {
        // Regression test: ensure primed variables like x' are correctly replaced
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' (+ x 1))")
            .build();

        // At step 0: x' -> x_1, x -> x_0
        let trans = ts.instantiate_transition(0);
        assert_eq!(trans, "(= x_1 (+ x_0 1))");

        // At step 5: x' -> x_6, x -> x_5
        let trans = ts.instantiate_transition(5);
        assert_eq!(trans, "(= x_6 (+ x_5 1))");
    }

    #[test]
    fn test_transition_instantiation_multiple_vars() {
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .variable("y", SmtType::Int)
            .init("(and (= x 0) (= y 0))")
            .transition("(and (= x' (+ x 1)) (= y' y))")
            .build();

        let trans = ts.instantiate_transition(0);
        assert_eq!(trans, "(and (= x_1 (+ x_0 1)) (= y_1 y_0))");
    }

    #[test]
    fn test_instantiation_no_variables() {
        let ts = TransitionSystem::new();
        let formula = StateFormula::new("true");
        let instantiated = ts.instantiate(&formula, 0);
        assert_eq!(instantiated, "true");
    }

    #[test]
    fn test_instantiation_complex_formula() {
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .variable("y", SmtType::Int)
            .build();

        let formula = StateFormula::new("(and (>= x 0) (< y 10) (= (+ x y) 5))");
        let instantiated = ts.instantiate(&formula, 2);
        assert_eq!(
            instantiated,
            "(and (>= x_2 0) (< y_2 10) (= (+ x_2 y_2) 5))"
        );
    }

    // ======== TransitionSystemBuilder tests ========

    #[test]
    fn test_builder_new() {
        let builder = TransitionSystemBuilder::new();
        let ts = builder.build();
        assert!(ts.variables.is_empty());
    }

    #[test]
    fn test_builder_default() {
        let builder = TransitionSystemBuilder::default();
        let ts = builder.build();
        assert!(ts.variables.is_empty());
    }

    #[test]
    fn test_builder_invariant() {
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .invariant("(>= x 0)")
            .invariant("(< x 100)")
            .build();

        assert_eq!(ts.invariants.len(), 2);
        assert_eq!(ts.invariants[0].smt_formula, "(>= x 0)");
        assert_eq!(ts.invariants[1].smt_formula, "(< x 100)");
    }

    #[test]
    fn test_builder_multiple_properties() {
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .property("p1", "first", "(> x 0)")
            .property("p2", "second", "(< x 100)")
            .build();

        assert_eq!(ts.properties.len(), 2);
        assert_eq!(ts.properties[0].id, "p1");
        assert_eq!(ts.properties[1].id, "p2");
    }

    #[test]
    fn test_builder_all_types() {
        let ts = TransitionSystemBuilder::new()
            .variable("i", SmtType::Int)
            .variable("b", SmtType::Bool)
            .variable("r", SmtType::Real)
            .variable("bv", SmtType::BitVec(32))
            .variable(
                "arr",
                SmtType::Array {
                    index: Box::new(SmtType::Int),
                    element: Box::new(SmtType::Int),
                },
            )
            .build();

        assert_eq!(ts.variables.len(), 5);
    }

    // ======== Serialization round-trip tests ========

    #[test]
    fn test_transition_system_serialization() {
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' (+ x 1))")
            .property("p1", "positive", "(>= x 0)")
            .invariant("(>= x 0)")
            .build();

        let json = serde_json::to_string(&ts).unwrap();
        let parsed: TransitionSystem = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.variables.len(), 1);
        assert_eq!(parsed.properties.len(), 1);
        assert_eq!(parsed.invariants.len(), 1);
        assert_eq!(parsed.init.smt_formula, "(= x 0)");
    }
}
