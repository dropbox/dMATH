//! Model representation for Z4
//!
//! A model is a satisfying assignment for all variables in a formula.

use crate::term::Constant;
use hashbrown::HashMap;

/// A satisfying assignment
#[derive(Debug, Clone, Default)]
pub struct Model {
    /// Variable assignments
    pub assignments: HashMap<String, Constant>,
    /// Function interpretations
    pub functions: HashMap<String, FunctionInterpretation>,
}

/// Interpretation of an uninterpreted function
#[derive(Debug, Clone)]
pub struct FunctionInterpretation {
    /// Explicit mappings from arguments to results
    pub entries: Vec<(Vec<Constant>, Constant)>,
    /// Default value for unmapped inputs
    pub default: Option<Constant>,
}

impl Model {
    /// Create a new empty model
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the value of a variable
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Constant> {
        self.assignments.get(name)
    }
}
