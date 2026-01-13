//! Predicates (uninterpreted relations) in CHC problems

use crate::ChcSort;
use std::fmt;

/// Unique identifier for a predicate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PredicateId(pub(crate) u32);

impl PredicateId {
    /// Create a new predicate ID
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw ID value
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for PredicateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "P{}", self.0)
    }
}

/// A predicate declaration
///
/// A predicate represents an uninterpreted relation that the solver
/// will find an interpretation for.
#[derive(Debug, Clone)]
pub struct Predicate {
    /// Unique identifier
    pub id: PredicateId,
    /// Human-readable name
    pub name: String,
    /// Sorts of arguments
    pub arg_sorts: Vec<ChcSort>,
}

impl Predicate {
    /// Create a new predicate
    pub fn new(id: PredicateId, name: impl Into<String>, arg_sorts: Vec<ChcSort>) -> Self {
        Self {
            id,
            name: name.into(),
            arg_sorts,
        }
    }

    /// Get the arity (number of arguments)
    pub fn arity(&self) -> usize {
        self.arg_sorts.len()
    }
}

impl fmt::Display for Predicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(", self.name)?;
        for (i, sort) in self.arg_sorts.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{sort:?}")?;
        }
        write!(f, ")")
    }
}
