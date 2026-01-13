//! Sort system for Z4
//!
//! Sorts are the types of terms in SMT-LIB.

use std::fmt;

/// A sort (type) in the SMT-LIB language.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Sort {
    /// Boolean sort
    Bool,
    /// Integer sort
    Int,
    /// Real sort
    Real,
    /// Bitvector sort with width
    BitVec(u32),
    /// Array sort with index and element sorts
    Array(Box<Sort>, Box<Sort>),
    /// String sort
    String,
    /// Floating-point sort with exponent and significand bits
    FloatingPoint(u32, u32),
    /// Uninterpreted sort
    Uninterpreted(String),
    /// Datatype sort
    Datatype(String),
}

impl fmt::Display for Sort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sort::Bool => write!(f, "Bool"),
            Sort::Int => write!(f, "Int"),
            Sort::Real => write!(f, "Real"),
            Sort::BitVec(w) => write!(f, "(_ BitVec {})", w),
            Sort::Array(idx, elem) => write!(f, "(Array {} {})", idx, elem),
            Sort::String => write!(f, "String"),
            Sort::FloatingPoint(eb, sb) => write!(f, "(_ FloatingPoint {} {})", eb, sb),
            Sort::Uninterpreted(name) => write!(f, "{}", name),
            Sort::Datatype(name) => write!(f, "{}", name),
        }
    }
}
