//! Z4 Strings - String theory solver
//!
//! Implements word equations and regular expression constraints.

#![warn(missing_docs)]
#![warn(clippy::all)]

use z4_core::term::TermId;
use z4_core::{TheoryPropagation, TheoryResult, TheorySolver};

/// String theory solver
pub struct StringSolver {
    // TODO: Add word equation solver and automata
}

impl StringSolver {
    /// Create a new string solver
    pub fn new() -> Self {
        StringSolver {}
    }
}

impl Default for StringSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl TheorySolver for StringSolver {
    fn assert_literal(&mut self, _literal: TermId, _value: bool) {}
    fn check(&mut self) -> TheoryResult {
        TheoryResult::Sat
    }
    fn propagate(&mut self) -> Vec<TheoryPropagation> {
        Vec::new()
    }
    fn push(&mut self) {}
    fn pop(&mut self) {}
    fn reset(&mut self) {}
}
