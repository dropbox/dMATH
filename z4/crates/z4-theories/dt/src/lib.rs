//! Z4 DT - Algebraic Datatypes theory solver
//!
//! Implements reasoning about algebraic datatypes (constructors, selectors, testers).

#![warn(missing_docs)]
#![warn(clippy::all)]

use z4_core::term::TermId;
use z4_core::{TheoryPropagation, TheoryResult, TheorySolver};

/// Datatype theory solver
pub struct DtSolver {
    // TODO: Add datatype definitions and cycle detection
}

impl DtSolver {
    /// Create a new DT solver
    #[must_use]
    pub fn new() -> Self {
        DtSolver {}
    }
}

impl Default for DtSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl TheorySolver for DtSolver {
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
