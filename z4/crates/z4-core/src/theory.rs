//! Theory solver trait for Z4
//!
//! All theory solvers implement this trait to integrate with the DPLL(T) framework.

use crate::term::TermId;
use num_bigint::BigInt;
use num_rational::BigRational;

/// A signed theory literal (term + Boolean value).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TheoryLit {
    /// The term representing the (Boolean) atom.
    pub term: TermId,
    /// The Boolean value of the atom.
    pub value: bool,
}

impl TheoryLit {
    /// Create a new signed theory literal.
    #[must_use]
    pub fn new(term: TermId, value: bool) -> Self {
        Self { term, value }
    }
}

/// A request from a theory solver to split on an integer variable.
///
/// Used for branch-and-bound in LIA: when the LRA relaxation gives x = 2.5,
/// the solver requests a split to force (x <= 2) OR (x >= 3).
#[derive(Debug, Clone)]
pub struct SplitRequest {
    /// The integer variable to split on
    pub variable: TermId,
    /// The non-integer value from the LRA relaxation
    pub value: BigRational,
    /// Floor of the value (lower bound in the split)
    pub floor: BigInt,
    /// Ceiling of the value (upper bound in the split)
    pub ceil: BigInt,
}

/// A request from a theory solver to split on a disequality.
///
/// Used when a disequality `x != c` is violated by the current model (x = c)
/// but the variable has slack (can take other values). The DPLL(T) layer
/// should create atoms `x < c` and `x > c` and add the clause `(x < c) OR (x > c)`.
#[derive(Debug, Clone)]
pub struct DisequlitySplitRequest {
    /// The variable/expression that must be different from the excluded value
    pub variable: TermId,
    /// The value that is excluded by the disequality
    pub excluded_value: BigRational,
}

/// A request from a theory solver to split on a multi-variable expression.
///
/// Used when a multi-variable disequality `E != F` (or `E - F != 0`) is violated.
/// Single-value enumeration doesn't work for these - we need to split on
/// `E < F OR E > F` directly. The DPLL(T) layer should parse the disequality
/// term to extract LHS and RHS, then create atoms for the comparison.
#[derive(Debug, Clone)]
pub struct ExpressionSplitRequest {
    /// The disequality term that was violated (the `distinct` or negated `=` term).
    /// The SMT layer should extract LHS and RHS from this term.
    pub disequality_term: TermId,
}

/// Result of a theory check
#[derive(Debug, Clone)]
pub enum TheoryResult {
    /// The current assignment is satisfiable
    Sat,
    /// The current assignment is unsatisfiable, with a conflicting set of signed literals.
    ///
    /// The returned set represents assignments that cannot all hold simultaneously.
    /// The DPLL(T) layer negates these literals to produce a blocking clause.
    Unsat(Vec<TheoryLit>),
    /// Unknown (theory solver could not determine)
    Unknown,
    /// Theory needs to split on an integer variable for branch-and-bound.
    ///
    /// The DPLL layer should create atoms `var <= floor` and `var >= ceil`,
    /// add the clause `(var <= floor) OR (var >= ceil)`, and continue solving.
    NeedSplit(SplitRequest),
    /// Theory needs to split on a disequality.
    ///
    /// The DPLL layer should create atoms `var < value` and `var > value`,
    /// add the clause `(var < value) OR (var > value)`, and continue solving.
    NeedDisequlitySplit(DisequlitySplitRequest),
    /// Theory needs to split on a multi-variable expression disequality.
    ///
    /// Used when `E != F` is violated but single-value enumeration would be infinite.
    /// The DPLL layer should parse the disequality term to get LHS and RHS,
    /// then create atoms `LHS < RHS` and `LHS > RHS`, add the clause
    /// `(LHS < RHS) OR (LHS > RHS)`, and continue solving.
    NeedExpressionSplit(ExpressionSplitRequest),
}

/// A propagated literal from a theory solver
#[derive(Debug, Clone)]
pub struct TheoryPropagation {
    /// The propagated literal
    pub literal: TheoryLit,
    /// The reason (antecedents) for the propagation
    pub reason: Vec<TheoryLit>,
}

/// Trait for theory solvers
pub trait TheorySolver {
    /// Assert a literal to the theory solver
    fn assert_literal(&mut self, literal: TermId, value: bool);

    /// Check consistency of current assignment
    fn check(&mut self) -> TheoryResult;

    /// Get propagated literals
    fn propagate(&mut self) -> Vec<TheoryPropagation>;

    /// Push a new scope
    fn push(&mut self);

    /// Pop to previous scope
    fn pop(&mut self);

    /// Reset the solver completely, clearing all state
    fn reset(&mut self);

    /// Soft reset: clear assertions but preserve learned state.
    ///
    /// This is called between SAT model iterations in DPLL(T). Unlike `reset()`,
    /// this preserves learned information (e.g., HNF cuts in LIA) that remains
    /// valid across different SAT assignments.
    ///
    /// Default implementation calls `reset()`. Theory solvers with learned state
    /// should override this to preserve that state.
    fn soft_reset(&mut self) {
        self.reset();
    }
}
