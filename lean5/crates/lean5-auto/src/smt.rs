//! SMT solver core combining SAT + theory solvers (DPLL(T))
//!
//! This module implements the DPLL(T) framework for Satisfiability Modulo Theories.
//! It combines a SAT solver (CDCL) with theory solvers to handle first-order logic
//! with interpreted symbols.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                         SMT Solver                               │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  ┌─────────────┐        ┌─────────────────────────────────────┐ │
//! │  │    CDCL     │◄──────►│         Theory Combination          │ │
//! │  │  SAT Core   │        │                                     │ │
//! │  └─────────────┘        │  ┌─────────┐ ┌─────────┐ ┌───────┐  │ │
//! │                         │  │Equality │ │  Arith  │ │Arrays │  │ │
//! │                         │  │(E-graph)│ │  (LRA)  │ │       │  │ │
//! │                         │  └─────────┘ └─────────┘ └───────┘  │ │
//! │                         └─────────────────────────────────────┘ │
//! │                                                                  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # DPLL(T) Algorithm
//!
//! 1. SAT solver makes decisions and propagates boolean constraints
//! 2. After each propagation, theory solvers check consistency
//! 3. Theory solvers can:
//!    - Propagate theory consequences (theory propagation)
//!    - Detect conflicts (theory conflict)
//! 4. On conflict, learn a clause and backtrack
//! 5. Repeat until SAT (with theory-consistent model) or UNSAT

use crate::cdcl::{CdclSolver, ClauseRef, Lit, SolveResult, Var};
use crate::egraph::Symbol;
use std::any::Any;
use std::collections::HashMap;

/// A theory literal - represents an atomic formula in a theory
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TheoryLiteral {
    /// Equality between two terms: t1 = t2
    Eq(TermId, TermId),
    /// Disequality: t1 ≠ t2
    Neq(TermId, TermId),
    /// Arithmetic: t1 < t2
    Lt(TermId, TermId),
    /// Arithmetic: t1 ≤ t2
    Le(TermId, TermId),
    /// Boolean variable (uninterpreted)
    Bool(u32),
}

impl TheoryLiteral {
    /// Negate this theory literal
    #[must_use]
    pub fn negate(&self) -> Self {
        match self {
            TheoryLiteral::Eq(a, b) => TheoryLiteral::Neq(*a, *b),
            TheoryLiteral::Neq(a, b) => TheoryLiteral::Eq(*a, *b),
            TheoryLiteral::Lt(a, b) => TheoryLiteral::Le(*b, *a), // ¬(a < b) ≡ b ≤ a
            TheoryLiteral::Le(a, b) => TheoryLiteral::Lt(*b, *a), // ¬(a ≤ b) ≡ b < a
            TheoryLiteral::Bool(v) => TheoryLiteral::Bool(*v),    // Handled by SAT solver
        }
    }
}

/// Term identifier in the SMT context
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TermId(pub u32);

/// Result of theory consistency check
#[derive(Clone, Debug)]
pub enum TheoryCheckResult {
    /// Theory is consistent with current assignment
    Consistent,
    /// Theory found a conflict - returns the conflicting literals
    Conflict(Vec<Lit>),
    /// Theory propagated new literals
    Propagation(Vec<Lit>),
}

/// Trait for theory solvers
pub trait TheorySolver: Send + Sync {
    /// Called when a literal becomes true in the SAT solver
    /// Returns propagated literals or a conflict
    fn assert_literal(&mut self, lit: Lit, theory_lit: &TheoryLiteral) -> TheoryCheckResult;

    /// Check full consistency of the theory
    fn check(&self) -> TheoryCheckResult;

    /// Backtrack to a given decision level
    fn backtrack(&mut self, level: u32);

    /// Push a new decision level
    fn push(&mut self);

    /// Get the name of this theory (for debugging)
    fn name(&self) -> &'static str;

    /// Set the terms used by this theory (for theories that need term structure)
    /// Default implementation does nothing.
    fn set_terms(&mut self, _terms: Vec<SmtTerm>) {
        // Default: do nothing
    }

    /// Downcast support for accessing concrete theory implementations
    fn as_any(&self) -> &dyn Any;

    /// Mutable downcast support
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// SMT solver combining CDCL SAT solver with theory solvers
pub struct SmtSolver {
    /// The underlying CDCL SAT solver
    sat: CdclSolver,
    /// Theory solvers
    theories: Vec<Box<dyn TheorySolver>>,
    /// Mapping from SAT variables to theory literals
    var_to_theory: HashMap<Var, TheoryLiteral>,
    /// Mapping from theory literals to SAT variables
    theory_to_var: HashMap<TheoryLiteral, Var>,
    /// Term storage
    terms: Vec<SmtTerm>,
    /// Mapping from term representation to term ID
    term_map: HashMap<SmtTerm, TermId>,
    /// Current decision level (for future incremental solving)
    #[allow(dead_code)]
    decision_level: u32,
}

/// Internal term representation
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SmtTerm {
    /// A constant/variable
    Const(Symbol),
    /// Function application
    App(Symbol, Vec<TermId>),
    /// Integer constant
    Int(i64),
    /// Rational constant (numerator, denominator)
    Rat(i64, i64),
}

impl SmtSolver {
    /// Create a new SMT solver
    pub fn new() -> Self {
        SmtSolver {
            sat: CdclSolver::new(0),
            theories: Vec::new(),
            var_to_theory: HashMap::new(),
            theory_to_var: HashMap::new(),
            terms: Vec::new(),
            term_map: HashMap::new(),
            decision_level: 0,
        }
    }

    /// Add a theory solver, returns its index
    pub fn add_theory(&mut self, theory: Box<dyn TheorySolver>) -> usize {
        let idx = self.theories.len();
        self.theories.push(theory);
        idx
    }

    /// Get a reference to a theory solver by index
    pub fn get_theory(&self, idx: usize) -> Option<&dyn TheorySolver> {
        self.theories.get(idx).map(AsRef::as_ref)
    }

    /// Get a mutable reference to a theory solver by index
    pub fn get_theory_mut(&mut self, idx: usize) -> Option<&mut Box<dyn TheorySolver>> {
        self.theories.get_mut(idx)
    }

    /// Get a typed reference to a theory solver by index
    pub fn get_theory_typed<T: 'static>(&self, idx: usize) -> Option<&T> {
        self.theories
            .get(idx)
            .and_then(|t| t.as_any().downcast_ref::<T>())
    }

    /// Get a typed mutable reference to a theory solver by index
    pub fn get_theory_typed_mut<T: 'static>(&mut self, idx: usize) -> Option<&mut T> {
        self.theories
            .get_mut(idx)
            .and_then(|t| t.as_any_mut().downcast_mut::<T>())
    }

    /// Get a reference to the terms
    pub fn terms(&self) -> &[SmtTerm] {
        &self.terms
    }

    /// Create a new constant term
    pub fn const_term(&mut self, name: impl Into<Symbol>) -> TermId {
        let term = SmtTerm::Const(name.into());
        self.intern_term(term)
    }

    /// Create a function application term
    pub fn app_term(&mut self, name: impl Into<Symbol>, args: Vec<TermId>) -> TermId {
        let term = SmtTerm::App(name.into(), args);
        self.intern_term(term)
    }

    /// Create an integer constant term
    pub fn int_term(&mut self, value: i64) -> TermId {
        let term = SmtTerm::Int(value);
        self.intern_term(term)
    }

    /// Create a select (array read) term: select(array, index) → value
    pub fn select_term(&mut self, array: TermId, index: TermId) -> TermId {
        self.app_term("select", vec![array, index])
    }

    /// Create a store (array write) term: store(array, index, value) → new_array
    pub fn store_term(&mut self, array: TermId, index: TermId, value: TermId) -> TermId {
        self.app_term("store", vec![array, index, value])
    }

    /// Intern a term (deduplicate)
    fn intern_term(&mut self, term: SmtTerm) -> TermId {
        if let Some(&id) = self.term_map.get(&term) {
            return id;
        }
        let id = TermId(
            u32::try_from(self.terms.len()).expect("term count exceeded u32::MAX during interning"),
        );
        self.terms.push(term.clone());
        self.term_map.insert(term, id);
        id
    }

    /// Get or create a SAT variable for a theory literal
    fn get_or_create_var(&mut self, lit: TheoryLiteral) -> Var {
        if let Some(&var) = self.theory_to_var.get(&lit) {
            return var;
        }
        let var = self.sat.new_var();
        self.var_to_theory.insert(var, lit.clone());
        self.theory_to_var.insert(lit, var);
        var
    }

    /// Assert an equality constraint: t1 = t2
    pub fn assert_eq(&mut self, t1: TermId, t2: TermId) {
        let lit = TheoryLiteral::Eq(t1, t2);
        let var = self.get_or_create_var(lit);
        // Add unit clause forcing this equality
        self.sat.add_clause(vec![Lit::pos(var)]);
    }

    /// Assert a disequality constraint: t1 ≠ t2
    pub fn assert_neq(&mut self, t1: TermId, t2: TermId) {
        let var = self.get_or_create_var(TheoryLiteral::Eq(t1, t2));
        // Add unit clause forcing this disequality (negation of equality var)
        self.sat.add_clause(vec![Lit::neg(var)]);
    }

    /// Add a clause over theory literals
    pub fn add_clause(&mut self, theory_lits: Vec<TheoryLiteral>) -> Option<ClauseRef> {
        let sat_lits: Vec<Lit> = theory_lits
            .into_iter()
            .map(|tl| {
                let (base_lit, positive) = match &tl {
                    TheoryLiteral::Eq(a, b) => (TheoryLiteral::Eq(*a, *b), true),
                    TheoryLiteral::Neq(a, b) => (TheoryLiteral::Eq(*a, *b), false),
                    TheoryLiteral::Lt(a, b) => (TheoryLiteral::Lt(*a, *b), true),
                    TheoryLiteral::Le(a, b) => (TheoryLiteral::Le(*a, *b), true),
                    TheoryLiteral::Bool(v) => (TheoryLiteral::Bool(*v), true),
                };
                let var = self.get_or_create_var(base_lit);
                if positive {
                    Lit::pos(var)
                } else {
                    Lit::neg(var)
                }
            })
            .collect();
        self.sat.add_clause(sat_lits)
    }

    /// Solve the SMT problem
    pub fn solve(&mut self) -> SmtResult {
        // Run SAT solver with theory integration
        self.sat_solve_with_theory()
    }

    /// Run SAT solver with theory integration
    fn sat_solve_with_theory(&mut self) -> SmtResult {
        // For now, use the basic approach:
        // 1. Let SAT solver find a complete assignment
        // 2. Check with theories
        // 3. If conflict, add blocking clause and retry

        match self.sat.solve() {
            SolveResult::Sat(model) => {
                // Check theory consistency
                match self.check_theories(&model) {
                    TheoryCheckResult::Consistent => {
                        // Build SMT model
                        let smt_model = self.build_model(&model);
                        SmtResult::Sat(smt_model)
                    }
                    TheoryCheckResult::Conflict(conflict_lits) => {
                        // Learn the conflict clause and retry
                        // The conflict clause is the negation of the conflicting assignment
                        let clause: Vec<Lit> = conflict_lits.iter().map(|l| l.not()).collect();
                        if self.sat.add_clause(clause).is_none() {
                            // Adding the conflict clause made it UNSAT
                            return SmtResult::Unsat;
                        }
                        // Retry SAT solving
                        self.sat_solve_with_theory()
                    }
                    TheoryCheckResult::Propagation(_) => {
                        // Theory propagation during final check is unusual
                        // but could happen - treat as consistent for now
                        let smt_model = self.build_model(&model);
                        SmtResult::Sat(smt_model)
                    }
                }
            }
            SolveResult::Unsat => SmtResult::Unsat,
            SolveResult::Unknown => SmtResult::Unknown,
        }
    }

    /// Check all theories for consistency
    fn check_theories(&mut self, model: &[bool]) -> TheoryCheckResult {
        // Share terms with all theories
        for theory in &mut self.theories {
            theory.set_terms(self.terms.clone());
        }

        // First, assert all literals to theories
        for (&var, theory_lit) in &self.var_to_theory {
            let value = model[var.index()];
            let effective_lit = if value {
                theory_lit.clone()
            } else {
                theory_lit.negate()
            };

            let sat_lit = if value { Lit::pos(var) } else { Lit::neg(var) };

            for theory in &mut self.theories {
                match theory.assert_literal(sat_lit, &effective_lit) {
                    TheoryCheckResult::Conflict(lits) => {
                        return TheoryCheckResult::Conflict(lits);
                    }
                    TheoryCheckResult::Propagation(lits) => {
                        // Could add these to propagate, but for offline check, ignore
                        let _ = lits;
                    }
                    TheoryCheckResult::Consistent => {}
                }
            }
        }

        // Then check full consistency
        for theory in &self.theories {
            match theory.check() {
                TheoryCheckResult::Conflict(lits) => {
                    return TheoryCheckResult::Conflict(lits);
                }
                result @ TheoryCheckResult::Propagation(_) => {
                    return result;
                }
                TheoryCheckResult::Consistent => {}
            }
        }

        TheoryCheckResult::Consistent
    }

    /// Build an SMT model from a SAT model
    fn build_model(&self, sat_model: &[bool]) -> SmtModel {
        let mut equalities = Vec::new();
        let mut disequalities = Vec::new();

        for (&var, theory_lit) in &self.var_to_theory {
            let value = sat_model[var.index()];
            if let TheoryLiteral::Eq(a, b) = theory_lit {
                if value {
                    equalities.push((*a, *b));
                } else {
                    disequalities.push((*a, *b));
                }
            }
        }

        SmtModel {
            sat_model: sat_model.to_vec(),
            equalities,
            disequalities,
        }
    }

    /// Get a term by ID
    pub fn get_term(&self, id: TermId) -> Option<&SmtTerm> {
        self.terms.get(id.0 as usize)
    }

    /// Get statistics
    pub fn stats(&self) -> SmtStats {
        let sat_stats = self.sat.stats();
        SmtStats {
            num_vars: self.sat.num_vars(),
            num_clauses: self.sat.num_clauses(),
            num_terms: self.terms.len(),
            sat_conflicts: sat_stats.conflicts,
            sat_decisions: sat_stats.decisions,
        }
    }
}

impl Default for SmtSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of SMT solving
#[derive(Clone, Debug)]
pub enum SmtResult {
    /// Satisfiable with the given model
    Sat(SmtModel),
    /// Unsatisfiable
    Unsat,
    /// Unknown (resource limit or incomplete)
    Unknown,
}

/// SMT model
#[derive(Clone, Debug)]
pub struct SmtModel {
    /// The underlying SAT model
    pub sat_model: Vec<bool>,
    /// Equalities that hold
    pub equalities: Vec<(TermId, TermId)>,
    /// Disequalities that hold
    pub disequalities: Vec<(TermId, TermId)>,
}

/// SMT solver statistics
#[derive(Clone, Debug, Default)]
pub struct SmtStats {
    pub num_vars: usize,
    pub num_clauses: usize,
    pub num_terms: usize,
    pub sat_conflicts: u64,
    pub sat_decisions: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smt_basic() {
        let mut smt = SmtSolver::new();

        // Create terms
        let a = smt.const_term("a");
        let b = smt.const_term("b");

        // Assert a = b
        smt.assert_eq(a, b);

        // Should be SAT (no theory solver to check)
        match smt.solve() {
            SmtResult::Sat(model) => {
                assert!(model.equalities.contains(&(a, b)));
            }
            _ => panic!("Expected SAT"),
        }
    }

    #[test]
    fn test_smt_conflict_basic() {
        let mut smt = SmtSolver::new();

        let a = smt.const_term("a");
        let b = smt.const_term("b");

        // Assert a = b and a ≠ b - pure SAT conflict
        smt.assert_eq(a, b);
        smt.assert_neq(a, b);

        // Should be UNSAT from SAT level
        match smt.solve() {
            SmtResult::Unsat => {}
            _ => panic!("Expected UNSAT"),
        }
    }

    #[test]
    fn test_smt_term_interning() {
        let mut smt = SmtSolver::new();

        let a1 = smt.const_term("a");
        let a2 = smt.const_term("a");
        let b = smt.const_term("b");

        // Same name should return same term ID
        assert_eq!(a1, a2);
        assert_ne!(a1, b);
    }

    #[test]
    fn test_smt_app_terms() {
        let mut smt = SmtSolver::new();

        let a = smt.const_term("a");
        let b = smt.const_term("b");
        let fa = smt.app_term("f", vec![a]);
        let fb = smt.app_term("f", vec![b]);
        let fa2 = smt.app_term("f", vec![a]);

        // Same application should return same term ID
        assert_eq!(fa, fa2);
        assert_ne!(fa, fb);
    }

    #[test]
    fn test_smt_clause() {
        let mut smt = SmtSolver::new();

        let a = smt.const_term("a");
        let b = smt.const_term("b");
        let c = smt.const_term("c");

        // a = b OR a = c
        smt.add_clause(vec![TheoryLiteral::Eq(a, b), TheoryLiteral::Eq(a, c)]);

        // Should be SAT
        match smt.solve() {
            SmtResult::Sat(_) => {}
            _ => panic!("Expected SAT"),
        }
    }

    #[test]
    fn test_smt_stats() {
        let mut smt = SmtSolver::new();

        let a = smt.const_term("a");
        let b = smt.const_term("b");
        smt.assert_eq(a, b);

        let stats = smt.stats();
        assert_eq!(stats.num_terms, 2);
        assert!(stats.num_vars >= 1);
    }

    #[test]
    fn test_smt_disjunction_choice() {
        let mut smt = SmtSolver::new();

        let a = smt.const_term("a");
        let b = smt.const_term("b");
        let c = smt.const_term("c");

        // (a = b) OR (a = c)
        smt.add_clause(vec![TheoryLiteral::Eq(a, b), TheoryLiteral::Eq(a, c)]);

        // Force a ≠ b
        smt.assert_neq(a, b);

        // Now a = c must be true
        match smt.solve() {
            SmtResult::Sat(model) => {
                assert!(model.equalities.contains(&(a, c)));
                assert!(model.disequalities.contains(&(a, b)));
            }
            _ => panic!("Expected SAT"),
        }
    }

    #[test]
    fn test_smt_multiple_equalities() {
        let mut smt = SmtSolver::new();

        let a = smt.const_term("a");
        let b = smt.const_term("b");
        let c = smt.const_term("c");
        let d = smt.const_term("d");

        // a = b AND c = d
        smt.assert_eq(a, b);
        smt.assert_eq(c, d);

        match smt.solve() {
            SmtResult::Sat(model) => {
                assert!(model.equalities.contains(&(a, b)));
                assert!(model.equalities.contains(&(c, d)));
            }
            _ => panic!("Expected SAT"),
        }
    }
}
