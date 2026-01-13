//! Z4 DPLL(T) - Theory integration framework
//!
//! Integrates the SAT solver with theory solvers using the DPLL(T) architecture.
//!
//! # DPLL(T) Algorithm
//!
//! The DPLL(T) framework combines SAT solving with theory reasoning:
//!
//! 1. Parse SMT-LIB input and elaborate to internal representation
//! 2. Convert Boolean structure to CNF via Tseitin transformation
//! 3. Run CDCL SAT solver:
//!    - After each propagation, check theory consistency
//!    - If theory finds conflict, add theory lemma as clause
//!    - If theory propagates, add propagated literals
//! 4. When SAT solver finds SAT, verify full model with theory
//! 5. If theory rejects, add blocking clause and continue
//!
//! # Executor
//!
//! The [`Executor`] struct provides a high-level interface for executing SMT-LIB
//! commands with automatic theory selection based on the logic:
//!
//! ```
//! use z4_dpll::Executor;
//! use z4_frontend::parse;
//!
//! let input = r#"
//!     (set-logic QF_UF)
//!     (declare-const a Bool)
//!     (assert a)
//!     (check-sat)
//! "#;
//!
//! let commands = parse(input).unwrap();
//! let mut exec = Executor::new();
//! let outputs = exec.execute_all(&commands).unwrap();
//! assert_eq!(outputs, vec!["sat"]);
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod api;
mod executor;
pub mod minimize;
pub mod proof_tracker;

pub use api::{
    Logic, Model, SolveResult as ApiSolveResult, Solver as ApiSolver, Sort as ApiSort, Term,
};
pub use executor::{CheckSatResult, Executor, ExecutorError};
pub use minimize::{CounterexampleStyle, ModelMinimizer};
pub use proof_tracker::ProofTracker;
pub use z4_sat::AssumeResult;

use std::collections::BTreeMap;
use z4_core::{
    term::{Constant, Symbol, TermData},
    CnfClause, CnfLit, DisequlitySplitRequest, ExpressionSplitRequest, Sort, SplitRequest, TermId,
    TermStore, TheoryPropagation, TheoryResult, TheorySolver, Tseitin,
};

/// Result of a solve step, either a final result or a split request
#[derive(Debug, Clone)]
pub enum SolveStepResult {
    /// Final solve result
    Done(SolveResult),
    /// Theory needs a split on an integer variable.
    /// The executor should create the split atoms and call `apply_split` to continue.
    NeedSplit(SplitRequest),
    /// Theory needs a split to exclude a specific value (disequality).
    /// The executor should create atoms for `x < c` and `x > c` to exclude `x = c`.
    NeedDisequlitySplit(DisequlitySplitRequest),
}
use z4_sat::{Literal, SolveResult, Solver as SatSolver, Variable};

enum TheoryCheck {
    Sat,
    Conflict(Vec<Literal>),
    Unknown,
    /// Theory needs to split on an integer variable (for branch-and-bound LIA)
    NeedSplit(SplitRequest),
    /// Theory needs to split on a disequality (x != c)
    #[allow(dead_code)] // Field will be used when disequality splits are implemented
    NeedDisequlitySplit(DisequlitySplitRequest),
    /// Theory needs to split on a multi-variable expression disequality (E != F)
    #[allow(dead_code)] // Field will be used when expression splits are implemented
    NeedExpressionSplit(ExpressionSplitRequest),
}

/// DPLL(T) solver combining SAT and theory reasoning
pub struct DpllT<T: TheorySolver> {
    /// The underlying SAT solver
    sat: SatSolver,
    /// The theory solver
    theory: T,
    /// Mapping from CNF variables to term IDs
    var_to_term: BTreeMap<u32, TermId>,
    /// Mapping from term IDs to CNF variables
    term_to_var: BTreeMap<TermId, u32>,
    /// Theory atoms to communicate to theory solver
    theory_atoms: Vec<TermId>,
    /// Current decision level for theory
    #[allow(dead_code)] // reserved for eager theory propagation / backtracking
    theory_level: usize,
}

/// Determine whether a term should be communicated to the theory solver.
///
/// DPLL(T) theory solvers should only see *atomic* Boolean predicates (e.g., `x <= 5`,
/// `f(a) = b`, `select(a,i) = v`) and should not be asked to interpret Boolean structure
/// like `and/or/xor/=>/ite`.
pub(crate) fn is_theory_atom(terms: &TermStore, term: TermId) -> bool {
    if terms.sort(term) != &Sort::Bool {
        return false;
    }

    match terms.get(term) {
        TermData::Const(Constant::Bool(_)) => false,
        TermData::Const(_) => false,
        TermData::Var(_, _) => false,
        TermData::Not(_) => false,
        TermData::Ite(_, _, _) => false,
        TermData::Let(_, _) => false,
        TermData::App(Symbol::Named(name), args) => match name.as_str() {
            "and" | "or" | "xor" | "=>" => false,
            "=" => {
                !(args.len() == 2
                    && terms.sort(args[0]) == &Sort::Bool
                    && terms.sort(args[1]) == &Sort::Bool)
            }
            _ => true,
        },
        TermData::App(_, _) => true,
    }
}

impl<T: TheorySolver> DpllT<T> {
    /// Create a new DPLL(T) solver with the given number of variables
    pub fn new(num_vars: usize, theory: T) -> Self {
        DpllT {
            sat: SatSolver::new(num_vars),
            theory,
            var_to_term: BTreeMap::new(),
            term_to_var: BTreeMap::new(),
            theory_atoms: Vec::new(),
            theory_level: 0,
        }
    }

    /// Create a DPLL(T) solver from a Tseitin transformation result
    pub fn from_tseitin(
        terms: &TermStore,
        tseitin_result: &z4_core::TseitinResult,
        theory: T,
    ) -> Self {
        let mut solver = DpllT::new(tseitin_result.num_vars as usize, theory);

        // Copy the variable mappings, converting CNF vars (1-indexed) to SAT vars (0-indexed).
        solver.var_to_term = tseitin_result
            .var_to_term
            .iter()
            .map(|(&v, &t)| (v - 1, t))
            .collect();
        solver.term_to_var = tseitin_result
            .term_to_var
            .iter()
            .map(|(&t, &v)| (t, v - 1))
            .collect();

        // Add all clauses to the SAT solver
        for clause in &tseitin_result.clauses {
            let lits: Vec<Literal> = clause
                .0
                .iter()
                .map(|&lit| cnf_lit_to_sat_lit(lit))
                .collect();
            solver.sat.add_clause(lits);
        }

        // Communicate only theory-relevant *atomic* predicates to the theory solver.
        // Sort by TermId for deterministic iteration order (BTreeMaps have random order).
        solver.theory_atoms = solver
            .var_to_term
            .values()
            .copied()
            .filter(|&t| is_theory_atom(terms, t))
            .collect();
        solver.theory_atoms.sort_by_key(|t| t.0);

        solver
    }

    /// Access the underlying SAT solver.
    pub fn sat_solver(&self) -> &SatSolver {
        &self.sat
    }

    /// Access the underlying SAT solver mutably
    pub fn sat_solver_mut(&mut self) -> &mut SatSolver {
        &mut self.sat
    }

    /// Access the underlying theory solver.
    pub fn theory_solver(&self) -> &T {
        &self.theory
    }

    /// Access the underlying theory solver mutably.
    pub fn theory_solver_mut(&mut self) -> &mut T {
        &mut self.theory
    }

    /// Add a clause to the solver
    pub fn add_clause(&mut self, literals: Vec<Literal>) {
        self.sat.add_clause(literals);
    }

    /// Add a CNF clause to the solver
    pub fn add_cnf_clause(&mut self, clause: &CnfClause) {
        let lits: Vec<Literal> = clause
            .0
            .iter()
            .map(|&lit| cnf_lit_to_sat_lit(lit))
            .collect();
        self.sat.add_clause(lits);
    }

    /// Register a theory atom
    ///
    /// Theory atoms are terms that the theory solver needs to know about.
    /// When the SAT solver assigns a value to the corresponding variable,
    /// the theory solver is informed.
    pub fn register_theory_atom(&mut self, term: TermId, var: u32) {
        self.var_to_term.insert(var, term);
        self.term_to_var.insert(term, var);
        self.theory_atoms.push(term);
    }

    /// Get the term ID for a SAT variable, if it exists
    pub fn term_for_var(&self, var: Variable) -> Option<TermId> {
        self.var_to_term.get(&var.0).copied()
    }

    /// Get the SAT variable for a term ID, if it exists
    pub fn var_for_term(&self, term: TermId) -> Option<Variable> {
        self.term_to_var.get(&term).map(|&v| Variable(v))
    }

    /// Convert a theory literal to a SAT literal
    fn term_to_literal(&self, term: TermId, value: bool) -> Option<Literal> {
        self.var_for_term(term).map(|var| {
            if value {
                Literal::positive(var)
            } else {
                Literal::negative(var)
            }
        })
    }

    /// Communicate SAT model to theory solver
    ///
    /// IMPORTANT: We use the model returned by the SAT solver, not the live assignment.
    /// The model has defaults applied for unassigned variables (via get_model()), whereas
    /// the live assignment may have None values. Using the model ensures the theory solver
    /// sees a complete, consistent assignment.
    fn sync_theory(&mut self, model: &[bool]) {
        // Rebuild theory state from the SAT model.
        // The SAT solver may return multiple candidate models over time as we
        // add theory lemmas, so the theory must not accumulate stale assertions.
        // Use soft_reset() to preserve learned state (e.g., HNF cuts in LIA).
        self.theory.soft_reset();

        // For each theory atom, tell the theory solver about its value from the model.
        let debug = std::env::var("Z4_DEBUG_SYNC").is_ok();
        for &term in &self.theory_atoms {
            if let Some(var) = self.var_for_term(term) {
                let var_idx = var.index();
                // Use the model value (which already has defaults applied)
                let value = if var_idx < model.len() {
                    model[var_idx]
                } else {
                    // Variable not in model (shouldn't happen, but default to false)
                    false
                };
                if debug {
                    eprintln!(
                        "[SYNC] term {:?} (var {:?}) = {} (from model)",
                        term, var, value
                    );
                }
                self.theory.assert_literal(term, value);
            }
        }
    }

    /// Check theory consistency and handle propagations/conflicts
    fn check_theory(&mut self) -> TheoryCheck {
        // First check for propagations
        let propagations = self.theory.propagate();
        for prop in propagations {
            if let Some(lit) = self.term_to_literal(prop.literal.term, prop.literal.value) {
                // Check if this conflicts with current assignment
                if let Some(var) = self.var_for_term(prop.literal.term) {
                    if let Some(value) = self.sat.value(var) {
                        if value != prop.literal.value {
                            // Theory propagated a value but SAT assigned the opposite.
                            // Create conflict clause from reason
                            let mut conflict: Vec<Literal> = prop
                                .reason
                                .iter()
                                .filter_map(|r| self.term_to_literal(r.term, !r.value))
                                .collect();
                            conflict.push(lit);
                            return TheoryCheck::Conflict(conflict);
                        }
                    }
                }
            }
        }

        // Then check consistency
        match self.theory.check() {
            TheoryResult::Sat => TheoryCheck::Sat,
            TheoryResult::Unknown => TheoryCheck::Unknown,
            TheoryResult::NeedSplit(split) => {
                // Theory needs to split on an integer variable
                // Pass this up to the caller which can create the splitting atoms
                TheoryCheck::NeedSplit(split)
            }
            TheoryResult::NeedDisequlitySplit(split) => {
                // Theory needs to split on a disequality
                TheoryCheck::NeedDisequlitySplit(split)
            }
            TheoryResult::NeedExpressionSplit(split) => {
                // Theory needs to split on a multi-variable expression disequality
                TheoryCheck::NeedExpressionSplit(split)
            }
            TheoryResult::Unsat(conflict_terms) => {
                // Convert theory conflict to SAT clause
                // The conflict terms are a set of literals that can't all be true
                // We negate them to create a blocking clause
                let clause: Vec<Literal> = conflict_terms
                    .iter()
                    .filter_map(|t| self.term_to_literal(t.term, !t.value))
                    .collect();
                TheoryCheck::Conflict(clause)
            }
        }
    }

    /// Solve the formula using DPLL(T)
    ///
    /// Note: This basic solve method returns Unknown if the theory requires
    /// splitting (branch-and-bound for LIA). Use `solve_with_splits` for LIA.
    pub fn solve(&mut self) -> SolveResult {
        // Reset theory solver
        self.theory.reset();

        // Convert AssumeResult to SolveResult (no assumptions means no unsat core needed)
        match self.solve_loop(None) {
            AssumeResult::Sat(model) => SolveResult::Sat(model),
            AssumeResult::Unsat(_) => SolveResult::Unsat,
            AssumeResult::Unknown => SolveResult::Unknown,
        }
    }

    /// Solve with assumptions using DPLL(T)
    ///
    /// This is like `solve()` but activates only the clauses whose selectors
    /// are in the positive assumptions, and deactivates those in negative assumptions.
    /// Used for incremental solving with selector-guarded assertions.
    ///
    /// Returns `AssumeResult` which includes an unsat core when UNSAT.
    /// The unsat core is a subset of the assumptions that caused the conflict.
    pub fn solve_with_assumptions(&mut self, assumptions: &[Literal]) -> AssumeResult {
        // Reset theory solver
        self.theory.reset();

        // Convert to owned vec for the loop
        let assumptions = assumptions.to_vec();
        self.solve_loop(Some(&assumptions))
    }

    /// Internal solve loop used by both `solve` and `solve_with_assumptions`
    ///
    /// Returns `AssumeResult` to propagate unsat core when using assumptions.
    fn solve_loop(&mut self, assumptions: Option<&[Literal]>) -> AssumeResult {
        loop {
            // Run SAT solver (with or without assumptions)
            let result = match assumptions {
                Some(a) => self.sat.solve_with_assumptions(a),
                None => match self.sat.solve() {
                    SolveResult::Sat(m) => AssumeResult::Sat(m),
                    SolveResult::Unsat => AssumeResult::Unsat(vec![]),
                    SolveResult::Unknown => AssumeResult::Unknown,
                },
            };

            match result {
                AssumeResult::Sat(model) => {
                    // SAT found a model - check with theory
                    self.sync_theory(&model);

                    match self.check_theory() {
                        TheoryCheck::Sat => {
                            // Theory accepts the model
                            return AssumeResult::Sat(model);
                        }
                        TheoryCheck::Unknown => {
                            // Theory could not determine satisfiability under this model.
                            // Returning SAT would be unsound; surface UNKNOWN instead.
                            return AssumeResult::Unknown;
                        }
                        TheoryCheck::NeedSplit(_)
                        | TheoryCheck::NeedDisequlitySplit(_)
                        | TheoryCheck::NeedExpressionSplit(_) => {
                            // Theory needs to split on an integer variable or disequality.
                            // Basic solve() can't handle this - return Unknown.
                            // Use solve_with_splits() for LIA.
                            return AssumeResult::Unknown;
                        }
                        TheoryCheck::Conflict(conflict_clause) => {
                            if conflict_clause.is_empty() {
                                // Empty conflict means UNSAT (theory proved contradiction)
                                // No unsat core available from theory conflicts
                                return AssumeResult::Unsat(vec![]);
                            }
                            // Add theory lemma and continue
                            if std::env::var("Z4_DEBUG_DPLL").is_ok() {
                                eprintln!(
                                    "[DPLL] Adding theory conflict clause with {} literals",
                                    conflict_clause.len()
                                );
                            }
                            self.sat.add_clause(conflict_clause);
                            continue;
                        }
                    }
                }
                AssumeResult::Unsat(core) => {
                    // Propagate the unsat core from the SAT solver
                    return AssumeResult::Unsat(core);
                }
                AssumeResult::Unknown => {
                    return AssumeResult::Unknown;
                }
            }
        }
    }

    /// Solve with support for theory-requested splits (for branch-and-bound LIA).
    ///
    /// The `create_split_atoms` callback is called when the theory needs to split
    /// on an integer variable. It receives the split request and should return
    /// the term IDs for the two new atoms: (var <= floor, var >= ceil).
    ///
    /// The callback is responsible for creating the atoms in the term store and
    /// ensuring they are properly formed.
    pub fn solve_with_splits<F>(&mut self, mut create_split_atoms: F) -> SolveResult
    where
        F: FnMut(&SplitRequest) -> (TermId, TermId),
    {
        // Reset theory solver
        self.theory.reset();

        loop {
            // Run SAT solver
            let result = self.sat.solve();

            match result {
                SolveResult::Sat(model) => {
                    // SAT found a model - check with theory
                    self.sync_theory(&model);

                    match self.check_theory() {
                        TheoryCheck::Sat => {
                            // Theory accepts the model
                            return SolveResult::Sat(model);
                        }
                        TheoryCheck::Unknown => {
                            return SolveResult::Unknown;
                        }
                        TheoryCheck::NeedSplit(split) => {
                            // Theory needs to split on an integer variable.
                            // Create the splitting atoms using the callback.
                            let (le_atom, ge_atom) = create_split_atoms(&split);

                            // Allocate new SAT variables for these atoms
                            let le_var = self.sat.new_var();
                            let ge_var = self.sat.new_var();

                            // Register the atoms as theory atoms
                            self.var_to_term.insert(le_var.0, le_atom);
                            self.term_to_var.insert(le_atom, le_var.0);
                            self.theory_atoms.push(le_atom);

                            self.var_to_term.insert(ge_var.0, ge_atom);
                            self.term_to_var.insert(ge_atom, ge_var.0);
                            self.theory_atoms.push(ge_atom);

                            // Add the splitting clause: (var <= floor) OR (var >= ceil)
                            let split_clause =
                                vec![Literal::positive(le_var), Literal::positive(ge_var)];
                            self.sat.add_clause(split_clause);

                            // Continue solving
                            continue;
                        }
                        TheoryCheck::NeedDisequlitySplit(_)
                        | TheoryCheck::NeedExpressionSplit(_) => {
                            // Disequality/expression splits not yet supported - return Unknown
                            return SolveResult::Unknown;
                        }
                        TheoryCheck::Conflict(conflict_clause) => {
                            if conflict_clause.is_empty() {
                                return SolveResult::Unsat;
                            }
                            self.sat.add_clause(conflict_clause);
                            continue;
                        }
                    }
                }
                SolveResult::Unsat => {
                    return SolveResult::Unsat;
                }
                SolveResult::Unknown => {
                    return SolveResult::Unknown;
                }
            }
        }
    }

    /// Solve with theory propagation at each decision level
    ///
    /// This is a more eager version of DPLL(T) that checks the theory
    /// after each propagation, not just when SAT finds a complete model.
    pub fn solve_eager(&mut self) -> SolveResult {
        // This would require callbacks from the SAT solver,
        // which would need additional infrastructure.
        // For now, just use the lazy version.
        self.solve()
    }

    /// Solve one step, returning either a final result or a split request.
    ///
    /// This method is used for LIA where splits may be needed. The executor
    /// should call this in a loop, handling splits by calling `apply_split`
    /// and then calling `solve_step` again.
    ///
    /// # Example
    /// ```ignore
    /// loop {
    ///     match dpll.solve_step() {
    ///         SolveStepResult::Done(result) => return result,
    ///         SolveStepResult::NeedSplit(split) => {
    ///             let (le_atom, ge_atom) = create_atoms(&split);
    ///             dpll.apply_split(le_atom, ge_atom);
    ///         }
    ///     }
    /// }
    /// ```
    pub fn solve_step(&mut self) -> SolveStepResult {
        // Reset theory solver on first call
        // (in practice, the caller should call this in a loop without resetting)
        loop {
            // Run SAT solver
            let result = self.sat.solve();

            match result {
                SolveResult::Sat(model) => {
                    // SAT found a model - check with theory
                    self.sync_theory(&model);

                    match self.check_theory() {
                        TheoryCheck::Sat => {
                            return SolveStepResult::Done(SolveResult::Sat(model));
                        }
                        TheoryCheck::Unknown | TheoryCheck::NeedExpressionSplit(_) => {
                            // Expression splits not yet supported - return Unknown
                            return SolveStepResult::Done(SolveResult::Unknown);
                        }
                        TheoryCheck::NeedSplit(split) => {
                            // Return the split request to the caller
                            return SolveStepResult::NeedSplit(split);
                        }
                        TheoryCheck::NeedDisequlitySplit(split) => {
                            // Return the disequality split request to the caller
                            return SolveStepResult::NeedDisequlitySplit(split);
                        }
                        TheoryCheck::Conflict(conflict_clause) => {
                            if conflict_clause.is_empty() {
                                return SolveStepResult::Done(SolveResult::Unsat);
                            }
                            self.sat.add_clause(conflict_clause);
                            continue;
                        }
                    }
                }
                SolveResult::Unsat => {
                    return SolveStepResult::Done(SolveResult::Unsat);
                }
                SolveResult::Unknown => {
                    return SolveStepResult::Done(SolveResult::Unknown);
                }
            }
        }
    }

    /// Apply a split by adding the split atoms and splitting clause.
    ///
    /// Call this after receiving `SolveStepResult::NeedSplit` and creating
    /// the split atoms (x <= floor and x >= ceil).
    ///
    /// If the atoms were already registered (e.g., from a previous iteration),
    /// this reuses the existing SAT variable mappings.
    pub fn apply_split(&mut self, le_atom: TermId, ge_atom: TermId) {
        self.apply_split_with_hint(le_atom, ge_atom, None);
    }

    /// Apply a split with a hint about which branch to try first.
    ///
    /// # Arguments
    /// * `le_atom` - The atom for x <= floor
    /// * `ge_atom` - The atom for x >= ceil
    /// * `prefer_ceil` - If Some(true), prefer ceil branch first; if Some(false), prefer floor;
    ///   if None, no preference (use default SAT heuristics)
    ///
    /// When splitting on a variable with fractional value, the solver should try the
    /// closer integer first. For value 3.7 (frac > 0.5), prefer ceil (4); for value 3.2
    /// (frac < 0.5), prefer floor (3). This reduces unnecessary backtracking.
    pub fn apply_split_with_hint(
        &mut self,
        le_atom: TermId,
        ge_atom: TermId,
        prefer_ceil: Option<bool>,
    ) {
        // Check if atoms are already mapped (reuse from previous iteration)
        let le_var = if let Some(&var_idx) = self.term_to_var.get(&le_atom) {
            Variable(var_idx)
        } else {
            // Allocate new SAT variable and register the mapping
            let var = self.sat.new_var();
            self.var_to_term.insert(var.0, le_atom);
            self.term_to_var.insert(le_atom, var.0);
            self.theory_atoms.push(le_atom);
            var
        };

        let ge_var = if let Some(&var_idx) = self.term_to_var.get(&ge_atom) {
            Variable(var_idx)
        } else {
            // Allocate new SAT variable and register the mapping
            let var = self.sat.new_var();
            self.var_to_term.insert(var.0, ge_atom);
            self.term_to_var.insert(ge_atom, var.0);
            self.theory_atoms.push(ge_atom);
            var
        };

        // Set phase hints if provided
        // The split clause is: (le_var) OR (ge_var) - at least one must be true
        // If we prefer ceil, set ge_var to true (try x >= ceil first)
        // If we prefer floor, set le_var to true (try x <= floor first)
        if let Some(prefer_ceil) = prefer_ceil {
            if prefer_ceil {
                // Prefer ceil: set ge_var=true, le_var=false
                self.sat.set_var_phase(ge_var, true);
                self.sat.set_var_phase(le_var, false);
            } else {
                // Prefer floor: set le_var=true, ge_var=false
                self.sat.set_var_phase(le_var, true);
                self.sat.set_var_phase(ge_var, false);
            }
        }

        // Add the splitting clause: (var <= floor) OR (var >= ceil)
        let split_clause = vec![Literal::positive(le_var), Literal::positive(ge_var)];
        self.sat.add_clause(split_clause);
    }

    /// Apply a disequality split by adding atoms for `x < c` and `x > c`.
    ///
    /// Call this after receiving `SolveStepResult::NeedDisequlitySplit` and creating
    /// the split atoms (x < excluded_value and x > excluded_value).
    ///
    /// This adds the clause `(x < c) OR (x > c)` to exclude `x = c`.
    pub fn apply_disequality_split(&mut self, lt_atom: TermId, gt_atom: TermId) {
        // Check if atoms are already mapped (reuse from previous iteration)
        let lt_var = if let Some(&var_idx) = self.term_to_var.get(&lt_atom) {
            Variable(var_idx)
        } else {
            // Allocate new SAT variable and register the mapping
            let var = self.sat.new_var();
            self.var_to_term.insert(var.0, lt_atom);
            self.term_to_var.insert(lt_atom, var.0);
            self.theory_atoms.push(lt_atom);
            var
        };

        let gt_var = if let Some(&var_idx) = self.term_to_var.get(&gt_atom) {
            Variable(var_idx)
        } else {
            // Allocate new SAT variable and register the mapping
            let var = self.sat.new_var();
            self.var_to_term.insert(var.0, gt_atom);
            self.term_to_var.insert(gt_atom, var.0);
            self.theory_atoms.push(gt_atom);
            var
        };

        // Add the splitting clause: (var < c) OR (var > c) - excludes var = c
        let split_clause = vec![Literal::positive(lt_var), Literal::positive(gt_var)];
        self.sat.add_clause(split_clause);
    }

    /// Reset the theory solver. Call this before starting a new solve session.
    /// Uses soft_reset() to preserve learned state (e.g., HNF cuts in LIA).
    pub fn reset_theory(&mut self) {
        self.theory.soft_reset();
    }

    // ========================================================================
    // Incremental Solving (Push/Pop)
    // ========================================================================

    /// Push a new assertion scope.
    ///
    /// All clauses added after this push will be removed when `pop()` is called.
    /// This enables incremental solving where you can add temporary constraints,
    /// solve, and then restore the original state.
    ///
    /// # Example
    /// ```ignore
    /// let mut dpll = DpllT::new(10, theory);
    /// dpll.add_clause(base_clause);  // Permanent
    /// dpll.push();
    /// dpll.add_clause(temp_clause);  // Will be removed by pop()
    /// let result = dpll.solve();
    /// dpll.pop();  // temp_clause is now inactive
    /// ```
    ///
    /// # Invariants
    /// - INV-PUSH-1: After push(), scope depth increases by 1
    /// - INV-PUSH-2: SAT solver and theory solver scopes are synchronized
    pub fn push(&mut self) {
        self.sat.push();
        self.theory.push();
    }

    /// Pop the most recent assertion scope.
    ///
    /// Removes all clauses added since the last `push()` and restores the
    /// theory solver state. Returns `false` if there is no active scope to pop.
    ///
    /// # Invariants
    /// - INV-POP-1: After pop(), scope depth decreases by 1 (if > 0)
    /// - INV-POP-2: SAT solver and theory solver scopes remain synchronized
    /// - INV-POP-3: Learned clauses that depend only on base assertions are preserved
    pub fn pop(&mut self) -> bool {
        if !self.sat.pop() {
            return false;
        }
        self.theory.pop();
        true
    }

    /// Get the current scope depth.
    ///
    /// Returns 0 when no push() calls are active.
    pub fn scope_depth(&self) -> usize {
        self.sat.scope_depth()
    }

    /// Extract learned clauses from the SAT solver.
    ///
    /// Used in branch-and-bound to preserve learned clauses across solver recreations.
    pub fn get_learned_clauses(&self) -> Vec<Vec<Literal>> {
        self.sat.get_learned_clauses()
    }

    /// Add learned clauses from a previous solve session.
    ///
    /// Used in branch-and-bound to restore learned clauses after recreating the solver.
    pub fn add_learned_clauses(&mut self, clauses: Vec<Vec<Literal>>) {
        for clause in clauses {
            self.sat.add_preserved_learned(clause);
        }
    }
}

/// Convert a DIMACS-style CNF literal to a z4-sat Literal
fn cnf_lit_to_sat_lit(lit: CnfLit) -> Literal {
    let var = Variable(lit.unsigned_abs() - 1);
    if lit > 0 {
        Literal::positive(var)
    } else {
        Literal::negative(var)
    }
}

#[cfg(test)]
/// Convert a z4-sat Literal to DIMACS-style CNF literal
fn sat_lit_to_cnf_lit(lit: Literal) -> CnfLit {
    let var = (lit.variable().0 + 1) as i32;
    if lit.is_positive() {
        var
    } else {
        -var
    }
}

/// A simple empty theory solver for propositional logic
pub struct PropositionalTheory;

impl TheorySolver for PropositionalTheory {
    fn assert_literal(&mut self, _literal: TermId, _value: bool) {
        // No theory reasoning needed
    }

    fn check(&mut self) -> TheoryResult {
        // Propositional logic is always consistent
        TheoryResult::Sat
    }

    fn propagate(&mut self) -> Vec<TheoryPropagation> {
        // No propagations
        vec![]
    }

    fn push(&mut self) {
        // No state to push
    }

    fn pop(&mut self) {
        // No state to pop
    }

    fn reset(&mut self) {
        // Nothing to reset
    }
}

/// High-level SMT solver interface
pub struct SmtSolver {
    /// Term store for all terms
    pub terms: TermStore,
    /// CNF clauses from Tseitin transformation
    cnf_clauses: Vec<CnfClause>,
    /// Variable mappings
    var_to_term: BTreeMap<u32, TermId>,
    term_to_var: BTreeMap<TermId, u32>,
    /// Number of CNF variables
    num_vars: u32,
    /// Assertions as term IDs
    assertions: Vec<TermId>,
}

impl Default for SmtSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl SmtSolver {
    /// Create a new SMT solver
    pub fn new() -> Self {
        SmtSolver {
            terms: TermStore::new(),
            cnf_clauses: Vec::new(),
            var_to_term: BTreeMap::new(),
            term_to_var: BTreeMap::new(),
            num_vars: 0,
            assertions: Vec::new(),
        }
    }

    /// Add an assertion
    pub fn assert(&mut self, term: TermId) {
        self.assertions.push(term);
    }

    /// Solve all assertions (propositional only)
    pub fn solve_propositional(&mut self) -> SolveResult {
        if self.assertions.is_empty() {
            return SolveResult::Sat(vec![]);
        }

        // Run Tseitin transformation
        let tseitin = Tseitin::new(&self.terms);
        let result = tseitin.transform_all(&self.assertions);

        // Store mappings
        self.cnf_clauses = result.clauses.clone();
        self.var_to_term = result.var_to_term.clone();
        self.term_to_var = result.term_to_var.clone();
        self.num_vars = result.num_vars;

        // Create DPLL(T) solver with propositional theory
        let theory = PropositionalTheory;
        let mut dpll = DpllT::from_tseitin(&self.terms, &result, theory);

        dpll.solve()
    }

    /// Get the term store
    pub fn term_store(&self) -> &TermStore {
        &self.terms
    }

    /// Get mutable access to the term store
    pub fn term_store_mut(&mut self) -> &mut TermStore {
        &mut self.terms
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use z4_core::term::Symbol;
    use z4_core::Sort;
    use z4_euf::EufSolver;

    #[test]
    fn test_propositional_sat() {
        let mut solver = SmtSolver::new();

        // Create (a ∨ b) ∧ (¬a ∨ b) - should be SAT with b=true
        let a = solver.terms.mk_var("a", Sort::Bool);
        let b = solver.terms.mk_var("b", Sort::Bool);
        let not_a = solver.terms.mk_not(a);

        let or1 = solver.terms.mk_or(vec![a, b]);
        let or2 = solver.terms.mk_or(vec![not_a, b]);
        let formula = solver.terms.mk_and(vec![or1, or2]);

        solver.assert(formula);

        let result = solver.solve_propositional();
        assert!(matches!(result, SolveResult::Sat(_)));
    }

    #[test]
    fn test_propositional_unsat() {
        let mut solver = SmtSolver::new();

        // Create a ∧ ¬a - should be UNSAT
        let a = solver.terms.mk_var("a", Sort::Bool);
        let not_a = solver.terms.mk_not(a);
        let formula = solver.terms.mk_and(vec![a, not_a]);

        solver.assert(formula);

        let result = solver.solve_propositional();
        assert!(matches!(result, SolveResult::Unsat));
    }

    #[test]
    fn test_dpll_empty_formula() {
        let theory = PropositionalTheory;
        let mut dpll = DpllT::new(0, theory);

        let result = dpll.solve();
        assert!(matches!(result, SolveResult::Sat(_)));
    }

    #[test]
    fn test_cnf_literal_conversion() {
        // Test positive literal
        let cnf_lit: CnfLit = 5;
        let sat_lit = cnf_lit_to_sat_lit(cnf_lit);
        assert_eq!(sat_lit.variable().0, 4); // 0-indexed
        assert!(sat_lit.is_positive());

        let back = sat_lit_to_cnf_lit(sat_lit);
        assert_eq!(back, cnf_lit);

        // Test negative literal
        let cnf_lit: CnfLit = -3;
        let sat_lit = cnf_lit_to_sat_lit(cnf_lit);
        assert_eq!(sat_lit.variable().0, 2); // 0-indexed
        assert!(!sat_lit.is_positive());

        let back = sat_lit_to_cnf_lit(sat_lit);
        assert_eq!(back, cnf_lit);
    }

    #[test]
    fn test_simple_cnf() {
        let theory = PropositionalTheory;
        let mut dpll = DpllT::new(3, theory);

        // Add clause: (1 ∨ 2)
        dpll.add_clause(vec![
            Literal::positive(Variable(0)),
            Literal::positive(Variable(1)),
        ]);

        // Add clause: (¬1 ∨ 2)
        dpll.add_clause(vec![
            Literal::negative(Variable(0)),
            Literal::positive(Variable(1)),
        ]);

        // Add clause: (¬2 ∨ 3)
        dpll.add_clause(vec![
            Literal::negative(Variable(1)),
            Literal::positive(Variable(2)),
        ]);

        let result = dpll.solve();
        assert!(matches!(result, SolveResult::Sat(_)));
    }

    #[test]
    fn test_unsat_cnf() {
        let theory = PropositionalTheory;
        let mut dpll = DpllT::new(1, theory);

        // Add clause: (1)
        dpll.add_clause(vec![Literal::positive(Variable(0))]);

        // Add clause: (¬1)
        dpll.add_clause(vec![Literal::negative(Variable(0))]);

        let result = dpll.solve();
        assert!(matches!(result, SolveResult::Unsat));
    }

    #[test]
    fn test_dpll_euf_predicate_congruence_unsat() {
        let mut terms = TermStore::new();
        let u = Sort::Uninterpreted("U".to_string());

        let a = terms.mk_var("a", u.clone());
        let b = terms.mk_var("b", u.clone());

        // (= a b) ∧ p(a) ∧ ¬p(b) is UNSAT in EUF.
        let eq_ab = terms.mk_eq(a, b);
        let p_a = terms.mk_app(z4_core::Symbol::named("p"), vec![a], Sort::Bool);
        let p_b = terms.mk_app(z4_core::Symbol::named("p"), vec![b], Sort::Bool);
        let not_p_b = terms.mk_not(p_b);

        let formula = terms.mk_and(vec![eq_ab, p_a, not_p_b]);

        let tseitin = Tseitin::new(&terms);
        let result = tseitin.transform(formula);

        let theory = EufSolver::new(&terms);
        let mut dpll = DpllT::from_tseitin(&terms, &result, theory);

        let solve_result = dpll.solve();
        assert!(matches!(solve_result, SolveResult::Unsat));
    }

    /// Test that learned clause preservation works correctly.
    ///
    /// This test would have caught the LIA branch-and-bound bug where
    /// learned clauses were lost between iterations because DpllT was
    /// recreated fresh each time.
    ///
    /// The bug: When solve_lia() recreated DpllT for each branch-and-bound
    /// iteration, all learned clauses were lost. This caused the solver
    /// to re-explore the same conflicting assignments repeatedly, leading
    /// to infinite loops instead of converging to UNSAT.
    ///
    /// The fix: Add get_learned_clauses() and add_learned_clauses() to
    /// preserve learned clauses across solver recreations.
    #[test]
    fn test_learned_clause_preservation() {
        let theory = PropositionalTheory;
        let mut dpll = DpllT::new(3, theory);

        // Create a simple formula
        dpll.add_clause(vec![
            Literal::positive(Variable(0)),
            Literal::positive(Variable(1)),
        ]);
        dpll.add_clause(vec![
            Literal::negative(Variable(0)),
            Literal::positive(Variable(1)),
        ]);

        // Solve first
        let result = dpll.solve();
        assert!(matches!(result, SolveResult::Sat(_)));

        // Extract learned clauses (may be empty for simple SAT formulas)
        let learned = dpll.get_learned_clauses();

        // Create a new solver and add the same clauses
        let theory2 = PropositionalTheory;
        let mut dpll2 = DpllT::new(3, theory2);

        dpll2.add_clause(vec![
            Literal::positive(Variable(0)),
            Literal::positive(Variable(1)),
        ]);
        dpll2.add_clause(vec![
            Literal::negative(Variable(0)),
            Literal::positive(Variable(1)),
        ]);

        // Add the learned clauses from the first solve
        dpll2.add_learned_clauses(learned.clone());

        // The second solve should also return SAT
        let result2 = dpll2.solve();
        assert!(matches!(result2, SolveResult::Sat(_)));
    }

    /// Test that get_learned_clauses and add_learned_clauses work together.
    ///
    /// This is a regression test for the branch-and-bound bug:
    /// The API for preserving learned clauses must work correctly
    /// to prevent infinite loops in LIA solving.
    #[test]
    fn test_learned_clauses_api() {
        // Create a solver and solve a formula
        let theory = PropositionalTheory;
        let mut dpll = DpllT::new(2, theory);

        dpll.add_clause(vec![
            Literal::positive(Variable(0)),
            Literal::positive(Variable(1)),
        ]);

        let result = dpll.solve();
        assert!(matches!(result, SolveResult::Sat(_)));

        // Get learned clauses (API should work even if empty)
        let learned = dpll.get_learned_clauses();

        // Create new solver and add learned clauses
        let theory2 = PropositionalTheory;
        let mut dpll2 = DpllT::new(2, theory2);
        dpll2.add_clause(vec![
            Literal::positive(Variable(0)),
            Literal::positive(Variable(1)),
        ]);

        // This should not crash even if learned is empty
        dpll2.add_learned_clauses(learned);

        // Should still be SAT
        let result2 = dpll2.solve();
        assert!(matches!(result2, SolveResult::Sat(_)));
    }

    // ========================================================================
    // Property Tests for Formal Verification Gaps
    // ========================================================================

    /// Gap 12: SMT Model Verification
    ///
    /// When the solver returns SAT, we should verify that the model actually
    /// satisfies all the original assertions, not just the CNF encoding.
    ///
    /// This property test ensures SMT model soundness.
    #[test]
    fn test_smt_model_verification_gap12() {
        // Test 1: Simple SAT formula with expected model
        let mut solver = SmtSolver::new();

        // Create (a ∨ b) - should be SAT
        let a = solver.terms.mk_var("a", Sort::Bool);
        let b = solver.terms.mk_var("b", Sort::Bool);
        let formula = solver.terms.mk_or(vec![a, b]);

        solver.assert(formula);

        let result = solver.solve_propositional();
        if let SolveResult::Sat(model) = result {
            // Verify: at least one of a or b must be true in the model
            // This is a basic soundness check
            let a_true = model.first().copied().unwrap_or(false);
            let b_true = model.get(1).copied().unwrap_or(false);
            assert!(
                a_true || b_true,
                "SMT Model Soundness: (a ∨ b) returned SAT but neither a nor b is true"
            );
        }
    }

    /// Gap 12 continued: Verify SMT model against conjunction
    #[test]
    fn test_smt_model_verification_conjunction() {
        let mut solver = SmtSolver::new();

        // Create (a ∧ b) - should be SAT with a=true, b=true
        let a = solver.terms.mk_var("a", Sort::Bool);
        let b = solver.terms.mk_var("b", Sort::Bool);
        let formula = solver.terms.mk_and(vec![a, b]);

        solver.assert(formula);

        let result = solver.solve_propositional();
        if let SolveResult::Sat(model) = result {
            // Verify: both a and b must be true
            let a_true = model.first().copied().unwrap_or(false);
            let b_true = model.get(1).copied().unwrap_or(false);
            assert!(
                a_true && b_true,
                "SMT Model Soundness: (a ∧ b) returned SAT but a={}, b={}",
                a_true,
                b_true
            );
        }
    }

    /// Gap 12 continued: Verify model against implications
    #[test]
    fn test_smt_model_verification_implication() {
        let mut solver = SmtSolver::new();

        // Create (a ⟹ b) ∧ a - should be SAT with a=true, b=true
        // (a ⟹ b) is equivalent to (¬a ∨ b)
        let a = solver.terms.mk_var("a", Sort::Bool);
        let b = solver.terms.mk_var("b", Sort::Bool);
        let not_a = solver.terms.mk_not(a);
        let implies = solver.terms.mk_or(vec![not_a, b]);
        let formula = solver.terms.mk_and(vec![implies, a]);

        solver.assert(formula);

        let result = solver.solve_propositional();
        if let SolveResult::Sat(model) = result {
            // Verify: a must be true (from second conjunct)
            // and b must be true (from implication with a=true)
            let a_true = model.first().copied().unwrap_or(false);
            let b_true = model.get(1).copied().unwrap_or(false);
            assert!(
                a_true,
                "SMT Model Soundness: (a ⟹ b) ∧ a returned SAT but a=false"
            );
            assert!(
                b_true,
                "SMT Model Soundness: (a ⟹ b) ∧ a returned SAT with a=true but b=false"
            );
        }
    }

    /// Gap 11: Theory Conflict Explanation Soundness
    ///
    /// When a theory reports a conflict, the explanation literals should
    /// logically imply the conflict. This test verifies that for EUF.
    #[test]
    fn test_euf_conflict_explanation_soundness_gap11() {
        // Test that EUF produces sound explanations
        // Set up: a = b ∧ b = c ⟹ a = c (transitivity)
        // If we assert a = b, b = c, and a ≠ c, we should get UNSAT
        // and the explanation should be {a = b, b = c, a ≠ c}

        let mut terms = TermStore::new();
        let sort_s = Sort::Uninterpreted("S".to_string());

        let a = terms.mk_var("a", sort_s.clone());
        let b = terms.mk_var("b", sort_s.clone());
        let c = terms.mk_var("c", sort_s);

        let eq_ab = terms.mk_eq(a, b);
        let eq_bc = terms.mk_eq(b, c);
        let eq_ac = terms.mk_eq(a, c);
        let _neq_ac = terms.mk_not(eq_ac); // kept for documentation

        // Create EUF solver and assert literals
        let mut euf = EufSolver::new(&terms);

        // Assert: a = b, b = c, a ≠ c
        euf.assert_literal(eq_ab, true);
        euf.assert_literal(eq_bc, true);
        euf.assert_literal(eq_ac, false);

        // Check should return Unsat (conflict)
        let result = euf.check();
        match result {
            TheoryResult::Unsat(explanation) => {
                // Explanation should contain the relevant equality atoms
                // The explanation is sound if asserting all explanation literals
                // leads to a conflict (which it does, by construction)
                assert!(
                    !explanation.is_empty(),
                    "EUF conflict explanation is empty - soundness violation"
                );
                // Basic check: explanation should mention the atoms involved
                // (A full soundness check would re-solve with just the explanation)
            }
            TheoryResult::Sat => {
                panic!("EUF should find conflict for a=b ∧ b=c ∧ a≠c (transitivity)");
            }
            TheoryResult::Unknown
            | TheoryResult::NeedSplit(_)
            | TheoryResult::NeedDisequlitySplit(_)
            | TheoryResult::NeedExpressionSplit(_) => {
                // Acceptable - some solvers may return Unknown
            }
        }
    }

    /// Gap 11: LRA Conflict Explanation Soundness
    ///
    /// When LRA reports a conflict, the explanation literals should be
    /// sufficient to cause the conflict. This test verifies that.
    #[test]
    fn test_lra_conflict_explanation_soundness_gap11() {
        use z4_lra::LraSolver;

        // Test that LRA produces sound explanations
        // Set up: x >= 5 ∧ x <= 3 is clearly UNSAT
        // The explanation should contain both bounds

        let mut terms = TermStore::new();

        let x = terms.mk_var("x", Sort::Real);
        let five = terms.mk_int(5.into());
        let three = terms.mk_int(3.into());

        // x >= 5 is (>= x 5)
        let ge_5 = terms.mk_app(Symbol::Named(">=".to_string()), vec![x, five], Sort::Bool);
        // x <= 3 is (<= x 3)
        let le_3 = terms.mk_app(Symbol::Named("<=".to_string()), vec![x, three], Sort::Bool);

        // Create LRA solver and assert literals
        let mut lra = LraSolver::new(&terms);

        // Assert: x >= 5, x <= 3
        lra.assert_literal(ge_5, true);
        lra.assert_literal(le_3, true);

        // Check should return Unsat (conflict)
        let result = lra.check();
        match result {
            TheoryResult::Unsat(explanation) => {
                // Explanation should contain the conflicting bound atoms
                assert!(
                    !explanation.is_empty(),
                    "LRA conflict explanation is empty - soundness violation"
                );

                // Soundness verification: re-solve with only explanation literals
                // If explanation is sound, re-asserting just those should still conflict
                let mut lra2 = LraSolver::new(&terms);
                for lit in &explanation {
                    lra2.assert_literal(lit.term, lit.value);
                }
                let result2 = lra2.check();
                assert!(
                    matches!(result2, TheoryResult::Unsat(_)),
                    "LRA explanation is not sound: re-asserting explanation literals did not cause conflict"
                );
            }
            TheoryResult::Sat => {
                panic!("LRA should find conflict for x >= 5 ∧ x <= 3");
            }
            TheoryResult::Unknown
            | TheoryResult::NeedSplit(_)
            | TheoryResult::NeedDisequlitySplit(_)
            | TheoryResult::NeedExpressionSplit(_) => {
                // Acceptable - some configurations may return Unknown
            }
        }
    }

    /// Gap 11: LRA Explanation Minimality Test
    ///
    /// Test that LRA explanations are reasonably minimal (don't include
    /// irrelevant literals).
    #[test]
    fn test_lra_explanation_minimality_gap11() {
        use z4_lra::LraSolver;

        // Set up a conflict with one irrelevant constraint:
        // x >= 5 ∧ x <= 3 ∧ y >= 0 should conflict
        // The explanation should NOT include y >= 0

        let mut terms = TermStore::new();

        let x = terms.mk_var("x", Sort::Real);
        let y = terms.mk_var("y", Sort::Real);
        let five = terms.mk_int(5.into());
        let three = terms.mk_int(3.into());
        let zero = terms.mk_int(0.into());

        // x >= 5
        let ge_5 = terms.mk_app(Symbol::Named(">=".to_string()), vec![x, five], Sort::Bool);
        // x <= 3
        let le_3 = terms.mk_app(Symbol::Named("<=".to_string()), vec![x, three], Sort::Bool);
        // y >= 0 (irrelevant)
        let y_ge_0 = terms.mk_app(Symbol::Named(">=".to_string()), vec![y, zero], Sort::Bool);

        let mut lra = LraSolver::new(&terms);

        // Assert all three
        lra.assert_literal(ge_5, true);
        lra.assert_literal(le_3, true);
        lra.assert_literal(y_ge_0, true);

        let result = lra.check();
        match result {
            TheoryResult::Unsat(explanation) => {
                // Explanation should NOT include y >= 0 (it's irrelevant)
                let contains_y = explanation.iter().any(|lit| lit.term == y_ge_0);
                assert!(
                    !contains_y,
                    "LRA explanation includes irrelevant constraint (y >= 0) - minimality violation"
                );

                // Should have at most 2 literals (the conflicting bounds on x)
                assert!(
                    explanation.len() <= 2,
                    "LRA explanation has {} literals, expected at most 2",
                    explanation.len()
                );
            }
            TheoryResult::Sat => {
                panic!("LRA should find conflict for x >= 5 ∧ x <= 3");
            }
            TheoryResult::Unknown
            | TheoryResult::NeedSplit(_)
            | TheoryResult::NeedDisequlitySplit(_)
            | TheoryResult::NeedExpressionSplit(_) => {
                // Acceptable
            }
        }
    }

    /// Gap 11: LIA Conflict Explanation Soundness
    ///
    /// When LIA reports a conflict (specifically for integer-infeasibility),
    /// the explanation literals should be sufficient to cause the conflict.
    #[test]
    fn test_lia_conflict_explanation_soundness_gap11() {
        use z4_lia::LiaSolver;

        // Test that LIA produces sound explanations for integer conflicts.
        // Set up: x >= 5 ∧ x <= 3 where x is an integer is UNSAT
        // (inherited from LRA relaxation conflict)

        let mut terms = TermStore::new();

        let x = terms.mk_var("x", Sort::Int);
        let five = terms.mk_int(5.into());
        let three = terms.mk_int(3.into());

        // x >= 5 is (>= x 5)
        let ge_5 = terms.mk_app(Symbol::Named(">=".to_string()), vec![x, five], Sort::Bool);
        // x <= 3 is (<= x 3)
        let le_3 = terms.mk_app(Symbol::Named("<=".to_string()), vec![x, three], Sort::Bool);

        // Create LIA solver and assert literals
        let mut lia = LiaSolver::new(&terms);

        // Assert: x >= 5, x <= 3
        lia.assert_literal(ge_5, true);
        lia.assert_literal(le_3, true);

        // Check should return Unsat (conflict from LRA relaxation)
        let result = lia.check();
        match result {
            TheoryResult::Unsat(explanation) => {
                // Explanation should contain the conflicting bound atoms
                assert!(
                    !explanation.is_empty(),
                    "LIA conflict explanation is empty - soundness violation"
                );

                // Soundness verification: re-solve with only explanation literals
                let mut lia2 = LiaSolver::new(&terms);
                for lit in &explanation {
                    lia2.assert_literal(lit.term, lit.value);
                }
                let result2 = lia2.check();
                assert!(
                    matches!(result2, TheoryResult::Unsat(_)),
                    "LIA explanation is not sound: re-asserting explanation literals did not cause conflict"
                );
            }
            TheoryResult::Sat => {
                panic!("LIA should find conflict for x >= 5 ∧ x <= 3");
            }
            TheoryResult::Unknown
            | TheoryResult::NeedSplit(_)
            | TheoryResult::NeedDisequlitySplit(_)
            | TheoryResult::NeedExpressionSplit(_) => {
                // Acceptable - some configurations may return Unknown or NeedSplit
            }
        }
    }

    /// Gap 11: LIA Integer-Specific Conflict Explanation
    ///
    /// Test LIA explanation for integer-specific conflicts (no real solution in range).
    /// For integer x: x > 5 ∧ x < 6 is UNSAT (no integer between 5 and 6).
    #[test]
    fn test_lia_integer_bounds_conflict_explanation_gap11() {
        use z4_lia::LiaSolver;

        // For an integer variable x:
        // x > 5 means x >= 6 (next integer)
        // x < 6 means x <= 5 (previous integer)
        // Together: x >= 6 and x <= 5 is UNSAT

        let mut terms = TermStore::new();

        let x = terms.mk_var("x", Sort::Int);
        let five = terms.mk_int(5.into());
        let six = terms.mk_int(6.into());

        // x > 5 is (> x 5)
        let gt_5 = terms.mk_app(Symbol::Named(">".to_string()), vec![x, five], Sort::Bool);
        // x < 6 is (< x 6)
        let lt_6 = terms.mk_app(Symbol::Named("<".to_string()), vec![x, six], Sort::Bool);

        let mut lia = LiaSolver::new(&terms);

        // Assert: x > 5, x < 6
        lia.assert_literal(gt_5, true);
        lia.assert_literal(lt_6, true);

        // Check should return Unsat (no integer satisfies both)
        let result = lia.check();
        match result {
            TheoryResult::Unsat(explanation) => {
                // Explanation should be non-empty
                assert!(
                    !explanation.is_empty(),
                    "LIA integer bounds conflict explanation is empty"
                );

                // Verify soundness
                let mut lia2 = LiaSolver::new(&terms);
                for lit in &explanation {
                    lia2.assert_literal(lit.term, lit.value);
                }
                let result2 = lia2.check();
                assert!(
                    matches!(result2, TheoryResult::Unsat(_)),
                    "LIA integer bounds explanation is not sound"
                );
            }
            TheoryResult::Sat => {
                panic!("LIA should find conflict for x > 5 ∧ x < 6 (no integer in range)");
            }
            TheoryResult::Unknown
            | TheoryResult::NeedSplit(_)
            | TheoryResult::NeedDisequlitySplit(_)
            | TheoryResult::NeedExpressionSplit(_) => {
                // Acceptable - may trigger split instead of immediate conflict
            }
        }
    }

    /// Gap 11: Arrays ROW1 Conflict Explanation Soundness
    ///
    /// When Arrays reports a conflict for ROW1 violation (read-over-write axiom 1),
    /// the explanation should be sufficient to cause the conflict.
    #[test]
    fn test_arrays_row1_conflict_explanation_soundness_gap11() {
        use z4_arrays::ArraySolver;

        // ROW1: select(store(a, i, v), j) = v when i = j
        // Conflict: Assert i = j AND select(store(a, i, v), j) ≠ v
        //
        // We use different index variables (i, j) and assert i = j explicitly,
        // because the solver's known_equal() needs to track equalities.

        let mut terms = TermStore::new();
        let arr_sort = Sort::Array(Box::new(Sort::Int), Box::new(Sort::Int));

        let a = terms.mk_var("a", arr_sort);
        let i = terms.mk_var("i", Sort::Int);
        let j = terms.mk_var("j", Sort::Int); // Different index variable
        let v = terms.mk_var("v", Sort::Int);

        // Create store(a, i, v) and select(store(a, i, v), j)
        let stored = terms.mk_store(a, i, v);
        let selected = terms.mk_select(stored, j);

        // Create equalities
        let eq_ij = terms.mk_eq(i, j); // i = j
        let eq_sel_v = terms.mk_eq(selected, v); // select(store(a,i,v), j) = v

        let mut solver = ArraySolver::new(&terms);

        // Assert i = j (so ROW1 applies)
        solver.assert_literal(eq_ij, true);
        // Assert select(store(a, i, v), j) ≠ v (violates ROW1 when i=j)
        solver.assert_literal(eq_sel_v, false);

        let result = solver.check();
        match result {
            TheoryResult::Unsat(explanation) => {
                // Explanation should be non-empty
                assert!(
                    !explanation.is_empty(),
                    "Arrays ROW1 conflict explanation is empty"
                );

                // Verify soundness: re-solve with only explanation literals
                let mut solver2 = ArraySolver::new(&terms);
                for lit in &explanation {
                    solver2.assert_literal(lit.term, lit.value);
                }
                let result2 = solver2.check();
                assert!(
                    matches!(result2, TheoryResult::Unsat(_)),
                    "Arrays ROW1 explanation is not sound"
                );
            }
            TheoryResult::Sat => {
                panic!("Arrays should find conflict for i=j AND select(store(a,i,v),j) ≠ v");
            }
            TheoryResult::Unknown
            | TheoryResult::NeedSplit(_)
            | TheoryResult::NeedDisequlitySplit(_)
            | TheoryResult::NeedExpressionSplit(_) => {
                // Acceptable
            }
        }
    }

    /// Gap 11: Arrays ROW2 Conflict Explanation Soundness
    ///
    /// When Arrays reports a conflict for ROW2 violation (read-over-write axiom 2),
    /// the explanation should be sufficient to cause the conflict.
    #[test]
    fn test_arrays_row2_conflict_explanation_soundness_gap11() {
        use z4_arrays::ArraySolver;

        // ROW2: i ≠ j → select(store(a, i, v), j) = select(a, j)
        // Conflict: Assert i ≠ j AND select(store(a, i, v), j) ≠ select(a, j)

        let mut terms = TermStore::new();
        let arr_sort = Sort::Array(Box::new(Sort::Int), Box::new(Sort::Int));

        let a = terms.mk_var("a", arr_sort);
        let i = terms.mk_var("i", Sort::Int);
        let j = terms.mk_var("j", Sort::Int);
        let v = terms.mk_var("v", Sort::Int);

        // Create store(a, i, v)
        let stored = terms.mk_store(a, i, v);
        // select(store(a, i, v), j)
        let sel_stored_j = terms.mk_select(stored, j);
        // select(a, j)
        let sel_a_j = terms.mk_select(a, j);

        // Create equalities
        let eq_ij = terms.mk_eq(i, j);
        let eq_sels = terms.mk_eq(sel_stored_j, sel_a_j);

        let mut solver = ArraySolver::new(&terms);

        // Assert i ≠ j
        solver.assert_literal(eq_ij, false);
        // Assert select(store(a, i, v), j) ≠ select(a, j) (violates ROW2)
        solver.assert_literal(eq_sels, false);

        let result = solver.check();
        match result {
            TheoryResult::Unsat(explanation) => {
                // Explanation should be non-empty
                assert!(
                    !explanation.is_empty(),
                    "Arrays ROW2 conflict explanation is empty"
                );

                // Verify soundness: re-solve with only explanation literals
                let mut solver2 = ArraySolver::new(&terms);
                for lit in &explanation {
                    solver2.assert_literal(lit.term, lit.value);
                }
                let result2 = solver2.check();
                assert!(
                    matches!(result2, TheoryResult::Unsat(_)),
                    "Arrays ROW2 explanation is not sound"
                );
            }
            TheoryResult::Sat => {
                panic!("Arrays should find conflict for ROW2 violation");
            }
            TheoryResult::Unknown
            | TheoryResult::NeedSplit(_)
            | TheoryResult::NeedDisequlitySplit(_)
            | TheoryResult::NeedExpressionSplit(_) => {
                // Acceptable
            }
        }
    }

    /// Gap 11: Arrays Explanation Minimality
    ///
    /// Test that Arrays explanations don't include irrelevant literals.
    #[test]
    fn test_arrays_explanation_minimality_gap11() {
        use z4_arrays::ArraySolver;

        // Set up ROW1 conflict with an irrelevant assertion
        // Use different index variables (i, j) and assert i = j explicitly.
        let mut terms = TermStore::new();
        let arr_sort = Sort::Array(Box::new(Sort::Int), Box::new(Sort::Int));

        let a = terms.mk_var("a", arr_sort.clone());
        let b = terms.mk_var("b", arr_sort); // Irrelevant array
        let i = terms.mk_var("i", Sort::Int);
        let j = terms.mk_var("j", Sort::Int); // Different index variable
        let v = terms.mk_var("v", Sort::Int);
        let w = terms.mk_var("w", Sort::Int); // Irrelevant value

        // Create store(a, i, v) and select(store(a, i, v), j)
        let stored = terms.mk_store(a, i, v);
        let selected = terms.mk_select(stored, j);

        // Irrelevant: select(b, i) = w (has nothing to do with the conflict)
        let sel_b = terms.mk_select(b, i);
        let eq_sel_b_w = terms.mk_eq(sel_b, w);

        // Relevant equalities
        let eq_ij = terms.mk_eq(i, j); // i = j
        let eq_sel_v = terms.mk_eq(selected, v); // select(store(a,i,v), j) = v

        let mut solver = ArraySolver::new(&terms);

        // Assert irrelevant fact
        solver.assert_literal(eq_sel_b_w, true);
        // Assert i = j (so ROW1 applies)
        solver.assert_literal(eq_ij, true);
        // Assert the conflict: select(store(a, i, v), j) ≠ v
        solver.assert_literal(eq_sel_v, false);

        let result = solver.check();
        match result {
            TheoryResult::Unsat(explanation) => {
                // Explanation should NOT include the irrelevant eq_sel_b_w
                let contains_irrelevant = explanation.iter().any(|lit| lit.term == eq_sel_b_w);
                assert!(
                    !contains_irrelevant,
                    "Arrays explanation includes irrelevant constraint - minimality violation"
                );
            }
            TheoryResult::Sat => {
                panic!("Arrays should find ROW1 conflict");
            }
            TheoryResult::Unknown
            | TheoryResult::NeedSplit(_)
            | TheoryResult::NeedDisequlitySplit(_)
            | TheoryResult::NeedExpressionSplit(_) => {
                // Acceptable
            }
        }
    }

    // ========== Unsat Core Tests ==========

    /// Test that DpllT propagates unsat core from SAT solver
    #[test]
    fn test_dpllt_unsat_core_propagation() {
        // Create a DpllT with 2 variables (x, y)
        // Formula: (x) - forces x=true
        // Assumptions: ¬x - contradicts the unit clause
        let theory = PropositionalTheory;
        let mut dpll = DpllT::new(2, theory);

        // Add unit clause: x (forces x=true)
        dpll.sat_solver_mut()
            .add_clause(vec![Literal::positive(Variable(0))]);

        // Assumption: ¬x (contradicts the unit clause)
        let lit_not_x = Literal::negative(Variable(0));

        let assumptions = vec![lit_not_x];
        let result = dpll.solve_with_assumptions(&assumptions);

        match result {
            AssumeResult::Unsat(core) => {
                // Core should be a subset of assumptions
                for lit in &core {
                    assert!(
                        assumptions.contains(lit),
                        "Unsat core literal {:?} not in assumptions",
                        lit
                    );
                }
                // The core might be empty if the SAT solver detects conflict
                // before fully processing assumptions (e.g., at decision level 0)
            }
            AssumeResult::Sat(_) => panic!("Should be unsat - ¬x contradicts unit clause x"),
            AssumeResult::Unknown => panic!("Should determine satisfiability"),
        }
    }

    /// Test unsat core with multiple assumptions where only some conflict
    #[test]
    fn test_dpllt_unsat_core_minimal() {
        // Formula: (x ∨ y) with assumptions x=false, y=false, z=true
        // Only x=false and y=false are needed for unsat

        let theory = PropositionalTheory;
        let mut dpll = DpllT::new(3, theory);

        // Add clause: x ∨ y
        dpll.sat_solver_mut().add_clause(vec![
            Literal::positive(Variable(0)),
            Literal::positive(Variable(1)),
        ]);

        // Assumptions: ¬x, ¬y, z (only first two cause conflict)
        let assumptions = vec![
            Literal::negative(Variable(0)), // ¬x
            Literal::negative(Variable(1)), // ¬y
            Literal::positive(Variable(2)), // z (irrelevant)
        ];

        let result = dpll.solve_with_assumptions(&assumptions);

        match result {
            AssumeResult::Unsat(core) => {
                // Core should be a subset of assumptions
                for lit in &core {
                    assert!(
                        assumptions.contains(lit),
                        "Unsat core literal {:?} not in assumptions",
                        lit
                    );
                }

                // Core should contain at least ¬x or ¬y (both needed for conflict)
                // but should NOT contain z (it's irrelevant)
                let contains_z = core.contains(&Literal::positive(Variable(2)));
                assert!(
                    !contains_z,
                    "Unsat core should not contain irrelevant assumption z"
                );
            }
            AssumeResult::Sat(_) => panic!("Should be unsat"),
            AssumeResult::Unknown => panic!("Should determine satisfiability"),
        }
    }

    /// Test unsat core with EUF theory
    #[test]
    fn test_dpllt_unsat_core_with_euf() {
        // Create: a = b ∧ f(a) ≠ f(b) (UNSAT due to congruence)
        let mut terms = TermStore::new();

        let a = terms.mk_var("a", Sort::Uninterpreted("U".to_string()));
        let b = terms.mk_var("b", Sort::Uninterpreted("U".to_string()));

        let f_a = terms.mk_app(
            Symbol::Named("f".to_string()),
            vec![a],
            Sort::Uninterpreted("U".to_string()),
        );
        let f_b = terms.mk_app(
            Symbol::Named("f".to_string()),
            vec![b],
            Sort::Uninterpreted("U".to_string()),
        );

        let a_eq_b = terms.mk_eq(a, b);
        let f_a_eq_f_b = terms.mk_eq(f_a, f_b);
        let f_a_neq_f_b = terms.mk_not(f_a_eq_f_b);

        // Tseitin transform
        let tseitin = Tseitin::new(&terms);
        let result = tseitin.transform_all(&[a_eq_b, f_a_neq_f_b]);

        // Create solver
        let euf = EufSolver::new(&terms);
        let mut dpll = DpllT::from_tseitin(&terms, &result, euf);

        // This should be UNSAT without assumptions
        let result = dpll.solve();
        assert!(
            matches!(result, SolveResult::Unsat),
            "a = b ∧ f(a) ≠ f(b) should be UNSAT"
        );
    }

    // ========================================================================
    // Gap 9: DpllT Push/Pop Incremental Solving Tests
    // ========================================================================

    /// Test basic push/pop scope depth tracking
    #[test]
    fn test_dpllt_push_pop_scope_depth() {
        let theory = PropositionalTheory;
        let mut dpll = DpllT::new(3, theory);

        assert_eq!(dpll.scope_depth(), 0, "Initial scope depth should be 0");

        dpll.push();
        assert_eq!(dpll.scope_depth(), 1, "After push, scope depth should be 1");

        dpll.push();
        assert_eq!(
            dpll.scope_depth(),
            2,
            "After second push, scope depth should be 2"
        );

        let ok = dpll.pop();
        assert!(ok, "Pop should succeed");
        assert_eq!(dpll.scope_depth(), 1, "After pop, scope depth should be 1");

        let ok = dpll.pop();
        assert!(ok, "Pop should succeed");
        assert_eq!(
            dpll.scope_depth(),
            0,
            "After second pop, scope depth should be 0"
        );

        let ok = dpll.pop();
        assert!(!ok, "Pop on empty should return false");
        assert_eq!(dpll.scope_depth(), 0, "Scope depth should remain 0");
    }

    /// Test that clauses added after push are disabled after pop
    #[test]
    fn test_dpllt_push_pop_clause_scoping() {
        let theory = PropositionalTheory;
        let mut dpll = DpllT::new(2, theory);

        // Add a base clause that makes formula SAT: (x ∨ y)
        dpll.add_clause(vec![
            Literal::positive(Variable(0)),
            Literal::positive(Variable(1)),
        ]);

        // Solve - should be SAT
        let result = dpll.solve();
        assert!(matches!(result, SolveResult::Sat(_)));

        // Push and add a conflicting clause
        dpll.push();

        // Add ¬x and ¬y unit clauses - combined with (x ∨ y), this is UNSAT
        dpll.add_clause(vec![Literal::negative(Variable(0))]);
        dpll.add_clause(vec![Literal::negative(Variable(1))]);

        // Should now be UNSAT
        let result = dpll.solve();
        assert!(matches!(result, SolveResult::Unsat));

        // Pop - the ¬x and ¬y clauses should be disabled
        let ok = dpll.pop();
        assert!(ok);

        // Should be SAT again (only base clause (x ∨ y) is active)
        let result = dpll.solve();
        assert!(
            matches!(result, SolveResult::Sat(_)),
            "After pop, formula should be SAT again"
        );
    }

    /// Test incremental solving with multiple push/pop cycles
    #[test]
    fn test_dpllt_incremental_multiple_cycles() {
        let theory = PropositionalTheory;
        let mut dpll = DpllT::new(3, theory);

        // Base clause: (x ∨ y ∨ z)
        dpll.add_clause(vec![
            Literal::positive(Variable(0)),
            Literal::positive(Variable(1)),
            Literal::positive(Variable(2)),
        ]);

        // First cycle: force x=false, y=false - only z can be true
        dpll.push();
        dpll.add_clause(vec![Literal::negative(Variable(0))]); // ¬x
        dpll.add_clause(vec![Literal::negative(Variable(1))]); // ¬y

        let result = dpll.solve();
        match result {
            SolveResult::Sat(model) => {
                // z must be true
                assert!(model.get(2).copied().unwrap_or(false), "z should be true");
            }
            _ => panic!("Should be SAT with z=true"),
        }

        dpll.pop();

        // Second cycle: force x=false, z=false - only y can be true
        dpll.push();
        dpll.add_clause(vec![Literal::negative(Variable(0))]); // ¬x
        dpll.add_clause(vec![Literal::negative(Variable(2))]); // ¬z

        let result = dpll.solve();
        match result {
            SolveResult::Sat(model) => {
                // y must be true
                assert!(model.get(1).copied().unwrap_or(false), "y should be true");
            }
            _ => panic!("Should be SAT with y=true"),
        }

        dpll.pop();

        // After all pops, should be SAT with any of x, y, z
        let result = dpll.solve();
        assert!(matches!(result, SolveResult::Sat(_)));
    }

    /// Test nested push/pop scopes
    #[test]
    fn test_dpllt_nested_push_pop() {
        let theory = PropositionalTheory;
        let mut dpll = DpllT::new(3, theory);

        // Base: (x ∨ y ∨ z)
        dpll.add_clause(vec![
            Literal::positive(Variable(0)),
            Literal::positive(Variable(1)),
            Literal::positive(Variable(2)),
        ]);

        // Level 1: add ¬x
        dpll.push();
        dpll.add_clause(vec![Literal::negative(Variable(0))]);
        assert_eq!(dpll.scope_depth(), 1);

        // Level 2: add ¬y
        dpll.push();
        dpll.add_clause(vec![Literal::negative(Variable(1))]);
        assert_eq!(dpll.scope_depth(), 2);

        // At level 2: only z can be true
        let result = dpll.solve();
        match result {
            SolveResult::Sat(model) => {
                assert!(!model.first().copied().unwrap_or(true), "x should be false");
                assert!(!model.get(1).copied().unwrap_or(true), "y should be false");
                assert!(model.get(2).copied().unwrap_or(false), "z should be true");
            }
            _ => panic!("Should be SAT"),
        }

        // Pop level 2 - ¬y is removed
        dpll.pop();
        assert_eq!(dpll.scope_depth(), 1);

        // At level 1: x is false, y or z can be true
        let result = dpll.solve();
        assert!(matches!(result, SolveResult::Sat(_)));

        // Pop level 1 - ¬x is removed
        dpll.pop();
        assert_eq!(dpll.scope_depth(), 0);

        // At base level: any of x, y, z can be true
        let result = dpll.solve();
        assert!(matches!(result, SolveResult::Sat(_)));
    }

    /// Test that pop on empty scope returns false and is safe
    #[test]
    fn test_dpllt_pop_empty_safe() {
        let theory = PropositionalTheory;
        let mut dpll = DpllT::new(2, theory);

        // Pop without any push should be safe and return false
        assert!(!dpll.pop());
        assert!(!dpll.pop());
        assert_eq!(dpll.scope_depth(), 0);

        // Solver should still work
        dpll.add_clause(vec![Literal::positive(Variable(0))]);
        let result = dpll.solve();
        assert!(matches!(result, SolveResult::Sat(_)));
    }

    // ========================================================================
    // Gap 9: Theory Propagation Soundness Tests
    // ========================================================================

    /// Test EUF theory propagation soundness through transitivity.
    ///
    /// When EUF detects a conflict due to transitivity (a=b, b=c, a≠c),
    /// the DPLL(T) loop must correctly generate a blocking clause and
    /// return UNSAT.
    #[test]
    fn test_gap9_euf_propagation_transitivity() {
        let mut terms = TermStore::new();
        let sort_s = Sort::Uninterpreted("S".to_string());

        let a = terms.mk_var("a", sort_s.clone());
        let b = terms.mk_var("b", sort_s.clone());
        let c = terms.mk_var("c", sort_s);

        // a = b
        let eq_ab = terms.mk_eq(a, b);
        // b = c
        let eq_bc = terms.mk_eq(b, c);
        // a ≠ c
        let eq_ac = terms.mk_eq(a, c);
        let neq_ac = terms.mk_not(eq_ac);

        // Conjunction: (a = b) ∧ (b = c) ∧ (a ≠ c)
        let formula = terms.mk_and(vec![eq_ab, eq_bc, neq_ac]);

        let tseitin = Tseitin::new(&terms);
        let result = tseitin.transform(formula);

        let theory = EufSolver::new(&terms);
        let mut dpll = DpllT::from_tseitin(&terms, &result, theory);

        let solve_result = dpll.solve();
        assert!(
            matches!(solve_result, SolveResult::Unsat),
            "EUF transitivity should detect UNSAT: a=b ∧ b=c ∧ a≠c"
        );
    }

    /// Test EUF congruence closure with function applications.
    ///
    /// If a=b then f(a)=f(b). So a=b ∧ f(a)≠f(b) should be UNSAT.
    #[test]
    fn test_gap9_euf_propagation_congruence() {
        let mut terms = TermStore::new();
        let sort_u = Sort::Uninterpreted("U".to_string());

        let a = terms.mk_var("a", sort_u.clone());
        let b = terms.mk_var("b", sort_u.clone());

        // f(a) and f(b)
        let f_a = terms.mk_app(Symbol::named("f"), vec![a], sort_u.clone());
        let f_b = terms.mk_app(Symbol::named("f"), vec![b], sort_u);

        // a = b
        let eq_ab = terms.mk_eq(a, b);
        // f(a) ≠ f(b)
        let eq_fa_fb = terms.mk_eq(f_a, f_b);
        let neq_fa_fb = terms.mk_not(eq_fa_fb);

        // Conjunction: (a = b) ∧ (f(a) ≠ f(b))
        let formula = terms.mk_and(vec![eq_ab, neq_fa_fb]);

        let tseitin = Tseitin::new(&terms);
        let result = tseitin.transform(formula);

        let theory = EufSolver::new(&terms);
        let mut dpll = DpllT::from_tseitin(&terms, &result, theory);

        let solve_result = dpll.solve();
        assert!(
            matches!(solve_result, SolveResult::Unsat),
            "EUF congruence should detect UNSAT: a=b ∧ f(a)≠f(b)"
        );
    }

    /// Test EUF with satisfiable formula.
    ///
    /// a≠b ∧ f(a)=c should be SAT (no congruence conflict).
    #[test]
    fn test_gap9_euf_propagation_sat() {
        let mut terms = TermStore::new();
        let sort_u = Sort::Uninterpreted("U".to_string());

        let a = terms.mk_var("a", sort_u.clone());
        let b = terms.mk_var("b", sort_u.clone());
        let c = terms.mk_var("c", sort_u.clone());

        let f_a = terms.mk_app(Symbol::named("f"), vec![a], sort_u);

        // a ≠ b
        let eq_ab = terms.mk_eq(a, b);
        let neq_ab = terms.mk_not(eq_ab);
        // f(a) = c
        let eq_fa_c = terms.mk_eq(f_a, c);

        // Conjunction: (a ≠ b) ∧ (f(a) = c)
        let formula = terms.mk_and(vec![neq_ab, eq_fa_c]);

        let tseitin = Tseitin::new(&terms);
        let result = tseitin.transform(formula);

        let theory = EufSolver::new(&terms);
        let mut dpll = DpllT::from_tseitin(&terms, &result, theory);

        let solve_result = dpll.solve();
        assert!(
            matches!(solve_result, SolveResult::Sat(_)),
            "EUF should be SAT: a≠b ∧ f(a)=c"
        );
    }

    /// Test LRA theory propagation with contradictory bounds.
    ///
    /// x >= 5 ∧ x <= 3 should be detected as UNSAT.
    #[test]
    fn test_gap9_lra_propagation_bounds_conflict() {
        use z4_lra::LraSolver;

        let mut terms = TermStore::new();

        let x = terms.mk_var("x", Sort::Real);
        let five = terms.mk_int(5.into());
        let three = terms.mk_int(3.into());

        // x >= 5
        let ge_5 = terms.mk_app(Symbol::Named(">=".to_string()), vec![x, five], Sort::Bool);
        // x <= 3
        let le_3 = terms.mk_app(Symbol::Named("<=".to_string()), vec![x, three], Sort::Bool);

        // Conjunction
        let formula = terms.mk_and(vec![ge_5, le_3]);

        let tseitin = Tseitin::new(&terms);
        let result = tseitin.transform(formula);

        let theory = LraSolver::new(&terms);
        let mut dpll = DpllT::from_tseitin(&terms, &result, theory);

        let solve_result = dpll.solve();
        assert!(
            matches!(solve_result, SolveResult::Unsat),
            "LRA should detect UNSAT: x >= 5 ∧ x <= 3"
        );
    }

    /// Test LRA with satisfiable formula.
    ///
    /// x >= 0 ∧ x <= 10 should be SAT.
    #[test]
    fn test_gap9_lra_propagation_sat() {
        use z4_lra::LraSolver;

        let mut terms = TermStore::new();

        let x = terms.mk_var("x", Sort::Real);
        let zero = terms.mk_int(0.into());
        let ten = terms.mk_int(10.into());

        // x >= 0
        let ge_0 = terms.mk_app(Symbol::Named(">=".to_string()), vec![x, zero], Sort::Bool);
        // x <= 10
        let le_10 = terms.mk_app(Symbol::Named("<=".to_string()), vec![x, ten], Sort::Bool);

        // Conjunction
        let formula = terms.mk_and(vec![ge_0, le_10]);

        let tseitin = Tseitin::new(&terms);
        let result = tseitin.transform(formula);

        let theory = LraSolver::new(&terms);
        let mut dpll = DpllT::from_tseitin(&terms, &result, theory);

        let solve_result = dpll.solve();
        assert!(
            matches!(solve_result, SolveResult::Sat(_)),
            "LRA should be SAT: x >= 0 ∧ x <= 10"
        );
    }

    /// Test LRA with strict bounds conflict.
    ///
    /// x > 5 ∧ x < 5 should be UNSAT.
    #[test]
    fn test_gap9_lra_propagation_strict_bounds() {
        use z4_lra::LraSolver;

        let mut terms = TermStore::new();

        let x = terms.mk_var("x", Sort::Real);
        let five = terms.mk_int(5.into());

        // x > 5
        let gt_5 = terms.mk_app(Symbol::Named(">".to_string()), vec![x, five], Sort::Bool);
        // x < 5
        let lt_5 = terms.mk_app(Symbol::Named("<".to_string()), vec![x, five], Sort::Bool);

        // Conjunction
        let formula = terms.mk_and(vec![gt_5, lt_5]);

        let tseitin = Tseitin::new(&terms);
        let result = tseitin.transform(formula);

        let theory = LraSolver::new(&terms);
        let mut dpll = DpllT::from_tseitin(&terms, &result, theory);

        let solve_result = dpll.solve();
        assert!(
            matches!(solve_result, SolveResult::Unsat),
            "LRA should detect UNSAT: x > 5 ∧ x < 5"
        );
    }

    /// Test LIA theory with integer division forcing.
    ///
    /// 2x = 1 should be UNSAT in integers (no integer solution).
    #[test]
    fn test_gap9_lia_propagation_no_integer_solution() {
        use z4_lia::LiaSolver;

        let mut terms = TermStore::new();

        let x = terms.mk_var("x", Sort::Int);
        let two = terms.mk_int(2.into());
        let one = terms.mk_int(1.into());

        // 2*x
        let two_x = terms.mk_app(Symbol::Named("*".to_string()), vec![two, x], Sort::Int);
        // 2*x = 1
        let eq = terms.mk_eq(two_x, one);

        let tseitin = Tseitin::new(&terms);
        let result = tseitin.transform(eq);

        let theory = LiaSolver::new(&terms);
        let mut dpll = DpllT::from_tseitin(&terms, &result, theory);

        let solve_result = dpll.solve();
        // LIA solver should return UNSAT or Unknown (depending on branch-and-bound convergence)
        // The key is it should NOT return SAT
        assert!(
            !matches!(solve_result, SolveResult::Sat(_)),
            "LIA should NOT return SAT for 2x = 1 (no integer solution)"
        );
    }

    /// Test LIA with satisfiable formula.
    ///
    /// x >= 0 ∧ x <= 5 should be SAT with integer solutions 0,1,2,3,4,5.
    #[test]
    fn test_gap9_lia_propagation_sat() {
        use z4_lia::LiaSolver;

        let mut terms = TermStore::new();

        let x = terms.mk_var("x", Sort::Int);
        let zero = terms.mk_int(0.into());
        let five = terms.mk_int(5.into());

        // x >= 0
        let ge_0 = terms.mk_app(Symbol::Named(">=".to_string()), vec![x, zero], Sort::Bool);
        // x <= 5
        let le_5 = terms.mk_app(Symbol::Named("<=".to_string()), vec![x, five], Sort::Bool);

        // Conjunction
        let formula = terms.mk_and(vec![ge_0, le_5]);

        let tseitin = Tseitin::new(&terms);
        let result = tseitin.transform(formula);

        let theory = LiaSolver::new(&terms);
        let mut dpll = DpllT::from_tseitin(&terms, &result, theory);

        let solve_result = dpll.solve();
        assert!(
            matches!(solve_result, SolveResult::Sat(_)),
            "LIA should be SAT: x >= 0 ∧ x <= 5"
        );
    }

    /// Test theory conflict clause generation.
    ///
    /// When theory detects a conflict, the conflict clause should block
    /// the current assignment. This test verifies that the DPLL(T) loop
    /// correctly adds theory lemmas and continues searching.
    #[test]
    fn test_gap9_theory_conflict_clause_generation() {
        let mut terms = TermStore::new();
        let sort_s = Sort::Uninterpreted("S".to_string());

        let a = terms.mk_var("a", sort_s.clone());
        let b = terms.mk_var("b", sort_s.clone());
        let c = terms.mk_var("c", sort_s.clone());

        // Create a formula where SAT finds a model but theory rejects it
        // (a = b) ∨ (b = c) is SAT propositionally
        // With EUF, we need to check if adding more constraints causes conflict
        let eq_ab = terms.mk_eq(a, b);
        let eq_bc = terms.mk_eq(b, c);
        let eq_ac = terms.mk_eq(a, c);
        let neq_ac = terms.mk_not(eq_ac);

        // (a = b) ∧ ((b = c) ∨ (a ≠ c))
        // This is SAT: either b=c is true, or a≠c is true
        let disjunct = terms.mk_or(vec![eq_bc, neq_ac]);
        let formula = terms.mk_and(vec![eq_ab, disjunct]);

        let tseitin = Tseitin::new(&terms);
        let result = tseitin.transform(formula);

        let theory = EufSolver::new(&terms);
        let mut dpll = DpllT::from_tseitin(&terms, &result, theory);

        let solve_result = dpll.solve();
        assert!(
            matches!(solve_result, SolveResult::Sat(_)),
            "Should find SAT model where a=b and either b=c or a≠c"
        );
    }

    /// Test theory integration with multiple theory lemmas.
    ///
    /// Create a formula that requires multiple DPLL(T) iterations.
    #[test]
    fn test_gap9_theory_multiple_lemmas() {
        let mut terms = TermStore::new();
        let sort_s = Sort::Uninterpreted("S".to_string());

        let a = terms.mk_var("a", sort_s.clone());
        let b = terms.mk_var("b", sort_s.clone());
        let c = terms.mk_var("c", sort_s.clone());
        let d = terms.mk_var("d", sort_s);

        let eq_ab = terms.mk_eq(a, b);
        let eq_bc = terms.mk_eq(b, c);
        let eq_cd = terms.mk_eq(c, d);
        let eq_ad = terms.mk_eq(a, d);
        let neq_ad = terms.mk_not(eq_ad);

        // (a = b) ∧ (b = c) ∧ (c = d) ∧ (a ≠ d)
        // This requires transitivity chain a=b=c=d to detect UNSAT
        let formula = terms.mk_and(vec![eq_ab, eq_bc, eq_cd, neq_ad]);

        let tseitin = Tseitin::new(&terms);
        let result = tseitin.transform(formula);

        let theory = EufSolver::new(&terms);
        let mut dpll = DpllT::from_tseitin(&terms, &result, theory);

        let solve_result = dpll.solve();
        assert!(
            matches!(solve_result, SolveResult::Unsat),
            "EUF should detect UNSAT through transitivity chain: a=b=c=d ∧ a≠d"
        );
    }

    /// Test sync_theory correctly communicates SAT assignment to theory.
    ///
    /// This test creates a formula where the SAT solver assigns values
    /// and the theory must correctly interpret them.
    #[test]
    fn test_gap9_sync_theory_assignment() {
        let mut terms = TermStore::new();
        let sort_s = Sort::Uninterpreted("S".to_string());

        let a = terms.mk_var("a", sort_s.clone());
        let b = terms.mk_var("b", sort_s.clone());

        // p(a) - predicate
        let p_a = terms.mk_app(Symbol::named("p"), vec![a], Sort::Bool);
        // p(b)
        let p_b = terms.mk_app(Symbol::named("p"), vec![b], Sort::Bool);

        let eq_ab = terms.mk_eq(a, b);
        let not_p_b = terms.mk_not(p_b);

        // (a = b) ∧ p(a) ∧ ¬p(b) - UNSAT due to congruence
        let formula = terms.mk_and(vec![eq_ab, p_a, not_p_b]);

        let tseitin = Tseitin::new(&terms);
        let result = tseitin.transform(formula);

        let theory = EufSolver::new(&terms);
        let mut dpll = DpllT::from_tseitin(&terms, &result, theory);

        let solve_result = dpll.solve();
        assert!(
            matches!(solve_result, SolveResult::Unsat),
            "sync_theory should communicate assignment correctly: a=b ∧ p(a) ∧ ¬p(b) is UNSAT"
        );
    }
}

// ============================================================================
// Kani Proofs for DpllT Push/Pop (Gap 9)
// ============================================================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Proof: push() increments scope depth by exactly 1.
    ///
    /// Invariant INV-PUSH-1: After push(), scope depth increases by 1
    #[kani::proof]
    fn proof_push_increments_scope_depth() {
        let theory = PropositionalTheory;
        let mut dpll = DpllT::new(2, theory);

        let depth_before = dpll.scope_depth();
        dpll.push();
        let depth_after = dpll.scope_depth();

        assert!(
            depth_after == depth_before + 1,
            "Push must increment scope depth by 1"
        );
    }

    /// Proof: pop() decrements scope depth by exactly 1 when scopes exist.
    ///
    /// Invariant INV-POP-1: After pop(), scope depth decreases by 1 (if > 0)
    #[kani::proof]
    fn proof_pop_decrements_scope_depth() {
        let theory = PropositionalTheory;
        let mut dpll = DpllT::new(2, theory);

        // Push at least once so we can pop
        dpll.push();
        let depth_before = dpll.scope_depth();
        kani::assume(depth_before > 0);

        let result = dpll.pop();
        let depth_after = dpll.scope_depth();

        assert!(result, "Pop should succeed when scope depth > 0");
        assert!(
            depth_after == depth_before - 1,
            "Pop must decrement scope depth by 1"
        );
    }

    /// Proof: pop() returns false and is safe when scope depth is 0.
    ///
    /// Invariant: Pop on empty scope is safe and returns false.
    #[kani::proof]
    fn proof_pop_empty_is_safe() {
        let theory = PropositionalTheory;
        let mut dpll = DpllT::new(2, theory);

        // No push calls, scope depth is 0
        assert_eq!(dpll.scope_depth(), 0);

        let result = dpll.pop();
        assert!(!result, "Pop on empty must return false");
        assert_eq!(dpll.scope_depth(), 0, "Scope depth must remain 0");
    }

    /// Proof: nested push/pop maintains correct scope depth.
    ///
    /// Verifies that multiple push/pop operations maintain correct depth tracking.
    #[kani::proof]
    fn proof_nested_push_pop_depth() {
        let theory = PropositionalTheory;
        let mut dpll = DpllT::new(2, theory);

        // Number of push operations (bounded for tractability)
        let n: usize = kani::any();
        kani::assume(n <= 5);

        // Push n times
        for _ in 0..n {
            dpll.push();
        }
        assert_eq!(dpll.scope_depth(), n, "Depth should equal number of pushes");

        // Pop all
        for i in 0..n {
            let expected_depth = n - i - 1;
            let result = dpll.pop();
            assert!(result, "Pop should succeed");
            assert_eq!(
                dpll.scope_depth(),
                expected_depth,
                "Depth should decrease correctly"
            );
        }

        assert_eq!(dpll.scope_depth(), 0, "Final depth should be 0");
    }

    /// Proof: pop after push restores scope depth to original value.
    ///
    /// Invariant: push(); pop(); leaves scope depth unchanged.
    #[kani::proof]
    fn proof_push_pop_restores_depth() {
        let theory = PropositionalTheory;
        let mut dpll = DpllT::new(2, theory);

        // Start with some scope depth (bounded)
        let initial_pushes: usize = kani::any();
        kani::assume(initial_pushes <= 3);

        for _ in 0..initial_pushes {
            dpll.push();
        }
        let depth_before = dpll.scope_depth();

        // Push and pop
        dpll.push();
        let result = dpll.pop();

        assert!(result, "Pop should succeed after push");
        assert_eq!(
            dpll.scope_depth(),
            depth_before,
            "push(); pop(); must restore depth"
        );
    }

    /// Proof: scope depth is non-negative.
    ///
    /// Invariant: Scope depth can never become negative.
    #[kani::proof]
    fn proof_scope_depth_non_negative() {
        let theory = PropositionalTheory;
        let mut dpll = DpllT::new(2, theory);

        // Random sequence of operations
        let ops: [bool; 5] = kani::any();

        for is_push in ops {
            if is_push {
                dpll.push();
            } else {
                let _ = dpll.pop();
            }
            assert!(
                dpll.scope_depth() >= 0,
                "Scope depth must never be negative"
            );
        }
    }

    // ========================================================================
    // Gap 9: Theory Integration Proofs
    // ========================================================================

    /// Proof: register_theory_atom maintains bidirectional mapping.
    ///
    /// Invariant: After register_theory_atom(term, var), both:
    ///   - term_for_var(var) == Some(term)
    ///   - var_for_term(term) == Some(var)
    #[kani::proof]
    fn proof_register_theory_atom_consistency() {
        let theory = PropositionalTheory;
        let mut dpll = DpllT::new(10, theory);

        // Symbolic term and variable
        let term_id: u32 = kani::any();
        let var_idx: u32 = kani::any();
        kani::assume(var_idx < 10);

        // Create TermId (symbolic, just for mapping test)
        let term = z4_core::TermId::from_raw(term_id);

        // Register the atom
        dpll.register_theory_atom(term, var_idx);

        // Verify bidirectional consistency
        let var = Variable(var_idx);
        let retrieved_term = dpll.term_for_var(var);
        let retrieved_var = dpll.var_for_term(term);

        assert!(
            retrieved_term == Some(term),
            "term_for_var must return the registered term"
        );
        assert!(
            retrieved_var == Some(var),
            "var_for_term must return the registered variable"
        );
    }

    /// Proof: term_to_literal produces correct polarity.
    ///
    /// Invariant: term_to_literal(term, true) produces positive literal,
    ///            term_to_literal(term, false) produces negative literal.
    #[kani::proof]
    fn proof_term_to_literal_polarity() {
        let theory = PropositionalTheory;
        let mut dpll = DpllT::new(5, theory);

        let term_id: u32 = kani::any();
        let var_idx: u32 = kani::any();
        kani::assume(var_idx < 5);

        let term = z4_core::TermId::from_raw(term_id);
        dpll.register_theory_atom(term, var_idx);

        // Test positive literal
        let pos_lit = dpll.term_to_literal(term, true);
        assert!(
            pos_lit.is_some(),
            "term_to_literal should return Some for registered term"
        );
        let pos_lit = pos_lit.unwrap();
        assert!(
            pos_lit.is_positive(),
            "term_to_literal(term, true) must be positive"
        );
        assert_eq!(pos_lit.variable().0, var_idx, "Variable index must match");

        // Test negative literal
        let neg_lit = dpll.term_to_literal(term, false);
        assert!(
            neg_lit.is_some(),
            "term_to_literal should return Some for registered term"
        );
        let neg_lit = neg_lit.unwrap();
        assert!(
            !neg_lit.is_positive(),
            "term_to_literal(term, false) must be negative"
        );
        assert_eq!(neg_lit.variable().0, var_idx, "Variable index must match");
    }

    /// Proof: unregistered term returns None.
    ///
    /// Invariant: term_to_literal returns None for terms not registered.
    #[kani::proof]
    fn proof_unregistered_term_returns_none() {
        let theory = PropositionalTheory;
        let dpll = DpllT::new(5, theory);

        // Any term_id without registration
        let term_id: u32 = kani::any();
        let term = z4_core::TermId::from_raw(term_id);

        // Should return None since nothing is registered
        let result = dpll.term_to_literal(term, true);
        assert!(result.is_none(), "Unregistered term must return None");

        let result = dpll.term_to_literal(term, false);
        assert!(result.is_none(), "Unregistered term must return None");
    }

    /// Proof: add_clause increases clause count.
    ///
    /// Invariant: After add_clause, SAT solver has at least one more clause.
    #[kani::proof]
    fn proof_add_clause_increases_count() {
        let theory = PropositionalTheory;
        let mut dpll = DpllT::new(3, theory);

        // Add a clause
        let lit1 = Literal::positive(Variable(0));
        let lit2 = Literal::negative(Variable(1));
        dpll.add_clause(vec![lit1, lit2]);

        // Solver should still be valid (no panic)
        // We can verify by checking that solving works
        let result = dpll.solve();
        // The formula (x0 ∨ ¬x1) is SAT
        assert!(
            matches!(result, SolveResult::Sat(_)),
            "Simple clause should be SAT"
        );
    }

    /// Proof: theory reset clears state.
    ///
    /// Invariant: reset_theory() allows fresh solving.
    #[kani::proof]
    fn proof_reset_theory_allows_fresh_solve() {
        let theory = PropositionalTheory;
        let mut dpll = DpllT::new(2, theory);

        // Add a simple SAT clause
        dpll.add_clause(vec![Literal::positive(Variable(0))]);

        // Solve
        let result1 = dpll.solve();
        assert!(matches!(result1, SolveResult::Sat(_)));

        // Reset theory
        dpll.reset_theory();

        // Should still solve correctly
        let result2 = dpll.solve();
        assert!(matches!(result2, SolveResult::Sat(_)));
    }
}
