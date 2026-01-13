//! QCDCL Solver
//!
//! Quantified Conflict-Driven Clause Learning algorithm for QBF.
//!
//! ## Algorithm Overview
//!
//! QCDCL extends CDCL for quantified formulas:
//!
//! 1. **Quantifier-aware propagation**: A clause can only propagate an existential
//!    literal. Universal literals cannot be forced because the adversary controls them.
//!
//! 2. **Universal reduction**: Universal literals at the "tail" of a clause
//!    (with level >= max existential level) can be removed.
//!
//! 3. **Two-sided learning**: Learn clauses on existential conflicts,
//!    learn cubes on universal "wins".
//!
//! ## Current Implementation Status
//!
//! **Implemented:**
//! - Basic QCDCL with quantifier-aware unit propagation
//! - Universal reduction (Q-resolution)
//! - Universally-blocked clause detection
//! - Tautology detection (clauses with x ∨ ¬x)
//! - Q-resolution based 1-UIP conflict analysis
//! - VSIDS-style activity-based decision heuristic
//! - Two-watched literals for efficient propagation
//! - Cube learning for solution states (existential wins)
//! - Cube propagation for blocking universal search paths
//! - Certificate generation (constant Skolem/Herbrand functions)
//!
//! **Not Yet Implemented (Future Work):**
//! - Dependency learning (DepQBF-style)
//! - Clause database management (learned clause deletion)
//! - Long-distance resolution
//!
//! ## References
//! - Zhang & Malik, "Conflict Driven Learning in a Quantified Boolean Satisfiability Solver"
//! - Lonsing & Biere, "DepQBF: A Dependency-Aware QBF Solver"

use crate::formula::QbfFormula;
use hashbrown::HashSet;
use z4_sat::{Literal, Variable};

/// Result of QBF solving
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QbfResult {
    /// Formula is satisfiable (true for all universal assignments)
    Sat(Certificate),
    /// Formula is unsatisfiable (false for some universal assignment)
    Unsat(Certificate),
    /// Unknown result (timeout, resource limit)
    Unknown,
}

/// Certificate for QBF result
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Certificate {
    /// Skolem functions mapping existential vars to boolean functions of outer universals
    Skolem(Vec<SkolemFunction>),
    /// Herbrand functions mapping universal vars to boolean functions of outer existentials
    Herbrand(Vec<HerbrandFunction>),
    /// No certificate (for simple true/false results)
    None,
}

/// A Skolem function for an existential variable
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkolemFunction {
    /// The existential variable
    pub variable: u32,
    /// The function as a truth table (indexed by universal variable assignments)
    /// For now, just store a constant value
    pub value: bool,
}

/// A Herbrand function for a universal variable
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HerbrandFunction {
    /// The universal variable
    pub variable: u32,
    /// The counterexample value
    pub value: bool,
}

/// Assignment state for a variable
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Assignment {
    /// Variable is unassigned
    Unassigned,
    /// Variable is assigned true
    True,
    /// Variable is assigned false
    False,
}

impl Assignment {
    fn to_bool(self) -> Option<bool> {
        match self {
            Assignment::True => Some(true),
            Assignment::False => Some(false),
            Assignment::Unassigned => None,
        }
    }

    fn is_assigned(self) -> bool {
        self != Assignment::Unassigned
    }
}

/// Reason for an assignment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reason {
    /// Decision (no reason)
    Decision,
    /// Propagated from clause
    Propagated(usize),
}

/// QCDCL Solver
pub struct QbfSolver {
    /// The formula
    formula: QbfFormula,
    /// Variable assignments (0-indexed)
    assignments: Vec<Assignment>,
    /// Decision level for each variable (0-indexed)
    levels: Vec<u32>,
    /// Reason for each assignment (0-indexed)
    reasons: Vec<Reason>,
    /// Assignment trail (in order of assignment)
    trail: Vec<Literal>,
    /// Decision level boundaries in trail
    trail_lim: Vec<usize>,
    /// Current decision level
    decision_level: u32,
    /// Learned clauses (disjunctions - block existential search paths)
    learned: Vec<Vec<Literal>>,
    /// Learned cubes (conjunctions - block universal search paths)
    /// A cube represents a winning strategy for the existential player
    cubes: Vec<Vec<Literal>>,
    /// Variables in quantifier order for decisions
    decision_order: Vec<u32>,
    /// Variable activity for VSIDS-style decisions (0-indexed by var-1)
    activity: Vec<f64>,
    /// VSIDS increment amount
    var_inc: f64,
    /// VSIDS decay factor (smaller => faster decay)
    var_decay: f64,
    /// Number of conflicts
    conflicts: u64,
    /// Number of propagations
    propagations: u64,
    /// Number of decisions
    decisions: u64,
    /// Two-watched literal data: watches[lit_idx] = list of (clause_idx, other_watch)
    /// lit_idx = var * 2 + (1 if negative else 0)
    watches: Vec<Vec<WatchInfo>>,
    /// Position in trail for propagation queue
    qhead: usize,
}

/// Watch information for a clause
#[derive(Debug, Clone, Copy)]
struct WatchInfo {
    /// Clause index (high bit indicates learned clause)
    clause_idx: usize,
    /// The other watched literal (for quick filtering)
    blocker: Literal,
}

/// Bit flag to distinguish learned clauses in watch lists
const LEARNED_CLAUSE_BIT: usize = 1 << (usize::BITS - 1);

impl QbfSolver {
    /// Create a new QBF solver for the given formula
    pub fn new(formula: QbfFormula) -> Self {
        let num_vars = formula.num_vars;

        // Build decision order from quantifier prefix
        let mut decision_order = Vec::with_capacity(num_vars);
        for block in &formula.prefix {
            for &var in &block.variables {
                decision_order.push(var);
            }
        }

        // Add any unquantified variables (implicitly existential at outermost)
        let quantified: HashSet<u32> = decision_order.iter().copied().collect();
        for v in 1..=num_vars as u32 {
            if !quantified.contains(&v) {
                decision_order.insert(0, v); // Add at front (outermost)
            }
        }

        // Initialize watch lists: 2 entries per variable (positive and negative)
        let watches = vec![Vec::new(); num_vars * 2 + 2];

        let mut solver = Self {
            formula,
            assignments: vec![Assignment::Unassigned; num_vars],
            levels: vec![0; num_vars],
            reasons: vec![Reason::Decision; num_vars],
            trail: Vec::with_capacity(num_vars),
            trail_lim: Vec::new(),
            decision_level: 0,
            learned: Vec::new(),
            cubes: Vec::new(),
            decision_order,
            activity: vec![0.0; num_vars],
            var_inc: 1.0,
            var_decay: 0.95,
            conflicts: 0,
            propagations: 0,
            decisions: 0,
            watches,
            qhead: 0,
        };

        // Initialize watches for all original clauses
        solver.init_watches();
        solver
    }

    /// Convert a literal to its watch index
    #[inline]
    fn lit_to_watch_idx(lit: Literal) -> usize {
        let var = lit.variable().0 as usize;
        var * 2 + if lit.is_positive() { 0 } else { 1 }
    }

    /// Initialize watches for all original clauses
    fn init_watches(&mut self) {
        for i in 0..self.formula.clauses.len() {
            let clause = &self.formula.clauses[i];
            if clause.len() >= 2 {
                // Watch first two literals
                let lit0 = clause[0];
                let lit1 = clause[1];
                let idx0 = Self::lit_to_watch_idx(lit0);
                let idx1 = Self::lit_to_watch_idx(lit1);
                self.watches[idx0].push(WatchInfo {
                    clause_idx: i,
                    blocker: lit1,
                });
                self.watches[idx1].push(WatchInfo {
                    clause_idx: i,
                    blocker: lit0,
                });
            }
            // Unit and empty clauses don't need watches - handled specially
        }
    }

    /// Add watches for a learned clause
    fn add_learned_watches(&mut self, clause_idx: usize) {
        let clause = &self.learned[clause_idx];
        if clause.len() >= 2 {
            let lit0 = clause[0];
            let lit1 = clause[1];
            let idx0 = Self::lit_to_watch_idx(lit0);
            let idx1 = Self::lit_to_watch_idx(lit1);
            // Mark as learned with high bit
            let marked_idx = clause_idx | LEARNED_CLAUSE_BIT;
            self.watches[idx0].push(WatchInfo {
                clause_idx: marked_idx,
                blocker: lit1,
            });
            self.watches[idx1].push(WatchInfo {
                clause_idx: marked_idx,
                blocker: lit0,
            });
        }
    }

    /// Solve the QBF formula
    pub fn solve(&mut self) -> QbfResult {
        self.solve_with_limit(1_000_000)
    }

    /// Solve with iteration limit (for debugging)
    pub fn solve_with_limit(&mut self, max_iterations: u64) -> QbfResult {
        // Apply initial universal reduction
        self.apply_universal_reduction();

        // Check for empty clause (immediate UNSAT)
        if self.has_empty_clause() {
            return QbfResult::Unsat(Certificate::None);
        }

        let mut iterations: u64 = 0;
        loop {
            iterations += 1;
            if iterations > max_iterations {
                return QbfResult::Unknown;
            }

            // Unit propagation
            match self.propagate() {
                PropResult::Ok => {}
                PropResult::Conflict(clause_idx) => {
                    self.conflicts += 1;

                    if self.decision_level == 0 {
                        // Conflict at level 0 - UNSAT
                        return QbfResult::Unsat(self.build_herbrand_certificate());
                    }

                    // Analyze conflict and learn
                    let (learned_clause, backtrack_level) = self.analyze_conflict(clause_idx);
                    self.bump_clause_activity(&learned_clause);
                    self.var_decay_activity();

                    // Backtrack
                    self.backtrack(backtrack_level);

                    // Add learned clause and its watches
                    if !learned_clause.is_empty() {
                        self.learned.push(learned_clause);
                        let learned_idx = self.learned.len() - 1;
                        self.add_learned_watches(learned_idx);
                    }

                    // Continue to propagate the learned clause before deciding
                    continue;
                }
            }

            // Check if all variables assigned
            if self.all_assigned() {
                // Check if formula is satisfied
                if self.is_satisfied() {
                    return QbfResult::Sat(self.build_skolem_certificate());
                } else {
                    // Should not happen with correct propagation
                    return QbfResult::Unknown;
                }
            }

            // Check for partial solution (all clauses satisfied but not all vars assigned)
            // This is a "solution" state where we can learn a cube
            if self.is_satisfied() {
                // All clauses satisfied - existential player wins for this universal path
                // Learn a cube to block this universal search path
                if let Some(cube_result) = self.learn_cube_from_solution() {
                    match cube_result {
                        CubeResult::Learned(backtrack_level) => {
                            self.backtrack(backtrack_level);
                            continue;
                        }
                        CubeResult::Solved => {
                            // All universal paths lead to SAT
                            return QbfResult::Sat(self.build_skolem_certificate());
                        }
                    }
                }
            }

            // Make a decision
            match self.decide() {
                Some(_) => {
                    self.decisions += 1;
                }
                None => {
                    // No more decisions possible but not all assigned?
                    // This shouldn't happen
                    return QbfResult::Unknown;
                }
            }
        }
    }

    /// Apply universal reduction to all clauses
    fn apply_universal_reduction(&mut self) {
        let reduced: Vec<Vec<Literal>> = self
            .formula
            .clauses
            .iter()
            .map(|c| self.formula.universal_reduce(c))
            .collect();
        self.formula.clauses = reduced;
    }

    /// Check if any clause is empty
    fn has_empty_clause(&self) -> bool {
        self.formula.clauses.iter().any(|c| c.is_empty())
            || self.learned.iter().any(|c| c.is_empty())
    }

    /// Get variable assignment
    fn value(&self, var: u32) -> Assignment {
        if var > 0 && (var as usize) <= self.assignments.len() {
            self.assignments[var as usize - 1]
        } else {
            Assignment::Unassigned
        }
    }

    /// Get literal value
    fn lit_value(&self, lit: Literal) -> Assignment {
        let var_val = self.value(lit.variable().0);
        match var_val {
            Assignment::Unassigned => Assignment::Unassigned,
            Assignment::True => {
                if lit.is_positive() {
                    Assignment::True
                } else {
                    Assignment::False
                }
            }
            Assignment::False => {
                if lit.is_positive() {
                    Assignment::False
                } else {
                    Assignment::True
                }
            }
        }
    }

    /// Assign a variable
    fn assign(&mut self, lit: Literal, reason: Reason) {
        let var = lit.variable().0;
        let value = if lit.is_positive() {
            Assignment::True
        } else {
            Assignment::False
        };

        self.assignments[var as usize - 1] = value;
        self.levels[var as usize - 1] = self.decision_level;
        self.reasons[var as usize - 1] = reason;
        self.trail.push(lit);
    }

    /// Unassign variables back to a given trail position
    fn unassign_to(&mut self, pos: usize) {
        while self.trail.len() > pos {
            let lit = self.trail.pop().unwrap();
            let var = lit.variable().0;
            self.assignments[var as usize - 1] = Assignment::Unassigned;
        }
    }

    /// Check if all variables are assigned
    fn all_assigned(&self) -> bool {
        self.assignments.iter().all(|a| a.is_assigned())
    }

    /// Check if formula is satisfied under current assignment
    fn is_satisfied(&self) -> bool {
        // Check all original clauses
        for clause in &self.formula.clauses {
            if !self.clause_satisfied(clause) {
                return false;
            }
        }
        // Check all learned clauses
        for clause in &self.learned {
            if !self.clause_satisfied(clause) {
                return false;
            }
        }
        true
    }

    /// Check if a clause is satisfied
    fn clause_satisfied(&self, clause: &[Literal]) -> bool {
        clause
            .iter()
            .any(|&lit| self.lit_value(lit) == Assignment::True)
    }

    /// Unit propagation using two-watched literals
    ///
    /// For QBF, we use a hybrid approach:
    /// 1. Track two watched literals per clause (for clauses with >= 2 literals)
    /// 2. When a literal becomes false, check only clauses watching it
    /// 3. Try to find a new watch; if not possible, do full QBF analysis
    fn propagate(&mut self) -> PropResult {
        // First, handle unit clauses (not covered by 2WL)
        if let Some(conflict) = self.propagate_unit_clauses() {
            return PropResult::Conflict(conflict);
        }

        // Main propagation loop using watched literals
        while self.qhead < self.trail.len() {
            let false_lit = self.trail[self.qhead];
            self.qhead += 1;

            // Check clauses watching the negation of this literal
            // (those clauses now have one of their watched literals false)
            let watch_idx = Self::lit_to_watch_idx(false_lit.negated());

            // Process watches - we'll rebuild the watch list as we go
            let mut i = 0;
            while i < self.watches[watch_idx].len() {
                let watch = self.watches[watch_idx][i];
                let is_learned = (watch.clause_idx & LEARNED_CLAUSE_BIT) != 0;
                let clause_idx = watch.clause_idx & !LEARNED_CLAUSE_BIT;

                // Quick check: if blocker is true, clause is satisfied
                if self.lit_value(watch.blocker) == Assignment::True {
                    i += 1;
                    continue;
                }

                // Get the clause
                let clause = if is_learned {
                    &self.learned[clause_idx]
                } else {
                    &self.formula.clauses[clause_idx]
                };

                // Find the positions of the two watched literals
                let watched_lit = false_lit.negated();
                let (w0_pos, w1_pos) =
                    self.find_watch_positions(clause, watched_lit, watch.blocker);

                // Try to find a new watch
                if let Some(new_watch) = self.find_new_watch(clause, w0_pos, w1_pos) {
                    // Found a new watch - update
                    let new_watch_lit = clause[new_watch];
                    let new_watch_idx = Self::lit_to_watch_idx(new_watch_lit);

                    // Remove from current watch list
                    self.watches[watch_idx].swap_remove(i);

                    // Add to new watch list
                    self.watches[new_watch_idx].push(WatchInfo {
                        clause_idx: watch.clause_idx,
                        blocker: watch.blocker,
                    });

                    // Don't increment i - we swapped in a new element
                    continue;
                }

                // No new watch found - do full QBF clause analysis
                let global_clause_idx = if is_learned {
                    self.formula.clauses.len() + clause_idx
                } else {
                    clause_idx
                };

                match self.check_clause_unit(clause, is_learned) {
                    ClauseStatus::Satisfied => {
                        // Shouldn't normally happen if blocker check worked
                        i += 1;
                    }
                    ClauseStatus::Falsified => {
                        return PropResult::Conflict(global_clause_idx);
                    }
                    ClauseStatus::UniversallyBlocked => {
                        return PropResult::Conflict(global_clause_idx);
                    }
                    ClauseStatus::Unit(prop_lit) => {
                        self.propagations += 1;
                        self.assign(prop_lit, Reason::Propagated(global_clause_idx));
                        i += 1;
                    }
                    ClauseStatus::Unresolved => {
                        i += 1;
                    }
                }
            }
        }

        PropResult::Ok
    }

    /// Propagate unit clauses (not covered by 2WL)
    fn propagate_unit_clauses(&mut self) -> Option<usize> {
        // Check original unit clauses
        for i in 0..self.formula.clauses.len() {
            let clause = &self.formula.clauses[i];
            if clause.len() == 1 {
                let lit = clause[0];
                match self.lit_value(lit) {
                    Assignment::True => {}
                    Assignment::False => return Some(i),
                    Assignment::Unassigned => {
                        if self.formula.lit_is_existential(lit) {
                            self.propagations += 1;
                            self.assign(lit, Reason::Propagated(i));
                        } else {
                            // Universal unit clause - blocked
                            return Some(i);
                        }
                    }
                }
            } else if clause.is_empty() {
                return Some(i);
            }
        }

        // Check learned unit clauses
        for i in 0..self.learned.len() {
            let clause = &self.learned[i];
            if clause.len() == 1 {
                let lit = clause[0];
                let global_idx = self.formula.clauses.len() + i;
                match self.lit_value(lit) {
                    Assignment::True => {}
                    Assignment::False => return Some(global_idx),
                    Assignment::Unassigned => {
                        // Learned clauses can propagate universals
                        self.propagations += 1;
                        self.assign(lit, Reason::Propagated(global_idx));
                    }
                }
            } else if clause.is_empty() {
                return Some(self.formula.clauses.len() + i);
            }
        }

        // Check cubes for propagation
        // Cubes are stored as negated literals, so they propagate like clauses
        // A cube (¬u1 ∨ ¬u2) propagates when all but one literal is false
        // (i.e., u1 is true and u2 is unassigned → propagate ¬u2 = u2 = false)
        self.propagate_cubes()
    }

    /// Propagate cubes
    ///
    /// Cubes are stored as disjunctions of negated literals.
    /// When all but one literal is false (original literal is true),
    /// the remaining literal must be true (original must be false).
    fn propagate_cubes(&mut self) -> Option<usize> {
        for i in 0..self.cubes.len() {
            let cube = &self.cubes[i];
            if cube.is_empty() {
                // Empty cube means formula is SAT (handled elsewhere)
                continue;
            }

            // Check cube status
            let mut num_false = 0;
            let mut num_unassigned = 0;
            let mut unassigned_lit = None;
            let mut has_true = false;

            for &lit in cube {
                match self.lit_value(lit) {
                    Assignment::True => {
                        has_true = true;
                        break;
                    }
                    Assignment::False => {
                        num_false += 1;
                    }
                    Assignment::Unassigned => {
                        num_unassigned += 1;
                        unassigned_lit = Some(lit);
                    }
                }
            }

            if has_true {
                // Cube is satisfied, no propagation needed
                continue;
            }

            if num_unassigned == 0 && num_false == cube.len() {
                // All literals false - this is a conflict!
                // The cube says "at least one of these must be true"
                // But all are false - should not happen if cube is correct
                // This indicates SAT (the existential always wins)
                continue;
            }

            if num_unassigned == 1 && num_false == cube.len() - 1 {
                // Unit propagation: one literal must be true
                // This forces a universal variable
                let lit = unassigned_lit.unwrap();
                self.propagations += 1;
                // Use a special reason marker for cube propagation
                // We'll use a large value that's distinct from clause indices
                let reason_idx = usize::MAX / 2 + i;
                self.assign(lit, Reason::Propagated(reason_idx));
            }
        }

        None
    }

    /// Find positions of two watched literals in a clause
    fn find_watch_positions(&self, clause: &[Literal], w0: Literal, w1: Literal) -> (usize, usize) {
        let mut pos0 = 0;
        let mut pos1 = 1;
        for (i, &lit) in clause.iter().enumerate() {
            if lit == w0 {
                pos0 = i;
            } else if lit == w1 {
                pos1 = i;
            }
        }
        (pos0, pos1)
    }

    /// Try to find a new literal to watch (not at positions w0_pos or w1_pos)
    fn find_new_watch(&self, clause: &[Literal], w0_pos: usize, w1_pos: usize) -> Option<usize> {
        for (i, &lit) in clause.iter().enumerate() {
            if i == w0_pos || i == w1_pos {
                continue;
            }
            // Can watch if not false
            if self.lit_value(lit) != Assignment::False {
                return Some(i);
            }
        }
        None
    }

    /// Check clause status for unit propagation
    ///
    /// Key insight for QBF: A clause is "universally blocked" if:
    /// - No existential literals are unassigned
    /// - No literals are satisfied
    /// - Universal literals remain unassigned
    /// - The clause is NOT a tautology (doesn't contain both x and ¬x)
    ///
    /// For learned clauses, a single universal literal CAN be propagated because
    /// it represents a constraint the SAT player derived. For original clauses,
    /// single universal literals are blocked (adversary controls them).
    fn check_clause_unit(&self, clause: &[Literal], is_learned: bool) -> ClauseStatus {
        let mut unassigned_existential = None;
        let mut num_unassigned_exist = 0;
        let mut unassigned_univ_lits = Vec::new();
        let mut has_satisfied = false;

        for &lit in clause {
            match self.lit_value(lit) {
                Assignment::True => {
                    has_satisfied = true;
                    break;
                }
                Assignment::False => {}
                Assignment::Unassigned => {
                    if self.formula.lit_is_existential(lit) {
                        num_unassigned_exist += 1;
                        unassigned_existential = Some(lit);
                    } else {
                        unassigned_univ_lits.push(lit);
                    }
                }
            }
        }

        if has_satisfied {
            ClauseStatus::Satisfied
        } else if num_unassigned_exist == 0 && unassigned_univ_lits.is_empty() {
            // All literals are false
            ClauseStatus::Falsified
        } else if num_unassigned_exist == 0 && unassigned_univ_lits.len() == 1 && is_learned {
            // Single universal literal in LEARNED clause - propagate it!
            // This represents a constraint the SAT player derived.
            ClauseStatus::Unit(unassigned_univ_lits[0])
        } else if num_unassigned_exist == 0 && !unassigned_univ_lits.is_empty() {
            // Universal literals remain - check if UNSAT player can falsify
            // They CAN'T falsify if the clause is a tautology (contains both x and ¬x)
            let is_tautology = self.contains_complementary(&unassigned_univ_lits);
            if is_tautology {
                // Tautology - always satisfied regardless of universal assignment
                ClauseStatus::Satisfied
            } else {
                // Non-tautology with only universals - UNSAT player can falsify
                ClauseStatus::UniversallyBlocked
            }
        } else if num_unassigned_exist == 1 && unassigned_univ_lits.is_empty() {
            // Single existential literal - can propagate
            ClauseStatus::Unit(unassigned_existential.unwrap())
        } else {
            ClauseStatus::Unresolved
        }
    }

    /// Check if a set of literals contains a complementary pair (x and ¬x)
    fn contains_complementary(&self, lits: &[Literal]) -> bool {
        for i in 0..lits.len() {
            for j in (i + 1)..lits.len() {
                // Check if lits[i] and lits[j] are complements (same var, different polarity)
                if lits[i].variable() == lits[j].variable()
                    && lits[i].is_positive() != lits[j].is_positive()
                {
                    return true;
                }
            }
        }
        false
    }

    fn bump_clause_activity(&mut self, clause: &[Literal]) {
        for &lit in clause {
            self.bump_var_activity(lit.variable().0);
        }
    }

    fn bump_var_activity(&mut self, var: u32) {
        let idx = var as usize - 1;
        self.activity[idx] += self.var_inc;
        if self.activity[idx] > 1e100 {
            self.rescale_var_activity();
        }
    }

    fn var_decay_activity(&mut self) {
        self.var_inc *= 1.0 / self.var_decay;
        if self.var_inc > 1e100 {
            self.rescale_var_activity();
        }
    }

    fn rescale_var_activity(&mut self) {
        for act in &mut self.activity {
            *act *= 1e-100;
        }
        self.var_inc *= 1e-100;
    }

    fn pick_branch_var(&self) -> Option<u32> {
        let mut min_level: Option<u32> = None;
        for &var in &self.decision_order {
            if self.value(var) == Assignment::Unassigned {
                let lvl = self.formula.var_level(var);
                min_level = Some(min_level.map_or(lvl, |cur| cur.min(lvl)));
            }
        }
        let min_level = min_level?;

        let mut best_var: Option<u32> = None;
        let mut best_activity = f64::NEG_INFINITY;
        for &var in &self.decision_order {
            if self.value(var) != Assignment::Unassigned || self.formula.var_level(var) != min_level
            {
                continue;
            }

            let act = self.activity[var as usize - 1];
            if best_var.is_none()
                || act > best_activity
                || (act == best_activity && var < best_var.unwrap())
            {
                best_var = Some(var);
                best_activity = act;
            }
        }

        best_var
    }

    /// Make a decision
    fn decide(&mut self) -> Option<Literal> {
        let var = self.pick_branch_var()?;

        // New decision level
        self.decision_level += 1;
        self.trail_lim.push(self.trail.len());

        // Decide: try true first for existential, false for universal
        let polarity = self.formula.is_existential(var);
        let lit = if polarity {
            Literal::positive(Variable(var))
        } else {
            Literal::negative(Variable(var))
        };

        self.assign(lit, Reason::Decision);
        Some(lit)
    }

    /// Backtrack to a given level
    fn backtrack(&mut self, level: u32) {
        if level < self.decision_level {
            let trail_pos = if level == 0 {
                0
            } else {
                self.trail_lim[level as usize - 1]
            };

            self.unassign_to(trail_pos);

            // Reset propagation queue head
            self.qhead = trail_pos;

            // Truncate trail limits
            self.trail_lim.truncate(level as usize);
            self.decision_level = level;
        }
    }

    /// Analyze conflict and return (learned clause, backtrack level)
    ///
    /// Implements Q-resolution based 1-UIP conflict analysis:
    /// 1. Start with the conflict clause
    /// 2. Resolve backwards using reason clauses until reaching 1-UIP
    /// 3. Apply universal reduction at each resolution step (Q-resolution)
    /// 4. The resulting clause has exactly one literal at the current decision level
    ///
    /// This learns more specific clauses than simple decision-negation,
    /// leading to better pruning and faster solving.
    fn analyze_conflict(&mut self, conflict_clause_idx: usize) -> (Vec<Literal>, u32) {
        // Get the conflict clause
        let conflict_clause = if conflict_clause_idx < self.formula.clauses.len() {
            self.formula.clauses[conflict_clause_idx].clone()
        } else {
            self.learned[conflict_clause_idx - self.formula.clauses.len()].clone()
        };

        // Track which variables we've seen during analysis
        let mut seen = vec![false; self.formula.num_vars];

        // The learned clause under construction
        let mut learned: Vec<Literal> = Vec::new();

        // Count of literals at current decision level that need resolution
        let mut counter = 0;

        // Start with the conflict clause
        for &lit in &conflict_clause {
            let var = lit.variable().0;
            let level = self.levels[var as usize - 1];

            if !seen[var as usize - 1] {
                seen[var as usize - 1] = true;
                if level == self.decision_level {
                    counter += 1;
                } else if level > 0 {
                    // Literal from earlier level goes directly to learned clause
                    learned.push(lit.negated());
                }
            }
        }

        // Work backwards through the trail
        let mut trail_idx = self.trail.len();
        let mut asserting_lit: Option<Literal> = None;

        while counter > 0 {
            // Find the next literal on the trail that we've seen
            trail_idx -= 1;
            let lit = self.trail[trail_idx];
            let var = lit.variable().0;

            if !seen[var as usize - 1] {
                continue;
            }

            // This variable was involved in the conflict
            seen[var as usize - 1] = false;
            counter -= 1;

            if counter == 0 {
                // This is the 1-UIP - the asserting literal
                asserting_lit = Some(lit.negated());
                break;
            }

            // Get the reason clause and resolve
            let reason = self.reasons[var as usize - 1];
            if let Reason::Propagated(reason_idx) = reason {
                let reason_clause = if reason_idx < self.formula.clauses.len() {
                    self.formula.clauses[reason_idx].clone()
                } else {
                    self.learned[reason_idx - self.formula.clauses.len()].clone()
                };

                // Add literals from reason clause (except the propagated literal)
                for &reason_lit in &reason_clause {
                    let reason_var = reason_lit.variable().0;
                    if reason_var == var {
                        continue; // Skip the resolved literal
                    }

                    let level = self.levels[reason_var as usize - 1];
                    if !seen[reason_var as usize - 1] {
                        seen[reason_var as usize - 1] = true;
                        if level == self.decision_level {
                            counter += 1;
                        } else if level > 0 {
                            learned.push(reason_lit.negated());
                        }
                    }
                }
            }
            // If reason is Decision, we shouldn't be here with counter > 0
        }

        // Add the asserting literal
        if let Some(lit) = asserting_lit {
            learned.push(lit);
        }

        // Apply universal reduction (Q-resolution)
        // This removes universal literals that are "blocked"
        let reduced = self.formula.universal_reduce(&learned);

        // Compute backtrack level (second highest level in learned clause)
        let backtrack_level = self.compute_backtrack_level(&reduced);

        (reduced, backtrack_level)
    }

    /// Compute the backtrack level for a learned clause
    /// Returns the second-highest decision level (or 0 for unit clauses)
    fn compute_backtrack_level(&self, clause: &[Literal]) -> u32 {
        if clause.is_empty() || clause.len() == 1 {
            return 0;
        }

        let mut max_level = 0;
        let mut second_level = 0;

        for lit in clause {
            let level = self.levels[lit.variable().0 as usize - 1];
            if level > max_level {
                second_level = max_level;
                max_level = level;
            } else if level > second_level && level < max_level {
                second_level = level;
            }
        }

        second_level
    }

    /// Learn a cube from a partial solution state
    ///
    /// Called when all clauses are satisfied but not all variables are assigned.
    /// This means the existential player has a winning strategy for the current
    /// universal assignments. We learn a cube (conjunction) representing this.
    ///
    /// The cube blocks this universal search path - the universal player cannot
    /// make these same choices and expect to win.
    fn learn_cube_from_solution(&mut self) -> Option<CubeResult> {
        // Find the first unassigned universal variable (if any)
        // If no unassigned universals, the existential player wins for all paths
        let has_unassigned_universal = self.decision_order.iter().any(|&var| {
            self.value(var) == Assignment::Unassigned && self.formula.is_universal(var)
        });

        if !has_unassigned_universal {
            // All universal variables are assigned, and formula is satisfied
            // This is a complete solution - SAT
            return Some(CubeResult::Solved);
        }

        // Build the cube from universal decisions on the trail
        // The cube contains the universal literals that led to this solution
        let cube: Vec<Literal> = self
            .trail
            .iter()
            .filter(|lit| self.formula.lit_is_universal(**lit))
            .copied()
            .collect();

        if cube.is_empty() {
            // No universal decisions made yet - can't learn useful cube
            return None;
        }

        // Apply existential reduction (dual of universal reduction)
        // Remove existential literals that are "outer" to the universals
        let reduced_cube = self.existential_reduce_cube(&cube);

        if reduced_cube.is_empty() {
            // Cube reduced to empty - formula is SAT
            return Some(CubeResult::Solved);
        }

        // Compute backtrack level for the cube
        // We backtrack to the second-highest level and flip the universal decision
        let backtrack_level = self.compute_cube_backtrack_level(&reduced_cube);

        // Add the cube (stored as negated literals for propagation)
        // A cube C = (l1 ∧ l2 ∧ ... ∧ ln) is blocked when any li is false
        // So we store it as blocking condition: ¬l1 ∨ ¬l2 ∨ ... ∨ ¬ln
        let negated_cube: Vec<Literal> = reduced_cube.iter().map(|l| l.negated()).collect();
        self.cubes.push(negated_cube);

        self.bump_clause_activity(&reduced_cube);
        self.var_decay_activity();

        if self.decision_level == 0 {
            // Cube learned at level 0 - all paths lead to SAT
            return Some(CubeResult::Solved);
        }

        Some(CubeResult::Learned(backtrack_level))
    }

    /// Apply existential reduction to a cube
    ///
    /// This is the dual of universal reduction:
    /// Remove existential literals whose level is >= the minimum universal level.
    /// These existential choices don't affect the universal winning strategy.
    fn existential_reduce_cube(&self, cube: &[Literal]) -> Vec<Literal> {
        // Find the minimum universal level in the cube
        let min_univ_level = cube
            .iter()
            .filter(|lit| self.formula.lit_is_universal(**lit))
            .map(|lit| self.formula.lit_level(*lit))
            .min();

        match min_univ_level {
            Some(min_level) => {
                cube.iter()
                    .filter(|lit| {
                        // Keep universal literals and existential literals with level < min_univ
                        self.formula.lit_is_universal(**lit)
                            || self.formula.lit_level(**lit) < min_level
                    })
                    .copied()
                    .collect()
            }
            None => {
                // No universal literals - keep everything
                cube.to_vec()
            }
        }
    }

    /// Compute backtrack level for a cube
    /// Similar to clause backtrack, but we want to flip a universal decision
    fn compute_cube_backtrack_level(&self, cube: &[Literal]) -> u32 {
        if cube.is_empty() || cube.len() == 1 {
            return 0;
        }

        // Find the second-highest decision level among universal variables in cube
        let mut levels: Vec<u32> = cube
            .iter()
            .filter(|lit| self.formula.lit_is_universal(**lit))
            .map(|lit| self.levels[lit.variable().0 as usize - 1])
            .collect();

        levels.sort_unstable();
        levels.dedup();

        if levels.len() < 2 {
            return levels.first().copied().unwrap_or(0).saturating_sub(1);
        }

        // Return second-highest level
        levels[levels.len() - 2]
    }

    /// Build Skolem certificate for SAT result
    fn build_skolem_certificate(&self) -> Certificate {
        let mut functions = Vec::new();
        for &var in &self.decision_order {
            if self.formula.is_existential(var) {
                let value = self.value(var).to_bool().unwrap_or(false);
                functions.push(SkolemFunction {
                    variable: var,
                    value,
                });
            }
        }
        Certificate::Skolem(functions)
    }

    /// Build Herbrand certificate for UNSAT result
    fn build_herbrand_certificate(&self) -> Certificate {
        let mut functions = Vec::new();
        for &var in &self.decision_order {
            if self.formula.is_universal(var) {
                let value = self.value(var).to_bool().unwrap_or(false);
                functions.push(HerbrandFunction {
                    variable: var,
                    value,
                });
            }
        }
        Certificate::Herbrand(functions)
    }

    /// Get statistics
    pub fn stats(&self) -> QbfStats {
        QbfStats {
            conflicts: self.conflicts,
            propagations: self.propagations,
            decisions: self.decisions,
            learned_clauses: self.learned.len(),
            learned_cubes: self.cubes.len(),
        }
    }
}

/// Result of propagation
enum PropResult {
    Ok,
    Conflict(usize),
}

/// Result of cube learning
enum CubeResult {
    /// Learned a cube, backtrack to this level
    Learned(u32),
    /// Formula is solved (SAT for all universal paths)
    Solved,
}

/// Status of a clause during propagation
enum ClauseStatus {
    Satisfied,
    Falsified,
    /// Only universal literals remain unassigned - UNSAT player wins
    UniversallyBlocked,
    Unit(Literal),
    Unresolved,
}

/// QBF solver statistics
#[derive(Debug, Clone, Default)]
pub struct QbfStats {
    /// Number of conflicts
    pub conflicts: u64,
    /// Number of propagations
    pub propagations: u64,
    /// Number of decisions
    pub decisions: u64,
    /// Number of learned clauses
    pub learned_clauses: usize,
    /// Number of learned cubes
    pub learned_cubes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_qdimacs;

    #[test]
    fn test_simple_sat_qbf() {
        // ∃x. x
        // This is SAT: just set x = true
        let input = "p cnf 1 1\ne 1 0\n1 0\n";
        let formula = parse_qdimacs(input).unwrap();
        let mut solver = QbfSolver::new(formula);
        let result = solver.solve();
        assert!(matches!(result, QbfResult::Sat(_)));
    }

    #[test]
    fn test_simple_unsat_qbf() {
        // ∃x. (x ∧ ¬x)
        // This is UNSAT
        let input = "p cnf 1 2\ne 1 0\n1 0\n-1 0\n";
        let formula = parse_qdimacs(input).unwrap();
        let mut solver = QbfSolver::new(formula);
        let result = solver.solve();
        assert!(matches!(result, QbfResult::Unsat(_)));
    }

    #[test]
    fn test_universal_sat() {
        // ∀x. (x ∨ ¬x)
        // This is SAT (tautology)
        let input = "p cnf 1 1\na 1 0\n1 -1 0\n";
        let formula = parse_qdimacs(input).unwrap();
        let mut solver = QbfSolver::new(formula);
        let result = solver.solve();
        assert!(matches!(result, QbfResult::Sat(_)));
    }

    #[test]
    fn test_universal_unsat() {
        // ∀x. x
        // This is UNSAT: when x = false, clause is false
        let input = "p cnf 1 1\na 1 0\n1 0\n";
        let formula = parse_qdimacs(input).unwrap();
        let mut solver = QbfSolver::new(formula);
        let result = solver.solve();
        assert!(matches!(result, QbfResult::Unsat(_)));
    }

    #[test]
    fn test_exists_forall_sat() {
        // ∃x∀y. (x ∨ y) ∧ (x ∨ ¬y)
        // SAT: set x = true, then both clauses satisfied regardless of y
        let input = "p cnf 2 2\ne 1 0\na 2 0\n1 2 0\n1 -2 0\n";
        let formula = parse_qdimacs(input).unwrap();
        let mut solver = QbfSolver::new(formula);
        let result = solver.solve();
        assert!(matches!(result, QbfResult::Sat(_)));
    }

    #[test]
    fn test_exists_forall_unsat() {
        // ∃x∀y. (x ∨ y) ∧ (¬x ∨ ¬y)
        // UNSAT:
        // - If x = true, adversary sets y = true, second clause false
        // - If x = false, adversary sets y = false, first clause false
        let input = "p cnf 2 2\ne 1 0\na 2 0\n1 2 0\n-1 -2 0\n";
        let formula = parse_qdimacs(input).unwrap();
        let mut solver = QbfSolver::new(formula);
        let result = solver.solve();
        assert!(matches!(result, QbfResult::Unsat(_)));
    }

    #[test]
    fn test_forall_exists_sat() {
        // ∀x∃y. (x ∨ y) ∧ (¬x ∨ ¬y)
        // SAT: for any x, set y = ¬x
        // - If x = true, set y = false: (T∨F)∧(F∨T) = T∧T = T
        // - If x = false, set y = true: (F∨T)∧(T∨F) = T∧T = T
        let input = "p cnf 2 2\na 1 0\ne 2 0\n1 2 0\n-1 -2 0\n";
        let formula = parse_qdimacs(input).unwrap();
        let mut solver = QbfSolver::new(formula);
        let result = solver.solve();
        assert!(matches!(result, QbfResult::Sat(_)));
    }

    #[test]
    fn test_universal_reduction() {
        // ∃x∀y. (x ∨ y)
        // After universal reduction of y (level 1 >= max_exist 0), clause becomes (x)
        // SAT: set x = true
        let input = "p cnf 2 1\ne 1 0\na 2 0\n1 2 0\n";
        let formula = parse_qdimacs(input).unwrap();
        let mut solver = QbfSolver::new(formula);
        let result = solver.solve();
        assert!(matches!(result, QbfResult::Sat(_)));
    }

    #[test]
    fn test_stats() {
        let input = "p cnf 2 2\ne 1 0\na 2 0\n1 2 0\n-1 -2 0\n";
        let formula = parse_qdimacs(input).unwrap();
        let mut solver = QbfSolver::new(formula);
        solver.solve();
        let stats = solver.stats();
        // Should have some activity
        assert!(stats.decisions > 0 || stats.propagations > 0 || stats.conflicts > 0);
    }

    #[test]
    fn test_three_quantifier_blocks_sat() {
        // ∃x∀y∃z. (x ∨ y ∨ z) ∧ (¬x ∨ ¬y ∨ z) ∧ (x ∨ ¬y ∨ ¬z)
        // SAT: set x = true, z = true
        // For any y:
        //   y=T: (T∨T∨T) ∧ (F∨F∨T) ∧ (T∨F∨F) = T ∧ T ∧ T = T
        //   y=F: (T∨F∨T) ∧ (F∨T∨T) ∧ (T∨T∨F) = T ∧ T ∧ T = T
        let input = r#"
p cnf 3 3
e 1 0
a 2 0
e 3 0
1 2 3 0
-1 -2 3 0
1 -2 -3 0
"#;
        let formula = parse_qdimacs(input).unwrap();
        let mut solver = QbfSolver::new(formula);
        let result = solver.solve();
        assert!(matches!(result, QbfResult::Sat(_)));
    }

    #[test]
    fn test_three_quantifier_blocks_unsat() {
        // ∃x∀y∃z. (x ∨ y) ∧ (¬x ∨ ¬y) ∧ (¬z)
        // With z forced false, we need x to satisfy (x∨y) and (¬x∨¬y) for all y
        // But that's the same as test_exists_forall_unsat (x handles y)
        // Actually let's make a simpler UNSAT case:
        // ∃x∀y∃z. (y) ∧ (¬y)
        // This is UNSAT because for y=T or y=F, one clause fails
        let input = r#"
p cnf 3 2
e 1 0
a 2 0
e 3 0
2 0
-2 0
"#;
        let formula = parse_qdimacs(input).unwrap();
        let mut solver = QbfSolver::new(formula);
        let result = solver.solve();
        assert!(matches!(result, QbfResult::Unsat(_)));
    }

    #[test]
    fn test_multiple_existential_per_block() {
        // ∃x₁x₂∀y. (x₁ ∨ x₂ ∨ y) ∧ (x₁ ∨ x₂ ∨ ¬y)
        // SAT: set x₁ = true (or x₂ = true)
        let input = r#"
p cnf 3 2
e 1 2 0
a 3 0
1 2 3 0
1 2 -3 0
"#;
        let formula = parse_qdimacs(input).unwrap();
        let mut solver = QbfSolver::new(formula);
        let result = solver.solve();
        assert!(matches!(result, QbfResult::Sat(_)));
    }

    #[test]
    fn test_multiple_universal_per_block() {
        // ∃x∀y₁y₂. (x ∨ y₁ ∨ y₂) ∧ (x ∨ y₁ ∨ ¬y₂) ∧ (x ∨ ¬y₁ ∨ y₂) ∧ (x ∨ ¬y₁ ∨ ¬y₂)
        // SAT: set x = true (satisfies all clauses regardless of y₁, y₂)
        let input = r#"
p cnf 3 4
e 1 0
a 2 3 0
1 2 3 0
1 2 -3 0
1 -2 3 0
1 -2 -3 0
"#;
        let formula = parse_qdimacs(input).unwrap();
        let mut solver = QbfSolver::new(formula);
        let result = solver.solve();
        assert!(matches!(result, QbfResult::Sat(_)));
    }

    #[test]
    fn test_forall_exists_unsat_dependency() {
        // ∀x∃y. (x ∨ y) ∧ (¬x ∨ y) ∧ (x ∨ ¬y) ∧ (¬x ∨ ¬y)
        // UNSAT: y cannot satisfy all clauses for all x
        // x=T: need y for (¬x∨y), need ¬y for (x∨¬y) - contradiction
        // x=F: need y for (x∨y), need ¬y for (¬x∨¬y) - contradiction
        let input = r#"
p cnf 2 4
a 1 0
e 2 0
1 2 0
-1 2 0
1 -2 0
-1 -2 0
"#;
        let formula = parse_qdimacs(input).unwrap();
        let mut solver = QbfSolver::new(formula);
        let result = solver.solve_with_limit(100);
        assert!(
            matches!(result, QbfResult::Unsat(_)),
            "Expected Unsat, got {:?}",
            result
        );
    }

    #[test]
    fn test_cube_learning_basic() {
        // Test that cube learning correctly handles partial solutions
        // ∃x∀y∀z. (x ∨ y) ∧ (x ∨ ¬y) ∧ (x ∨ z) ∧ (x ∨ ¬z)
        // SAT: x = true satisfies all clauses regardless of y, z
        // This should trigger cube learning when x=true makes everything SAT
        let input = r#"
p cnf 3 4
e 1 0
a 2 3 0
1 2 0
1 -2 0
1 3 0
1 -3 0
"#;
        let formula = parse_qdimacs(input).unwrap();
        let mut solver = QbfSolver::new(formula);
        let result = solver.solve();
        assert!(matches!(result, QbfResult::Sat(_)));

        // Check that cube learning happened
        let stats = solver.stats();
        assert!(
            stats.learned_cubes > 0 || stats.decisions <= 2,
            "Expected cube learning or quick SAT detection"
        );
    }

    #[test]
    fn test_cube_learning_multiple_universals() {
        // Test cube learning with multiple universal variables
        // ∃x₁x₂∀y₁y₂. (x₁ ∨ y₁) ∧ (x₁ ∨ ¬y₁) ∧ (x₂ ∨ y₂) ∧ (x₂ ∨ ¬y₂)
        // SAT: x₁ = true, x₂ = true
        // The solver should learn cubes to avoid exploring all y₁, y₂ combinations
        let input = r#"
p cnf 4 4
e 1 2 0
a 3 4 0
1 3 0
1 -3 0
2 4 0
2 -4 0
"#;
        let formula = parse_qdimacs(input).unwrap();
        let mut solver = QbfSolver::new(formula);
        let result = solver.solve();
        assert!(matches!(result, QbfResult::Sat(_)));
    }

    #[test]
    fn test_cube_propagation() {
        // Test that cube propagation works correctly
        // ∃x∀y∀z. (x ∨ y ∨ z) ∧ (x ∨ y ∨ ¬z) ∧ (x ∨ ¬y ∨ z) ∧ (x ∨ ¬y ∨ ¬z)
        // SAT: x = true satisfies all
        // After x = true, z = true leads to SAT → cube (z) learned
        // Cube propagation should then try z = false automatically
        let input = r#"
p cnf 3 4
e 1 0
a 2 3 0
1 2 3 0
1 2 -3 0
1 -2 3 0
1 -2 -3 0
"#;
        let formula = parse_qdimacs(input).unwrap();
        let mut solver = QbfSolver::new(formula);
        let result = solver.solve();
        assert!(matches!(result, QbfResult::Sat(_)));

        let stats = solver.stats();
        // With cube learning and propagation, we should solve efficiently
        assert!(
            stats.decisions <= 10,
            "Too many decisions: {}",
            stats.decisions
        );
    }
}
