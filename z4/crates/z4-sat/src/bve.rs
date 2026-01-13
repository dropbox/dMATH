//! Bounded Variable Elimination (BVE)
//!
//! Implements variable elimination as an inprocessing technique.
//! For a variable x, we can eliminate it by:
//! 1. Collecting all clauses containing x (positive occurrences)
//! 2. Collecting all clauses containing ~x (negative occurrences)
//! 3. Computing all resolvents between positive and negative clauses
//! 4. If the total size of resolvents <= original clauses, eliminate x
//!
//! The "bounded" part ensures we only eliminate if it doesn't increase the
//! formula size too much (bounded by a growth limit).
//!
//! Reference: Een & Biere, "Effective Preprocessing in SAT through Variable
//! and Clause Elimination", SAT 2005.

use crate::clause_db::ClauseDB;
use crate::literal::{Literal, Variable};

/// Maximum number of clauses that can be added by eliminating a variable
/// (relative to the number of clauses removed)
const BVE_GROWTH_LIMIT: usize = 0;

/// Maximum number of occurrences for a variable to be considered for elimination
/// (to avoid combinatorial explosion)
const MAX_OCCURRENCES: usize = 10;

/// Statistics for BVE operations
#[derive(Debug, Clone, Default)]
pub struct BVEStats {
    /// Number of variables eliminated
    pub vars_eliminated: u64,
    /// Number of clauses removed (before resolvents added)
    pub clauses_removed: u64,
    /// Number of resolvents added
    pub resolvents_added: u64,
    /// Number of tautological resolvents skipped
    pub tautologies_skipped: u64,
    /// Number of elimination rounds
    pub rounds: u64,
}

/// Result of attempting to eliminate a variable
#[derive(Debug, Clone)]
pub struct EliminationResult {
    /// The variable that was eliminated
    pub variable: Variable,
    /// Indices of clauses to delete (containing the eliminated variable)
    pub to_delete: Vec<usize>,
    /// New resolvents to add
    pub resolvents: Vec<Vec<Literal>>,
    /// Whether elimination was performed
    pub eliminated: bool,
}

/// Occurrence list for BVE - tracks which clauses contain each literal
#[derive(Debug, Clone)]
pub struct BVEOccList {
    /// For each literal index, list of clause indices containing that literal
    occ: Vec<Vec<usize>>,
}

impl BVEOccList {
    /// Create a new occurrence list for n variables
    pub fn new(num_vars: usize) -> Self {
        BVEOccList {
            occ: vec![Vec::new(); num_vars * 2],
        }
    }

    /// Ensure the occurrence list can index literals for `num_vars` variables.
    pub fn ensure_num_vars(&mut self, num_vars: usize) {
        let target = num_vars.saturating_mul(2);
        if self.occ.len() < target {
            self.occ.resize_with(target, Vec::new);
        }
    }

    /// Add a clause to occurrence lists
    pub fn add_clause(&mut self, clause_idx: usize, literals: &[Literal]) {
        for &lit in literals {
            let idx = lit.index();
            if idx < self.occ.len() {
                self.occ[idx].push(clause_idx);
            }
        }
    }

    /// Get clauses containing a literal
    pub fn get(&self, lit: Literal) -> &[usize] {
        let idx = lit.index();
        if idx < self.occ.len() {
            &self.occ[idx]
        } else {
            &[]
        }
    }

    /// Get number of clauses containing a literal
    pub fn count(&self, lit: Literal) -> usize {
        self.get(lit).len()
    }

    /// Clear all occurrence lists
    pub fn clear(&mut self) {
        for list in &mut self.occ {
            list.clear();
        }
    }

    /// Remove a clause from occurrence lists
    pub fn remove_clause(&mut self, clause_idx: usize, literals: &[Literal]) {
        for &lit in literals {
            let idx = lit.index();
            if idx < self.occ.len() {
                if let Some(pos) = self.occ[idx].iter().position(|&c| c == clause_idx) {
                    self.occ[idx].swap_remove(pos);
                }
            }
        }
    }
}

/// Bounded Variable Elimination engine
pub struct BVE {
    /// Occurrence lists
    occ: BVEOccList,
    /// Statistics
    stats: BVEStats,
    /// Number of variables
    num_vars: usize,
    /// Variables that have been eliminated (cannot be eliminated again)
    eliminated: Vec<bool>,
    /// Temporary buffer for resolvent computation
    resolvent_buf: Vec<Literal>,
    /// Temporary mark array for tautology checking
    marks: Vec<i8>, // 0 = unmarked, 1 = positive, -1 = negative
}

struct ResolveAcc<'a> {
    clauses_removed: usize,
    resolvents: &'a mut Vec<Vec<Literal>>,
    total_literals_added: &'a mut usize,
    found_empty_resolvent: &'a mut bool,
}

impl BVE {
    /// Create a new BVE engine for n variables
    pub fn new(num_vars: usize) -> Self {
        BVE {
            occ: BVEOccList::new(num_vars),
            stats: BVEStats::default(),
            num_vars,
            eliminated: vec![false; num_vars],
            resolvent_buf: Vec::new(),
            marks: vec![0; num_vars],
        }
    }

    /// Ensure internal buffers can handle `num_vars` variables.
    pub fn ensure_num_vars(&mut self, num_vars: usize) {
        if self.num_vars >= num_vars {
            return;
        }
        self.num_vars = num_vars;
        self.occ.ensure_num_vars(num_vars);
        if self.eliminated.len() < num_vars {
            self.eliminated.resize(num_vars, false);
        }
        if self.marks.len() < num_vars {
            self.marks.resize(num_vars, 0);
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &BVEStats {
        &self.stats
    }

    /// Check if a variable has been eliminated
    pub fn is_eliminated(&self, var: Variable) -> bool {
        let idx = var.index();
        idx < self.eliminated.len() && self.eliminated[idx]
    }

    /// Initialize/rebuild occurrence lists from clause database
    pub fn rebuild(&mut self, clauses: &ClauseDB) {
        self.occ.clear();

        for idx in clauses.indices() {
            if clauses.header(idx).is_empty() {
                continue;
            }
            self.occ.add_clause(idx, clauses.literals(idx));
        }
    }

    /// Compute the resolvent of two clauses on a variable
    ///
    /// Returns None if the resolvent is tautological (contains both L and ~L).
    /// Otherwise returns the resolvent clause.
    fn resolve(
        &mut self,
        pos_clause: &[Literal],
        neg_clause: &[Literal],
        var: Variable,
    ) -> Option<Vec<Literal>> {
        self.resolvent_buf.clear();

        // Clear marks
        for &lit in pos_clause {
            self.marks[lit.variable().index()] = 0;
        }
        for &lit in neg_clause {
            self.marks[lit.variable().index()] = 0;
        }

        // Add literals from positive clause (except the pivot)
        for &lit in pos_clause {
            if lit.variable() == var {
                continue;
            }
            let var_idx = lit.variable().index();
            let sign: i8 = if lit.is_positive() { 1 } else { -1 };

            if self.marks[var_idx] == -sign {
                // Tautology: both L and ~L present
                self.stats.tautologies_skipped += 1;
                return None;
            }
            if self.marks[var_idx] == 0 {
                self.marks[var_idx] = sign;
                self.resolvent_buf.push(lit);
            }
            // If marks[var_idx] == sign, literal is duplicate, skip
        }

        // Add literals from negative clause (except the pivot)
        for &lit in neg_clause {
            if lit.variable() == var {
                continue;
            }
            let var_idx = lit.variable().index();
            let sign: i8 = if lit.is_positive() { 1 } else { -1 };

            if self.marks[var_idx] == -sign {
                // Tautology: both L and ~L present
                self.stats.tautologies_skipped += 1;
                return None;
            }
            if self.marks[var_idx] == 0 {
                self.marks[var_idx] = sign;
                self.resolvent_buf.push(lit);
            }
            // If marks[var_idx] == sign, literal is duplicate, skip
        }

        Some(self.resolvent_buf.clone())
    }

    /// Check if eliminating a variable would increase the formula size too much
    ///
    /// Returns (can_eliminate, resolvents) where:
    /// - can_eliminate: true if the elimination is bounded
    /// - resolvents: the computed resolvents (only if can_eliminate is true)
    fn check_bounded_elimination(
        &mut self,
        var: Variable,
        clauses: &ClauseDB,
        gate_defining_clauses: Option<&[usize]>,
    ) -> (bool, Vec<Vec<Literal>>) {
        let pos_lit = Literal::positive(var);
        let neg_lit = Literal::negative(var);

        let pos_clauses: Vec<usize> = self.occ.get(pos_lit).to_vec();
        let neg_clauses: Vec<usize> = self.occ.get(neg_lit).to_vec();

        // Quick bound check: if product of occurrences is too large, skip
        let pos_count = pos_clauses.len();
        let neg_count = neg_clauses.len();

        if pos_count == 0 || neg_count == 0 {
            // Variable is pure - can be eliminated trivially
            // All clauses containing the variable can be removed
            return (true, Vec::new());
        }

        if pos_count > MAX_OCCURRENCES || neg_count > MAX_OCCURRENCES {
            return (false, Vec::new());
        }

        // Number of clauses that would be removed
        let clauses_removed = pos_count + neg_count;

        // Compute all resolvents and count
        let mut resolvents = Vec::new();
        let mut total_literals_added = 0usize;
        let mut total_literals_removed = 0usize;

        // Count literals in removed clauses
        for &c_idx in &pos_clauses {
            if c_idx < clauses.len() && !clauses.header(c_idx).is_empty() {
                total_literals_removed += clauses.header(c_idx).len();
            }
        }
        for &c_idx in &neg_clauses {
            if c_idx < clauses.len() && !clauses.header(c_idx).is_empty() {
                total_literals_removed += clauses.header(c_idx).len();
            }
        }

        let mut pos_gate = Vec::new();
        let mut pos_non_gate = Vec::new();
        let mut neg_gate = Vec::new();
        let mut neg_non_gate = Vec::new();

        if let Some(defining) = gate_defining_clauses {
            for &idx in &pos_clauses {
                if defining.contains(&idx) {
                    pos_gate.push(idx);
                } else {
                    pos_non_gate.push(idx);
                }
            }
            for &idx in &neg_clauses {
                if defining.contains(&idx) {
                    neg_gate.push(idx);
                } else {
                    neg_non_gate.push(idx);
                }
            }
        } else {
            pos_non_gate = pos_clauses.clone();
            neg_non_gate = neg_clauses.clone();
        }

        // Restricted resolution (Een & Biere SAT'05): if a functional gate
        // definition for `var` is known, only resolve between gate clauses and
        // non-gate clauses, skipping gate/gate and non-gate/non-gate pairs.
        let mut found_empty_resolvent = false;
        {
            let mut acc = ResolveAcc {
                clauses_removed,
                resolvents: &mut resolvents,
                total_literals_added: &mut total_literals_added,
                found_empty_resolvent: &mut found_empty_resolvent,
            };

            if gate_defining_clauses.is_some() {
                for &pos_idx in &pos_gate {
                    for &neg_idx in &neg_non_gate {
                        if !self.try_resolve_pair(var, clauses, pos_idx, neg_idx, &mut acc) {
                            return (false, Vec::new());
                        }
                    }
                }
                for &pos_idx in &pos_non_gate {
                    for &neg_idx in &neg_gate {
                        if !self.try_resolve_pair(var, clauses, pos_idx, neg_idx, &mut acc) {
                            return (false, Vec::new());
                        }
                    }
                }
            } else {
                // Full variable elimination resolution.
                for &pos_idx in &pos_non_gate {
                    for &neg_idx in &neg_non_gate {
                        if !self.try_resolve_pair(var, clauses, pos_idx, neg_idx, &mut acc) {
                            return (false, Vec::new());
                        }
                    }
                }
            }
        }

        if found_empty_resolvent {
            return (true, resolvents);
        }

        // Check the growth bound
        // We allow elimination if the number of resolvents <= clauses_removed + growth_limit
        // AND the total literal count doesn't increase too much
        let bounded = resolvents.len() <= clauses_removed + BVE_GROWTH_LIMIT
            && total_literals_added <= total_literals_removed + BVE_GROWTH_LIMIT * 10;

        (bounded, if bounded { resolvents } else { Vec::new() })
    }

    fn try_resolve_pair(
        &mut self,
        var: Variable,
        clauses: &ClauseDB,
        pos_idx: usize,
        neg_idx: usize,
        acc: &mut ResolveAcc<'_>,
    ) -> bool {
        if pos_idx >= clauses.len() || clauses.header(pos_idx).is_empty() {
            return true;
        }
        if neg_idx >= clauses.len() || clauses.header(neg_idx).is_empty() {
            return true;
        }

        let pos_lits = clauses.literals(pos_idx);
        let neg_lits = clauses.literals(neg_idx);

        if let Some(resolvent) = self.resolve(pos_lits, neg_lits, var) {
            if resolvent.is_empty() {
                acc.resolvents.push(resolvent);
                *acc.found_empty_resolvent = true;
                return true;
            }

            *acc.total_literals_added += resolvent.len();
            acc.resolvents.push(resolvent);

            if acc.resolvents.len() > acc.clauses_removed + BVE_GROWTH_LIMIT {
                return false;
            }
        }

        true
    }

    /// Find the best variable to eliminate
    ///
    /// Returns None if no suitable variable is found.
    fn find_elimination_candidate(&self, assignment: &[Option<bool>]) -> Option<Variable> {
        let mut best_var: Option<Variable> = None;
        let mut best_score = usize::MAX;

        for var_idx in 0..self.num_vars {
            let var = Variable(var_idx as u32);

            // Skip eliminated variables
            if self.eliminated[var_idx] {
                continue;
            }

            // Skip assigned variables
            if var_idx < assignment.len() && assignment[var_idx].is_some() {
                continue;
            }

            let pos_lit = Literal::positive(var);
            let neg_lit = Literal::negative(var);

            let pos_count = self.occ.count(pos_lit);
            let neg_count = self.occ.count(neg_lit);

            // Skip variables with too many occurrences
            if pos_count > MAX_OCCURRENCES || neg_count > MAX_OCCURRENCES {
                continue;
            }

            // Score: product of occurrences (lower is better)
            // This estimates the number of resolvents
            let score = pos_count.saturating_mul(neg_count);

            if score < best_score {
                best_score = score;
                best_var = Some(var);
            }
        }

        best_var
    }

    /// Try to eliminate a specific variable
    ///
    /// Returns an EliminationResult describing what should be done.
    pub fn try_eliminate(&mut self, var: Variable, clauses: &ClauseDB) -> EliminationResult {
        self.try_eliminate_with_gate(var, clauses, None)
    }

    /// Try to eliminate a specific variable, optionally using restricted
    /// resolution if `gate_defining_clauses` is provided.
    pub fn try_eliminate_with_gate(
        &mut self,
        var: Variable,
        clauses: &ClauseDB,
        gate_defining_clauses: Option<&[usize]>,
    ) -> EliminationResult {
        let var_idx = var.index();

        // Check if already eliminated
        if var_idx < self.eliminated.len() && self.eliminated[var_idx] {
            return EliminationResult {
                variable: var,
                to_delete: Vec::new(),
                resolvents: Vec::new(),
                eliminated: false,
            };
        }

        // Check if elimination is bounded
        let (can_eliminate, resolvents) =
            self.check_bounded_elimination(var, clauses, gate_defining_clauses);

        if !can_eliminate {
            return EliminationResult {
                variable: var,
                to_delete: Vec::new(),
                resolvents: Vec::new(),
                eliminated: false,
            };
        }

        // Collect clauses to delete
        let pos_lit = Literal::positive(var);
        let neg_lit = Literal::negative(var);

        let mut to_delete = Vec::new();
        for &c_idx in self.occ.get(pos_lit) {
            if c_idx < clauses.len() && !clauses.header(c_idx).is_empty() {
                to_delete.push(c_idx);
            }
        }
        for &c_idx in self.occ.get(neg_lit) {
            if c_idx < clauses.len() && !clauses.header(c_idx).is_empty() {
                to_delete.push(c_idx);
            }
        }

        // Mark variable as eliminated
        if var_idx < self.eliminated.len() {
            self.eliminated[var_idx] = true;
        }

        // Update statistics
        self.stats.vars_eliminated += 1;
        self.stats.clauses_removed += to_delete.len() as u64;
        self.stats.resolvents_added += resolvents.len() as u64;

        EliminationResult {
            variable: var,
            to_delete,
            resolvents,
            eliminated: true,
        }
    }

    /// Run BVE as inprocessing
    ///
    /// Attempts to eliminate variables to simplify the formula.
    /// Should be called at decision level 0 (e.g., after a restart).
    ///
    /// Returns a list of EliminationResults describing the eliminations performed.
    pub fn run_elimination(
        &mut self,
        clauses: &ClauseDB,
        assignment: &[Option<bool>],
        max_eliminations: usize,
    ) -> Vec<EliminationResult> {
        self.run_elimination_with_gate_provider(
            clauses,
            assignment,
            max_eliminations,
            |_var, _pos_occs, _neg_occs, _clauses| None,
        )
    }

    /// Run BVE as inprocessing, optionally providing gate information for
    /// restricted resolution.
    pub fn run_elimination_with_gate_provider<F>(
        &mut self,
        clauses: &ClauseDB,
        assignment: &[Option<bool>],
        max_eliminations: usize,
        mut gate_provider: F,
    ) -> Vec<EliminationResult>
    where
        F: FnMut(Variable, &[usize], &[usize], &ClauseDB) -> Option<Vec<usize>>,
    {
        self.stats.rounds += 1;
        let mut results = Vec::new();
        let mut eliminations = 0;

        while eliminations < max_eliminations {
            // Find a candidate variable
            let var = match self.find_elimination_candidate(assignment) {
                Some(v) => v,
                None => break,
            };

            let pos_lit = Literal::positive(var);
            let neg_lit = Literal::negative(var);
            let pos_occs = self.occ.get(pos_lit).to_vec();
            let neg_occs = self.occ.get(neg_lit).to_vec();
            let gate_defining = gate_provider(var, &pos_occs, &neg_occs, clauses);

            // Try to eliminate it
            let result = self.try_eliminate_with_gate(var, clauses, gate_defining.as_deref());

            if result.eliminated {
                eliminations += 1;

                // Update occurrence lists for deleted clauses
                for &c_idx in &result.to_delete {
                    if c_idx < clauses.len() && !clauses.header(c_idx).is_empty() {
                        self.occ.remove_clause(c_idx, clauses.literals(c_idx));
                    }
                }

                results.push(result);
            } else {
                // Mark as not eliminable to avoid retrying
                let var_idx = var.index();
                if var_idx < self.eliminated.len() {
                    self.eliminated[var_idx] = true;
                }
            }
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lit(var: u32, positive: bool) -> Literal {
        if positive {
            Literal::positive(Variable(var))
        } else {
            Literal::negative(Variable(var))
        }
    }

    #[test]
    fn test_bve_occurrence_list() {
        let mut occ = BVEOccList::new(5);

        let clause1 = vec![lit(0, true), lit(1, false)];
        let clause2 = vec![lit(0, true), lit(2, true)];

        occ.add_clause(0, &clause1);
        occ.add_clause(1, &clause2);

        // lit(0, true) appears in both clauses
        assert_eq!(occ.count(lit(0, true)), 2);
        assert!(occ.get(lit(0, true)).contains(&0));
        assert!(occ.get(lit(0, true)).contains(&1));

        // lit(1, false) appears only in clause 0
        assert_eq!(occ.count(lit(1, false)), 1);
        assert!(occ.get(lit(1, false)).contains(&0));
    }

    #[test]
    fn test_bve_resolve_basic() {
        let mut bve = BVE::new(5);

        // C1 = {x0, x1}, C2 = {~x0, x2}
        // Resolvent on x0 should be {x1, x2}
        let c1 = vec![lit(0, true), lit(1, true)];
        let c2 = vec![lit(0, false), lit(2, true)];

        let result = bve.resolve(&c1, &c2, Variable(0));
        assert!(result.is_some());

        let resolvent = result.unwrap();
        assert_eq!(resolvent.len(), 2);
        assert!(resolvent.contains(&lit(1, true)));
        assert!(resolvent.contains(&lit(2, true)));
    }

    #[test]
    fn test_bve_resolve_tautology() {
        let mut bve = BVE::new(5);

        // C1 = {x0, x1}, C2 = {~x0, ~x1}
        // Resolvent on x0 would be {x1, ~x1} - tautology
        let c1 = vec![lit(0, true), lit(1, true)];
        let c2 = vec![lit(0, false), lit(1, false)];

        let result = bve.resolve(&c1, &c2, Variable(0));
        assert!(result.is_none());
    }

    #[test]
    fn test_bve_resolve_duplicates() {
        let mut bve = BVE::new(5);

        // C1 = {x0, x1, x2}, C2 = {~x0, x1, x3}
        // Resolvent on x0 should be {x1, x2, x3} (x1 appears once)
        let c1 = vec![lit(0, true), lit(1, true), lit(2, true)];
        let c2 = vec![lit(0, false), lit(1, true), lit(3, true)];

        let result = bve.resolve(&c1, &c2, Variable(0));
        assert!(result.is_some());

        let resolvent = result.unwrap();
        assert_eq!(resolvent.len(), 3);
        assert!(resolvent.contains(&lit(1, true)));
        assert!(resolvent.contains(&lit(2, true)));
        assert!(resolvent.contains(&lit(3, true)));
    }

    #[test]
    fn test_bve_pure_literal() {
        let mut bve = BVE::new(5);

        // Clauses: {x0, x1}, {x0, x2}
        // x0 appears only positively - it's pure
        let mut clauses = ClauseDB::new();
        clauses.add(&[lit(0, true), lit(1, true)], false);
        clauses.add(&[lit(0, true), lit(2, true)], false);

        bve.rebuild(&clauses);

        // x0 should be eliminable (pure literal)
        let result = bve.try_eliminate(Variable(0), &clauses);
        assert!(result.eliminated);
        assert_eq!(result.to_delete.len(), 2); // Both clauses removed
        assert_eq!(result.resolvents.len(), 0); // No resolvents (pure)
    }

    #[test]
    fn test_bve_bounded_check() {
        let mut bve = BVE::new(5);

        // Simple case: x0 appears in 2 clauses, elimination should be bounded
        // C0 = {x0, x1}, C1 = {~x0, x2}
        // Resolvent = {x1, x2} - one new clause vs two removed
        let mut clauses = ClauseDB::new();
        clauses.add(&[lit(0, true), lit(1, true)], false);
        clauses.add(&[lit(0, false), lit(2, true)], false);

        bve.rebuild(&clauses);

        let result = bve.try_eliminate(Variable(0), &clauses);
        assert!(result.eliminated);
        assert_eq!(result.to_delete.len(), 2);
        assert_eq!(result.resolvents.len(), 1);
    }

    #[test]
    fn test_bve_not_bounded() {
        let mut bve = BVE::new(10);

        // Create a case where elimination would add too many clauses
        // x0 appears in 3 positive clauses and 3 negative clauses
        // This could create 9 resolvents vs removing 6 clauses - not bounded with limit 0
        let mut clauses = ClauseDB::new();
        clauses.add(&[lit(0, true), lit(1, true)], false);
        clauses.add(&[lit(0, true), lit(2, true)], false);
        clauses.add(&[lit(0, true), lit(3, true)], false);
        clauses.add(&[lit(0, false), lit(4, true)], false);
        clauses.add(&[lit(0, false), lit(5, true)], false);
        clauses.add(&[lit(0, false), lit(6, true)], false);

        bve.rebuild(&clauses);

        let result = bve.try_eliminate(Variable(0), &clauses);
        // Should not be eliminated because 9 resolvents > 6 removed clauses
        assert!(!result.eliminated);
    }

    #[test]
    fn test_bve_run_elimination() {
        let mut bve = BVE::new(5);

        // Simple formula where x0 can be eliminated
        // x0 appears in both polarities (good candidate)
        let mut clauses = ClauseDB::new();
        clauses.add(&[lit(0, true), lit(1, true)], false);
        clauses.add(&[lit(0, false), lit(2, true)], false);
        clauses.add(&[lit(1, true), lit(2, false)], false); // Independent

        bve.rebuild(&clauses);

        let assignment = vec![None; 5];
        let results = bve.run_elimination(&clauses, &assignment, 10);

        // Some variable should be eliminated (could be x0 or other pure literals)
        assert!(!results.is_empty());
        assert!(results.iter().any(|r| r.eliminated));
        // x0 specifically should be eliminable when tried directly
        let mut bve2 = BVE::new(5);
        bve2.rebuild(&clauses);
        let result = bve2.try_eliminate(Variable(0), &clauses);
        assert!(result.eliminated, "x0 should be eliminable");
    }

    #[test]
    fn test_bve_skip_assigned() {
        let mut bve = BVE::new(5);

        let mut clauses = ClauseDB::new();
        clauses.add(&[lit(0, true), lit(1, true)], false);
        clauses.add(&[lit(0, false), lit(2, true)], false);

        bve.rebuild(&clauses);

        // x0 is assigned
        let mut assignment = vec![None; 5];
        assignment[0] = Some(true);

        let candidate = bve.find_elimination_candidate(&assignment);
        // Should not select x0 since it's assigned
        assert!(candidate.is_none() || candidate.unwrap() != Variable(0));
    }

    #[test]
    fn test_bve_stats() {
        let mut bve = BVE::new(5);

        // Create a formula where only x0 can be eliminated
        // x0 appears in both polarities, x1 and x2 also appear in both
        let mut clauses = ClauseDB::new();
        clauses.add(&[lit(0, true), lit(1, true)], false);
        clauses.add(&[lit(0, false), lit(2, true)], false);
        clauses.add(&[lit(1, false), lit(2, false)], false); // Makes x1, x2 also balanced

        bve.rebuild(&clauses);

        let assignment = vec![None; 5];
        let _ = bve.run_elimination(&clauses, &assignment, 10);

        let stats = bve.stats();
        // At least one variable should be eliminated
        assert!(
            stats.vars_eliminated >= 1,
            "At least one var should be eliminated"
        );
        assert!(
            stats.clauses_removed >= 2,
            "At least 2 clauses should be removed"
        );
        assert_eq!(stats.rounds, 1, "Should run 1 round");
    }

    #[test]
    fn test_bve_restricted_resolution_gate_vs_non_gate() {
        let mut bve = BVE::new(4);

        // x0 <-> x1 (gate clauses), plus two non-gate clauses containing x0:
        // (x0 v ~x1) (gate)
        // (~x0 v x1) (gate)
        // (x0 v x2) (non-gate)
        // (~x0 v x3) (non-gate)
        let mut clauses = ClauseDB::new();
        clauses.add(&[lit(0, true), lit(1, false)], false);
        clauses.add(&[lit(0, false), lit(1, true)], false);
        clauses.add(&[lit(0, true), lit(2, true)], false);
        clauses.add(&[lit(0, false), lit(3, true)], false);

        bve.rebuild(&clauses);

        let result = bve.try_eliminate_with_gate(Variable(0), &clauses, Some(&[0, 1]));
        assert!(result.eliminated);

        // Restricted resolution should produce:
        // (x1 v x2) and (~x1 v x3), but not (x2 v x3).
        let mut has_x1_x2 = false;
        let mut has_not_x1_x3 = false;
        let mut has_x2_x3 = false;

        for r in &result.resolvents {
            if r.len() == 2 && r.contains(&lit(1, true)) && r.contains(&lit(2, true)) {
                has_x1_x2 = true;
            }
            if r.len() == 2 && r.contains(&lit(1, false)) && r.contains(&lit(3, true)) {
                has_not_x1_x3 = true;
            }
            if r.len() == 2 && r.contains(&lit(2, true)) && r.contains(&lit(3, true)) {
                has_x2_x3 = true;
            }
        }

        assert!(has_x1_x2);
        assert!(has_not_x1_x3);
        assert!(!has_x2_x3);
    }
}
