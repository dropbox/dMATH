//! Blocked Clause Elimination (BCE)
//!
//! A clause C is blocked on literal L if for every clause D containing ~L,
//! the resolvent of C and D on L is a tautology (contains both a literal and
//! its negation). If a clause is blocked, it can be safely removed without
//! changing satisfiability.
//!
//! BCE is a powerful preprocessing/inprocessing technique that can significantly
//! reduce formula size, especially on structured instances.
//!
//! Reference: Järvisalo, Biere, Heule, "Blocked Clause Elimination", TACAS 2010.

use crate::clause_db::ClauseDB;
use crate::literal::{Literal, Variable};

/// Maximum number of clauses to check per blocked literal candidate
const MAX_RESOLUTION_CHECKS: usize = 50;

/// Statistics for BCE operations
#[derive(Debug, Clone, Default)]
pub struct BCEStats {
    /// Number of blocked clauses eliminated
    pub clauses_eliminated: u64,
    /// Number of blocking checks performed
    pub checks_performed: u64,
    /// Number of clauses skipped (too many resolutions)
    pub skipped_expensive: u64,
    /// Number of BCE rounds
    pub rounds: u64,
}

/// Occurrence list for BCE - tracks which clauses contain each literal
#[derive(Debug, Clone)]
pub struct BCEOccList {
    /// For each literal index, list of clause indices containing that literal
    occ: Vec<Vec<usize>>,
}

impl BCEOccList {
    /// Create a new occurrence list for n variables
    #[must_use]
    pub fn new(num_vars: usize) -> Self {
        BCEOccList {
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

/// Blocked Clause Elimination engine
pub struct BCE {
    /// Occurrence lists
    occ: BCEOccList,
    /// Statistics
    stats: BCEStats,
    /// Temporary mark array for tautology checking (per variable: 0=unmarked, 1=pos, -1=neg)
    marks: Vec<i8>,
    /// Clauses that have been checked and found not blocked (avoid rechecking)
    checked: Vec<bool>,
}

impl BCE {
    /// Create a new BCE engine for n variables
    pub fn new(num_vars: usize) -> Self {
        BCE {
            occ: BCEOccList::new(num_vars),
            stats: BCEStats::default(),
            marks: vec![0; num_vars],
            checked: Vec::new(),
        }
    }

    /// Ensure internal buffers can handle `num_vars` variables.
    pub fn ensure_num_vars(&mut self, num_vars: usize) {
        self.occ.ensure_num_vars(num_vars);
        if self.marks.len() < num_vars {
            self.marks.resize(num_vars, 0);
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &BCEStats {
        &self.stats
    }

    /// Initialize/rebuild occurrence lists from clause database
    pub fn rebuild(&mut self, clauses: &ClauseDB) {
        self.occ.clear();
        self.checked = vec![false; clauses.len()];

        for idx in clauses.indices() {
            if clauses.header(idx).is_empty() {
                continue;
            }
            self.occ.add_clause(idx, clauses.literals(idx));
        }
    }

    /// Clear the marks array for literals in a clause
    fn clear_marks(&mut self, clause: &[Literal]) {
        for &lit in clause {
            self.marks[lit.variable().index()] = 0;
        }
    }

    /// Check if resolving two clauses on a variable produces a tautology
    ///
    /// Returns true if the resolvent would be a tautology.
    fn is_tautological_resolvent(
        &mut self,
        clause_c: &[Literal],
        clause_d: &[Literal],
        pivot_var: Variable,
    ) -> bool {
        // Clear marks from previous check
        self.clear_marks(clause_c);
        self.clear_marks(clause_d);

        // Mark literals from clause C (except the pivot)
        for &lit in clause_c {
            if lit.variable() == pivot_var {
                continue;
            }
            let var_idx = lit.variable().index();
            let sign: i8 = if lit.is_positive() { 1 } else { -1 };
            self.marks[var_idx] = sign;
        }

        // Check literals from clause D (except the pivot)
        for &lit in clause_d {
            if lit.variable() == pivot_var {
                continue;
            }
            let var_idx = lit.variable().index();
            let sign: i8 = if lit.is_positive() { 1 } else { -1 };

            // If opposite sign is already marked, we have a tautology
            if self.marks[var_idx] == -sign {
                // Clear marks before returning
                self.clear_marks(clause_c);
                self.clear_marks(clause_d);
                return true;
            }
        }

        // No tautology found
        self.clear_marks(clause_c);
        self.clear_marks(clause_d);
        false
    }

    /// Check if a clause is blocked on a given literal
    ///
    /// A clause C is blocked on literal L if for every clause D containing ~L,
    /// the resolvent of C and D on L is a tautology.
    fn is_blocked(&mut self, clause_idx: usize, blocking_lit: Literal, clauses: &ClauseDB) -> bool {
        let clause_c_header = clauses.header(clause_idx);
        if clause_c_header.is_empty() {
            return false;
        }
        let clause_c = clauses.literals(clause_idx);

        // Get clauses containing the negation of the blocking literal
        let neg_lit = blocking_lit.negated();
        let neg_occ = self.occ.get(neg_lit).to_vec(); // Clone to avoid borrow issues

        // If no clauses contain ~L, the clause is trivially blocked
        if neg_occ.is_empty() {
            return true;
        }

        // Too many resolution partners - skip for efficiency
        if neg_occ.len() > MAX_RESOLUTION_CHECKS {
            self.stats.skipped_expensive += 1;
            return false;
        }

        let pivot_var = blocking_lit.variable();

        // Check each clause D containing ~L
        for &d_idx in &neg_occ {
            if d_idx == clause_idx {
                continue; // Skip self
            }
            if d_idx >= clauses.len() || clauses.header(d_idx).is_empty() {
                continue;
            }

            self.stats.checks_performed += 1;

            // If any resolvent is not tautological, the clause is not blocked on L
            if !self.is_tautological_resolvent(clause_c, clauses.literals(d_idx), pivot_var) {
                return false;
            }
        }

        // All resolvents are tautological - clause is blocked
        true
    }

    /// Find a blocking literal for a clause
    ///
    /// Returns Some(literal) if the clause is blocked on that literal, None otherwise.
    fn find_blocking_literal(&mut self, clause_idx: usize, clauses: &ClauseDB) -> Option<Literal> {
        let header = clauses.header(clause_idx);
        if header.is_empty() || header.len() < 2 {
            return None;
        }
        let lits = clauses.literals(clause_idx);

        // Try each literal in the clause as a potential blocking literal
        // Prefer literals with fewer occurrences of their negation
        let mut candidates: Vec<(Literal, usize)> = lits
            .iter()
            .map(|&lit| (lit, self.occ.count(lit.negated())))
            .collect();

        // Sort by occurrence count (ascending) - fewer occurrences means fewer checks
        candidates.sort_by_key(|&(_, count)| count);

        candidates
            .into_iter()
            .map(|(lit, _)| lit)
            .find(|&lit| self.is_blocked(clause_idx, lit, clauses))
    }

    /// Run BCE as inprocessing
    ///
    /// Attempts to eliminate blocked clauses to simplify the formula.
    /// Should be called at decision level 0 (e.g., after a restart).
    ///
    /// Returns a list of clause indices that were eliminated.
    pub fn run_elimination(&mut self, clauses: &ClauseDB, max_eliminations: usize) -> Vec<usize> {
        self.stats.rounds += 1;
        let mut eliminated = Vec::new();

        // Check learned clauses first (more likely to be blocked and less important),
        // then check non-learned clauses.
        let mut candidates: Vec<usize> = Vec::new();

        // First pass: learned clauses
        for idx in clauses.indices() {
            let header = clauses.header(idx);
            if header.is_empty() || header.len() < 2 {
                continue;
            }
            if idx < self.checked.len() && self.checked[idx] {
                continue;
            }
            if header.is_learned() {
                candidates.push(idx);
            }
        }

        // Second pass: non-learned clauses
        for idx in clauses.indices() {
            let header = clauses.header(idx);
            if header.is_empty() || header.len() < 2 || header.is_learned() {
                continue;
            }
            if idx < self.checked.len() && self.checked[idx] {
                continue;
            }
            candidates.push(idx);
        }

        for clause_idx in candidates.iter().take(max_eliminations * 2) {
            let clause_idx = *clause_idx;

            // Skip if already eliminated in this round
            if eliminated.contains(&clause_idx) {
                continue;
            }

            // Skip empty or unit clauses
            let header = clauses.header(clause_idx);
            if clause_idx >= clauses.len() || header.is_empty() || header.len() < 2 {
                continue;
            }

            // Try to find a blocking literal
            if let Some(_blocking_lit) = self.find_blocking_literal(clause_idx, clauses) {
                eliminated.push(clause_idx);
                self.stats.clauses_eliminated += 1;

                // Update occurrence lists
                self.occ
                    .remove_clause(clause_idx, clauses.literals(clause_idx));

                if eliminated.len() >= max_eliminations {
                    break;
                }
            } else {
                // Mark as checked to avoid rechecking
                if clause_idx < self.checked.len() {
                    self.checked[clause_idx] = true;
                }
            }
        }

        eliminated
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
    fn test_bce_occurrence_list() {
        let mut occ = BCEOccList::new(5);

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
    fn test_bce_tautology_detection() {
        let mut bce = BCE::new(5);

        // C = {x0, x1}, D = {~x0, ~x1}
        // Resolvent on x0 = {x1, ~x1} - tautology
        let c = vec![lit(0, true), lit(1, true)];
        let d = vec![lit(0, false), lit(1, false)];

        assert!(bce.is_tautological_resolvent(&c, &d, Variable(0)));
    }

    #[test]
    fn test_bce_non_tautology() {
        let mut bce = BCE::new(5);

        // C = {x0, x1}, D = {~x0, x2}
        // Resolvent on x0 = {x1, x2} - NOT a tautology
        let c = vec![lit(0, true), lit(1, true)];
        let d = vec![lit(0, false), lit(2, true)];

        assert!(!bce.is_tautological_resolvent(&c, &d, Variable(0)));
    }

    #[test]
    fn test_bce_simple_blocked() {
        let mut bce = BCE::new(5);

        // C0 = {x0, x1}
        // C1 = {~x0, ~x1}  <- only clause with ~x0
        // Resolving C0 on x0 with C1 gives {x1, ~x1} - tautology
        // So C0 is blocked on x0
        let mut clauses = ClauseDB::new();
        clauses.add(&[lit(0, true), lit(1, true)], false);
        clauses.add(&[lit(0, false), lit(1, false)], false);

        bce.rebuild(&clauses);

        // C0 should be blocked on x0
        assert!(bce.is_blocked(0, lit(0, true), &clauses));
    }

    #[test]
    fn test_bce_not_blocked() {
        let mut bce = BCE::new(5);

        // C0 = {x0, x1}
        // C1 = {~x0, x2}  <- resolvent would be {x1, x2}, not a tautology
        // So C0 is NOT blocked on x0
        let mut clauses = ClauseDB::new();
        clauses.add(&[lit(0, true), lit(1, true)], false);
        clauses.add(&[lit(0, false), lit(2, true)], false);

        bce.rebuild(&clauses);

        // C0 should NOT be blocked on x0
        assert!(!bce.is_blocked(0, lit(0, true), &clauses));
    }

    #[test]
    fn test_bce_blocked_no_negation() {
        let mut bce = BCE::new(5);

        // C0 = {x0, x1}
        // No clause contains ~x0
        // So C0 is trivially blocked on x0
        let mut clauses = ClauseDB::new();
        clauses.add(&[lit(0, true), lit(1, true)], false);
        clauses.add(&[lit(2, true), lit(3, true)], false); // No ~x0

        bce.rebuild(&clauses);

        // C0 should be blocked on x0 (no resolution partners)
        assert!(bce.is_blocked(0, lit(0, true), &clauses));
    }

    #[test]
    fn test_bce_find_blocking_literal() {
        let mut bce = BCE::new(5);

        // C0 = {x0, x1}
        // C1 = {~x0, ~x1}
        // C0 is blocked on x0 (or x1)
        let mut clauses = ClauseDB::new();
        clauses.add(&[lit(0, true), lit(1, true)], false);
        clauses.add(&[lit(0, false), lit(1, false)], false);

        bce.rebuild(&clauses);

        // Should find a blocking literal
        let blocking = bce.find_blocking_literal(0, &clauses);
        assert!(blocking.is_some());
    }

    #[test]
    fn test_bce_run_elimination() {
        let mut bce = BCE::new(5);

        // C0 = {x0, x1} - blocked on x0 (resolvent with C1 is tautology)
        // C1 = {~x0, ~x1}
        // C2 = {x2, x3} - NOT blocked (no constraints)
        let mut clauses = ClauseDB::new();
        clauses.add(&[lit(0, true), lit(1, true)], false);
        clauses.add(&[lit(0, false), lit(1, false)], false);
        clauses.add(&[lit(2, true), lit(3, true)], false);

        bce.rebuild(&clauses);

        let eliminated = bce.run_elimination(&clauses, 10);

        // At least one clause should be blocked (either C0 or C1, or both)
        assert!(
            !eliminated.is_empty(),
            "Expected at least one blocked clause"
        );
    }

    #[test]
    fn test_bce_stats() {
        let mut bce = BCE::new(5);

        let mut clauses = ClauseDB::new();
        clauses.add(&[lit(0, true), lit(1, true)], false);
        clauses.add(&[lit(0, false), lit(1, false)], false);

        bce.rebuild(&clauses);
        let _ = bce.run_elimination(&clauses, 10);

        let stats = bce.stats();
        assert_eq!(stats.rounds, 1);
        // Should have eliminated some clauses
        assert!(
            stats.clauses_eliminated > 0 || stats.checks_performed > 0,
            "Expected some activity"
        );
    }

    #[test]
    fn test_bce_multiple_blocking_literals() {
        let mut bce = BCE::new(5);

        // C0 = {x0, x1, x2}
        // C1 = {~x0, ~x1}  <- blocking partner for x0 gives tautology
        // C2 = {~x1, ~x2}  <- blocking partner for x1 gives tautology
        //
        // Resolving C0 on x0 with C1: {x1, x2, ~x1} - tautology
        // Resolving C0 on x1 with C1: {x0, x2, ~x0} - tautology
        // Resolving C0 on x1 with C2: {x0, x2, ~x2} - tautology
        //
        // C0 should be blocked on either x0 or x1
        let mut clauses = ClauseDB::new();
        clauses.add(&[lit(0, true), lit(1, true), lit(2, true)], false);
        clauses.add(&[lit(0, false), lit(1, false)], false);
        clauses.add(&[lit(1, false), lit(2, false)], false);

        bce.rebuild(&clauses);

        // Check that C0 is blocked on x0 (resolvent with C1 is {x1, x2, ~x1} - tautology)
        let blocked_on_x0 = bce.is_blocked(0, lit(0, true), &clauses);

        // C0 should be blocked on at least one literal
        let blocking = bce.find_blocking_literal(0, &clauses);
        assert!(
            blocked_on_x0 || blocking.is_some(),
            "C0 should be blocked on some literal"
        );
    }
}
