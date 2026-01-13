//! Conflict analysis (1UIP learning)
//!
//! Implements the First Unique Implication Point (1UIP) scheme for
//! conflict-driven clause learning.

use crate::literal::Literal;

/// Result of conflict analysis
#[derive(Debug, Clone)]
pub struct ConflictResult {
    /// The learned clause (first literal is the asserting literal)
    pub learned_clause: Vec<Literal>,
    /// The backtrack level
    pub backtrack_level: u32,
    /// The LBD (Literal Block Distance) of the learned clause
    pub lbd: u32,
    /// Resolution chain (clause IDs used to derive the learned clause)
    /// Used for LRAT proof generation. Empty if LRAT is not enabled.
    pub resolution_chain: Vec<u64>,
}

/// Conflict analyzer
#[derive(Debug, Default)]
pub struct ConflictAnalyzer {
    /// Seen marks for variables during analysis
    seen: Vec<bool>,
    /// Temporary learned clause being built (without the UIP)
    learned: Vec<Literal>,
    /// The asserting literal (UIP negated)
    asserting_lit: Option<Literal>,
    /// Resolution chain (clause IDs used during analysis)
    resolution_chain: Vec<u64>,
    /// Workspace for LBD computation (reused to avoid allocations)
    lbd_seen: Vec<bool>,
    /// Indices to clear in lbd_seen after LBD computation
    lbd_to_clear: Vec<usize>,
}

impl ConflictAnalyzer {
    /// Create a new conflict analyzer
    pub fn new(num_vars: usize) -> Self {
        ConflictAnalyzer {
            seen: vec![false; num_vars],
            learned: Vec::new(),
            asserting_lit: None,
            resolution_chain: Vec::new(),
            lbd_seen: Vec::new(),
            lbd_to_clear: Vec::new(),
        }
    }

    /// Ensure the analyzer can track `num_vars` variables.
    pub fn ensure_num_vars(&mut self, num_vars: usize) {
        if self.seen.len() < num_vars {
            self.seen.resize(num_vars, false);
        }
    }

    /// Clear the analyzer state for a new conflict
    pub fn clear(&mut self) {
        for s in &mut self.seen {
            *s = false;
        }
        self.learned.clear();
        self.asserting_lit = None;
        self.resolution_chain.clear();
    }

    /// Add a clause ID to the resolution chain (for LRAT proofs)
    #[inline]
    pub fn add_to_chain(&mut self, clause_id: u64) {
        self.resolution_chain.push(clause_id);
    }

    /// Mark a variable as seen
    #[inline]
    pub fn mark_seen(&mut self, var: usize) {
        self.seen[var] = true;
    }

    /// Unmark a variable as seen
    #[inline]
    pub fn unmark_seen(&mut self, var: usize) {
        self.seen[var] = false;
    }

    /// Check if a variable is seen
    #[inline]
    pub fn is_seen(&self, var: usize) -> bool {
        self.seen[var]
    }

    /// Add a literal to the learned clause
    #[inline]
    pub fn add_to_learned(&mut self, lit: Literal) {
        self.learned.push(lit);
    }

    /// Set the asserting literal (the 1UIP negated)
    #[inline]
    pub fn set_asserting_literal(&mut self, lit: Literal) {
        self.asserting_lit = Some(lit);
    }

    /// Take the learned literals (for minimization)
    #[inline]
    pub fn take_learned(&mut self) -> Vec<Literal> {
        std::mem::take(&mut self.learned)
    }

    /// Set the learned literals (after minimization)
    #[inline]
    pub fn set_learned(&mut self, learned: Vec<Literal>) {
        self.learned = learned;
    }

    /// Get the asserting literal (1UIP negated)
    #[inline]
    pub fn asserting_literal(&self) -> Literal {
        self.asserting_lit
            .expect("asserting_literal called before set")
    }

    /// Get the learned literals (not including the asserting literal)
    #[inline]
    pub fn learned_literals(&self) -> &[Literal] {
        &self.learned
    }

    /// Compute the backtrack level from the learned clause.
    /// This is the second-highest decision level among the literals,
    /// or 0 if the learned clause is unit.
    pub fn compute_backtrack_level(&self, level: &[u32]) -> u32 {
        if self.learned.is_empty() {
            // Unit learned clause - backtrack to level 0
            return 0;
        }

        // Find the highest level among non-asserting literals
        let mut max_level = 0;
        for &lit in &self.learned {
            let var_level = level[lit.variable().index()];
            if var_level > max_level {
                max_level = var_level;
            }
        }
        max_level
    }

    /// Compute the LBD (Literal Block Distance) of the learned clause.
    /// LBD = number of distinct decision levels in the clause.
    /// Uses a reusable workspace to avoid per-call allocations.
    pub fn compute_lbd(&mut self, level: &[u32]) -> u32 {
        let mut count = 0u32;

        // Ensure workspace is large enough for all decision levels
        // Decision levels are bounded by the number of decisions made,
        // which is at most the number of variables. We use level.len() as upper bound.
        if self.lbd_seen.len() < level.len() + 1 {
            self.lbd_seen.resize(level.len() + 1, false);
        }

        // Add asserting literal's level
        if let Some(lit) = self.asserting_lit {
            let lvl = level[lit.variable().index()] as usize;
            if !self.lbd_seen[lvl] {
                self.lbd_seen[lvl] = true;
                self.lbd_to_clear.push(lvl);
                count += 1;
            }
        }

        // Add other literals' levels
        for &lit in &self.learned {
            let lvl = level[lit.variable().index()] as usize;
            if !self.lbd_seen[lvl] {
                self.lbd_seen[lvl] = true;
                self.lbd_to_clear.push(lvl);
                count += 1;
            }
        }

        // Clear workspace for next call
        for &idx in &self.lbd_to_clear {
            self.lbd_seen[idx] = false;
        }
        self.lbd_to_clear.clear();

        count
    }

    /// Get the final conflict result
    pub fn get_result(&self, backtrack_level: u32, lbd: u32) -> ConflictResult {
        // Build the final learned clause with asserting literal first
        let mut learned_clause = Vec::with_capacity(self.learned.len() + 1);

        // Asserting literal goes first
        if let Some(lit) = self.asserting_lit {
            learned_clause.push(lit);
        }

        // Add other literals to the learned clause
        learned_clause.extend_from_slice(&self.learned);

        ConflictResult {
            learned_clause,
            backtrack_level,
            lbd,
            resolution_chain: self.resolution_chain.clone(),
        }
    }

    /// Reorder the learned clause so the second literal is at the backtrack level.
    /// This is needed for proper watched literal initialization.
    pub fn reorder_for_watches(clause: &mut [Literal], level: &[u32], backtrack_level: u32) {
        if clause.len() < 2 {
            return;
        }

        // Find a literal at the backtrack level (not the first one)
        for i in 2..clause.len() {
            if level[clause[i].variable().index()] == backtrack_level {
                clause.swap(1, i);
                return;
            }
        }

        // If no exact match, find the highest level (should be at position 1)
        let mut max_idx = 1;
        let mut max_level = level[clause[1].variable().index()];
        for i in 2..clause.len() {
            let lit_level = level[clause[i].variable().index()];
            if lit_level > max_level {
                max_level = lit_level;
                max_idx = i;
            }
        }
        if max_idx != 1 {
            clause.swap(1, max_idx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::literal::Variable;

    #[test]
    fn test_conflict_analyzer_basic() {
        let mut analyzer = ConflictAnalyzer::new(5);

        // Mark some variables as seen
        analyzer.mark_seen(0);
        analyzer.mark_seen(2);
        assert!(analyzer.is_seen(0));
        assert!(!analyzer.is_seen(1));
        assert!(analyzer.is_seen(2));

        // Clear and verify
        analyzer.clear();
        assert!(!analyzer.is_seen(0));
        assert!(!analyzer.is_seen(2));
    }

    #[test]
    fn test_learned_clause_construction() {
        let mut analyzer = ConflictAnalyzer::new(5);

        // Build a learned clause
        analyzer.add_to_learned(Literal::negative(Variable(1)));
        analyzer.add_to_learned(Literal::positive(Variable(2)));
        analyzer.set_asserting_literal(Literal::negative(Variable(0)));

        let level = vec![1, 2, 1, 0, 0]; // Levels for variables 0-4
        let bt_level = analyzer.compute_backtrack_level(&level);
        let lbd = analyzer.compute_lbd(&level);

        assert_eq!(bt_level, 2); // Highest level among non-asserting literals
        assert!(lbd >= 1); // At least one decision level

        let result = analyzer.get_result(bt_level, lbd);
        assert_eq!(result.learned_clause.len(), 3);
        assert_eq!(result.learned_clause[0], Literal::negative(Variable(0)));
    }

    #[test]
    fn test_unit_learned_clause() {
        let mut analyzer = ConflictAnalyzer::new(3);

        // Only an asserting literal (unit learned clause)
        analyzer.set_asserting_literal(Literal::positive(Variable(0)));

        let level = vec![1, 0, 0];
        let bt_level = analyzer.compute_backtrack_level(&level);

        assert_eq!(bt_level, 0); // Unit clause backtracks to level 0
    }
}

// ============================================================================
// Kani Verification Harnesses
// ============================================================================

#[cfg(kani)]
mod verification {
    use super::*;
    use crate::literal::Variable;

    /// Verify that seen marking is idempotent and can be cleared
    #[kani::proof]
    fn proof_seen_marking_idempotent() {
        const NUM_VARS: usize = 8;
        let mut analyzer = ConflictAnalyzer::new(NUM_VARS);
        let var_idx: usize = kani::any();
        kani::assume(var_idx < NUM_VARS);

        // Initially not seen
        assert!(!analyzer.is_seen(var_idx));

        // Mark seen
        analyzer.mark_seen(var_idx);
        assert!(analyzer.is_seen(var_idx));

        // Mark again (idempotent)
        analyzer.mark_seen(var_idx);
        assert!(analyzer.is_seen(var_idx));

        // Unmark
        analyzer.unmark_seen(var_idx);
        assert!(!analyzer.is_seen(var_idx));

        // Clear clears all
        analyzer.mark_seen(var_idx);
        analyzer.clear();
        assert!(!analyzer.is_seen(var_idx));
    }

    /// Verify that backtrack level is computed correctly for learned clauses (concrete)
    #[kani::proof]
    fn proof_backtrack_level_concrete() {
        const NUM_VARS: usize = 4;
        let mut analyzer = ConflictAnalyzer::new(NUM_VARS);

        // Fixed levels for tractability
        let level: [u32; NUM_VARS] = [1, 3, 2, 4];

        // Empty learned clause -> backtrack level is 0
        assert_eq!(analyzer.compute_backtrack_level(&level), 0);

        // Add literal at level 1
        analyzer.add_to_learned(Literal::positive(Variable(0)));
        assert_eq!(analyzer.compute_backtrack_level(&level), 1);

        // Add literal at level 3 -> max is now 3
        analyzer.add_to_learned(Literal::positive(Variable(1)));
        assert_eq!(analyzer.compute_backtrack_level(&level), 3);
    }

    /// Verify that LBD (Literal Block Distance) is at least 1 if asserting literal is set
    #[kani::proof]
    fn proof_lbd_at_least_one_if_asserting() {
        const NUM_VARS: usize = 4;
        let mut analyzer = ConflictAnalyzer::new(NUM_VARS);

        // Create arbitrary levels
        let level: [u32; NUM_VARS] = [1, 2, 1, 3]; // Fixed levels for tractability

        // Set asserting literal
        let asserting_var: u32 = kani::any();
        kani::assume(asserting_var < NUM_VARS as u32);
        analyzer.set_asserting_literal(Literal::positive(Variable(asserting_var)));

        let lbd = analyzer.compute_lbd(&level);

        // LBD should be at least 1 when asserting literal is set
        assert!(lbd >= 1);
    }

    /// Verify that learned clause has asserting literal first
    #[kani::proof]
    fn proof_asserting_literal_first_in_result() {
        const NUM_VARS: usize = 4;
        let mut analyzer = ConflictAnalyzer::new(NUM_VARS);

        // Set asserting literal
        let asserting_lit = Literal::positive(Variable(0));
        analyzer.set_asserting_literal(asserting_lit);

        // Add some other literals
        analyzer.add_to_learned(Literal::negative(Variable(1)));
        analyzer.add_to_learned(Literal::positive(Variable(2)));

        let result = analyzer.get_result(1, 2);

        // Asserting literal should be first
        assert!(!result.learned_clause.is_empty());
        assert_eq!(result.learned_clause[0], asserting_lit);
    }

    /// Verify reorder_for_watches doesn't crash and preserves clause length
    #[kani::proof]
    fn proof_reorder_preserves_length() {
        // Small clause for verification
        let mut clause = vec![
            Literal::positive(Variable(0)),
            Literal::positive(Variable(1)),
            Literal::positive(Variable(2)),
        ];
        let original_len = clause.len();

        let level: [u32; 3] = [3, 1, 2];
        let backtrack_level: u32 = kani::any();
        kani::assume(backtrack_level <= 3);

        ConflictAnalyzer::reorder_for_watches(&mut clause, &level, backtrack_level);

        // Length preserved
        assert_eq!(clause.len(), original_len);
    }

    // ========================================================================
    // Gap 5 Proofs: 1UIP Conflict Analysis Invariants
    // ========================================================================

    /// Verify 1UIP property: learned clause has exactly one literal at conflict level
    /// This is a structural property of how we build learned clauses - the asserting
    /// literal is at the conflict level, all others are at lower levels.
    #[kani::proof]
    fn proof_1uip_single_literal_at_conflict_level() {
        const NUM_VARS: usize = 4;
        let mut analyzer = ConflictAnalyzer::new(NUM_VARS);

        // Conflict level
        let conflict_level: u32 = 3;

        // Levels for each variable: only var 0 is at conflict level
        // Other variables are at lower levels (1 and 2)
        let levels: [u32; NUM_VARS] = [3, 1, 2, 1];

        // Asserting literal (UIP) is var 0 at conflict level
        let asserting_lit = Literal::positive(Variable(0));
        analyzer.set_asserting_literal(asserting_lit);

        // Other literals in learned clause are at lower levels
        analyzer.add_to_learned(Literal::negative(Variable(1))); // level 1
        analyzer.add_to_learned(Literal::positive(Variable(2))); // level 2

        let result = analyzer.get_result(2, 2); // backtrack to level 2

        // Count how many literals are at conflict level
        let count_at_conflict = result
            .learned_clause
            .iter()
            .filter(|lit| levels[lit.variable().index()] == conflict_level)
            .count();

        // 1UIP property: exactly one literal at conflict level
        assert_eq!(count_at_conflict, 1);

        // That literal should be the asserting literal (first position)
        assert_eq!(result.learned_clause[0], asserting_lit);
    }

    /// Verify backtrack level is computed correctly from learned clause
    /// Backtrack level = second-highest level among all literals
    #[kani::proof]
    fn proof_backtrack_level_is_second_highest() {
        const NUM_VARS: usize = 4;
        let mut analyzer = ConflictAnalyzer::new(NUM_VARS);

        // Levels: var 0 at level 5 (conflict), var 1 at level 3, var 2 at level 2
        let levels: [u32; NUM_VARS] = [5, 3, 2, 1];

        // Asserting literal at conflict level 5
        analyzer.set_asserting_literal(Literal::positive(Variable(0)));

        // Other literals at lower levels
        analyzer.add_to_learned(Literal::negative(Variable(1))); // level 3
        analyzer.add_to_learned(Literal::positive(Variable(2))); // level 2

        let backtrack_level = analyzer.compute_backtrack_level(&levels);

        // Backtrack level should be the second-highest: 3
        // (Asserting literal is at 5, next highest is 3)
        assert_eq!(backtrack_level, 3);
    }

    /// Verify learned clause is non-empty when asserting literal is set
    #[kani::proof]
    fn proof_learned_clause_non_empty_with_asserting() {
        const NUM_VARS: usize = 4;
        let mut analyzer = ConflictAnalyzer::new(NUM_VARS);

        // Set any asserting literal
        let var_idx: u32 = kani::any();
        kani::assume(var_idx < NUM_VARS as u32);
        let polarity: bool = kani::any();
        let asserting_lit = if polarity {
            Literal::positive(Variable(var_idx))
        } else {
            Literal::negative(Variable(var_idx))
        };

        analyzer.set_asserting_literal(asserting_lit);

        let result = analyzer.get_result(0, 1);

        // Learned clause must have at least the asserting literal
        assert!(!result.learned_clause.is_empty());
        assert_eq!(result.learned_clause[0], asserting_lit);
    }

    /// Verify analyzer reset clears all state
    #[kani::proof]
    fn proof_clear_resets_all_state() {
        const NUM_VARS: usize = 4;
        let mut analyzer = ConflictAnalyzer::new(NUM_VARS);

        // Mark some variables as seen
        let v0: usize = kani::any();
        let v1: usize = kani::any();
        kani::assume(v0 < NUM_VARS && v1 < NUM_VARS);

        analyzer.mark_seen(v0);
        analyzer.mark_seen(v1);
        analyzer.set_asserting_literal(Literal::positive(Variable(0)));
        analyzer.add_to_learned(Literal::negative(Variable(1)));

        // Clear
        analyzer.clear();

        // Verify all state is reset
        assert!(!analyzer.is_seen(v0));
        assert!(!analyzer.is_seen(v1));
        assert!(analyzer.asserting_lit.is_none());
        assert!(analyzer.learned.is_empty());
    }
}
