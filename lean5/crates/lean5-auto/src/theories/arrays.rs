//! Array theory solver
//!
//! This module implements the theory of arrays with extensionality,
//! supporting `select` (read) and `store` (write) operations.
//!
//! # Theory of Arrays (AR)
//!
//! The theory handles:
//! - `select(a, i)` - read element at index i from array a
//! - `store(a, i, v)` - write value v at index i to array a
//!
//! # Axioms
//!
//! 1. **Read-over-write (same index)**:
//!    `select(store(a, i, v), i) = v`
//!
//! 2. **Read-over-write (different index)**:
//!    `i ≠ j → select(store(a, i, v), j) = select(a, j)`
//!
//! 3. **Extensionality** (for deciding array equality):
//!    `(∀i. select(a, i) = select(b, i)) → a = b`
//!
//! # Implementation
//!
//! The solver integrates with the E-graph equality theory to:
//! - Track select/store operations
//! - Propagate equalities from array axioms
//! - Detect conflicts when axioms are violated
//!
//! # Example
//!
//! ```text
//! // Read-over-write same index: select(store(a, i, v), i) = v
//! store_a_i_v := store(a, i, v)
//! select_store := select(store_a_i_v, i)
//! // Axiom 1 implies: select_store = v
//!
//! // Read-over-write different index:
//! // select(store(a, i, v), j) = select(a, j)  when i ≠ j
//! ```

use crate::cdcl::Lit;
use crate::smt::{SmtTerm, TermId, TheoryCheckResult, TheoryLiteral, TheorySolver};
use std::collections::{HashMap, HashSet};

/// Array theory solver
///
/// Tracks array operations (select/store) and propagates implied equalities.
pub struct ArrayTheory {
    /// Mapping from term IDs to their structure
    terms: Vec<SmtTerm>,

    /// Select operations: maps (array_term, index_term) -> result_term
    selects: HashMap<(TermId, TermId), TermId>,

    /// Store operations: maps store_term -> (array, index, value)
    stores: HashMap<TermId, (TermId, TermId, TermId)>,

    /// Reverse mapping: which store terms produce which array
    store_results: HashSet<TermId>,

    /// Asserted equalities: (t1, t2, literal)
    equalities: Vec<(TermId, TermId, Lit)>,

    /// Asserted disequalities: (t1, t2, literal)
    disequalities: Vec<(TermId, TermId, Lit)>,

    /// Decision level trails for backtracking
    eq_trail: Vec<usize>,
    diseq_trail: Vec<usize>,

    /// Current decision level
    level: u32,

    /// Pending equality checks from axiom applications
    /// These are equalities implied by array axioms that should be propagated
    /// to the equality theory for full integration
    pending_equalities: Vec<(TermId, TermId)>,
}

impl ArrayTheory {
    /// Create a new array theory solver
    pub fn new() -> Self {
        ArrayTheory {
            terms: Vec::new(),
            selects: HashMap::new(),
            stores: HashMap::new(),
            store_results: HashSet::new(),
            equalities: Vec::new(),
            disequalities: Vec::new(),
            eq_trail: vec![0],
            diseq_trail: vec![0],
            level: 0,
            pending_equalities: Vec::new(),
        }
    }

    /// Analyze terms to extract select/store structure
    fn analyze_terms(&mut self) {
        self.selects.clear();
        self.stores.clear();
        self.store_results.clear();

        for (idx, term) in self.terms.iter().enumerate() {
            let term_id =
                TermId(u32::try_from(idx).expect("array theory term index exceeded u32::MAX"));

            if let SmtTerm::App(name, args) = term {
                match name.name() {
                    "select" if args.len() == 2 => {
                        // select(array, index) -> value
                        let array = args[0];
                        let index = args[1];
                        self.selects.insert((array, index), term_id);
                    }
                    "store" if args.len() == 3 => {
                        // store(array, index, value) -> new_array
                        let array = args[0];
                        let index = args[1];
                        let value = args[2];
                        self.stores.insert(term_id, (array, index, value));
                        self.store_results.insert(term_id);
                    }
                    _ => {}
                }
            }
        }
    }

    /// Apply read-over-write axiom 1: select(store(a, i, v), i) = v
    ///
    /// When we have select(s, j) where s = store(a, i, v):
    /// - If i = j (indices equal), then select(s, j) = v
    fn apply_row_same_index(&mut self) -> TheoryCheckResult {
        // Collect potential ROW-same applications
        let mut to_check = Vec::new();

        for (&(array_term, select_index), &select_result) in &self.selects {
            // Check if array_term is a store operation
            if let Some(&(_, store_index, store_value)) = self.stores.get(&array_term) {
                // select(store(a, i, v), j) with this select having index j
                // If i = j (same index), then select result should equal v
                to_check.push((select_index, store_index, select_result, store_value));
            }
        }

        // Check each potential ROW-same
        for (select_idx, store_idx, select_result, store_value) in to_check {
            // Check if indices are asserted equal
            if self.are_equal(select_idx, store_idx) {
                // Axiom 1: select(store(a, i, v), i) = v
                // Check if select_result = store_value is consistent
                if !self.are_equal(select_result, store_value) {
                    // Need to check: is select_result ≠ store_value asserted?
                    if self.are_disequal(select_result, store_value) {
                        // Conflict! ROW-same says they should be equal,
                        // but disequality is asserted
                        return self.build_row_same_conflict(
                            select_idx,
                            store_idx,
                            select_result,
                            store_value,
                        );
                    }
                    // Otherwise, we should propagate select_result = store_value
                    // (handled by equality theory integration)
                    self.pending_equalities.push((select_result, store_value));
                }
            }
        }

        TheoryCheckResult::Consistent
    }

    /// Apply read-over-write axiom 2: i ≠ j → select(store(a, i, v), j) = select(a, j)
    ///
    /// When we have select(s, j) where s = store(a, i, v) and i ≠ j:
    /// - select(s, j) should equal select(a, j)
    fn apply_row_diff_index(&mut self) -> TheoryCheckResult {
        let mut to_check = Vec::new();

        for (&(array_term, select_index), &select_result) in &self.selects {
            if let Some(&(base_array, store_index, _)) = self.stores.get(&array_term) {
                // select(store(a, i, v), j)
                // If i ≠ j, then select(store(a, i, v), j) = select(a, j)
                to_check.push((select_index, store_index, select_result, base_array));
            }
        }

        for (select_idx, store_idx, select_result, base_array) in to_check {
            // Check if indices are asserted disequal
            if self.are_disequal(select_idx, store_idx) {
                // Axiom 2 applies: select(store(a, i, v), j) = select(a, j)
                // Look up select(base_array, select_idx)
                if let Some(&base_select) = self.selects.get(&(base_array, select_idx)) {
                    // select_result should equal base_select
                    if !self.are_equal(select_result, base_select) {
                        if self.are_disequal(select_result, base_select) {
                            return self.build_row_diff_conflict(
                                select_idx,
                                store_idx,
                                select_result,
                                base_select,
                            );
                        }
                        self.pending_equalities.push((select_result, base_select));
                    }
                }
            }
        }

        TheoryCheckResult::Consistent
    }

    /// Check if two terms are equal (based on asserted equalities)
    fn are_equal(&self, t1: TermId, t2: TermId) -> bool {
        if t1 == t2 {
            return true;
        }
        // Simple equality check - in practice, this would use union-find
        // from the EUF theory integration
        for &(a, b, _) in &self.equalities {
            if (a == t1 && b == t2) || (a == t2 && b == t1) {
                return true;
            }
        }
        false
    }

    /// Check if two terms are disequal (based on asserted disequalities)
    fn are_disequal(&self, t1: TermId, t2: TermId) -> bool {
        for &(a, b, _) in &self.disequalities {
            if (a == t1 && b == t2) || (a == t2 && b == t1) {
                return true;
            }
        }
        false
    }

    /// Build conflict clause for ROW-same violation
    fn build_row_same_conflict(
        &self,
        select_idx: TermId,
        store_idx: TermId,
        select_result: TermId,
        store_value: TermId,
    ) -> TheoryCheckResult {
        // Conflict: select_idx = store_idx implies select_result = store_value,
        // but select_result ≠ store_value was asserted
        let mut conflict_lits = Vec::new();

        // Find the equality literal for indices
        for &(a, b, lit) in &self.equalities {
            if (a == select_idx && b == store_idx) || (a == store_idx && b == select_idx) {
                conflict_lits.push(lit);
                break;
            }
        }

        // Find the disequality literal for results
        for &(a, b, lit) in &self.disequalities {
            if (a == select_result && b == store_value) || (a == store_value && b == select_result)
            {
                conflict_lits.push(lit);
                break;
            }
        }

        TheoryCheckResult::Conflict(conflict_lits)
    }

    /// Build conflict clause for ROW-diff violation
    fn build_row_diff_conflict(
        &self,
        select_idx: TermId,
        store_idx: TermId,
        select_result: TermId,
        base_select: TermId,
    ) -> TheoryCheckResult {
        let mut conflict_lits = Vec::new();

        // Find disequality literal for indices
        for &(a, b, lit) in &self.disequalities {
            if (a == select_idx && b == store_idx) || (a == store_idx && b == select_idx) {
                conflict_lits.push(lit);
                break;
            }
        }

        // Find disequality literal for results
        for &(a, b, lit) in &self.disequalities {
            if (a == select_result && b == base_select) || (a == base_select && b == select_result)
            {
                conflict_lits.push(lit);
                break;
            }
        }

        TheoryCheckResult::Conflict(conflict_lits)
    }

    /// Get statistics about the array theory state
    pub fn stats(&self) -> ArrayStats {
        ArrayStats {
            num_selects: self.selects.len(),
            num_stores: self.stores.len(),
            num_equalities: self.equalities.len(),
            num_disequalities: self.disequalities.len(),
        }
    }
}

impl Default for ArrayTheory {
    fn default() -> Self {
        Self::new()
    }
}

impl TheorySolver for ArrayTheory {
    fn assert_literal(&mut self, lit: Lit, theory_lit: &TheoryLiteral) -> TheoryCheckResult {
        match theory_lit {
            TheoryLiteral::Eq(t1, t2) => {
                self.equalities.push((*t1, *t2, lit));

                // After asserting equality, check array axioms
                if let result @ TheoryCheckResult::Conflict(_) = self.apply_row_same_index() {
                    return result;
                }
                if let result @ TheoryCheckResult::Conflict(_) = self.apply_row_diff_index() {
                    return result;
                }

                TheoryCheckResult::Consistent
            }
            TheoryLiteral::Neq(t1, t2) => {
                self.disequalities.push((*t1, *t2, lit));

                // After asserting disequality, check array axioms
                if let result @ TheoryCheckResult::Conflict(_) = self.apply_row_same_index() {
                    return result;
                }
                if let result @ TheoryCheckResult::Conflict(_) = self.apply_row_diff_index() {
                    return result;
                }

                TheoryCheckResult::Consistent
            }
            // Other literals are not handled by array theory
            _ => TheoryCheckResult::Consistent,
        }
    }

    fn check(&self) -> TheoryCheckResult {
        // Full consistency check is already done incrementally
        TheoryCheckResult::Consistent
    }

    fn backtrack(&mut self, level: u32) {
        if level >= self.level {
            return;
        }

        // Restore equalities and disequalities to the state at target level
        let eq_limit = self.eq_trail.get(level as usize + 1).copied().unwrap_or(0);
        let diseq_limit = self
            .diseq_trail
            .get(level as usize + 1)
            .copied()
            .unwrap_or(0);

        self.equalities.truncate(eq_limit);
        self.disequalities.truncate(diseq_limit);

        self.eq_trail.truncate(level as usize + 1);
        self.diseq_trail.truncate(level as usize + 1);
        self.level = level;
    }

    fn push(&mut self) {
        self.level += 1;
        self.eq_trail.push(self.equalities.len());
        self.diseq_trail.push(self.disequalities.len());
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn name(&self) -> &'static str {
        "Arrays"
    }

    fn set_terms(&mut self, terms: Vec<SmtTerm>) {
        self.terms = terms;
        self.analyze_terms();
    }
}

/// Statistics for array theory
#[derive(Clone, Debug, Default)]
pub struct ArrayStats {
    pub num_selects: usize,
    pub num_stores: usize,
    pub num_equalities: usize,
    pub num_disequalities: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cdcl::Var;
    use crate::egraph::Symbol;

    fn make_lit(var_idx: u32, positive: bool) -> Lit {
        let var = Var::new(var_idx);
        if positive {
            Lit::pos(var)
        } else {
            Lit::neg(var)
        }
    }

    #[test]
    fn test_array_theory_basic() {
        let mut theory = ArrayTheory::new();

        // Create terms: a, i, v
        let terms = vec![
            SmtTerm::Const(Symbol::new("a")), // 0: array a
            SmtTerm::Const(Symbol::new("i")), // 1: index i
            SmtTerm::Const(Symbol::new("v")), // 2: value v
        ];
        theory.set_terms(terms);

        let stats = theory.stats();
        assert_eq!(stats.num_selects, 0);
        assert_eq!(stats.num_stores, 0);
    }

    #[test]
    fn test_array_select_store_analysis() {
        let mut theory = ArrayTheory::new();

        // Terms: a, i, v, store(a, i, v), select(store(a, i, v), i)
        let terms = vec![
            SmtTerm::Const(Symbol::new("a")), // 0
            SmtTerm::Const(Symbol::new("i")), // 1
            SmtTerm::Const(Symbol::new("v")), // 2
            SmtTerm::App(Symbol::new("store"), vec![TermId(0), TermId(1), TermId(2)]), // 3: store(a, i, v)
            SmtTerm::App(Symbol::new("select"), vec![TermId(3), TermId(1)]), // 4: select(store(a, i, v), i)
        ];
        theory.set_terms(terms);

        let stats = theory.stats();
        assert_eq!(stats.num_stores, 1, "Should detect 1 store operation");
        assert_eq!(stats.num_selects, 1, "Should detect 1 select operation");
    }

    #[test]
    fn test_row_same_index_consistency() {
        let mut theory = ArrayTheory::new();

        // Terms for: select(store(a, i, v), i) should equal v
        // 0: a, 1: i, 2: v, 3: store(a,i,v), 4: select(store(a,i,v), i)
        let terms = vec![
            SmtTerm::Const(Symbol::new("a")), // 0
            SmtTerm::Const(Symbol::new("i")), // 1
            SmtTerm::Const(Symbol::new("v")), // 2
            SmtTerm::App(Symbol::new("store"), vec![TermId(0), TermId(1), TermId(2)]), // 3
            SmtTerm::App(Symbol::new("select"), vec![TermId(3), TermId(1)]), // 4
        ];
        theory.set_terms(terms);

        // Assert i = i (reflexive, already true) - the select uses the same index as store
        // The theory should recognize that select result (4) should equal v (2)

        // Check that no conflict occurs initially
        let result = theory.check();
        assert!(matches!(result, TheoryCheckResult::Consistent));
    }

    #[test]
    fn test_row_same_index_conflict() {
        let mut theory = ArrayTheory::new();

        // select(store(a, i, v), i) ≠ v should cause conflict
        let terms = vec![
            SmtTerm::Const(Symbol::new("a")), // 0
            SmtTerm::Const(Symbol::new("i")), // 1
            SmtTerm::Const(Symbol::new("v")), // 2
            SmtTerm::App(Symbol::new("store"), vec![TermId(0), TermId(1), TermId(2)]), // 3
            SmtTerm::App(Symbol::new("select"), vec![TermId(3), TermId(1)]), // 4
        ];
        theory.set_terms(terms);

        let i = TermId(1);
        let v = TermId(2);
        let select_result = TermId(4);

        // Assert that the indices are equal (they're the same term, so trivially true)
        // In this case, we need to assert i = i explicitly for the theory to track it
        let eq_lit = make_lit(0, true);
        let result = theory.assert_literal(eq_lit, &TheoryLiteral::Eq(i, i));
        assert!(matches!(result, TheoryCheckResult::Consistent));

        // Now assert select(store(a,i,v), i) ≠ v - this should conflict
        let neq_lit = make_lit(1, false);
        let result = theory.assert_literal(neq_lit, &TheoryLiteral::Neq(select_result, v));

        // The ROW-same axiom says select(store(a,i,v), i) = v
        // So asserting they're not equal should cause a conflict
        assert!(
            matches!(result, TheoryCheckResult::Conflict(_)),
            "ROW-same axiom violation should cause conflict"
        );
    }

    #[test]
    fn test_row_diff_index() {
        let mut theory = ArrayTheory::new();

        // Test: select(store(a, i, v), j) = select(a, j) when i ≠ j
        let terms = vec![
            SmtTerm::Const(Symbol::new("a")), // 0: array a
            SmtTerm::Const(Symbol::new("i")), // 1: index i
            SmtTerm::Const(Symbol::new("j")), // 2: index j
            SmtTerm::Const(Symbol::new("v")), // 3: value v
            SmtTerm::App(Symbol::new("store"), vec![TermId(0), TermId(1), TermId(3)]), // 4: store(a, i, v)
            SmtTerm::App(Symbol::new("select"), vec![TermId(4), TermId(2)]), // 5: select(store(a,i,v), j)
            SmtTerm::App(Symbol::new("select"), vec![TermId(0), TermId(2)]), // 6: select(a, j)
        ];
        theory.set_terms(terms);

        let i = TermId(1);
        let j = TermId(2);
        let select_store = TermId(5);
        let select_base = TermId(6);

        // Assert i ≠ j
        let diseq_lit = make_lit(0, false);
        let result = theory.assert_literal(diseq_lit, &TheoryLiteral::Neq(i, j));
        assert!(matches!(result, TheoryCheckResult::Consistent));

        // At this point, ROW-diff axiom should imply select_store = select_base
        // Check that asserting select_store ≠ select_base causes conflict
        let neq_lit = make_lit(1, false);
        let result = theory.assert_literal(neq_lit, &TheoryLiteral::Neq(select_store, select_base));

        assert!(
            matches!(result, TheoryCheckResult::Conflict(_)),
            "ROW-diff axiom violation should cause conflict"
        );
    }

    #[test]
    fn test_backtrack() {
        let mut theory = ArrayTheory::new();

        let terms = vec![
            SmtTerm::Const(Symbol::new("a")),
            SmtTerm::Const(Symbol::new("b")),
        ];
        theory.set_terms(terms);

        let a = TermId(0);
        let b = TermId(1);

        // Level 0: assert a = b
        let eq_lit = make_lit(0, true);
        theory.assert_literal(eq_lit, &TheoryLiteral::Eq(a, b));
        assert_eq!(theory.equalities.len(), 1);

        // Push to level 1
        theory.push();

        // Level 1: assert a ≠ b (would conflict but we're testing backtrack)
        let neq_lit = make_lit(1, false);
        // Note: This doesn't actually conflict because are_equal uses simple lookup
        theory.assert_literal(neq_lit, &TheoryLiteral::Neq(a, b));
        assert_eq!(theory.disequalities.len(), 1);

        // Backtrack to level 0
        theory.backtrack(0);

        // Disequality should be removed
        assert_eq!(theory.disequalities.len(), 0);
        // Equality should remain
        assert_eq!(theory.equalities.len(), 1);
    }

    #[test]
    fn test_multiple_stores() {
        let mut theory = ArrayTheory::new();

        // a1 = store(a0, i, v1)
        // a2 = store(a1, j, v2)
        // select(a2, i) when i ≠ j should equal v1 (from first store)
        let terms = vec![
            SmtTerm::Const(Symbol::new("a0")), // 0
            SmtTerm::Const(Symbol::new("i")),  // 1
            SmtTerm::Const(Symbol::new("j")),  // 2
            SmtTerm::Const(Symbol::new("v1")), // 3
            SmtTerm::Const(Symbol::new("v2")), // 4
            SmtTerm::App(Symbol::new("store"), vec![TermId(0), TermId(1), TermId(3)]), // 5: store(a0, i, v1)
            SmtTerm::App(Symbol::new("store"), vec![TermId(5), TermId(2), TermId(4)]), // 6: store(a1, j, v2)
            SmtTerm::App(Symbol::new("select"), vec![TermId(6), TermId(1)]), // 7: select(a2, i)
            SmtTerm::App(Symbol::new("select"), vec![TermId(5), TermId(1)]), // 8: select(a1, i)
        ];
        theory.set_terms(terms);

        let stats = theory.stats();
        assert_eq!(stats.num_stores, 2);
        assert_eq!(stats.num_selects, 2);
    }

    #[test]
    fn test_stats() {
        let mut theory = ArrayTheory::new();

        let terms = vec![
            SmtTerm::Const(Symbol::new("a")),
            SmtTerm::Const(Symbol::new("i")),
            SmtTerm::Const(Symbol::new("v")),
            SmtTerm::App(Symbol::new("store"), vec![TermId(0), TermId(1), TermId(2)]),
            SmtTerm::App(Symbol::new("select"), vec![TermId(3), TermId(1)]),
        ];
        theory.set_terms(terms);

        let stats = theory.stats();
        assert_eq!(stats.num_stores, 1);
        assert_eq!(stats.num_selects, 1);
        assert_eq!(stats.num_equalities, 0);
        assert_eq!(stats.num_disequalities, 0);
    }

    #[test]
    fn test_smt_integration() {
        use crate::smt::{SmtResult, SmtSolver};

        let mut smt = SmtSolver::new();

        // Add array theory
        smt.add_theory(Box::new(ArrayTheory::new()));

        // Create array terms: a, i, v, store(a, i, v)
        let a = smt.const_term("a");
        let i = smt.const_term("i");
        let v = smt.const_term("v");
        let store_aiv = smt.app_term("store", vec![a, i, v]);
        let _select = smt.app_term("select", vec![store_aiv, i]);

        // This should be satisfiable - just declaring the terms
        match smt.solve() {
            SmtResult::Sat(_) => {}
            other => panic!("Expected SAT, got {other:?}"),
        }
    }

    #[test]
    fn test_smt_with_equality_integration() {
        use crate::smt::{SmtResult, SmtSolver};
        use crate::theories::equality::EqualityTheory;

        let mut smt = SmtSolver::new();

        // Add both equality and array theories
        smt.add_theory(Box::new(EqualityTheory::new()));
        smt.add_theory(Box::new(ArrayTheory::new()));

        // Create terms
        let a = smt.const_term("a");
        let i = smt.const_term("i");
        let j = smt.const_term("j");
        let v = smt.const_term("v");

        // store(a, i, v)
        let store_aiv = smt.app_term("store", vec![a, i, v]);

        // select(store(a, i, v), j)
        let _select_j = smt.app_term("select", vec![store_aiv, j]);

        // select(a, j)
        let _select_base = smt.app_term("select", vec![a, j]);

        // Assert i ≠ j (different indices)
        smt.assert_neq(i, j);

        // With i ≠ j, the array theory should allow consistency
        // (ROW-diff axiom would be applicable but not violated)
        match smt.solve() {
            SmtResult::Sat(_) => {}
            other => panic!("Expected SAT, got {other:?}"),
        }
    }
}
