//! 2-Watched Literal scheme

use crate::literal::Literal;

/// Index of a clause in the clause database
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(kani, derive(kani::Arbitrary))]
pub struct ClauseRef(pub u32);

/// A watcher entry (8 bytes)
///
/// For binary clauses: `blocker` stores the other literal (not just a hint)
/// For longer clauses: `blocker` is a hint for early satisfaction check
///
/// The high bit of `clause` indicates whether this is a binary clause.
#[derive(Debug, Clone, Copy)]
pub struct Watcher {
    /// The clause being watched. High bit set if this is a binary clause.
    clause: ClauseRef,
    /// For binary clauses: the other literal in the clause
    /// For non-binary clauses: blocker literal for faster filtering
    blocker: Literal,
}

impl Watcher {
    /// High bit flag for binary clauses
    const BINARY_FLAG: u32 = 0x8000_0000;

    /// Create a watcher for a binary clause
    #[inline]
    pub fn binary(clause: ClauseRef, other_lit: Literal) -> Self {
        Watcher {
            clause: ClauseRef(clause.0 | Self::BINARY_FLAG),
            blocker: other_lit,
        }
    }

    /// Create a watcher for a non-binary clause (3+ literals)
    #[inline]
    pub fn new(clause: ClauseRef, blocker: Literal) -> Self {
        debug_assert!(clause.0 & Self::BINARY_FLAG == 0, "ClauseRef too large");
        Watcher { clause, blocker }
    }

    /// Check if this is a binary clause watcher
    #[inline]
    pub fn is_binary(&self) -> bool {
        self.clause.0 & Self::BINARY_FLAG != 0
    }

    /// Get the clause reference (strips binary flag)
    #[inline]
    pub fn clause_ref(&self) -> ClauseRef {
        ClauseRef(self.clause.0 & !Self::BINARY_FLAG)
    }

    /// Get the blocker/other literal
    #[inline]
    pub fn blocker(&self) -> Literal {
        self.blocker
    }

    /// Set the blocker (for updating when clause becomes satisfied)
    #[inline]
    pub fn set_blocker(&mut self, lit: Literal) {
        self.blocker = lit;
    }
}

/// Watched literal lists
#[derive(Debug, Default)]
pub struct WatchedLists {
    /// For each literal, the list of clauses watching it
    watches: Vec<Vec<Watcher>>,
}

impl WatchedLists {
    /// Create new watched lists for n variables
    pub fn new(num_vars: usize) -> Self {
        WatchedLists {
            watches: vec![Vec::new(); num_vars * 2],
        }
    }

    /// Ensure the watched lists can index literals for `num_vars` variables.
    pub fn ensure_num_vars(&mut self, num_vars: usize) {
        let target = num_vars.saturating_mul(2);
        if self.watches.len() < target {
            self.watches.resize_with(target, Vec::new);
        }
    }

    /// Clear all watch lists without deallocating the outer Vec.
    /// This is more efficient than creating a new WatchedLists for incremental solving.
    pub fn clear(&mut self) {
        for watch_list in &mut self.watches {
            watch_list.clear();
        }
    }

    /// Add a watcher for a literal
    #[inline]
    pub fn add_watch(&mut self, lit: Literal, watcher: Watcher) {
        self.watches[lit.index()].push(watcher);
    }

    /// Get watchers for a literal
    #[inline]
    pub fn get_watches(&self, lit: Literal) -> &[Watcher] {
        &self.watches[lit.index()]
    }

    /// Get mutable watchers for a literal
    #[inline]
    pub fn get_watches_mut(&mut self, lit: Literal) -> &mut Vec<Watcher> {
        &mut self.watches[lit.index()]
    }

    /// Count total watches for a clause (used for verification)
    #[cfg(any(test, kani))]
    pub fn count_watches_for_clause(&self, clause_ref: ClauseRef) -> usize {
        let mut count = 0;
        for watch_list in &self.watches {
            for watcher in watch_list {
                if watcher.clause_ref() == clause_ref {
                    count += 1;
                }
            }
        }
        count
    }

    /// Get the number of literal indices (for verification)
    #[cfg(any(test, kani))]
    pub fn num_literals(&self) -> usize {
        self.watches.len()
    }

    /// Get the number of watch lists (for memory statistics)
    pub fn num_watch_lists(&self) -> usize {
        self.watches.len()
    }

    /// Get the capacity of a specific watch list (for memory statistics)
    pub fn watch_list_capacity(&self, lit_idx: usize) -> usize {
        if lit_idx < self.watches.len() {
            self.watches[lit_idx].capacity()
        } else {
            0
        }
    }

    /// Get raw access to watch list for in-place modification
    /// Returns (ptr, len) tuple for the watch list
    /// SAFETY: The caller must ensure no concurrent access to this watch list
    #[inline]
    pub fn get_watch_list_raw(&mut self, lit: Literal) -> (*mut Watcher, usize) {
        let list = &mut self.watches[lit.index()];
        (list.as_mut_ptr(), list.len())
    }

    /// Truncate a watch list to the given length
    /// SAFETY: new_len must be <= current length
    #[inline]
    pub fn truncate_watches(&mut self, lit: Literal, new_len: usize) {
        self.watches[lit.index()].truncate(new_len);
    }

    /// Get the length of a watch list
    #[inline]
    pub fn watch_count(&self, lit: Literal) -> usize {
        self.watches[lit.index()].len()
    }

    /// Prefetch the watch list for a literal
    ///
    /// This is called when assigning a literal to prefetch the watch list
    /// that will be processed when the literal's negation is propagated.
    /// This follows CaDiCaL's technique of prefetching in search_assign.
    #[inline]
    pub fn prefetch(&self, lit: Literal) {
        let list = &self.watches[lit.index()];
        if !list.is_empty() {
            // Prefetch the first element of the watch list
            // This brings the first cache line into L1 cache
            let ptr = list.as_ptr();
            // Use architecture-specific prefetch instructions
            #[cfg(target_arch = "x86_64")]
            {
                // SAFETY: list is non-empty, so ptr is valid
                unsafe {
                    std::arch::x86_64::_mm_prefetch(
                        ptr as *const i8,
                        std::arch::x86_64::_MM_HINT_T1,
                    );
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                // SAFETY: list is non-empty, so ptr is valid
                // Use inline assembly for aarch64 prefetch (PRFM instruction)
                unsafe {
                    std::arch::asm!(
                        "prfm pldl1strm, [{ptr}]",
                        ptr = in(reg) ptr,
                        options(nostack, preserves_flags)
                    );
                }
            }
            // For other architectures, no prefetch
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                let _ = ptr;
            }
        }
    }
}

// ============================================================================
// Kani Verification Harnesses
// ============================================================================

#[cfg(kani)]
mod verification {
    use super::*;
    use crate::literal::Variable;

    /// Watcher struct preserves its fields correctly (non-binary)
    #[kani::proof]
    fn watcher_fields_preserved() {
        let clause: ClauseRef = kani::any();
        let blocker: Literal = kani::any();

        // Bound to prevent overflow and avoid binary flag collision
        kani::assume(clause.0 < 1000);
        kani::assume(blocker.0 < 1000);

        let watcher = Watcher::new(clause, blocker);

        // Fields are preserved
        assert!(watcher.clause_ref() == clause);
        assert!(watcher.blocker() == blocker);
        assert!(!watcher.is_binary());
    }

    /// Binary watcher preserves its fields correctly
    #[kani::proof]
    fn binary_watcher_fields_preserved() {
        let clause: ClauseRef = kani::any();
        let other_lit: Literal = kani::any();

        // Bound to prevent overflow
        kani::assume(clause.0 < 1000);
        kani::assume(other_lit.0 < 1000);

        let watcher = Watcher::binary(clause, other_lit);

        // Fields are preserved, binary flag is set
        assert!(watcher.clause_ref() == clause);
        assert!(watcher.blocker() == other_lit);
        assert!(watcher.is_binary());
    }

    /// ClauseRef is correctly identified
    #[kani::proof]
    fn clause_ref_equality() {
        let a: ClauseRef = kani::any();
        let b: ClauseRef = kani::any();

        kani::assume(a.0 < 1000 && b.0 < 1000);

        // Equality is based on inner value
        if a.0 == b.0 {
            assert!(a == b);
        }
        if a != b {
            assert!(a.0 != b.0);
        }
    }

    /// Literal index calculation is consistent for watched lists
    /// This verifies the watched list indexing scheme
    #[kani::proof]
    fn literal_index_for_watches() {
        let var: Variable = kani::any();
        kani::assume(var.0 < 100);

        let pos = Literal::positive(var);
        let neg = Literal::negative(var);

        // Positive and negative have different indices
        assert!(pos.index() != neg.index());

        // Both indices are within bounds for a watched list of size 2*num_vars
        // where num_vars > var.0
        let expected_max_index = (var.0 as usize + 1) * 2;
        assert!(pos.index() < expected_max_index);
        assert!(neg.index() < expected_max_index);
    }

    /// Adding a watch increases the watch count by exactly one
    #[kani::proof]
    fn watch_add_increases_count() {
        let num_vars = 4;
        let mut watches = WatchedLists::new(num_vars);

        let var_idx: u32 = kani::any();
        kani::assume(var_idx < num_vars as u32);
        let lit = Literal::positive(Variable(var_idx));

        let clause: ClauseRef = kani::any();
        let blocker: Literal = kani::any();
        kani::assume(clause.0 < 100 && blocker.0 < 100);

        let before = watches.watch_count(lit);
        watches.add_watch(lit, Watcher::new(clause, blocker));
        let after = watches.watch_count(lit);

        assert_eq!(after, before + 1);
    }

    /// Set blocker preserves clause_ref and is_binary flag
    #[kani::proof]
    fn set_blocker_preserves_fields() {
        let clause: ClauseRef = kani::any();
        let blocker1: Literal = kani::any();
        let blocker2: Literal = kani::any();

        kani::assume(clause.0 < 1000);
        kani::assume(blocker1.0 < 1000 && blocker2.0 < 1000);

        // Test non-binary watcher
        let mut watcher = Watcher::new(clause, blocker1);
        let original_clause = watcher.clause_ref();
        let original_is_binary = watcher.is_binary();

        watcher.set_blocker(blocker2);

        // clause_ref and is_binary should be unchanged
        assert_eq!(watcher.clause_ref(), original_clause);
        assert_eq!(watcher.is_binary(), original_is_binary);
        // blocker should be updated
        assert_eq!(watcher.blocker(), blocker2);
    }

    /// Binary watcher set_blocker also preserves fields
    #[kani::proof]
    fn binary_set_blocker_preserves_fields() {
        let clause: ClauseRef = kani::any();
        let blocker1: Literal = kani::any();
        let blocker2: Literal = kani::any();

        kani::assume(clause.0 < 1000);
        kani::assume(blocker1.0 < 1000 && blocker2.0 < 1000);

        // Test binary watcher
        let mut watcher = Watcher::binary(clause, blocker1);
        let original_clause = watcher.clause_ref();
        let original_is_binary = watcher.is_binary();

        watcher.set_blocker(blocker2);

        // clause_ref and is_binary should be unchanged
        assert_eq!(watcher.clause_ref(), original_clause);
        assert_eq!(watcher.is_binary(), original_is_binary);
        // blocker should be updated
        assert_eq!(watcher.blocker(), blocker2);
    }

    /// Watched list clear resets all counts to zero
    #[kani::proof]
    fn watch_clear_resets_counts() {
        let num_vars = 4;
        let mut watches = WatchedLists::new(num_vars);

        // Add some watchers
        let lit = Literal::positive(Variable(0));
        watches.add_watch(lit, Watcher::new(ClauseRef(0), Literal(1)));
        watches.add_watch(lit, Watcher::new(ClauseRef(1), Literal(2)));

        // Clear
        watches.clear();

        // All counts should be zero
        for var_idx in 0..num_vars {
            let pos_lit = Literal::positive(Variable(var_idx as u32));
            let neg_lit = Literal::negative(Variable(var_idx as u32));
            assert_eq!(watches.watch_count(pos_lit), 0);
            assert_eq!(watches.watch_count(neg_lit), 0);
        }
    }
}

// ============================================================================
// Property Tests (for Vec-based operations)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::literal::Variable;
    use proptest::prelude::*;

    proptest! {
        /// Adding a watch increases the count by one
        #[test]
        fn prop_add_watch_increases_count(var_idx in 0u32..10) {
            let mut watches = WatchedLists::new(10);
            let lit = Literal::positive(Variable(var_idx));
            let blocker = Literal::negative(Variable(var_idx));
            let clause_ref = ClauseRef(0);

            let before = watches.get_watches(lit).len();
            watches.add_watch(lit, Watcher::new(clause_ref, blocker));
            let after = watches.get_watches(lit).len();

            prop_assert_eq!(after, before + 1);
        }

        /// Watches are empty after initialization
        #[test]
        fn prop_watches_initially_empty(num_vars in 1usize..20, var_idx in 0u32..20) {
            prop_assume!(var_idx < num_vars as u32);
            let watches = WatchedLists::new(num_vars);
            let pos = Literal::positive(Variable(var_idx));
            let neg = Literal::negative(Variable(var_idx));

            prop_assert!(watches.get_watches(pos).is_empty());
            prop_assert!(watches.get_watches(neg).is_empty());
        }

        /// Blocker and clause are preserved when adding a watch
        #[test]
        fn prop_watcher_preserved(
            var1 in 0u32..10,
            var2 in 0u32..10,
            clause_id in 0u32..100
        ) {
            let mut watches = WatchedLists::new(10);
            let lit = Literal::positive(Variable(var1));
            let blocker = Literal::negative(Variable(var2));
            let clause_ref = ClauseRef(clause_id);

            watches.add_watch(lit, Watcher::new(clause_ref, blocker));

            let watchers = watches.get_watches(lit);
            prop_assert_eq!(watchers.len(), 1);
            prop_assert_eq!(watchers[0].clause_ref(), clause_ref);
            prop_assert_eq!(watchers[0].blocker(), blocker);
        }

        /// Multiple watches can be added to the same literal
        #[test]
        fn prop_multiple_watches(
            var_idx in 0u32..10,
            num_watches in 1usize..10
        ) {
            let mut watches = WatchedLists::new(10);
            let lit = Literal::positive(Variable(var_idx));

            for i in 0..num_watches {
                watches.add_watch(lit, Watcher::new(
                    ClauseRef(i as u32),
                    Literal::positive(Variable(0)),
                ));
            }

            prop_assert_eq!(watches.get_watches(lit).len(), num_watches);
        }

        /// Clear empties all watch lists
        #[test]
        fn prop_clear_empties_all(
            num_vars in 1usize..20,
            var_idx in 0u32..20
        ) {
            prop_assume!(var_idx < num_vars as u32);
            let mut watches = WatchedLists::new(num_vars);
            let pos = Literal::positive(Variable(var_idx));
            let neg = Literal::negative(Variable(var_idx));

            // Add some watches
            watches.add_watch(pos, Watcher::new(ClauseRef(0), neg));
            watches.add_watch(neg, Watcher::new(ClauseRef(1), pos));

            // Verify they were added
            prop_assert_eq!(watches.get_watches(pos).len(), 1);
            prop_assert_eq!(watches.get_watches(neg).len(), 1);

            // Clear and verify empty
            watches.clear();
            prop_assert!(watches.get_watches(pos).is_empty());
            prop_assert!(watches.get_watches(neg).is_empty());
        }
    }
}
