//! Warmup-based phase initialization
//!
//! Implements CaDiCaL-style warmup for finding good initial phases.
//! Unlike walk (which uses ProbSAT random walk), warmup leverages CDCL's
//! propagation strength to set phases efficiently.
//!
//! ## How It Works
//!
//! 1. Make decisions using standard heuristics (VMTF in focused mode)
//! 2. Propagate using 2-watched literals (ignoring conflicts)
//! 3. Save resulting assignment as target phases for local search
//!
//! ## Why Warmup is Better Than Walk for Phase Initialization
//!
//! - Walk: O(n²) per flip due to break-value computation requiring clause scans
//! - Warmup: O(1) amortized per propagation using 2-watched literals
//!
//! CaDiCaL uses warmup to prepare phases before walk. For small/medium instances,
//! warmup alone often provides sufficient phase quality without the walk overhead.

use crate::clause_db::ClauseDB;
use crate::Literal;

/// Statistics from warmup phase
#[derive(Debug, Default, Clone)]
pub struct WarmupStats {
    /// Number of warmup rounds executed
    pub warmup_count: u64,
    /// Number of decisions made during warmup
    pub decisions: u64,
    /// Number of propagations made during warmup
    pub propagations: u64,
    /// Number of conflicts encountered (and ignored)
    pub conflicts: u64,
}

/// Warmup state for propagation
pub struct Warmup {
    /// Current assignment (None = unassigned)
    assignment: Vec<Option<bool>>,
    /// Trail of assigned literals
    trail: Vec<Literal>,
    /// Propagation pointer into trail
    propagated: usize,
    /// Watch lists: watches[2*var + sign] = list of clause indices watching -lit
    watches: Vec<Vec<WatchEntry>>,
    /// Number of variables
    num_vars: usize,
}

/// Entry in watch list
#[derive(Clone, Copy)]
struct WatchEntry {
    /// Blocking literal (optimization to avoid clause access)
    blocking: Literal,
    /// Clause index
    clause_idx: usize,
}

impl Warmup {
    /// Create warmup state for n variables
    pub fn new(num_vars: usize) -> Self {
        Warmup {
            assignment: vec![None; num_vars],
            trail: Vec::with_capacity(num_vars),
            propagated: 0,
            watches: vec![Vec::new(); num_vars * 2],
            num_vars,
        }
    }

    /// Reset state for new warmup round
    pub fn reset(&mut self) {
        for val in &mut self.assignment {
            *val = None;
        }
        self.trail.clear();
        self.propagated = 0;
        for watch in &mut self.watches {
            watch.clear();
        }
    }

    /// Initialize watch lists from clause database
    fn init_watches(&mut self, clause_db: &ClauseDB) {
        for watch in &mut self.watches {
            watch.clear();
        }

        for idx in clause_db.indices() {
            let header = clause_db.header(idx);
            if header.is_empty() || header.is_learned() {
                continue;
            }

            let len = header.len();
            if len < 2 {
                continue; // Unit clauses handled separately
            }

            let lits = clause_db.literals(idx);
            let lit0 = lits[0];
            let lit1 = lits[1];

            // Watch -lit0 with blocking literal lit1
            let watch_idx0 = self.watch_index(lit0.negated());
            self.watches[watch_idx0].push(WatchEntry {
                blocking: lit1,
                clause_idx: idx,
            });

            // Watch -lit1 with blocking literal lit0
            let watch_idx1 = self.watch_index(lit1.negated());
            self.watches[watch_idx1].push(WatchEntry {
                blocking: lit0,
                clause_idx: idx,
            });
        }
    }

    /// Get watch list index for a literal
    #[inline]
    fn watch_index(&self, lit: Literal) -> usize {
        let var = lit.variable().index();
        var * 2 + if lit.is_positive() { 0 } else { 1 }
    }

    /// Get value of a literal under current assignment
    #[inline]
    fn value(&self, lit: Literal) -> Option<bool> {
        let var = lit.variable().index();
        self.assignment[var].map(|v| if lit.is_positive() { v } else { !v })
    }

    /// Assign a literal during warmup
    fn assign(&mut self, lit: Literal) {
        let var = lit.variable().index();
        let val = lit.is_positive();
        debug_assert!(self.assignment[var].is_none());
        self.assignment[var] = Some(val);
        self.trail.push(lit);
    }

    /// Propagate beyond conflicts (warmup-specific)
    /// Returns number of conflicts encountered
    fn propagate_beyond_conflict(
        &mut self,
        clause_db: &ClauseDB,
        stats: &mut WarmupStats,
    ) -> usize {
        let mut conflicts = 0;

        while self.propagated < self.trail.len() {
            let lit = self.trail[self.propagated];
            self.propagated += 1;
            stats.propagations += 1;

            // Get watches for negation of propagated literal
            let watch_idx = self.watch_index(lit);
            let mut watches = std::mem::take(&mut self.watches[watch_idx]);
            let mut j = 0;

            for i in 0..watches.len() {
                let entry = watches[i];

                // Check blocking literal first (optimization)
                if let Some(true) = self.value(entry.blocking) {
                    watches[j] = entry;
                    j += 1;
                    continue;
                }

                // Need to look at clause
                let header = clause_db.header(entry.clause_idx);
                if header.is_empty() {
                    // Clause was deleted, skip
                    continue;
                }

                let lits = clause_db.literals(entry.clause_idx);
                let len = lits.len();

                // Find the other watched literal
                let lit0 = lits[0];
                let lit1 = lits[1];
                let neg_lit = lit.negated();
                let other = if lit0 == neg_lit { lit1 } else { lit0 };

                // Check if other watched literal is satisfied
                if let Some(true) = self.value(other) {
                    watches[j] = WatchEntry {
                        blocking: other,
                        clause_idx: entry.clause_idx,
                    };
                    j += 1;
                    continue;
                }

                // Look for new literal to watch
                let mut found = false;
                for &new_lit in &lits[2..len] {
                    if self.value(new_lit) != Some(false) {
                        // Found a new literal to watch (unassigned or true)
                        // Swap it to position 0 or 1
                        // We can't actually swap in clause_db, so we just update watches
                        let new_watch_idx = self.watch_index(new_lit.negated());
                        self.watches[new_watch_idx].push(WatchEntry {
                            blocking: other,
                            clause_idx: entry.clause_idx,
                        });
                        found = true;
                        break;
                    }
                }

                if found {
                    // Watch moved, don't keep this entry
                    continue;
                }

                // All other literals are false
                // Check if other watched literal is unit or conflict
                watches[j] = entry;
                j += 1;

                if let Some(false) = self.value(other) {
                    // Conflict - ignore it (warmup behavior)
                    conflicts += 1;
                    stats.conflicts += 1;
                } else if self.value(other).is_none() {
                    // Unit propagation
                    self.assign(other);
                }
            }

            watches.truncate(j);
            self.watches[watch_idx] = watches;
        }

        conflicts
    }

    /// Perform warmup: propagate with decisions to set phases
    pub fn warmup(
        &mut self,
        clause_db: &ClauseDB,
        phases: &[Option<bool>],
        target_phases: &mut [Option<bool>],
        stats: &mut WarmupStats,
    ) {
        stats.warmup_count += 1;

        // Reset state
        self.reset();

        // Initialize watches
        self.init_watches(clause_db);

        // Handle unit clauses first
        for idx in clause_db.indices() {
            let header = clause_db.header(idx);
            if header.is_empty() || header.is_learned() {
                continue;
            }
            if header.len() == 1 {
                let lit = clause_db.literals(idx)[0];
                let var = lit.variable().index();
                if self.assignment[var].is_none() {
                    self.assign(lit);
                }
            }
        }

        // Propagate unit clauses
        self.propagate_beyond_conflict(clause_db, stats);

        // Make decisions until all variables assigned
        // Note: We need the index-based loop here because we mutate `self` inside the loop
        // via `self.assign()` and `self.propagate_beyond_conflict()`, which prevents us from
        // borrowing `self.assignment` in an iterator.
        #[allow(clippy::needless_range_loop)]
        for var in 0..self.num_vars {
            if self.assignment[var].is_some() {
                continue;
            }

            // Use existing phase or default to true
            let phase = phases[var].unwrap_or(true);
            let variable = crate::Variable(var as u32);
            let lit = if phase {
                Literal::positive(variable)
            } else {
                Literal::negative(variable)
            };

            self.assign(lit);
            stats.decisions += 1;

            // Propagate (ignoring conflicts)
            self.propagate_beyond_conflict(clause_db, stats);
        }

        // Copy warmup assignment to target phases
        for (&assignment, target) in self
            .assignment
            .iter()
            .zip(target_phases.iter_mut())
            .take(self.num_vars)
        {
            if let Some(val) = assignment {
                *target = Some(val);
            }
        }
    }
}

/// Performs warmup phase initialization.
///
/// Uses CDCL propagation (ignoring conflicts) to find good initial phases.
/// This is more efficient than walk for small/medium instances because
/// it uses O(1) amortized 2-watched literal propagation instead of O(n²)
/// break-value computation.
pub fn warmup(
    clause_db: &ClauseDB,
    num_vars: usize,
    phases: &[Option<bool>],
    target_phases: &mut [Option<bool>],
    stats: &mut WarmupStats,
) {
    let mut state = Warmup::new(num_vars);
    state.warmup(clause_db, phases, target_phases, stats);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warmup_stats_default() {
        let stats = WarmupStats::default();
        assert_eq!(stats.warmup_count, 0);
        assert_eq!(stats.decisions, 0);
        assert_eq!(stats.propagations, 0);
        assert_eq!(stats.conflicts, 0);
    }

    #[test]
    fn test_warmup_state_creation() {
        let state = Warmup::new(10);
        assert_eq!(state.num_vars, 10);
        assert_eq!(state.assignment.len(), 10);
        assert_eq!(state.trail.len(), 0);
        assert_eq!(state.watches.len(), 20); // 2 * num_vars
    }

    #[test]
    fn test_warmup_reset() {
        let mut state = Warmup::new(5);

        // Simulate some state
        state.assignment[0] = Some(true);
        state.trail.push(Literal::positive(crate::Variable(0)));
        state.propagated = 1;

        // Reset
        state.reset();

        assert!(state.assignment.iter().all(|a| a.is_none()));
        assert_eq!(state.trail.len(), 0);
        assert_eq!(state.propagated, 0);
    }

    #[test]
    fn test_warmup_value() {
        let mut state = Warmup::new(3);

        // var0 = true
        state.assignment[0] = Some(true);
        // var1 = false
        state.assignment[1] = Some(false);
        // var2 unassigned

        // Positive literal for true variable -> true
        assert_eq!(
            state.value(Literal::positive(crate::Variable(0))),
            Some(true)
        );
        // Negative literal for true variable -> false
        assert_eq!(
            state.value(Literal::negative(crate::Variable(0))),
            Some(false)
        );
        // Positive literal for false variable -> false
        assert_eq!(
            state.value(Literal::positive(crate::Variable(1))),
            Some(false)
        );
        // Negative literal for false variable -> true
        assert_eq!(
            state.value(Literal::negative(crate::Variable(1))),
            Some(true)
        );
        // Unassigned variable -> None
        assert_eq!(state.value(Literal::positive(crate::Variable(2))), None);
    }

    #[test]
    fn test_warmup_watch_index() {
        let state = Warmup::new(10);

        // Positive literal for var 0 -> index 0
        assert_eq!(state.watch_index(Literal::positive(crate::Variable(0))), 0);
        // Negative literal for var 0 -> index 1
        assert_eq!(state.watch_index(Literal::negative(crate::Variable(0))), 1);
        // Positive literal for var 5 -> index 10
        assert_eq!(state.watch_index(Literal::positive(crate::Variable(5))), 10);
        // Negative literal for var 5 -> index 11
        assert_eq!(state.watch_index(Literal::negative(crate::Variable(5))), 11);
    }
}
