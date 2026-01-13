//! Failed Literal Probing
//!
//! Failed literal probing is an inprocessing technique that detects literals
//! which lead to conflicts when assumed, determining their forced assignments.
//!
//! ## Algorithm
//!
//! For each candidate probe literal `p`:
//! 1. Temporarily assign `p` at decision level 1
//! 2. Propagate to completion
//! 3. If conflict is found: `p` is a "failed literal" and `¬p` must be true
//! 4. If no conflict: backtrack and try next probe
//!
//! ## Probe Selection
//!
//! We focus on "root" literals in the binary implication graph - literals
//! that appear negated in binary clauses but not positively. These are the
//! most likely to produce useful information through probing.
//!
//! ## References
//!
//! - CaDiCaL probe.cpp
//! - Heule & van Maaren, "Look-Ahead Based SAT Solvers" (2009)

use crate::clause_db::ClauseDB;
use crate::literal::{Literal, Variable};
use crate::watched::ClauseRef;

/// Statistics for failed literal probing
#[derive(Debug, Clone, Default)]
pub struct ProbeStats {
    /// Number of probing rounds
    pub rounds: u64,
    /// Number of literals probed
    pub probed: u64,
    /// Number of failed literals found
    pub failed: u64,
    /// Number of units derived (from failed literals)
    pub units_derived: u64,
}

/// Result of a probing round
#[derive(Debug, Clone)]
pub struct ProbeResult {
    /// Literals that were found to be failed (their negation is forced)
    pub failed_literals: Vec<Literal>,
    /// Number of literals probed
    pub probed_count: u64,
    /// Whether UNSAT was detected (conflict at level 0)
    pub is_unsat: bool,
}

/// Failed literal prober
///
/// Identifies and probes candidate literals to find forced assignments.
pub struct Prober {
    /// Number of variables
    num_vars: usize,
    /// Probe candidates (literals to probe)
    probes: Vec<Literal>,
    /// Occurrence counts: how many times each literal appears in binary clauses
    /// Index: literal.index()
    bin_occs: Vec<u32>,
    /// Last propagation fixed point for each literal
    /// If `propfixed[lit]` >= current_fixed, no need to reprobe
    propfixed: Vec<i64>,
    /// Statistics
    stats: ProbeStats,
}

impl Prober {
    /// Create a new prober for the given number of variables
    pub fn new(num_vars: usize) -> Self {
        let num_lits = num_vars * 2;
        Prober {
            num_vars,
            probes: Vec::new(),
            bin_occs: vec![0; num_lits],
            propfixed: vec![-1; num_lits],
            stats: ProbeStats::default(),
        }
    }

    /// Ensure internal buffers can handle `num_vars` variables.
    pub fn ensure_num_vars(&mut self, num_vars: usize) {
        if self.num_vars >= num_vars {
            return;
        }
        self.num_vars = num_vars;
        let num_lits = num_vars.saturating_mul(2);
        if self.bin_occs.len() < num_lits {
            self.bin_occs.resize(num_lits, 0);
        }
        if self.propfixed.len() < num_lits {
            self.propfixed.resize(num_lits, -1);
        }
    }

    /// Get probing statistics
    pub fn stats(&self) -> &ProbeStats {
        &self.stats
    }

    /// Reset occurrence counts
    fn reset_occs(&mut self) {
        self.bin_occs.fill(0);
    }

    /// Count binary clause occurrences
    ///
    /// For each binary clause {a, b}:
    /// - Increment `bin_occs[a]` and `bin_occs[b]`
    fn count_binary_occurrences(&mut self, clauses: &ClauseDB, assignment: &[Option<bool>]) {
        self.reset_occs();

        for idx in clauses.indices() {
            // Skip empty/deleted clauses
            let header = clauses.header(idx);
            if header.is_empty() {
                continue;
            }

            let lits = clauses.literals(idx);

            // Check if it's effectively binary (ignoring falsified literals at level 0)
            let mut unassigned: Vec<Literal> = Vec::new();
            let mut is_satisfied = false;

            for &lit in lits {
                let var_idx = lit.variable().index();
                // Skip literals with out-of-bounds variables
                if var_idx >= assignment.len() {
                    continue;
                }
                match assignment[var_idx] {
                    Some(true) if lit.is_positive() => {
                        is_satisfied = true;
                        break;
                    }
                    Some(false) if !lit.is_positive() => {
                        is_satisfied = true;
                        break;
                    }
                    None => {
                        unassigned.push(lit);
                    }
                    _ => {} // falsified literal, skip
                }
            }

            if is_satisfied {
                continue;
            }

            // Only count actual binary clauses
            if unassigned.len() == 2 {
                let a = unassigned[0];
                let b = unassigned[1];
                // Bounds check for bin_occs array
                if a.index() < self.bin_occs.len() && b.index() < self.bin_occs.len() {
                    self.bin_occs[a.index()] += 1;
                    self.bin_occs[b.index()] += 1;
                }
            }
        }
    }

    /// Generate probe candidates
    ///
    /// Probes are "root" literals in the binary implication graph:
    /// - Appear negated in binary clauses
    /// - Do not appear positively in binary clauses
    ///
    /// These are good candidates because probing them exercises the
    /// binary implication chain without immediately satisfying it.
    pub fn generate_probes(
        &mut self,
        clauses: &ClauseDB,
        assignment: &[Option<bool>],
        current_fixed: i64,
    ) {
        self.probes.clear();
        self.count_binary_occurrences(clauses, assignment);

        // Limit iteration to actual assignment length and bin_occs capacity
        let max_lits = self.bin_occs.len();
        let max_vars = self.num_vars.min(assignment.len()).min(max_lits / 2);

        #[allow(clippy::needless_range_loop)]
        for var_idx in 0..max_vars {
            // Skip assigned variables
            if assignment[var_idx].is_some() {
                continue;
            }

            let var = Variable(var_idx as u32);
            let pos_lit = Literal::positive(var);
            let neg_lit = Literal::negative(var);

            // Bounds check for bin_occs and propfixed
            if pos_lit.index() >= max_lits || neg_lit.index() >= max_lits {
                continue;
            }
            if pos_lit.index() >= self.propfixed.len() || neg_lit.index() >= self.propfixed.len() {
                continue;
            }

            let pos_occs = self.bin_occs[pos_lit.index()];
            let neg_occs = self.bin_occs[neg_lit.index()];

            // Look for "roots": one polarity has occurrences, other doesn't
            if pos_occs > 0 && neg_occs == 0 {
                // Positive appears in binary clauses but not negative
                // Probing positive is more likely to propagate
                // Skip if already probed without changes
                if self.propfixed[pos_lit.index()] < current_fixed {
                    self.probes.push(pos_lit);
                }
            } else if neg_occs > 0 && pos_occs == 0 {
                // Negative appears in binary clauses but not positive
                if self.propfixed[neg_lit.index()] < current_fixed {
                    self.probes.push(neg_lit);
                }
            }
        }

        // Sort probes by occurrence count (higher first) for better pruning
        let bin_occs_len = self.bin_occs.len();
        self.probes.sort_by(|a, b| {
            let a_idx = a.negated().index();
            let b_idx = b.negated().index();
            let a_occs = if a_idx < bin_occs_len {
                self.bin_occs[a_idx]
            } else {
                0
            };
            let b_occs = if b_idx < bin_occs_len {
                self.bin_occs[b_idx]
            } else {
                0
            };
            b_occs.cmp(&a_occs)
        });
    }

    /// Get the next probe candidate, or None if exhausted
    pub fn next_probe(&mut self) -> Option<Literal> {
        self.probes.pop()
    }

    /// Mark a literal as probed at the current fixed point
    pub fn mark_probed(&mut self, lit: Literal, current_fixed: i64) {
        let idx = lit.index();
        if idx < self.propfixed.len() {
            self.propfixed[idx] = current_fixed;
        }
    }

    /// Reset the prober for a new solving session
    pub fn reset(&mut self) {
        self.probes.clear();
        self.propfixed.fill(-1);
    }

    /// Update statistics after finding a failed literal
    pub fn record_failed(&mut self) {
        self.stats.failed += 1;
        self.stats.units_derived += 1;
    }

    /// Update statistics after probing a literal
    pub fn record_probed(&mut self) {
        self.stats.probed += 1;
    }

    /// Update statistics after completing a round
    pub fn record_round(&mut self) {
        self.stats.rounds += 1;
    }

    /// Check if we have any probes remaining
    pub fn has_probes(&self) -> bool {
        !self.probes.is_empty()
    }

    /// Get remaining probe count
    pub fn remaining_probes(&self) -> usize {
        self.probes.len()
    }
}

/// Find the 1UIP dominator in a failed literal conflict
///
/// Given a conflict at decision level 1, finds the unique implication point
/// (UIP) - the single literal at level 1 that all paths to the conflict
/// pass through.
///
/// Returns the negation of the UIP (the forced literal).
pub fn find_failed_literal_uip(
    conflict_clause: &[Literal],
    trail: &[Literal],
    level: &[u32],
    reason: &[Option<ClauseRef>],
    clauses: &ClauseDB,
) -> Option<Literal> {
    // For failed literal probing at level 1, we find the dominator
    // of all literals in the conflict clause that are at level 1

    let mut seen: Vec<bool> = vec![false; level.len()];
    let mut count_at_level_1 = 0;

    // Mark literals from conflict clause
    for &lit in conflict_clause {
        let var_idx = lit.variable().index();
        let var_level = level[var_idx];

        if var_level == 1 {
            seen[var_idx] = true;
            count_at_level_1 += 1;
        }
    }

    if count_at_level_1 == 0 {
        return None;
    }

    // Walk backward through trail to find 1UIP
    let mut uip: Option<Literal> = None;

    for &lit in trail.iter().rev() {
        let var_idx = lit.variable().index();

        if !seen[var_idx] {
            continue;
        }

        seen[var_idx] = false;
        count_at_level_1 -= 1;

        if count_at_level_1 == 0 {
            // This is the 1UIP
            uip = Some(lit);
            break;
        }

        // Expand the reason clause
        if let Some(reason_ref) = reason[var_idx] {
            let clause_idx = reason_ref.0 as usize;
            if clause_idx < clauses.len() && !clauses.header(clause_idx).is_empty() {
                let reason_lits = clauses.literals(clause_idx);
                for &reason_lit in reason_lits {
                    let reason_var_idx = reason_lit.variable().index();
                    if reason_var_idx == var_idx {
                        continue;
                    }
                    let reason_level = level[reason_var_idx];
                    if reason_level == 1 && !seen[reason_var_idx] {
                        seen[reason_var_idx] = true;
                        count_at_level_1 += 1;
                    }
                }
            }
        }
    }

    // Return the negation of the UIP (the forced literal)
    uip.map(|l| l.negated())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prober_new() {
        let prober = Prober::new(10);
        assert_eq!(prober.num_vars, 10);
        assert!(prober.probes.is_empty());
        assert_eq!(prober.stats.rounds, 0);
    }

    #[test]
    fn test_count_binary_occurrences() {
        let mut prober = Prober::new(3);
        let assignment: Vec<Option<bool>> = vec![None, None, None];

        // Create clauses: (x0 v x1), (¬x0 v x2)
        let mut clauses = ClauseDB::new();
        clauses.add(
            &[
                Literal::positive(Variable(0)),
                Literal::positive(Variable(1)),
            ],
            false,
        );
        clauses.add(
            &[
                Literal::negative(Variable(0)),
                Literal::positive(Variable(2)),
            ],
            false,
        );

        prober.count_binary_occurrences(&clauses, &assignment);

        // x0 appears once positive, once negative
        assert_eq!(prober.bin_occs[Literal::positive(Variable(0)).index()], 1);
        assert_eq!(prober.bin_occs[Literal::negative(Variable(0)).index()], 1);
        // x1 appears once positive
        assert_eq!(prober.bin_occs[Literal::positive(Variable(1)).index()], 1);
        assert_eq!(prober.bin_occs[Literal::negative(Variable(1)).index()], 0);
        // x2 appears once positive
        assert_eq!(prober.bin_occs[Literal::positive(Variable(2)).index()], 1);
        assert_eq!(prober.bin_occs[Literal::negative(Variable(2)).index()], 0);
    }

    #[test]
    fn test_generate_probes_root_detection() {
        let mut prober = Prober::new(3);
        let assignment: Vec<Option<bool>> = vec![None, None, None];

        // Create binary clauses where x1 is a root (appears only negated)
        // (x0 v ¬x1), (x2 v ¬x1)
        let mut clauses = ClauseDB::new();
        clauses.add(
            &[
                Literal::positive(Variable(0)),
                Literal::negative(Variable(1)),
            ],
            false,
        );
        clauses.add(
            &[
                Literal::positive(Variable(2)),
                Literal::negative(Variable(1)),
            ],
            false,
        );

        prober.generate_probes(&clauses, &assignment, 0);

        // ¬x1 is a root (appears twice, x1 never appears positive)
        // x0 appears once positive, never negative - should be probe
        // x2 appears once positive, never negative - should be probe
        assert!(!prober.probes.is_empty());

        // Verify we have some probes
        let mut found_neg_x1 = false;
        for probe in &prober.probes {
            if *probe == Literal::negative(Variable(1)) {
                found_neg_x1 = true;
            }
        }
        assert!(found_neg_x1, "¬x1 should be a probe candidate");
    }

    #[test]
    fn test_probe_stats() {
        let mut prober = Prober::new(5);

        prober.record_probed();
        prober.record_probed();
        prober.record_failed();
        prober.record_round();

        assert_eq!(prober.stats.probed, 2);
        assert_eq!(prober.stats.failed, 1);
        assert_eq!(prober.stats.units_derived, 1);
        assert_eq!(prober.stats.rounds, 1);
    }

    #[test]
    fn test_mark_probed() {
        let mut prober = Prober::new(3);
        let lit = Literal::positive(Variable(1));

        assert!(prober.propfixed[lit.index()] < 0);

        prober.mark_probed(lit, 5);
        assert_eq!(prober.propfixed[lit.index()], 5);

        // Should skip probing if current_fixed hasn't changed
        let assignment: Vec<Option<bool>> = vec![None, None, None];
        let mut clauses = ClauseDB::new();
        clauses.add(
            &[
                Literal::negative(Variable(0)),
                Literal::positive(Variable(1)),
            ],
            false,
        );

        prober.generate_probes(&clauses, &assignment, 5);
        // lit should not be in probes because propfixed[lit] >= current_fixed
        assert!(
            !prober.probes.contains(&lit),
            "Already probed literal should be skipped"
        );
    }
}
