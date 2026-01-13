//! Vivification - clause strengthening via unit propagation
//!
//! Vivification is an inprocessing technique that strengthens clauses by testing
//! if literals can be removed. For each clause C = (l1 ∨ l2 ∨ ... ∨ ln):
//!
//! 1. For each literal l_i in order:
//!    - Temporarily assume ¬l_i
//!    - Propagate unit clauses
//!    - If conflict or another literal in C becomes unit, the clause can be shortened
//!
//! This is based on the technique described in:
//! - Piette, Hamadi, Saïs: "Vivification" (SAT 2008)
//! - CaDiCaL implementation (src/vivify.cpp)
//!
//! For DRAT proof correctness:
//! - When a clause is strengthened, we first add the new clause, then delete the old one
//! - The new clause must be a DRAT derivation of the old (which it always is since it's a subset)

use crate::clause_db::ClauseDB;
use crate::literal::Literal;
use crate::watched::ClauseRef;

/// Result of attempting to vivify a single clause
#[derive(Debug, Clone)]
pub struct VivifyResult {
    /// The strengthened clause (may be same as original if no strengthening)
    pub strengthened: Vec<Literal>,
    /// Whether the clause was actually strengthened
    pub was_strengthened: bool,
    /// Whether vivification proved the clause is satisfied (can be deleted)
    pub is_satisfied: bool,
}

/// Statistics for vivification
#[derive(Debug, Default, Clone)]
pub struct VivifyStats {
    /// Number of clauses examined
    pub clauses_examined: u64,
    /// Number of clauses strengthened
    pub clauses_strengthened: u64,
    /// Total literals removed
    pub literals_removed: u64,
    /// Number of clauses found to be satisfied
    pub clauses_satisfied: u64,
}

/// Internal state for vivification propagation
pub struct VivifyState {
    /// Saved assignment values
    saved_assignment: Vec<Option<bool>>,
    /// Saved levels
    saved_level: Vec<u32>,
    /// Temporary trail for vivification
    temp_trail: Vec<Literal>,
    /// Marks for which variables are assigned in vivification
    temp_assigned: Vec<bool>,
}

impl VivifyState {
    /// Create a new vivification state for n variables
    pub fn new(num_vars: usize) -> Self {
        VivifyState {
            saved_assignment: vec![None; num_vars],
            saved_level: vec![0; num_vars],
            temp_trail: Vec::with_capacity(num_vars),
            temp_assigned: vec![false; num_vars],
        }
    }

    /// Ensure the vivification state can track `num_vars` variables.
    pub fn ensure_num_vars(&mut self, num_vars: usize) {
        if self.saved_assignment.len() < num_vars {
            self.saved_assignment.resize(num_vars, None);
        }
        if self.saved_level.len() < num_vars {
            self.saved_level.resize(num_vars, 0);
        }
        if self.temp_assigned.len() < num_vars {
            self.temp_assigned.resize(num_vars, false);
        }
    }

    /// Save the current solver state for variables in the clause
    pub fn save_state(&mut self, assignment: &[Option<bool>], level: &[u32], clause: &[Literal]) {
        for lit in clause {
            let var_idx = lit.variable().index();
            self.saved_assignment[var_idx] = assignment[var_idx];
            self.saved_level[var_idx] = level[var_idx];
        }
    }

    /// Clear temporary state
    pub fn clear(&mut self) {
        for &lit in &self.temp_trail {
            self.temp_assigned[lit.variable().index()] = false;
        }
        self.temp_trail.clear();
    }

    /// Restore the saved state to the solver
    pub fn restore_state(
        &mut self,
        assignment: &mut [Option<bool>],
        level: &mut [u32],
        clause: &[Literal],
    ) {
        // First restore clause variables
        for lit in clause {
            let var_idx = lit.variable().index();
            assignment[var_idx] = self.saved_assignment[var_idx];
            level[var_idx] = self.saved_level[var_idx];
        }
        // Then restore temp trail variables
        for &lit in &self.temp_trail {
            let var_idx = lit.variable().index();
            assignment[var_idx] = self.saved_assignment[var_idx];
            level[var_idx] = self.saved_level[var_idx];
            self.temp_assigned[var_idx] = false;
        }
        self.temp_trail.clear();
    }
}

/// Vivification engine that works with a solver
pub struct Vivifier {
    state: VivifyState,
    stats: VivifyStats,
    /// Maximum number of clauses to vivify per call
    max_clauses_per_call: usize,
    /// Minimum clause length to consider for vivification
    min_clause_len: usize,
}

impl Vivifier {
    /// Create a new vivifier for n variables
    pub fn new(num_vars: usize) -> Self {
        Vivifier {
            state: VivifyState::new(num_vars),
            stats: VivifyStats::default(),
            max_clauses_per_call: 1000,
            min_clause_len: 3, // Don't vivify binary clauses (too important)
        }
    }

    /// Ensure the vivifier can track `num_vars` variables.
    pub fn ensure_num_vars(&mut self, num_vars: usize) {
        self.state.ensure_num_vars(num_vars);
    }

    /// Get vivification statistics
    pub fn stats(&self) -> &VivifyStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = VivifyStats::default();
    }

    /// Set maximum clauses to process per call
    pub fn set_max_clauses(&mut self, max: usize) {
        self.max_clauses_per_call = max;
    }

    /// Try to vivify a single clause, returning the result
    ///
    /// This is a "pure" operation that doesn't modify the solver state permanently.
    /// The caller is responsible for updating the clause database based on the result.
    ///
    /// # Arguments
    /// * `clause_lits` - The literals in the clause to vivify
    /// * `assignment` - Current variable assignments (will be temporarily modified)
    /// * `level` - Variable decision levels (will be temporarily modified)
    /// * `clauses` - The clause database (for propagation)
    /// * `watches` - Watch lists (for propagation)
    #[allow(clippy::too_many_arguments)]
    pub fn vivify_clause(
        &mut self,
        clause_lits: &[Literal],
        assignment: &mut [Option<bool>],
        level: &mut [u32],
        clauses: &ClauseDB,
        watches: &crate::watched::WatchedLists,
        reason: &[Option<ClauseRef>],
        clause_ref: ClauseRef,
    ) -> VivifyResult {
        self.stats.clauses_examined += 1;

        // Save state for literals in this clause
        self.state.save_state(assignment, level, clause_lits);

        let mut simplified: Vec<Literal> = Vec::with_capacity(clause_lits.len());
        for &lit in clause_lits {
            match Self::lit_value(assignment, lit) {
                Some(true) => {
                    self.state.restore_state(assignment, level, clause_lits);
                    self.stats.clauses_satisfied += 1;
                    return VivifyResult {
                        strengthened: clause_lits.to_vec(),
                        was_strengthened: false,
                        is_satisfied: true,
                    };
                }
                Some(false) => {}
                None => simplified.push(lit),
            }
        }

        // All literals false at level 0 means the formula is already inconsistent.
        if simplified.is_empty() {
            self.state.restore_state(assignment, level, clause_lits);
            return VivifyResult {
                strengthened: clause_lits.to_vec(),
                was_strengthened: false,
                is_satisfied: false,
            };
        }

        let simplified_only = simplified.len() < clause_lits.len();

        // Don't vivify (via propagation) short clauses, but do allow simplification.
        if simplified.len() < self.min_clause_len {
            self.state.restore_state(assignment, level, clause_lits);
            return VivifyResult {
                strengthened: simplified,
                was_strengthened: simplified_only,
                is_satisfied: false,
            };
        }

        let mut assumptions: Vec<Literal> = Vec::with_capacity(simplified.len());
        let mut strengthened: Option<Vec<Literal>> = None;

        for (i, &lit) in simplified.iter().enumerate() {
            match Self::lit_value(assignment, lit) {
                Some(true) => {
                    let mut out = assumptions.clone();
                    if !out.contains(&lit) {
                        out.push(lit);
                    }
                    strengthened = Some(out);
                    break;
                }
                Some(false) => continue,
                None => {}
            }

            // Assume the negation of this literal.
            let neg_lit = lit.negated();
            let var_idx = lit.variable().index();
            assignment[var_idx] = Some(neg_lit.is_positive());
            level[var_idx] = 0;
            self.state.temp_trail.push(neg_lit);
            self.state.temp_assigned[var_idx] = true;
            assumptions.push(lit);

            let prop_result = self.propagate_vivify(
                assignment,
                level,
                clauses,
                watches,
                reason,
                clause_ref,
                &simplified[i + 1..],
            );

            match prop_result {
                VivifyPropResult::Conflict => {
                    strengthened = Some(assumptions.clone());
                    break;
                }
                VivifyPropResult::ImpliedTrue(implied_lit) => {
                    let mut out = assumptions.clone();
                    if !out.contains(&implied_lit) {
                        out.push(implied_lit);
                    }
                    strengthened = Some(out);
                    break;
                }
                VivifyPropResult::NoChange => {}
            }
        }

        self.state.restore_state(assignment, level, clause_lits);

        if let Some(strengthened) = strengthened {
            if strengthened.len() < clause_lits.len() {
                self.stats.clauses_strengthened += 1;
                self.stats.literals_removed += (clause_lits.len() - strengthened.len()) as u64;
                return VivifyResult {
                    strengthened,
                    was_strengthened: true,
                    is_satisfied: false,
                };
            }
        }

        if simplified_only {
            self.stats.clauses_strengthened += 1;
            self.stats.literals_removed += (clause_lits.len() - simplified.len()) as u64;
            return VivifyResult {
                strengthened: simplified,
                was_strengthened: true,
                is_satisfied: false,
            };
        }

        VivifyResult {
            strengthened: clause_lits.to_vec(),
            was_strengthened: false,
            is_satisfied: false,
        }
    }

    /// Get the value of a literal under current assignment
    #[inline]
    fn lit_value(assignment: &[Option<bool>], lit: Literal) -> Option<bool> {
        assignment[lit.variable().index()].map(|v| if lit.is_positive() { v } else { !v })
    }

    /// Propagate during vivification, checking for conflicts or implications
    /// that would allow us to strengthen the clause.
    #[allow(clippy::too_many_arguments)]
    fn propagate_vivify(
        &mut self,
        assignment: &mut [Option<bool>],
        level: &mut [u32],
        clauses: &ClauseDB,
        watches: &crate::watched::WatchedLists,
        _reason: &[Option<ClauseRef>],
        vivify_clause: ClauseRef,
        remaining_lits: &[Literal],
    ) -> VivifyPropResult {
        let mut qhead = 0;

        while qhead < self.state.temp_trail.len() {
            let p = self.state.temp_trail[qhead];
            qhead += 1;

            let false_lit = p.negated();
            let watch_list = watches.get_watches(false_lit);

            for watcher in watch_list {
                if watcher.clause_ref() == vivify_clause {
                    continue;
                }
                let clause_idx = watcher.clause_ref().0 as usize;
                let header = clauses.header(clause_idx);

                if header.is_empty() {
                    continue; // Deleted clause
                }

                // Quick check with blocker
                if Self::lit_value(assignment, watcher.blocker()) == Some(true) {
                    continue;
                }

                // Check if clause is satisfied or find unit
                let mut num_false = 0;
                let mut unit_lit: Option<Literal> = None;
                let mut is_satisfied = false;

                for &lit in clauses.literals(clause_idx) {
                    match Self::lit_value(assignment, lit) {
                        Some(true) => {
                            is_satisfied = true;
                            break;
                        }
                        Some(false) => num_false += 1,
                        None => {
                            if unit_lit.is_some() {
                                // More than one unassigned - not unit
                                unit_lit = None;
                                break;
                            }
                            unit_lit = Some(lit);
                        }
                    }
                }

                if is_satisfied {
                    continue;
                }

                if num_false == header.len() {
                    // Conflict!
                    return VivifyPropResult::Conflict;
                }

                if let Some(unit) = unit_lit {
                    // Check if this unit is in the remaining literals of the vivify clause
                    if remaining_lits.contains(&unit) {
                        return VivifyPropResult::ImpliedTrue(unit);
                    }

                    // Propagate the unit
                    let var_idx = unit.variable().index();
                    let val = unit.is_positive();
                    match assignment[var_idx] {
                        Some(existing) if existing != val => return VivifyPropResult::Conflict,
                        Some(_) => continue,
                        None => {}
                    }

                    // Save original state before modifying (for restore)
                    self.state.saved_assignment[var_idx] = assignment[var_idx];
                    self.state.saved_level[var_idx] = level[var_idx];

                    assignment[var_idx] = Some(val);
                    level[var_idx] = 0;
                    self.state.temp_trail.push(unit);
                    self.state.temp_assigned[var_idx] = true;
                }
            }
        }

        VivifyPropResult::NoChange
    }
}

/// Result of vivification propagation
#[derive(Debug)]
enum VivifyPropResult {
    /// Found a conflict - the assumed literal's negation is implied
    Conflict,
    /// A literal in the remaining clause became true via propagation
    ImpliedTrue(Literal),
    /// No useful implication found
    NoChange,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::literal::Variable;
    use crate::watched::WatchedLists;

    fn make_clause(lits: &[(u32, bool)]) -> Vec<Literal> {
        lits.iter()
            .map(|&(var, positive)| {
                if positive {
                    Literal::positive(Variable(var))
                } else {
                    Literal::negative(Variable(var))
                }
            })
            .collect()
    }

    #[test]
    fn test_vivify_state_save_restore() {
        let mut state = VivifyState::new(5);
        let mut assignment = vec![Some(true), Some(false), None, None, None];
        let mut level = vec![0, 1, 0, 0, 0];

        let clause = make_clause(&[(0, true), (1, false), (2, true)]);
        state.save_state(&assignment, &level, &clause);

        // Modify
        assignment[0] = Some(false);
        assignment[2] = Some(true);
        level[0] = 5;

        // Restore
        state.restore_state(&mut assignment, &mut level, &clause);

        assert_eq!(assignment[0], Some(true));
        assert_eq!(assignment[1], Some(false));
        assert_eq!(assignment[2], None);
        assert_eq!(level[0], 0);
    }

    #[test]
    fn test_vivifier_stats() {
        let mut vivifier = Vivifier::new(10);
        assert_eq!(vivifier.stats().clauses_examined, 0);

        vivifier.stats.clauses_examined = 5;
        vivifier.stats.clauses_strengthened = 2;

        assert_eq!(vivifier.stats().clauses_examined, 5);
        assert_eq!(vivifier.stats().clauses_strengthened, 2);

        vivifier.reset_stats();
        assert_eq!(vivifier.stats().clauses_examined, 0);
    }

    #[test]
    fn test_vivify_short_clause_unchanged() {
        let mut vivifier = Vivifier::new(5);
        let mut assignment = vec![None; 5];
        let mut level = vec![0; 5];
        let clauses = ClauseDB::new();
        let watches = WatchedLists::new(5);
        let reason = vec![None; 5];

        // Binary clause should not be vivified
        let clause = make_clause(&[(0, true), (1, true)]);
        let result = vivifier.vivify_clause(
            &clause,
            &mut assignment,
            &mut level,
            &clauses,
            &watches,
            &reason,
            ClauseRef(0),
        );

        assert!(!result.was_strengthened);
        assert_eq!(result.strengthened.len(), 2);
    }

    #[test]
    fn test_vivify_satisfied_clause() {
        let mut vivifier = Vivifier::new(5);
        let mut assignment = vec![None; 5];
        let mut level = vec![0; 5];
        let clauses = ClauseDB::new();
        let watches = WatchedLists::new(5);
        let reason = vec![None; 5];

        // Make x0 true, so clause (x0 OR x1 OR x2) is satisfied
        assignment[0] = Some(true);

        let clause = make_clause(&[(0, true), (1, true), (2, true)]);
        let result = vivifier.vivify_clause(
            &clause,
            &mut assignment,
            &mut level,
            &clauses,
            &watches,
            &reason,
            ClauseRef(0),
        );

        assert!(result.is_satisfied);
        assert!(!result.was_strengthened);
    }

    #[test]
    fn test_vivify_remove_false_literal() {
        let mut vivifier = Vivifier::new(5);
        let mut assignment = vec![None; 5];
        let mut level = vec![0; 5];
        let clauses = ClauseDB::new();
        let watches = WatchedLists::new(5);
        let reason = vec![None; 5];

        // Make x0 false, so it can be removed from (x0 OR x1 OR x2)
        assignment[0] = Some(false);

        let clause = make_clause(&[(0, true), (1, true), (2, true)]);
        let result = vivifier.vivify_clause(
            &clause,
            &mut assignment,
            &mut level,
            &clauses,
            &watches,
            &reason,
            ClauseRef(0),
        );

        assert!(result.was_strengthened);
        assert_eq!(result.strengthened.len(), 2);
        assert!(!result
            .strengthened
            .contains(&Literal::positive(Variable(0))));
    }
}
