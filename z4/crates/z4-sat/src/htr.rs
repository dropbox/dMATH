//! Hyper-Ternary Resolution (HTR)
//!
//! Implements hyper-ternary resolution as an inprocessing technique.
//! This technique resolves pairs of ternary (3-literal) clauses to produce
//! new binary or ternary resolvents.
//!
//! The key insight is that resolving two ternary clauses can produce:
//! - A binary clause (2 literals) - highly valuable, can delete both antecedents
//! - A ternary clause (3 literals) - useful for further propagation
//! - A quaternary clause (4 literals) - discarded (not beneficial)
//! - A tautology - discarded
//!
//! Binary resolvents are particularly valuable because they:
//! 1. Strengthen unit propagation
//! 2. Subsume both antecedent clauses (which can be deleted)
//!
//! Reference: Heule, Järvisalo, Lonsing, "Clause Elimination for SAT and QSAT",
//! Journal of Artificial Intelligence Research, 2015.
//! Also: Biere, "Lingeling and Friends at the SAT Competition 2011".

use crate::clause_db::ClauseDB;
use crate::literal::{Literal, Variable};

// Use deterministic hasher for Kani (avoids CCRandomGenerateBytes foreign function call)
#[cfg(kani)]
use std::hash::{BuildHasherDefault, DefaultHasher};
#[cfg(kani)]
type DetHashSet<T> = std::collections::HashSet<T, BuildHasherDefault<DefaultHasher>>;

#[cfg(not(kani))]
use std::collections::HashSet as DetHashSet;

/// Maximum number of occurrences for a literal to be considered
/// (to avoid combinatorial explosion)
const MAX_OCCURRENCES: usize = 100;

/// Statistics for HTR operations
#[derive(Debug, Clone, Default)]
pub struct HTRStats {
    /// Number of hyper-ternary resolution rounds
    pub rounds: u64,
    /// Number of ternary resolvents added
    pub ternary_resolvents: u64,
    /// Number of binary resolvents added
    pub binary_resolvents: u64,
    /// Number of clause pairs checked
    pub pairs_checked: u64,
    /// Number of tautological resolvents skipped
    pub tautologies_skipped: u64,
    /// Number of duplicate resolvents skipped
    pub duplicates_skipped: u64,
    /// Number of quaternary (or larger) resolvents skipped
    pub too_large_skipped: u64,
}

/// Result of a single hyper-ternary resolution
#[derive(Debug, Clone)]
pub struct HTRResult {
    /// New resolvents to add (binary or ternary)
    pub resolvents: Vec<Vec<Literal>>,
    /// Clause indices to delete (antecedents subsumed by binary resolvents)
    pub to_delete: Vec<usize>,
}

/// Occurrence list for HTR - tracks which clauses contain each literal
#[derive(Debug, Clone)]
pub struct HTROccList {
    /// For each literal index, list of clause indices containing that literal
    occ: Vec<Vec<usize>>,
}

impl HTROccList {
    /// Create a new occurrence list for n variables
    pub fn new(num_vars: usize) -> Self {
        HTROccList {
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
}

/// Hyper-Ternary Resolution engine
pub struct HTR {
    /// Occurrence lists
    occ: HTROccList,
    /// Statistics
    stats: HTRStats,
    /// Number of variables
    num_vars: usize,
    /// Temporary mark array for tautology/duplicate checking
    marks: Vec<i8>, // 0 = unmarked, 1 = positive, -1 = negative
    /// Set of existing binary clauses (for duplicate detection)
    /// Stored as (min_lit_idx, max_lit_idx) pairs
    existing_binary: DetHashSet<(u32, u32)>,
    /// Set of existing ternary clauses (for duplicate detection)
    /// Stored as sorted (lit1, lit2, lit3) triples
    existing_ternary: DetHashSet<(u32, u32, u32)>,
}

impl HTR {
    /// Create a new HTR engine for n variables
    pub fn new(num_vars: usize) -> Self {
        HTR {
            occ: HTROccList::new(num_vars),
            stats: HTRStats::default(),
            num_vars,
            marks: vec![0; num_vars],
            existing_binary: DetHashSet::default(),
            existing_ternary: DetHashSet::default(),
        }
    }

    /// Ensure internal buffers can handle `num_vars` variables.
    pub fn ensure_num_vars(&mut self, num_vars: usize) {
        if self.num_vars >= num_vars {
            return;
        }
        self.num_vars = num_vars;
        self.occ.ensure_num_vars(num_vars);
        if self.marks.len() < num_vars {
            self.marks.resize(num_vars, 0);
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &HTRStats {
        &self.stats
    }

    /// Normalize a binary clause to a canonical form for hashing
    fn normalize_binary(a: Literal, b: Literal) -> (u32, u32) {
        let a_raw = a.0;
        let b_raw = b.0;
        if a_raw <= b_raw {
            (a_raw, b_raw)
        } else {
            (b_raw, a_raw)
        }
    }

    /// Normalize a ternary clause to a canonical form for hashing
    fn normalize_ternary(a: Literal, b: Literal, c: Literal) -> (u32, u32, u32) {
        let mut lits = [a.0, b.0, c.0];
        lits.sort_unstable();
        (lits[0], lits[1], lits[2])
    }

    /// Initialize/rebuild occurrence lists and existing clause sets
    pub fn rebuild(&mut self, clauses: &ClauseDB) {
        self.occ.clear();
        self.existing_binary.clear();
        self.existing_ternary.clear();

        for idx in clauses.indices() {
            let header = clauses.header(idx);
            if header.is_empty() {
                continue;
            }

            let len = header.len();
            let lits = clauses.literals(idx);

            // Only track binary and ternary clauses
            if len == 2 {
                let key = Self::normalize_binary(lits[0], lits[1]);
                self.existing_binary.insert(key);
                self.occ.add_clause(idx, lits);
            } else if len == 3 {
                let key = Self::normalize_ternary(lits[0], lits[1], lits[2]);
                self.existing_ternary.insert(key);
                self.occ.add_clause(idx, lits);
            }
        }
    }

    /// Check if a binary clause already exists (or is subsumed)
    fn binary_exists(&self, a: Literal, b: Literal) -> bool {
        let key = Self::normalize_binary(a, b);
        self.existing_binary.contains(&key)
    }

    /// Check if a ternary clause already exists (or is subsumed by a binary)
    fn ternary_exists(&self, a: Literal, b: Literal, c: Literal) -> bool {
        // Check if subsumed by any binary clause
        if self.binary_exists(a, b) || self.binary_exists(a, c) || self.binary_exists(b, c) {
            return true;
        }

        // Check if exact ternary exists
        let key = Self::normalize_ternary(a, b, c);
        self.existing_ternary.contains(&key)
    }

    /// Try to resolve two ternary clauses on a pivot literal
    ///
    /// Returns Some(resolvent) if successful, None if:
    /// - Resolvent is tautological
    /// - Resolvent has more than 3 literals
    /// - Resolvent already exists
    fn try_resolve(
        &mut self,
        clause_c: &[Literal],
        clause_d: &[Literal],
        pivot: Literal,
    ) -> Option<Vec<Literal>> {
        debug_assert_eq!(clause_c.len(), 3);
        debug_assert_eq!(clause_d.len(), 3);

        self.stats.pairs_checked += 1;

        // Clear marks
        for &lit in clause_c {
            self.marks[lit.variable().index()] = 0;
        }
        for &lit in clause_d {
            self.marks[lit.variable().index()] = 0;
        }

        let mut resolvent = Vec::with_capacity(4);

        // Add literals from first clause (except pivot)
        for &lit in clause_c {
            if lit == pivot {
                continue;
            }
            let var_idx = lit.variable().index();
            let sign: i8 = if lit.is_positive() { 1 } else { -1 };
            self.marks[var_idx] = sign;
            resolvent.push(lit);
        }

        // Add literals from second clause (except negated pivot)
        let neg_pivot = pivot.negated();
        for &lit in clause_d {
            if lit == neg_pivot {
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
                // New literal
                self.marks[var_idx] = sign;
                resolvent.push(lit);
            }
            // If marks[var_idx] == sign, literal is duplicate, skip
        }

        // Check resolvent size
        if resolvent.len() > 3 {
            self.stats.too_large_skipped += 1;
            return None;
        }

        // Check for duplicates
        match resolvent.len() {
            2 => {
                if self.binary_exists(resolvent[0], resolvent[1]) {
                    self.stats.duplicates_skipped += 1;
                    return None;
                }
            }
            3 => {
                if self.ternary_exists(resolvent[0], resolvent[1], resolvent[2]) {
                    self.stats.duplicates_skipped += 1;
                    return None;
                }
            }
            _ => {
                // Unit clause (len == 1) or empty - should not happen with ternary inputs
                // but we'll accept it if it does
            }
        }

        Some(resolvent)
    }

    /// Run hyper-ternary resolution on a single pivot variable
    ///
    /// Returns (resolvents, antecedents_to_delete)
    fn resolve_on_pivot(
        &mut self,
        var: Variable,
        clauses: &ClauseDB,
        assignment: &[Option<bool>],
    ) -> (Vec<Vec<Literal>>, Vec<(usize, usize)>) {
        let pos_lit = Literal::positive(var);
        let neg_lit = Literal::negative(var);

        // Get ternary clauses containing positive and negative occurrences
        let pos_clauses: Vec<usize> = self.occ.get(pos_lit).to_vec();
        let neg_clauses: Vec<usize> = self.occ.get(neg_lit).to_vec();

        let mut resolvents = Vec::new();
        let mut antecedents_to_delete = Vec::new();

        // Try all pairs
        for &pos_idx in &pos_clauses {
            let pos_header = clauses.header(pos_idx);
            if pos_idx >= clauses.len() || pos_header.is_empty() {
                continue;
            }
            if pos_header.len() != 3 {
                continue;
            }

            // Skip if any literal in clause is assigned
            let pos_lits = clauses.literals(pos_idx);
            let mut assigned = false;
            for &lit in pos_lits {
                if assignment
                    .get(lit.variable().index())
                    .copied()
                    .flatten()
                    .is_some()
                {
                    assigned = true;
                    break;
                }
            }
            if assigned {
                continue;
            }

            for &neg_idx in &neg_clauses {
                let neg_header = clauses.header(neg_idx);
                if neg_idx >= clauses.len() || neg_header.is_empty() {
                    continue;
                }
                if neg_header.len() != 3 {
                    continue;
                }

                // Skip if any literal in clause is assigned
                let neg_lits = clauses.literals(neg_idx);
                let mut assigned = false;
                for &lit in neg_lits {
                    if assignment
                        .get(lit.variable().index())
                        .copied()
                        .flatten()
                        .is_some()
                    {
                        assigned = true;
                        break;
                    }
                }
                if assigned {
                    continue;
                }

                // Try to resolve
                if let Some(resolvent) = self.try_resolve(pos_lits, neg_lits, pos_lit) {
                    let is_binary = resolvent.len() == 2;

                    // Add to existing set to avoid duplicate detection
                    match resolvent.len() {
                        2 => {
                            let key = Self::normalize_binary(resolvent[0], resolvent[1]);
                            self.existing_binary.insert(key);
                            self.stats.binary_resolvents += 1;

                            // Binary resolvents subsume both antecedents
                            antecedents_to_delete.push((pos_idx, neg_idx));
                        }
                        3 => {
                            let key =
                                Self::normalize_ternary(resolvent[0], resolvent[1], resolvent[2]);
                            self.existing_ternary.insert(key);
                            self.stats.ternary_resolvents += 1;
                        }
                        _ => {}
                    }

                    resolvents.push(resolvent);

                    // If we derived a binary clause, stop processing this positive clause
                    // (it will be deleted anyway)
                    if is_binary {
                        break;
                    }
                }
            }
        }

        (resolvents, antecedents_to_delete)
    }

    /// Run hyper-ternary resolution as inprocessing
    ///
    /// Iterates over variables and tries to resolve ternary clauses.
    /// Should be called at decision level 0 (e.g., after a restart).
    ///
    /// Returns an HTRResult with new resolvents and clauses to delete.
    pub fn run(
        &mut self,
        clauses: &ClauseDB,
        assignment: &[Option<bool>],
        max_resolvents: usize,
    ) -> HTRResult {
        self.stats.rounds += 1;

        let mut all_resolvents = Vec::new();
        let mut to_delete = Vec::new();
        let mut deleted_set: DetHashSet<usize> = DetHashSet::default();

        // Iterate over all variables
        for var_idx in 0..self.num_vars {
            if all_resolvents.len() >= max_resolvents {
                break;
            }

            let var = Variable(var_idx as u32);

            // Skip if variable is assigned
            if var_idx < assignment.len() && assignment[var_idx].is_some() {
                continue;
            }

            // Check occurrence counts to avoid expensive operations
            let pos_lit = Literal::positive(var);
            let neg_lit = Literal::negative(var);
            let pos_count = self.occ.count(pos_lit);
            let neg_count = self.occ.count(neg_lit);

            if pos_count == 0 || neg_count == 0 {
                continue;
            }
            if pos_count > MAX_OCCURRENCES || neg_count > MAX_OCCURRENCES {
                continue;
            }

            // Run resolution on this pivot
            let (resolvents, antecedents) = self.resolve_on_pivot(var, clauses, assignment);

            // Collect resolvents
            for resolvent in resolvents {
                if all_resolvents.len() >= max_resolvents {
                    break;
                }
                all_resolvents.push(resolvent);
            }

            // Collect unique antecedents to delete
            for (c_idx, d_idx) in antecedents {
                if !deleted_set.contains(&c_idx) {
                    deleted_set.insert(c_idx);
                    to_delete.push(c_idx);
                }
                if !deleted_set.contains(&d_idx) {
                    deleted_set.insert(d_idx);
                    to_delete.push(d_idx);
                }
            }
        }

        HTRResult {
            resolvents: all_resolvents,
            to_delete,
        }
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
    fn test_htr_occurrence_list() {
        let mut occ = HTROccList::new(5);

        let clause1 = vec![lit(0, true), lit(1, false), lit(2, true)];
        let clause2 = vec![lit(0, true), lit(3, true), lit(4, false)];

        occ.add_clause(0, &clause1);
        occ.add_clause(1, &clause2);

        // lit(0, true) appears in both clauses
        assert_eq!(occ.count(lit(0, true)), 2);
        assert!(occ.get(lit(0, true)).contains(&0));
        assert!(occ.get(lit(0, true)).contains(&1));
    }

    #[test]
    fn test_htr_basic_resolve() {
        let mut htr = HTR::new(10);

        // C1 = {x0, x1, x2}, C2 = {~x0, x3, x4}
        // Resolvent on x0 should be {x1, x2, x3, x4} - too large, rejected
        let c1 = vec![lit(0, true), lit(1, true), lit(2, true)];
        let c2 = vec![lit(0, false), lit(3, true), lit(4, true)];

        let result = htr.try_resolve(&c1, &c2, lit(0, true));
        assert!(result.is_none()); // 4 literals - too large
        assert_eq!(htr.stats.too_large_skipped, 1);
    }

    #[test]
    fn test_htr_binary_resolvent() {
        let mut htr = HTR::new(10);

        // C1 = {x0, x1, x2}, C2 = {~x0, x1, x2}
        // Resolvent on x0 should be {x1, x2} - binary!
        let c1 = vec![lit(0, true), lit(1, true), lit(2, true)];
        let c2 = vec![lit(0, false), lit(1, true), lit(2, true)];

        let result = htr.try_resolve(&c1, &c2, lit(0, true));
        assert!(result.is_some());

        let resolvent = result.unwrap();
        assert_eq!(resolvent.len(), 2);
        assert!(resolvent.contains(&lit(1, true)));
        assert!(resolvent.contains(&lit(2, true)));
    }

    #[test]
    fn test_htr_ternary_resolvent() {
        let mut htr = HTR::new(10);

        // C1 = {x0, x1, x2}, C2 = {~x0, x1, x3}
        // Resolvent on x0 should be {x1, x2, x3} - ternary
        let c1 = vec![lit(0, true), lit(1, true), lit(2, true)];
        let c2 = vec![lit(0, false), lit(1, true), lit(3, true)];

        let result = htr.try_resolve(&c1, &c2, lit(0, true));
        assert!(result.is_some());

        let resolvent = result.unwrap();
        assert_eq!(resolvent.len(), 3);
        assert!(resolvent.contains(&lit(1, true)));
        assert!(resolvent.contains(&lit(2, true)));
        assert!(resolvent.contains(&lit(3, true)));
    }

    #[test]
    fn test_htr_tautology_detection() {
        let mut htr = HTR::new(10);

        // C1 = {x0, x1, x2}, C2 = {~x0, ~x1, x3}
        // Resolvent would be {x1, ~x1, x2, x3} - tautology
        let c1 = vec![lit(0, true), lit(1, true), lit(2, true)];
        let c2 = vec![lit(0, false), lit(1, false), lit(3, true)];

        let result = htr.try_resolve(&c1, &c2, lit(0, true));
        assert!(result.is_none());
        assert_eq!(htr.stats.tautologies_skipped, 1);
    }

    #[test]
    fn test_htr_duplicate_detection() {
        let mut htr = HTR::new(10);

        // Pre-populate with existing binary clause {x1, x2}
        htr.existing_binary
            .insert(HTR::normalize_binary(lit(1, true), lit(2, true)));

        // Try to derive the same binary
        let c1 = vec![lit(0, true), lit(1, true), lit(2, true)];
        let c2 = vec![lit(0, false), lit(1, true), lit(2, true)];

        let result = htr.try_resolve(&c1, &c2, lit(0, true));
        assert!(result.is_none());
        assert_eq!(htr.stats.duplicates_skipped, 1);
    }

    #[test]
    fn test_htr_run_integration() {
        let mut htr = HTR::new(10);

        // Create clauses that allow hyper-ternary resolution
        // C0 = {x0, x1, x2}, C1 = {~x0, x1, x3}
        // These should produce ternary resolvent {x1, x2, x3}
        let mut clauses = ClauseDB::new();
        clauses.add(&[lit(0, true), lit(1, true), lit(2, true)], false);
        clauses.add(&[lit(0, false), lit(1, true), lit(3, true)], false);

        htr.rebuild(&clauses);

        let assignment = vec![None; 10];
        let result = htr.run(&clauses, &assignment, 100);

        // Should produce at least one resolvent
        assert!(!result.resolvents.is_empty());
        assert!(result.resolvents.iter().any(|r| r.len() == 3));
    }

    #[test]
    fn test_htr_binary_deletion() {
        let mut htr = HTR::new(10);

        // Create clauses where binary resolvent can be derived
        // C0 = {x0, x1, x2}, C1 = {~x0, x1, x2}
        // Binary resolvent {x1, x2} subsumes both
        let mut clauses = ClauseDB::new();
        clauses.add(&[lit(0, true), lit(1, true), lit(2, true)], false);
        clauses.add(&[lit(0, false), lit(1, true), lit(2, true)], false);

        htr.rebuild(&clauses);

        let assignment = vec![None; 10];
        let result = htr.run(&clauses, &assignment, 100);

        // Should have binary resolvent and both antecedents marked for deletion
        assert!(result.resolvents.iter().any(|r| r.len() == 2));
        assert_eq!(result.to_delete.len(), 2);
        assert!(result.to_delete.contains(&0));
        assert!(result.to_delete.contains(&1));
    }

    #[test]
    fn test_htr_skips_assigned() {
        let mut htr = HTR::new(10);

        let mut clauses = ClauseDB::new();
        clauses.add(&[lit(0, true), lit(1, true), lit(2, true)], false);
        clauses.add(&[lit(0, false), lit(1, true), lit(3, true)], false);

        htr.rebuild(&clauses);

        // Variable 0 is assigned - should skip resolution
        let mut assignment = vec![None; 10];
        assignment[0] = Some(true);

        let result = htr.run(&clauses, &assignment, 100);

        // Should produce no resolvents since pivot variable is assigned
        assert!(result.resolvents.is_empty());
    }

    #[test]
    fn test_htr_stats() {
        let mut htr = HTR::new(10);

        // Create clauses for resolution
        let mut clauses = ClauseDB::new();
        clauses.add(&[lit(0, true), lit(1, true), lit(2, true)], false);
        clauses.add(&[lit(0, false), lit(1, true), lit(2, true)], false);
        clauses.add(&[lit(0, false), lit(1, true), lit(3, true)], false);

        htr.rebuild(&clauses);

        let assignment = vec![None; 10];
        let _ = htr.run(&clauses, &assignment, 100);

        let stats = htr.stats();
        assert_eq!(stats.rounds, 1);
        assert!(stats.pairs_checked > 0);
        // Should have at least one binary resolvent from C0 and C1
        assert!(stats.binary_resolvents >= 1 || stats.ternary_resolvents >= 1);
    }

    #[test]
    fn test_htr_subsumption_by_binary() {
        let mut htr = HTR::new(10);

        // Pre-populate with binary clause that subsumes potential ternary resolvent
        // Binary {x1, x2} subsumes any ternary {x1, x2, ?}
        htr.existing_binary
            .insert(HTR::normalize_binary(lit(1, true), lit(2, true)));

        // C1 = {x0, x1, x3}, C2 = {~x0, x2, x3}
        // Resolvent would be {x1, x2, x3} - subsumed by {x1, x2}
        // Actually wait, {x1, x2} doesn't contain x3, so it doesn't subsume
        // Let me fix the test

        // Actually the subsumption check is: if any binary subset exists, skip
        // So {x1, x2} being a subset of {x1, x2, x3} means the ternary is subsumed

        // Resolvent would be {x1, x2, x3}
        // Check if subsumed by existing binary {x1, x2}
        assert!(htr.ternary_exists(lit(1, true), lit(2, true), lit(3, true)));
    }

    #[test]
    fn test_normalize_binary() {
        let a = lit(3, true);
        let b = lit(1, false);

        let (x, y) = HTR::normalize_binary(a, b);
        let (x2, y2) = HTR::normalize_binary(b, a);

        // Should be the same regardless of order
        assert_eq!((x, y), (x2, y2));
        assert!(x <= y);
    }

    #[test]
    fn test_normalize_ternary() {
        let a = lit(5, true);
        let b = lit(2, false);
        let c = lit(7, true);

        let t1 = HTR::normalize_ternary(a, b, c);
        let t2 = HTR::normalize_ternary(c, a, b);
        let t3 = HTR::normalize_ternary(b, c, a);

        // Should all be the same
        assert_eq!(t1, t2);
        assert_eq!(t2, t3);
        assert!(t1.0 <= t1.1 && t1.1 <= t1.2);
    }
}
