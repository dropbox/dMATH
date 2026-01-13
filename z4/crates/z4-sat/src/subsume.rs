//! Subsumption and self-subsumption checking
//!
//! Implements clause subsumption for inprocessing:
//! - Forward subsumption: A new clause subsumes (removes) existing clauses
//! - Backward subsumption: An existing clause subsumes a new clause
//! - Self-subsumption (strengthening): Remove redundant literals from clauses
//!
//! A clause C subsumes clause D if C ⊆ D (all literals in C are in D).
//! The subsumed clause D can be removed because C logically implies D.
//!
//! Self-subsumption: If C = L ∨ A and D = ¬L ∨ A ∨ B, then D can be
//! strengthened to A ∨ B by resolving C and D on L.

use crate::clause_db::ClauseDB;
use crate::literal::Literal;

/// 64-bit signature for quick subsumption filtering
///
/// The signature is a bitwise OR of (1 << (var % 64)) for each variable in the clause.
/// If sig(C) & sig(D) != sig(C), then C cannot subsume D.
pub type ClauseSignature = u64;

/// Compute the signature of a clause
#[inline]
pub fn clause_signature(literals: &[Literal]) -> ClauseSignature {
    let mut sig: ClauseSignature = 0;
    for lit in literals {
        let var_idx = lit.variable().index();
        sig |= 1u64 << (var_idx % 64);
    }
    sig
}

/// Occurrence list: for each literal, list of clauses containing that literal
#[derive(Debug, Clone)]
pub struct OccurrenceList {
    /// For each literal code (2*var + polarity), the list of clause indices
    occ: Vec<Vec<usize>>,
}

impl OccurrenceList {
    /// Create a new occurrence list for n variables
    pub fn new(num_vars: usize) -> Self {
        OccurrenceList {
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

    /// Get the literal code for indexing
    #[inline]
    fn lit_code(lit: Literal) -> usize {
        lit.index()
    }

    /// Add a clause to occurrence lists
    pub fn add_clause(&mut self, clause_idx: usize, literals: &[Literal]) {
        for &lit in literals {
            let code = Self::lit_code(lit);
            if code < self.occ.len() {
                self.occ[code].push(clause_idx);
            }
        }
    }

    /// Remove a clause from occurrence lists
    pub fn remove_clause(&mut self, clause_idx: usize, literals: &[Literal]) {
        for &lit in literals {
            let code = Self::lit_code(lit);
            if code < self.occ.len() {
                if let Some(pos) = self.occ[code].iter().position(|&idx| idx == clause_idx) {
                    self.occ[code].swap_remove(pos);
                }
            }
        }
    }

    /// Get clauses containing a literal
    pub fn get(&self, lit: Literal) -> &[usize] {
        let code = Self::lit_code(lit);
        if code < self.occ.len() {
            &self.occ[code]
        } else {
            &[]
        }
    }

    /// Clear all occurrence lists
    pub fn clear(&mut self) {
        for list in &mut self.occ {
            list.clear();
        }
    }
}

/// Statistics for subsumption operations
#[derive(Debug, Clone, Default)]
pub struct SubsumeStats {
    /// Number of clauses removed by forward subsumption
    pub forward_subsumed: u64,
    /// Number of clauses removed by backward subsumption
    pub backward_subsumed: u64,
    /// Number of literals removed by self-subsumption
    pub strengthened_literals: u64,
    /// Number of clauses strengthened by self-subsumption
    pub strengthened_clauses: u64,
    /// Number of subsumption checks performed
    pub checks: u64,
}

/// Result of subsumption checking
#[derive(Debug, Clone)]
pub struct SubsumeResult {
    /// Indices of clauses that were subsumed (should be deleted)
    pub subsumed: Vec<usize>,
    /// Clauses that were strengthened: (clause_idx, new_literals)
    pub strengthened: Vec<(usize, Vec<Literal>)>,
}

impl Default for SubsumeResult {
    fn default() -> Self {
        Self::new()
    }
}

impl SubsumeResult {
    /// Create a new empty result
    pub fn new() -> Self {
        SubsumeResult {
            subsumed: Vec::new(),
            strengthened: Vec::new(),
        }
    }
}

/// Subsumption checking engine
pub struct Subsumer {
    /// Occurrence lists for each literal
    occ: OccurrenceList,
    /// Signatures for each clause
    signatures: Vec<ClauseSignature>,
    /// Statistics
    stats: SubsumeStats,
    /// Temporary mark array for subsumption checking
    marks: Vec<bool>,
}

impl Subsumer {
    /// Create a new subsumption checker for n variables
    pub fn new(num_vars: usize) -> Self {
        Subsumer {
            occ: OccurrenceList::new(num_vars),
            signatures: Vec::new(),
            stats: SubsumeStats::default(),
            marks: vec![false; num_vars],
        }
    }

    /// Ensure internal buffers can handle `num_vars` variables.
    pub fn ensure_num_vars(&mut self, num_vars: usize) {
        self.occ.ensure_num_vars(num_vars);
        if self.marks.len() < num_vars {
            self.marks.resize(num_vars, false);
        }
    }

    /// Initialize/rebuild occurrence lists from clause database
    pub fn rebuild(&mut self, clauses: &ClauseDB) {
        self.occ.clear();
        self.signatures.clear();
        self.signatures.reserve(clauses.len());

        for idx in clauses.indices() {
            if clauses.header(idx).is_empty() {
                self.signatures.push(0);
                continue;
            }
            let lits = clauses.literals(idx);
            self.occ.add_clause(idx, lits);
            self.signatures.push(clause_signature(lits));
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &SubsumeStats {
        &self.stats
    }

    /// Check if clause C subsumes clause D (all literals in C are in D)
    ///
    /// Precondition: sig(C) & sig(D) == sig(C) (signature check passed)
    fn clause_subsumes(&mut self, c: &[Literal], d: &[Literal]) -> bool {
        // C cannot subsume D if C is longer
        if c.len() > d.len() {
            return false;
        }

        self.stats.checks += 1;

        // Mark all literals in D
        for &lit in d {
            let var_idx = lit.variable().index();
            if var_idx < self.marks.len() {
                // Use the sign bit to distinguish polarity
                // We store the literal code to check exact match
                self.marks[var_idx] = true;
            }
        }

        // Check if all literals in C are in D
        let mut all_found = true;
        for &lit_c in c {
            let var_idx = lit_c.variable().index();
            if var_idx >= self.marks.len() || !self.marks[var_idx] {
                all_found = false;
                break;
            }
            // Need to verify polarity match
            if !d.contains(&lit_c) {
                all_found = false;
                break;
            }
        }

        // Clear marks
        for &lit in d {
            let var_idx = lit.variable().index();
            if var_idx < self.marks.len() {
                self.marks[var_idx] = false;
            }
        }

        all_found
    }

    /// Check for self-subsumption: C = L ∨ A, D = ¬L ∨ A ∨ B
    /// Returns Some(literal_to_remove) if D can be strengthened
    ///
    /// Self-subsumption finds a literal L in C such that:
    /// - ¬L is in D
    /// - All other literals of C are in D
    ///
    /// Then ¬L can be removed from D (D is strengthened to A ∨ B)
    fn find_self_subsumption(&mut self, c: &[Literal], d: &[Literal]) -> Option<Literal> {
        // C\{L} must be subset of D for some L in C
        // This means |C| <= |D| + 1
        if c.len() > d.len() + 1 {
            return None;
        }

        self.stats.checks += 1;

        // Try each literal in C as the pivot
        for (pivot_idx, &pivot) in c.iter().enumerate() {
            let neg_pivot = pivot.negated();

            // Check if ¬pivot is in D
            if !d.contains(&neg_pivot) {
                continue;
            }

            // Check if all other literals in C are in D
            let mut all_others_found = true;
            for (i, &lit) in c.iter().enumerate() {
                if i == pivot_idx {
                    continue;
                }
                if !d.contains(&lit) {
                    all_others_found = false;
                    break;
                }
            }

            if all_others_found {
                // D can be strengthened by removing ¬pivot
                return Some(neg_pivot);
            }
        }

        None
    }

    /// Forward subsumption: check if clause C subsumes any existing clause
    ///
    /// Returns indices of clauses subsumed by C
    pub fn forward_subsumption(
        &mut self,
        c_idx: usize,
        c_lits: &[Literal],
        c_sig: ClauseSignature,
        clauses: &ClauseDB,
    ) -> Vec<usize> {
        let mut subsumed = Vec::new();

        if c_lits.is_empty() {
            return subsumed;
        }

        // Find the literal in C with the smallest occurrence list
        let min_lit = c_lits
            .iter()
            .min_by_key(|&&lit| self.occ.get(lit).len())
            .copied()
            .unwrap();

        // Collect candidates first to avoid borrow issues
        let candidates: Vec<(usize, Vec<Literal>)> = self
            .occ
            .get(min_lit)
            .iter()
            .filter_map(|&d_idx| {
                if d_idx == c_idx || d_idx >= clauses.len() {
                    return None;
                }
                let d_header = clauses.header(d_idx);
                if d_header.is_empty() {
                    return None;
                }

                // Quick signature check
                let d_sig = if d_idx < self.signatures.len() {
                    self.signatures[d_idx]
                } else {
                    clause_signature(clauses.literals(d_idx))
                };

                // If sig(C) & sig(D) != sig(C), C cannot subsume D
                if c_sig & d_sig != c_sig {
                    return None;
                }

                Some((d_idx, clauses.literals(d_idx).to_vec()))
            })
            .collect();

        // Check each candidate for subsumption
        for (d_idx, d_lits) in candidates {
            if self.clause_subsumes(c_lits, &d_lits) {
                subsumed.push(d_idx);
                self.stats.forward_subsumed += 1;
            }
        }

        subsumed
    }

    /// Backward subsumption: check if any existing clause subsumes clause C
    ///
    /// Returns the index of a clause that subsumes C, or None
    pub fn backward_subsumption(
        &mut self,
        c_lits: &[Literal],
        c_sig: ClauseSignature,
        clauses: &ClauseDB,
    ) -> Option<usize> {
        if c_lits.is_empty() {
            return None;
        }

        // Find the literal in C with the smallest occurrence list
        let min_lit = c_lits
            .iter()
            .min_by_key(|&&lit| self.occ.get(lit).len())
            .copied()
            .unwrap();

        // Collect candidates first to avoid borrow issues
        let candidates: Vec<(usize, Vec<Literal>, ClauseSignature)> = self
            .occ
            .get(min_lit)
            .iter()
            .filter_map(|&d_idx| {
                if d_idx >= clauses.len() {
                    return None;
                }
                let d_header = clauses.header(d_idx);
                if d_header.is_empty() {
                    return None;
                }

                let d_lits = clauses.literals(d_idx);

                // Backward: D subsumes C, so D must be shorter or equal
                if d_lits.len() > c_lits.len() {
                    return None;
                }

                // Quick signature check: if sig(D) & sig(C) != sig(D), D cannot subsume C
                let d_sig = if d_idx < self.signatures.len() {
                    self.signatures[d_idx]
                } else {
                    clause_signature(d_lits)
                };

                if d_sig & c_sig != d_sig {
                    return None;
                }

                Some((d_idx, d_lits.to_vec(), d_sig))
            })
            .collect();

        // Check each candidate for subsumption
        for (d_idx, d_lits, _d_sig) in candidates {
            if self.clause_subsumes(&d_lits, c_lits) {
                self.stats.backward_subsumed += 1;
                return Some(d_idx);
            }
        }

        None
    }

    /// Self-subsumption (strengthening): find clauses that can be strengthened by C
    ///
    /// Returns list of (clause_idx, literal_to_remove) pairs
    pub fn self_subsumption(
        &mut self,
        c_lits: &[Literal],
        _c_sig: ClauseSignature,
        clauses: &ClauseDB,
    ) -> Vec<(usize, Literal)> {
        let mut strengthening = Vec::new();

        if c_lits.len() < 2 {
            return strengthening;
        }

        // Collect all candidates first to avoid borrow issues
        let mut candidates: Vec<(usize, Vec<Literal>, Literal)> = Vec::new();

        // For each literal L in C, look for clauses containing ¬L
        for &pivot in c_lits {
            let neg_pivot = pivot.negated();

            for &d_idx in self.occ.get(neg_pivot) {
                if d_idx >= clauses.len() {
                    continue;
                }
                let d_header = clauses.header(d_idx);
                if d_header.is_empty() || d_header.len() < 2 {
                    continue;
                }

                // Quick check: D must be longer than C for strengthening to be useful
                // (Otherwise it would be forward subsumption, not self-subsumption)
                if d_header.len() <= c_lits.len() {
                    continue;
                }

                candidates.push((d_idx, clauses.literals(d_idx).to_vec(), neg_pivot));
            }
        }

        // Now check each candidate for self-subsumption
        for (d_idx, d_lits, neg_pivot) in candidates {
            // D must contain ¬pivot and all other literals of C
            if let Some(lit_to_remove) = self.find_self_subsumption(c_lits, &d_lits) {
                if lit_to_remove == neg_pivot {
                    strengthening.push((d_idx, lit_to_remove));
                    self.stats.strengthened_clauses += 1;
                    self.stats.strengthened_literals += 1;
                }
            }
        }

        strengthening
    }

    /// Run subsumption as inprocessing on a subset of clauses
    ///
    /// This is the main entry point for subsumption inprocessing.
    /// It should be called at decision level 0 (e.g., after a restart).
    ///
    /// Returns a SubsumeResult with clauses to delete and strengthen.
    pub fn run_subsumption(
        &mut self,
        clauses: &ClauseDB,
        start_idx: usize,
        max_clauses: usize,
    ) -> SubsumeResult {
        let mut result = SubsumeResult::new();

        // Process clauses starting from start_idx
        let end_idx = (start_idx + max_clauses).min(clauses.len());

        for c_idx in start_idx..end_idx {
            let c_header = clauses.header(c_idx);
            if c_header.is_empty() {
                continue;
            }

            let c_lits = clauses.literals(c_idx).to_vec();
            let c_sig = if c_idx < self.signatures.len() {
                self.signatures[c_idx]
            } else {
                clause_signature(&c_lits)
            };

            // Forward subsumption: does C subsume any other clause?
            let subsumed = self.forward_subsumption(c_idx, &c_lits, c_sig, clauses);
            result.subsumed.extend(subsumed);

            // Self-subsumption: can C strengthen any other clause?
            let strengthening = self.self_subsumption(&c_lits, c_sig, clauses);
            for (d_idx, lit_to_remove) in strengthening {
                // Compute the strengthened clause
                let new_lits: Vec<Literal> = clauses
                    .literals(d_idx)
                    .iter()
                    .filter(|&&l| l != lit_to_remove)
                    .copied()
                    .collect();
                result.strengthened.push((d_idx, new_lits));
            }
        }

        result
    }

    /// Update occurrence lists after a clause is added
    pub fn on_clause_added(&mut self, clause_idx: usize, literals: &[Literal]) {
        while self.signatures.len() <= clause_idx {
            self.signatures.push(0);
        }
        self.signatures[clause_idx] = clause_signature(literals);
        self.occ.add_clause(clause_idx, literals);
    }

    /// Update occurrence lists after a clause is deleted
    pub fn on_clause_deleted(&mut self, clause_idx: usize, literals: &[Literal]) {
        self.occ.remove_clause(clause_idx, literals);
        if clause_idx < self.signatures.len() {
            self.signatures[clause_idx] = 0;
        }
    }

    /// Update occurrence lists after a clause is strengthened
    pub fn on_clause_strengthened(
        &mut self,
        clause_idx: usize,
        old_literals: &[Literal],
        new_literals: &[Literal],
    ) {
        self.occ.remove_clause(clause_idx, old_literals);
        self.occ.add_clause(clause_idx, new_literals);
        if clause_idx < self.signatures.len() {
            self.signatures[clause_idx] = clause_signature(new_literals);
        }
    }

    /// Run subsumption on learned clauses (does not delete or strengthen non-learned clauses).
    pub fn run_subsumption_learned(
        &mut self,
        clauses: &ClauseDB,
        max_clauses: usize,
    ) -> SubsumeResult {
        let mut result = SubsumeResult::new();

        let mut processed = 0usize;
        for c_idx in clauses.indices() {
            if processed >= max_clauses {
                break;
            }
            let c_header = clauses.header(c_idx);
            if c_header.is_empty() || !c_header.is_learned() {
                continue;
            }

            processed += 1;

            let c_lits = clauses.literals(c_idx).to_vec();
            let c_sig = if c_idx < self.signatures.len() {
                self.signatures[c_idx]
            } else {
                clause_signature(&c_lits)
            };

            let subsumed = self.forward_subsumption(c_idx, &c_lits, c_sig, clauses);
            result.subsumed.extend(
                subsumed
                    .into_iter()
                    .filter(|&idx| clauses.header(idx).is_learned()),
            );

            let strengthening = self.self_subsumption(&c_lits, c_sig, clauses);
            for (d_idx, lit_to_remove) in strengthening {
                let d_header = clauses.header(d_idx);
                if d_header.is_empty() || !d_header.is_learned() {
                    continue;
                }
                let new_lits: Vec<Literal> = clauses
                    .literals(d_idx)
                    .iter()
                    .filter(|&&l| l != lit_to_remove)
                    .copied()
                    .collect();
                result.strengthened.push((d_idx, new_lits));
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::literal::Variable;

    fn lit(var: u32, positive: bool) -> Literal {
        if positive {
            Literal::positive(Variable(var))
        } else {
            Literal::negative(Variable(var))
        }
    }

    #[test]
    fn test_clause_signature() {
        // Signature should be consistent
        let lits1 = vec![lit(0, true), lit(1, false), lit(2, true)];
        let sig1 = clause_signature(&lits1);

        // Same literals, different order -> same signature
        let lits2 = vec![lit(2, true), lit(0, true), lit(1, false)];
        let sig2 = clause_signature(&lits2);
        assert_eq!(sig1, sig2);

        // Different polarity -> same signature (signature is per variable)
        let lits3 = vec![lit(0, false), lit(1, true), lit(2, false)];
        let sig3 = clause_signature(&lits3);
        assert_eq!(sig1, sig3);
    }

    #[test]
    fn test_signature_filtering() {
        // Clause {0, 1} cannot subsume {2, 3} (different variables)
        let c = vec![lit(0, true), lit(1, true)];
        let d = vec![lit(2, true), lit(3, true)];
        let sig_c = clause_signature(&c);
        let sig_d = clause_signature(&d);

        // sig_c & sig_d should NOT equal sig_c
        assert_ne!(sig_c & sig_d, sig_c);
    }

    #[test]
    fn test_subsumption_basic() {
        let mut subsumer = Subsumer::new(5);

        // C = {0, 1} subsumes D = {0, 1, 2}
        let c = vec![lit(0, true), lit(1, true)];
        let d = vec![lit(0, true), lit(1, true), lit(2, true)];

        assert!(subsumer.clause_subsumes(&c, &d));
        assert!(!subsumer.clause_subsumes(&d, &c)); // D does not subsume C
    }

    #[test]
    fn test_subsumption_polarity() {
        let mut subsumer = Subsumer::new(5);

        // C = {0, ¬1} does NOT subsume D = {0, 1, 2} (polarity mismatch on 1)
        let c = vec![lit(0, true), lit(1, false)];
        let d = vec![lit(0, true), lit(1, true), lit(2, true)];

        assert!(!subsumer.clause_subsumes(&c, &d));
    }

    #[test]
    fn test_self_subsumption_basic() {
        let mut subsumer = Subsumer::new(5);

        // C = {0, 1}, D = {¬0, 1, 2}
        // C can strengthen D by removing ¬0 (result: {1, 2})
        let c = vec![lit(0, true), lit(1, true)];
        let d = vec![lit(0, false), lit(1, true), lit(2, true)];

        let result = subsumer.find_self_subsumption(&c, &d);
        assert_eq!(result, Some(lit(0, false)));
    }

    #[test]
    fn test_self_subsumption_no_match() {
        let mut subsumer = Subsumer::new(5);

        // C = {0, 1}, D = {2, 3}
        // No self-subsumption possible
        let c = vec![lit(0, true), lit(1, true)];
        let d = vec![lit(2, true), lit(3, true)];

        let result = subsumer.find_self_subsumption(&c, &d);
        assert_eq!(result, None);
    }

    #[test]
    fn test_occurrence_list() {
        let mut occ = OccurrenceList::new(5);

        let clause1 = vec![lit(0, true), lit(1, false)];
        let clause2 = vec![lit(0, true), lit(2, true)];

        occ.add_clause(0, &clause1);
        occ.add_clause(1, &clause2);

        // lit(0, true) appears in both clauses
        let occ_lit0 = occ.get(lit(0, true));
        assert!(occ_lit0.contains(&0));
        assert!(occ_lit0.contains(&1));

        // lit(1, false) appears only in clause 0
        let occ_lit1 = occ.get(lit(1, false));
        assert!(occ_lit1.contains(&0));
        assert!(!occ_lit1.contains(&1));

        // Remove clause 0
        occ.remove_clause(0, &clause1);
        let occ_lit0_after = occ.get(lit(0, true));
        assert!(!occ_lit0_after.contains(&0));
        assert!(occ_lit0_after.contains(&1));
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn test_forward_subsumption() {
        let mut subsumer = Subsumer::new(5);

        // Clauses: C0 = {0, 1}, C1 = {0, 1, 2}, C2 = {2, 3}
        let mut clauses = ClauseDB::new();
        clauses.add(&[lit(0, true), lit(1, true)], false);
        clauses.add(&[lit(0, true), lit(1, true), lit(2, true)], false);
        clauses.add(&[lit(2, true), lit(3, true)], false);

        subsumer.rebuild(&clauses);

        // C0 should subsume C1
        let c0_lits = clauses.literals(0).to_vec();
        let c0_sig = clause_signature(&c0_lits);
        let subsumed = subsumer.forward_subsumption(0, &c0_lits, c0_sig, &clauses);

        assert!(subsumed.contains(&1)); // C1 is subsumed
        assert!(!subsumed.contains(&2)); // C2 is not subsumed
    }

    #[test]
    fn test_backward_subsumption() {
        let mut subsumer = Subsumer::new(5);

        // Clauses: C0 = {0, 1}, C1 = {2, 3}
        let mut clauses = ClauseDB::new();
        clauses.add(&[lit(0, true), lit(1, true)], false);
        clauses.add(&[lit(2, true), lit(3, true)], false);

        subsumer.rebuild(&clauses);

        // New clause {0, 1, 2} should be subsumed by C0
        let new_lits = vec![lit(0, true), lit(1, true), lit(2, true)];
        let new_sig = clause_signature(&new_lits);
        let result = subsumer.backward_subsumption(&new_lits, new_sig, &clauses);

        assert_eq!(result, Some(0)); // C0 subsumes the new clause
    }

    #[test]
    fn test_run_subsumption() {
        let mut subsumer = Subsumer::new(5);

        // Clauses setup for subsumption testing
        let mut clauses = ClauseDB::new();
        // C0: {0, 1} - will subsume C2
        clauses.add(&[lit(0, true), lit(1, true)], false);
        // C1: {2, 3} - independent
        clauses.add(&[lit(2, true), lit(3, true)], false);
        // C2: {0, 1, 4} - will be subsumed by C0
        clauses.add(&[lit(0, true), lit(1, true), lit(4, true)], false);
        // C3: {¬0, 1, 4} - can be strengthened by C0 to {1, 4}
        clauses.add(&[lit(0, false), lit(1, true), lit(4, true)], false);

        subsumer.rebuild(&clauses);

        let result = subsumer.run_subsumption(&clauses, 0, clauses.len());

        // C2 should be subsumed by C0
        assert!(result.subsumed.contains(&2));

        // C3 should be strengthened (remove ¬0)
        let strengthened_c3 = result.strengthened.iter().find(|(idx, _)| *idx == 3);
        assert!(strengthened_c3.is_some());
        let (_, new_lits) = strengthened_c3.unwrap();
        assert_eq!(new_lits.len(), 2);
        assert!(new_lits.contains(&lit(1, true)));
        assert!(new_lits.contains(&lit(4, true)));
        assert!(!new_lits.contains(&lit(0, false)));
    }
}
