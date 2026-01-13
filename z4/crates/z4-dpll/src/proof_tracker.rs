//! Proof Tracking for SMT Solving
//!
//! This module provides proof generation during SMT solving. When enabled,
//! the solver collects proof steps that can be exported in Alethe format
//! for independent verification using tools like carcara.
//!
//! ## tRust Integration
//!
//! Proof certificates are critical for tRust (verified Rust compiler):
//! - Verification conditions are checked by Z4
//! - Proof certificates allow independent verification of results
//! - Unsat proofs are especially important for proving safety properties
//!
//! ## Alethe Proof Format
//!
//! The proof tracker generates steps compatible with the Alethe format:
//! - `assume`: Input assertions from the SMT-LIB problem
//! - `step`: Inference steps with rules, premises, and conclusion clauses
//! - Theory lemmas are recorded with appropriate theory-specific rules
//!
//! ## Usage
//!
//! ```ignore
//! use z4_dpll::{Executor, ProofTracker};
//!
//! let mut exec = Executor::new();
//! exec.set_produce_proofs(true);
//!
//! // After check-sat returns unsat...
//! if let Some(proof) = exec.get_proof() {
//!     let alethe = z4_proof::export_alethe(&proof, exec.terms());
//!     println!("{}", alethe);
//! }
//! ```

use hashbrown::HashMap;
use z4_core::proof::{AletheRule, Proof, ProofId};
use z4_core::TermId;

/// Proof tracker for collecting SMT proof steps during solving
///
/// The tracker collects:
/// 1. Assumptions from input assertions
/// 2. Theory lemmas from theory solver conflicts
/// 3. Resolution steps from SAT solver (when available)
#[derive(Debug, Default)]
pub struct ProofTracker {
    /// The accumulated proof
    proof: Proof,
    /// Mapping from assertion term IDs to their proof step IDs
    assumption_map: HashMap<TermId, ProofId>,
    /// Mapping from theory lemma clauses (as sorted term IDs) to proof step IDs
    lemma_map: HashMap<Vec<TermId>, ProofId>,
    /// Whether proof tracking is enabled
    enabled: bool,
    /// Theory name for the current solving context
    theory_name: String,
}

impl ProofTracker {
    /// Create a new proof tracker (disabled by default)
    #[must_use]
    pub fn new() -> Self {
        ProofTracker {
            proof: Proof::new(),
            assumption_map: HashMap::new(),
            lemma_map: HashMap::new(),
            enabled: false,
            theory_name: "UNKNOWN".to_string(),
        }
    }

    /// Enable proof tracking
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable proof tracking
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Check if proof tracking is enabled
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Set the theory name for subsequent theory lemmas
    pub fn set_theory(&mut self, theory: impl Into<String>) {
        self.theory_name = theory.into();
    }

    /// Reset the tracker for a new solving session
    pub fn reset(&mut self) {
        self.proof = Proof::new();
        self.assumption_map.clear();
        self.lemma_map.clear();
        // Keep enabled state and theory name
    }

    /// Record an assumption (input assertion)
    ///
    /// Returns the proof step ID for this assumption, or None if tracking is disabled.
    pub fn add_assumption(&mut self, term: TermId, name: Option<String>) -> Option<ProofId> {
        if !self.enabled {
            return None;
        }

        // Check if we already have this assumption
        if let Some(&id) = self.assumption_map.get(&term) {
            return Some(id);
        }

        let id = self.proof.add_assume(term, name);
        self.assumption_map.insert(term, id);
        Some(id)
    }

    /// Record a theory lemma (conflict clause from theory solver)
    ///
    /// The clause is the disjunction of literals that the theory solver derived.
    /// Returns the proof step ID for this lemma, or None if tracking is disabled.
    pub fn add_theory_lemma(&mut self, clause: Vec<TermId>) -> Option<ProofId> {
        if !self.enabled {
            return None;
        }

        // Create a canonical key for deduplication
        let mut key = clause.clone();
        key.sort_by_key(|t| t.0);

        // Check if we already have this lemma
        if let Some(&id) = self.lemma_map.get(&key) {
            return Some(id);
        }

        let id = self.proof.add_theory_lemma(&self.theory_name, clause);
        self.lemma_map.insert(key, id);
        Some(id)
    }

    /// Record a resolution step
    ///
    /// Resolution combines two clauses using a pivot literal:
    /// C1 = (A ∨ p) and C2 = (B ∨ ¬p) resolve to C = (A ∨ B)
    pub fn add_resolution(
        &mut self,
        pivot: TermId,
        clause1: ProofId,
        clause2: ProofId,
    ) -> Option<ProofId> {
        if !self.enabled {
            return None;
        }

        Some(self.proof.add_resolution(pivot, clause1, clause2))
    }

    /// Record a generic proof step with a rule
    pub fn add_step(
        &mut self,
        rule: AletheRule,
        clause: Vec<TermId>,
        premises: Vec<ProofId>,
        args: Vec<TermId>,
    ) -> Option<ProofId> {
        if !self.enabled {
            return None;
        }

        Some(self.proof.add_rule_step(rule, clause, premises, args))
    }

    /// Record a reflexivity step: t = t
    pub fn add_refl(&mut self, eq_term: TermId) -> Option<ProofId> {
        self.add_step(AletheRule::Refl, vec![eq_term], vec![], vec![])
    }

    /// Record a transitivity step: a = b, b = c => a = c
    pub fn add_trans(
        &mut self,
        conclusion: TermId,
        premise1: ProofId,
        premise2: ProofId,
    ) -> Option<ProofId> {
        self.add_step(
            AletheRule::Trans,
            vec![conclusion],
            vec![premise1, premise2],
            vec![],
        )
    }

    /// Record a congruence step: f(a) = f(b) if a = b
    pub fn add_cong(&mut self, conclusion: TermId, premises: Vec<ProofId>) -> Option<ProofId> {
        self.add_step(AletheRule::Cong, vec![conclusion], premises, vec![])
    }

    /// Record a linear arithmetic lemma
    pub fn add_la_lemma(&mut self, clause: Vec<TermId>) -> Option<ProofId> {
        self.add_step(AletheRule::LaGeneric, clause, vec![], vec![])
    }

    /// Record an integer arithmetic lemma
    pub fn add_lia_lemma(&mut self, clause: Vec<TermId>) -> Option<ProofId> {
        self.add_step(AletheRule::LiaGeneric, clause, vec![], vec![])
    }

    /// Record a trust step (unverified, placeholder)
    pub fn add_trust(&mut self, clause: Vec<TermId>) -> Option<ProofId> {
        self.add_step(AletheRule::Trust, clause, vec![], vec![])
    }

    /// Record the final contradiction (empty clause)
    pub fn add_contradiction(&mut self, premises: Vec<ProofId>) -> Option<ProofId> {
        self.add_step(AletheRule::Resolution, vec![], premises, vec![])
    }

    /// Get the accumulated proof
    #[must_use]
    pub fn get_proof(&self) -> &Proof {
        &self.proof
    }

    /// Take ownership of the accumulated proof
    pub fn take_proof(&mut self) -> Proof {
        std::mem::take(&mut self.proof)
    }

    /// Get the number of proof steps
    #[must_use]
    pub fn num_steps(&self) -> usize {
        self.proof.len()
    }

    /// Get the proof step ID for an assumption term
    #[must_use]
    pub fn assumption_id(&self, term: TermId) -> Option<ProofId> {
        self.assumption_map.get(&term).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracker_disabled_by_default() {
        let tracker = ProofTracker::new();
        assert!(!tracker.is_enabled());
    }

    #[test]
    fn test_enable_disable() {
        let mut tracker = ProofTracker::new();
        tracker.enable();
        assert!(tracker.is_enabled());
        tracker.disable();
        assert!(!tracker.is_enabled());
    }

    #[test]
    fn test_assumption_when_disabled() {
        let mut tracker = ProofTracker::new();
        let result = tracker.add_assumption(TermId(1), None);
        assert!(result.is_none());
    }

    #[test]
    fn test_assumption_when_enabled() {
        let mut tracker = ProofTracker::new();
        tracker.enable();

        let id = tracker.add_assumption(TermId(1), Some("h1".to_string()));
        assert!(id.is_some());
        assert_eq!(tracker.num_steps(), 1);

        // Adding same assumption returns same ID
        let id2 = tracker.add_assumption(TermId(1), None);
        assert_eq!(id, id2);
        assert_eq!(tracker.num_steps(), 1);
    }

    #[test]
    fn test_theory_lemma() {
        let mut tracker = ProofTracker::new();
        tracker.enable();
        tracker.set_theory("EUF");

        let clause = vec![TermId(1), TermId(2)];
        let id = tracker.add_theory_lemma(clause.clone());
        assert!(id.is_some());
        assert_eq!(tracker.num_steps(), 1);

        // Adding same lemma (even reordered) returns same ID
        let clause2 = vec![TermId(2), TermId(1)];
        let id2 = tracker.add_theory_lemma(clause2);
        assert_eq!(id, id2);
        assert_eq!(tracker.num_steps(), 1);
    }

    #[test]
    fn test_resolution() {
        let mut tracker = ProofTracker::new();
        tracker.enable();

        let h1 = tracker
            .add_assumption(TermId(1), Some("h1".to_string()))
            .unwrap();
        let h2 = tracker
            .add_assumption(TermId(2), Some("h2".to_string()))
            .unwrap();

        let res = tracker.add_resolution(TermId(3), h1, h2);
        assert!(res.is_some());
        assert_eq!(tracker.num_steps(), 3);
    }

    #[test]
    fn test_reset() {
        let mut tracker = ProofTracker::new();
        tracker.enable();
        tracker.add_assumption(TermId(1), None);
        assert_eq!(tracker.num_steps(), 1);

        tracker.reset();
        assert_eq!(tracker.num_steps(), 0);
        assert!(tracker.is_enabled()); // Enabled state preserved
    }

    #[test]
    fn test_take_proof() {
        let mut tracker = ProofTracker::new();
        tracker.enable();
        tracker.add_assumption(TermId(1), None);

        let proof = tracker.take_proof();
        assert_eq!(proof.len(), 1);
        assert_eq!(tracker.num_steps(), 0);
    }

    #[test]
    fn test_proof_step_types() {
        let mut tracker = ProofTracker::new();
        tracker.enable();
        tracker.set_theory("LRA");

        // Add various step types
        let h1 = tracker.add_assumption(TermId(1), None).unwrap();
        let _ = tracker.add_refl(TermId(2));
        let _ = tracker.add_la_lemma(vec![TermId(3)]);
        let _ = tracker.add_trust(vec![TermId(4)]);
        let _ = tracker.add_contradiction(vec![h1]);

        assert_eq!(tracker.num_steps(), 5);
    }

    #[test]
    fn test_assumption_lookup() {
        let mut tracker = ProofTracker::new();
        tracker.enable();

        let id = tracker
            .add_assumption(TermId(42), Some("hyp".to_string()))
            .unwrap();
        assert_eq!(tracker.assumption_id(TermId(42)), Some(id));
        assert_eq!(tracker.assumption_id(TermId(99)), None);
    }
}
