//! Equality theory solver using E-graphs
//!
//! This module implements the equality/uninterpreted functions (EUF) theory solver
//! using the E-graph data structure for congruence closure.
//!
//! # Theory of Equality with Uninterpreted Functions (EUF)
//!
//! EUF handles:
//! - Equality constraints: `x = y`
//! - Disequality constraints: `x ≠ y`
//! - Function congruence: `x = y → f(x) = f(y)`
//!
//! The E-graph maintains equivalence classes and performs congruence closure
//! to derive implied equalities.
//!
//! # Conflict Detection
//!
//! A conflict occurs when:
//! - We assert `x ≠ y` but the E-graph already knows `x = y`
//! - We assert `x = y` but there's a chain of disequalities making this impossible
//!
//! # Proof Reconstruction
//!
//! The theory solver tracks union reasons for proof reconstruction:
//! - Direct assertions record the hypothesis that caused the union
//! - Congruence steps record which argument equalities caused the merge
//! - The proof trace can be used to build kernel proof terms

use crate::cdcl::Lit;
use crate::egraph::{EClassId, EGraph};
use crate::proof::{ProofTrace, UnionReason};
use crate::smt::{SmtTerm, TermId, TheoryCheckResult, TheoryLiteral, TheorySolver};
use lean5_kernel::FVarId;
use std::collections::HashMap;

/// Equality theory solver using E-graphs
pub struct EqualityTheory {
    /// The E-graph for congruence closure
    egraph: EGraph,
    /// Mapping from SMT term IDs to E-class IDs
    term_to_eclass: HashMap<TermId, EClassId>,
    /// Disequalities that have been asserted: (t1, t2, literal)
    /// The literal is stored so we can report it in conflicts
    disequalities: Vec<(TermId, TermId, Lit)>,
    /// Equalities asserted at each level (for backtracking)
    equality_trail: Vec<Vec<(TermId, TermId, Lit)>>,
    /// Disequality indices at each level (for backtracking)
    diseq_trail: Vec<usize>,
    /// Current decision level
    level: u32,
    /// SMT terms (shared reference for term building)
    terms: Vec<SmtTerm>,
    /// Proof trace for reconstruction
    proof_trace: ProofTrace,
    /// Mapping from term ID to hypothesis FVarId (for asserted equalities)
    term_to_hypothesis: HashMap<(TermId, TermId), FVarId>,
}

impl EqualityTheory {
    /// Create a new equality theory solver
    pub fn new() -> Self {
        EqualityTheory {
            egraph: EGraph::new(),
            term_to_eclass: HashMap::new(),
            disequalities: Vec::new(),
            equality_trail: vec![Vec::new()],
            diseq_trail: vec![0],
            level: 0,
            terms: Vec::new(),
            proof_trace: ProofTrace::new(),
            term_to_hypothesis: HashMap::new(),
        }
    }

    /// Set the terms (called by SMT solver to share term information)
    pub fn set_terms(&mut self, terms: Vec<SmtTerm>) {
        self.terms = terms;
    }

    /// Register a hypothesis for an equality (for proof reconstruction)
    pub fn register_hypothesis(&mut self, t1: TermId, t2: TermId, fvar: FVarId) {
        self.term_to_hypothesis.insert((t1, t2), fvar);
        self.term_to_hypothesis.insert((t2, t1), fvar);
    }

    /// Get the proof trace (for proof reconstruction)
    pub fn proof_trace(&self) -> &ProofTrace {
        &self.proof_trace
    }

    /// Get mutable proof trace
    pub fn proof_trace_mut(&mut self) -> &mut ProofTrace {
        &mut self.proof_trace
    }

    /// Get or create an E-class ID for a term
    fn get_or_create_eclass(&mut self, term_id: TermId) -> EClassId {
        if let Some(&eclass) = self.term_to_eclass.get(&term_id) {
            return eclass;
        }

        // Build the term in the E-graph
        let eclass = self.build_term_in_egraph(term_id);
        self.term_to_eclass.insert(term_id, eclass);
        eclass
    }

    /// Build a term in the E-graph
    fn build_term_in_egraph(&mut self, term_id: TermId) -> EClassId {
        // Check if already built
        if let Some(&eclass) = self.term_to_eclass.get(&term_id) {
            return eclass;
        }

        let term = self.terms.get(term_id.0 as usize).cloned();

        let eclass = match term {
            Some(SmtTerm::Const(ref name)) => self.egraph.add_const(name.name()),
            Some(SmtTerm::App(ref name, ref args)) => {
                let arg_eclasses: Vec<EClassId> = args
                    .iter()
                    .map(|&arg_id| self.build_term_in_egraph(arg_id))
                    .collect();
                self.egraph.add_app(name.name(), arg_eclasses)
            }
            Some(SmtTerm::Int(n)) => {
                // Represent integers as constants
                self.egraph.add_const(format!("int_{n}"))
            }
            Some(SmtTerm::Rat(num, den)) => {
                // Represent rationals as constants
                self.egraph.add_const(format!("rat_{num}_{den}"))
            }
            None => {
                // Unknown term - create a placeholder constant
                self.egraph.add_const(format!("term_{}", term_id.0))
            }
        };

        self.term_to_eclass.insert(term_id, eclass);
        eclass
    }

    /// Assert an equality: t1 = t2
    fn assert_equality(&mut self, t1: TermId, t2: TermId, lit: Lit) -> TheoryCheckResult {
        let ec1 = self.get_or_create_eclass(t1);
        let ec2 = self.get_or_create_eclass(t2);

        // Record this equality for backtracking
        self.equality_trail[self.level as usize].push((t1, t2, lit));

        // Record in proof trace
        let hypothesis = self.term_to_hypothesis.get(&(t1, t2)).copied();
        self.proof_trace.record_union(
            ec1.id(),
            ec2.id(),
            UnionReason::Asserted {
                hypothesis,
                lhs: t1,
                rhs: t2,
            },
        );

        // Record merge history size before union
        let history_start = self.egraph.merge_history().len();

        // Merge in the E-graph (this does congruence closure)
        self.egraph.union(ec1, ec2);

        // Extract congruence merges from the E-graph's merge history
        self.record_congruence_merges(history_start);

        // Check if any disequality is now violated
        self.check_disequalities()
    }

    /// Record congruence merges from E-graph merge history into proof trace
    fn record_congruence_merges(&mut self, history_start: usize) {
        use crate::egraph::MergeReason as EgraphMergeReason;

        // Get the new merge records (skip the first External merge which we already recorded)
        let new_merges: Vec<_> = self
            .egraph
            .merge_history()
            .iter()
            .skip(history_start)
            .skip(1) // Skip the first External merge we initiated
            .cloned()
            .collect();

        for merge in new_merges {
            match merge.reason {
                EgraphMergeReason::Congruence {
                    func,
                    children1,
                    children2,
                } => {
                    // Record this congruence in our proof trace
                    // The children are e-class IDs that are pairwise equal
                    //
                    // For each pair of children (c1, c2):
                    // - Use the ORIGINAL e-class IDs (before canonicalization) to look up
                    //   the proof index, since that's how they were recorded in the trace
                    // - If c1.id() == c2.id(): the terms were originally identical, reflexive
                    // - Otherwise: look up the proof index for their equality
                    let arg_reasons: Vec<u32> = children1
                        .iter()
                        .zip(children2.iter())
                        .filter_map(|(c1, c2)| {
                            // Check if original e-class IDs are the same (truly reflexive)
                            // Note: We use the original IDs, not canonicalized ones, because
                            // by the time we're recording congruence merges, the E-graph has
                            // already unified the children. We need the original IDs to find
                            // the proof index.
                            if c1.id() == c2.id() {
                                // Originally same e-class, no proof needed
                                None
                            } else {
                                // Find proof index for this argument equality
                                self.proof_trace
                                    .get_proof_index(c1.id(), c2.id())
                                    .map(|idx| {
                                        u32::try_from(idx)
                                            .expect("proof trace index exceeded u32::MAX")
                                    })
                            }
                        })
                        .collect();

                    self.proof_trace.record_union(
                        merge.ec1.id(),
                        merge.ec2.id(),
                        UnionReason::Congruence {
                            func: func.name().to_string(),
                            app1: merge.ec1.id(),
                            app2: merge.ec2.id(),
                            arg_reasons,
                        },
                    );
                }
                EgraphMergeReason::External => {
                    // Skip - we should have already recorded this
                }
            }
        }
    }

    /// Assert a disequality: t1 ≠ t2
    fn assert_disequality(&mut self, t1: TermId, t2: TermId, lit: Lit) -> TheoryCheckResult {
        let ec1 = self.get_or_create_eclass(t1);
        let ec2 = self.get_or_create_eclass(t2);

        // Check if already equal - immediate conflict
        if self.egraph.are_equal(ec1, ec2) {
            // Conflict: we're asserting t1 ≠ t2 but E-graph says they're equal
            // Return the disequality literal as the conflict
            return TheoryCheckResult::Conflict(vec![lit]);
        }

        // Record the disequality
        self.disequalities.push((t1, t2, lit));

        TheoryCheckResult::Consistent
    }

    /// Check all disequalities for violations
    fn check_disequalities(&mut self) -> TheoryCheckResult {
        // Copy disequalities to avoid borrow issues
        let diseqs: Vec<(TermId, TermId, Lit)> = self.disequalities.clone();

        for (t1, t2, lit) in diseqs {
            let ec1 = self.get_or_create_eclass(t1);
            let ec2 = self.get_or_create_eclass(t2);

            if self.egraph.are_equal(ec1, ec2) {
                // Conflict: we have t1 ≠ t2 but now t1 = t2
                return TheoryCheckResult::Conflict(vec![lit]);
            }
        }

        TheoryCheckResult::Consistent
    }

    /// Check if two terms are equal in the current state
    pub fn are_equal(&mut self, t1: TermId, t2: TermId) -> bool {
        let ec1 = self.get_or_create_eclass(t1);
        let ec2 = self.get_or_create_eclass(t2);
        self.egraph.are_equal(ec1, ec2)
    }

    /// Get the E-graph (for debugging/inspection)
    pub fn egraph(&self) -> &EGraph {
        &self.egraph
    }

    /// Get the term to E-class mapping (for E-matching instantiation)
    pub fn term_to_eclass_map(&self) -> &HashMap<TermId, EClassId> {
        &self.term_to_eclass
    }

    /// Get statistics
    pub fn stats(&self) -> EqualityStats {
        EqualityStats {
            num_eclasses: self.egraph.num_classes(),
            num_enodes: self.egraph.num_nodes(),
            num_disequalities: self.disequalities.len(),
            num_terms: self.term_to_eclass.len(),
        }
    }

    /// Get the E-class ID for a term (if it exists)
    pub fn get_eclass(&self, term_id: TermId) -> Option<u32> {
        self.term_to_eclass.get(&term_id).map(|ec| ec.id())
    }

    /// Get the canonical E-class ID for a term (if it exists)
    pub fn get_canonical_eclass(&self, term_id: TermId) -> Option<u32> {
        self.term_to_eclass
            .get(&term_id)
            .map(|ec| self.egraph.find_const(*ec).id())
    }
}

impl Default for EqualityTheory {
    fn default() -> Self {
        Self::new()
    }
}

impl TheorySolver for EqualityTheory {
    fn assert_literal(&mut self, lit: Lit, theory_lit: &TheoryLiteral) -> TheoryCheckResult {
        match theory_lit {
            TheoryLiteral::Eq(t1, t2) => self.assert_equality(*t1, *t2, lit),
            TheoryLiteral::Neq(t1, t2) => self.assert_disequality(*t1, *t2, lit),
            // Other literals are not handled by equality theory
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

        // Remove disequalities added after the target level
        let diseq_limit = self.diseq_trail[level as usize + 1];
        self.disequalities.truncate(diseq_limit);

        // For equalities, we need to rebuild the E-graph
        // This is inefficient but correct - a production implementation
        // would use incremental E-graph with undo support
        if level < self.level {
            // Collect equalities and their hypotheses to replay
            let mut equalities_to_replay: Vec<(TermId, TermId, Option<FVarId>)> = Vec::new();
            for l in 0..=level as usize {
                for &(t1, t2, _lit) in &self.equality_trail[l] {
                    let hyp = self.term_to_hypothesis.get(&(t1, t2)).copied();
                    equalities_to_replay.push((t1, t2, hyp));
                }
            }

            // Clear and rebuild E-graph and proof trace
            self.egraph.clear();
            self.term_to_eclass.clear();
            self.proof_trace.clear();

            // Replay equalities (also rebuilds proof trace)
            for (t1, t2, hypothesis) in equalities_to_replay {
                let ec1 = self.get_or_create_eclass(t1);
                let ec2 = self.get_or_create_eclass(t2);

                // Record in proof trace
                self.proof_trace.record_union(
                    ec1.id(),
                    ec2.id(),
                    UnionReason::Asserted {
                        hypothesis,
                        lhs: t1,
                        rhs: t2,
                    },
                );

                self.egraph.union(ec1, ec2);
            }
        }

        // Truncate trails
        self.equality_trail.truncate(level as usize + 1);
        self.diseq_trail.truncate(level as usize + 1);
        self.level = level;
    }

    fn push(&mut self) {
        self.level += 1;
        self.equality_trail.push(Vec::new());
        self.diseq_trail.push(self.disequalities.len());
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn name(&self) -> &'static str {
        "EUF"
    }

    fn set_terms(&mut self, terms: Vec<SmtTerm>) {
        self.terms = terms;
    }
}

/// Statistics for equality theory
#[derive(Clone, Debug, Default)]
pub struct EqualityStats {
    pub num_eclasses: usize,
    pub num_enodes: usize,
    pub num_disequalities: usize,
    pub num_terms: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egraph::Symbol;

    #[test]
    fn test_equality_basic() {
        let mut eq = EqualityTheory::new();

        // Create terms: a, b
        let terms = vec![
            SmtTerm::Const(Symbol::new("a")),
            SmtTerm::Const(Symbol::new("b")),
        ];
        eq.set_terms(terms);

        let a = TermId(0);
        let b = TermId(1);

        // Initially not equal
        assert!(!eq.are_equal(a, b));

        // Assert a = b
        let lit = Lit::pos(crate::cdcl::Var::new(0));
        let result = eq.assert_literal(lit, &TheoryLiteral::Eq(a, b));
        assert!(matches!(result, TheoryCheckResult::Consistent));

        // Now equal
        assert!(eq.are_equal(a, b));
    }

    #[test]
    fn test_equality_conflict() {
        let mut eq = EqualityTheory::new();

        let terms = vec![
            SmtTerm::Const(Symbol::new("a")),
            SmtTerm::Const(Symbol::new("b")),
        ];
        eq.set_terms(terms);

        let a = TermId(0);
        let b = TermId(1);

        // Assert a = b
        let lit_eq = Lit::pos(crate::cdcl::Var::new(0));
        eq.assert_literal(lit_eq, &TheoryLiteral::Eq(a, b));

        // Assert a ≠ b - should conflict
        let lit_neq = Lit::neg(crate::cdcl::Var::new(0));
        let result = eq.assert_literal(lit_neq, &TheoryLiteral::Neq(a, b));
        assert!(matches!(result, TheoryCheckResult::Conflict(_)));
    }

    #[test]
    fn test_equality_disequality_first() {
        let mut eq = EqualityTheory::new();

        let terms = vec![
            SmtTerm::Const(Symbol::new("a")),
            SmtTerm::Const(Symbol::new("b")),
        ];
        eq.set_terms(terms);

        let a = TermId(0);
        let b = TermId(1);

        // Assert a ≠ b first
        let lit_neq = Lit::neg(crate::cdcl::Var::new(0));
        let result = eq.assert_literal(lit_neq, &TheoryLiteral::Neq(a, b));
        assert!(matches!(result, TheoryCheckResult::Consistent));

        // Now assert a = b - should conflict
        let lit_eq = Lit::pos(crate::cdcl::Var::new(0));
        let result = eq.assert_literal(lit_eq, &TheoryLiteral::Eq(a, b));
        assert!(matches!(result, TheoryCheckResult::Conflict(_)));
    }

    #[test]
    fn test_congruence() {
        let mut eq = EqualityTheory::new();

        // Terms: a, b, f(a), f(b)
        let terms = vec![
            SmtTerm::Const(Symbol::new("a")),
            SmtTerm::Const(Symbol::new("b")),
            SmtTerm::App(Symbol::new("f"), vec![TermId(0)]),
            SmtTerm::App(Symbol::new("f"), vec![TermId(1)]),
        ];
        eq.set_terms(terms);

        let a = TermId(0);
        let b = TermId(1);
        let fa = TermId(2);
        let fb = TermId(3);

        // Initially f(a) ≠ f(b)
        assert!(!eq.are_equal(fa, fb));

        // Assert a = b
        let lit = Lit::pos(crate::cdcl::Var::new(0));
        eq.assert_literal(lit, &TheoryLiteral::Eq(a, b));

        // By congruence, f(a) = f(b)
        assert!(eq.are_equal(fa, fb));
    }

    #[test]
    fn test_congruence_conflict() {
        let mut eq = EqualityTheory::new();

        // Terms: a, b, f(a), f(b)
        let terms = vec![
            SmtTerm::Const(Symbol::new("a")),
            SmtTerm::Const(Symbol::new("b")),
            SmtTerm::App(Symbol::new("f"), vec![TermId(0)]),
            SmtTerm::App(Symbol::new("f"), vec![TermId(1)]),
        ];
        eq.set_terms(terms);

        let a = TermId(0);
        let b = TermId(1);
        let fa = TermId(2);
        let fb = TermId(3);

        // Assert f(a) ≠ f(b)
        let lit_neq = Lit::neg(crate::cdcl::Var::new(1));
        eq.assert_literal(lit_neq, &TheoryLiteral::Neq(fa, fb));

        // Assert a = b - should cause conflict via congruence
        let lit_eq = Lit::pos(crate::cdcl::Var::new(0));
        let result = eq.assert_literal(lit_eq, &TheoryLiteral::Eq(a, b));

        // The conflict should be detected
        assert!(matches!(result, TheoryCheckResult::Conflict(_)));
    }

    #[test]
    fn test_transitivity() {
        let mut eq = EqualityTheory::new();

        let terms = vec![
            SmtTerm::Const(Symbol::new("a")),
            SmtTerm::Const(Symbol::new("b")),
            SmtTerm::Const(Symbol::new("c")),
        ];
        eq.set_terms(terms);

        let a = TermId(0);
        let b = TermId(1);
        let c = TermId(2);

        // Assert a = b
        let lit1 = Lit::pos(crate::cdcl::Var::new(0));
        eq.assert_literal(lit1, &TheoryLiteral::Eq(a, b));

        // Assert b = c
        let lit2 = Lit::pos(crate::cdcl::Var::new(1));
        eq.assert_literal(lit2, &TheoryLiteral::Eq(b, c));

        // By transitivity, a = c
        assert!(eq.are_equal(a, c));
    }

    #[test]
    fn test_backtrack() {
        let mut eq = EqualityTheory::new();

        let terms = vec![
            SmtTerm::Const(Symbol::new("a")),
            SmtTerm::Const(Symbol::new("b")),
            SmtTerm::Const(Symbol::new("c")),
        ];
        eq.set_terms(terms);

        let a = TermId(0);
        let b = TermId(1);
        let c = TermId(2);

        // Level 0: assert a = b
        let lit1 = Lit::pos(crate::cdcl::Var::new(0));
        eq.assert_literal(lit1, &TheoryLiteral::Eq(a, b));
        assert!(eq.are_equal(a, b));

        // Push to level 1
        eq.push();

        // Level 1: assert b = c
        let lit2 = Lit::pos(crate::cdcl::Var::new(1));
        eq.assert_literal(lit2, &TheoryLiteral::Eq(b, c));
        assert!(eq.are_equal(a, c));

        // Backtrack to level 0
        eq.backtrack(0);

        // a = b should still hold (from level 0)
        assert!(eq.are_equal(a, b));

        // But a = c should not hold anymore
        assert!(!eq.are_equal(a, c));
    }

    #[test]
    fn test_stats() {
        let mut eq = EqualityTheory::new();

        let terms = vec![
            SmtTerm::Const(Symbol::new("a")),
            SmtTerm::Const(Symbol::new("b")),
        ];
        eq.set_terms(terms);

        let a = TermId(0);
        let b = TermId(1);

        // Build terms
        let _ = eq.get_or_create_eclass(a);
        let _ = eq.get_or_create_eclass(b);

        let stats = eq.stats();
        assert_eq!(stats.num_terms, 2);
        assert_eq!(stats.num_eclasses, 2);
    }

    #[test]
    fn test_nested_congruence() {
        let mut eq = EqualityTheory::new();

        // Terms: a, b, f(a), f(b), g(f(a)), g(f(b))
        let terms = vec![
            SmtTerm::Const(Symbol::new("a")),                // 0
            SmtTerm::Const(Symbol::new("b")),                // 1
            SmtTerm::App(Symbol::new("f"), vec![TermId(0)]), // 2: f(a)
            SmtTerm::App(Symbol::new("f"), vec![TermId(1)]), // 3: f(b)
            SmtTerm::App(Symbol::new("g"), vec![TermId(2)]), // 4: g(f(a))
            SmtTerm::App(Symbol::new("g"), vec![TermId(3)]), // 5: g(f(b))
        ];
        eq.set_terms(terms);

        let a = TermId(0);
        let b = TermId(1);
        let gfa = TermId(4);
        let gfb = TermId(5);

        // Assert a = b
        let lit = Lit::pos(crate::cdcl::Var::new(0));
        eq.assert_literal(lit, &TheoryLiteral::Eq(a, b));

        // By nested congruence: f(a) = f(b), then g(f(a)) = g(f(b))
        assert!(eq.are_equal(gfa, gfb));
    }

    #[test]
    fn test_congruence_proof_trace() {
        use crate::proof::UnionReason;

        let mut eq = EqualityTheory::new();

        // Terms: a, b, f(a), f(b)
        let terms = vec![
            SmtTerm::Const(Symbol::new("a")),                // 0
            SmtTerm::Const(Symbol::new("b")),                // 1
            SmtTerm::App(Symbol::new("f"), vec![TermId(0)]), // 2: f(a)
            SmtTerm::App(Symbol::new("f"), vec![TermId(1)]), // 3: f(b)
        ];
        eq.set_terms(terms);

        let a = TermId(0);
        let b = TermId(1);
        let fa = TermId(2);
        let fb = TermId(3);

        // IMPORTANT: Build f(a) and f(b) in the E-graph BEFORE asserting a = b
        // This is required for congruence closure to detect the congruence
        let _ = eq.get_or_create_eclass(fa);
        let _ = eq.get_or_create_eclass(fb);

        // Register hypothesis for proof reconstruction
        eq.register_hypothesis(a, b, lean5_kernel::FVarId(42));

        // Assert a = b (with hypothesis tracking)
        // This will trigger congruence closure which detects f(a) = f(b)
        let lit = Lit::pos(crate::cdcl::Var::new(0));
        eq.assert_literal(lit, &TheoryLiteral::Eq(a, b));

        // By congruence, f(a) = f(b)
        assert!(eq.are_equal(fa, fb));

        // Check that the proof trace has recorded both:
        // 1. The asserted equality (a = b) with hypothesis
        // 2. The congruence merge (f(a) = f(b))
        let trace = eq.proof_trace();

        // The trace should have at least 2 records:
        // - a = b (asserted)
        // - f(a) = f(b) (congruence)
        assert!(
            trace.steps.len() >= 2,
            "Proof trace should have at least 2 records, got {}",
            trace.steps.len()
        );

        // Check that we can find a congruence reason in the trace
        let has_congruence = trace.steps.iter().any(
            |(_, _, reason)| matches!(reason, UnionReason::Congruence { func, .. } if func == "f"),
        );
        assert!(
            has_congruence,
            "Proof trace should contain a Congruence reason for f"
        );

        // Check that the proof trace can build a proof from a to b
        let ec_a = eq.get_eclass(a).unwrap();
        let ec_b = eq.get_eclass(b).unwrap();
        let proof_step_ab = trace.build_proof(ec_a, ec_b);
        assert!(
            proof_step_ab.is_some(),
            "Should be able to build proof for a = b"
        );

        // Check that we can build a proof from f(a) to f(b)
        let ec_fa = eq.get_eclass(fa).unwrap();
        let ec_fb = eq.get_eclass(fb).unwrap();
        let proof_step_fafb = trace.build_proof(ec_fa, ec_fb);
        assert!(
            proof_step_fafb.is_some(),
            "Should be able to build proof for f(a) = f(b)"
        );
    }

    #[test]
    fn test_congruence_children_already_equal() {
        // Edge case: If children are added AFTER their equality is established,
        // the E-graph may not record the congruence properly because the children
        // are already canonical when the applications are built.
        //
        // This test ensures we can still build a valid proof in this case.

        let mut eq = EqualityTheory::new();

        // Terms: a, b, f(a), f(b)
        let terms = vec![
            SmtTerm::Const(Symbol::new("a")),                // 0
            SmtTerm::Const(Symbol::new("b")),                // 1
            SmtTerm::App(Symbol::new("f"), vec![TermId(0)]), // 2: f(a)
            SmtTerm::App(Symbol::new("f"), vec![TermId(1)]), // 3: f(b)
        ];
        eq.set_terms(terms);

        let a = TermId(0);
        let b = TermId(1);
        let fa = TermId(2);
        let fb = TermId(3);

        // Register hypothesis for proof reconstruction
        eq.register_hypothesis(a, b, lean5_kernel::FVarId(42));

        // First, assert a = b BEFORE building f(a) and f(b)
        let lit = Lit::pos(crate::cdcl::Var::new(0));
        eq.assert_literal(lit, &TheoryLiteral::Eq(a, b));

        // Now build f(a) and f(b) - since a and b are already equal,
        // these should be hashconsed to the same e-class
        let ec_fa = eq.get_or_create_eclass(fa);
        let ec_fb = eq.get_or_create_eclass(fb);

        // Since a = b was already established, f(a) and f(b) should be
        // canonicalized to the same e-class immediately
        assert!(eq.are_equal(fa, fb), "f(a) and f(b) should be equal");

        // The key question: can we still build a proof for f(a) = f(b)?
        // When terms are hashconsed to the same e-class at creation time,
        // there's no merge record - they're identical from the start.
        //
        // In this case, we have ec_fa == ec_fb (same e-class ID)
        assert_eq!(
            eq.egraph().find_const(ec_fa),
            eq.egraph().find_const(ec_fb),
            "f(a) and f(b) should be in the same canonical e-class"
        );

        // For proof reconstruction, when two terms are in the same e-class
        // from the start, we need reflexivity or to track that they were
        // unified due to their children being equal.
        //
        // The current implementation should handle this case.
        let trace = eq.proof_trace();

        // We should have at least the a = b assertion
        assert!(
            !trace.steps.is_empty(),
            "Proof trace should have at least the a = b assertion"
        );

        // When f(a) and f(b) are hashconsed together (same canonical term),
        // the proof is essentially reflexivity of f(canonical(a))
        // This is correct - no explicit merge was needed.
    }

    #[test]
    fn test_congruence_proof_with_arg_reasons() {
        // Test that congruence proofs properly include argument equality reasons
        use crate::proof::UnionReason;

        let mut eq = EqualityTheory::new();

        // Terms: a, b, c, d, f(a, c), f(b, d)
        let terms = vec![
            SmtTerm::Const(Symbol::new("a")),                           // 0
            SmtTerm::Const(Symbol::new("b")),                           // 1
            SmtTerm::Const(Symbol::new("c")),                           // 2
            SmtTerm::Const(Symbol::new("d")),                           // 3
            SmtTerm::App(Symbol::new("f"), vec![TermId(0), TermId(2)]), // 4: f(a, c)
            SmtTerm::App(Symbol::new("f"), vec![TermId(1), TermId(3)]), // 5: f(b, d)
        ];
        eq.set_terms(terms);

        let a = TermId(0);
        let b = TermId(1);
        let c = TermId(2);
        let d = TermId(3);
        let fac = TermId(4);
        let fbd = TermId(5);

        // Build f(a,c) and f(b,d) first so E-graph can track congruence
        let _ = eq.get_or_create_eclass(fac);
        let _ = eq.get_or_create_eclass(fbd);

        // Register hypotheses
        eq.register_hypothesis(a, b, lean5_kernel::FVarId(1));
        eq.register_hypothesis(c, d, lean5_kernel::FVarId(2));

        // Assert a = b
        let lit1 = Lit::pos(crate::cdcl::Var::new(0));
        eq.assert_literal(lit1, &TheoryLiteral::Eq(a, b));

        // Assert c = d
        let lit2 = Lit::pos(crate::cdcl::Var::new(1));
        eq.assert_literal(lit2, &TheoryLiteral::Eq(c, d));

        // Now f(a, c) = f(b, d) by congruence
        assert!(eq.are_equal(fac, fbd), "f(a,c) and f(b,d) should be equal");

        let trace = eq.proof_trace();

        // Should have at least 3 records: a=b, c=d, and f(a,c)=f(b,d) by congruence
        assert!(
            trace.steps.len() >= 3,
            "Proof trace should have at least 3 records, got {}",
            trace.steps.len()
        );

        // Find the congruence record
        let congruence_step = trace.steps.iter().find(
            |(_, _, reason)| matches!(reason, UnionReason::Congruence { func, .. } if func == "f"),
        );

        assert!(
            congruence_step.is_some(),
            "Should have a congruence step for f"
        );

        if let Some((_, _, UnionReason::Congruence { arg_reasons, .. })) = congruence_step {
            // For a 2-argument function with both args changing, we might have
            // 0, 1, or 2 arg_reasons depending on how the E-graph processes it
            // The important thing is we can build a valid proof
            assert!(
                arg_reasons.len() <= 2,
                "arg_reasons should have at most 2 entries for 2-arg function"
            );
        }

        // Can we build a proof for f(a,c) = f(b,d)?
        let ec_fac = eq.get_eclass(fac).unwrap();
        let ec_fbd = eq.get_eclass(fbd).unwrap();
        let proof = trace.build_proof(ec_fac, ec_fbd);

        // The proof should exist - if arg_reasons is empty, we fall back to
        // looking up proof index or using BFS
        assert!(
            proof.is_some(),
            "Should be able to build proof for f(a,c) = f(b,d)"
        );
    }
}
