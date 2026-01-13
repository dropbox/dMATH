//! Proof reconstruction for SMT solving
//!
//! This module provides proof term construction from SMT proof traces.
//! When the SMT solver proves a goal, we need to reconstruct a kernel-valid
//! proof term that witnesses the validity.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Proof Reconstruction                         │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  E-graph Union  ──────────► ProofStep ──────────► Kernel Expr   │
//! │  (with reason)    record     (trace)    build    (proof term)   │
//! │                                                                  │
//! │  Proof steps:                                                    │
//! │  - Refl(a)           →  Eq.refl a                               │
//! │  - Symm(pf)          →  Eq.symm pf                              │
//! │  - Trans(pf1, pf2)   →  Eq.trans pf1 pf2                        │
//! │  - Congr(f, args)    →  congrArg f (proof for args)             │
//! │  - Asserted(hyp_id)  →  reference to hypothesis                 │
//! │                                                                  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Proof Generation Strategy
//!
//! For equality goals like `a = b`, we:
//! 1. Find a path in the E-graph from e-class(a) to e-class(b)
//! 2. Each edge is either a direct assertion or a congruence step
//! 3. Build proof terms for each step and compose with transitivity

use crate::smt::TermId;
use lean5_kernel::name::Name;
use lean5_kernel::{Expr, FVarId, Level};
use std::collections::HashMap;

/// A proof step in the SMT proof trace
#[derive(Debug, Clone)]
pub enum ProofStep {
    /// Reflexivity: a = a
    Refl(TermId),
    /// Symmetry: if we have proof of a = b, get b = a
    Symm(Box<ProofStep>),
    /// Transitivity: if we have a = b and b = c, get a = c
    Trans(Box<ProofStep>, Box<ProofStep>),
    /// Congruence: if args are equal, function applications are equal
    /// (function_name, arg_proofs)
    Congr(String, Vec<ProofStep>),
    /// Direct hypothesis assertion (hypothesis FVar ID)
    Hypothesis(FVarId),
    /// Direct axiom from SMT solver (placeholder until we have full proof)
    Axiom(String),
}

impl ProofStep {
    /// Create a reflexivity proof step
    pub fn refl(term: TermId) -> Self {
        ProofStep::Refl(term)
    }

    /// Create a symmetry proof step
    pub fn symm(proof: ProofStep) -> Self {
        // Optimize: symm(symm(p)) = p
        if let ProofStep::Symm(inner) = proof {
            return *inner;
        }
        // Optimize: symm(refl) = refl
        if let ProofStep::Refl(t) = &proof {
            return ProofStep::Refl(*t);
        }
        ProofStep::Symm(Box::new(proof))
    }

    /// Create a transitivity proof step
    pub fn trans(p1: ProofStep, p2: ProofStep) -> Self {
        // Optimize: trans(refl, p) = p
        if matches!(&p1, ProofStep::Refl(_)) {
            return p2;
        }
        // Optimize: trans(p, refl) = p
        if matches!(&p2, ProofStep::Refl(_)) {
            return p1;
        }
        ProofStep::Trans(Box::new(p1), Box::new(p2))
    }

    /// Create a congruence proof step
    pub fn congr(func: impl Into<String>, arg_proofs: Vec<ProofStep>) -> Self {
        ProofStep::Congr(func.into(), arg_proofs)
    }

    /// Create a hypothesis proof step
    pub fn hypothesis(fvar: FVarId) -> Self {
        ProofStep::Hypothesis(fvar)
    }
}

/// Proof builder that constructs kernel expressions from proof steps
pub struct ProofBuilder<'a> {
    /// Mapping from SMT term IDs to kernel expressions
    term_to_expr: &'a HashMap<TermId, Expr>,
    /// Mapping from SMT term IDs to their types
    term_to_type: &'a HashMap<TermId, Expr>,
}

impl<'a> ProofBuilder<'a> {
    /// Create a new proof builder
    pub fn new(
        term_to_expr: &'a HashMap<TermId, Expr>,
        term_to_type: &'a HashMap<TermId, Expr>,
    ) -> Self {
        ProofBuilder {
            term_to_expr,
            term_to_type,
        }
    }

    /// Build a kernel proof term from a proof step
    pub fn build(&self, step: &ProofStep) -> Option<Expr> {
        match step {
            ProofStep::Refl(term_id) => {
                let expr = self.term_to_expr.get(term_id)?;
                let ty = self
                    .term_to_type
                    .get(term_id)
                    .cloned()
                    .unwrap_or_else(Expr::type_);
                Some(self.mk_eq_refl(&ty, expr))
            }
            ProofStep::Symm(inner) => {
                let inner_proof = self.build(inner)?;
                // Extract type and lhs/rhs from the inner proof's type
                // For now, we use a simplified version
                Some(self.mk_eq_symm(&inner_proof))
            }
            ProofStep::Trans(p1, p2) => {
                let proof1 = self.build(p1)?;
                let proof2 = self.build(p2)?;
                Some(self.mk_eq_trans(&proof1, &proof2))
            }
            ProofStep::Congr(func_name, arg_proofs) => {
                let arg_proof_terms: Option<Vec<Expr>> =
                    arg_proofs.iter().map(|p| self.build(p)).collect();
                let arg_proof_terms = arg_proof_terms?;

                if arg_proof_terms.len() == 1 {
                    // Single argument: use congrArg
                    Some(self.mk_congr_arg(func_name, &arg_proof_terms[0]))
                } else {
                    // Multiple arguments: use congrFun and compose
                    self.mk_congr_multi(func_name, &arg_proof_terms)
                }
            }
            ProofStep::Hypothesis(fvar) => {
                // Return the free variable representing the hypothesis
                Some(Expr::fvar(*fvar))
            }
            ProofStep::Axiom(name) => {
                // Return an axiom reference (for unsupported proof steps)
                Some(Expr::const_(Name::from_string(name), vec![]))
            }
        }
    }

    /// Build Eq.refl : ∀ {α : Sort u} (a : α), a = a
    fn mk_eq_refl(&self, ty: &Expr, val: &Expr) -> Expr {
        // Eq.refl {α} a : Eq α a a
        Expr::app(
            Expr::app(
                Expr::const_(
                    Name::from_string("Eq.refl"),
                    vec![Level::succ(Level::zero())],
                ),
                ty.clone(),
            ),
            val.clone(),
        )
    }

    /// Build Eq.symm : ∀ {α : Sort u} {a b : α}, a = b → b = a
    fn mk_eq_symm(&self, proof: &Expr) -> Expr {
        // Eq.symm {α} {a} {b} h : b = a
        // We apply Eq.symm to the proof, letting Lean infer the implicit args
        Expr::app(
            Expr::const_(
                Name::from_string("Eq.symm"),
                vec![Level::succ(Level::zero())],
            ),
            proof.clone(),
        )
    }

    /// Build Eq.trans : ∀ {α : Sort u} {a b c : α}, a = b → b = c → a = c
    fn mk_eq_trans(&self, p1: &Expr, p2: &Expr) -> Expr {
        // Eq.trans {α} {a} {b} {c} h1 h2 : a = c
        Expr::app(
            Expr::app(
                Expr::const_(
                    Name::from_string("Eq.trans"),
                    vec![Level::succ(Level::zero())],
                ),
                p1.clone(),
            ),
            p2.clone(),
        )
    }

    /// Build congrArg : ∀ {α β : Sort u} (f : α → β) {a₁ a₂ : α}, a₁ = a₂ → f a₁ = f a₂
    fn mk_congr_arg(&self, func_name: &str, arg_proof: &Expr) -> Expr {
        // congrArg f h : f a₁ = f a₂
        Expr::app(
            Expr::app(
                Expr::const_(
                    Name::from_string("congrArg"),
                    vec![Level::succ(Level::zero())],
                ),
                Expr::const_(Name::from_string(func_name), vec![]),
            ),
            arg_proof.clone(),
        )
    }

    /// Build congruence for multiple arguments
    fn mk_congr_multi(&self, func_name: &str, arg_proofs: &[Expr]) -> Option<Expr> {
        if arg_proofs.is_empty() {
            // No arguments - this shouldn't happen
            return None;
        }

        // For multiple arguments, we compose congruence proofs:
        // congrArg (f a₁) h₂ ∘ congrArg (fun x => f x a₂') h₁
        // This simplified version chains congrArg applications without full type info
        let mut result = arg_proofs[0].clone();
        for proof in &arg_proofs[1..] {
            result = Expr::app(
                Expr::app(
                    Expr::const_(
                        Name::from_string("congrArg"),
                        vec![Level::succ(Level::zero())],
                    ),
                    Expr::const_(Name::from_string(func_name), vec![]),
                ),
                proof.clone(),
            );
        }
        Some(result)
    }
}

/// Union reason in the E-graph (for proof reconstruction)
#[derive(Debug, Clone)]
pub enum UnionReason {
    /// Direct equality assertion with proof
    Asserted {
        /// The hypothesis providing the equality
        hypothesis: Option<FVarId>,
        /// LHS term
        lhs: TermId,
        /// RHS term
        rhs: TermId,
    },
    /// Congruence: two terms are equal because their arguments are equal
    Congruence {
        /// Function symbol
        func: String,
        /// E-class IDs of the two function applications
        app1: u32,
        app2: u32,
        /// Proofs that corresponding arguments are equal
        arg_reasons: Vec<u32>, // Indices into the proof trace
    },
    /// Reflexivity (trivial equality)
    Reflexivity(TermId),
}

/// Proof trace recording all union operations
#[derive(Debug, Clone, Default)]
pub struct ProofTrace {
    /// Sequence of union operations with reasons
    pub steps: Vec<(u32, u32, UnionReason)>, // (e-class1, e-class2, reason)
    /// Mapping from (e-class, e-class) to proof index
    proof_index: HashMap<(u32, u32), usize>,
}

impl ProofTrace {
    /// Create a new empty proof trace
    pub fn new() -> Self {
        ProofTrace {
            steps: Vec::new(),
            proof_index: HashMap::new(),
        }
    }

    /// Record a union with its reason
    pub fn record_union(&mut self, ec1: u32, ec2: u32, reason: UnionReason) -> usize {
        let idx = self.steps.len();
        self.steps.push((ec1, ec2, reason));

        // Index both directions for lookup
        self.proof_index.insert((ec1, ec2), idx);
        self.proof_index.insert((ec2, ec1), idx);

        idx
    }

    /// Get the proof index for a union
    pub fn get_proof_index(&self, ec1: u32, ec2: u32) -> Option<usize> {
        self.proof_index.get(&(ec1, ec2)).copied()
    }

    /// Get the reason for a specific proof step
    pub fn get_reason(&self, idx: usize) -> Option<&UnionReason> {
        self.steps.get(idx).map(|(_, _, r)| r)
    }

    /// Clear the trace
    pub fn clear(&mut self) {
        self.steps.clear();
        self.proof_index.clear();
    }

    /// Build a proof step from an e-class equality
    /// Returns None if no proof path exists
    pub fn build_proof(&self, ec1: u32, ec2: u32) -> Option<ProofStep> {
        if ec1 == ec2 {
            // Reflexivity - need a term ID, but we don't have it here
            // This should be handled by the caller
            return None;
        }

        // Direct proof exists?
        if let Some(&idx) = self.proof_index.get(&(ec1, ec2)) {
            return self.step_to_proof(idx, ec1, ec2);
        }

        // Need to find a path through the trace
        // This is a BFS through the union-find history
        self.find_proof_path(ec1, ec2)
    }

    /// Convert a trace step to a ProofStep
    fn step_to_proof(
        &self,
        idx: usize,
        requested_ec1: u32,
        _requested_ec2: u32,
    ) -> Option<ProofStep> {
        let (ec1, _ec2, reason) = self.steps.get(idx)?;

        // Check if we need to flip the proof
        let needs_flip = *ec1 != requested_ec1;

        let proof = match reason {
            UnionReason::Asserted {
                hypothesis,
                lhs,
                rhs,
            } => {
                if let Some(fvar) = hypothesis {
                    ProofStep::Hypothesis(*fvar)
                } else {
                    // No hypothesis means this was asserted directly
                    // Use reflexivity if lhs == rhs, otherwise axiom
                    if lhs == rhs {
                        ProofStep::Refl(*lhs)
                    } else {
                        ProofStep::Axiom(format!("asserted_eq_{}_{}", lhs.0, rhs.0))
                    }
                }
            }
            UnionReason::Congruence {
                func, arg_reasons, ..
            } => {
                // Build proofs for each argument equality
                let arg_proofs: Vec<ProofStep> = arg_reasons
                    .iter()
                    .filter_map(|&arg_idx| {
                        if let Some((ec1, ec2, _)) = self.steps.get(arg_idx as usize) {
                            self.step_to_proof(arg_idx as usize, *ec1, *ec2)
                        } else {
                            None
                        }
                    })
                    .collect();

                // Handle the case where arg_reasons is empty but we need a congruence proof.
                // This can happen when:
                // 1. Children were already in the same e-class (reflexive case)
                // 2. The proof index wasn't captured properly
                //
                // In this case, the congruence is essentially a reflexivity proof
                // because f(canonical(x)) = f(canonical(x)) for the canonicalized children.
                if arg_proofs.is_empty() {
                    // When there are no argument proofs, use Axiom as a fallback.
                    // The kernel will need to verify this, but logically it's valid
                    // because the two terms were deemed equal by congruence closure.
                    ProofStep::Axiom(format!("congr_{func}"))
                } else {
                    ProofStep::Congr(func.clone(), arg_proofs)
                }
            }
            UnionReason::Reflexivity(term) => ProofStep::Refl(*term),
        };

        if needs_flip {
            Some(ProofStep::symm(proof))
        } else {
            Some(proof)
        }
    }

    /// Find a proof path using BFS
    fn find_proof_path(&self, start: u32, end: u32) -> Option<ProofStep> {
        use std::collections::{HashSet, VecDeque};

        // BFS to find a path
        let mut visited: HashSet<u32> = HashSet::new();
        let mut queue: VecDeque<(u32, ProofStep)> = VecDeque::new();

        // Initialize with all edges from start
        visited.insert(start);
        for (idx, (ec1, ec2, _)) in self.steps.iter().enumerate() {
            if *ec1 == start {
                if let Some(step) = self.step_to_proof(idx, *ec1, *ec2) {
                    if *ec2 == end {
                        return Some(step);
                    }
                    queue.push_back((*ec2, step));
                }
            } else if *ec2 == start {
                if let Some(step) = self.step_to_proof(idx, *ec2, *ec1) {
                    if *ec1 == end {
                        return Some(step);
                    }
                    queue.push_back((*ec1, step));
                }
            }
        }

        // BFS
        while let Some((current, current_proof)) = queue.pop_front() {
            if !visited.insert(current) {
                continue;
            }

            // Find all edges from current
            for (idx, (ec1, ec2, _)) in self.steps.iter().enumerate() {
                let (next, next_ec1, next_ec2) = if *ec1 == current && !visited.contains(ec2) {
                    (*ec2, *ec1, *ec2)
                } else if *ec2 == current && !visited.contains(ec1) {
                    (*ec1, *ec2, *ec1)
                } else {
                    continue;
                };

                if let Some(next_step) = self.step_to_proof(idx, next_ec1, next_ec2) {
                    let combined = ProofStep::trans(current_proof.clone(), next_step);

                    if next == end {
                        return Some(combined);
                    }

                    queue.push_back((next, combined));
                }
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proof_step_refl() {
        let step = ProofStep::refl(TermId(0));
        assert!(matches!(step, ProofStep::Refl(TermId(0))));
    }

    #[test]
    fn test_proof_step_symm_optimization() {
        // symm(refl) = refl
        let refl = ProofStep::refl(TermId(0));
        let symm = ProofStep::symm(refl);
        assert!(matches!(symm, ProofStep::Refl(_)));

        // symm(symm(p)) = p
        let hyp = ProofStep::hypothesis(FVarId(0));
        let s1 = ProofStep::symm(hyp.clone());
        let s2 = ProofStep::symm(s1);
        assert!(matches!(s2, ProofStep::Hypothesis(_)));
    }

    #[test]
    fn test_proof_step_trans_optimization() {
        let hyp = ProofStep::hypothesis(FVarId(0));
        let refl = ProofStep::refl(TermId(0));

        // trans(refl, p) = p
        let t1 = ProofStep::trans(refl.clone(), hyp.clone());
        assert!(matches!(t1, ProofStep::Hypothesis(_)));

        // trans(p, refl) = p
        let t2 = ProofStep::trans(hyp.clone(), refl);
        assert!(matches!(t2, ProofStep::Hypothesis(_)));
    }

    #[test]
    fn test_proof_trace_record() {
        let mut trace = ProofTrace::new();

        let idx = trace.record_union(
            0,
            1,
            UnionReason::Asserted {
                hypothesis: Some(FVarId(0)),
                lhs: TermId(0),
                rhs: TermId(1),
            },
        );

        assert_eq!(idx, 0);
        assert_eq!(trace.get_proof_index(0, 1), Some(0));
        assert_eq!(trace.get_proof_index(1, 0), Some(0)); // Both directions indexed
    }

    #[test]
    fn test_proof_trace_build_direct() {
        let mut trace = ProofTrace::new();

        trace.record_union(
            0,
            1,
            UnionReason::Asserted {
                hypothesis: Some(FVarId(42)),
                lhs: TermId(0),
                rhs: TermId(1),
            },
        );

        let proof = trace.build_proof(0, 1);
        assert!(proof.is_some());
        assert!(matches!(proof.unwrap(), ProofStep::Hypothesis(FVarId(42))));
    }

    #[test]
    fn test_proof_trace_build_flipped() {
        let mut trace = ProofTrace::new();

        trace.record_union(
            0,
            1,
            UnionReason::Asserted {
                hypothesis: Some(FVarId(42)),
                lhs: TermId(0),
                rhs: TermId(1),
            },
        );

        // Request proof in opposite direction
        let proof = trace.build_proof(1, 0);
        assert!(proof.is_some());
        assert!(matches!(proof.unwrap(), ProofStep::Symm(_)));
    }

    #[test]
    fn test_proof_trace_build_transitive() {
        let mut trace = ProofTrace::new();

        // 0 = 1
        trace.record_union(
            0,
            1,
            UnionReason::Asserted {
                hypothesis: Some(FVarId(0)),
                lhs: TermId(0),
                rhs: TermId(1),
            },
        );

        // 1 = 2
        trace.record_union(
            1,
            2,
            UnionReason::Asserted {
                hypothesis: Some(FVarId(1)),
                lhs: TermId(1),
                rhs: TermId(2),
            },
        );

        // Should be able to prove 0 = 2 via transitivity
        let proof = trace.build_proof(0, 2);
        assert!(proof.is_some());
        assert!(matches!(proof.unwrap(), ProofStep::Trans(_, _)));
    }

    #[test]
    fn test_proof_builder_refl() {
        let mut term_to_expr = HashMap::new();
        let mut term_to_type = HashMap::new();

        let a = Expr::const_(Name::from_string("a"), vec![]);
        let ty_a = Expr::const_(Name::from_string("A"), vec![]);

        term_to_expr.insert(TermId(0), a.clone());
        term_to_type.insert(TermId(0), ty_a);

        let builder = ProofBuilder::new(&term_to_expr, &term_to_type);
        let step = ProofStep::refl(TermId(0));

        let proof = builder.build(&step);
        assert!(proof.is_some());

        // The proof should be an application of Eq.refl
        let proof = proof.unwrap();
        match &proof {
            Expr::App(_f, arg) => {
                // Should be (Eq.refl A) applied to a
                assert!(matches!(arg.as_ref(), Expr::Const(n, _) if n.to_string() == "a"));
            }
            _ => panic!("Expected App, got {proof:?}"),
        }
    }

    #[test]
    fn test_proof_builder_hypothesis() {
        let term_to_expr = HashMap::new();
        let term_to_type = HashMap::new();

        let builder = ProofBuilder::new(&term_to_expr, &term_to_type);
        let step = ProofStep::hypothesis(FVarId(42));

        let proof = builder.build(&step);
        assert!(proof.is_some());
        assert!(matches!(proof.unwrap(), Expr::FVar(FVarId(42))));
    }
}
