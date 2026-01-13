//! Proof obligation extraction
//!
//! A proof obligation is a formula that needs to be proved.
//! This module extracts obligations from theorems and their proofs.

use sha2::{Digest, Sha256};
use tla_core::ast::{
    BoundVar, Expr, Module, Proof, ProofHint, ProofStep, ProofStepKind, TheoremDecl, Unit,
};
use tla_core::span::{Span, Spanned};

use crate::error::ProofResult;

/// A proof obligation - a formula that must be proved
#[derive(Debug, Clone)]
pub struct Obligation {
    /// Unique identifier for caching
    pub id: ObligationId,
    /// The goal to prove
    pub goal: Spanned<Expr>,
    /// Assumptions available for proving
    pub assumptions: Vec<Spanned<Expr>>,
    /// Definitions that can be expanded
    pub definitions: Vec<String>,
    /// Source location for error reporting
    pub span: Span,
    /// Human-readable description
    pub description: String,
}

/// Unique identifier for an obligation (content-addressed)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ObligationId(pub String);

impl ObligationId {
    /// Create a new obligation ID from a fingerprint
    pub fn from_fingerprint(data: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let result = hasher.finalize();
        ObligationId(hex::encode(&result[..16]))
    }
}

impl std::fmt::Display for ObligationId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Extracts proof obligations from a module
pub struct ObligationExtractor {
    /// Operator definitions available for expansion
    definitions: Vec<(String, Spanned<Expr>)>,
    /// Known facts (theorems, lemmas, assumptions)
    facts: Vec<(String, Spanned<Expr>)>,
}

impl ObligationExtractor {
    pub fn new() -> Self {
        Self {
            definitions: Vec::new(),
            facts: Vec::new(),
        }
    }

    /// Add a definition to the context
    pub fn add_definition(&mut self, name: String, body: Spanned<Expr>) {
        self.definitions.push((name, body));
    }

    /// Add a fact to the context
    pub fn add_fact(&mut self, name: String, body: Spanned<Expr>) {
        self.facts.push((name, body));
    }

    /// Extract all obligations from a module
    pub fn extract_module(&mut self, module: &Module) -> ProofResult<Vec<Obligation>> {
        let mut obligations = Vec::new();

        // First pass: collect definitions and assumptions
        for unit in &module.units {
            match &unit.node {
                Unit::Operator(op) => {
                    self.add_definition(op.name.node.clone(), op.body.clone());
                }
                Unit::Assume(assume) => {
                    let name = assume
                        .name
                        .as_ref()
                        .map(|n| n.node.clone())
                        .unwrap_or_else(|| format!("_ASSUME_{}", self.facts.len()));
                    self.add_fact(name, assume.expr.clone());
                }
                _ => {}
            }
        }

        // Second pass: extract obligations from theorems
        for unit in &module.units {
            if let Unit::Theorem(thm) = &unit.node {
                let thm_obligations = self.extract_theorem(thm)?;
                obligations.extend(thm_obligations);
            }
        }

        Ok(obligations)
    }

    /// Extract obligations from a theorem
    pub fn extract_theorem(&self, thm: &TheoremDecl) -> ProofResult<Vec<Obligation>> {
        let thm_name = thm
            .name
            .as_ref()
            .map(|n| n.node.clone())
            .unwrap_or_else(|| "_THEOREM".to_string());

        match &thm.proof {
            Some(proof) => {
                // Theorem has a proof - extract structured obligations
                self.extract_proof_obligations(&thm_name, &thm.body, &proof.node, proof.span)
            }
            None => {
                // No proof - create a single obligation for the whole theorem
                let obl = self.create_obligation(&thm_name, thm.body.clone(), &[], thm.body.span);
                Ok(vec![obl])
            }
        }
    }

    /// Extract obligations from a proof structure
    fn extract_proof_obligations(
        &self,
        name: &str,
        goal: &Spanned<Expr>,
        proof: &Proof,
        span: Span,
    ) -> ProofResult<Vec<Obligation>> {
        match proof {
            Proof::Obvious => {
                // OBVIOUS - create obligation with goal and current assumptions
                let obl =
                    self.create_obligation(&format!("{}_obvious", name), goal.clone(), &[], span);
                Ok(vec![obl])
            }
            Proof::Omitted => {
                // OMITTED - no obligation, just skip
                Ok(vec![])
            }
            Proof::By(hints) => {
                // BY hints - create obligation with specified hints as assumptions
                let assumptions = self.resolve_hints(hints)?;
                let obl = self.create_obligation(
                    &format!("{}_by", name),
                    goal.clone(),
                    &assumptions,
                    span,
                );
                Ok(vec![obl])
            }
            Proof::Steps(steps) => {
                // Structured proof - create obligations for each step
                self.extract_step_obligations(name, goal, steps)
            }
        }
    }

    /// Extract obligations from proof steps
    fn extract_step_obligations(
        &self,
        name: &str,
        goal: &Spanned<Expr>,
        steps: &[ProofStep],
    ) -> ProofResult<Vec<Obligation>> {
        let mut obligations = Vec::new();
        let mut step_facts: Vec<(String, Spanned<Expr>)> = Vec::new();

        for (i, step) in steps.iter().enumerate() {
            let step_name = step
                .label
                .as_ref()
                .map(|l| l.node.clone())
                .unwrap_or_else(|| format!("{}", i + 1));

            let full_name = format!("{}_{}", name, step_name);

            match &step.kind {
                ProofStepKind::Assert(assertion, proof) => {
                    // <n>. assertion - must prove assertion given previous steps
                    let assumptions: Vec<Spanned<Expr>> =
                        step_facts.iter().map(|(_, e)| e.clone()).collect();

                    if let Some(step_proof) = proof {
                        let step_obls = self.extract_proof_obligations(
                            &full_name,
                            assertion,
                            &step_proof.node,
                            step_proof.span,
                        )?;
                        obligations.extend(step_obls);
                    } else {
                        let obl = self.create_obligation(
                            &full_name,
                            assertion.clone(),
                            &assumptions,
                            assertion.span,
                        );
                        obligations.push(obl);
                    }

                    // Add this step as a fact for subsequent steps
                    step_facts.push((step_name, assertion.clone()));
                }
                ProofStepKind::Suffices(assertion, proof) => {
                    // SUFFICES - if we can prove assertion, then goal follows
                    // Two obligations: (1) prove assertion => goal, (2) prove assertion

                    // First: prove assertion => goal
                    let impl_goal = Spanned::new(
                        Expr::Implies(Box::new(assertion.clone()), Box::new(goal.clone())),
                        assertion.span,
                    );
                    let impl_obl = self.create_obligation(
                        &format!("{}_impl", full_name),
                        impl_goal,
                        &[],
                        assertion.span,
                    );
                    obligations.push(impl_obl);

                    // Second: prove the assertion
                    if let Some(step_proof) = proof {
                        let step_obls = self.extract_proof_obligations(
                            &full_name,
                            assertion,
                            &step_proof.node,
                            step_proof.span,
                        )?;
                        obligations.extend(step_obls);
                    } else {
                        let obl = self.create_obligation(
                            &full_name,
                            assertion.clone(),
                            &[],
                            assertion.span,
                        );
                        obligations.push(obl);
                    }
                }
                ProofStepKind::Have(assertion) => {
                    // HAVE - assertion must follow from context (implicit proof)
                    let assumptions: Vec<Spanned<Expr>> =
                        step_facts.iter().map(|(_, e)| e.clone()).collect();

                    let obl = self.create_obligation(
                        &full_name,
                        assertion.clone(),
                        &assumptions,
                        assertion.span,
                    );
                    obligations.push(obl);

                    step_facts.push((step_name, assertion.clone()));
                }
                ProofStepKind::Take(_bounds) => {
                    // TAKE x \in S - introduces universally quantified variable
                    // This modifies the proof context, no immediate obligation
                    // For now, we just record that we've taken these variables
                }
                ProofStepKind::Witness(_exprs) => {
                    // WITNESS e - provides witness for existential
                    // This affects the proof context
                }
                ProofStepKind::Pick(bounds, assertion, _proof) => {
                    // PICK x \in S : P - choose witness satisfying P
                    // Creates obligation to prove \E x \in S : P
                    let existential = self.make_existential(bounds, assertion);
                    let assumptions: Vec<Spanned<Expr>> =
                        step_facts.iter().map(|(_, e)| e.clone()).collect();

                    let obl = self.create_obligation(
                        &full_name,
                        existential,
                        &assumptions,
                        assertion.span,
                    );
                    obligations.push(obl);

                    // The assertion with bound variables becomes a fact
                    step_facts.push((step_name, assertion.clone()));
                }
                ProofStepKind::UseOrHide { use_: _, facts: _ } => {
                    // USE/HIDE - modifies what facts are available
                }
                ProofStepKind::Define(_defs) => {
                    // DEFINE - introduces local definitions
                    // These are available for subsequent steps
                }
                ProofStepKind::Qed(proof) => {
                    // QED - final step, must prove the goal
                    let assumptions: Vec<Spanned<Expr>> =
                        step_facts.iter().map(|(_, e)| e.clone()).collect();

                    if let Some(qed_proof) = proof {
                        let step_obls = self.extract_proof_obligations(
                            &format!("{}_qed", name),
                            goal,
                            &qed_proof.node,
                            qed_proof.span,
                        )?;
                        obligations.extend(step_obls);
                    } else {
                        let obl = self.create_obligation(
                            &format!("{}_qed", name),
                            goal.clone(),
                            &assumptions,
                            goal.span,
                        );
                        obligations.push(obl);
                    }
                }
            }
        }

        Ok(obligations)
    }

    /// Resolve proof hints to expressions
    fn resolve_hints(&self, hints: &[ProofHint]) -> ProofResult<Vec<Spanned<Expr>>> {
        let mut result = Vec::new();

        for hint in hints {
            match hint {
                ProofHint::Ref(name) => {
                    // Look up fact by name
                    if let Some((_, expr)) = self.facts.iter().find(|(n, _)| n == &name.node) {
                        result.push(expr.clone());
                    }
                    // If not found, we don't error - the backend will handle it
                }
                ProofHint::Def(_names) => {
                    // DEF names - these are definitions to expand
                    // We don't add them as assumptions, they modify how the goal is interpreted
                }
                ProofHint::Module(_) => {
                    // Module reference - would need module resolution
                }
            }
        }

        Ok(result)
    }

    /// Create an existential from bounds and predicate
    fn make_existential(&self, bounds: &[BoundVar], pred: &Spanned<Expr>) -> Spanned<Expr> {
        Spanned::new(
            Expr::Exists(bounds.to_vec(), Box::new(pred.clone())),
            pred.span,
        )
    }

    /// Create an obligation with a fingerprint
    fn create_obligation(
        &self,
        name: &str,
        goal: Spanned<Expr>,
        assumptions: &[Spanned<Expr>],
        span: Span,
    ) -> Obligation {
        // Create fingerprint from goal and assumptions
        let mut data = Vec::new();
        data.extend_from_slice(name.as_bytes());
        data.extend_from_slice(format!("{:?}", goal.node).as_bytes());
        for assumption in assumptions {
            data.extend_from_slice(format!("{:?}", assumption.node).as_bytes());
        }

        let id = ObligationId::from_fingerprint(&data);

        Obligation {
            id,
            goal,
            assumptions: assumptions.to_vec(),
            definitions: self.definitions.iter().map(|(n, _)| n.clone()).collect(),
            span,
            description: name.to_string(),
        }
    }
}

impl Default for ObligationExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper to convert fingerprint bytes to hex string
mod hex {
    pub fn encode(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{:02x}", b)).collect()
    }
}
