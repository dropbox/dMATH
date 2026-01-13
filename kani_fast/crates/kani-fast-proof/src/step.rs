//! Proof steps for different verification backends
//!
//! Each backend produces different proof artifacts:
//! - SAT solvers: DRAT clauses
//! - SMT solvers: Alethe-style inference steps
//! - CHC solvers: Invariant predicates
//! - Lean: Proof terms

use serde::{Deserialize, Serialize};
use std::fmt::Write;

/// A single step in a proof
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProofStep {
    /// SAT proof step (DRAT format)
    Drat(DratStep),
    /// SMT proof step (Alethe-style)
    Smt(SmtStep),
    /// CHC proof step (invariant discovery)
    Chc(ChcStep),
    /// Lean proof step (proof term)
    Lean(LeanStep),
    /// Trust step (skip verification, just trust the result)
    Trust {
        /// Justification for trusting this step
        reason: String,
        /// The claim being trusted
        claim: String,
    },
}

impl ProofStep {
    /// Create a DRAT addition step
    pub fn drat_add(literals: Vec<i32>) -> Self {
        Self::Drat(DratStep::Add(literals))
    }

    /// Create a DRAT deletion step
    pub fn drat_delete(literals: Vec<i32>) -> Self {
        Self::Drat(DratStep::Delete(literals))
    }

    /// Create an SMT inference step
    pub fn smt_infer(
        rule: impl Into<String>,
        premises: Vec<usize>,
        conclusion: impl Into<String>,
    ) -> Self {
        Self::Smt(SmtStep::Inference {
            rule: rule.into(),
            premises,
            conclusion: conclusion.into(),
        })
    }

    /// Create a CHC invariant step
    pub fn chc_invariant(
        name: impl Into<String>,
        params: Vec<String>,
        formula: impl Into<String>,
    ) -> Self {
        Self::Chc(ChcStep::Invariant {
            name: name.into(),
            params,
            formula: formula.into(),
        })
    }

    /// Create a Lean proof term step
    pub fn lean_term(term: impl Into<String>) -> Self {
        Self::Lean(LeanStep::Term(term.into()))
    }

    /// Create a Lean tactic step
    pub fn lean_tactic(tactic: impl Into<String>) -> Self {
        Self::Lean(LeanStep::Tactic(tactic.into()))
    }

    /// Create a trust step
    pub fn trust(reason: impl Into<String>, claim: impl Into<String>) -> Self {
        Self::Trust {
            reason: reason.into(),
            claim: claim.into(),
        }
    }

    /// Check if this is a trust step
    pub fn is_trust(&self) -> bool {
        matches!(self, Self::Trust { .. })
    }

    /// Get the proof format for this step
    pub fn format(&self) -> StepFormat {
        match self {
            Self::Drat(_) => StepFormat::Drat,
            Self::Smt(_) => StepFormat::Smt,
            Self::Chc(_) => StepFormat::Chc,
            Self::Lean(_) => StepFormat::Lean,
            Self::Trust { .. } => StepFormat::Trust,
        }
    }
}

/// Format of a proof step
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepFormat {
    Drat,
    Smt,
    Chc,
    Lean,
    Trust,
}

/// DRAT (Deletion Resolution Asymmetric Tautology) proof step
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DratStep {
    /// Add a clause (learned)
    Add(Vec<i32>),
    /// Delete a clause
    Delete(Vec<i32>),
}

impl DratStep {
    /// Get the literals in this step
    pub fn literals(&self) -> &[i32] {
        match self {
            Self::Add(lits) | Self::Delete(lits) => lits,
        }
    }

    /// Check if this is an addition
    pub fn is_add(&self) -> bool {
        matches!(self, Self::Add(_))
    }

    /// Check if this is a deletion
    pub fn is_delete(&self) -> bool {
        matches!(self, Self::Delete(_))
    }

    /// Convert to DRAT text format
    pub fn to_drat_text(&self) -> String {
        match self {
            Self::Add(lits) => {
                let mut s = String::with_capacity(lits.len() * 4 + 1);
                for l in lits {
                    let _ = write!(s, "{l} ");
                }
                s.push('0');
                s
            }
            Self::Delete(lits) => {
                let mut s = String::with_capacity(lits.len() * 4 + 3);
                s.push_str("d ");
                for l in lits {
                    let _ = write!(s, "{l} ");
                }
                s.push('0');
                s
            }
        }
    }
}

/// SMT proof step (Alethe-style)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SmtStep {
    /// Assumption/axiom
    Assume {
        /// Name of the assumption
        name: String,
        /// The assumed formula
        formula: String,
    },
    /// Inference step
    Inference {
        /// Rule name (e.g., "resolution", "congruence", "refl")
        rule: String,
        /// Indices of premise steps
        premises: Vec<usize>,
        /// Conclusion formula
        conclusion: String,
    },
    /// Theory lemma
    TheoryLemma {
        /// Theory name (e.g., "LIA", "UF", "BV")
        theory: String,
        /// The lemma formula
        formula: String,
    },
}

impl SmtStep {
    /// Create an assumption
    pub fn assume(name: impl Into<String>, formula: impl Into<String>) -> Self {
        Self::Assume {
            name: name.into(),
            formula: formula.into(),
        }
    }

    /// Create an inference
    pub fn infer(
        rule: impl Into<String>,
        premises: Vec<usize>,
        conclusion: impl Into<String>,
    ) -> Self {
        Self::Inference {
            rule: rule.into(),
            premises,
            conclusion: conclusion.into(),
        }
    }

    /// Create a theory lemma
    pub fn theory_lemma(theory: impl Into<String>, formula: impl Into<String>) -> Self {
        Self::TheoryLemma {
            theory: theory.into(),
            formula: formula.into(),
        }
    }
}

/// CHC (Constrained Horn Clauses) proof step
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChcStep {
    /// Invariant predicate definition
    Invariant {
        /// Predicate name
        name: String,
        /// Parameter names
        params: Vec<String>,
        /// Formula in SMT-LIB2 format
        formula: String,
    },
    /// Initiation proof (Init -> Inv)
    Initiation {
        /// Initial condition
        init: String,
        /// Invariant formula
        invariant: String,
    },
    /// Consecution proof (Inv ∧ Trans -> Inv')
    Consecution {
        /// Pre-state invariant
        pre_invariant: String,
        /// Transition relation
        transition: String,
        /// Post-state invariant
        post_invariant: String,
    },
    /// Property proof (Inv -> Property)
    Property {
        /// Invariant formula
        invariant: String,
        /// Property formula
        property: String,
    },
}

impl ChcStep {
    /// Create an invariant definition
    pub fn invariant(
        name: impl Into<String>,
        params: Vec<String>,
        formula: impl Into<String>,
    ) -> Self {
        Self::Invariant {
            name: name.into(),
            params,
            formula: formula.into(),
        }
    }

    /// Create an initiation step
    pub fn initiation(init: impl Into<String>, invariant: impl Into<String>) -> Self {
        Self::Initiation {
            init: init.into(),
            invariant: invariant.into(),
        }
    }

    /// Create a consecution step
    pub fn consecution(
        pre_invariant: impl Into<String>,
        transition: impl Into<String>,
        post_invariant: impl Into<String>,
    ) -> Self {
        Self::Consecution {
            pre_invariant: pre_invariant.into(),
            transition: transition.into(),
            post_invariant: post_invariant.into(),
        }
    }

    /// Create a property step
    pub fn property(invariant: impl Into<String>, property: impl Into<String>) -> Self {
        Self::Property {
            invariant: invariant.into(),
            property: property.into(),
        }
    }
}

/// Lean proof step
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LeanStep {
    /// Raw proof term (lambda calculus expression)
    Term(String),
    /// Tactic application
    Tactic(String),
    /// Import statement
    Import(String),
    /// Variable declaration
    Variable { name: String, ty: String },
    /// Theorem statement
    Theorem {
        name: String,
        statement: String,
        proof: Vec<LeanStep>,
    },
}

impl LeanStep {
    /// Create a term step
    pub fn term(t: impl Into<String>) -> Self {
        Self::Term(t.into())
    }

    /// Create a tactic step
    pub fn tactic(t: impl Into<String>) -> Self {
        Self::Tactic(t.into())
    }

    /// Create an import step
    pub fn import(module: impl Into<String>) -> Self {
        Self::Import(module.into())
    }

    /// Create a variable declaration
    pub fn variable(name: impl Into<String>, ty: impl Into<String>) -> Self {
        Self::Variable {
            name: name.into(),
            ty: ty.into(),
        }
    }

    /// Create a theorem
    pub fn theorem(
        name: impl Into<String>,
        statement: impl Into<String>,
        proof: Vec<LeanStep>,
    ) -> Self {
        Self::Theorem {
            name: name.into(),
            statement: statement.into(),
            proof,
        }
    }

    /// Convert to Lean 5 source code
    pub fn to_lean5(&self) -> String {
        match self {
            Self::Term(t) => t.clone(),
            Self::Tactic(t) => t.clone(),
            Self::Import(m) => format!("import {m}"),
            Self::Variable { name, ty } => format!("variable ({name} : {ty})"),
            Self::Theorem {
                name,
                statement,
                proof,
            } => {
                let mut s = format!("theorem {name} : {statement} := by\n");
                for step in proof {
                    s.push_str("  ");
                    s.push_str(&step.to_lean5());
                    s.push('\n');
                }
                s
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== ProofStep Tests ====================

    #[test]
    fn test_proof_step_drat_add() {
        let step = ProofStep::drat_add(vec![1, 2, 3]);
        assert!(matches!(step, ProofStep::Drat(DratStep::Add(_))));
        assert_eq!(step.format(), StepFormat::Drat);
    }

    #[test]
    fn test_proof_step_drat_delete() {
        let step = ProofStep::drat_delete(vec![1, 2]);
        assert!(matches!(step, ProofStep::Drat(DratStep::Delete(_))));
    }

    #[test]
    fn test_proof_step_smt_infer() {
        let step = ProofStep::smt_infer("resolution", vec![0, 1], "(or a b)");
        assert!(matches!(step, ProofStep::Smt(SmtStep::Inference { .. })));
        assert_eq!(step.format(), StepFormat::Smt);
    }

    #[test]
    fn test_proof_step_chc_invariant() {
        let step = ProofStep::chc_invariant("inv", vec!["x".to_string()], "(>= x 0)");
        assert!(matches!(step, ProofStep::Chc(ChcStep::Invariant { .. })));
        assert_eq!(step.format(), StepFormat::Chc);
    }

    #[test]
    fn test_proof_step_lean_term() {
        let step = ProofStep::lean_term("fun x => x");
        assert!(matches!(step, ProofStep::Lean(LeanStep::Term(_))));
        assert_eq!(step.format(), StepFormat::Lean);
    }

    #[test]
    fn test_proof_step_lean_tactic() {
        let step = ProofStep::lean_tactic("omega");
        assert!(matches!(step, ProofStep::Lean(LeanStep::Tactic(_))));
    }

    #[test]
    fn test_proof_step_trust() {
        let step = ProofStep::trust("external prover", "(= 1 1)");
        assert!(step.is_trust());
        assert_eq!(step.format(), StepFormat::Trust);
    }

    #[test]
    fn test_proof_step_not_trust() {
        let step = ProofStep::drat_add(vec![1]);
        assert!(!step.is_trust());
    }

    #[test]
    fn test_proof_step_serialize() {
        let step = ProofStep::chc_invariant("inv", vec!["x".to_string()], "(>= x 0)");
        let json = serde_json::to_string(&step).unwrap();
        let parsed: ProofStep = serde_json::from_str(&json).unwrap();
        assert_eq!(step, parsed);
    }

    #[test]
    fn test_proof_step_clone() {
        let step = ProofStep::drat_add(vec![1, 2, 3]);
        let cloned = step.clone();
        assert_eq!(step, cloned);
    }

    // ==================== DratStep Tests ====================

    #[test]
    fn test_drat_step_add() {
        let step = DratStep::Add(vec![1, 2, 3]);
        assert!(step.is_add());
        assert!(!step.is_delete());
        assert_eq!(step.literals(), &[1, 2, 3]);
    }

    #[test]
    fn test_drat_step_delete() {
        let step = DratStep::Delete(vec![1, 2]);
        assert!(step.is_delete());
        assert!(!step.is_add());
        assert_eq!(step.literals(), &[1, 2]);
    }

    #[test]
    fn test_drat_step_to_text_add() {
        let step = DratStep::Add(vec![1, -2, 3]);
        assert_eq!(step.to_drat_text(), "1 -2 3 0");
    }

    #[test]
    fn test_drat_step_to_text_delete() {
        let step = DratStep::Delete(vec![1, 2]);
        assert_eq!(step.to_drat_text(), "d 1 2 0");
    }

    #[test]
    fn test_drat_step_empty() {
        let step = DratStep::Add(vec![]);
        assert_eq!(step.to_drat_text(), "0");
    }

    #[test]
    fn test_drat_step_serialize() {
        let step = DratStep::Add(vec![1, 2, 3]);
        let json = serde_json::to_string(&step).unwrap();
        let parsed: DratStep = serde_json::from_str(&json).unwrap();
        assert_eq!(step, parsed);
    }

    // ==================== SmtStep Tests ====================

    #[test]
    fn test_smt_step_assume() {
        let step = SmtStep::assume("a1", "(> x 0)");
        assert!(matches!(step, SmtStep::Assume { .. }));
    }

    #[test]
    fn test_smt_step_infer() {
        let step = SmtStep::infer("resolution", vec![0, 1], "(or a b)");
        if let SmtStep::Inference {
            rule,
            premises,
            conclusion,
        } = step
        {
            assert_eq!(rule, "resolution");
            assert_eq!(premises, vec![0, 1]);
            assert_eq!(conclusion, "(or a b)");
        } else {
            panic!("Expected Inference");
        }
    }

    #[test]
    fn test_smt_step_theory_lemma() {
        let step = SmtStep::theory_lemma("LIA", "(>= (+ x 1) 1)");
        if let SmtStep::TheoryLemma { theory, formula } = step {
            assert_eq!(theory, "LIA");
            assert_eq!(formula, "(>= (+ x 1) 1)");
        } else {
            panic!("Expected TheoryLemma");
        }
    }

    #[test]
    fn test_smt_step_serialize() {
        let step = SmtStep::infer("mp", vec![0], "Q");
        let json = serde_json::to_string(&step).unwrap();
        let parsed: SmtStep = serde_json::from_str(&json).unwrap();
        assert_eq!(step, parsed);
    }

    // ==================== ChcStep Tests ====================

    #[test]
    fn test_chc_step_invariant() {
        let step = ChcStep::invariant(
            "inv",
            vec!["x".to_string(), "y".to_string()],
            "(and (>= x 0) (>= y 0))",
        );
        if let ChcStep::Invariant {
            name,
            params,
            formula,
        } = step
        {
            assert_eq!(name, "inv");
            assert_eq!(params, vec!["x", "y"]);
            assert!(formula.contains("and"));
        } else {
            panic!("Expected Invariant");
        }
    }

    #[test]
    fn test_chc_step_initiation() {
        let step = ChcStep::initiation("(= x 0)", "(>= x 0)");
        if let ChcStep::Initiation { init, invariant } = step {
            assert_eq!(init, "(= x 0)");
            assert_eq!(invariant, "(>= x 0)");
        } else {
            panic!("Expected Initiation");
        }
    }

    #[test]
    fn test_chc_step_consecution() {
        let step = ChcStep::consecution("(>= x 0)", "(= x' (+ x 1))", "(>= x' 0)");
        if let ChcStep::Consecution {
            pre_invariant,
            transition,
            post_invariant,
        } = step
        {
            assert_eq!(pre_invariant, "(>= x 0)");
            assert!(transition.contains("x'"));
            assert!(post_invariant.contains("x'"));
        } else {
            panic!("Expected Consecution");
        }
    }

    #[test]
    fn test_chc_step_property() {
        let step = ChcStep::property("(>= x 0)", "(not (< x 0))");
        if let ChcStep::Property {
            invariant,
            property,
        } = step
        {
            assert_eq!(invariant, "(>= x 0)");
            assert!(property.contains("not"));
        } else {
            panic!("Expected Property");
        }
    }

    #[test]
    fn test_chc_step_serialize() {
        let step = ChcStep::invariant("inv", vec!["x".to_string()], "(>= x 0)");
        let json = serde_json::to_string(&step).unwrap();
        let parsed: ChcStep = serde_json::from_str(&json).unwrap();
        assert_eq!(step, parsed);
    }

    // ==================== LeanStep Tests ====================

    #[test]
    fn test_lean_step_term() {
        let step = LeanStep::term("fun x => x");
        if let LeanStep::Term(t) = &step {
            assert_eq!(t, "fun x => x");
        } else {
            panic!("Expected Term");
        }
        assert_eq!(step.to_lean5(), "fun x => x");
    }

    #[test]
    fn test_lean_step_tactic() {
        let step = LeanStep::tactic("omega");
        assert_eq!(step.to_lean5(), "omega");
    }

    #[test]
    fn test_lean_step_import() {
        let step = LeanStep::import("Mathlib.Tactic");
        assert_eq!(step.to_lean5(), "import Mathlib.Tactic");
    }

    #[test]
    fn test_lean_step_variable() {
        let step = LeanStep::variable("x", "Int");
        assert_eq!(step.to_lean5(), "variable (x : Int)");
    }

    #[test]
    fn test_lean_step_theorem() {
        let step = LeanStep::theorem("test_thm", "x >= 0", vec![LeanStep::tactic("omega")]);
        let lean = step.to_lean5();
        assert!(lean.contains("theorem test_thm : x >= 0 := by"));
        assert!(lean.contains("omega"));
    }

    #[test]
    fn test_lean_step_serialize() {
        let step = LeanStep::tactic("simp");
        let json = serde_json::to_string(&step).unwrap();
        let parsed: LeanStep = serde_json::from_str(&json).unwrap();
        assert_eq!(step, parsed);
    }

    #[test]
    fn test_lean_step_clone() {
        let step = LeanStep::term("id");
        let cloned = step.clone();
        assert_eq!(step, cloned);
    }

    // ==================== StepFormat Tests ====================

    #[test]
    fn test_step_format_serialize() {
        let format = StepFormat::Chc;
        let json = serde_json::to_string(&format).unwrap();
        let parsed: StepFormat = serde_json::from_str(&json).unwrap();
        assert_eq!(format, parsed);
    }

    #[test]
    fn test_step_format_clone() {
        let format = StepFormat::Lean;
        let cloned = format;
        assert_eq!(format, cloned);
    }

    #[test]
    fn test_step_format_debug() {
        let format = StepFormat::Drat;
        let debug = format!("{:?}", format);
        assert_eq!(debug, "Drat");
    }
}
