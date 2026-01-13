//! Proof representation for Z4
//!
//! Proofs can be produced for unsatisfiable formulas.
//! Supports export to Alethe format for independent verification.
//!
//! ## Alethe Proof Format
//!
//! The Alethe format (used by carcara proof checker) has three main commands:
//! - `assume`: Input assertions from the problem
//! - `step`: Proof steps with a rule name, premises, and conclusion clause
//! - `anchor`: Subproofs (for nested reasoning)
//!
//! Example Alethe proof:
//! ```text
//! (assume h1 (= a b))
//! (assume h2 (= b c))
//! (step t1 (cl (= a c)) :rule trans :premises (h1 h2))
//! (step t2 (cl (not (= a c)) (= a c)) :rule equiv_pos1 :premises (t1))
//! ```

use crate::term::TermId;

/// A proof step (Alethe-compatible)
#[derive(Debug, Clone)]
pub enum ProofStep {
    /// Input assertion from the problem
    Assume(TermId),

    /// Resolution inference (SAT solver)
    Resolution {
        /// Pivot literal (resolved on)
        pivot: TermId,
        /// First clause premise
        clause1: ProofId,
        /// Second clause premise
        clause2: ProofId,
    },

    /// Theory lemma (from theory solver)
    TheoryLemma {
        /// Theory name (e.g., "EUF", "LRA", "LIA", "BV")
        theory: String,
        /// The lemma clause (disjunction of literals)
        clause: Vec<TermId>,
    },

    /// Generic proof step (Alethe-style)
    Step {
        /// The rule name (e.g., "trans", "cong", "and", "resolution")
        rule: AletheRule,
        /// The conclusion clause (disjunction of literals)
        clause: Vec<TermId>,
        /// Premise step IDs
        premises: Vec<ProofId>,
        /// Additional arguments (rule-specific)
        args: Vec<TermId>,
    },

    /// Subproof anchor (start of nested proof)
    Anchor {
        /// The step that ends this subproof
        end_step: ProofId,
        /// Variables introduced in this subproof
        variables: Vec<(String, crate::sort::Sort)>,
    },
}

/// Alethe proof rules
///
/// These rules correspond to the rules supported by carcara.
/// See: <https://github.com/ufmg-smite/carcara>
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AletheRule {
    // === Boolean rules ===
    /// True introduction
    True,
    /// False elimination
    False,
    /// Negation of true
    NotTrue,
    /// Negation of false
    NotFalse,
    /// And introduction
    And,
    /// And elimination (position i)
    AndPos(u32),
    /// And negation
    AndNeg,
    /// Not-and
    NotAnd,
    /// Or introduction
    Or,
    /// Or elimination (position i)
    OrPos(u32),
    /// Or negation
    OrNeg,
    /// Not-or
    NotOr,
    /// Implication introduction
    Implies,
    /// Implication negation 1
    ImpliesNeg1,
    /// Implication negation 2
    ImpliesNeg2,
    /// Not-implies 1
    NotImplies1,
    /// Not-implies 2
    NotImplies2,
    /// Equivalence introduction
    Equiv,
    /// Equivalence positive 1
    EquivPos1,
    /// Equivalence positive 2
    EquivPos2,
    /// Equivalence negative 1
    EquivNeg1,
    /// Equivalence negative 2
    EquivNeg2,
    /// Not-equivalence 1
    NotEquiv1,
    /// Not-equivalence 2
    NotEquiv2,
    /// ITE introduction
    Ite,
    /// ITE positive 1
    ItePos1,
    /// ITE positive 2
    ItePos2,
    /// ITE negative 1
    IteNeg1,
    /// ITE negative 2
    IteNeg2,
    /// Not-ITE 1
    NotIte1,
    /// Not-ITE 2
    NotIte2,

    // === Resolution ===
    /// Propositional resolution
    Resolution,
    /// Theory resolution (resolution on theory literals)
    ThResolution,
    /// Contraction (remove duplicate literals)
    Contraction,

    // === Equality ===
    /// Reflexivity: t = t
    Refl,
    /// Symmetry: a = b => b = a
    Symm,
    /// Transitivity: a = b, b = c => a = c
    Trans,
    /// Congruence: f(a) = f(b) if a = b
    Cong,
    /// Equality reflexivity (eq_reflexive)
    EqReflexive,
    /// Equality transitive
    EqTransitive,
    /// Equality congruent
    EqCongruent,
    /// Equality congruent predicate
    EqCongruentPred,

    // === Arithmetic ===
    /// Linear arithmetic tautology
    LaTautology,
    /// Linear arithmetic generic
    LaGeneric,
    /// Linear arithmetic disequality
    LaDisequality,
    /// Linear arithmetic totality
    LaTotality,
    /// Multiply by positive
    LaMultPos,
    /// Multiply by negative
    LaMultNeg,
    /// Linear integer arithmetic generic (SMT calls LIA solver)
    LiaGeneric,

    // === Quantifiers ===
    /// Forall instantiation
    ForallInst,
    /// Skolemization
    Skolem,

    // === Subproof rules ===
    /// Subproof (nested proof)
    Subproof,
    /// Bind (variable binding)
    Bind,

    // === Simplification ===
    /// Generic simplification
    AllSimplify,
    /// Boolean simplification
    BoolSimplify,
    /// Arithmetic simplification
    ArithSimplify,

    // === Special ===
    /// Hole (placeholder, should be elaborated)
    Hole,
    /// DRUP (clause addition verified by unit propagation)
    Drup,
    /// Trust (unverified step)
    Trust,
    /// Custom rule (extension)
    Custom(String),
}

impl AletheRule {
    /// Get the Alethe rule name as a string
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            AletheRule::True => "true",
            AletheRule::False => "false",
            AletheRule::NotTrue => "not_true",
            AletheRule::NotFalse => "not_false",
            AletheRule::And => "and",
            AletheRule::AndPos(_) => "and_pos",
            AletheRule::AndNeg => "and_neg",
            AletheRule::NotAnd => "not_and",
            AletheRule::Or => "or",
            AletheRule::OrPos(_) => "or_pos",
            AletheRule::OrNeg => "or_neg",
            AletheRule::NotOr => "not_or",
            AletheRule::Implies => "implies",
            AletheRule::ImpliesNeg1 => "implies_neg1",
            AletheRule::ImpliesNeg2 => "implies_neg2",
            AletheRule::NotImplies1 => "not_implies1",
            AletheRule::NotImplies2 => "not_implies2",
            AletheRule::Equiv => "equiv",
            AletheRule::EquivPos1 => "equiv_pos1",
            AletheRule::EquivPos2 => "equiv_pos2",
            AletheRule::EquivNeg1 => "equiv_neg1",
            AletheRule::EquivNeg2 => "equiv_neg2",
            AletheRule::NotEquiv1 => "not_equiv1",
            AletheRule::NotEquiv2 => "not_equiv2",
            AletheRule::Ite => "ite",
            AletheRule::ItePos1 => "ite_pos1",
            AletheRule::ItePos2 => "ite_pos2",
            AletheRule::IteNeg1 => "ite_neg1",
            AletheRule::IteNeg2 => "ite_neg2",
            AletheRule::NotIte1 => "not_ite1",
            AletheRule::NotIte2 => "not_ite2",
            AletheRule::Resolution => "resolution",
            AletheRule::ThResolution => "th_resolution",
            AletheRule::Contraction => "contraction",
            AletheRule::Refl => "refl",
            AletheRule::Symm => "symm",
            AletheRule::Trans => "trans",
            AletheRule::Cong => "cong",
            AletheRule::EqReflexive => "eq_reflexive",
            AletheRule::EqTransitive => "eq_transitive",
            AletheRule::EqCongruent => "eq_congruent",
            AletheRule::EqCongruentPred => "eq_congruent_pred",
            AletheRule::LaTautology => "la_tautology",
            AletheRule::LaGeneric => "la_generic",
            AletheRule::LaDisequality => "la_disequality",
            AletheRule::LaTotality => "la_totality",
            AletheRule::LaMultPos => "la_mult_pos",
            AletheRule::LaMultNeg => "la_mult_neg",
            AletheRule::LiaGeneric => "lia_generic",
            AletheRule::ForallInst => "forall_inst",
            AletheRule::Skolem => "sko_forall",
            AletheRule::Subproof => "subproof",
            AletheRule::Bind => "bind",
            AletheRule::AllSimplify => "all_simplify",
            AletheRule::BoolSimplify => "bool_simplify",
            AletheRule::ArithSimplify => "arith_simplify",
            AletheRule::Hole => "hole",
            AletheRule::Drup => "drup",
            AletheRule::Trust => "trust",
            AletheRule::Custom(name) => name,
        }
    }
}

impl std::fmt::Display for AletheRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Proof step identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProofId(pub u32);

impl std::fmt::Display for ProofId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "t{}", self.0)
    }
}

/// A complete proof (Alethe-compatible)
#[derive(Debug, Clone, Default)]
pub struct Proof {
    /// Proof steps
    pub steps: Vec<ProofStep>,
    /// Named step IDs (for assume commands)
    pub named_steps: std::collections::HashMap<String, ProofId>,
}

impl Proof {
    /// Create a new empty proof
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a proof step
    #[allow(clippy::cast_possible_truncation)] // Proof step count is bounded well under u32::MAX
    pub fn add_step(&mut self, step: ProofStep) -> ProofId {
        let id = ProofId(self.steps.len() as u32);
        self.steps.push(step);
        id
    }

    /// Add an assumption and optionally name it
    pub fn add_assume(&mut self, term: TermId, name: Option<String>) -> ProofId {
        let id = self.add_step(ProofStep::Assume(term));
        if let Some(n) = name {
            self.named_steps.insert(n, id);
        }
        id
    }

    /// Add a generic step with a rule
    pub fn add_rule_step(
        &mut self,
        rule: AletheRule,
        clause: Vec<TermId>,
        premises: Vec<ProofId>,
        args: Vec<TermId>,
    ) -> ProofId {
        self.add_step(ProofStep::Step {
            rule,
            clause,
            premises,
            args,
        })
    }

    /// Add a resolution step
    pub fn add_resolution(&mut self, pivot: TermId, clause1: ProofId, clause2: ProofId) -> ProofId {
        self.add_step(ProofStep::Resolution {
            pivot,
            clause1,
            clause2,
        })
    }

    /// Add a theory lemma
    pub fn add_theory_lemma(&mut self, theory: impl Into<String>, clause: Vec<TermId>) -> ProofId {
        self.add_step(ProofStep::TheoryLemma {
            theory: theory.into(),
            clause,
        })
    }

    /// Get a step by ID
    #[must_use]
    pub fn get_step(&self, id: ProofId) -> Option<&ProofStep> {
        self.steps.get(id.0 as usize)
    }

    /// Get the number of steps
    #[must_use]
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Check if the proof is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}
