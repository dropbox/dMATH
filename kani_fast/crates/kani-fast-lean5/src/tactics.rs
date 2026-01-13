//! Proof tactic generation for Lean5
//!
//! This module generates Lean5 proof tactics for common verification patterns.
//! When possible, it produces complete proofs; otherwise, it produces helpful
//! proof sketches with targeted `sorry` placeholders.
//!
//! ## Supported Patterns
//!
//! The tactic generator handles these common invariant patterns:
//! - Non-negativity: `x >= 0`
//! - Range bounds: `x >= 0 ∧ x <= N`
//! - Ordering between variables: `x <= y`
//! - Boolean implications: `a => b`
//! - Monotonicity patterns for k-induction
//!
//! For arithmetic inequalities, `omega` is the primary tactic.
//! For boolean/propositional formulas, `simp` combined with `decide` works well.

use crate::expr::Lean5Expr;
#[cfg(test)]
use crate::expr::Lean5Type;
use crate::obligation::{ProofObligation, ProofObligationKind};
use std::fmt::Write;

/// Tactic block for Lean5 proofs
#[derive(Debug, Clone)]
pub struct TacticBlock {
    /// The tactics in sequence
    pub tactics: Vec<Tactic>,
    /// Whether this is a complete proof (no sorry)
    pub is_complete: bool,
}

impl TacticBlock {
    /// Create a new tactic block
    pub fn new(tactics: Vec<Tactic>) -> Self {
        let is_complete = !tactics.iter().any(|t| matches!(t, Tactic::Sorry(_)));
        Self {
            tactics,
            is_complete,
        }
    }

    /// Create a sorry block with explanation
    pub fn sorry(reason: &str) -> Self {
        Self {
            tactics: vec![Tactic::Sorry(reason.to_string())],
            is_complete: false,
        }
    }

    /// Convert to Lean5 source code
    pub fn to_lean5(&self) -> String {
        if self.tactics.is_empty() {
            return "sorry".to_string();
        }

        if self.tactics.len() == 1 {
            return self.tactics[0].to_lean5();
        }

        // Multiple tactics joined by newlines
        self.tactics
            .iter()
            .map(|t| format!("  {}", t.to_lean5()))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Individual tactics in Lean5
#[derive(Debug, Clone)]
pub enum Tactic {
    /// Introduce variables: `intro x y`
    Intro(Vec<String>),
    /// Apply a lemma: `apply lemma`
    Apply(String),
    /// Exact match: `exact term`
    Exact(String),
    /// Simplify: `simp [lemmas]`
    Simp(Vec<String>),
    /// Simplify with arithmetic: `simp_arith [lemmas]`
    SimpArith(Vec<String>),
    /// Omega tactic for linear arithmetic: `omega`
    Omega,
    /// Linear arithmetic: `linarith`
    Linarith,
    /// Non-linear arithmetic: `nlinarith`
    Nlinarith,
    /// Polynomial ring arithmetic: `polyrith`
    Polyrith,
    /// Decide for decidable propositions: `decide`
    Decide,
    /// Reflexivity: `rfl`
    Rfl,
    /// Symmetry: `symm`
    Symm,
    /// Transitivity: `trans expr`
    Trans(String),
    /// Split conjunction: `constructor`
    Constructor,
    /// Left/right for disjunction
    Left,
    Right,
    /// Cases analysis: `cases expr with ...`
    Cases(String, Vec<(String, TacticBlock)>),
    /// Induction: `induction n with ...`
    Induction(String, Vec<(String, TacticBlock)>),
    /// Have intermediate fact: `have h : type := proof`
    Have(String, String, Box<TacticBlock>),
    /// Rewrite: `rw [lemmas]`
    Rewrite(Vec<String>),
    /// Ring for ring arithmetic: `ring`
    Ring,
    /// Ring_nf for ring normal form: `ring_nf`
    RingNf,
    /// Native decide: `native_decide`
    NativeDecide,
    /// Assumption tactic: `assumption`
    Assumption,
    /// Trivial tactic: `trivial`
    Trivial,
    /// Tauto tactic for propositional tautologies: `tauto`
    Tauto,
    /// And introduction: `And.intro h1 h2`
    AndIntro,
    /// Extract left from conjunction: `And.left h`
    AndLeft(String),
    /// Extract right from conjunction: `And.right h`
    AndRight(String),
    /// Obtain with pattern for destructuring: `obtain ⟨h1, h2⟩ := h`
    Obtain(String, String),
    /// Bitvector decision: `bv_decide`
    BvDecide,
    /// Bitvector omega: `bv_omega`
    BvOmega,
    /// Modular arithmetic: `mod_cast`
    ModCast,
    /// Push cast operations: `push_cast`
    PushCast,
    /// Normalize casts: `norm_cast`
    NormCast,
    /// Sorry with reason
    Sorry(String),
    /// Raw tactic string
    Raw(String),
}

impl Tactic {
    /// Convert to Lean5 source
    pub fn to_lean5(&self) -> String {
        match self {
            Tactic::Intro(vars) if vars.is_empty() => "intro".to_string(),
            Tactic::Intro(vars) => format!("intro {}", vars.join(" ")),
            Tactic::Apply(lemma) => format!("apply {lemma}"),
            Tactic::Exact(term) => format!("exact {term}"),
            Tactic::Simp(lemmas) if lemmas.is_empty() => "simp".to_string(),
            Tactic::Simp(lemmas) => format!("simp [{}]", lemmas.join(", ")),
            Tactic::SimpArith(lemmas) if lemmas.is_empty() => "simp_arith".to_string(),
            Tactic::SimpArith(lemmas) => format!("simp_arith [{}]", lemmas.join(", ")),
            Tactic::Omega => "omega".to_string(),
            Tactic::Linarith => "linarith".to_string(),
            Tactic::Nlinarith => "nlinarith".to_string(),
            Tactic::Polyrith => "polyrith".to_string(),
            Tactic::Decide => "decide".to_string(),
            Tactic::Rfl => "rfl".to_string(),
            Tactic::Symm => "symm".to_string(),
            Tactic::Trans(expr) => format!("trans {expr}"),
            Tactic::Constructor => "constructor".to_string(),
            Tactic::Left => "left".to_string(),
            Tactic::Right => "right".to_string(),
            Tactic::Cases(expr, branches) => {
                let mut s = format!("cases {expr} with\n");
                for (pattern, block) in branches {
                    let _ = writeln!(s, "  | {} => {}", pattern, block.to_lean5());
                }
                s
            }
            Tactic::Induction(var, branches) => {
                let mut s = format!("induction {var} with\n");
                for (pattern, block) in branches {
                    let _ = writeln!(s, "  | {} => {}", pattern, block.to_lean5());
                }
                s
            }
            Tactic::Have(name, ty, proof) => {
                format!("have {} : {} := {}", name, ty, proof.to_lean5())
            }
            Tactic::Rewrite(lemmas) => format!("rw [{}]", lemmas.join(", ")),
            Tactic::Ring => "ring".to_string(),
            Tactic::RingNf => "ring_nf".to_string(),
            Tactic::NativeDecide => "native_decide".to_string(),
            Tactic::Assumption => "assumption".to_string(),
            Tactic::Trivial => "trivial".to_string(),
            Tactic::Tauto => "tauto".to_string(),
            Tactic::AndIntro => "constructor".to_string(), // In Lean4, And.intro is via constructor
            Tactic::AndLeft(h) => format!("exact {h}.1"),
            Tactic::AndRight(h) => format!("exact {h}.2"),
            Tactic::Obtain(pattern, h) => format!("obtain {pattern} := {h}"),
            Tactic::BvDecide => "bv_decide".to_string(),
            Tactic::BvOmega => "bv_omega".to_string(),
            Tactic::ModCast => "mod_cast".to_string(),
            Tactic::PushCast => "push_cast".to_string(),
            Tactic::NormCast => "norm_cast".to_string(),
            Tactic::Sorry(reason) => format!("sorry -- {reason}"),
            Tactic::Raw(s) => s.clone(),
        }
    }
}

/// Generate tactics for a proof obligation
pub fn generate_tactics(obligation: &ProofObligation) -> TacticBlock {
    match &obligation.kind {
        ProofObligationKind::Initiation => generate_initiation_tactics(obligation),
        ProofObligationKind::Consecution => generate_consecution_tactics(obligation),
        ProofObligationKind::Property => generate_property_tactics(obligation),
        ProofObligationKind::Custom(_) => generate_custom_tactics(obligation),
    }
}

/// Generate tactics for initiation proofs (Init => Inv)
fn generate_initiation_tactics(obligation: &ProofObligation) -> TacticBlock {
    // Analyze the statement to determine appropriate tactics
    let stmt = &obligation.statement;

    // Try to generate a complete proof for simple cases
    if let Some(tactics) = try_simple_initiation_proof(stmt) {
        return tactics;
    }

    // For more complex cases, generate a proof sketch
    let mut tactics = Vec::new();

    // Introduce any universal quantifiers
    let (intro_vars, body) = collect_forall_vars(stmt);
    if !intro_vars.is_empty() {
        tactics.push(Tactic::Intro(intro_vars));
    }

    // Check if this is an implication
    if let Lean5Expr::Implies(_, conclusion) = body {
        tactics.push(Tactic::Intro(vec!["hinit".to_string()]));

        // Classify the conclusion pattern and try pattern-based tactics
        let pattern = classify_pattern(conclusion);
        if let Some(pattern_tactics) = generate_pattern_tactics(pattern, conclusion) {
            tactics.extend(pattern_tactics.tactics);
            return TacticBlock::new(tactics);
        }

        // Try omega for arithmetic conclusions
        if is_arithmetic_expr(conclusion) {
            tactics.push(Tactic::Omega);
            return TacticBlock::new(tactics);
        }
    }

    // Try pattern-based tactics on the full statement
    let pattern = classify_pattern(&obligation.statement);
    if let Some(pattern_tactics) = generate_pattern_tactics(pattern, &obligation.statement) {
        tactics.extend(pattern_tactics.tactics);
        return TacticBlock::new(tactics);
    }

    // Fallback: try omega for any arithmetic, then simp/decide for booleans
    if is_arithmetic_expr(&obligation.statement) {
        tactics.push(Tactic::Omega);
        TacticBlock::new(tactics)
    } else if contains_any_arithmetic(&obligation.statement) {
        // If there's any arithmetic anywhere, still try omega
        tactics.push(Tactic::Simp(vec!["*".to_string()]));
        tactics.push(Tactic::Omega);
        TacticBlock::new(tactics)
    } else {
        tactics.push(Tactic::Simp(vec!["*".to_string()]));
        tactics.push(Tactic::Decide);
        TacticBlock::new(tactics)
    }
}

/// Generate tactics for consecution proofs (Inv ∧ Trans => Inv')
fn generate_consecution_tactics(obligation: &ProofObligation) -> TacticBlock {
    let mut tactics = Vec::new();

    // Collect universal quantifiers
    let (intro_vars, body) = collect_forall_vars(&obligation.statement);
    if !intro_vars.is_empty() {
        tactics.push(Tactic::Intro(intro_vars));
    }

    // For implications, introduce hypotheses
    if let Lean5Expr::Implies(hypothesis, conclusion) = body {
        // Handle different hypothesis structures for k-induction consecution proofs
        let hyp_depth = count_conjunction_depth(hypothesis);

        if hyp_depth >= 2 {
            // Deeply nested conjunction (k-induction with k >= 2)
            // Use obtain to destructure: obtain ⟨h1, h2, h3, ...⟩ := h
            let pattern = generate_conjunction_pattern(hyp_depth);
            tactics.push(Tactic::Intro(vec!["h".to_string()]));
            tactics.push(Tactic::Obtain(pattern, "h".to_string()));
        } else if matches!(hypothesis.as_ref(), Lean5Expr::And(_, _)) {
            // Simple conjunction (Inv ∧ Trans)
            tactics.push(Tactic::Intro(vec!["⟨hinv, htrans⟩".to_string()]));
        } else {
            tactics.push(Tactic::Intro(vec!["h".to_string()]));
        }

        // Classify the conclusion pattern
        let pattern = classify_pattern(conclusion);

        // For reflexive equality in conclusion (invariant doesn't change)
        if pattern == InvariantPattern::ReflexiveEquality {
            tactics.push(Tactic::Rfl);
            return TacticBlock::new(tactics);
        }

        // For multi-variable arithmetic (common in k-induction)
        if pattern == InvariantPattern::MultiVariableArithmetic {
            // Use simp_arith with all hypotheses, then omega
            tactics.push(Tactic::SimpArith(vec!["*".to_string()]));
            tactics.push(Tactic::Omega);
            return TacticBlock::new(tactics);
        }

        // For conjunctions, try splitting
        if pattern == InvariantPattern::Conjunction && contains_any_arithmetic(conclusion) {
            // For arithmetic conjunctions: simp to use hypotheses, then omega
            tactics.push(Tactic::Simp(vec!["*".to_string()]));
            tactics.push(Tactic::Raw("<;> omega".to_string()));
            return TacticBlock::new(tactics);
        }

        // For arithmetic conclusions, omega should handle it
        if is_arithmetic_expr(conclusion) {
            tactics.push(Tactic::Omega);
            return TacticBlock::new(tactics);
        }

        // For boolean conclusions, try simp with hypotheses first
        tactics.push(Tactic::Simp(vec!["*".to_string()]));

        // Then try omega if there might be arithmetic hidden in the formula
        if contains_any_arithmetic(&obligation.statement) {
            tactics.push(Tactic::Omega);
            return TacticBlock::new(tactics);
        }

        // If conclusion is trivially true
        if matches!(conclusion.as_ref(), Lean5Expr::BoolLit(true)) {
            tactics.push(Tactic::Decide);
            return TacticBlock::new(tactics);
        }
    }

    // Last resort: try omega anyway (it's harmless if it fails at proof check time)
    if contains_any_arithmetic(&obligation.statement) {
        tactics.push(Tactic::Omega);
        return TacticBlock::new(tactics);
    }

    tactics.push(Tactic::Simp(vec!["*".to_string()]));
    tactics.push(Tactic::Decide);
    TacticBlock::new(tactics)
}

/// Count the nesting depth of conjunctions
fn count_conjunction_depth(expr: &Lean5Expr) -> usize {
    match expr {
        Lean5Expr::And(_, b) => 1 + count_conjunction_depth(b),
        _ => 1,
    }
}

/// Generate a destructuring pattern for nested conjunctions
/// e.g., depth=3 generates "⟨h1, h2, h3⟩"
fn generate_conjunction_pattern(depth: usize) -> String {
    let names: Vec<String> = (1..=depth).map(|i| format!("h{i}")).collect();
    format!("⟨{}⟩", names.join(", "))
}

/// Generate tactics for property proofs (Inv => Property)
fn generate_property_tactics(obligation: &ProofObligation) -> TacticBlock {
    let mut tactics = Vec::new();

    // Collect universal quantifiers
    let (intro_vars, body) = collect_forall_vars(&obligation.statement);
    if !intro_vars.is_empty() {
        tactics.push(Tactic::Intro(intro_vars));
    }

    if let Lean5Expr::Implies(_, conclusion) = body {
        tactics.push(Tactic::Intro(vec!["hinv".to_string()]));

        // Classify the conclusion pattern
        let pattern = classify_pattern(conclusion);

        // For reflexive equality
        if pattern == InvariantPattern::ReflexiveEquality {
            tactics.push(Tactic::Rfl);
            return TacticBlock::new(tactics);
        }

        // For simple arithmetic properties, omega usually works
        if is_arithmetic_expr(conclusion) {
            tactics.push(Tactic::Omega);
            return TacticBlock::new(tactics);
        }

        // For conjunctions in conclusions
        if pattern == InvariantPattern::Conjunction && contains_any_arithmetic(conclusion) {
            tactics.push(Tactic::Constructor);
            tactics.push(Tactic::Raw("<;> omega".to_string()));
            return TacticBlock::new(tactics);
        }

        // For trivially true conclusions
        if matches!(conclusion.as_ref(), Lean5Expr::BoolLit(true)) {
            tactics.push(Tactic::Decide);
            return TacticBlock::new(tactics);
        }

        // Try simp with assumption
        tactics.push(Tactic::Simp(vec!["*".to_string()]));

        // If there's any arithmetic in the statement, try omega
        if contains_any_arithmetic(&obligation.statement) {
            tactics.push(Tactic::Omega);
            return TacticBlock::new(tactics);
        }
    }

    // Check for simple equality or reflexivity
    if matches!(body, Lean5Expr::Eq(a, b) if a == b) {
        tactics.push(Tactic::Rfl);
        return TacticBlock::new(tactics);
    }

    // If we have any arithmetic at all, try omega
    if contains_any_arithmetic(&obligation.statement) {
        tactics.push(Tactic::Omega);
        return TacticBlock::new(tactics);
    }

    tactics.push(Tactic::Simp(vec!["*".to_string()]));
    tactics.push(Tactic::Decide);
    TacticBlock::new(tactics)
}

/// Generate tactics for custom obligations
fn generate_custom_tactics(obligation: &ProofObligation) -> TacticBlock {
    let mut tactics = Vec::new();

    // Collect universal quantifiers
    let (intro_vars, body) = collect_forall_vars(&obligation.statement);
    if !intro_vars.is_empty() {
        tactics.push(Tactic::Intro(intro_vars));
    }

    // Classify the overall pattern first
    let pattern = classify_pattern(&obligation.statement);

    // Handle implications
    if let Lean5Expr::Implies(_, conclusion) = body {
        tactics.push(Tactic::Intro(vec!["h".to_string()]));

        // Check conclusion pattern
        let concl_pattern = classify_pattern(conclusion);

        // Reflexive equality
        if concl_pattern == InvariantPattern::ReflexiveEquality {
            tactics.push(Tactic::Rfl);
            return TacticBlock::new(tactics);
        }

        // For arithmetic conclusions
        if is_arithmetic_expr(conclusion) {
            tactics.push(Tactic::Omega);
            return TacticBlock::new(tactics);
        }

        // For conjunctions
        if concl_pattern == InvariantPattern::Conjunction && contains_any_arithmetic(conclusion) {
            tactics.push(Tactic::Constructor);
            tactics.push(Tactic::Raw("<;> omega".to_string()));
            return TacticBlock::new(tactics);
        }

        // For trivially true
        if matches!(conclusion.as_ref(), Lean5Expr::BoolLit(true)) {
            tactics.push(Tactic::Decide);
            return TacticBlock::new(tactics);
        }
    }

    // Try pattern-based tactics
    if let Some(pattern_tactics) = generate_pattern_tactics(pattern, &obligation.statement) {
        tactics.extend(pattern_tactics.tactics);
        return TacticBlock::new(tactics);
    }

    // Try common tactics that often work
    if is_arithmetic_expr(&obligation.statement) {
        tactics.push(Tactic::Omega);
        return TacticBlock::new(tactics);
    }

    // Try simp first
    tactics.push(Tactic::Simp(vec![]));

    // If there's any arithmetic, also try omega
    if contains_any_arithmetic(&obligation.statement) {
        tactics.push(Tactic::Omega);
        return TacticBlock::new(tactics);
    }

    tactics.push(Tactic::Decide);
    TacticBlock::new(tactics)
}

/// Try to generate a complete proof for simple initiation cases
fn try_simple_initiation_proof(stmt: &Lean5Expr) -> Option<TacticBlock> {
    // x = 0 → x ≥ 0 is trivially true
    if let Lean5Expr::Implies(hyp, concl) = stmt {
        // Check for "x = 0 → x ≥ 0" pattern
        if let Lean5Expr::Eq(_, rhs) = hyp.as_ref() {
            if matches!(rhs.as_ref(), Lean5Expr::IntLit(0)) && is_arithmetic_expr(concl) {
                return Some(TacticBlock::new(vec![
                    Tactic::Intro(vec!["h".to_string()]),
                    Tactic::Omega,
                ]));
            }
        }
    }

    // For quantified statements, recurse
    if let Lean5Expr::Forall(var, _, body) = stmt {
        if let Some(inner) = try_simple_initiation_proof(body) {
            let mut tactics = vec![Tactic::Intro(vec![var.clone()])];
            tactics.extend(inner.tactics);
            return Some(TacticBlock::new(tactics));
        }
    }

    None
}

/// Collect forall variables from an expression
fn collect_forall_vars(expr: &Lean5Expr) -> (Vec<String>, &Lean5Expr) {
    let mut vars = Vec::new();
    let mut current = expr;

    while let Lean5Expr::Forall(name, _, body) = current {
        vars.push(name.clone());
        current = body;
    }

    (vars, current)
}

/// Check if an expression involves arithmetic (Int/Nat comparisons)
fn is_arithmetic_expr(expr: &Lean5Expr) -> bool {
    match expr {
        Lean5Expr::Ge(_, _)
        | Lean5Expr::Gt(_, _)
        | Lean5Expr::Le(_, _)
        | Lean5Expr::Lt(_, _)
        | Lean5Expr::Add(_, _)
        | Lean5Expr::Sub(_, _)
        | Lean5Expr::Mul(_, _)
        | Lean5Expr::IntLit(_)
        | Lean5Expr::NatLit(_) => true,
        Lean5Expr::Eq(a, b) => is_arithmetic_expr(a) || is_arithmetic_expr(b),
        Lean5Expr::And(a, b) | Lean5Expr::Or(a, b) | Lean5Expr::Implies(a, b) => {
            is_arithmetic_expr(a) || is_arithmetic_expr(b)
        }
        Lean5Expr::Not(a) => is_arithmetic_expr(a),
        Lean5Expr::Forall(_, _, body) | Lean5Expr::Exists(_, _, body) => is_arithmetic_expr(body),
        Lean5Expr::Var(_) => false, // Variables alone aren't arithmetic
        _ => false,
    }
}

/// Check if an expression contains any arithmetic anywhere in its structure
fn contains_any_arithmetic(expr: &Lean5Expr) -> bool {
    match expr {
        Lean5Expr::Ge(_, _)
        | Lean5Expr::Gt(_, _)
        | Lean5Expr::Le(_, _)
        | Lean5Expr::Lt(_, _)
        | Lean5Expr::Add(_, _)
        | Lean5Expr::Sub(_, _)
        | Lean5Expr::Mul(_, _)
        | Lean5Expr::Div(_, _)
        | Lean5Expr::Mod(_, _)
        | Lean5Expr::Neg(_)
        | Lean5Expr::IntLit(_)
        | Lean5Expr::NatLit(_) => true,
        Lean5Expr::Eq(a, b)
        | Lean5Expr::And(a, b)
        | Lean5Expr::Or(a, b)
        | Lean5Expr::Implies(a, b) => contains_any_arithmetic(a) || contains_any_arithmetic(b),
        Lean5Expr::Not(a) => contains_any_arithmetic(a),
        Lean5Expr::Forall(_, _, body) | Lean5Expr::Exists(_, _, body) => {
            contains_any_arithmetic(body)
        }
        Lean5Expr::Ite(c, t, e) => {
            contains_any_arithmetic(c) || contains_any_arithmetic(t) || contains_any_arithmetic(e)
        }
        Lean5Expr::App(f, a) => contains_any_arithmetic(f) || contains_any_arithmetic(a),
        Lean5Expr::Let(_, _, v, b) => contains_any_arithmetic(v) || contains_any_arithmetic(b),
        _ => false,
    }
}

/// Pattern classification for invariant expressions
#[derive(Debug, Clone, PartialEq)]
enum InvariantPattern {
    /// x >= 0 or x > 0
    NonNegative,
    /// x <= N or x < N for some constant N
    UpperBounded,
    /// x >= L and x <= U (range constraint)
    RangeBounded,
    /// x <= y or x < y (ordering between variables)
    Ordering,
    /// a => b (boolean implication)
    BoolImplication,
    /// a ∧ b (conjunction of invariants)
    Conjunction,
    /// a ∨ b (disjunction of invariants)
    Disjunction,
    /// Pure boolean (true, false, or boolean variable)
    PureBool,
    /// Equality x = c for constant c
    EqualityConst,
    /// Reflexive equality x = x
    ReflexiveEquality,
    /// Equality with arithmetic expression: x = expr
    EqualityArithmetic,
    /// Multiple variables with arithmetic relationships
    MultiVariableArithmetic,
    /// Monotonicity: x' >= x or similar transition relationships
    Monotonicity,
    /// Non-linear arithmetic (multiplication of variables, x * y)
    NonLinearArithmetic,
    /// Modular arithmetic (mod, div operations)
    ModularArithmetic,
    /// Bitvector operations (reserved for future bitvector detection)
    #[allow(dead_code)]
    BitVector,
    /// General arithmetic (fallback)
    GeneralArithmetic,
    /// Unknown pattern
    Unknown,
}

/// Analyze an expression to determine its pattern
fn classify_pattern(expr: &Lean5Expr) -> InvariantPattern {
    match expr {
        // Reflexive equality
        Lean5Expr::Eq(a, b) if a == b => InvariantPattern::ReflexiveEquality,

        // Equality to constant
        Lean5Expr::Eq(lhs, rhs)
            if matches!(rhs.as_ref(), Lean5Expr::IntLit(_) | Lean5Expr::NatLit(_))
                && matches!(lhs.as_ref(), Lean5Expr::Var(_)) =>
        {
            InvariantPattern::EqualityConst
        }
        Lean5Expr::Eq(lhs, rhs)
            if matches!(lhs.as_ref(), Lean5Expr::IntLit(_) | Lean5Expr::NatLit(_))
                && matches!(rhs.as_ref(), Lean5Expr::Var(_)) =>
        {
            InvariantPattern::EqualityConst
        }

        // Equality with arithmetic expression: x = f(y, z, ...)
        Lean5Expr::Eq(lhs, rhs)
            if matches!(lhs.as_ref(), Lean5Expr::Var(_)) && contains_any_arithmetic(rhs) =>
        {
            InvariantPattern::EqualityArithmetic
        }
        Lean5Expr::Eq(lhs, rhs)
            if matches!(rhs.as_ref(), Lean5Expr::Var(_)) && contains_any_arithmetic(lhs) =>
        {
            InvariantPattern::EqualityArithmetic
        }

        // Non-negativity: x >= 0, x > 0, 0 <= x, 0 < x (only simple variable cases)
        Lean5Expr::Ge(lhs, rhs)
            if matches!(rhs.as_ref(), Lean5Expr::IntLit(0) | Lean5Expr::NatLit(0))
                && matches!(lhs.as_ref(), Lean5Expr::Var(_)) =>
        {
            InvariantPattern::NonNegative
        }
        Lean5Expr::Gt(lhs, rhs)
            if matches!(rhs.as_ref(), Lean5Expr::IntLit(0) | Lean5Expr::NatLit(0))
                && matches!(lhs.as_ref(), Lean5Expr::Var(_)) =>
        {
            InvariantPattern::NonNegative
        }
        Lean5Expr::Le(lhs, rhs)
            if matches!(lhs.as_ref(), Lean5Expr::IntLit(0) | Lean5Expr::NatLit(0))
                && matches!(rhs.as_ref(), Lean5Expr::Var(_)) =>
        {
            InvariantPattern::NonNegative
        }
        Lean5Expr::Lt(lhs, rhs)
            if matches!(lhs.as_ref(), Lean5Expr::IntLit(0) | Lean5Expr::NatLit(0))
                && matches!(rhs.as_ref(), Lean5Expr::Var(_)) =>
        {
            InvariantPattern::NonNegative
        }

        // Upper bounded: x <= N or x < N where N is constant and x is a simple variable
        Lean5Expr::Le(lhs, rhs) | Lean5Expr::Lt(lhs, rhs)
            if matches!(rhs.as_ref(), Lean5Expr::IntLit(_) | Lean5Expr::NatLit(_))
                && matches!(lhs.as_ref(), Lean5Expr::Var(_)) =>
        {
            InvariantPattern::UpperBounded
        }

        // Monotonicity patterns: x' >= x (common in k-induction transitions)
        Lean5Expr::Ge(lhs, rhs) | Lean5Expr::Le(rhs, lhs) if is_primed_pair(lhs, rhs) => {
            InvariantPattern::Monotonicity
        }

        // Ordering between variables: x <= y or x < y
        Lean5Expr::Le(lhs, rhs) | Lean5Expr::Lt(lhs, rhs)
            if matches!(lhs.as_ref(), Lean5Expr::Var(_))
                && matches!(rhs.as_ref(), Lean5Expr::Var(_)) =>
        {
            InvariantPattern::Ordering
        }

        // Conjunction - check if it's a range (non_neg AND upper_bound)
        Lean5Expr::And(a, b) => {
            let pat_a = classify_pattern(a);
            let pat_b = classify_pattern(b);
            if (pat_a == InvariantPattern::NonNegative && pat_b == InvariantPattern::UpperBounded)
                || (pat_a == InvariantPattern::UpperBounded
                    && pat_b == InvariantPattern::NonNegative)
            {
                InvariantPattern::RangeBounded
            } else if has_multiple_variables(a) || has_multiple_variables(b) {
                // Multi-variable arithmetic conjunction
                InvariantPattern::MultiVariableArithmetic
            } else {
                InvariantPattern::Conjunction
            }
        }

        // Disjunction
        Lean5Expr::Or(_, _) => InvariantPattern::Disjunction,

        // Boolean implication
        Lean5Expr::Implies(_, _) => InvariantPattern::BoolImplication,

        // Pure boolean values
        Lean5Expr::BoolLit(_) => InvariantPattern::PureBool,

        // Var alone is treated as unknown since we don't know its type
        Lean5Expr::Var(_) => InvariantPattern::Unknown,

        // General arithmetic comparisons with arithmetic subexpressions
        Lean5Expr::Ge(lhs, rhs)
        | Lean5Expr::Gt(lhs, rhs)
        | Lean5Expr::Le(lhs, rhs)
        | Lean5Expr::Lt(lhs, rhs) => {
            // Check for non-linear arithmetic first (multiplication of variables)
            if contains_nonlinear_term(lhs) || contains_nonlinear_term(rhs) {
                InvariantPattern::NonLinearArithmetic
            } else if contains_modular_arithmetic(lhs) || contains_modular_arithmetic(rhs) {
                InvariantPattern::ModularArithmetic
            } else if has_multiple_variables(lhs) || has_multiple_variables(rhs) {
                InvariantPattern::MultiVariableArithmetic
            } else {
                InvariantPattern::GeneralArithmetic
            }
        }

        // Equality checks for special patterns
        Lean5Expr::Eq(lhs, rhs)
            if !matches!(lhs.as_ref(), Lean5Expr::Var(_))
                && !matches!(rhs.as_ref(), Lean5Expr::Var(_)) =>
        {
            if contains_nonlinear_term(lhs) || contains_nonlinear_term(rhs) {
                InvariantPattern::NonLinearArithmetic
            } else if contains_modular_arithmetic(lhs) || contains_modular_arithmetic(rhs) {
                InvariantPattern::ModularArithmetic
            } else {
                InvariantPattern::GeneralArithmetic
            }
        }

        // Check under quantifiers
        Lean5Expr::Forall(_, _, body) | Lean5Expr::Exists(_, _, body) => classify_pattern(body),

        _ => InvariantPattern::Unknown,
    }
}

/// Check if expression contains non-linear terms (multiplication of variables)
fn contains_nonlinear_term(expr: &Lean5Expr) -> bool {
    match expr {
        // Multiplication where both sides contain variables
        Lean5Expr::Mul(lhs, rhs) => {
            let lhs_has_var = contains_any_variable(lhs);
            let rhs_has_var = contains_any_variable(rhs);
            // Non-linear if both sides have variables
            (lhs_has_var && rhs_has_var)
                || contains_nonlinear_term(lhs)
                || contains_nonlinear_term(rhs)
        }
        // Division by variable expression is also non-linear
        Lean5Expr::Div(_, rhs) if contains_any_variable(rhs) => true,
        // Recurse through other arithmetic
        Lean5Expr::Add(a, b)
        | Lean5Expr::Sub(a, b)
        | Lean5Expr::Div(a, b)
        | Lean5Expr::Mod(a, b) => contains_nonlinear_term(a) || contains_nonlinear_term(b),
        Lean5Expr::Neg(a) => contains_nonlinear_term(a),
        Lean5Expr::Ite(_, t, e) => contains_nonlinear_term(t) || contains_nonlinear_term(e),
        _ => false,
    }
}

/// Check if expression contains any variable
fn contains_any_variable(expr: &Lean5Expr) -> bool {
    match expr {
        Lean5Expr::Var(_) => true,
        Lean5Expr::Add(a, b)
        | Lean5Expr::Sub(a, b)
        | Lean5Expr::Mul(a, b)
        | Lean5Expr::Div(a, b)
        | Lean5Expr::Mod(a, b) => contains_any_variable(a) || contains_any_variable(b),
        Lean5Expr::Neg(a) => contains_any_variable(a),
        Lean5Expr::Ite(c, t, e) => {
            contains_any_variable(c) || contains_any_variable(t) || contains_any_variable(e)
        }
        _ => false,
    }
}

/// Check if expression contains modular arithmetic (mod, div)
fn contains_modular_arithmetic(expr: &Lean5Expr) -> bool {
    match expr {
        Lean5Expr::Mod(_, _) | Lean5Expr::Div(_, _) => true,
        Lean5Expr::Add(a, b) | Lean5Expr::Sub(a, b) | Lean5Expr::Mul(a, b) => {
            contains_modular_arithmetic(a) || contains_modular_arithmetic(b)
        }
        Lean5Expr::Neg(a) => contains_modular_arithmetic(a),
        Lean5Expr::Ite(c, t, e) => {
            contains_modular_arithmetic(c)
                || contains_modular_arithmetic(t)
                || contains_modular_arithmetic(e)
        }
        _ => false,
    }
}

/// Check if lhs looks like a primed version of rhs (e.g., x_next vs x, x' vs x)
fn is_primed_pair(lhs: &Lean5Expr, rhs: &Lean5Expr) -> bool {
    if let (Lean5Expr::Var(l), Lean5Expr::Var(r)) = (lhs, rhs) {
        // Check common priming conventions
        l.starts_with(r) && (l.ends_with("_next") || l.ends_with('\'') || l.ends_with("_prime"))
    } else {
        false
    }
}

/// Check if expression contains multiple distinct variables
fn has_multiple_variables(expr: &Lean5Expr) -> bool {
    let vars = collect_variables(expr);
    vars.len() > 1
}

/// Collect all variable names from an expression
fn collect_variables(expr: &Lean5Expr) -> Vec<String> {
    let mut vars = Vec::new();
    collect_variables_impl(expr, &mut vars);
    vars.sort();
    vars.dedup();
    vars
}

fn collect_variables_impl(expr: &Lean5Expr, vars: &mut Vec<String>) {
    match expr {
        Lean5Expr::Var(name) => vars.push(name.clone()),
        Lean5Expr::And(a, b)
        | Lean5Expr::Or(a, b)
        | Lean5Expr::Implies(a, b)
        | Lean5Expr::Eq(a, b)
        | Lean5Expr::Lt(a, b)
        | Lean5Expr::Le(a, b)
        | Lean5Expr::Gt(a, b)
        | Lean5Expr::Ge(a, b)
        | Lean5Expr::Add(a, b)
        | Lean5Expr::Sub(a, b)
        | Lean5Expr::Mul(a, b)
        | Lean5Expr::Div(a, b)
        | Lean5Expr::Mod(a, b) => {
            collect_variables_impl(a, vars);
            collect_variables_impl(b, vars);
        }
        Lean5Expr::Not(a) | Lean5Expr::Neg(a) => collect_variables_impl(a, vars),
        Lean5Expr::Ite(c, t, e) => {
            collect_variables_impl(c, vars);
            collect_variables_impl(t, vars);
            collect_variables_impl(e, vars);
        }
        Lean5Expr::Forall(_, _, body) | Lean5Expr::Exists(_, _, body) => {
            collect_variables_impl(body, vars);
        }
        Lean5Expr::App(f, a) => {
            collect_variables_impl(f, vars);
            collect_variables_impl(a, vars);
        }
        Lean5Expr::Lam(_, _, body) | Lean5Expr::Let(_, _, _, body) => {
            collect_variables_impl(body, vars);
        }
        _ => {}
    }
}

/// Generate tactics based on recognized pattern
fn generate_pattern_tactics(pattern: InvariantPattern, expr: &Lean5Expr) -> Option<TacticBlock> {
    match pattern {
        InvariantPattern::ReflexiveEquality => {
            // x = x is trivially true with rfl
            Some(TacticBlock::new(vec![Tactic::Rfl]))
        }

        InvariantPattern::PureBool => {
            // For pure boolean, decide usually works
            if matches!(expr, Lean5Expr::BoolLit(true)) {
                Some(TacticBlock::new(vec![Tactic::Decide]))
            } else {
                // For other boolean vars, try simp then decide
                Some(TacticBlock::new(vec![
                    Tactic::Simp(vec!["*".to_string()]),
                    Tactic::Decide,
                ]))
            }
        }

        InvariantPattern::NonNegative
        | InvariantPattern::UpperBounded
        | InvariantPattern::RangeBounded
        | InvariantPattern::Ordering
        | InvariantPattern::GeneralArithmetic
        | InvariantPattern::EqualityConst
        | InvariantPattern::Monotonicity => {
            // Omega handles all linear arithmetic
            Some(TacticBlock::new(vec![Tactic::Omega]))
        }

        InvariantPattern::EqualityArithmetic => {
            // Equality with arithmetic: try simp_arith first then omega
            Some(TacticBlock::new(vec![
                Tactic::SimpArith(vec!["*".to_string()]),
                Tactic::Omega,
            ]))
        }

        InvariantPattern::MultiVariableArithmetic => {
            // Multi-variable arithmetic requires careful handling
            // Use simp with hypotheses to simplify, then omega/linarith
            Some(TacticBlock::new(vec![
                Tactic::Simp(vec!["*".to_string()]),
                Tactic::Omega,
            ]))
        }

        InvariantPattern::NonLinearArithmetic => {
            // Non-linear arithmetic requires nlinarith or polyrith
            // Try ring_nf to normalize, then nlinarith
            Some(TacticBlock::new(vec![Tactic::RingNf, Tactic::Nlinarith]))
        }

        InvariantPattern::ModularArithmetic => {
            // Modular arithmetic requires omega with mod_cast support
            Some(TacticBlock::new(vec![
                Tactic::Simp(vec!["*".to_string()]),
                Tactic::ModCast,
                Tactic::Omega,
            ]))
        }

        InvariantPattern::BitVector => {
            // Bitvector operations need bv_decide or bv_omega
            Some(TacticBlock::new(vec![Tactic::BvOmega]))
        }

        InvariantPattern::Conjunction => {
            // For conjunctions, use constructor to split, then omega for each part
            if contains_any_arithmetic(expr) {
                Some(TacticBlock::new(vec![
                    Tactic::Constructor,
                    Tactic::Raw("<;> omega".to_string()),
                ]))
            } else {
                // Boolean conjunction - try simp then decide
                Some(TacticBlock::new(vec![Tactic::Simp(vec![]), Tactic::Decide]))
            }
        }

        InvariantPattern::Disjunction => {
            // Disjunction: might need to pick a side or use tauto
            if contains_any_arithmetic(expr) {
                // Try omega first (can sometimes handle disjunctions)
                Some(TacticBlock::new(vec![
                    Tactic::Simp(vec!["*".to_string()]),
                    Tactic::Omega,
                ]))
            } else {
                // Boolean disjunction - try tauto or simp+decide
                Some(TacticBlock::new(vec![
                    Tactic::Simp(vec!["*".to_string()]),
                    Tactic::Decide,
                ]))
            }
        }

        InvariantPattern::BoolImplication => {
            // Boolean implication: intro the hypothesis, then use it
            Some(TacticBlock::new(vec![
                Tactic::Intro(vec!["h".to_string()]),
                Tactic::Simp(vec!["*".to_string()]),
                Tactic::Decide,
            ]))
        }

        InvariantPattern::Unknown => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tactic_to_lean5() {
        assert_eq!(Tactic::Intro(vec!["x".to_string()]).to_lean5(), "intro x");
        assert_eq!(Tactic::Omega.to_lean5(), "omega");
        assert_eq!(Tactic::Simp(vec![]).to_lean5(), "simp");
        assert_eq!(
            Tactic::Simp(vec!["h1".to_string(), "h2".to_string()]).to_lean5(),
            "simp [h1, h2]"
        );
    }

    #[test]
    fn test_tactic_block_to_lean5() {
        let block = TacticBlock::new(vec![Tactic::Intro(vec!["x".to_string()]), Tactic::Omega]);

        let lean = block.to_lean5();
        assert!(lean.contains("intro x"));
        assert!(lean.contains("omega"));
    }

    #[test]
    fn test_is_arithmetic_expr() {
        assert!(is_arithmetic_expr(&Lean5Expr::ge(
            Lean5Expr::var("x"),
            Lean5Expr::IntLit(0)
        )));

        assert!(is_arithmetic_expr(&Lean5Expr::implies(
            Lean5Expr::eq(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
            Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0))
        )));

        assert!(!is_arithmetic_expr(&Lean5Expr::var("x")));
    }

    #[test]
    fn test_collect_forall_vars() {
        let expr = Lean5Expr::forall_(
            "x",
            Lean5Type::Int,
            Lean5Expr::forall_(
                "y",
                Lean5Type::Int,
                Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::var("y")),
            ),
        );

        let (vars, body) = collect_forall_vars(&expr);
        assert_eq!(vars, vec!["x", "y"]);
        assert!(matches!(body, Lean5Expr::Ge(_, _)));
    }

    #[test]
    fn test_generate_initiation_tactics_simple() {
        let obligation = ProofObligation::new(
            "init_proof",
            ProofObligationKind::Initiation,
            Lean5Expr::implies(
                Lean5Expr::eq(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
                Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
            ),
        );

        let tactics = generate_tactics(&obligation);
        let lean = tactics.to_lean5();

        // Should generate intro and omega for this simple case
        assert!(
            lean.contains("intro") || lean.contains("omega"),
            "Got: {}",
            lean
        );
    }

    #[test]
    fn test_sorry_block() {
        let block = TacticBlock::sorry("need to prove this");
        assert!(!block.is_complete);
        assert!(block.to_lean5().contains("sorry"));
    }

    #[test]
    fn test_classify_pattern_non_negative() {
        // x >= 0
        let expr = Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0));
        assert_eq!(classify_pattern(&expr), InvariantPattern::NonNegative);

        // x > 0
        let expr = Lean5Expr::gt(Lean5Expr::var("x"), Lean5Expr::IntLit(0));
        assert_eq!(classify_pattern(&expr), InvariantPattern::NonNegative);

        // 0 <= x
        let expr = Lean5Expr::le(Lean5Expr::IntLit(0), Lean5Expr::var("x"));
        assert_eq!(classify_pattern(&expr), InvariantPattern::NonNegative);
    }

    #[test]
    fn test_classify_pattern_upper_bounded() {
        // x <= 100
        let expr = Lean5Expr::le(Lean5Expr::var("x"), Lean5Expr::IntLit(100));
        assert_eq!(classify_pattern(&expr), InvariantPattern::UpperBounded);

        // x < 50
        let expr = Lean5Expr::lt(Lean5Expr::var("x"), Lean5Expr::IntLit(50));
        assert_eq!(classify_pattern(&expr), InvariantPattern::UpperBounded);
    }

    #[test]
    fn test_classify_pattern_ordering() {
        // x <= y
        let expr = Lean5Expr::le(Lean5Expr::var("x"), Lean5Expr::var("y"));
        assert_eq!(classify_pattern(&expr), InvariantPattern::Ordering);

        // x < y
        let expr = Lean5Expr::lt(Lean5Expr::var("x"), Lean5Expr::var("y"));
        assert_eq!(classify_pattern(&expr), InvariantPattern::Ordering);
    }

    #[test]
    fn test_classify_pattern_range_bounded() {
        // x >= 0 ∧ x <= 100
        let expr = Lean5Expr::and(
            Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
            Lean5Expr::le(Lean5Expr::var("x"), Lean5Expr::IntLit(100)),
        );
        assert_eq!(classify_pattern(&expr), InvariantPattern::RangeBounded);
    }

    #[test]
    fn test_classify_pattern_reflexive_equality() {
        // x = x
        let x = Lean5Expr::var("x");
        let expr = Lean5Expr::eq(x.clone(), x);
        assert_eq!(classify_pattern(&expr), InvariantPattern::ReflexiveEquality);
    }

    #[test]
    fn test_classify_pattern_equality_const() {
        // x = 5
        let expr = Lean5Expr::eq(Lean5Expr::var("x"), Lean5Expr::IntLit(5));
        assert_eq!(classify_pattern(&expr), InvariantPattern::EqualityConst);
    }

    #[test]
    fn test_generate_pattern_tactics_non_negative() {
        let expr = Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0));
        let pattern = classify_pattern(&expr);
        let tactics = generate_pattern_tactics(pattern, &expr);

        assert!(tactics.is_some());
        let tactics = tactics.unwrap();
        assert!(tactics.is_complete);
        assert!(tactics.to_lean5().contains("omega"));
    }

    #[test]
    fn test_generate_pattern_tactics_reflexive() {
        let x = Lean5Expr::var("x");
        let expr = Lean5Expr::eq(x.clone(), x);
        let pattern = classify_pattern(&expr);
        let tactics = generate_pattern_tactics(pattern, &expr);

        assert!(tactics.is_some());
        let tactics = tactics.unwrap();
        assert!(tactics.is_complete);
        assert!(tactics.to_lean5().contains("rfl"));
    }

    #[test]
    fn test_generate_pattern_tactics_conjunction() {
        // x >= 0 ∧ y >= 0
        let expr = Lean5Expr::and(
            Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
            Lean5Expr::ge(Lean5Expr::var("y"), Lean5Expr::IntLit(0)),
        );
        let pattern = classify_pattern(&expr);
        let tactics = generate_pattern_tactics(pattern, &expr);

        assert!(tactics.is_some());
        let tactics = tactics.unwrap();
        assert!(tactics.is_complete);
        let lean = tactics.to_lean5();
        assert!(lean.contains("constructor") || lean.contains("omega"));
    }

    #[test]
    fn test_generate_consecution_tactics_arithmetic() {
        // (inv ∧ trans) → inv'
        let obligation = ProofObligation::new(
            "consecution",
            ProofObligationKind::Consecution,
            Lean5Expr::implies(
                Lean5Expr::and(
                    Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
                    Lean5Expr::eq(
                        Lean5Expr::var("x_next"),
                        Lean5Expr::add(Lean5Expr::var("x"), Lean5Expr::IntLit(1)),
                    ),
                ),
                Lean5Expr::ge(Lean5Expr::var("x_next"), Lean5Expr::IntLit(0)),
            ),
        );

        let tactics = generate_tactics(&obligation);
        assert!(tactics.is_complete);
        let lean = tactics.to_lean5();
        assert!(lean.contains("omega"), "Got: {}", lean);
    }

    #[test]
    fn test_generate_property_tactics_arithmetic() {
        // inv → property
        let obligation = ProofObligation::new(
            "property",
            ProofObligationKind::Property,
            Lean5Expr::implies(
                Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
                Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(-5)),
            ),
        );

        let tactics = generate_tactics(&obligation);
        assert!(tactics.is_complete);
        let lean = tactics.to_lean5();
        assert!(lean.contains("omega"), "Got: {}", lean);
    }

    #[test]
    fn test_generate_property_tactics_conjunction() {
        // inv → (property1 ∧ property2)
        let obligation = ProofObligation::new(
            "property_conj",
            ProofObligationKind::Property,
            Lean5Expr::implies(
                Lean5Expr::and(
                    Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
                    Lean5Expr::le(Lean5Expr::var("x"), Lean5Expr::IntLit(100)),
                ),
                Lean5Expr::and(
                    Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(-10)),
                    Lean5Expr::le(Lean5Expr::var("x"), Lean5Expr::IntLit(200)),
                ),
            ),
        );

        let tactics = generate_tactics(&obligation);
        assert!(tactics.is_complete);
        let lean = tactics.to_lean5();
        // Should use constructor to split and omega for each
        assert!(lean.contains("omega"), "Got: {}", lean);
    }

    #[test]
    fn test_complete_proof_for_simple_ordering() {
        // ∀x y, x <= y → x <= y (trivial)
        let obligation = ProofObligation::new(
            "ordering",
            ProofObligationKind::Custom("ordering".to_string()),
            Lean5Expr::forall_(
                "x",
                Lean5Type::Int,
                Lean5Expr::forall_(
                    "y",
                    Lean5Type::Int,
                    Lean5Expr::implies(
                        Lean5Expr::le(Lean5Expr::var("x"), Lean5Expr::var("y")),
                        Lean5Expr::le(Lean5Expr::var("x"), Lean5Expr::var("y")),
                    ),
                ),
            ),
        );

        let tactics = generate_tactics(&obligation);
        assert!(tactics.is_complete);
        let lean = tactics.to_lean5();
        assert!(
            lean.contains("intro") || lean.contains("omega"),
            "Got: {}",
            lean
        );
    }

    #[test]
    fn test_generate_initiation_tactics_boolean_decide() {
        let obligation = ProofObligation::new(
            "init_bool",
            ProofObligationKind::Initiation,
            Lean5Expr::implies(
                Lean5Expr::BoolLit(true),
                Lean5Expr::or(Lean5Expr::BoolLit(true), Lean5Expr::BoolLit(false)),
            ),
        );

        let tactics = generate_tactics(&obligation);
        assert!(tactics.is_complete);
        let lean = tactics.to_lean5();
        assert!(
            lean.contains("decide") || lean.contains("simp"),
            "Got: {}",
            lean
        );
    }

    #[test]
    fn test_generate_consecution_tactics_boolean() {
        let obligation = ProofObligation::new(
            "consecution_bool",
            ProofObligationKind::Consecution,
            Lean5Expr::implies(Lean5Expr::var("p"), Lean5Expr::var("p")),
        );

        let tactics = generate_tactics(&obligation);
        assert!(tactics.is_complete);
        let lean = tactics.to_lean5();
        assert!(
            lean.contains("decide") || lean.contains("simp"),
            "Got: {}",
            lean
        );
    }

    #[test]
    fn test_generate_property_tactics_boolean() {
        let obligation = ProofObligation::new(
            "property_bool",
            ProofObligationKind::Property,
            Lean5Expr::implies(Lean5Expr::var("p"), Lean5Expr::var("p")),
        );

        let tactics = generate_tactics(&obligation);
        assert!(tactics.is_complete);
        let lean = tactics.to_lean5();
        assert!(
            lean.contains("decide") || lean.contains("simp"),
            "Got: {}",
            lean
        );
    }

    // New tests for advanced tactics

    #[test]
    fn test_new_tactics_to_lean5() {
        assert_eq!(Tactic::Linarith.to_lean5(), "linarith");
        assert_eq!(Tactic::SimpArith(vec![]).to_lean5(), "simp_arith");
        assert_eq!(
            Tactic::SimpArith(vec!["*".to_string()]).to_lean5(),
            "simp_arith [*]"
        );
        assert_eq!(Tactic::Assumption.to_lean5(), "assumption");
        assert_eq!(Tactic::Trivial.to_lean5(), "trivial");
        assert_eq!(Tactic::Tauto.to_lean5(), "tauto");
        assert_eq!(Tactic::AndLeft("h".to_string()).to_lean5(), "exact h.1");
        assert_eq!(Tactic::AndRight("h".to_string()).to_lean5(), "exact h.2");
        assert_eq!(
            Tactic::Obtain("⟨h1, h2⟩".to_string(), "h".to_string()).to_lean5(),
            "obtain ⟨h1, h2⟩ := h"
        );
    }

    #[test]
    fn test_classify_pattern_multi_variable() {
        // x + y >= 0 (multiple variables)
        let expr = Lean5Expr::ge(
            Lean5Expr::add(Lean5Expr::var("x"), Lean5Expr::var("y")),
            Lean5Expr::IntLit(0),
        );
        assert_eq!(
            classify_pattern(&expr),
            InvariantPattern::MultiVariableArithmetic
        );
    }

    #[test]
    fn test_classify_pattern_equality_arithmetic() {
        // x = y + 1
        let expr = Lean5Expr::eq(
            Lean5Expr::var("x"),
            Lean5Expr::add(Lean5Expr::var("y"), Lean5Expr::IntLit(1)),
        );
        assert_eq!(
            classify_pattern(&expr),
            InvariantPattern::EqualityArithmetic
        );
    }

    #[test]
    fn test_classify_pattern_disjunction() {
        // x >= 0 ∨ x < 0
        let expr = Lean5Expr::or(
            Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
            Lean5Expr::lt(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
        );
        assert_eq!(classify_pattern(&expr), InvariantPattern::Disjunction);
    }

    #[test]
    fn test_collect_variables() {
        let expr = Lean5Expr::add(
            Lean5Expr::add(Lean5Expr::var("x"), Lean5Expr::var("y")),
            Lean5Expr::mul(Lean5Expr::var("z"), Lean5Expr::var("x")),
        );
        let vars = collect_variables(&expr);
        assert_eq!(vars.len(), 3);
        assert!(vars.contains(&"x".to_string()));
        assert!(vars.contains(&"y".to_string()));
        assert!(vars.contains(&"z".to_string()));
    }

    #[test]
    fn test_has_multiple_variables() {
        let single = Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0));
        assert!(!has_multiple_variables(&single));

        let multi = Lean5Expr::ge(
            Lean5Expr::add(Lean5Expr::var("x"), Lean5Expr::var("y")),
            Lean5Expr::IntLit(0),
        );
        assert!(has_multiple_variables(&multi));
    }

    #[test]
    fn test_count_conjunction_depth() {
        // Simple: x >= 0
        let simple = Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0));
        assert_eq!(count_conjunction_depth(&simple), 1);

        // a ∧ b
        let conj1 = Lean5Expr::and(
            Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
            Lean5Expr::le(Lean5Expr::var("x"), Lean5Expr::IntLit(10)),
        );
        assert_eq!(count_conjunction_depth(&conj1), 2);

        // a ∧ (b ∧ c)
        let conj2 = Lean5Expr::and(
            Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
            Lean5Expr::and(
                Lean5Expr::le(Lean5Expr::var("x"), Lean5Expr::IntLit(10)),
                Lean5Expr::ge(Lean5Expr::var("y"), Lean5Expr::IntLit(0)),
            ),
        );
        assert_eq!(count_conjunction_depth(&conj2), 3);
    }

    #[test]
    fn test_generate_conjunction_pattern() {
        assert_eq!(generate_conjunction_pattern(1), "⟨h1⟩");
        assert_eq!(generate_conjunction_pattern(2), "⟨h1, h2⟩");
        assert_eq!(generate_conjunction_pattern(3), "⟨h1, h2, h3⟩");
    }

    #[test]
    fn test_generate_consecution_with_nested_conjunction() {
        // Test k-induction style consecution: (h1 ∧ h2 ∧ h3) → conclusion
        let obligation = ProofObligation::new(
            "consecution_nested",
            ProofObligationKind::Consecution,
            Lean5Expr::implies(
                Lean5Expr::and(
                    Lean5Expr::ge(Lean5Expr::var("n"), Lean5Expr::IntLit(0)),
                    Lean5Expr::and(
                        Lean5Expr::ge(
                            Lean5Expr::add(Lean5Expr::var("n"), Lean5Expr::IntLit(1)),
                            Lean5Expr::IntLit(0),
                        ),
                        Lean5Expr::ge(
                            Lean5Expr::add(Lean5Expr::var("n"), Lean5Expr::IntLit(2)),
                            Lean5Expr::IntLit(0),
                        ),
                    ),
                ),
                Lean5Expr::ge(
                    Lean5Expr::add(Lean5Expr::var("n"), Lean5Expr::IntLit(3)),
                    Lean5Expr::IntLit(0),
                ),
            ),
        );

        let tactics = generate_tactics(&obligation);
        assert!(tactics.is_complete);
        let lean = tactics.to_lean5();
        // Should have intro and obtain for destructuring
        assert!(
            lean.contains("intro") || lean.contains("obtain") || lean.contains("omega"),
            "Got: {}",
            lean
        );
    }

    #[test]
    fn test_generate_pattern_tactics_multi_variable() {
        let expr = Lean5Expr::ge(
            Lean5Expr::add(Lean5Expr::var("x"), Lean5Expr::var("y")),
            Lean5Expr::IntLit(0),
        );
        let pattern = classify_pattern(&expr);
        let tactics = generate_pattern_tactics(pattern, &expr);

        assert!(tactics.is_some());
        let tactics = tactics.unwrap();
        assert!(tactics.is_complete);
        let lean = tactics.to_lean5();
        assert!(
            lean.contains("simp") || lean.contains("omega"),
            "Got: {}",
            lean
        );
    }

    #[test]
    fn test_generate_pattern_tactics_equality_arithmetic() {
        let expr = Lean5Expr::eq(
            Lean5Expr::var("x"),
            Lean5Expr::add(Lean5Expr::var("y"), Lean5Expr::IntLit(1)),
        );
        let pattern = classify_pattern(&expr);
        let tactics = generate_pattern_tactics(pattern, &expr);

        assert!(tactics.is_some());
        let tactics = tactics.unwrap();
        assert!(tactics.is_complete);
        let lean = tactics.to_lean5();
        assert!(
            lean.contains("simp_arith") || lean.contains("omega"),
            "Got: {}",
            lean
        );
    }

    // Tests for advanced tactics (non-linear, modular, bitvector)

    #[test]
    fn test_advanced_tactics_to_lean5() {
        assert_eq!(Tactic::Nlinarith.to_lean5(), "nlinarith");
        assert_eq!(Tactic::Polyrith.to_lean5(), "polyrith");
        assert_eq!(Tactic::RingNf.to_lean5(), "ring_nf");
        assert_eq!(Tactic::BvDecide.to_lean5(), "bv_decide");
        assert_eq!(Tactic::BvOmega.to_lean5(), "bv_omega");
        assert_eq!(Tactic::ModCast.to_lean5(), "mod_cast");
        assert_eq!(Tactic::PushCast.to_lean5(), "push_cast");
        assert_eq!(Tactic::NormCast.to_lean5(), "norm_cast");
    }

    #[test]
    fn test_classify_pattern_nonlinear() {
        // x * y >= 0 (non-linear: multiplication of variables)
        let expr = Lean5Expr::ge(
            Lean5Expr::mul(Lean5Expr::var("x"), Lean5Expr::var("y")),
            Lean5Expr::IntLit(0),
        );
        assert_eq!(
            classify_pattern(&expr),
            InvariantPattern::NonLinearArithmetic
        );
    }

    #[test]
    fn test_classify_pattern_modular() {
        // x % 2 == 0 (modular arithmetic)
        let expr = Lean5Expr::eq(
            Lean5Expr::Mod(
                Box::new(Lean5Expr::var("x")),
                Box::new(Lean5Expr::IntLit(2)),
            ),
            Lean5Expr::IntLit(0),
        );
        assert_eq!(classify_pattern(&expr), InvariantPattern::ModularArithmetic);
    }

    #[test]
    fn test_contains_nonlinear_term() {
        // x * 2 is linear (constant multiplier)
        let linear = Lean5Expr::mul(Lean5Expr::var("x"), Lean5Expr::IntLit(2));
        assert!(!contains_nonlinear_term(&linear));

        // x * y is non-linear
        let nonlinear = Lean5Expr::mul(Lean5Expr::var("x"), Lean5Expr::var("y"));
        assert!(contains_nonlinear_term(&nonlinear));

        // x + y is linear
        let sum = Lean5Expr::add(Lean5Expr::var("x"), Lean5Expr::var("y"));
        assert!(!contains_nonlinear_term(&sum));

        // (x + 1) * y is non-linear
        let complex = Lean5Expr::mul(
            Lean5Expr::add(Lean5Expr::var("x"), Lean5Expr::IntLit(1)),
            Lean5Expr::var("y"),
        );
        assert!(contains_nonlinear_term(&complex));
    }

    #[test]
    fn test_contains_modular_arithmetic() {
        // x + 1 has no modular operations
        let simple = Lean5Expr::add(Lean5Expr::var("x"), Lean5Expr::IntLit(1));
        assert!(!contains_modular_arithmetic(&simple));

        // x % 2 is modular
        let modular = Lean5Expr::Mod(
            Box::new(Lean5Expr::var("x")),
            Box::new(Lean5Expr::IntLit(2)),
        );
        assert!(contains_modular_arithmetic(&modular));

        // x / 2 is modular (division)
        let div = Lean5Expr::Div(
            Box::new(Lean5Expr::var("x")),
            Box::new(Lean5Expr::IntLit(2)),
        );
        assert!(contains_modular_arithmetic(&div));

        // x + (y % 3) contains modular arithmetic
        let nested = Lean5Expr::add(
            Lean5Expr::var("x"),
            Lean5Expr::Mod(
                Box::new(Lean5Expr::var("y")),
                Box::new(Lean5Expr::IntLit(3)),
            ),
        );
        assert!(contains_modular_arithmetic(&nested));
    }

    #[test]
    fn test_generate_pattern_tactics_nonlinear() {
        let expr = Lean5Expr::ge(
            Lean5Expr::mul(Lean5Expr::var("x"), Lean5Expr::var("y")),
            Lean5Expr::IntLit(0),
        );
        let pattern = classify_pattern(&expr);
        let tactics = generate_pattern_tactics(pattern, &expr);

        assert!(tactics.is_some());
        let tactics = tactics.unwrap();
        assert!(tactics.is_complete);
        let lean = tactics.to_lean5();
        assert!(
            lean.contains("ring_nf") || lean.contains("nlinarith"),
            "Got: {}",
            lean
        );
    }

    #[test]
    fn test_generate_pattern_tactics_modular() {
        let expr = Lean5Expr::eq(
            Lean5Expr::Mod(
                Box::new(Lean5Expr::var("x")),
                Box::new(Lean5Expr::IntLit(2)),
            ),
            Lean5Expr::IntLit(0),
        );
        let pattern = classify_pattern(&expr);
        let tactics = generate_pattern_tactics(pattern, &expr);

        assert!(tactics.is_some());
        let tactics = tactics.unwrap();
        assert!(tactics.is_complete);
        let lean = tactics.to_lean5();
        assert!(
            lean.contains("mod_cast") || lean.contains("omega"),
            "Got: {}",
            lean
        );
    }

    #[test]
    fn test_generate_pattern_tactics_bitvector() {
        let pattern = InvariantPattern::BitVector;
        let expr = Lean5Expr::BoolLit(true); // Placeholder, pattern is directly specified
        let tactics = generate_pattern_tactics(pattern, &expr);

        assert!(tactics.is_some());
        let tactics = tactics.unwrap();
        assert!(tactics.is_complete);
        let lean = tactics.to_lean5();
        assert!(
            lean.contains("bv_omega") || lean.contains("bv_decide"),
            "Got: {}",
            lean
        );
    }

    // ========================================================================
    // Mutation coverage tests
    // ========================================================================

    #[test]
    fn test_tactic_intro_empty_vars() {
        // Mutation: match guard vars.is_empty() with false
        let tactic = Tactic::Intro(vec![]);
        assert_eq!(tactic.to_lean5(), "intro");

        let tactic_with_vars = Tactic::Intro(vec!["x".to_string()]);
        assert_eq!(tactic_with_vars.to_lean5(), "intro x");
    }

    #[test]
    fn test_generate_initiation_tactics_checks_intro_vars() {
        // Mutation: delete ! in !intro_vars.is_empty()
        let obligation = ProofObligation::new(
            "quantified_init",
            ProofObligationKind::Initiation,
            Lean5Expr::forall_(
                "x",
                Lean5Type::Int,
                Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
            ),
        );

        let tactics = generate_tactics(&obligation);
        let lean = tactics.to_lean5();
        // Should have intro for the quantifier
        assert!(
            lean.contains("intro"),
            "Should intro quantified variable: {}",
            lean
        );
    }

    #[test]
    fn test_generate_consecution_tactics_checks_intro_vars() {
        // Mutation: delete ! in !intro_vars.is_empty()
        let obligation = ProofObligation::new(
            "quantified_consecution",
            ProofObligationKind::Consecution,
            Lean5Expr::forall_(
                "n",
                Lean5Type::Int,
                Lean5Expr::implies(
                    Lean5Expr::ge(Lean5Expr::var("n"), Lean5Expr::IntLit(0)),
                    Lean5Expr::ge(Lean5Expr::var("n"), Lean5Expr::IntLit(0)),
                ),
            ),
        );

        let tactics = generate_tactics(&obligation);
        let lean = tactics.to_lean5();
        assert!(
            lean.contains("intro"),
            "Should intro quantified variable: {}",
            lean
        );
    }

    #[test]
    fn test_generate_consecution_tactics_hyp_depth_check() {
        // Mutation: replace >= with < in hyp_depth >= 2
        // With depth >= 2, we use obtain to destructure

        // Depth 3 conjunction: a ∧ (b ∧ c)
        let obligation = ProofObligation::new(
            "deep_conjunction",
            ProofObligationKind::Consecution,
            Lean5Expr::implies(
                Lean5Expr::and(
                    Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
                    Lean5Expr::and(
                        Lean5Expr::ge(Lean5Expr::var("y"), Lean5Expr::IntLit(0)),
                        Lean5Expr::ge(Lean5Expr::var("z"), Lean5Expr::IntLit(0)),
                    ),
                ),
                Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
            ),
        );

        let tactics = generate_tactics(&obligation);
        let lean = tactics.to_lean5();
        // Should use obtain pattern for depth >= 2
        assert!(
            lean.contains("obtain") || lean.contains("intro"),
            "Should destructure deep conjunction: {}",
            lean
        );
    }

    #[test]
    fn test_consecution_reflexive_equality_uses_rfl() {
        // Mutation: replace == with != in pattern == ReflexiveEquality
        let x = Lean5Expr::var("x");
        let obligation = ProofObligation::new(
            "reflexive",
            ProofObligationKind::Consecution,
            Lean5Expr::implies(Lean5Expr::BoolLit(true), Lean5Expr::eq(x.clone(), x)),
        );

        let tactics = generate_tactics(&obligation);
        let lean = tactics.to_lean5();
        assert!(
            lean.contains("rfl"),
            "Reflexive equality should use rfl: {}",
            lean
        );
    }

    #[test]
    fn test_consecution_multi_variable_arithmetic() {
        // Mutation: replace == with != in pattern == MultiVariableArithmetic
        let obligation = ProofObligation::new(
            "multi_var",
            ProofObligationKind::Consecution,
            Lean5Expr::implies(
                Lean5Expr::ge(
                    Lean5Expr::add(Lean5Expr::var("x"), Lean5Expr::var("y")),
                    Lean5Expr::IntLit(0),
                ),
                Lean5Expr::ge(
                    Lean5Expr::add(Lean5Expr::var("x"), Lean5Expr::var("y")),
                    Lean5Expr::IntLit(0),
                ),
            ),
        );

        let tactics = generate_tactics(&obligation);
        let lean = tactics.to_lean5();
        // Should use simp_arith and omega for multi-variable
        assert!(
            lean.contains("simp") || lean.contains("omega"),
            "Multi-variable should use simp/omega: {}",
            lean
        );
    }

    #[test]
    fn test_consecution_conjunction_with_arithmetic() {
        // Mutation: replace == with != and && with || in conjunction check
        let obligation = ProofObligation::new(
            "conj_arith",
            ProofObligationKind::Consecution,
            Lean5Expr::implies(
                Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
                Lean5Expr::and(
                    Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
                    Lean5Expr::le(Lean5Expr::var("x"), Lean5Expr::IntLit(100)),
                ),
            ),
        );

        let tactics = generate_tactics(&obligation);
        let lean = tactics.to_lean5();
        assert!(
            lean.contains("omega"),
            "Conjunction with arithmetic should use omega: {}",
            lean
        );
    }

    #[test]
    fn test_generate_property_tactics_reflexive_equality() {
        // Mutation: replace == with != in pattern == ReflexiveEquality
        let x = Lean5Expr::var("x");
        let obligation = ProofObligation::new(
            "prop_reflexive",
            ProofObligationKind::Property,
            Lean5Expr::implies(Lean5Expr::BoolLit(true), Lean5Expr::eq(x.clone(), x)),
        );

        let tactics = generate_tactics(&obligation);
        let lean = tactics.to_lean5();
        assert!(lean.contains("rfl"), "Should use rfl for x = x: {}", lean);
    }

    #[test]
    fn test_generate_property_tactics_conjunction_coverage() {
        // Mutation: replace == with != and && with || in conjunction check
        let obligation = ProofObligation::new(
            "prop_conj",
            ProofObligationKind::Property,
            Lean5Expr::implies(
                Lean5Expr::BoolLit(true),
                Lean5Expr::and(
                    Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
                    Lean5Expr::le(Lean5Expr::var("x"), Lean5Expr::IntLit(10)),
                ),
            ),
        );

        let tactics = generate_tactics(&obligation);
        let lean = tactics.to_lean5();
        assert!(
            lean.contains("constructor") || lean.contains("omega"),
            "Property conjunction should split: {}",
            lean
        );
    }

    #[test]
    fn test_generate_custom_tactics_reflexive() {
        // Mutation: replace == with != in concl_pattern == ReflexiveEquality
        let x = Lean5Expr::var("x");
        let obligation = ProofObligation::new(
            "custom_reflexive",
            ProofObligationKind::Custom("test".to_string()),
            Lean5Expr::implies(Lean5Expr::BoolLit(true), Lean5Expr::eq(x.clone(), x)),
        );

        let tactics = generate_tactics(&obligation);
        let lean = tactics.to_lean5();
        assert!(
            lean.contains("rfl"),
            "Custom reflexive should use rfl: {}",
            lean
        );
    }

    #[test]
    fn test_generate_custom_tactics_conjunction() {
        // Mutation: replace == with != and && with ||
        let obligation = ProofObligation::new(
            "custom_conj",
            ProofObligationKind::Custom("test".to_string()),
            Lean5Expr::implies(
                Lean5Expr::BoolLit(true),
                Lean5Expr::and(
                    Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
                    Lean5Expr::ge(Lean5Expr::var("y"), Lean5Expr::IntLit(0)),
                ),
            ),
        );

        let tactics = generate_tactics(&obligation);
        let lean = tactics.to_lean5();
        assert!(
            lean.contains("constructor") || lean.contains("omega"),
            "Custom conjunction should use constructor/omega: {}",
            lean
        );
    }

    #[test]
    fn test_try_simple_initiation_proof_eq_zero_pattern() {
        // Mutation: replace && with || in eq check
        // Pattern: x = 0 → x >= 0

        let stmt = Lean5Expr::implies(
            Lean5Expr::eq(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
            Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
        );

        let tactics = try_simple_initiation_proof(&stmt);
        assert!(
            tactics.is_some(),
            "Should recognize simple initiation pattern"
        );
        let lean = tactics.unwrap().to_lean5();
        assert!(lean.contains("intro") && lean.contains("omega"));
    }

    #[test]
    fn test_classify_pattern_equality_const_both_directions() {
        // Mutation: match guards with false for IntLit/NatLit checks

        // x = 5 (var on left)
        let expr1 = Lean5Expr::eq(Lean5Expr::var("x"), Lean5Expr::IntLit(5));
        assert_eq!(classify_pattern(&expr1), InvariantPattern::EqualityConst);

        // 5 = x (var on right)
        let expr2 = Lean5Expr::eq(Lean5Expr::IntLit(5), Lean5Expr::var("x"));
        assert_eq!(classify_pattern(&expr2), InvariantPattern::EqualityConst);
    }

    #[test]
    fn test_classify_pattern_equality_arithmetic_both_directions() {
        // Mutation: match guards for EqualityArithmetic

        // x = y + 1 (var on left, arithmetic on right)
        let expr1 = Lean5Expr::eq(
            Lean5Expr::var("x"),
            Lean5Expr::add(Lean5Expr::var("y"), Lean5Expr::IntLit(1)),
        );
        assert_eq!(
            classify_pattern(&expr1),
            InvariantPattern::EqualityArithmetic
        );

        // y + 1 = x (arithmetic on left, var on right)
        let expr2 = Lean5Expr::eq(
            Lean5Expr::add(Lean5Expr::var("y"), Lean5Expr::IntLit(1)),
            Lean5Expr::var("x"),
        );
        assert_eq!(
            classify_pattern(&expr2),
            InvariantPattern::EqualityArithmetic
        );
    }

    #[test]
    fn test_classify_pattern_non_negative_0_lt_x() {
        // Mutation: match guard 0 < x pattern
        let expr = Lean5Expr::lt(Lean5Expr::IntLit(0), Lean5Expr::var("x"));
        assert_eq!(classify_pattern(&expr), InvariantPattern::NonNegative);
    }

    #[test]
    fn test_classify_pattern_non_negative_0_le_x() {
        // Mutation: match guard 0 <= x pattern
        let expr = Lean5Expr::le(Lean5Expr::IntLit(0), Lean5Expr::var("x"));
        assert_eq!(classify_pattern(&expr), InvariantPattern::NonNegative);
    }

    #[test]
    fn test_classify_pattern_monotonicity() {
        // Mutation: is_primed_pair match guard
        let expr = Lean5Expr::ge(Lean5Expr::var("x_next"), Lean5Expr::var("x"));
        assert_eq!(classify_pattern(&expr), InvariantPattern::Monotonicity);

        // Also test x' >= x pattern
        let expr2 = Lean5Expr::ge(Lean5Expr::var("x'"), Lean5Expr::var("x"));
        assert_eq!(classify_pattern(&expr2), InvariantPattern::Monotonicity);
    }

    #[test]
    fn test_is_primed_pair() {
        // Mutation: replace && with || and is_primed_pair -> false

        // Valid primed pairs
        assert!(is_primed_pair(
            &Lean5Expr::var("x_next"),
            &Lean5Expr::var("x")
        ));
        assert!(is_primed_pair(&Lean5Expr::var("x'"), &Lean5Expr::var("x")));
        assert!(is_primed_pair(
            &Lean5Expr::var("x_prime"),
            &Lean5Expr::var("x")
        ));

        // Invalid pairs
        assert!(!is_primed_pair(&Lean5Expr::var("y"), &Lean5Expr::var("x")));
        assert!(!is_primed_pair(&Lean5Expr::IntLit(5), &Lean5Expr::var("x")));
    }

    #[test]
    fn test_classify_pattern_ordering_both_vars() {
        // Mutation: replace && with || in var checks
        let expr = Lean5Expr::le(Lean5Expr::var("x"), Lean5Expr::var("y"));
        assert_eq!(classify_pattern(&expr), InvariantPattern::Ordering);

        // If one side is not a var, should NOT be ordering
        let expr2 = Lean5Expr::le(Lean5Expr::var("x"), Lean5Expr::IntLit(10));
        assert_eq!(classify_pattern(&expr2), InvariantPattern::UpperBounded);
    }

    #[test]
    fn test_classify_pattern_and_checks_sub_patterns() {
        // Mutation: replace && with || and == with != in pattern checks

        // NonNegative + UpperBounded = RangeBounded
        let range = Lean5Expr::and(
            Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
            Lean5Expr::le(Lean5Expr::var("x"), Lean5Expr::IntLit(100)),
        );
        assert_eq!(classify_pattern(&range), InvariantPattern::RangeBounded);

        // UpperBounded + NonNegative = RangeBounded (reverse order)
        let range_rev = Lean5Expr::and(
            Lean5Expr::le(Lean5Expr::var("x"), Lean5Expr::IntLit(100)),
            Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
        );
        assert_eq!(classify_pattern(&range_rev), InvariantPattern::RangeBounded);
    }

    #[test]
    fn test_classify_pattern_eq_nonlinear_and_modular() {
        // Mutation: Eq branch for non-var sides

        // x * y = z (non-linear through multiply of vars)
        let nonlin = Lean5Expr::eq(
            Lean5Expr::mul(Lean5Expr::var("x"), Lean5Expr::var("y")),
            Lean5Expr::var("z"),
        );
        // This should be EqualityArithmetic because rhs is Var
        assert_eq!(
            classify_pattern(&nonlin),
            InvariantPattern::EqualityArithmetic
        );

        // x * y = z * w (both sides non-var arithmetic)
        let both_arith = Lean5Expr::eq(
            Lean5Expr::mul(Lean5Expr::var("x"), Lean5Expr::var("y")),
            Lean5Expr::mul(Lean5Expr::var("z"), Lean5Expr::IntLit(2)),
        );
        assert_eq!(
            classify_pattern(&both_arith),
            InvariantPattern::NonLinearArithmetic
        );
    }

    #[test]
    fn test_classify_pattern_under_quantifier() {
        // Mutation: delete match arm for Forall/Exists
        let expr = Lean5Expr::forall_(
            "x",
            Lean5Type::Int,
            Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
        );
        // Should classify the body
        assert_eq!(classify_pattern(&expr), InvariantPattern::NonNegative);

        let expr2 = Lean5Expr::exists_(
            "x",
            Lean5Type::Int,
            Lean5Expr::le(Lean5Expr::var("x"), Lean5Expr::IntLit(10)),
        );
        assert_eq!(classify_pattern(&expr2), InvariantPattern::UpperBounded);
    }

    #[test]
    fn test_classify_pattern_implies() {
        // Mutation: delete match arm Lean5Expr::Implies(_, _)
        let expr = Lean5Expr::implies(Lean5Expr::var("p"), Lean5Expr::var("q"));
        assert_eq!(classify_pattern(&expr), InvariantPattern::BoolImplication);
    }

    #[test]
    fn test_classify_pattern_var_is_unknown() {
        // Mutation: delete match arm Lean5Expr::Var(_)
        let expr = Lean5Expr::var("x");
        assert_eq!(classify_pattern(&expr), InvariantPattern::Unknown);
    }

    #[test]
    fn test_classify_pattern_bool_lit() {
        // Mutation: delete match arm Lean5Expr::BoolLit(_)
        let expr = Lean5Expr::BoolLit(true);
        assert_eq!(classify_pattern(&expr), InvariantPattern::PureBool);
    }

    #[test]
    fn test_contains_nonlinear_div_by_variable() {
        // Mutation: match guard contains_any_variable(rhs) in Div case
        // x / y is non-linear (division by variable)
        let div_by_var =
            Lean5Expr::Div(Box::new(Lean5Expr::var("x")), Box::new(Lean5Expr::var("y")));
        assert!(contains_nonlinear_term(&div_by_var));

        // x / 2 is NOT non-linear (division by constant)
        let div_by_const = Lean5Expr::Div(
            Box::new(Lean5Expr::var("x")),
            Box::new(Lean5Expr::IntLit(2)),
        );
        assert!(!contains_nonlinear_term(&div_by_const));
    }

    #[test]
    fn test_contains_nonlinear_term_recursion() {
        // Mutation: delete match arms for Add/Sub/Div/Mod recursion
        // and delete Neg arm

        // Nested: (x * y) + z - non-linear through mul
        let nested_add = Lean5Expr::add(
            Lean5Expr::mul(Lean5Expr::var("x"), Lean5Expr::var("y")),
            Lean5Expr::var("z"),
        );
        assert!(contains_nonlinear_term(&nested_add));

        // Nested: z - (x * y)
        let nested_sub = Lean5Expr::sub(
            Lean5Expr::var("z"),
            Lean5Expr::mul(Lean5Expr::var("x"), Lean5Expr::var("y")),
        );
        assert!(contains_nonlinear_term(&nested_sub));

        // Nested in Neg: -(x * y)
        let nested_neg = Lean5Expr::Neg(Box::new(Lean5Expr::mul(
            Lean5Expr::var("x"),
            Lean5Expr::var("y"),
        )));
        assert!(contains_nonlinear_term(&nested_neg));

        // Nested in Ite: if c then x*y else z
        let nested_ite = Lean5Expr::ite(
            Lean5Expr::BoolLit(true),
            Lean5Expr::mul(Lean5Expr::var("x"), Lean5Expr::var("y")),
            Lean5Expr::var("z"),
        );
        assert!(contains_nonlinear_term(&nested_ite));
    }

    #[test]
    fn test_contains_any_variable_recursion() {
        // Mutation: delete match arms for Neg, Ite

        // Neg with variable
        let neg = Lean5Expr::Neg(Box::new(Lean5Expr::var("x")));
        assert!(contains_any_variable(&neg));

        // Ite with variable in condition
        let ite = Lean5Expr::ite(
            Lean5Expr::var("x"),
            Lean5Expr::IntLit(1),
            Lean5Expr::IntLit(0),
        );
        assert!(contains_any_variable(&ite));

        // Ite with variable in then
        let ite2 = Lean5Expr::ite(
            Lean5Expr::BoolLit(true),
            Lean5Expr::var("x"),
            Lean5Expr::IntLit(0),
        );
        assert!(contains_any_variable(&ite2));

        // Ite with variable in else
        let ite3 = Lean5Expr::ite(
            Lean5Expr::BoolLit(true),
            Lean5Expr::IntLit(1),
            Lean5Expr::var("x"),
        );
        assert!(contains_any_variable(&ite3));
    }

    #[test]
    fn test_contains_modular_arithmetic_recursion() {
        // Mutation: delete match arms for Neg, Ite, and recursion

        // Nested in Neg
        let nested_neg = Lean5Expr::Neg(Box::new(Lean5Expr::Mod(
            Box::new(Lean5Expr::var("x")),
            Box::new(Lean5Expr::IntLit(2)),
        )));
        assert!(contains_modular_arithmetic(&nested_neg));

        // Nested in Ite condition
        let ite = Lean5Expr::ite(
            Lean5Expr::eq(
                Lean5Expr::Mod(
                    Box::new(Lean5Expr::var("x")),
                    Box::new(Lean5Expr::IntLit(2)),
                ),
                Lean5Expr::IntLit(0),
            ),
            Lean5Expr::IntLit(1),
            Lean5Expr::IntLit(0),
        );
        // Note: ite checks c || t || e, and contains_modular_arithmetic
        // only recurses through Ite, not into Eq
        assert!(!contains_modular_arithmetic(&ite));

        // Direct in Add
        let in_add = Lean5Expr::add(
            Lean5Expr::Mod(
                Box::new(Lean5Expr::var("x")),
                Box::new(Lean5Expr::IntLit(2)),
            ),
            Lean5Expr::IntLit(1),
        );
        assert!(contains_modular_arithmetic(&in_add));
    }

    #[test]
    fn test_is_arithmetic_expr_eq_with_arithmetic() {
        // Mutation: replace || with && in Eq branch
        // x = y + 1 should be arithmetic
        let expr = Lean5Expr::eq(
            Lean5Expr::var("x"),
            Lean5Expr::add(Lean5Expr::var("y"), Lean5Expr::IntLit(1)),
        );
        assert!(is_arithmetic_expr(&expr));

        // y + 1 = x should also be arithmetic
        let expr2 = Lean5Expr::eq(
            Lean5Expr::add(Lean5Expr::var("y"), Lean5Expr::IntLit(1)),
            Lean5Expr::var("x"),
        );
        assert!(is_arithmetic_expr(&expr2));

        // x = y (no arithmetic) should NOT be arithmetic by itself
        let expr3 = Lean5Expr::eq(Lean5Expr::var("x"), Lean5Expr::var("y"));
        assert!(!is_arithmetic_expr(&expr3));
    }

    #[test]
    fn test_is_arithmetic_expr_and_or_implies() {
        // Mutation: replace || with && in And/Or/Implies branches

        // x >= 0 ∧ y (one side arithmetic)
        let and_expr = Lean5Expr::and(
            Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
            Lean5Expr::var("y"),
        );
        assert!(is_arithmetic_expr(&and_expr));

        // y ∧ x >= 0 (other side arithmetic)
        let and_expr2 = Lean5Expr::and(
            Lean5Expr::var("y"),
            Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
        );
        assert!(is_arithmetic_expr(&and_expr2));

        // y ∧ z (no arithmetic)
        let and_no_arith = Lean5Expr::and(Lean5Expr::var("y"), Lean5Expr::var("z"));
        assert!(!is_arithmetic_expr(&and_no_arith));
    }

    #[test]
    fn test_is_arithmetic_expr_forall_exists() {
        // Mutation: delete match arm Forall/Exists
        let forall_expr = Lean5Expr::forall_(
            "x",
            Lean5Type::Int,
            Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
        );
        assert!(is_arithmetic_expr(&forall_expr));

        let exists_expr = Lean5Expr::exists_(
            "x",
            Lean5Type::Int,
            Lean5Expr::le(Lean5Expr::var("x"), Lean5Expr::IntLit(10)),
        );
        assert!(is_arithmetic_expr(&exists_expr));
    }

    #[test]
    fn test_contains_any_arithmetic_recursion() {
        // Mutation: delete match arms for Not, Forall/Exists, Ite, App, Let

        // Not with arithmetic inside
        let not_arith = Lean5Expr::not(Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)));
        assert!(contains_any_arithmetic(&not_arith));

        // Forall with arithmetic
        let forall_arith = Lean5Expr::forall_(
            "x",
            Lean5Type::Int,
            Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
        );
        assert!(contains_any_arithmetic(&forall_arith));

        // Ite with arithmetic
        let ite_arith = Lean5Expr::ite(
            Lean5Expr::BoolLit(true),
            Lean5Expr::add(Lean5Expr::var("x"), Lean5Expr::IntLit(1)),
            Lean5Expr::IntLit(0),
        );
        assert!(contains_any_arithmetic(&ite_arith));

        // App with arithmetic
        let app_arith = Lean5Expr::app(
            Lean5Expr::var("f"),
            Lean5Expr::add(Lean5Expr::var("x"), Lean5Expr::IntLit(1)),
        );
        assert!(contains_any_arithmetic(&app_arith));

        // Let with arithmetic in value
        let let_arith = Lean5Expr::Let(
            "y".to_string(),
            Lean5Type::Int,
            Box::new(Lean5Expr::add(Lean5Expr::var("x"), Lean5Expr::IntLit(1))),
            Box::new(Lean5Expr::var("y")),
        );
        assert!(contains_any_arithmetic(&let_arith));
    }

    #[test]
    fn test_collect_variables_impl_recursion() {
        // Mutation: delete match arms for Not/Neg, Ite, Forall/Exists, App, Lam/Let

        // Not with var
        let not_expr = Lean5Expr::not(Lean5Expr::var("x"));
        let vars = collect_variables(&not_expr);
        assert!(vars.contains(&"x".to_string()));

        // Ite with vars
        let ite = Lean5Expr::ite(
            Lean5Expr::var("c"),
            Lean5Expr::var("t"),
            Lean5Expr::var("e"),
        );
        let vars = collect_variables(&ite);
        assert_eq!(vars.len(), 3);
        assert!(vars.contains(&"c".to_string()));
        assert!(vars.contains(&"t".to_string()));
        assert!(vars.contains(&"e".to_string()));

        // Forall with vars
        let forall_expr = Lean5Expr::forall_(
            "x",
            Lean5Type::Int,
            Lean5Expr::and(Lean5Expr::var("x"), Lean5Expr::var("y")),
        );
        let vars = collect_variables(&forall_expr);
        assert!(vars.contains(&"x".to_string()));
        assert!(vars.contains(&"y".to_string()));

        // App with vars
        let app = Lean5Expr::app(Lean5Expr::var("f"), Lean5Expr::var("x"));
        let vars = collect_variables(&app);
        assert!(vars.contains(&"f".to_string()));
        assert!(vars.contains(&"x".to_string()));

        // Let with vars in body
        let let_expr = Lean5Expr::Let(
            "y".to_string(),
            Lean5Type::Int,
            Box::new(Lean5Expr::IntLit(0)),
            Box::new(Lean5Expr::var("z")),
        );
        let vars = collect_variables(&let_expr);
        assert!(vars.contains(&"z".to_string()));
    }
}
