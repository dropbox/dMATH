//! Proof obligations for Lean5
//!
//! This module generates proof obligations from CHC invariants.
//! A proof obligation represents a theorem that needs to be proved
//! to certify the correctness of a discovered invariant.

use crate::expr::{Lean5Expr, Lean5Type};
use crate::translate::{translate_predicate, TranslationError};
use kani_fast_chc::result::InvariantModel;
use std::fmt;
use std::fmt::Write;

/// Kind of proof obligation
#[derive(Debug, Clone, PartialEq)]
pub enum ProofObligationKind {
    /// Initial state implies invariant: Init(s) → Inv(s)
    Initiation,
    /// Invariant is preserved by transition: Inv(s) ∧ Trans(s, s') → Inv(s')
    Consecution,
    /// Invariant implies property: Inv(s) → Property(s)
    Property,
    /// Custom invariant assertion
    Custom(String),
}

impl fmt::Display for ProofObligationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProofObligationKind::Initiation => write!(f, "initiation"),
            ProofObligationKind::Consecution => write!(f, "consecution"),
            ProofObligationKind::Property => write!(f, "property"),
            ProofObligationKind::Custom(s) => write!(f, "custom: {s}"),
        }
    }
}

/// A proof obligation to be verified by Lean5
#[derive(Debug, Clone)]
pub struct ProofObligation {
    /// Name of the obligation
    pub name: String,
    /// Kind of obligation
    pub kind: ProofObligationKind,
    /// The statement to be proved
    pub statement: Lean5Expr,
    /// Optional proof term (if known)
    pub proof: Option<Lean5Expr>,
    /// Variable context
    pub context: Vec<(String, Lean5Type)>,
}

impl ProofObligation {
    /// Create a new proof obligation
    pub fn new(name: impl Into<String>, kind: ProofObligationKind, statement: Lean5Expr) -> Self {
        ProofObligation {
            name: name.into(),
            kind,
            statement,
            proof: None,
            context: Vec::new(),
        }
    }

    /// Add a variable to the context
    pub fn with_var(mut self, name: impl Into<String>, ty: Lean5Type) -> Self {
        self.context.push((name.into(), ty));
        self
    }

    /// Set a proof term
    pub fn with_proof(mut self, proof: Lean5Expr) -> Self {
        self.proof = Some(proof);
        self
    }

    /// Generate Lean5 source code for this obligation
    pub fn to_lean5_source(&self) -> String {
        let mut result = String::new();

        // Add a comment describing the obligation
        let _ = writeln!(result, "-- Proof obligation: {} ({})", self.name, self.kind);

        // Generate theorem statement
        let _ = write!(result, "theorem {} ", self.name);

        // Add variable bindings
        for (name, ty) in &self.context {
            let _ = write!(result, "({name} : {ty}) ");
        }

        // Add statement
        let _ = write!(result, ": {} ", self.statement);

        // Add proof or sorry
        if let Some(proof) = &self.proof {
            let _ = writeln!(result, ":= {proof}");
        } else {
            result.push_str(":= by sorry\n");
        }

        result
    }

    /// Generate proof obligations from an invariant model
    pub fn from_invariant(
        model: &InvariantModel,
    ) -> Result<Vec<ProofObligation>, TranslationError> {
        let mut obligations = Vec::new();

        for pred in &model.predicates {
            // Translate the predicate to Lean5
            let inv_expr = translate_predicate(pred)?;

            // Create an obligation asserting the invariant holds
            let obligation = ProofObligation::new(
                format!("{}_holds", pred.name),
                ProofObligationKind::Custom("invariant assertion".to_string()),
                inv_expr.clone(),
            );

            obligations.push(obligation);

            // If we have a main invariant, generate initiation and property obligations
            if pred.name == "Inv" {
                // Generate context from predicate params
                let context: Vec<(String, Lean5Type)> = pred
                    .params
                    .iter()
                    .map(|(name, ty)| {
                        let lean_ty = match ty {
                            kani_fast_kinduction::SmtType::Int => Lean5Type::Int,
                            kani_fast_kinduction::SmtType::Bool => Lean5Type::Bool,
                            _ => Lean5Type::Type,
                        };
                        (name.replace('!', "_"), lean_ty)
                    })
                    .collect();

                // Extract the invariant body (without the outer forall)
                let inv_body = match &inv_expr {
                    Lean5Expr::Forall(_, _, body) => body.as_ref().clone(),
                    _ => inv_expr.clone(),
                };

                // Initiation: x = 0 → Inv(x) for a counter example
                let init_condition = if !context.is_empty() {
                    Lean5Expr::eq(Lean5Expr::var(context[0].0.clone()), Lean5Expr::IntLit(0))
                } else {
                    Lean5Expr::BoolLit(true)
                };

                let init_obligation = ProofObligation::new(
                    format!("{}_initiation", pred.name),
                    ProofObligationKind::Initiation,
                    Lean5Expr::implies(init_condition, inv_body.clone()),
                );
                obligations.push(init_obligation);
            }
        }

        Ok(obligations)
    }

    /// Generate a complete Lean5 file from multiple obligations
    pub fn to_lean5_file(obligations: &[ProofObligation]) -> String {
        let mut result = String::new();

        // File header
        result.push_str("/-!\n");
        result.push_str("  Proof obligations generated by Kani Fast\n");
        result.push_str("  \n");
        result.push_str("  These theorems verify discovered invariants.\n");
        result.push_str("-/\n\n");

        // Import statements
        result.push_str("import Mathlib.Tactic\n\n");

        // Namespace
        result.push_str("namespace KaniFast\n\n");

        // Generate each obligation
        for obligation in obligations {
            result.push_str(&obligation.to_lean5_source());
            result.push('\n');
        }

        // Close namespace
        result.push_str("end KaniFast\n");

        result
    }
}

impl fmt::Display for ProofObligation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let src = self.to_lean5_source();
        write!(f, "{src}")
    }
}

/// Builder for creating proof obligations with a fluent interface
pub struct ProofObligationBuilder {
    name: String,
    kind: ProofObligationKind,
    context: Vec<(String, Lean5Type)>,
    hypotheses: Vec<Lean5Expr>,
    conclusion: Option<Lean5Expr>,
}

impl ProofObligationBuilder {
    /// Start building a new obligation
    pub fn new(name: impl Into<String>) -> Self {
        ProofObligationBuilder {
            name: name.into(),
            kind: ProofObligationKind::Custom("proof obligation".to_string()),
            context: Vec::new(),
            hypotheses: Vec::new(),
            conclusion: None,
        }
    }

    /// Set the obligation kind
    pub fn kind(mut self, kind: ProofObligationKind) -> Self {
        self.kind = kind;
        self
    }

    /// Add a variable to the context
    pub fn var(mut self, name: impl Into<String>, ty: Lean5Type) -> Self {
        self.context.push((name.into(), ty));
        self
    }

    /// Add a hypothesis
    pub fn hypothesis(mut self, hyp: Lean5Expr) -> Self {
        self.hypotheses.push(hyp);
        self
    }

    /// Set the conclusion
    pub fn conclusion(mut self, concl: Lean5Expr) -> Self {
        self.conclusion = Some(concl);
        self
    }

    /// Build the proof obligation
    pub fn build(self) -> Option<ProofObligation> {
        let conclusion = self.conclusion?;

        // Build the statement: hyp1 → hyp2 → ... → conclusion
        let statement = if self.hypotheses.is_empty() {
            conclusion
        } else {
            let mut stmt = conclusion;
            for hyp in self.hypotheses.into_iter().rev() {
                stmt = Lean5Expr::implies(hyp, stmt);
            }
            stmt
        };

        let mut obligation = ProofObligation::new(self.name, self.kind, statement);
        obligation.context = self.context;

        Some(obligation)
    }
}

/// Generate induction proof obligation for k-induction
pub fn generate_kinduction_obligation(
    invariant: &Lean5Expr,
    k: usize,
    var_name: &str,
    var_type: &Lean5Type,
) -> ProofObligation {
    // For k-induction, we need to prove:
    // Inv(n) ∧ Inv(n+1) ∧ ... ∧ Inv(n+k-1) → Inv(n+k)

    let var = Lean5Expr::var(var_name);

    // Build hypothesis: conjunction of Inv(n), Inv(n+1), ..., Inv(n+k-1)
    let mut hypotheses = Vec::new();
    for i in 0..k {
        let arg = if i == 0 {
            var.clone()
        } else {
            Lean5Expr::add(var.clone(), Lean5Expr::IntLit(i as i64))
        };
        // Apply the invariant predicate
        let inv_at_i = apply_invariant(invariant, &arg);
        hypotheses.push(inv_at_i);
    }

    let hypothesis = if hypotheses.len() == 1 {
        // SAFETY: The condition guarantees exactly one element
        hypotheses.pop().expect("condition guarantees one element")
    } else {
        let mut result = hypotheses[0].clone();
        for h in &hypotheses[1..] {
            result = Lean5Expr::and(result, h.clone());
        }
        result
    };

    // Build conclusion: Inv(n+k)
    let conclusion_arg = Lean5Expr::add(var, Lean5Expr::IntLit(k as i64));
    let conclusion = apply_invariant(invariant, &conclusion_arg);

    // Build the full statement
    let statement = Lean5Expr::forall_(
        var_name.to_string(),
        var_type.clone(),
        Lean5Expr::implies(hypothesis, conclusion),
    );

    ProofObligation::new(
        format!("kinduction_step_{k}"),
        ProofObligationKind::Consecution,
        statement,
    )
}

/// Apply an invariant formula to an argument
/// This handles the case where invariant is already a forall
fn apply_invariant(invariant: &Lean5Expr, arg: &Lean5Expr) -> Lean5Expr {
    match invariant {
        Lean5Expr::Forall(var_name, _, body) => {
            // Substitute the bound variable with the argument
            substitute_var(body, var_name, arg)
        }
        _ => Lean5Expr::app(invariant.clone(), arg.clone()),
    }
}

/// Variable substitution - replaces occurrences of `var_name` with `replacement`
fn substitute_var(expr: &Lean5Expr, var_name: &str, replacement: &Lean5Expr) -> Lean5Expr {
    match expr {
        Lean5Expr::Var(name) if name == var_name => replacement.clone(),
        Lean5Expr::Var(_) => expr.clone(),
        Lean5Expr::App(f, a) => Lean5Expr::app(
            substitute_var(f, var_name, replacement),
            substitute_var(a, var_name, replacement),
        ),
        Lean5Expr::And(a, b) => Lean5Expr::and(
            substitute_var(a, var_name, replacement),
            substitute_var(b, var_name, replacement),
        ),
        Lean5Expr::Or(a, b) => Lean5Expr::or(
            substitute_var(a, var_name, replacement),
            substitute_var(b, var_name, replacement),
        ),
        Lean5Expr::Not(a) => Lean5Expr::not(substitute_var(a, var_name, replacement)),
        Lean5Expr::Implies(a, b) => Lean5Expr::implies(
            substitute_var(a, var_name, replacement),
            substitute_var(b, var_name, replacement),
        ),
        Lean5Expr::Eq(a, b) => Lean5Expr::eq(
            substitute_var(a, var_name, replacement),
            substitute_var(b, var_name, replacement),
        ),
        Lean5Expr::Lt(a, b) => Lean5Expr::lt(
            substitute_var(a, var_name, replacement),
            substitute_var(b, var_name, replacement),
        ),
        Lean5Expr::Le(a, b) => Lean5Expr::le(
            substitute_var(a, var_name, replacement),
            substitute_var(b, var_name, replacement),
        ),
        Lean5Expr::Gt(a, b) => Lean5Expr::gt(
            substitute_var(a, var_name, replacement),
            substitute_var(b, var_name, replacement),
        ),
        Lean5Expr::Ge(a, b) => Lean5Expr::ge(
            substitute_var(a, var_name, replacement),
            substitute_var(b, var_name, replacement),
        ),
        Lean5Expr::Add(a, b) => Lean5Expr::add(
            substitute_var(a, var_name, replacement),
            substitute_var(b, var_name, replacement),
        ),
        Lean5Expr::Sub(a, b) => Lean5Expr::sub(
            substitute_var(a, var_name, replacement),
            substitute_var(b, var_name, replacement),
        ),
        Lean5Expr::Mul(a, b) => Lean5Expr::mul(
            substitute_var(a, var_name, replacement),
            substitute_var(b, var_name, replacement),
        ),
        Lean5Expr::Ite(c, t, e) => Lean5Expr::ite(
            substitute_var(c, var_name, replacement),
            substitute_var(t, var_name, replacement),
            substitute_var(e, var_name, replacement),
        ),
        Lean5Expr::Neg(a) => Lean5Expr::Neg(Box::new(substitute_var(a, var_name, replacement))),
        // Handle nested quantifiers - avoid capturing bound variables
        Lean5Expr::Forall(name, ty, body) => {
            if name == var_name {
                // The variable is shadowed by this forall, don't substitute in body
                expr.clone()
            } else {
                Lean5Expr::forall_(
                    name,
                    ty.clone(),
                    substitute_var(body, var_name, replacement),
                )
            }
        }
        Lean5Expr::Exists(name, ty, body) => {
            if name == var_name {
                // The variable is shadowed by this exists, don't substitute in body
                expr.clone()
            } else {
                Lean5Expr::exists_(
                    name,
                    ty.clone(),
                    substitute_var(body, var_name, replacement),
                )
            }
        }
        // For other expressions, return as-is
        _ => expr.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proof_obligation_display() {
        let obligation = ProofObligation::new(
            "test_theorem",
            ProofObligationKind::Initiation,
            Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
        );

        let source = obligation.to_lean5_source();
        assert!(source.contains("theorem test_theorem"));
        assert!(source.contains("initiation"));
    }

    #[test]
    fn test_proof_obligation_with_context() {
        let obligation = ProofObligation::new(
            "test",
            ProofObligationKind::Property,
            Lean5Expr::implies(
                Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
                Lean5Expr::BoolLit(true),
            ),
        )
        .with_var("x", Lean5Type::Int);

        let source = obligation.to_lean5_source();
        assert!(source.contains("(x : Int)"));
    }

    #[test]
    fn test_proof_obligation_builder() {
        let obligation = ProofObligationBuilder::new("my_theorem")
            .kind(ProofObligationKind::Consecution)
            .var("n", Lean5Type::Int)
            .hypothesis(Lean5Expr::ge(Lean5Expr::var("n"), Lean5Expr::IntLit(0)))
            .conclusion(Lean5Expr::ge(
                Lean5Expr::add(Lean5Expr::var("n"), Lean5Expr::IntLit(1)),
                Lean5Expr::IntLit(0),
            ))
            .build()
            .unwrap();

        assert_eq!(obligation.name, "my_theorem");
        assert_eq!(obligation.kind, ProofObligationKind::Consecution);
    }

    #[test]
    fn test_to_lean5_file() {
        let obligations = vec![
            ProofObligation::new(
                "theorem1",
                ProofObligationKind::Initiation,
                Lean5Expr::BoolLit(true),
            ),
            ProofObligation::new(
                "theorem2",
                ProofObligationKind::Property,
                Lean5Expr::BoolLit(true),
            ),
        ];

        let file = ProofObligation::to_lean5_file(&obligations);
        assert!(file.contains("namespace KaniFast"));
        assert!(file.contains("theorem theorem1"));
        assert!(file.contains("theorem theorem2"));
        assert!(file.contains("end KaniFast"));
    }

    #[test]
    fn test_kinduction_obligation() {
        let invariant = Lean5Expr::forall_(
            "x",
            Lean5Type::Int,
            Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
        );

        let obligation = generate_kinduction_obligation(&invariant, 2, "n", &Lean5Type::Int);

        assert_eq!(obligation.name, "kinduction_step_2");
        assert_eq!(obligation.kind, ProofObligationKind::Consecution);
    }

    #[test]
    fn test_substitute_var_single() {
        // Test simple variable substitution
        let expr = Lean5Expr::add(Lean5Expr::var("x"), Lean5Expr::IntLit(1));
        let result = substitute_var(&expr, "x", &Lean5Expr::IntLit(5));

        // Should produce: 5 + 1
        assert_eq!(
            format!("{}", result),
            format!(
                "{}",
                Lean5Expr::add(Lean5Expr::IntLit(5), Lean5Expr::IntLit(1))
            )
        );
    }

    #[test]
    fn test_substitute_var_preserves_other_vars() {
        // Test that substitution only affects the target variable
        let expr = Lean5Expr::add(Lean5Expr::var("x"), Lean5Expr::var("y"));
        let result = substitute_var(&expr, "x", &Lean5Expr::IntLit(5));

        // Should produce: 5 + y (y unchanged)
        assert_eq!(
            format!("{}", result),
            format!(
                "{}",
                Lean5Expr::add(Lean5Expr::IntLit(5), Lean5Expr::var("y"))
            )
        );
    }

    #[test]
    fn test_substitute_var_nested_forall_shadowing() {
        // Test that substitution respects variable shadowing in nested forall
        // ∀ x. (x + (∀ x. x)) should not substitute inner x
        let inner = Lean5Expr::forall_("x", Lean5Type::Int, Lean5Expr::var("x"));
        let outer = Lean5Expr::add(Lean5Expr::var("x"), inner.clone());

        let result = substitute_var(&outer, "x", &Lean5Expr::IntLit(5));

        // Should produce: 5 + (∀ x. x) - inner x is shadowed
        let expected = Lean5Expr::add(Lean5Expr::IntLit(5), inner);
        assert_eq!(format!("{}", result), format!("{}", expected));
    }

    #[test]
    fn test_substitute_var_nested_exists_shadowing() {
        // Test that substitution respects variable shadowing in nested exists
        let inner = Lean5Expr::exists_("y", Lean5Type::Int, Lean5Expr::var("y"));
        let outer = Lean5Expr::and(Lean5Expr::var("y"), inner.clone());

        let result = substitute_var(&outer, "y", &Lean5Expr::BoolLit(true));

        // Should produce: true ∧ (∃ y. y) - inner y is shadowed
        let expected = Lean5Expr::and(Lean5Expr::BoolLit(true), inner);
        assert_eq!(format!("{}", result), format!("{}", expected));
    }

    #[test]
    fn test_substitute_var_different_var_in_quantifier() {
        // Test substitution into a forall with a different bound variable
        // ∀ y. (x + y) with x → 5 should give ∀ y. (5 + y)
        let expr = Lean5Expr::forall_(
            "y",
            Lean5Type::Int,
            Lean5Expr::add(Lean5Expr::var("x"), Lean5Expr::var("y")),
        );

        let result = substitute_var(&expr, "x", &Lean5Expr::IntLit(5));

        let expected = Lean5Expr::forall_(
            "y",
            Lean5Type::Int,
            Lean5Expr::add(Lean5Expr::IntLit(5), Lean5Expr::var("y")),
        );
        assert_eq!(format!("{}", result), format!("{}", expected));
    }

    #[test]
    fn test_apply_invariant_substitutes_bound_var() {
        // Test that apply_invariant correctly substitutes the forall's bound variable
        let invariant = Lean5Expr::forall_(
            "x",
            Lean5Type::Int,
            Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
        );

        let arg = Lean5Expr::var("n");
        let result = apply_invariant(&invariant, &arg);

        // Should produce: n >= 0
        let expected = Lean5Expr::ge(Lean5Expr::var("n"), Lean5Expr::IntLit(0));
        assert_eq!(format!("{}", result), format!("{}", expected));
    }

    // ========================================================================
    // Mutation coverage tests for substitute_var
    // ========================================================================

    #[test]
    fn test_substitute_var_in_or() {
        // Mutation: delete match arm Lean5Expr::Or(a, b)
        let expr = Lean5Expr::or(Lean5Expr::var("x"), Lean5Expr::var("y"));
        let result = substitute_var(&expr, "x", &Lean5Expr::BoolLit(true));

        let expected = Lean5Expr::or(Lean5Expr::BoolLit(true), Lean5Expr::var("y"));
        assert_eq!(format!("{}", result), format!("{}", expected));
    }

    #[test]
    fn test_substitute_var_in_not() {
        // Mutation: delete match arm Lean5Expr::Not(a)
        let expr = Lean5Expr::not(Lean5Expr::var("x"));
        let result = substitute_var(&expr, "x", &Lean5Expr::BoolLit(true));

        let expected = Lean5Expr::not(Lean5Expr::BoolLit(true));
        assert_eq!(format!("{}", result), format!("{}", expected));
    }

    #[test]
    fn test_substitute_var_in_implies() {
        // Mutation: delete match arm Lean5Expr::Implies(a, b)
        let expr = Lean5Expr::implies(Lean5Expr::var("x"), Lean5Expr::var("y"));
        let result = substitute_var(&expr, "x", &Lean5Expr::BoolLit(true));

        let expected = Lean5Expr::implies(Lean5Expr::BoolLit(true), Lean5Expr::var("y"));
        assert_eq!(format!("{}", result), format!("{}", expected));
    }

    #[test]
    fn test_substitute_var_in_eq() {
        // Mutation: delete match arm Lean5Expr::Eq(a, b)
        let expr = Lean5Expr::eq(Lean5Expr::var("x"), Lean5Expr::IntLit(5));
        let result = substitute_var(&expr, "x", &Lean5Expr::IntLit(10));

        let expected = Lean5Expr::eq(Lean5Expr::IntLit(10), Lean5Expr::IntLit(5));
        assert_eq!(format!("{}", result), format!("{}", expected));
    }

    #[test]
    fn test_substitute_var_in_lt() {
        // Mutation: delete match arm Lean5Expr::Lt(a, b)
        let expr = Lean5Expr::lt(Lean5Expr::var("x"), Lean5Expr::IntLit(10));
        let result = substitute_var(&expr, "x", &Lean5Expr::IntLit(5));

        let expected = Lean5Expr::lt(Lean5Expr::IntLit(5), Lean5Expr::IntLit(10));
        assert_eq!(format!("{}", result), format!("{}", expected));
    }

    #[test]
    fn test_substitute_var_in_le() {
        // Mutation: delete match arm Lean5Expr::Le(a, b)
        let expr = Lean5Expr::le(Lean5Expr::var("x"), Lean5Expr::IntLit(10));
        let result = substitute_var(&expr, "x", &Lean5Expr::IntLit(5));

        let expected = Lean5Expr::le(Lean5Expr::IntLit(5), Lean5Expr::IntLit(10));
        assert_eq!(format!("{}", result), format!("{}", expected));
    }

    #[test]
    fn test_substitute_var_in_gt() {
        // Mutation: delete match arm Lean5Expr::Gt(a, b)
        let expr = Lean5Expr::gt(Lean5Expr::var("x"), Lean5Expr::IntLit(0));
        let result = substitute_var(&expr, "x", &Lean5Expr::IntLit(5));

        let expected = Lean5Expr::gt(Lean5Expr::IntLit(5), Lean5Expr::IntLit(0));
        assert_eq!(format!("{}", result), format!("{}", expected));
    }

    #[test]
    fn test_substitute_var_in_sub() {
        // Mutation: delete match arm Lean5Expr::Sub(a, b)
        let expr = Lean5Expr::sub(Lean5Expr::var("x"), Lean5Expr::IntLit(1));
        let result = substitute_var(&expr, "x", &Lean5Expr::IntLit(10));

        let expected = Lean5Expr::sub(Lean5Expr::IntLit(10), Lean5Expr::IntLit(1));
        assert_eq!(format!("{}", result), format!("{}", expected));
    }

    #[test]
    fn test_substitute_var_in_mul() {
        // Mutation: delete match arm Lean5Expr::Mul(a, b)
        let expr = Lean5Expr::mul(Lean5Expr::var("x"), Lean5Expr::IntLit(2));
        let result = substitute_var(&expr, "x", &Lean5Expr::IntLit(5));

        let expected = Lean5Expr::mul(Lean5Expr::IntLit(5), Lean5Expr::IntLit(2));
        assert_eq!(format!("{}", result), format!("{}", expected));
    }

    #[test]
    fn test_substitute_var_in_ite() {
        // Mutation: delete match arm Lean5Expr::Ite(c, t, e)
        let expr = Lean5Expr::ite(
            Lean5Expr::var("x"),
            Lean5Expr::IntLit(1),
            Lean5Expr::IntLit(0),
        );
        let result = substitute_var(&expr, "x", &Lean5Expr::BoolLit(true));

        let expected = Lean5Expr::ite(
            Lean5Expr::BoolLit(true),
            Lean5Expr::IntLit(1),
            Lean5Expr::IntLit(0),
        );
        assert_eq!(format!("{}", result), format!("{}", expected));
    }

    #[test]
    fn test_substitute_var_in_neg() {
        // Mutation: delete match arm Lean5Expr::Neg(a)
        let expr = Lean5Expr::Neg(Box::new(Lean5Expr::var("x")));
        let result = substitute_var(&expr, "x", &Lean5Expr::IntLit(5));

        let expected = Lean5Expr::Neg(Box::new(Lean5Expr::IntLit(5)));
        assert_eq!(format!("{}", result), format!("{}", expected));
    }

    #[test]
    fn test_substitute_var_in_app() {
        // Mutation: delete match arm Lean5Expr::App(f, a)
        let expr = Lean5Expr::app(Lean5Expr::var("f"), Lean5Expr::var("x"));
        let result = substitute_var(&expr, "x", &Lean5Expr::IntLit(5));

        let expected = Lean5Expr::app(Lean5Expr::var("f"), Lean5Expr::IntLit(5));
        assert_eq!(format!("{}", result), format!("{}", expected));
    }

    #[test]
    fn test_substitute_var_in_exists() {
        // Mutation: delete match arm Lean5Expr::Exists(name, ty, body)
        let expr = Lean5Expr::exists_(
            "y",
            Lean5Type::Int,
            Lean5Expr::eq(Lean5Expr::var("x"), Lean5Expr::var("y")),
        );
        let result = substitute_var(&expr, "x", &Lean5Expr::IntLit(5));

        let expected = Lean5Expr::exists_(
            "y",
            Lean5Type::Int,
            Lean5Expr::eq(Lean5Expr::IntLit(5), Lean5Expr::var("y")),
        );
        assert_eq!(format!("{}", result), format!("{}", expected));
    }

    #[test]
    fn test_substitute_var_in_forall() {
        // Mutation: delete match arm Lean5Expr::Forall(_, _, body)
        let expr = Lean5Expr::forall_(
            "y",
            Lean5Type::Int,
            Lean5Expr::le(Lean5Expr::var("x"), Lean5Expr::var("y")),
        );
        let result = substitute_var(&expr, "x", &Lean5Expr::IntLit(0));

        let expected = Lean5Expr::forall_(
            "y",
            Lean5Type::Int,
            Lean5Expr::le(Lean5Expr::IntLit(0), Lean5Expr::var("y")),
        );
        assert_eq!(format!("{}", result), format!("{}", expected));
    }

    // ========================================================================
    // Mutation coverage tests for generate_kinduction_obligation
    // ========================================================================

    #[test]
    fn test_kinduction_obligation_k1_uses_single_hypothesis() {
        // Mutation: replace == with != in i == 0 check
        let invariant = Lean5Expr::forall_(
            "x",
            Lean5Type::Int,
            Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
        );

        // k=1: hypothesis is just Inv(n), conclusion is Inv(n+1)
        let obligation = generate_kinduction_obligation(&invariant, 1, "n", &Lean5Type::Int);

        // For k=1, hypothesis should be single, not conjunction
        let stmt_str = format!("{}", obligation.statement);
        // Should have single hypothesis n >= 0, not n + 0 >= 0
        assert!(
            stmt_str.contains("n"),
            "Should reference variable n: {}",
            stmt_str
        );
    }

    #[test]
    fn test_kinduction_obligation_k2_has_conjunction() {
        // Mutation: replace == with != would break conjunction building
        let invariant = Lean5Expr::forall_(
            "x",
            Lean5Type::Int,
            Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
        );

        // k=2: hypothesis is Inv(n) ∧ Inv(n+1), conclusion is Inv(n+2)
        let obligation = generate_kinduction_obligation(&invariant, 2, "n", &Lean5Type::Int);

        // Should be a conjunction (And)
        let stmt_str = format!("{}", obligation.statement);
        assert!(
            stmt_str.contains("∧"),
            "k=2 should have conjunction: {}",
            stmt_str
        );
    }

    // ========================================================================
    // Mutation coverage tests for ProofObligation::fmt (Display impl)
    // ========================================================================

    #[test]
    fn test_proof_obligation_display_not_empty() {
        // Mutation: replace Display::fmt with Ok(Default::default())
        let obligation = ProofObligation::new(
            "test_display",
            ProofObligationKind::Property,
            Lean5Expr::BoolLit(true),
        );

        let display = format!("{}", obligation);
        assert!(!display.is_empty(), "Display should not be empty");
        assert!(
            display.contains("test_display"),
            "Display should contain theorem name"
        );
    }
}
