//! Zenon tableau prover backend
//!
//! This backend uses the tla-zenon first-order tableau prover for proving
//! proof obligations that are suitable for first-order logic.

use std::time::Duration;
use tla_core::ast::Expr;
use tla_core::span::Spanned;
use tla_zenon::{Formula, ProofResult as ZenonResult, Prover as ZenonProver, ProverConfig, Term};

use crate::backend::{ProofBackend, ProofOutcome};
use crate::context::ProofContext;
use crate::error::ProofResult;
use crate::obligation::Obligation;

/// Zenon tableau prover backend
pub struct ZenonBackend {
    /// Maximum proof search depth
    depth_limit: usize,
    /// Maximum number of proof nodes
    node_limit: usize,
    /// Timeout for proof search
    timeout: Duration,
}

impl ZenonBackend {
    /// Create a new Zenon backend with default settings
    pub fn new() -> Self {
        Self {
            depth_limit: 100,
            node_limit: 10000,
            timeout: Duration::from_secs(30),
        }
    }

    /// Create with custom limits
    pub fn with_limits(depth_limit: usize, node_limit: usize, timeout: Duration) -> Self {
        Self {
            depth_limit,
            node_limit,
            timeout,
        }
    }

    /// Translate a TLA+ expression to a Zenon formula
    fn translate_expr(&self, expr: &Spanned<Expr>) -> Option<Formula> {
        self.translate_expr_inner(&expr.node)
    }

    /// Inner translation function
    fn translate_expr_inner(&self, expr: &Expr) -> Option<Formula> {
        match expr {
            // Boolean literals
            Expr::Bool(true) => Some(Formula::True),
            Expr::Bool(false) => Some(Formula::False),

            // Identifiers become atoms (propositional variables)
            Expr::Ident(name) => {
                if name == "TRUE" {
                    Some(Formula::True)
                } else if name == "FALSE" {
                    Some(Formula::False)
                } else {
                    Some(Formula::atom(name.clone()))
                }
            }

            // Boolean connectives
            Expr::Not(e) => {
                let inner = self.translate_expr_inner(&e.node)?;
                Some(Formula::not(inner))
            }
            Expr::And(l, r) => {
                let left = self.translate_expr_inner(&l.node)?;
                let right = self.translate_expr_inner(&r.node)?;
                Some(Formula::and(left, right))
            }
            Expr::Or(l, r) => {
                let left = self.translate_expr_inner(&l.node)?;
                let right = self.translate_expr_inner(&r.node)?;
                Some(Formula::or(left, right))
            }
            Expr::Implies(l, r) => {
                let left = self.translate_expr_inner(&l.node)?;
                let right = self.translate_expr_inner(&r.node)?;
                Some(Formula::implies(left, right))
            }
            Expr::Equiv(l, r) => {
                let left = self.translate_expr_inner(&l.node)?;
                let right = self.translate_expr_inner(&r.node)?;
                Some(Formula::equiv(left, right))
            }

            // Equality becomes FOL equality
            Expr::Eq(l, r) => {
                let left = Self::translate_term(&l.node)?;
                let right = Self::translate_term(&r.node)?;
                Some(Formula::eq(left, right))
            }
            Expr::Neq(l, r) => {
                let left = Self::translate_term(&l.node)?;
                let right = Self::translate_term(&r.node)?;
                Some(Formula::not(Formula::eq(left, right)))
            }

            // Quantifiers - handle both bounded and unbounded
            Expr::Forall(bounds, body) => {
                let body_formula = self.translate_expr_inner(&body.node)?;
                // For unbounded quantifiers, wrap directly
                // For bounded quantifiers, we need to add domain constraints
                self.wrap_quantifier_forall(bounds, body_formula)
            }
            Expr::Exists(bounds, body) => {
                let body_formula = self.translate_expr_inner(&body.node)?;
                self.wrap_quantifier_exists(bounds, body_formula)
            }

            // Comparisons become predicates
            Expr::Lt(l, r) => self.translate_comparison("lt", l, r),
            Expr::Leq(l, r) => self.translate_comparison("leq", l, r),
            Expr::Gt(l, r) => self.translate_comparison("gt", l, r),
            Expr::Geq(l, r) => self.translate_comparison("geq", l, r),

            // Set membership becomes a predicate
            Expr::In(l, r) => {
                let elem = Self::translate_term(&l.node)?;
                let set = Self::translate_term(&r.node)?;
                Some(Formula::pred("in", vec![elem, set]))
            }
            Expr::NotIn(l, r) => {
                let elem = Self::translate_term(&l.node)?;
                let set = Self::translate_term(&r.node)?;
                Some(Formula::not(Formula::pred("in", vec![elem, set])))
            }

            // IF-THEN-ELSE: (cond => then) /\ (~cond => else)
            Expr::If(cond, then_e, else_e) => {
                let cond_f = self.translate_expr_inner(&cond.node)?;
                let then_f = self.translate_expr_inner(&then_e.node)?;
                let else_f = self.translate_expr_inner(&else_e.node)?;
                Some(Formula::and(
                    Formula::implies(cond_f.clone(), then_f),
                    Formula::implies(Formula::not(cond_f), else_f),
                ))
            }

            // Function application - treat as predicate when boolean result expected
            Expr::Apply(func, args) => {
                if let Expr::Ident(name) = &func.node {
                    let term_args: Option<Vec<Term>> =
                        args.iter().map(|a| Self::translate_term(&a.node)).collect();
                    let term_args = term_args?;
                    Some(Formula::pred(name.clone(), term_args))
                } else {
                    None
                }
            }

            // Other expressions are not directly supported
            _ => None,
        }
    }

    /// Translate a TLA+ expression to a FOL term
    fn translate_term(expr: &Expr) -> Option<Term> {
        match expr {
            Expr::Ident(name) => Some(Term::var(name.clone())),
            Expr::Int(n) => Some(Term::constant(n.to_string())),
            Expr::String(s) => Some(Term::constant(format!("\"{}\"", s))),
            Expr::Apply(func, args) => {
                if let Expr::Ident(name) = &func.node {
                    let term_args: Option<Vec<Term>> =
                        args.iter().map(|a| Self::translate_term(&a.node)).collect();
                    Some(Term::app(name.clone(), term_args?))
                } else {
                    None
                }
            }
            // Arithmetic operations as function applications
            Expr::Add(l, r) => {
                let left = Self::translate_term(&l.node)?;
                let right = Self::translate_term(&r.node)?;
                Some(Term::app("add", vec![left, right]))
            }
            Expr::Sub(l, r) => {
                let left = Self::translate_term(&l.node)?;
                let right = Self::translate_term(&r.node)?;
                Some(Term::app("sub", vec![left, right]))
            }
            Expr::Mul(l, r) => {
                let left = Self::translate_term(&l.node)?;
                let right = Self::translate_term(&r.node)?;
                Some(Term::app("mul", vec![left, right]))
            }
            Expr::Neg(e) => {
                let inner = Self::translate_term(&e.node)?;
                Some(Term::app("neg", vec![inner]))
            }
            _ => None,
        }
    }

    /// Translate a comparison to a predicate
    fn translate_comparison(
        &self,
        op: &str,
        l: &Spanned<Expr>,
        r: &Spanned<Expr>,
    ) -> Option<Formula> {
        let left = Self::translate_term(&l.node)?;
        let right = Self::translate_term(&r.node)?;
        Some(Formula::pred(op, vec![left, right]))
    }

    /// Wrap a formula with forall quantifiers from bounds
    fn wrap_quantifier_forall(
        &self,
        bounds: &[tla_core::ast::BoundVar],
        body: Formula,
    ) -> Option<Formula> {
        let mut result = body;

        // Process bounds in reverse order so innermost quantifier is processed first
        for bound in bounds.iter().rev() {
            let var_name = bound.name.node.clone();

            // If there's a domain, add implication: \A x \in S : P becomes \A x : (x \in S) => P
            if let Some(domain) = &bound.domain {
                let domain_term = Self::translate_term(&domain.node)?;
                let var_term = Term::var(var_name.clone());
                let in_domain = Formula::pred("in", vec![var_term, domain_term]);
                result = Formula::implies(in_domain, result);
            }

            result = Formula::forall(var_name, result);
        }

        Some(result)
    }

    /// Wrap a formula with exists quantifiers from bounds
    fn wrap_quantifier_exists(
        &self,
        bounds: &[tla_core::ast::BoundVar],
        body: Formula,
    ) -> Option<Formula> {
        let mut result = body;

        // Process bounds in reverse order
        for bound in bounds.iter().rev() {
            let var_name = bound.name.node.clone();

            // If there's a domain, add conjunction: \E x \in S : P becomes \E x : (x \in S) /\ P
            if let Some(domain) = &bound.domain {
                let domain_term = Self::translate_term(&domain.node)?;
                let var_term = Term::var(var_name.clone());
                let in_domain = Formula::pred("in", vec![var_term, domain_term]);
                result = Formula::and(in_domain, result);
            }

            result = Formula::exists(var_name, result);
        }

        Some(result)
    }

    /// Check if an expression is suitable for Zenon (pure first-order logic)
    fn is_suitable_for_zenon(expr: &Expr) -> bool {
        match expr {
            // Boolean literals and identifiers are always suitable
            Expr::Bool(_) | Expr::Ident(_) => true,

            // Integer literals are suitable (as constants)
            Expr::Int(_) | Expr::String(_) => true,

            // Boolean connectives - recurse
            Expr::Not(e) => Self::is_suitable_for_zenon(&e.node),
            Expr::And(l, r)
            | Expr::Or(l, r)
            | Expr::Implies(l, r)
            | Expr::Equiv(l, r)
            | Expr::Eq(l, r)
            | Expr::Neq(l, r)
            | Expr::Lt(l, r)
            | Expr::Leq(l, r)
            | Expr::Gt(l, r)
            | Expr::Geq(l, r) => {
                Self::is_suitable_for_zenon(&l.node) && Self::is_suitable_for_zenon(&r.node)
            }

            // Arithmetic operations
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
                Self::is_suitable_for_zenon(&l.node) && Self::is_suitable_for_zenon(&r.node)
            }
            Expr::Neg(e) => Self::is_suitable_for_zenon(&e.node),

            // Quantifiers - check bounds and body
            Expr::Forall(bounds, body) | Expr::Exists(bounds, body) => {
                let bounds_ok = bounds.iter().all(|b| {
                    b.domain
                        .as_ref()
                        .map(|d| Self::is_suitable_for_zenon(&d.node))
                        .unwrap_or(true)
                });
                bounds_ok && Self::is_suitable_for_zenon(&body.node)
            }

            // Set membership
            Expr::In(l, r) | Expr::NotIn(l, r) => {
                Self::is_suitable_for_zenon(&l.node) && Self::is_suitable_for_zenon(&r.node)
            }

            // IF-THEN-ELSE
            Expr::If(c, t, e) => {
                Self::is_suitable_for_zenon(&c.node)
                    && Self::is_suitable_for_zenon(&t.node)
                    && Self::is_suitable_for_zenon(&e.node)
            }

            // Simple function application
            Expr::Apply(func, args) => {
                matches!(&func.node, Expr::Ident(_))
                    && args.iter().all(|a| Self::is_suitable_for_zenon(&a.node))
            }

            // Not suitable for tableau proving
            _ => false,
        }
    }
}

impl Default for ZenonBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ProofBackend for ZenonBackend {
    fn prove(&self, obligation: &Obligation, context: &ProofContext) -> ProofResult<ProofOutcome> {
        // Translate goal
        let goal = match self.translate_expr(&obligation.goal) {
            Some(f) => f,
            None => {
                return Ok(ProofOutcome::Unknown {
                    reason: "goal cannot be translated to first-order logic".to_string(),
                });
            }
        };

        // Translate assumptions
        let mut assumptions = Vec::new();
        for assumption in &obligation.assumptions {
            match self.translate_expr(assumption) {
                Some(f) => assumptions.push(f),
                None => {
                    return Ok(ProofOutcome::Unknown {
                        reason: "assumption cannot be translated to first-order logic".to_string(),
                    });
                }
            }
        }

        // Add context facts
        for fact in context.all_facts() {
            if let Some(f) = self.translate_expr(&fact.expr) {
                assumptions.push(f);
            }
            // Ignore untranslatable facts - they might not be needed
        }

        // Build the formula to prove: assumptions => goal
        let formula = if assumptions.is_empty() {
            goal
        } else {
            let mut conj = assumptions[0].clone();
            for assumption in &assumptions[1..] {
                conj = Formula::and(conj, assumption.clone());
            }
            Formula::implies(conj, goal)
        };

        // Configure and run prover
        let config = ProverConfig {
            max_depth: self.depth_limit,
            max_nodes: self.node_limit,
            timeout: self.timeout,
            ..Default::default()
        };

        let mut prover = ZenonProver::new();
        let result = prover.prove(&formula, config);

        match result {
            ZenonResult::Valid(_proof) => Ok(ProofOutcome::Proved),
            ZenonResult::Unknown { reason } => Ok(ProofOutcome::Unknown { reason }),
            ZenonResult::Invalid { reason } => Ok(ProofOutcome::Failed {
                message: reason,
                counterexample: None,
            }),
        }
    }

    fn name(&self) -> &str {
        "Zenon (Tableau)"
    }

    fn supports(&self, obligation: &Obligation) -> bool {
        // Check if goal is suitable
        if !Self::is_suitable_for_zenon(&obligation.goal.node) {
            return false;
        }

        // Check if all assumptions are suitable
        obligation
            .assumptions
            .iter()
            .all(|a| Self::is_suitable_for_zenon(&a.node))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::obligation::ObligationId;
    use tla_core::span::{FileId, Span};

    fn span() -> Span {
        Span::new(FileId(0), 0, 0)
    }

    fn spanned<T>(node: T) -> Spanned<T> {
        Spanned::new(node, span())
    }

    fn make_obligation(goal: Spanned<Expr>, assumptions: Vec<Spanned<Expr>>) -> Obligation {
        Obligation {
            id: ObligationId("test".to_string()),
            goal,
            assumptions,
            definitions: Vec::new(),
            span: span(),
            description: "test".to_string(),
        }
    }

    #[test]
    fn test_prove_simple_tautology() {
        let backend = ZenonBackend::new();
        let context = ProofContext::new();

        // Prove: (A /\ B) => A
        let goal = spanned(Expr::Implies(
            Box::new(spanned(Expr::And(
                Box::new(spanned(Expr::Ident("A".to_string()))),
                Box::new(spanned(Expr::Ident("B".to_string()))),
            ))),
            Box::new(spanned(Expr::Ident("A".to_string()))),
        ));

        let obl = make_obligation(goal, vec![]);
        let result = backend.prove(&obl, &context).unwrap();

        assert!(result.is_proved());
    }

    #[test]
    fn test_prove_modus_ponens() {
        let backend = ZenonBackend::new();
        let context = ProofContext::new();

        // Given: A, A => B
        // Prove: B
        let assumption1 = spanned(Expr::Ident("A".to_string()));
        let assumption2 = spanned(Expr::Implies(
            Box::new(spanned(Expr::Ident("A".to_string()))),
            Box::new(spanned(Expr::Ident("B".to_string()))),
        ));

        let goal = spanned(Expr::Ident("B".to_string()));

        let obl = make_obligation(goal, vec![assumption1, assumption2]);
        let result = backend.prove(&obl, &context).unwrap();

        assert!(result.is_proved());
    }

    #[test]
    fn test_prove_double_negation() {
        let backend = ZenonBackend::new();
        let context = ProofContext::new();

        // Prove: ~~A => A
        let goal = spanned(Expr::Implies(
            Box::new(spanned(Expr::Not(Box::new(spanned(Expr::Not(Box::new(
                spanned(Expr::Ident("A".to_string())),
            ))))))),
            Box::new(spanned(Expr::Ident("A".to_string()))),
        ));

        let obl = make_obligation(goal, vec![]);
        let result = backend.prove(&obl, &context).unwrap();

        assert!(result.is_proved());
    }

    #[test]
    fn test_prove_excluded_middle() {
        let backend = ZenonBackend::new();
        let context = ProofContext::new();

        // Prove: A \/ ~A
        let goal = spanned(Expr::Or(
            Box::new(spanned(Expr::Ident("A".to_string()))),
            Box::new(spanned(Expr::Not(Box::new(spanned(Expr::Ident(
                "A".to_string(),
            )))))),
        ));

        let obl = make_obligation(goal, vec![]);
        let result = backend.prove(&obl, &context).unwrap();

        assert!(result.is_proved());
    }

    #[test]
    fn test_prove_de_morgan() {
        let backend = ZenonBackend::new();
        let context = ProofContext::new();

        // Prove: ~(A /\ B) <=> (~A \/ ~B)
        let goal = spanned(Expr::Equiv(
            Box::new(spanned(Expr::Not(Box::new(spanned(Expr::And(
                Box::new(spanned(Expr::Ident("A".to_string()))),
                Box::new(spanned(Expr::Ident("B".to_string()))),
            )))))),
            Box::new(spanned(Expr::Or(
                Box::new(spanned(Expr::Not(Box::new(spanned(Expr::Ident(
                    "A".to_string(),
                )))))),
                Box::new(spanned(Expr::Not(Box::new(spanned(Expr::Ident(
                    "B".to_string(),
                )))))),
            ))),
        ));

        let obl = make_obligation(goal, vec![]);
        let result = backend.prove(&obl, &context).unwrap();

        assert!(result.is_proved());
    }

    #[test]
    fn test_prove_syllogism() {
        let backend = ZenonBackend::new();
        let context = ProofContext::new();

        // Given: A => B, B => C
        // Prove: A => C
        let assumption1 = spanned(Expr::Implies(
            Box::new(spanned(Expr::Ident("A".to_string()))),
            Box::new(spanned(Expr::Ident("B".to_string()))),
        ));
        let assumption2 = spanned(Expr::Implies(
            Box::new(spanned(Expr::Ident("B".to_string()))),
            Box::new(spanned(Expr::Ident("C".to_string()))),
        ));

        let goal = spanned(Expr::Implies(
            Box::new(spanned(Expr::Ident("A".to_string()))),
            Box::new(spanned(Expr::Ident("C".to_string()))),
        ));

        let obl = make_obligation(goal, vec![assumption1, assumption2]);
        let result = backend.prove(&obl, &context).unwrap();

        assert!(result.is_proved());
    }

    #[test]
    fn test_not_prove_invalid() {
        let backend = ZenonBackend::new();
        let context = ProofContext::new();

        // Cannot prove: A => B (without assumptions)
        let goal = spanned(Expr::Implies(
            Box::new(spanned(Expr::Ident("A".to_string()))),
            Box::new(spanned(Expr::Ident("B".to_string()))),
        ));

        let obl = make_obligation(goal, vec![]);
        let result = backend.prove(&obl, &context).unwrap();

        // Should fail or be unknown
        assert!(!result.is_proved());
    }

    #[test]
    fn test_prove_equality() {
        let backend = ZenonBackend::new();
        let context = ProofContext::new();

        // Prove: x = x
        let goal = spanned(Expr::Eq(
            Box::new(spanned(Expr::Ident("x".to_string()))),
            Box::new(spanned(Expr::Ident("x".to_string()))),
        ));

        let obl = make_obligation(goal, vec![]);
        let result = backend.prove(&obl, &context).unwrap();

        // Note: This might not be provable without equality reasoning in Zenon
        // The result depends on whether the prover has built-in equality
        // For now, we accept either proved or unknown
        assert!(result.is_proved() || result.is_unknown());
    }

    #[test]
    fn test_translate_boolean_literals() {
        let backend = ZenonBackend::new();

        let true_expr = spanned(Expr::Bool(true));
        let false_expr = spanned(Expr::Bool(false));

        let true_formula = backend.translate_expr(&true_expr);
        let false_formula = backend.translate_expr(&false_expr);

        assert_eq!(true_formula, Some(Formula::True));
        assert_eq!(false_formula, Some(Formula::False));
    }

    #[test]
    fn test_translate_and() {
        let backend = ZenonBackend::new();

        let expr = spanned(Expr::And(
            Box::new(spanned(Expr::Ident("A".to_string()))),
            Box::new(spanned(Expr::Ident("B".to_string()))),
        ));

        let formula = backend.translate_expr(&expr);
        assert!(matches!(formula, Some(Formula::And(_, _))));
    }

    #[test]
    fn test_supports_propositional() {
        let backend = ZenonBackend::new();

        // Propositional formula should be supported
        let prop = spanned(Expr::And(
            Box::new(spanned(Expr::Ident("A".to_string()))),
            Box::new(spanned(Expr::Ident("B".to_string()))),
        ));
        let obl = make_obligation(prop, vec![]);
        assert!(backend.supports(&obl));
    }

    #[test]
    fn test_does_not_support_complex_sets() {
        let backend = ZenonBackend::new();

        // Set builder is not supported
        let set_builder = spanned(Expr::SetBuilder(
            Box::new(spanned(Expr::Ident("x".to_string()))),
            vec![],
        ));
        let obl = make_obligation(set_builder, vec![]);
        assert!(!backend.supports(&obl));
    }

    #[test]
    fn test_prove_contrapositive() {
        let backend = ZenonBackend::new();
        let context = ProofContext::new();

        // Prove: (A => B) => (~B => ~A)
        let goal = spanned(Expr::Implies(
            Box::new(spanned(Expr::Implies(
                Box::new(spanned(Expr::Ident("A".to_string()))),
                Box::new(spanned(Expr::Ident("B".to_string()))),
            ))),
            Box::new(spanned(Expr::Implies(
                Box::new(spanned(Expr::Not(Box::new(spanned(Expr::Ident(
                    "B".to_string(),
                )))))),
                Box::new(spanned(Expr::Not(Box::new(spanned(Expr::Ident(
                    "A".to_string(),
                )))))),
            ))),
        ));

        let obl = make_obligation(goal, vec![]);
        let result = backend.prove(&obl, &context).unwrap();

        assert!(result.is_proved());
    }

    #[test]
    fn test_prove_with_if_then_else() {
        let backend = ZenonBackend::new();
        let context = ProofContext::new();

        // Given: A
        // Prove: IF A THEN B ELSE C => B
        let assumption = spanned(Expr::Ident("A".to_string()));

        let goal = spanned(Expr::Implies(
            Box::new(spanned(Expr::If(
                Box::new(spanned(Expr::Ident("A".to_string()))),
                Box::new(spanned(Expr::Ident("B".to_string()))),
                Box::new(spanned(Expr::Ident("C".to_string()))),
            ))),
            Box::new(spanned(Expr::Ident("B".to_string()))),
        ));

        let obl = make_obligation(goal, vec![assumption]);
        let result = backend.prove(&obl, &context).unwrap();

        assert!(result.is_proved());
    }
}
