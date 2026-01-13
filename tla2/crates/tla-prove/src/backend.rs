//! Proof backend interface and SMT implementation
//!
//! The proof manager uses backends to discharge proof obligations.
//! Currently supports SMT (Z3), with planned support for Zenon and Isabelle.

use std::collections::HashMap;
use std::time::Duration;
use tla_core::ast::Expr;
use tla_core::span::Spanned;
use tla_smt::{SmtCheckResult, SmtContext, Sort};

use crate::context::ProofContext;
use crate::error::ProofResult;
use crate::obligation::Obligation;

/// Result of attempting to prove an obligation
#[derive(Debug, Clone)]
pub enum ProofOutcome {
    /// The obligation was proved
    Proved,
    /// The proof failed with a counterexample
    Failed {
        /// Description of the failure
        message: String,
        /// Counterexample values (if available)
        counterexample: Option<Vec<(String, String)>>,
    },
    /// The backend couldn't determine the result
    Unknown { reason: String },
}

impl ProofOutcome {
    pub fn is_proved(&self) -> bool {
        matches!(self, ProofOutcome::Proved)
    }

    pub fn is_failed(&self) -> bool {
        matches!(self, ProofOutcome::Failed { .. })
    }

    pub fn is_unknown(&self) -> bool {
        matches!(self, ProofOutcome::Unknown { .. })
    }
}

/// A proof backend that can attempt to prove obligations
pub trait ProofBackend {
    /// Attempt to prove an obligation
    fn prove(&self, obligation: &Obligation, context: &ProofContext) -> ProofResult<ProofOutcome>;

    /// The name of this backend
    fn name(&self) -> &str;

    /// Whether this backend supports the given obligation
    fn supports(&self, obligation: &Obligation) -> bool;
}

/// SMT backend using Z3
pub struct SmtBackend {
    timeout: Duration,
}

impl SmtBackend {
    /// Create a new SMT backend with default timeout (30 seconds)
    pub fn new() -> Self {
        Self {
            timeout: Duration::from_secs(30),
        }
    }

    /// Create an SMT backend with custom timeout
    pub fn with_timeout(timeout: Duration) -> Self {
        Self { timeout }
    }

    /// Infer variable types from an expression
    fn infer_vars(&self, expr: &Spanned<Expr>) -> Vec<(String, Sort)> {
        let mut var_types: HashMap<String, Sort> = HashMap::new();
        self.infer_var_types(&expr.node, &mut var_types, None);

        var_types.into_iter().collect()
    }

    /// Infer types for variables based on context
    #[allow(clippy::only_used_in_recursion)]
    fn infer_var_types(
        &self,
        expr: &Expr,
        var_types: &mut HashMap<String, Sort>,
        expected_sort: Option<Sort>,
    ) {
        match expr {
            Expr::Ident(name) => {
                // Skip built-in operators and boolean literals
                if !["TRUE", "FALSE", "BOOLEAN"].contains(&name.as_str()) {
                    // Use expected sort if provided, otherwise default to Int
                    let sort = expected_sort.unwrap_or(Sort::Int);
                    // Only update if not already set (first inference wins)
                    var_types.entry(name.clone()).or_insert(sort);
                }
            }
            Expr::Bool(_) | Expr::Int(_) | Expr::String(_) => {}
            Expr::Apply(func, args) => {
                self.infer_var_types(&func.node, var_types, None);
                for arg in args {
                    self.infer_var_types(&arg.node, var_types, None);
                }
            }
            Expr::Lambda(_, body) => {
                self.infer_var_types(&body.node, var_types, None);
            }
            // Boolean operators - operands must be Bool
            Expr::And(l, r) | Expr::Or(l, r) | Expr::Implies(l, r) | Expr::Equiv(l, r) => {
                self.infer_var_types(&l.node, var_types, Some(Sort::Bool));
                self.infer_var_types(&r.node, var_types, Some(Sort::Bool));
            }
            Expr::Not(e) => {
                self.infer_var_types(&e.node, var_types, Some(Sort::Bool));
            }
            Expr::Forall(bounds, body) | Expr::Exists(bounds, body) => {
                // Body is boolean
                self.infer_var_types(&body.node, var_types, Some(Sort::Bool));
                // Bound variable domains
                for bound in bounds {
                    if let Some(dom) = &bound.domain {
                        self.infer_var_types(&dom.node, var_types, None);
                    }
                }
            }
            Expr::Choose(bound, body) => {
                self.infer_var_types(&body.node, var_types, Some(Sort::Bool));
                if let Some(dom) = &bound.domain {
                    self.infer_var_types(&dom.node, var_types, None);
                }
            }
            Expr::If(cond, then_e, else_e) => {
                self.infer_var_types(&cond.node, var_types, Some(Sort::Bool));
                self.infer_var_types(&then_e.node, var_types, expected_sort.clone());
                self.infer_var_types(&else_e.node, var_types, expected_sort);
            }
            // Comparisons produce Bool, operate on Int (for arith) or any
            Expr::Lt(l, r) | Expr::Leq(l, r) | Expr::Gt(l, r) | Expr::Geq(l, r) => {
                self.infer_var_types(&l.node, var_types, Some(Sort::Int));
                self.infer_var_types(&r.node, var_types, Some(Sort::Int));
            }
            Expr::Eq(l, r) | Expr::Neq(l, r) => {
                // Equality can be on any type, try to infer
                self.infer_var_types(&l.node, var_types, None);
                self.infer_var_types(&r.node, var_types, None);
            }
            // Arithmetic operators - operands are Int
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::IntDiv(l, r)
            | Expr::Mod(l, r)
            | Expr::Pow(l, r)
            | Expr::Range(l, r) => {
                self.infer_var_types(&l.node, var_types, Some(Sort::Int));
                self.infer_var_types(&r.node, var_types, Some(Sort::Int));
            }
            Expr::Neg(e) => {
                self.infer_var_types(&e.node, var_types, Some(Sort::Int));
            }
            Expr::In(l, r)
            | Expr::NotIn(l, r)
            | Expr::Subseteq(l, r)
            | Expr::Union(l, r)
            | Expr::Intersect(l, r)
            | Expr::SetMinus(l, r) => {
                self.infer_var_types(&l.node, var_types, None);
                self.infer_var_types(&r.node, var_types, None);
            }
            Expr::SetEnum(elems) => {
                for e in elems {
                    self.infer_var_types(&e.node, var_types, None);
                }
            }
            Expr::SetBuilder(body, bounds) | Expr::FuncDef(bounds, body) => {
                self.infer_var_types(&body.node, var_types, None);
                for bound in bounds {
                    if let Some(dom) = &bound.domain {
                        self.infer_var_types(&dom.node, var_types, None);
                    }
                }
            }
            Expr::SetFilter(bound, body) => {
                self.infer_var_types(&body.node, var_types, Some(Sort::Bool));
                if let Some(dom) = &bound.domain {
                    self.infer_var_types(&dom.node, var_types, None);
                }
            }
            Expr::Powerset(e) | Expr::BigUnion(e) | Expr::Domain(e) => {
                self.infer_var_types(&e.node, var_types, None);
            }
            Expr::FuncApply(f, arg) => {
                self.infer_var_types(&f.node, var_types, None);
                self.infer_var_types(&arg.node, var_types, None);
            }
            Expr::FuncSet(dom, codom) => {
                self.infer_var_types(&dom.node, var_types, None);
                self.infer_var_types(&codom.node, var_types, None);
            }
            Expr::Except(base, specs) => {
                self.infer_var_types(&base.node, var_types, None);
                for spec in specs {
                    self.infer_var_types(&spec.value.node, var_types, None);
                }
            }
            Expr::Record(fields) | Expr::RecordSet(fields) => {
                for (_, v) in fields {
                    self.infer_var_types(&v.node, var_types, None);
                }
            }
            Expr::RecordAccess(e, _) => {
                self.infer_var_types(&e.node, var_types, None);
            }
            Expr::Tuple(elems) | Expr::Times(elems) => {
                for e in elems {
                    self.infer_var_types(&e.node, var_types, None);
                }
            }
            Expr::Prime(e) => {
                self.infer_var_types(&e.node, var_types, None);
            }
            Expr::Always(e) | Expr::Eventually(e) => {
                self.infer_var_types(&e.node, var_types, Some(Sort::Bool));
            }
            Expr::LeadsTo(l, r) => {
                self.infer_var_types(&l.node, var_types, Some(Sort::Bool));
                self.infer_var_types(&r.node, var_types, Some(Sort::Bool));
            }
            Expr::WeakFair(v, a) | Expr::StrongFair(v, a) => {
                self.infer_var_types(&v.node, var_types, None);
                self.infer_var_types(&a.node, var_types, Some(Sort::Bool));
            }
            Expr::Enabled(e) | Expr::Unchanged(e) => {
                self.infer_var_types(&e.node, var_types, Some(Sort::Bool));
            }
            Expr::Case(arms, other) => {
                for arm in arms {
                    self.infer_var_types(&arm.guard.node, var_types, Some(Sort::Bool));
                    self.infer_var_types(&arm.body.node, var_types, expected_sort.clone());
                }
                if let Some(o) = other {
                    self.infer_var_types(&o.node, var_types, expected_sort);
                }
            }
            Expr::Let(defs, body) => {
                for def in defs {
                    self.infer_var_types(&def.body.node, var_types, None);
                }
                self.infer_var_types(&body.node, var_types, expected_sort);
            }
            // These are resolved during lowering/evaluation, not in proof context
            Expr::OpRef(_) | Expr::ModuleRef(_, _, _) | Expr::InstanceExpr(_, _) => {}
        }
    }
}

impl Default for SmtBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ProofBackend for SmtBackend {
    fn prove(&self, obligation: &Obligation, context: &ProofContext) -> ProofResult<ProofOutcome> {
        let smt = SmtContext::new().with_timeout(self.timeout);

        // Collect all variables from goal and assumptions
        let mut all_vars = self.infer_vars(&obligation.goal);
        for assumption in &obligation.assumptions {
            let assumption_vars = self.infer_vars(assumption);
            for (name, sort) in assumption_vars {
                if !all_vars.iter().any(|(n, _)| n == &name) {
                    all_vars.push((name, sort));
                }
            }
        }

        // Add variables from context facts
        for fact in context.all_facts() {
            let fact_vars = self.infer_vars(&fact.expr);
            for (name, sort) in fact_vars {
                if !all_vars.iter().any(|(n, _)| n == &name) {
                    all_vars.push((name, sort));
                }
            }
        }

        // Combine assumptions: obligation assumptions + context facts
        let mut assumptions = obligation.assumptions.clone();
        for fact in context.all_facts() {
            assumptions.push(fact.expr.clone());
        }

        // Try to prove: assumptions => goal
        let result = smt.prove_implication(&assumptions, &obligation.goal, &all_vars)?;

        match result {
            SmtCheckResult::Unsat => {
                // Negation is unsat, so the implication holds
                Ok(ProofOutcome::Proved)
            }
            SmtCheckResult::Sat(model) => {
                // Found counterexample
                let counterexample: Vec<(String, String)> = model
                    .assignments
                    .iter()
                    .map(|(k, v)| (k.clone(), v.to_string()))
                    .collect();

                Ok(ProofOutcome::Failed {
                    message: "counterexample found".to_string(),
                    counterexample: Some(counterexample),
                })
            }
            SmtCheckResult::Unknown(reason) => Ok(ProofOutcome::Unknown { reason }),
        }
    }

    fn name(&self) -> &str {
        "SMT (Z3)"
    }

    fn supports(&self, _obligation: &Obligation) -> bool {
        // SMT can attempt any obligation, but may return Unknown
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::obligation::ObligationId;
    use num_bigint::BigInt;
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
        let backend = SmtBackend::new();
        let context = ProofContext::new();

        // Prove: x > 5 => x > 3
        let goal = spanned(Expr::Implies(
            Box::new(spanned(Expr::Gt(
                Box::new(spanned(Expr::Ident("x".to_string()))),
                Box::new(spanned(Expr::Int(BigInt::from(5)))),
            ))),
            Box::new(spanned(Expr::Gt(
                Box::new(spanned(Expr::Ident("x".to_string()))),
                Box::new(spanned(Expr::Int(BigInt::from(3)))),
            ))),
        ));

        let obl = make_obligation(goal, vec![]);
        let result = backend.prove(&obl, &context).unwrap();

        assert!(result.is_proved());
    }

    #[test]
    fn test_prove_with_assumption() {
        let backend = SmtBackend::new();
        let context = ProofContext::new();

        // Given: x > 5
        // Prove: x > 3
        let assumption = spanned(Expr::Gt(
            Box::new(spanned(Expr::Ident("x".to_string()))),
            Box::new(spanned(Expr::Int(BigInt::from(5)))),
        ));

        let goal = spanned(Expr::Gt(
            Box::new(spanned(Expr::Ident("x".to_string()))),
            Box::new(spanned(Expr::Int(BigInt::from(3)))),
        ));

        let obl = make_obligation(goal, vec![assumption]);
        let result = backend.prove(&obl, &context).unwrap();

        assert!(result.is_proved());
    }

    #[test]
    fn test_prove_false_claim() {
        let backend = SmtBackend::new();
        let context = ProofContext::new();

        // Try to prove: x > 3 => x > 5 (this is false!)
        let goal = spanned(Expr::Implies(
            Box::new(spanned(Expr::Gt(
                Box::new(spanned(Expr::Ident("x".to_string()))),
                Box::new(spanned(Expr::Int(BigInt::from(3)))),
            ))),
            Box::new(spanned(Expr::Gt(
                Box::new(spanned(Expr::Ident("x".to_string()))),
                Box::new(spanned(Expr::Int(BigInt::from(5)))),
            ))),
        ));

        let obl = make_obligation(goal, vec![]);
        let result = backend.prove(&obl, &context).unwrap();

        assert!(result.is_failed());
    }

    #[test]
    fn test_prove_with_context_fact() {
        let backend = SmtBackend::new();
        let mut context = ProofContext::new();

        // Add fact to context: x > 5
        let fact_expr = spanned(Expr::Gt(
            Box::new(spanned(Expr::Ident("x".to_string()))),
            Box::new(spanned(Expr::Int(BigInt::from(5)))),
        ));
        context.add_proved_fact("P".to_string(), fact_expr);

        // Prove: x > 3 (should follow from context)
        let goal = spanned(Expr::Gt(
            Box::new(spanned(Expr::Ident("x".to_string()))),
            Box::new(spanned(Expr::Int(BigInt::from(3)))),
        ));

        let obl = make_obligation(goal, vec![]);
        let result = backend.prove(&obl, &context).unwrap();

        assert!(result.is_proved());
    }

    #[test]
    fn test_boolean_proof() {
        let backend = SmtBackend::new();
        let context = ProofContext::new();

        // Prove: (a /\ b) => a
        let goal = spanned(Expr::Implies(
            Box::new(spanned(Expr::And(
                Box::new(spanned(Expr::Ident("a".to_string()))),
                Box::new(spanned(Expr::Ident("b".to_string()))),
            ))),
            Box::new(spanned(Expr::Ident("a".to_string()))),
        ));

        let obl = make_obligation(goal, vec![]);
        let result = backend.prove(&obl, &context).unwrap();

        assert!(result.is_proved());
    }
}
