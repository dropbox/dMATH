//! Translation from TLA+ expressions to Z3
//!
//! This module provides the translation layer between TLA+ AST and Z3's API.

use std::collections::HashMap;
use tla_core::ast::{BoundVar, Expr};
use tla_core::span::Spanned;
use z3::ast::{Ast, Bool, Dynamic, Int};
use z3::{Context, Sort as Z3Sort};

use crate::error::{SmtError, SmtResult};

/// Sort (type) in the SMT context
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sort {
    /// Boolean sort
    Bool,
    /// Integer sort
    Int,
    /// Uninterpreted sort (for TLA+ values we can't type precisely)
    Uninterpreted(String),
}

impl std::fmt::Display for Sort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Sort::Bool => write!(f, "Bool"),
            Sort::Int => write!(f, "Int"),
            Sort::Uninterpreted(name) => write!(f, "{}", name),
        }
    }
}

/// Translator from TLA+ expressions to Z3 AST
pub struct SmtTranslator<'ctx> {
    ctx: &'ctx Context,
    /// Variable declarations: name -> (Z3 sort, declared Z3 constant)
    vars: HashMap<String, (Sort, Dynamic<'ctx>)>,
    /// Uninterpreted sorts created
    sorts: HashMap<String, Z3Sort<'ctx>>,
}

impl<'ctx> SmtTranslator<'ctx> {
    /// Create a new translator with the given Z3 context
    pub fn new(ctx: &'ctx Context) -> Self {
        Self {
            ctx,
            vars: HashMap::new(),
            sorts: HashMap::new(),
        }
    }

    /// Declare a variable with the given sort
    pub fn declare_var(&mut self, name: &str, sort: Sort) -> SmtResult<()> {
        if self.vars.contains_key(name) {
            return Err(SmtError::TypeMismatch {
                name: name.to_string(),
                expected: format!("{}", self.vars[name].0),
                actual: format!("{}", sort),
            });
        }

        let z3_const = match &sort {
            Sort::Bool => {
                let b = Bool::new_const(self.ctx, name);
                Dynamic::from_ast(&b)
            }
            Sort::Int => {
                let i = Int::new_const(self.ctx, name);
                Dynamic::from_ast(&i)
            }
            Sort::Uninterpreted(sort_name) => {
                let _z3_sort = self.get_or_create_sort(sort_name);
                // Create an uninterpreted constant using Int as placeholder
                // TODO: Use proper uninterpreted sort constant when API allows
                let i = Int::new_const(self.ctx, name);
                Dynamic::from_ast(&i)
            }
        };

        self.vars.insert(name.to_string(), (sort, z3_const));
        Ok(())
    }

    /// Get or create an uninterpreted sort
    fn get_or_create_sort(&mut self, name: &str) -> Z3Sort<'ctx> {
        if let Some(sort) = self.sorts.get(name) {
            return sort.clone();
        }
        let sort = Z3Sort::uninterpreted(self.ctx, z3::Symbol::String(name.to_string()));
        self.sorts.insert(name.to_string(), sort.clone());
        sort
    }

    /// Translate a TLA+ expression to a Z3 boolean expression
    pub fn translate_bool(&self, expr: &Spanned<Expr>) -> SmtResult<Bool<'ctx>> {
        match &expr.node {
            Expr::Bool(b) => Ok(Bool::from_bool(self.ctx, *b)),

            Expr::Ident(name) => {
                let (sort, z3_const) = self
                    .vars
                    .get(name)
                    .ok_or_else(|| SmtError::UnknownVariable(name.clone()))?;
                if *sort != Sort::Bool {
                    return Err(SmtError::TypeMismatch {
                        name: name.clone(),
                        expected: "Bool".to_string(),
                        actual: format!("{}", sort),
                    });
                }
                z3_const.as_bool().ok_or_else(|| SmtError::TypeMismatch {
                    name: name.clone(),
                    expected: "Bool".to_string(),
                    actual: "non-Bool".to_string(),
                })
            }

            Expr::And(left, right) => {
                let l = self.translate_bool(left)?;
                let r = self.translate_bool(right)?;
                Ok(Bool::and(self.ctx, &[&l, &r]))
            }

            Expr::Or(left, right) => {
                let l = self.translate_bool(left)?;
                let r = self.translate_bool(right)?;
                Ok(Bool::or(self.ctx, &[&l, &r]))
            }

            Expr::Not(inner) => {
                let i = self.translate_bool(inner)?;
                Ok(i.not())
            }

            Expr::Implies(left, right) => {
                let l = self.translate_bool(left)?;
                let r = self.translate_bool(right)?;
                Ok(l.implies(&r))
            }

            Expr::Equiv(left, right) => {
                let l = self.translate_bool(left)?;
                let r = self.translate_bool(right)?;
                Ok(l.iff(&r))
            }

            Expr::Eq(left, right) => {
                // Try to translate as integers first, fall back to bool
                if let (Ok(l), Ok(r)) = (self.translate_int(left), self.translate_int(right)) {
                    Ok(l._eq(&r))
                } else if let (Ok(l), Ok(r)) =
                    (self.translate_bool(left), self.translate_bool(right))
                {
                    Ok(l.iff(&r))
                } else {
                    Err(SmtError::UntranslatableExpr(
                        "cannot determine type for equality comparison".to_string(),
                    ))
                }
            }

            Expr::Neq(left, right) => {
                // x /= y is equivalent to ~(x = y)
                let eq = self.translate_bool(&Spanned::new(
                    Expr::Eq(left.clone(), right.clone()),
                    expr.span,
                ))?;
                Ok(eq.not())
            }

            Expr::Lt(left, right) => {
                let l = self.translate_int(left)?;
                let r = self.translate_int(right)?;
                Ok(l.lt(&r))
            }

            Expr::Leq(left, right) => {
                let l = self.translate_int(left)?;
                let r = self.translate_int(right)?;
                Ok(l.le(&r))
            }

            Expr::Gt(left, right) => {
                let l = self.translate_int(left)?;
                let r = self.translate_int(right)?;
                Ok(l.gt(&r))
            }

            Expr::Geq(left, right) => {
                let l = self.translate_int(left)?;
                let r = self.translate_int(right)?;
                Ok(l.ge(&r))
            }

            Expr::If(cond, then_branch, else_branch) => {
                let c = self.translate_bool(cond)?;
                let t = self.translate_bool(then_branch)?;
                let e = self.translate_bool(else_branch)?;
                Ok(c.ite(&t, &e))
            }

            Expr::Forall(bounds, body) => self.translate_quantifier(bounds, body, true),

            Expr::Exists(bounds, body) => self.translate_quantifier(bounds, body, false),

            _ => Err(SmtError::UntranslatableExpr(format!(
                "expression type {:?} cannot be translated to Bool",
                std::mem::discriminant(&expr.node)
            ))),
        }
    }

    /// Translate a TLA+ expression to a Z3 integer expression
    pub fn translate_int(&self, expr: &Spanned<Expr>) -> SmtResult<Int<'ctx>> {
        match &expr.node {
            Expr::Int(n) => {
                // Convert BigInt to i64 for Z3 (with overflow check)
                let n_i64: i64 = n.try_into().map_err(|_| {
                    SmtError::UntranslatableExpr(format!("integer {} too large for SMT", n))
                })?;
                Ok(Int::from_i64(self.ctx, n_i64))
            }

            Expr::Ident(name) => {
                let (sort, z3_const) = self
                    .vars
                    .get(name)
                    .ok_or_else(|| SmtError::UnknownVariable(name.clone()))?;
                if *sort != Sort::Int {
                    return Err(SmtError::TypeMismatch {
                        name: name.clone(),
                        expected: "Int".to_string(),
                        actual: format!("{}", sort),
                    });
                }
                z3_const.as_int().ok_or_else(|| SmtError::TypeMismatch {
                    name: name.clone(),
                    expected: "Int".to_string(),
                    actual: "non-Int".to_string(),
                })
            }

            Expr::Add(left, right) => {
                let l = self.translate_int(left)?;
                let r = self.translate_int(right)?;
                Ok(Int::add(self.ctx, &[&l, &r]))
            }

            Expr::Sub(left, right) => {
                let l = self.translate_int(left)?;
                let r = self.translate_int(right)?;
                Ok(Int::sub(self.ctx, &[&l, &r]))
            }

            Expr::Mul(left, right) => {
                let l = self.translate_int(left)?;
                let r = self.translate_int(right)?;
                Ok(Int::mul(self.ctx, &[&l, &r]))
            }

            Expr::IntDiv(left, right) => {
                let l = self.translate_int(left)?;
                let r = self.translate_int(right)?;
                Ok(l.div(&r))
            }

            Expr::Mod(left, right) => {
                let l = self.translate_int(left)?;
                let r = self.translate_int(right)?;
                Ok(l.modulo(&r))
            }

            Expr::Neg(inner) => {
                let i = self.translate_int(inner)?;
                Ok(i.unary_minus())
            }

            Expr::If(cond, then_branch, else_branch) => {
                let c = self.translate_bool(cond)?;
                let t = self.translate_int(then_branch)?;
                let e = self.translate_int(else_branch)?;
                Ok(c.ite(&t, &e))
            }

            _ => Err(SmtError::UntranslatableExpr(format!(
                "expression type {:?} cannot be translated to Int",
                std::mem::discriminant(&expr.node)
            ))),
        }
    }

    /// Translate a quantified formula (forall or exists)
    fn translate_quantifier(
        &self,
        bounds: &[BoundVar],
        body: &Spanned<Expr>,
        is_forall: bool,
    ) -> SmtResult<Bool<'ctx>> {
        // Handle bounded quantifiers by domain type
        // Common cases:
        // - Int: treat as unbounded integer domain
        // - Nat: add x >= 0 constraint
        // - BOOLEAN: expand to conjunction/disjunction
        // - Finite set enum {e1,e2,...}: expand to conjunction/disjunction

        // Try to expand finite bounded quantifiers first
        if let Some(expanded) = self.try_expand_finite_quantifier(bounds, body, is_forall)? {
            return Ok(expanded);
        }

        // For infinite domains (Int, Nat), add domain constraints
        let mut bound_consts = Vec::new();
        let mut domain_constraints = Vec::new();

        for bound in bounds {
            let name = &bound.name.node;
            let z3_const = Int::new_const(self.ctx, name.as_str());

            // Check for domain constraints
            if let Some(domain) = &bound.domain {
                match &domain.node {
                    Expr::Ident(dom_name) if dom_name == "Int" => {
                        // Int domain: no additional constraint needed
                    }
                    Expr::Ident(dom_name) if dom_name == "Nat" => {
                        // Nat domain: add x >= 0 constraint
                        let zero = Int::from_i64(self.ctx, 0);
                        domain_constraints.push(z3_const.ge(&zero));
                    }
                    _ => {
                        // Unknown domain - return error for unsupported cases
                        return Err(SmtError::UnsupportedOp(
                            "bounded quantifier over non-standard domain not supported in SMT"
                                .to_string(),
                        ));
                    }
                }
            }

            bound_consts.push(z3_const);
        }

        // Translate the body in a context where bound variables are in scope
        let body_bool = self.translate_bool_with_bound_vars(body, bounds, &bound_consts)?;

        // Build the quantifier with domain constraints
        let bound_refs: Vec<_> = bound_consts.iter().map(|c| c as &dyn Ast).collect();

        let final_body = if domain_constraints.is_empty() {
            body_bool
        } else if is_forall {
            // forall x in Nat : P(x) => forall x. (x >= 0) => P(x)
            let domain_conj = Bool::and(self.ctx, &domain_constraints.iter().collect::<Vec<_>>());
            domain_conj.implies(&body_bool)
        } else {
            // exists x in Nat : P(x) => exists x. (x >= 0) /\ P(x)
            let domain_conj = Bool::and(self.ctx, &domain_constraints.iter().collect::<Vec<_>>());
            Bool::and(self.ctx, &[&domain_conj, &body_bool])
        };

        if is_forall {
            Ok(z3::ast::forall_const(
                self.ctx,
                &bound_refs,
                &[],
                &final_body,
            ))
        } else {
            Ok(z3::ast::exists_const(
                self.ctx,
                &bound_refs,
                &[],
                &final_body,
            ))
        }
    }

    /// Try to expand a bounded quantifier over a finite domain
    fn try_expand_finite_quantifier(
        &self,
        bounds: &[BoundVar],
        body: &Spanned<Expr>,
        is_forall: bool,
    ) -> SmtResult<Option<Bool<'ctx>>> {
        // Only handle single bound for now
        if bounds.len() != 1 {
            // For multiple bounds, check if all can be expanded
            return self.try_expand_multiple_finite_bounds(bounds, body, is_forall);
        }

        let bound = &bounds[0];
        let domain = match &bound.domain {
            Some(d) => d,
            None => return Ok(None), // Unbounded, not finite
        };

        // Check for BOOLEAN domain
        if let Expr::Ident(name) = &domain.node {
            if name == "BOOLEAN" {
                return self
                    .expand_boolean_quantifier(bound, body, is_forall)
                    .map(Some);
            }
        }

        // Check for finite set enumeration {e1, e2, ...}
        if let Expr::SetEnum(elements) = &domain.node {
            return self
                .expand_set_enum_quantifier(bound, elements, body, is_forall)
                .map(Some);
        }

        // Not a finite domain we can expand
        Ok(None)
    }

    /// Expand quantifier over multiple finite bounds
    fn try_expand_multiple_finite_bounds(
        &self,
        bounds: &[BoundVar],
        body: &Spanned<Expr>,
        is_forall: bool,
    ) -> SmtResult<Option<Bool<'ctx>>> {
        // Check if all bounds are over BOOLEAN or finite sets
        let all_finite = bounds.iter().all(|b| {
            b.domain.as_ref().is_some_and(|d| {
                matches!(&d.node, Expr::Ident(n) if n == "BOOLEAN")
                    || matches!(&d.node, Expr::SetEnum(_))
            })
        });

        if !all_finite {
            return Ok(None);
        }

        // For now, handle the special case of all BOOLEAN bounds
        let all_boolean = bounds.iter().all(|b| {
            b.domain
                .as_ref()
                .is_some_and(|d| matches!(&d.node, Expr::Ident(n) if n == "BOOLEAN"))
        });

        if all_boolean {
            return self
                .expand_multiple_boolean_quantifier(bounds, body, is_forall)
                .map(Some);
        }

        Ok(None)
    }

    /// Expand a quantifier over BOOLEAN: \A x \in BOOLEAN : P(x) => P(TRUE) /\ P(FALSE)
    fn expand_boolean_quantifier(
        &self,
        bound: &BoundVar,
        body: &Spanned<Expr>,
        is_forall: bool,
    ) -> SmtResult<Bool<'ctx>> {
        let var_name = &bound.name.node;

        // Substitute TRUE and FALSE into the body
        let body_true = self.substitute_var_bool(body, var_name, true)?;
        let body_false = self.substitute_var_bool(body, var_name, false)?;

        if is_forall {
            // P(TRUE) /\ P(FALSE)
            Ok(Bool::and(self.ctx, &[&body_true, &body_false]))
        } else {
            // P(TRUE) \/ P(FALSE)
            Ok(Bool::or(self.ctx, &[&body_true, &body_false]))
        }
    }

    /// Expand quantifier over multiple BOOLEAN bounds
    fn expand_multiple_boolean_quantifier(
        &self,
        bounds: &[BoundVar],
        body: &Spanned<Expr>,
        is_forall: bool,
    ) -> SmtResult<Bool<'ctx>> {
        // Generate all combinations of TRUE/FALSE for all bounds
        let n = bounds.len();
        let num_combinations = 1 << n;

        let mut substituted_bodies = Vec::new();

        for i in 0..num_combinations {
            // Create substitution map
            let substitutions: Vec<(&str, bool)> = bounds
                .iter()
                .enumerate()
                .map(|(j, b)| (b.name.node.as_str(), (i >> j) & 1 == 1))
                .collect();

            let substituted = self.substitute_multiple_bools(body, &substitutions)?;
            substituted_bodies.push(substituted);
        }

        let body_refs: Vec<_> = substituted_bodies.iter().collect();

        if is_forall {
            Ok(Bool::and(self.ctx, &body_refs))
        } else {
            Ok(Bool::or(self.ctx, &body_refs))
        }
    }

    /// Expand a quantifier over a finite set enumeration
    fn expand_set_enum_quantifier(
        &self,
        bound: &BoundVar,
        elements: &[Spanned<Expr>],
        body: &Spanned<Expr>,
        is_forall: bool,
    ) -> SmtResult<Bool<'ctx>> {
        if elements.is_empty() {
            // Empty set: \A x \in {} : P(x) is TRUE, \E x \in {} : P(x) is FALSE
            return Ok(Bool::from_bool(self.ctx, is_forall));
        }

        let var_name = &bound.name.node;
        let mut substituted_bodies = Vec::new();

        for elem in elements {
            let substituted = self.substitute_var_expr(body, var_name, elem)?;
            substituted_bodies.push(substituted);
        }

        let body_refs: Vec<_> = substituted_bodies.iter().collect();

        if is_forall {
            Ok(Bool::and(self.ctx, &body_refs))
        } else {
            Ok(Bool::or(self.ctx, &body_refs))
        }
    }

    /// Substitute a boolean value for a variable in an expression and translate
    fn substitute_var_bool(
        &self,
        expr: &Spanned<Expr>,
        var_name: &str,
        value: bool,
    ) -> SmtResult<Bool<'ctx>> {
        let substituted = self.substitute_bool_in_expr(&expr.node, var_name, value);
        self.translate_bool(&Spanned::new(substituted, expr.span))
    }

    /// Substitute multiple boolean values
    fn substitute_multiple_bools(
        &self,
        expr: &Spanned<Expr>,
        substitutions: &[(&str, bool)],
    ) -> SmtResult<Bool<'ctx>> {
        let mut result = expr.node.clone();
        for (var_name, value) in substitutions {
            result = self.substitute_bool_in_expr(&result, var_name, *value);
        }
        self.translate_bool(&Spanned::new(result, expr.span))
    }

    /// Substitute an expression for a variable and translate
    fn substitute_var_expr(
        &self,
        body: &Spanned<Expr>,
        var_name: &str,
        replacement: &Spanned<Expr>,
    ) -> SmtResult<Bool<'ctx>> {
        let substituted = self.substitute_expr_in_expr(&body.node, var_name, &replacement.node);
        self.translate_bool(&Spanned::new(substituted, body.span))
    }

    /// Recursively substitute a boolean value for a variable in an expression
    #[allow(clippy::only_used_in_recursion)]
    fn substitute_bool_in_expr(&self, expr: &Expr, var_name: &str, value: bool) -> Expr {
        match expr {
            Expr::Ident(name) if name == var_name => Expr::Bool(value),
            Expr::Ident(_) | Expr::Bool(_) | Expr::Int(_) | Expr::String(_) => expr.clone(),
            Expr::And(l, r) => Expr::And(
                Box::new(Spanned::new(
                    self.substitute_bool_in_expr(&l.node, var_name, value),
                    l.span,
                )),
                Box::new(Spanned::new(
                    self.substitute_bool_in_expr(&r.node, var_name, value),
                    r.span,
                )),
            ),
            Expr::Or(l, r) => Expr::Or(
                Box::new(Spanned::new(
                    self.substitute_bool_in_expr(&l.node, var_name, value),
                    l.span,
                )),
                Box::new(Spanned::new(
                    self.substitute_bool_in_expr(&r.node, var_name, value),
                    r.span,
                )),
            ),
            Expr::Not(e) => Expr::Not(Box::new(Spanned::new(
                self.substitute_bool_in_expr(&e.node, var_name, value),
                e.span,
            ))),
            Expr::Implies(l, r) => Expr::Implies(
                Box::new(Spanned::new(
                    self.substitute_bool_in_expr(&l.node, var_name, value),
                    l.span,
                )),
                Box::new(Spanned::new(
                    self.substitute_bool_in_expr(&r.node, var_name, value),
                    r.span,
                )),
            ),
            Expr::Equiv(l, r) => Expr::Equiv(
                Box::new(Spanned::new(
                    self.substitute_bool_in_expr(&l.node, var_name, value),
                    l.span,
                )),
                Box::new(Spanned::new(
                    self.substitute_bool_in_expr(&r.node, var_name, value),
                    r.span,
                )),
            ),
            Expr::Eq(l, r) => Expr::Eq(
                Box::new(Spanned::new(
                    self.substitute_bool_in_expr(&l.node, var_name, value),
                    l.span,
                )),
                Box::new(Spanned::new(
                    self.substitute_bool_in_expr(&r.node, var_name, value),
                    r.span,
                )),
            ),
            Expr::Neq(l, r) => Expr::Neq(
                Box::new(Spanned::new(
                    self.substitute_bool_in_expr(&l.node, var_name, value),
                    l.span,
                )),
                Box::new(Spanned::new(
                    self.substitute_bool_in_expr(&r.node, var_name, value),
                    r.span,
                )),
            ),
            Expr::If(c, t, e) => Expr::If(
                Box::new(Spanned::new(
                    self.substitute_bool_in_expr(&c.node, var_name, value),
                    c.span,
                )),
                Box::new(Spanned::new(
                    self.substitute_bool_in_expr(&t.node, var_name, value),
                    t.span,
                )),
                Box::new(Spanned::new(
                    self.substitute_bool_in_expr(&e.node, var_name, value),
                    e.span,
                )),
            ),
            // For other expressions, return as-is (bounded var doesn't appear in them sensibly)
            _ => expr.clone(),
        }
    }

    /// Recursively substitute an expression for a variable in an expression
    #[allow(clippy::only_used_in_recursion)]
    fn substitute_expr_in_expr(&self, expr: &Expr, var_name: &str, replacement: &Expr) -> Expr {
        match expr {
            Expr::Ident(name) if name == var_name => replacement.clone(),
            Expr::Ident(_) | Expr::Bool(_) | Expr::Int(_) | Expr::String(_) => expr.clone(),
            Expr::And(l, r) => Expr::And(
                Box::new(Spanned::new(
                    self.substitute_expr_in_expr(&l.node, var_name, replacement),
                    l.span,
                )),
                Box::new(Spanned::new(
                    self.substitute_expr_in_expr(&r.node, var_name, replacement),
                    r.span,
                )),
            ),
            Expr::Or(l, r) => Expr::Or(
                Box::new(Spanned::new(
                    self.substitute_expr_in_expr(&l.node, var_name, replacement),
                    l.span,
                )),
                Box::new(Spanned::new(
                    self.substitute_expr_in_expr(&r.node, var_name, replacement),
                    r.span,
                )),
            ),
            Expr::Not(e) => Expr::Not(Box::new(Spanned::new(
                self.substitute_expr_in_expr(&e.node, var_name, replacement),
                e.span,
            ))),
            Expr::Implies(l, r) => Expr::Implies(
                Box::new(Spanned::new(
                    self.substitute_expr_in_expr(&l.node, var_name, replacement),
                    l.span,
                )),
                Box::new(Spanned::new(
                    self.substitute_expr_in_expr(&r.node, var_name, replacement),
                    r.span,
                )),
            ),
            Expr::Equiv(l, r) => Expr::Equiv(
                Box::new(Spanned::new(
                    self.substitute_expr_in_expr(&l.node, var_name, replacement),
                    l.span,
                )),
                Box::new(Spanned::new(
                    self.substitute_expr_in_expr(&r.node, var_name, replacement),
                    r.span,
                )),
            ),
            Expr::Eq(l, r) => Expr::Eq(
                Box::new(Spanned::new(
                    self.substitute_expr_in_expr(&l.node, var_name, replacement),
                    l.span,
                )),
                Box::new(Spanned::new(
                    self.substitute_expr_in_expr(&r.node, var_name, replacement),
                    r.span,
                )),
            ),
            Expr::Neq(l, r) => Expr::Neq(
                Box::new(Spanned::new(
                    self.substitute_expr_in_expr(&l.node, var_name, replacement),
                    l.span,
                )),
                Box::new(Spanned::new(
                    self.substitute_expr_in_expr(&r.node, var_name, replacement),
                    r.span,
                )),
            ),
            Expr::Lt(l, r) => Expr::Lt(
                Box::new(Spanned::new(
                    self.substitute_expr_in_expr(&l.node, var_name, replacement),
                    l.span,
                )),
                Box::new(Spanned::new(
                    self.substitute_expr_in_expr(&r.node, var_name, replacement),
                    r.span,
                )),
            ),
            Expr::Leq(l, r) => Expr::Leq(
                Box::new(Spanned::new(
                    self.substitute_expr_in_expr(&l.node, var_name, replacement),
                    l.span,
                )),
                Box::new(Spanned::new(
                    self.substitute_expr_in_expr(&r.node, var_name, replacement),
                    r.span,
                )),
            ),
            Expr::Gt(l, r) => Expr::Gt(
                Box::new(Spanned::new(
                    self.substitute_expr_in_expr(&l.node, var_name, replacement),
                    l.span,
                )),
                Box::new(Spanned::new(
                    self.substitute_expr_in_expr(&r.node, var_name, replacement),
                    r.span,
                )),
            ),
            Expr::Geq(l, r) => Expr::Geq(
                Box::new(Spanned::new(
                    self.substitute_expr_in_expr(&l.node, var_name, replacement),
                    l.span,
                )),
                Box::new(Spanned::new(
                    self.substitute_expr_in_expr(&r.node, var_name, replacement),
                    r.span,
                )),
            ),
            Expr::Add(l, r) => Expr::Add(
                Box::new(Spanned::new(
                    self.substitute_expr_in_expr(&l.node, var_name, replacement),
                    l.span,
                )),
                Box::new(Spanned::new(
                    self.substitute_expr_in_expr(&r.node, var_name, replacement),
                    r.span,
                )),
            ),
            Expr::Sub(l, r) => Expr::Sub(
                Box::new(Spanned::new(
                    self.substitute_expr_in_expr(&l.node, var_name, replacement),
                    l.span,
                )),
                Box::new(Spanned::new(
                    self.substitute_expr_in_expr(&r.node, var_name, replacement),
                    r.span,
                )),
            ),
            Expr::Mul(l, r) => Expr::Mul(
                Box::new(Spanned::new(
                    self.substitute_expr_in_expr(&l.node, var_name, replacement),
                    l.span,
                )),
                Box::new(Spanned::new(
                    self.substitute_expr_in_expr(&r.node, var_name, replacement),
                    r.span,
                )),
            ),
            Expr::IntDiv(l, r) => Expr::IntDiv(
                Box::new(Spanned::new(
                    self.substitute_expr_in_expr(&l.node, var_name, replacement),
                    l.span,
                )),
                Box::new(Spanned::new(
                    self.substitute_expr_in_expr(&r.node, var_name, replacement),
                    r.span,
                )),
            ),
            Expr::Mod(l, r) => Expr::Mod(
                Box::new(Spanned::new(
                    self.substitute_expr_in_expr(&l.node, var_name, replacement),
                    l.span,
                )),
                Box::new(Spanned::new(
                    self.substitute_expr_in_expr(&r.node, var_name, replacement),
                    r.span,
                )),
            ),
            Expr::Neg(e) => Expr::Neg(Box::new(Spanned::new(
                self.substitute_expr_in_expr(&e.node, var_name, replacement),
                e.span,
            ))),
            Expr::If(c, t, e) => Expr::If(
                Box::new(Spanned::new(
                    self.substitute_expr_in_expr(&c.node, var_name, replacement),
                    c.span,
                )),
                Box::new(Spanned::new(
                    self.substitute_expr_in_expr(&t.node, var_name, replacement),
                    t.span,
                )),
                Box::new(Spanned::new(
                    self.substitute_expr_in_expr(&e.node, var_name, replacement),
                    e.span,
                )),
            ),
            // For other expressions, return as-is
            _ => expr.clone(),
        }
    }

    /// Translate a boolean expression with bound variables in scope
    fn translate_bool_with_bound_vars(
        &self,
        expr: &Spanned<Expr>,
        bounds: &[BoundVar],
        bound_consts: &[Int<'ctx>],
    ) -> SmtResult<Bool<'ctx>> {
        // Create a mutable translator with bound variables
        // For now, we use a simple approach: translate the body directly
        // but handle Ident nodes that match bound variable names
        self.translate_bool_with_int_bindings(expr, bounds, bound_consts)
    }

    /// Translate with integer bindings for bound variables
    fn translate_bool_with_int_bindings(
        &self,
        expr: &Spanned<Expr>,
        bounds: &[BoundVar],
        bound_consts: &[Int<'ctx>],
    ) -> SmtResult<Bool<'ctx>> {
        match &expr.node {
            Expr::Bool(b) => Ok(Bool::from_bool(self.ctx, *b)),

            Expr::Ident(name) => {
                // Check if this is a bound variable
                for bound in bounds.iter() {
                    if &bound.name.node == name {
                        // This shouldn't happen in a boolean context for an int variable
                        // unless we're comparing it
                        return Err(SmtError::TypeMismatch {
                            name: name.clone(),
                            expected: "Bool".to_string(),
                            actual: "bound Int".to_string(),
                        });
                    }
                }

                // Otherwise use the normal lookup
                let (sort, z3_const) = self
                    .vars
                    .get(name)
                    .ok_or_else(|| SmtError::UnknownVariable(name.clone()))?;
                if *sort != Sort::Bool {
                    return Err(SmtError::TypeMismatch {
                        name: name.clone(),
                        expected: "Bool".to_string(),
                        actual: format!("{}", sort),
                    });
                }
                z3_const.as_bool().ok_or_else(|| SmtError::TypeMismatch {
                    name: name.clone(),
                    expected: "Bool".to_string(),
                    actual: "non-Bool".to_string(),
                })
            }

            Expr::And(left, right) => {
                let l = self.translate_bool_with_int_bindings(left, bounds, bound_consts)?;
                let r = self.translate_bool_with_int_bindings(right, bounds, bound_consts)?;
                Ok(Bool::and(self.ctx, &[&l, &r]))
            }

            Expr::Or(left, right) => {
                let l = self.translate_bool_with_int_bindings(left, bounds, bound_consts)?;
                let r = self.translate_bool_with_int_bindings(right, bounds, bound_consts)?;
                Ok(Bool::or(self.ctx, &[&l, &r]))
            }

            Expr::Not(inner) => {
                let i = self.translate_bool_with_int_bindings(inner, bounds, bound_consts)?;
                Ok(i.not())
            }

            Expr::Implies(left, right) => {
                let l = self.translate_bool_with_int_bindings(left, bounds, bound_consts)?;
                let r = self.translate_bool_with_int_bindings(right, bounds, bound_consts)?;
                Ok(l.implies(&r))
            }

            Expr::Equiv(left, right) => {
                let l = self.translate_bool_with_int_bindings(left, bounds, bound_consts)?;
                let r = self.translate_bool_with_int_bindings(right, bounds, bound_consts)?;
                Ok(l.iff(&r))
            }

            Expr::Eq(left, right) => {
                // Try integer comparison first
                if let (Ok(l), Ok(r)) = (
                    self.translate_int_with_bindings(left, bounds, bound_consts),
                    self.translate_int_with_bindings(right, bounds, bound_consts),
                ) {
                    Ok(l._eq(&r))
                } else if let (Ok(l), Ok(r)) = (
                    self.translate_bool_with_int_bindings(left, bounds, bound_consts),
                    self.translate_bool_with_int_bindings(right, bounds, bound_consts),
                ) {
                    Ok(l.iff(&r))
                } else {
                    Err(SmtError::UntranslatableExpr(
                        "cannot determine type for equality comparison".to_string(),
                    ))
                }
            }

            Expr::Neq(left, right) => {
                let eq = self.translate_bool_with_int_bindings(
                    &Spanned::new(Expr::Eq(left.clone(), right.clone()), expr.span),
                    bounds,
                    bound_consts,
                )?;
                Ok(eq.not())
            }

            Expr::Lt(left, right) => {
                let l = self.translate_int_with_bindings(left, bounds, bound_consts)?;
                let r = self.translate_int_with_bindings(right, bounds, bound_consts)?;
                Ok(l.lt(&r))
            }

            Expr::Leq(left, right) => {
                let l = self.translate_int_with_bindings(left, bounds, bound_consts)?;
                let r = self.translate_int_with_bindings(right, bounds, bound_consts)?;
                Ok(l.le(&r))
            }

            Expr::Gt(left, right) => {
                let l = self.translate_int_with_bindings(left, bounds, bound_consts)?;
                let r = self.translate_int_with_bindings(right, bounds, bound_consts)?;
                Ok(l.gt(&r))
            }

            Expr::Geq(left, right) => {
                let l = self.translate_int_with_bindings(left, bounds, bound_consts)?;
                let r = self.translate_int_with_bindings(right, bounds, bound_consts)?;
                Ok(l.ge(&r))
            }

            Expr::If(cond, then_branch, else_branch) => {
                let c = self.translate_bool_with_int_bindings(cond, bounds, bound_consts)?;
                let t = self.translate_bool_with_int_bindings(then_branch, bounds, bound_consts)?;
                let e = self.translate_bool_with_int_bindings(else_branch, bounds, bound_consts)?;
                Ok(c.ite(&t, &e))
            }

            Expr::Forall(inner_bounds, body) => self.translate_quantifier(inner_bounds, body, true),

            Expr::Exists(inner_bounds, body) => {
                self.translate_quantifier(inner_bounds, body, false)
            }

            _ => Err(SmtError::UntranslatableExpr(format!(
                "expression type {:?} cannot be translated to Bool",
                std::mem::discriminant(&expr.node)
            ))),
        }
    }

    /// Translate an integer expression with bound variables
    fn translate_int_with_bindings(
        &self,
        expr: &Spanned<Expr>,
        bounds: &[BoundVar],
        bound_consts: &[Int<'ctx>],
    ) -> SmtResult<Int<'ctx>> {
        match &expr.node {
            Expr::Int(n) => {
                let n_i64: i64 = n.try_into().map_err(|_| {
                    SmtError::UntranslatableExpr(format!("integer {} too large for SMT", n))
                })?;
                Ok(Int::from_i64(self.ctx, n_i64))
            }

            Expr::Ident(name) => {
                // Check if this is a bound variable
                for (i, bound) in bounds.iter().enumerate() {
                    if &bound.name.node == name {
                        return Ok(bound_consts[i].clone());
                    }
                }

                // Otherwise use normal lookup
                let (sort, z3_const) = self
                    .vars
                    .get(name)
                    .ok_or_else(|| SmtError::UnknownVariable(name.clone()))?;
                if *sort != Sort::Int {
                    return Err(SmtError::TypeMismatch {
                        name: name.clone(),
                        expected: "Int".to_string(),
                        actual: format!("{}", sort),
                    });
                }
                z3_const.as_int().ok_or_else(|| SmtError::TypeMismatch {
                    name: name.clone(),
                    expected: "Int".to_string(),
                    actual: "non-Int".to_string(),
                })
            }

            Expr::Add(left, right) => {
                let l = self.translate_int_with_bindings(left, bounds, bound_consts)?;
                let r = self.translate_int_with_bindings(right, bounds, bound_consts)?;
                Ok(Int::add(self.ctx, &[&l, &r]))
            }

            Expr::Sub(left, right) => {
                let l = self.translate_int_with_bindings(left, bounds, bound_consts)?;
                let r = self.translate_int_with_bindings(right, bounds, bound_consts)?;
                Ok(Int::sub(self.ctx, &[&l, &r]))
            }

            Expr::Mul(left, right) => {
                let l = self.translate_int_with_bindings(left, bounds, bound_consts)?;
                let r = self.translate_int_with_bindings(right, bounds, bound_consts)?;
                Ok(Int::mul(self.ctx, &[&l, &r]))
            }

            Expr::IntDiv(left, right) => {
                let l = self.translate_int_with_bindings(left, bounds, bound_consts)?;
                let r = self.translate_int_with_bindings(right, bounds, bound_consts)?;
                Ok(l.div(&r))
            }

            Expr::Mod(left, right) => {
                let l = self.translate_int_with_bindings(left, bounds, bound_consts)?;
                let r = self.translate_int_with_bindings(right, bounds, bound_consts)?;
                Ok(l.modulo(&r))
            }

            Expr::Neg(inner) => {
                let i = self.translate_int_with_bindings(inner, bounds, bound_consts)?;
                Ok(i.unary_minus())
            }

            Expr::If(cond, then_branch, else_branch) => {
                let c = self.translate_bool_with_int_bindings(cond, bounds, bound_consts)?;
                let t = self.translate_int_with_bindings(then_branch, bounds, bound_consts)?;
                let e = self.translate_int_with_bindings(else_branch, bounds, bound_consts)?;
                Ok(c.ite(&t, &e))
            }

            _ => Err(SmtError::UntranslatableExpr(format!(
                "expression type {:?} cannot be translated to Int",
                std::mem::discriminant(&expr.node)
            ))),
        }
    }

    /// Get the declared variables
    pub fn vars(&self) -> &HashMap<String, (Sort, Dynamic<'ctx>)> {
        &self.vars
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;
    use tla_core::span::{FileId, Span};

    fn span() -> Span {
        Span::new(FileId(0), 0, 0)
    }

    fn spanned<T>(node: T) -> Spanned<T> {
        Spanned::new(node, span())
    }

    #[test]
    fn test_translate_bool_literal() {
        let cfg = z3::Config::new();
        let ctx = Context::new(&cfg);
        let translator = SmtTranslator::new(&ctx);

        let expr_true = spanned(Expr::Bool(true));
        let expr_false = spanned(Expr::Bool(false));

        let z3_true = translator.translate_bool(&expr_true).unwrap();
        let z3_false = translator.translate_bool(&expr_false).unwrap();

        // Verify they are different
        let solver = z3::Solver::new(&ctx);
        solver.assert(&z3_true.iff(&z3_false));
        assert_eq!(solver.check(), z3::SatResult::Unsat);
    }

    #[test]
    fn test_translate_int_literal() {
        let cfg = z3::Config::new();
        let ctx = Context::new(&cfg);
        let translator = SmtTranslator::new(&ctx);

        let expr = spanned(Expr::Int(BigInt::from(42)));
        let z3_int = translator.translate_int(&expr).unwrap();

        // Verify the value
        let solver = z3::Solver::new(&ctx);
        let expected = Int::from_i64(&ctx, 42);
        solver.assert(&z3_int._eq(&expected).not());
        assert_eq!(solver.check(), z3::SatResult::Unsat);
    }

    #[test]
    fn test_translate_and() {
        let cfg = z3::Config::new();
        let ctx = Context::new(&cfg);
        let translator = SmtTranslator::new(&ctx);

        let expr = spanned(Expr::And(
            Box::new(spanned(Expr::Bool(true))),
            Box::new(spanned(Expr::Bool(false))),
        ));

        let z3_and = translator.translate_bool(&expr).unwrap();

        // true /\ false = false
        let solver = z3::Solver::new(&ctx);
        solver.assert(&z3_and);
        assert_eq!(solver.check(), z3::SatResult::Unsat);
    }

    #[test]
    fn test_translate_or() {
        let cfg = z3::Config::new();
        let ctx = Context::new(&cfg);
        let translator = SmtTranslator::new(&ctx);

        let expr = spanned(Expr::Or(
            Box::new(spanned(Expr::Bool(true))),
            Box::new(spanned(Expr::Bool(false))),
        ));

        let z3_or = translator.translate_bool(&expr).unwrap();

        // true \/ false = true
        let solver = z3::Solver::new(&ctx);
        solver.assert(&z3_or.not());
        assert_eq!(solver.check(), z3::SatResult::Unsat);
    }

    #[test]
    fn test_translate_variable() {
        let cfg = z3::Config::new();
        let ctx = Context::new(&cfg);
        let mut translator = SmtTranslator::new(&ctx);

        translator.declare_var("x", Sort::Int).unwrap();
        translator.declare_var("b", Sort::Bool).unwrap();

        let x_expr = spanned(Expr::Ident("x".to_string()));
        let b_expr = spanned(Expr::Ident("b".to_string()));

        let x_int = translator.translate_int(&x_expr).unwrap();
        let b_bool = translator.translate_bool(&b_expr).unwrap();

        // x = 5 /\ b = true should be satisfiable
        let solver = z3::Solver::new(&ctx);
        let five = Int::from_i64(&ctx, 5);
        solver.assert(&x_int._eq(&five));
        solver.assert(&b_bool);
        assert_eq!(solver.check(), z3::SatResult::Sat);
    }

    #[test]
    fn test_translate_comparison() {
        let cfg = z3::Config::new();
        let ctx = Context::new(&cfg);
        let mut translator = SmtTranslator::new(&ctx);

        translator.declare_var("x", Sort::Int).unwrap();

        // x > 5 /\ x < 3 should be unsat
        let x_expr = spanned(Expr::Ident("x".to_string()));
        let gt_expr = spanned(Expr::Gt(
            Box::new(x_expr.clone()),
            Box::new(spanned(Expr::Int(BigInt::from(5)))),
        ));
        let lt_expr = spanned(Expr::Lt(
            Box::new(x_expr),
            Box::new(spanned(Expr::Int(BigInt::from(3)))),
        ));

        let gt = translator.translate_bool(&gt_expr).unwrap();
        let lt = translator.translate_bool(&lt_expr).unwrap();

        let solver = z3::Solver::new(&ctx);
        solver.assert(&gt);
        solver.assert(&lt);
        assert_eq!(solver.check(), z3::SatResult::Unsat);
    }

    #[test]
    fn test_translate_arithmetic() {
        let cfg = z3::Config::new();
        let ctx = Context::new(&cfg);
        let mut translator = SmtTranslator::new(&ctx);

        translator.declare_var("x", Sort::Int).unwrap();

        // x + 3 = 7 implies x = 4
        let x_expr = spanned(Expr::Ident("x".to_string()));
        let sum = spanned(Expr::Add(
            Box::new(x_expr.clone()),
            Box::new(spanned(Expr::Int(BigInt::from(3)))),
        ));
        let eq = spanned(Expr::Eq(
            Box::new(sum),
            Box::new(spanned(Expr::Int(BigInt::from(7)))),
        ));

        let z3_eq = translator.translate_bool(&eq).unwrap();

        let solver = z3::Solver::new(&ctx);
        solver.assert(&z3_eq);

        // x /= 4 should be unsat with x + 3 = 7
        let x_int = translator.translate_int(&x_expr).unwrap();
        let four = Int::from_i64(&ctx, 4);
        solver.assert(&x_int._eq(&four).not());

        assert_eq!(solver.check(), z3::SatResult::Unsat);
    }

    #[test]
    fn test_translate_implies() {
        let cfg = z3::Config::new();
        let ctx = Context::new(&cfg);
        let translator = SmtTranslator::new(&ctx);

        // true => false should be false
        let expr = spanned(Expr::Implies(
            Box::new(spanned(Expr::Bool(true))),
            Box::new(spanned(Expr::Bool(false))),
        ));

        let z3_implies = translator.translate_bool(&expr).unwrap();

        let solver = z3::Solver::new(&ctx);
        solver.assert(&z3_implies);
        assert_eq!(solver.check(), z3::SatResult::Unsat);
    }

    #[test]
    fn test_translate_iff() {
        let cfg = z3::Config::new();
        let ctx = Context::new(&cfg);
        let mut translator = SmtTranslator::new(&ctx);

        translator.declare_var("a", Sort::Bool).unwrap();
        translator.declare_var("b", Sort::Bool).unwrap();

        // (a <=> b) /\ a /\ ~b should be unsat
        let a_expr = spanned(Expr::Ident("a".to_string()));
        let b_expr = spanned(Expr::Ident("b".to_string()));

        let iff = spanned(Expr::Equiv(
            Box::new(a_expr.clone()),
            Box::new(b_expr.clone()),
        ));
        let not_b = spanned(Expr::Not(Box::new(b_expr)));

        let z3_iff = translator.translate_bool(&iff).unwrap();
        let z3_a = translator.translate_bool(&a_expr).unwrap();
        let z3_not_b = translator.translate_bool(&not_b).unwrap();

        let solver = z3::Solver::new(&ctx);
        solver.assert(&z3_iff);
        solver.assert(&z3_a);
        solver.assert(&z3_not_b);
        assert_eq!(solver.check(), z3::SatResult::Unsat);
    }

    #[test]
    fn test_translate_if_bool() {
        let cfg = z3::Config::new();
        let ctx = Context::new(&cfg);
        let mut translator = SmtTranslator::new(&ctx);

        translator.declare_var("c", Sort::Bool).unwrap();

        // IF c THEN TRUE ELSE FALSE is equivalent to c
        let c_expr = spanned(Expr::Ident("c".to_string()));
        let if_expr = spanned(Expr::If(
            Box::new(c_expr.clone()),
            Box::new(spanned(Expr::Bool(true))),
            Box::new(spanned(Expr::Bool(false))),
        ));

        let z3_if = translator.translate_bool(&if_expr).unwrap();
        let z3_c = translator.translate_bool(&c_expr).unwrap();

        // (IF c THEN TRUE ELSE FALSE) <=> c should always be true
        let solver = z3::Solver::new(&ctx);
        solver.assert(&z3_if.iff(&z3_c).not());
        assert_eq!(solver.check(), z3::SatResult::Unsat);
    }

    #[test]
    fn test_translate_if_int() {
        let cfg = z3::Config::new();
        let ctx = Context::new(&cfg);
        let mut translator = SmtTranslator::new(&ctx);

        translator.declare_var("c", Sort::Bool).unwrap();
        translator.declare_var("x", Sort::Int).unwrap();

        // IF c THEN 1 ELSE 2 = x, c = true implies x = 1
        let c_expr = spanned(Expr::Ident("c".to_string()));
        let x_expr = spanned(Expr::Ident("x".to_string()));

        let if_expr = spanned(Expr::If(
            Box::new(c_expr.clone()),
            Box::new(spanned(Expr::Int(BigInt::from(1)))),
            Box::new(spanned(Expr::Int(BigInt::from(2)))),
        ));

        let z3_if = translator.translate_int(&if_expr).unwrap();
        let z3_c = translator.translate_bool(&c_expr).unwrap();
        let z3_x = translator.translate_int(&x_expr).unwrap();
        let one = Int::from_i64(&ctx, 1);

        // c /\ (IF c THEN 1 ELSE 2) = x /\ x /= 1 should be unsat
        let solver = z3::Solver::new(&ctx);
        solver.assert(&z3_c);
        solver.assert(&z3_if._eq(&z3_x));
        solver.assert(&z3_x._eq(&one).not());
        assert_eq!(solver.check(), z3::SatResult::Unsat);
    }

    #[test]
    fn test_unknown_variable() {
        let cfg = z3::Config::new();
        let ctx = Context::new(&cfg);
        let translator = SmtTranslator::new(&ctx);

        let expr = spanned(Expr::Ident("unknown".to_string()));
        let result = translator.translate_bool(&expr);
        assert!(matches!(result, Err(SmtError::UnknownVariable(_))));
    }

    #[test]
    fn test_type_mismatch() {
        let cfg = z3::Config::new();
        let ctx = Context::new(&cfg);
        let mut translator = SmtTranslator::new(&ctx);

        translator.declare_var("x", Sort::Int).unwrap();

        let expr = spanned(Expr::Ident("x".to_string()));
        let result = translator.translate_bool(&expr);
        assert!(matches!(result, Err(SmtError::TypeMismatch { .. })));
    }

    #[test]
    fn test_bounded_forall_int() {
        // \A x \in Int : x > 5 => x > 3 (should be provable)
        let cfg = z3::Config::new();
        let ctx = Context::new(&cfg);
        let translator = SmtTranslator::new(&ctx);

        let expr = spanned(Expr::Forall(
            vec![BoundVar {
                name: spanned("x".to_string()),
                domain: Some(Box::new(spanned(Expr::Ident("Int".to_string())))),
                pattern: None,
            }],
            Box::new(spanned(Expr::Implies(
                Box::new(spanned(Expr::Gt(
                    Box::new(spanned(Expr::Ident("x".to_string()))),
                    Box::new(spanned(Expr::Int(BigInt::from(5)))),
                ))),
                Box::new(spanned(Expr::Gt(
                    Box::new(spanned(Expr::Ident("x".to_string()))),
                    Box::new(spanned(Expr::Int(BigInt::from(3)))),
                ))),
            ))),
        ));

        let z3_expr = translator.translate_bool(&expr).unwrap();

        // The formula should be valid (always true)
        let solver = z3::Solver::new(&ctx);
        solver.assert(&z3_expr.not());
        assert_eq!(solver.check(), z3::SatResult::Unsat);
    }

    #[test]
    fn test_bounded_forall_boolean_single() {
        // \A a \in BOOLEAN : a \/ ~a (should be provable - tautology)
        let cfg = z3::Config::new();
        let ctx = Context::new(&cfg);
        let translator = SmtTranslator::new(&ctx);

        let expr = spanned(Expr::Forall(
            vec![BoundVar {
                name: spanned("a".to_string()),
                domain: Some(Box::new(spanned(Expr::Ident("BOOLEAN".to_string())))),
                pattern: None,
            }],
            Box::new(spanned(Expr::Or(
                Box::new(spanned(Expr::Ident("a".to_string()))),
                Box::new(spanned(Expr::Not(Box::new(spanned(Expr::Ident(
                    "a".to_string(),
                )))))),
            ))),
        ));

        let z3_expr = translator.translate_bool(&expr).unwrap();

        // The formula should be valid
        let solver = z3::Solver::new(&ctx);
        solver.assert(&z3_expr.not());
        assert_eq!(solver.check(), z3::SatResult::Unsat);
    }

    #[test]
    fn test_bounded_forall_boolean_double() {
        // \A a, b \in BOOLEAN : (a /\ b) => a (should be provable)
        let cfg = z3::Config::new();
        let ctx = Context::new(&cfg);
        let translator = SmtTranslator::new(&ctx);

        let expr = spanned(Expr::Forall(
            vec![
                BoundVar {
                    name: spanned("a".to_string()),
                    domain: Some(Box::new(spanned(Expr::Ident("BOOLEAN".to_string())))),
                    pattern: None,
                },
                BoundVar {
                    name: spanned("b".to_string()),
                    domain: Some(Box::new(spanned(Expr::Ident("BOOLEAN".to_string())))),
                    pattern: None,
                },
            ],
            Box::new(spanned(Expr::Implies(
                Box::new(spanned(Expr::And(
                    Box::new(spanned(Expr::Ident("a".to_string()))),
                    Box::new(spanned(Expr::Ident("b".to_string()))),
                ))),
                Box::new(spanned(Expr::Ident("a".to_string()))),
            ))),
        ));

        let z3_expr = translator.translate_bool(&expr).unwrap();

        // The formula should be valid
        let solver = z3::Solver::new(&ctx);
        solver.assert(&z3_expr.not());
        assert_eq!(solver.check(), z3::SatResult::Unsat);
    }

    #[test]
    fn test_bounded_exists_set_enum() {
        // \E x \in {1, 2, 3} : x > 2 (should be satisfiable - x=3 works)
        let cfg = z3::Config::new();
        let ctx = Context::new(&cfg);
        let translator = SmtTranslator::new(&ctx);

        let expr = spanned(Expr::Exists(
            vec![BoundVar {
                name: spanned("x".to_string()),
                domain: Some(Box::new(spanned(Expr::SetEnum(vec![
                    spanned(Expr::Int(BigInt::from(1))),
                    spanned(Expr::Int(BigInt::from(2))),
                    spanned(Expr::Int(BigInt::from(3))),
                ])))),
                pattern: None,
            }],
            Box::new(spanned(Expr::Gt(
                Box::new(spanned(Expr::Ident("x".to_string()))),
                Box::new(spanned(Expr::Int(BigInt::from(2)))),
            ))),
        ));

        let z3_expr = translator.translate_bool(&expr).unwrap();

        // The formula should be satisfiable
        let solver = z3::Solver::new(&ctx);
        solver.assert(&z3_expr);
        assert_eq!(solver.check(), z3::SatResult::Sat);
    }

    #[test]
    fn test_bounded_forall_nat() {
        // \A x \in Nat : x >= 0 (should be provable by constraint)
        let cfg = z3::Config::new();
        let ctx = Context::new(&cfg);
        let translator = SmtTranslator::new(&ctx);

        let expr = spanned(Expr::Forall(
            vec![BoundVar {
                name: spanned("x".to_string()),
                domain: Some(Box::new(spanned(Expr::Ident("Nat".to_string())))),
                pattern: None,
            }],
            Box::new(spanned(Expr::Geq(
                Box::new(spanned(Expr::Ident("x".to_string()))),
                Box::new(spanned(Expr::Int(BigInt::from(0)))),
            ))),
        ));

        let z3_expr = translator.translate_bool(&expr).unwrap();

        // The formula should be valid
        let solver = z3::Solver::new(&ctx);
        solver.assert(&z3_expr.not());
        assert_eq!(solver.check(), z3::SatResult::Unsat);
    }
}
