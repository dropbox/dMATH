//! AST to LiveExpr conversion
//!
//! This module converts TLA+ AST expressions to the internal LiveExpr
//! representation used for liveness checking.
//!
//! Based on TLC's `astToLive` method in Liveness.java.

use super::live_expr::{ExprLevel, LiveExpr};
use crate::eval::{apply_substitutions, eval, EvalCtx, OpEnv};
use crate::Value;
use num_bigint::BigInt;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::Arc;
use tla_core::ast::{BoundPattern, BoundVar, Expr, ModuleTarget, Substitution};
use tla_core::Spanned;

/// Converter from AST to LiveExpr
///
/// Maintains state during conversion, including tag allocation for predicates.
pub struct AstToLive {
    /// Next tag to assign to predicates
    next_tag: Cell<u32>,
    /// Stack of module reference targets currently being inlined.
    ///
    /// When converting `M!Op` by inlining `Op`'s body, predicates inside that body
    /// may reference module-local operators (e.g., `Next`, `Init`). Those must be
    /// qualified so they can be evaluated later without module-local operator scope.
    target_stack: RefCell<Vec<Arc<ModuleTarget>>>,
    /// Stack of LET definitions currently in scope.
    ///
    /// When converting predicates inside `LET defs IN body`, the predicate expressions
    /// need to be wrapped in the LET so they can be evaluated later with the definitions
    /// in scope.
    let_defs_stack: RefCell<Vec<Vec<tla_core::ast::OperatorDef>>>,
}

struct TargetGuard<'a> {
    stack: &'a RefCell<Vec<Arc<ModuleTarget>>>,
}

impl Drop for TargetGuard<'_> {
    fn drop(&mut self) {
        self.stack.borrow_mut().pop();
    }
}

struct LetGuard<'a> {
    stack: &'a RefCell<Vec<Vec<tla_core::ast::OperatorDef>>>,
}

impl Drop for LetGuard<'_> {
    fn drop(&mut self) {
        self.stack.borrow_mut().pop();
    }
}

impl AstToLive {
    /// Create a new converter
    pub fn new() -> Self {
        Self {
            next_tag: Cell::new(1),
            target_stack: RefCell::new(Vec::new()),
            let_defs_stack: RefCell::new(Vec::new()),
        }
    }

    /// Push LET definitions onto the stack
    fn push_let_defs(&self, defs: Vec<tla_core::ast::OperatorDef>) -> LetGuard<'_> {
        self.let_defs_stack.borrow_mut().push(defs);
        LetGuard {
            stack: &self.let_defs_stack,
        }
    }

    /// Wrap an expression in any LET definitions currently in scope
    fn wrap_in_let_defs(&self, expr: Spanned<Expr>) -> Spanned<Expr> {
        let stack = self.let_defs_stack.borrow();
        if stack.is_empty() {
            return expr;
        }
        // Flatten all LET definitions and wrap the expression
        let all_defs: Vec<_> = stack.iter().flatten().cloned().collect();
        if all_defs.is_empty() {
            return expr;
        }
        Spanned {
            span: expr.span,
            node: Expr::Let(all_defs, Box::new(expr)),
        }
    }

    /// Allocate a new unique tag for a predicate
    pub fn alloc_tag(&self) -> u32 {
        let tag = self.next_tag.get();
        self.next_tag.set(tag + 1);
        tag
    }

    fn current_target(&self) -> Option<Arc<ModuleTarget>> {
        self.target_stack.borrow().last().cloned()
    }

    fn push_target(&self, target: Arc<ModuleTarget>) -> TargetGuard<'_> {
        self.target_stack.borrow_mut().push(target);
        TargetGuard {
            stack: &self.target_stack,
        }
    }

    fn qualify_predicate_expr(&self, ctx: &EvalCtx, expr: &Spanned<Expr>) -> Spanned<Expr> {
        let Some(target) = self.current_target() else {
            return expr.clone();
        };
        let Some(local_ops) = ctx.local_ops.as_ref() else {
            return expr.clone();
        };

        fn extend_bound_names_from_bv(
            bound: &BoundVar,
            out: &mut std::collections::HashSet<String>,
        ) {
            out.insert(bound.name.node.clone());
            if let Some(pattern) = &bound.pattern {
                match pattern {
                    BoundPattern::Var(v) => {
                        out.insert(v.node.clone());
                    }
                    BoundPattern::Tuple(vs) => {
                        for v in vs {
                            out.insert(v.node.clone());
                        }
                    }
                }
            }
        }

        fn clone_target(target: &ModuleTarget) -> ModuleTarget {
            target.clone()
        }

        fn qualify_expr(
            expr: &Spanned<Expr>,
            target: &ModuleTarget,
            local_ops: &crate::eval::OpEnv,
            bound: &std::collections::HashSet<String>,
        ) -> Spanned<Expr> {
            let node = match &expr.node {
                Expr::Bool(_) | Expr::Int(_) | Expr::String(_) | Expr::OpRef(_) => {
                    expr.node.clone()
                }

                Expr::Ident(name) => {
                    if bound.contains(name) {
                        Expr::Ident(name.clone())
                    } else if let Some(def) = local_ops.get(name) {
                        if def.params.is_empty() {
                            Expr::ModuleRef(target.clone(), name.clone(), Vec::new())
                        } else {
                            Expr::Ident(name.clone())
                        }
                    } else {
                        Expr::Ident(name.clone())
                    }
                }

                Expr::Apply(op, args) => {
                    let new_op = qualify_expr(op, target, local_ops, bound);
                    let new_args: Vec<_> = args
                        .iter()
                        .map(|a| qualify_expr(a, target, local_ops, bound))
                        .collect();
                    if let Expr::Ident(name) = &op.node {
                        if !bound.contains(name) && local_ops.contains_key(name) {
                            Expr::ModuleRef(target.clone(), name.clone(), new_args)
                        } else {
                            Expr::Apply(Box::new(new_op), new_args)
                        }
                    } else {
                        Expr::Apply(Box::new(new_op), new_args)
                    }
                }

                Expr::ModuleRef(t, op_name, args) => {
                    let new_args: Vec<_> = args
                        .iter()
                        .map(|a| qualify_expr(a, target, local_ops, bound))
                        .collect();
                    Expr::ModuleRef(clone_target(t), op_name.clone(), new_args)
                }
                Expr::InstanceExpr(module_name, subs) => Expr::InstanceExpr(
                    module_name.clone(),
                    subs.iter()
                        .map(|s| Substitution {
                            from: s.from.clone(),
                            to: qualify_expr(&s.to, target, local_ops, bound),
                        })
                        .collect(),
                ),

                Expr::Lambda(params, body) => {
                    let mut new_bound = bound.clone();
                    for p in params {
                        new_bound.insert(p.node.clone());
                    }
                    Expr::Lambda(
                        params.clone(),
                        Box::new(qualify_expr(body, target, local_ops, &new_bound)),
                    )
                }

                Expr::And(a, b) => Expr::And(
                    Box::new(qualify_expr(a, target, local_ops, bound)),
                    Box::new(qualify_expr(b, target, local_ops, bound)),
                ),
                Expr::Or(a, b) => Expr::Or(
                    Box::new(qualify_expr(a, target, local_ops, bound)),
                    Box::new(qualify_expr(b, target, local_ops, bound)),
                ),
                Expr::Not(a) => Expr::Not(Box::new(qualify_expr(a, target, local_ops, bound))),
                Expr::Implies(a, b) => Expr::Implies(
                    Box::new(qualify_expr(a, target, local_ops, bound)),
                    Box::new(qualify_expr(b, target, local_ops, bound)),
                ),
                Expr::Equiv(a, b) => Expr::Equiv(
                    Box::new(qualify_expr(a, target, local_ops, bound)),
                    Box::new(qualify_expr(b, target, local_ops, bound)),
                ),

                Expr::Forall(bounds, body) => {
                    let new_bounds: Vec<_> = bounds
                        .iter()
                        .map(|b| BoundVar {
                            name: b.name.clone(),
                            pattern: b.pattern.clone(),
                            domain: b
                                .domain
                                .as_ref()
                                .map(|d| Box::new(qualify_expr(d, target, local_ops, bound))),
                        })
                        .collect();

                    let mut new_bound = bound.clone();
                    for b in bounds {
                        extend_bound_names_from_bv(b, &mut new_bound);
                    }

                    Expr::Forall(
                        new_bounds,
                        Box::new(qualify_expr(body, target, local_ops, &new_bound)),
                    )
                }
                Expr::Exists(bounds, body) => {
                    let new_bounds: Vec<_> = bounds
                        .iter()
                        .map(|b| BoundVar {
                            name: b.name.clone(),
                            pattern: b.pattern.clone(),
                            domain: b
                                .domain
                                .as_ref()
                                .map(|d| Box::new(qualify_expr(d, target, local_ops, bound))),
                        })
                        .collect();

                    let mut new_bound = bound.clone();
                    for b in bounds {
                        extend_bound_names_from_bv(b, &mut new_bound);
                    }

                    Expr::Exists(
                        new_bounds,
                        Box::new(qualify_expr(body, target, local_ops, &new_bound)),
                    )
                }
                Expr::Choose(bv, body) => {
                    let new_bv = BoundVar {
                        name: bv.name.clone(),
                        pattern: bv.pattern.clone(),
                        domain: bv
                            .domain
                            .as_ref()
                            .map(|d| Box::new(qualify_expr(d, target, local_ops, bound))),
                    };

                    let mut new_bound = bound.clone();
                    extend_bound_names_from_bv(bv, &mut new_bound);
                    Expr::Choose(
                        new_bv,
                        Box::new(qualify_expr(body, target, local_ops, &new_bound)),
                    )
                }

                Expr::SetEnum(elems) => Expr::SetEnum(
                    elems
                        .iter()
                        .map(|e| qualify_expr(e, target, local_ops, bound))
                        .collect(),
                ),
                Expr::SetBuilder(body, bounds) => {
                    let new_bounds: Vec<_> = bounds
                        .iter()
                        .map(|b| BoundVar {
                            name: b.name.clone(),
                            pattern: b.pattern.clone(),
                            domain: b
                                .domain
                                .as_ref()
                                .map(|d| Box::new(qualify_expr(d, target, local_ops, bound))),
                        })
                        .collect();
                    let mut new_bound = bound.clone();
                    for b in bounds {
                        extend_bound_names_from_bv(b, &mut new_bound);
                    }
                    Expr::SetBuilder(
                        Box::new(qualify_expr(body, target, local_ops, &new_bound)),
                        new_bounds,
                    )
                }
                Expr::SetFilter(bv, pred) => {
                    let new_bv = BoundVar {
                        name: bv.name.clone(),
                        pattern: bv.pattern.clone(),
                        domain: bv
                            .domain
                            .as_ref()
                            .map(|d| Box::new(qualify_expr(d, target, local_ops, bound))),
                    };
                    let mut new_bound = bound.clone();
                    extend_bound_names_from_bv(bv, &mut new_bound);
                    Expr::SetFilter(
                        new_bv,
                        Box::new(qualify_expr(pred, target, local_ops, &new_bound)),
                    )
                }
                Expr::In(a, b) => Expr::In(
                    Box::new(qualify_expr(a, target, local_ops, bound)),
                    Box::new(qualify_expr(b, target, local_ops, bound)),
                ),
                Expr::NotIn(a, b) => Expr::NotIn(
                    Box::new(qualify_expr(a, target, local_ops, bound)),
                    Box::new(qualify_expr(b, target, local_ops, bound)),
                ),
                Expr::Subseteq(a, b) => Expr::Subseteq(
                    Box::new(qualify_expr(a, target, local_ops, bound)),
                    Box::new(qualify_expr(b, target, local_ops, bound)),
                ),
                Expr::Union(a, b) => Expr::Union(
                    Box::new(qualify_expr(a, target, local_ops, bound)),
                    Box::new(qualify_expr(b, target, local_ops, bound)),
                ),
                Expr::Intersect(a, b) => Expr::Intersect(
                    Box::new(qualify_expr(a, target, local_ops, bound)),
                    Box::new(qualify_expr(b, target, local_ops, bound)),
                ),
                Expr::SetMinus(a, b) => Expr::SetMinus(
                    Box::new(qualify_expr(a, target, local_ops, bound)),
                    Box::new(qualify_expr(b, target, local_ops, bound)),
                ),
                Expr::Powerset(a) => {
                    Expr::Powerset(Box::new(qualify_expr(a, target, local_ops, bound)))
                }
                Expr::BigUnion(a) => {
                    Expr::BigUnion(Box::new(qualify_expr(a, target, local_ops, bound)))
                }

                Expr::FuncDef(bounds, body) => {
                    let new_bounds: Vec<_> = bounds
                        .iter()
                        .map(|b| BoundVar {
                            name: b.name.clone(),
                            pattern: b.pattern.clone(),
                            domain: b
                                .domain
                                .as_ref()
                                .map(|d| Box::new(qualify_expr(d, target, local_ops, bound))),
                        })
                        .collect();
                    let mut new_bound = bound.clone();
                    for b in bounds {
                        extend_bound_names_from_bv(b, &mut new_bound);
                    }
                    Expr::FuncDef(
                        new_bounds,
                        Box::new(qualify_expr(body, target, local_ops, &new_bound)),
                    )
                }
                Expr::FuncApply(f, arg) => Expr::FuncApply(
                    Box::new(qualify_expr(f, target, local_ops, bound)),
                    Box::new(qualify_expr(arg, target, local_ops, bound)),
                ),
                Expr::Domain(f) => {
                    Expr::Domain(Box::new(qualify_expr(f, target, local_ops, bound)))
                }
                Expr::Except(base, specs) => {
                    let new_base = Box::new(qualify_expr(base, target, local_ops, bound));
                    let new_specs = specs
                        .iter()
                        .map(|spec| tla_core::ast::ExceptSpec {
                            path: spec
                                .path
                                .iter()
                                .map(|elem| match elem {
                                    tla_core::ast::ExceptPathElement::Index(idx) => {
                                        tla_core::ast::ExceptPathElement::Index(qualify_expr(
                                            idx, target, local_ops, bound,
                                        ))
                                    }
                                    tla_core::ast::ExceptPathElement::Field(f) => {
                                        tla_core::ast::ExceptPathElement::Field(f.clone())
                                    }
                                })
                                .collect(),
                            value: qualify_expr(&spec.value, target, local_ops, bound),
                        })
                        .collect();
                    Expr::Except(new_base, new_specs)
                }
                Expr::FuncSet(dom, ran) => Expr::FuncSet(
                    Box::new(qualify_expr(dom, target, local_ops, bound)),
                    Box::new(qualify_expr(ran, target, local_ops, bound)),
                ),

                Expr::Record(fields) => Expr::Record(
                    fields
                        .iter()
                        .map(|(k, v)| (k.clone(), qualify_expr(v, target, local_ops, bound)))
                        .collect(),
                ),
                Expr::RecordAccess(rec, field) => Expr::RecordAccess(
                    Box::new(qualify_expr(rec, target, local_ops, bound)),
                    field.clone(),
                ),
                Expr::RecordSet(fields) => Expr::RecordSet(
                    fields
                        .iter()
                        .map(|(k, v)| (k.clone(), qualify_expr(v, target, local_ops, bound)))
                        .collect(),
                ),

                Expr::Tuple(elems) => Expr::Tuple(
                    elems
                        .iter()
                        .map(|e| qualify_expr(e, target, local_ops, bound))
                        .collect(),
                ),
                Expr::Times(elems) => Expr::Times(
                    elems
                        .iter()
                        .map(|e| qualify_expr(e, target, local_ops, bound))
                        .collect(),
                ),

                Expr::Prime(a) => Expr::Prime(Box::new(qualify_expr(a, target, local_ops, bound))),
                Expr::Always(a) => {
                    Expr::Always(Box::new(qualify_expr(a, target, local_ops, bound)))
                }
                Expr::Eventually(a) => {
                    Expr::Eventually(Box::new(qualify_expr(a, target, local_ops, bound)))
                }
                Expr::LeadsTo(a, b) => Expr::LeadsTo(
                    Box::new(qualify_expr(a, target, local_ops, bound)),
                    Box::new(qualify_expr(b, target, local_ops, bound)),
                ),
                Expr::WeakFair(a, b) => Expr::WeakFair(
                    Box::new(qualify_expr(a, target, local_ops, bound)),
                    Box::new(qualify_expr(b, target, local_ops, bound)),
                ),
                Expr::StrongFair(a, b) => Expr::StrongFair(
                    Box::new(qualify_expr(a, target, local_ops, bound)),
                    Box::new(qualify_expr(b, target, local_ops, bound)),
                ),
                Expr::Enabled(a) => {
                    Expr::Enabled(Box::new(qualify_expr(a, target, local_ops, bound)))
                }
                Expr::Unchanged(a) => {
                    Expr::Unchanged(Box::new(qualify_expr(a, target, local_ops, bound)))
                }

                Expr::If(cond, then_e, else_e) => Expr::If(
                    Box::new(qualify_expr(cond, target, local_ops, bound)),
                    Box::new(qualify_expr(then_e, target, local_ops, bound)),
                    Box::new(qualify_expr(else_e, target, local_ops, bound)),
                ),
                Expr::Case(arms, other) => Expr::Case(
                    arms.iter()
                        .map(|arm| tla_core::ast::CaseArm {
                            guard: qualify_expr(&arm.guard, target, local_ops, bound),
                            body: qualify_expr(&arm.body, target, local_ops, bound),
                        })
                        .collect(),
                    other
                        .as_ref()
                        .map(|e| Box::new(qualify_expr(e, target, local_ops, bound))),
                ),
                Expr::Let(defs, body) => {
                    let new_defs: Vec<_> = defs
                        .iter()
                        .map(|d| tla_core::ast::OperatorDef {
                            name: d.name.clone(),
                            params: d.params.clone(),
                            body: qualify_expr(&d.body, target, local_ops, bound),
                            local: d.local,
                        })
                        .collect();
                    let mut new_bound = bound.clone();
                    for d in defs {
                        new_bound.insert(d.name.node.clone());
                    }
                    Expr::Let(
                        new_defs,
                        Box::new(qualify_expr(body, target, local_ops, &new_bound)),
                    )
                }

                Expr::Eq(a, b) => Expr::Eq(
                    Box::new(qualify_expr(a, target, local_ops, bound)),
                    Box::new(qualify_expr(b, target, local_ops, bound)),
                ),
                Expr::Neq(a, b) => Expr::Neq(
                    Box::new(qualify_expr(a, target, local_ops, bound)),
                    Box::new(qualify_expr(b, target, local_ops, bound)),
                ),
                Expr::Lt(a, b) => Expr::Lt(
                    Box::new(qualify_expr(a, target, local_ops, bound)),
                    Box::new(qualify_expr(b, target, local_ops, bound)),
                ),
                Expr::Leq(a, b) => Expr::Leq(
                    Box::new(qualify_expr(a, target, local_ops, bound)),
                    Box::new(qualify_expr(b, target, local_ops, bound)),
                ),
                Expr::Gt(a, b) => Expr::Gt(
                    Box::new(qualify_expr(a, target, local_ops, bound)),
                    Box::new(qualify_expr(b, target, local_ops, bound)),
                ),
                Expr::Geq(a, b) => Expr::Geq(
                    Box::new(qualify_expr(a, target, local_ops, bound)),
                    Box::new(qualify_expr(b, target, local_ops, bound)),
                ),

                Expr::Add(a, b) => Expr::Add(
                    Box::new(qualify_expr(a, target, local_ops, bound)),
                    Box::new(qualify_expr(b, target, local_ops, bound)),
                ),
                Expr::Sub(a, b) => Expr::Sub(
                    Box::new(qualify_expr(a, target, local_ops, bound)),
                    Box::new(qualify_expr(b, target, local_ops, bound)),
                ),
                Expr::Mul(a, b) => Expr::Mul(
                    Box::new(qualify_expr(a, target, local_ops, bound)),
                    Box::new(qualify_expr(b, target, local_ops, bound)),
                ),
                Expr::Div(a, b) => Expr::Div(
                    Box::new(qualify_expr(a, target, local_ops, bound)),
                    Box::new(qualify_expr(b, target, local_ops, bound)),
                ),
                Expr::IntDiv(a, b) => Expr::IntDiv(
                    Box::new(qualify_expr(a, target, local_ops, bound)),
                    Box::new(qualify_expr(b, target, local_ops, bound)),
                ),
                Expr::Mod(a, b) => Expr::Mod(
                    Box::new(qualify_expr(a, target, local_ops, bound)),
                    Box::new(qualify_expr(b, target, local_ops, bound)),
                ),
                Expr::Pow(a, b) => Expr::Pow(
                    Box::new(qualify_expr(a, target, local_ops, bound)),
                    Box::new(qualify_expr(b, target, local_ops, bound)),
                ),
                Expr::Neg(a) => Expr::Neg(Box::new(qualify_expr(a, target, local_ops, bound))),
                Expr::Range(a, b) => Expr::Range(
                    Box::new(qualify_expr(a, target, local_ops, bound)),
                    Box::new(qualify_expr(b, target, local_ops, bound)),
                ),
            };

            Spanned {
                node,
                span: expr.span,
            }
        }

        qualify_expr(
            expr,
            target.as_ref(),
            local_ops.as_ref(),
            &std::collections::HashSet::new(),
        )
    }

    /// Resolve an action expression by inlining operator references.
    ///
    /// When an action expression is stored in LiveExpr (e.g., for WF/SF fairness),
    /// operator references must be resolved/inlined so the expression can be
    /// evaluated later without requiring the original context to have those
    /// operators bound. This is especially important for INSTANCE'd modules where
    /// the referenced operators are only available through the instance context.
    fn resolve_action_expr(&self, ctx: &EvalCtx, expr: &Spanned<Expr>) -> Spanned<Expr> {
        if self.current_target().is_some() {
            return self.qualify_predicate_expr(ctx, expr);
        }
        let mut visited = std::collections::HashSet::new();
        let resolved_node = self.resolve_action_expr_node(ctx, &expr.node, &mut visited);
        Spanned {
            node: resolved_node,
            span: expr.span,
        }
    }

    /// Recursively resolve operator references in an expression node.
    ///
    /// This is used to inline operator definitions so that the resulting expression
    /// can be evaluated later without the original context's operator bindings.
    fn resolve_action_expr_node(
        &self,
        ctx: &EvalCtx,
        expr: &Expr,
        visited: &mut std::collections::HashSet<String>,
    ) -> Expr {
        match expr {
            // Identifier - try to resolve operator definition and inline it
            Expr::Ident(name) => {
                // Prevent infinite recursion for cyclic definitions
                if visited.contains(name) {
                    return expr.clone();
                }
                // Skip inlining for config-overridden constants - these have model values
                // that should be used instead of inlining the original operator definition.
                // E.g., `Done == CHOOSE v : v \notin Reg` with `Done = Done` in config
                // should use the model value @Done, not inline the CHOOSE.
                if ctx.is_config_constant(name) {
                    return expr.clone();
                }
                if let Some(op_def) = ctx.get_op(name) {
                    if op_def.params.is_empty() {
                        // IMPORTANT: When resolving an operator's body, we should NOT apply
                        // instance substitutions because the operator is defined in a module
                        // (possibly the outer module), and its body should be evaluated in
                        // that module's context, not the instance context.
                        //
                        // For example, if we have:
                        //   pcBar == IF \A q \in Procs : pc[q] = "Done" THEN "Done" ELSE "a"
                        //   R == INSTANCE Reachable WITH pc <- pcBar
                        //
                        // The `pc` inside pcBar refers to ParReach's pc variable, NOT
                        // Reachable's pc (which would be substituted with pcBar).
                        let outer_ctx = ctx.without_instance_substitutions();

                        // Mark as visited before recursing
                        visited.insert(name.clone());
                        let result =
                            self.resolve_action_expr_node(&outer_ctx, &op_def.body.node, visited);
                        visited.remove(name);
                        return result;
                    }
                }
                // Not an operator - check if there's an instance substitution for this identifier
                // (e.g., for VARIABLE names that get substituted via INSTANCE ... WITH ...)
                if let Some(subs) = ctx.instance_substitutions() {
                    for sub in subs {
                        if sub.from.node == *name {
                            // Recursively resolve the substituted expression, but WITHOUT
                            // the instance substitutions. The substitution expression is
                            // written in the outer module's context, so references to
                            // variables there should NOT be substituted.
                            // E.g., for `INSTANCE M WITH pc <- pcBar`, the `pcBar` expression
                            // may contain `pc` which refers to the outer module's `pc` variable.
                            let outer_ctx = ctx.without_instance_substitutions();
                            return self.resolve_action_expr_node(
                                &outer_ctx,
                                &sub.to.node,
                                visited,
                            );
                        }
                    }
                }
                expr.clone()
            }

            // ModuleRef - keep the reference intact.
            //
            // The evaluator can resolve `M!Op` with the correct instance substitutions and
            // module-local operator scope. Inlining module references here is unnecessary and can
            // be incorrect when the referenced operator body includes module-local operators with
            // parameters.
            Expr::ModuleRef(target, op_name, args) => Expr::ModuleRef(
                target.clone(),
                op_name.clone(),
                args.iter()
                    .map(|a| self.resolve_action_expr_spanned(ctx, a, visited))
                    .collect(),
            ),

            // === Compound expressions: recursively resolve subexpressions ===

            // Binary operators
            Expr::And(a, b) => Expr::And(
                Box::new(self.resolve_action_expr_spanned(ctx, a, visited)),
                Box::new(self.resolve_action_expr_spanned(ctx, b, visited)),
            ),
            Expr::Or(a, b) => Expr::Or(
                Box::new(self.resolve_action_expr_spanned(ctx, a, visited)),
                Box::new(self.resolve_action_expr_spanned(ctx, b, visited)),
            ),
            Expr::Implies(a, b) => Expr::Implies(
                Box::new(self.resolve_action_expr_spanned(ctx, a, visited)),
                Box::new(self.resolve_action_expr_spanned(ctx, b, visited)),
            ),
            Expr::Equiv(a, b) => Expr::Equiv(
                Box::new(self.resolve_action_expr_spanned(ctx, a, visited)),
                Box::new(self.resolve_action_expr_spanned(ctx, b, visited)),
            ),
            Expr::Eq(a, b) => Expr::Eq(
                Box::new(self.resolve_action_expr_spanned(ctx, a, visited)),
                Box::new(self.resolve_action_expr_spanned(ctx, b, visited)),
            ),
            Expr::Neq(a, b) => Expr::Neq(
                Box::new(self.resolve_action_expr_spanned(ctx, a, visited)),
                Box::new(self.resolve_action_expr_spanned(ctx, b, visited)),
            ),
            Expr::In(a, b) => Expr::In(
                Box::new(self.resolve_action_expr_spanned(ctx, a, visited)),
                Box::new(self.resolve_action_expr_spanned(ctx, b, visited)),
            ),
            Expr::NotIn(a, b) => Expr::NotIn(
                Box::new(self.resolve_action_expr_spanned(ctx, a, visited)),
                Box::new(self.resolve_action_expr_spanned(ctx, b, visited)),
            ),

            // Unary operators
            Expr::Not(e) => Expr::Not(Box::new(self.resolve_action_expr_spanned(ctx, e, visited))),
            Expr::Prime(e) => {
                Expr::Prime(Box::new(self.resolve_action_expr_spanned(ctx, e, visited)))
            }
            Expr::Enabled(e) => {
                Expr::Enabled(Box::new(self.resolve_action_expr_spanned(ctx, e, visited)))
            }
            Expr::Unchanged(e) => {
                Expr::Unchanged(Box::new(self.resolve_action_expr_spanned(ctx, e, visited)))
            }

            // IF-THEN-ELSE
            Expr::If(cond, then_e, else_e) => Expr::If(
                Box::new(self.resolve_action_expr_spanned(ctx, cond, visited)),
                Box::new(self.resolve_action_expr_spanned(ctx, then_e, visited)),
                Box::new(self.resolve_action_expr_spanned(ctx, else_e, visited)),
            ),

            // Quantifiers
            Expr::Exists(bounds, body) => {
                let new_bounds: Vec<_> = bounds
                    .iter()
                    .map(|b| BoundVar {
                        name: b.name.clone(),
                        pattern: b.pattern.clone(),
                        domain: b
                            .domain
                            .as_ref()
                            .map(|d| Box::new(self.resolve_action_expr_spanned(ctx, d, visited))),
                    })
                    .collect();
                Expr::Exists(
                    new_bounds,
                    Box::new(self.resolve_action_expr_spanned(ctx, body, visited)),
                )
            }
            Expr::Forall(bounds, body) => {
                let new_bounds: Vec<_> = bounds
                    .iter()
                    .map(|b| BoundVar {
                        name: b.name.clone(),
                        pattern: b.pattern.clone(),
                        domain: b
                            .domain
                            .as_ref()
                            .map(|d| Box::new(self.resolve_action_expr_spanned(ctx, d, visited))),
                    })
                    .collect();
                Expr::Forall(
                    new_bounds,
                    Box::new(self.resolve_action_expr_spanned(ctx, body, visited)),
                )
            }

            // LET-IN
            Expr::Let(defs, body) => {
                let new_defs: Vec<_> = defs
                    .iter()
                    .map(|d| tla_core::ast::OperatorDef {
                        name: d.name.clone(),
                        params: d.params.clone(),
                        body: self.resolve_action_expr_spanned(ctx, &d.body, visited),
                        local: d.local,
                    })
                    .collect();
                Expr::Let(
                    new_defs,
                    Box::new(self.resolve_action_expr_spanned(ctx, body, visited)),
                )
            }

            // Apply - try to inline parameterized operators
            Expr::Apply(op, args) => {
                // First resolve all arguments
                let new_args: Vec<_> = args
                    .iter()
                    .map(|a| self.resolve_action_expr_spanned(ctx, a, visited))
                    .collect();

                // Check if the operator is an Ident that resolves to a parameterized operator
                if let Expr::Ident(name) = &op.node {
                    // Prevent infinite recursion for cyclic definitions
                    if !visited.contains(name) {
                        if let Some(op_def) = ctx.get_op(name) {
                            // Inline the operator body with parameter substitution
                            if !op_def.params.is_empty()
                                && op_def.params.len() == new_args.len()
                            {
                                // Build a mapping from parameter names to argument expressions
                                let outer_ctx = ctx.without_instance_substitutions();

                                // Mark as visited before recursing
                                visited.insert(name.clone());

                                // Substitute parameters with arguments in the body
                                let substituted_body = self.substitute_params_in_expr(
                                    &op_def.body.node,
                                    &op_def.params,
                                    &new_args,
                                );

                                // Recursively resolve the substituted body
                                let result = self.resolve_action_expr_node(
                                    &outer_ctx,
                                    &substituted_body,
                                    visited,
                                );
                                visited.remove(name);
                                return result;
                            }
                        }
                    }
                }

                // Fall through: keep as Apply with resolved subexpressions
                Expr::Apply(
                    Box::new(self.resolve_action_expr_spanned(ctx, op, visited)),
                    new_args,
                )
            }

            // EXCEPT
            Expr::Except(base, specs) => {
                let new_specs: Vec<_> = specs
                    .iter()
                    .map(|s| {
                        let new_path: Vec<_> = s
                            .path
                            .iter()
                            .map(|p| match p {
                                tla_core::ast::ExceptPathElement::Index(idx) => {
                                    tla_core::ast::ExceptPathElement::Index(
                                        self.resolve_action_expr_spanned(ctx, idx, visited),
                                    )
                                }
                                tla_core::ast::ExceptPathElement::Field(f) => {
                                    tla_core::ast::ExceptPathElement::Field(f.clone())
                                }
                            })
                            .collect();
                        tla_core::ast::ExceptSpec {
                            path: new_path,
                            value: self.resolve_action_expr_spanned(ctx, &s.value, visited),
                        }
                    })
                    .collect();
                Expr::Except(
                    Box::new(self.resolve_action_expr_spanned(ctx, base, visited)),
                    new_specs,
                )
            }

            // Record
            Expr::Record(fields) => {
                let new_fields: Vec<_> = fields
                    .iter()
                    .map(|(k, v)| (k.clone(), self.resolve_action_expr_spanned(ctx, v, visited)))
                    .collect();
                Expr::Record(new_fields)
            }

            // Record access
            Expr::RecordAccess(record, field) => Expr::RecordAccess(
                Box::new(self.resolve_action_expr_spanned(ctx, record, visited)),
                field.clone(),
            ),

            // FuncApply
            Expr::FuncApply(func, arg) => Expr::FuncApply(
                Box::new(self.resolve_action_expr_spanned(ctx, func, visited)),
                Box::new(self.resolve_action_expr_spanned(ctx, arg, visited)),
            ),

            // Set operations
            Expr::SetEnum(elems) => {
                let new_elems: Vec<_> = elems
                    .iter()
                    .map(|e| self.resolve_action_expr_spanned(ctx, e, visited))
                    .collect();
                Expr::SetEnum(new_elems)
            }
            Expr::Union(a, b) => Expr::Union(
                Box::new(self.resolve_action_expr_spanned(ctx, a, visited)),
                Box::new(self.resolve_action_expr_spanned(ctx, b, visited)),
            ),
            Expr::Intersect(a, b) => Expr::Intersect(
                Box::new(self.resolve_action_expr_spanned(ctx, a, visited)),
                Box::new(self.resolve_action_expr_spanned(ctx, b, visited)),
            ),
            Expr::SetMinus(a, b) => Expr::SetMinus(
                Box::new(self.resolve_action_expr_spanned(ctx, a, visited)),
                Box::new(self.resolve_action_expr_spanned(ctx, b, visited)),
            ),

            // Tuple
            Expr::Tuple(elems) => {
                let new_elems: Vec<_> = elems
                    .iter()
                    .map(|e| self.resolve_action_expr_spanned(ctx, e, visited))
                    .collect();
                Expr::Tuple(new_elems)
            }

            // All other expressions: return unchanged (literals, etc.)
            _ => expr.clone(),
        }
    }

    /// Helper to resolve a spanned expression while threading through the visited set
    fn resolve_action_expr_spanned(
        &self,
        ctx: &EvalCtx,
        expr: &Spanned<Expr>,
        visited: &mut std::collections::HashSet<String>,
    ) -> Spanned<Expr> {
        Spanned {
            node: self.resolve_action_expr_node(ctx, &expr.node, visited),
            span: expr.span,
        }
    }

    /// Convert an AST expression to a LiveExpr
    ///
    /// The context is used to evaluate constant subexpressions.
    pub fn convert(&self, ctx: &EvalCtx, expr: &Spanned<Expr>) -> Result<LiveExpr, ConvertError> {
        self.convert_expr(ctx, &expr.node, Arc::new(expr.clone()))
    }

    /// Internal conversion with the original expression for predicates
    fn convert_expr(
        &self,
        ctx: &EvalCtx,
        expr: &Expr,
        original: Arc<Spanned<Expr>>,
    ) -> Result<LiveExpr, ConvertError> {
        // IMPORTANT: Handle ENABLED specially before level-based dispatch.
        // ENABLED is state-level but cannot be evaluated by `eval()` - it must
        // be converted to LiveExpr::Enabled and handled by eval_live_expr.
        // This also applies to expressions containing ENABLED (like ~ENABLED A).
        match expr {
            // ENABLED A -> LiveExpr::Enabled
            Expr::Enabled(inner) => {
                let qualified = self.qualify_predicate_expr(ctx, inner);
                return Ok(LiveExpr::enabled(Arc::new(qualified), self.alloc_tag()));
            }
            // NOT containing ENABLED -> LiveExpr::Not(convert inner)
            Expr::Not(inner) if Self::contains_enabled(&inner.node) => {
                let inner_live = self.convert(ctx, inner)?;
                return Ok(LiveExpr::not(inner_live));
            }
            // AND/OR containing ENABLED -> convert recursively
            Expr::And(a, b)
                if Self::contains_enabled(&a.node) || Self::contains_enabled(&b.node) =>
            {
                let left_live = self.convert(ctx, a)?;
                let right_live = self.convert(ctx, b)?;
                return Ok(LiveExpr::and(vec![left_live, right_live]));
            }
            Expr::Or(a, b)
                if Self::contains_enabled(&a.node) || Self::contains_enabled(&b.node) =>
            {
                let left_live = self.convert(ctx, a)?;
                let right_live = self.convert(ctx, b)?;
                return Ok(LiveExpr::or(vec![left_live, right_live]));
            }
            Expr::Implies(a, b)
                if Self::contains_enabled(&a.node) || Self::contains_enabled(&b.node) =>
            {
                let left_live = self.convert(ctx, a)?;
                let right_live = self.convert(ctx, b)?;
                return Ok(LiveExpr::or(vec![LiveExpr::not(left_live), right_live]));
            }
            // EQUIV containing ENABLED: A <=> B is equivalent to (A /\ B) \/ (~A /\ ~B)
            Expr::Equiv(a, b)
                if Self::contains_enabled(&a.node) || Self::contains_enabled(&b.node) =>
            {
                let left_live = self.convert(ctx, a)?;
                let right_live = self.convert(ctx, b)?;
                // (A /\ B) \/ (~A /\ ~B)
                let both_true = LiveExpr::and(vec![left_live.clone(), right_live.clone()]);
                let both_false =
                    LiveExpr::and(vec![LiveExpr::not(left_live), LiveExpr::not(right_live)]);
                return Ok(LiveExpr::or(vec![both_true, both_false]));
            }
            _ => {}
        }

        // Determine expression level first - use context to resolve operator definitions
        let level = self.get_level_with_ctx(ctx, expr);

        match level {
            ExprLevel::Constant => {
                // Try to evaluate constant expressions
                match eval(ctx, &original) {
                    Ok(Value::Bool(b)) => Ok(LiveExpr::Bool(b)),
                    Ok(_) => Err(ConvertError::NonBooleanConstant(original)),
                    Err(_) => {
                        // If evaluation fails, treat as state predicate
                        // This can happen with unbound variables in properties
                        let qualified = self.qualify_predicate_expr(ctx, original.as_ref());
                        let wrapped = self.wrap_in_let_defs(qualified);
                        Ok(LiveExpr::state_pred(Arc::new(wrapped), self.alloc_tag()))
                    }
                }
            }
            ExprLevel::State => {
                // State-level predicate
                let qualified = self.qualify_predicate_expr(ctx, original.as_ref());
                let wrapped = self.wrap_in_let_defs(qualified);
                Ok(LiveExpr::state_pred(Arc::new(wrapped), self.alloc_tag()))
            }
            ExprLevel::Action => {
                // Action-level predicate (contains primed variables)
                let qualified = self.qualify_predicate_expr(ctx, original.as_ref());
                let wrapped = self.wrap_in_let_defs(qualified);
                Ok(LiveExpr::action_pred(Arc::new(wrapped), self.alloc_tag()))
            }
            ExprLevel::Temporal => {
                // Contains temporal operators - need to recurse
                self.convert_temporal(ctx, expr, original)
            }
        }
    }

    /// Check if an expression contains ENABLED (at any nesting level)
    fn contains_enabled(expr: &Expr) -> bool {
        match expr {
            Expr::Enabled(_) => true,
            Expr::Not(e) => Self::contains_enabled(&e.node),
            Expr::And(a, b) | Expr::Or(a, b) | Expr::Implies(a, b) | Expr::Equiv(a, b) => {
                Self::contains_enabled(&a.node) || Self::contains_enabled(&b.node)
            }
            Expr::Forall(_, body) | Expr::Exists(_, body) => Self::contains_enabled(&body.node),
            Expr::If(cond, then_e, else_e) => {
                Self::contains_enabled(&cond.node)
                    || Self::contains_enabled(&then_e.node)
                    || Self::contains_enabled(&else_e.node)
            }
            Expr::Let(defs, body) => {
                defs.iter().any(|d| Self::contains_enabled(&d.body.node))
                    || Self::contains_enabled(&body.node)
            }
            _ => false,
        }
    }

    /// Convert temporal-level expressions
    fn convert_temporal(
        &self,
        ctx: &EvalCtx,
        expr: &Expr,
        original: Arc<Spanned<Expr>>,
    ) -> Result<LiveExpr, ConvertError> {
        fn resolve_instance_info(
            ctx: &EvalCtx,
            instance_name: &str,
        ) -> Option<(String, Vec<Substitution>)> {
            if let Some(info) = ctx.get_instance(instance_name) {
                return Some((info.module_name.clone(), info.substitutions.clone()));
            }
            if let Some(def) = ctx.get_op(instance_name) {
                if !def.params.is_empty() {
                    return None;
                }
                if let Expr::InstanceExpr(module_name, substitutions) = &def.body.node {
                    return Some((module_name.clone(), substitutions.clone()));
                }
            }
            None
        }

        fn compose_instance_substitutions(
            instance_subs: &[Substitution],
            outer_subs: Option<&[Substitution]>,
        ) -> Vec<Substitution> {
            let Some(outer_subs) = outer_subs else {
                return instance_subs.to_vec();
            };
            if outer_subs.is_empty() {
                return instance_subs.to_vec();
            }

            let mut combined = Vec::with_capacity(instance_subs.len() + outer_subs.len());
            let mut overridden: std::collections::HashSet<&str> = std::collections::HashSet::new();

            for sub in instance_subs {
                overridden.insert(sub.from.node.as_str());
                combined.push(Substitution {
                    from: sub.from.clone(),
                    to: apply_substitutions(&sub.to, outer_subs),
                });
            }

            for sub in outer_subs {
                if overridden.contains(sub.from.node.as_str()) {
                    continue;
                }
                combined.push(sub.clone());
            }

            combined
        }

        fn resolve_module_target_ctx(ctx: &EvalCtx, target: &ModuleTarget) -> Option<EvalCtx> {
            match target {
                ModuleTarget::Named(name) => {
                    let (module_name, instance_subs) = resolve_instance_info(ctx, name)?;
                    let effective_subs = compose_instance_substitutions(
                        &instance_subs,
                        ctx.instance_substitutions(),
                    );
                    let instance_ops = ctx
                        .instance_ops()
                        .get(&module_name)
                        .cloned()
                        .unwrap_or_default();
                    Some(
                        ctx.with_local_ops(instance_ops)
                            .with_instance_substitutions(effective_subs),
                    )
                }
                ModuleTarget::Parameterized(name, params) => {
                    let def = ctx.get_op(name)?;
                    let Expr::InstanceExpr(module_name, instance_subs) = &def.body.node else {
                        return None;
                    };
                    if def.params.len() != params.len() {
                        return None;
                    }

                    // Bind instance operator parameters into the WITH substitutions.
                    let param_subs: Vec<Substitution> = def
                        .params
                        .iter()
                        .zip(params.iter())
                        .map(|(p, arg)| Substitution {
                            from: p.name.clone(),
                            to: arg.clone(),
                        })
                        .collect();

                    let instantiated_subs: Vec<Substitution> = instance_subs
                        .iter()
                        .map(|s| Substitution {
                            from: s.from.clone(),
                            to: apply_substitutions(&s.to, &param_subs),
                        })
                        .collect();

                    let effective_subs = compose_instance_substitutions(
                        &instantiated_subs,
                        ctx.instance_substitutions(),
                    );
                    let instance_ops = ctx
                        .instance_ops()
                        .get(module_name)
                        .cloned()
                        .unwrap_or_default();
                    Some(
                        ctx.with_local_ops(instance_ops)
                            .with_instance_substitutions(effective_subs),
                    )
                }
                ModuleTarget::Chained(base_expr) => resolve_chained_target_ctx(ctx, base_expr),
            }
        }

        fn resolve_chained_target_ctx(ctx: &EvalCtx, base_expr: &Spanned<Expr>) -> Option<EvalCtx> {
            let Expr::ModuleRef(base_target, inst_name, inst_args) = &base_expr.node else {
                return None;
            };
            if !inst_args.is_empty() {
                return None;
            }

            let base_ctx = resolve_module_target_ctx(ctx, base_target)?;
            let inst_def = base_ctx.get_op(inst_name)?;
            let Expr::InstanceExpr(module_name, inst_subs) = &inst_def.body.node else {
                return None;
            };

            let effective_subs =
                compose_instance_substitutions(inst_subs, base_ctx.instance_substitutions());
            let instance_ops = ctx
                .instance_ops()
                .get(module_name)
                .cloned()
                .unwrap_or_default();

            Some(
                base_ctx
                    .with_local_ops(instance_ops)
                    .with_instance_substitutions(effective_subs),
            )
        }

        match expr {
            // Boolean constants
            Expr::Bool(b) => Ok(LiveExpr::Bool(*b)),

            // Conjunction
            Expr::And(left, right) => {
                let left_live = self.convert(ctx, left)?;
                let right_live = self.convert(ctx, right)?;
                Ok(LiveExpr::and(vec![left_live, right_live]))
            }

            // Disjunction
            Expr::Or(left, right) => {
                let left_live = self.convert(ctx, left)?;
                let right_live = self.convert(ctx, right)?;
                Ok(LiveExpr::or(vec![left_live, right_live]))
            }

            // Negation
            Expr::Not(inner) => {
                let inner_live = self.convert(ctx, inner)?;
                Ok(LiveExpr::not(inner_live))
            }

            // Implication: A => B becomes ~A \/ B
            Expr::Implies(left, right) => {
                let left_live = self.convert(ctx, left)?;
                let right_live = self.convert(ctx, right)?;
                Ok(LiveExpr::or(vec![LiveExpr::not(left_live), right_live]))
            }

            // Equivalence: A <=> B becomes (A /\ B) \/ (~A /\ ~B)
            Expr::Equiv(left, right) => {
                let left_live = self.convert(ctx, left)?;
                let right_live = self.convert(ctx, right)?;
                // (A /\ B) \/ (~A /\ ~B)
                let both_true = LiveExpr::and(vec![left_live.clone(), right_live.clone()]);
                let both_false =
                    LiveExpr::and(vec![LiveExpr::not(left_live), LiveExpr::not(right_live)]);
                Ok(LiveExpr::or(vec![both_true, both_false]))
            }

            // Temporal: Always []P
            Expr::Always(inner) => {
                let inner_live = self.convert(ctx, inner)?;
                Ok(LiveExpr::always(inner_live))
            }

            // Temporal: Eventually <>P
            Expr::Eventually(inner) => {
                let inner_live = self.convert(ctx, inner)?;
                Ok(LiveExpr::eventually(inner_live))
            }

            // Leads-to: P ~> Q expands to [](P => <>Q) = [](~P \/ <>Q)
            Expr::LeadsTo(left, right) => {
                let left_live = self.convert(ctx, left)?;
                let right_live = self.convert(ctx, right)?;
                // [](~P \/ <>Q)
                let implies_eventually = LiveExpr::or(vec![
                    LiveExpr::not(left_live),
                    LiveExpr::eventually(right_live),
                ]);
                Ok(LiveExpr::always(implies_eventually))
            }

            // Weak Fairness: WF_e(A) expands to []<>(~ENABLED<A>_e \/ <A>_e)
            // i.e., "infinitely often, either A is not enabled or A happens"
            Expr::WeakFair(subscript, action) => {
                if std::env::var("TLA2_DEBUG_WF").is_ok() {
                    eprintln!("[DEBUG WF] Processing WeakFair");
                    eprintln!("[DEBUG WF] action = {:?}", action.node);
                    eprintln!("[DEBUG WF] current_target = {:?}", self.current_target());
                    if let Some(ops) = ctx.local_ops.as_ref() {
                        eprintln!("[DEBUG WF] local_ops keys: {:?}", ops.keys().collect::<Vec<_>>());
                    } else {
                        eprintln!("[DEBUG WF] local_ops = None");
                    }
                }
                // Resolve any operator references in the action expression so it can be
                // evaluated later without the original context's operator bindings.
                let resolved_action = self.resolve_action_expr(ctx, action);
                if std::env::var("TLA2_DEBUG_WF").is_ok() {
                    eprintln!("[DEBUG WF] resolved_action = {:?}", resolved_action.node);
                }

                // Resolve the subscript expression - this is the tuple we check for changes.
                // For WF_vars(A), vars is typically <<v1, v2, ...>>.
                let resolved_subscript = self.resolve_action_expr(ctx, subscript);
                let subscript_arc = Some(Arc::new(resolved_subscript));

                // ~ENABLED<<A>>_e (action is not enabled w/ stuttering exclusion)
                let not_enabled = LiveExpr::not(LiveExpr::enabled_subscripted(
                    Arc::new(resolved_action.clone()),
                    subscript_arc.clone(),
                    self.alloc_tag(),
                ));

                // <<A>>_e (action happens with stuttering exclusion)
                // Check (e' ≠ e) where e is the subscript expression.
                let action_occurs = LiveExpr::and(vec![
                    LiveExpr::action_pred(Arc::new(resolved_action), self.alloc_tag()),
                    LiveExpr::state_changed(subscript_arc, self.alloc_tag()),
                ]);

                // []<>(~ENABLED<<A>>_e \/ <<A>>_e)
                let disj = LiveExpr::or(vec![not_enabled, action_occurs]);
                Ok(LiveExpr::always(LiveExpr::eventually(disj)))
            }

            // Strong Fairness: SF_e(A) expands to <>[]~ENABLED<A>_e \/ []<><A>_e
            // i.e., "eventually always not enabled, or infinitely often happens"
            Expr::StrongFair(subscript, action) => {
                // Resolve any operator references in the action expression so it can be
                // evaluated later without the original context's operator bindings.
                let resolved_action = self.resolve_action_expr(ctx, action);

                // Resolve the subscript expression - this is the tuple we check for changes.
                let resolved_subscript = self.resolve_action_expr(ctx, subscript);
                let subscript_arc = Some(Arc::new(resolved_subscript));

                // <>[]~ENABLED<<A>>_e (eventually always not enabled w/ stuttering exclusion)
                let not_enabled = LiveExpr::not(LiveExpr::enabled_subscripted(
                    Arc::new(resolved_action.clone()),
                    subscript_arc.clone(),
                    self.alloc_tag(),
                ));
                let eventually_always_disabled =
                    LiveExpr::eventually(LiveExpr::always(not_enabled));

                // []<> <<A>>_e (infinitely often happens w/ stuttering exclusion)
                let action_occurs = LiveExpr::and(vec![
                    LiveExpr::action_pred(Arc::new(resolved_action), self.alloc_tag()),
                    LiveExpr::state_changed(subscript_arc, self.alloc_tag()),
                ]);
                let infinitely_often = LiveExpr::always(LiveExpr::eventually(action_occurs));

                Ok(LiveExpr::or(vec![
                    eventually_always_disabled,
                    infinitely_often,
                ]))
            }

            // ENABLED A
            Expr::Enabled(inner) => {
                // Resolve any operator references in the action expression
                let resolved_action = self.resolve_action_expr(ctx, inner);
                Ok(LiveExpr::enabled(
                    Arc::new(resolved_action),
                    self.alloc_tag(),
                ))
            }

            // Prime: x' - this is action level, handled by get_level returning Action
            Expr::Prime(_) => {
                let qualified = self.qualify_predicate_expr(ctx, original.as_ref());
                let wrapped = self.wrap_in_let_defs(qualified);
                Ok(LiveExpr::action_pred(Arc::new(wrapped), self.alloc_tag()))
            }

            // IF-THEN-ELSE: IF g THEN a ELSE b becomes (g /\ a) \/ (~g /\ b)
            Expr::If(guard, then_expr, else_expr) => {
                let guard_live = self.convert(ctx, guard)?;
                let then_live = self.convert(ctx, then_expr)?;
                let else_live = self.convert(ctx, else_expr)?;

                let then_branch = LiveExpr::and(vec![guard_live.clone(), then_live]);
                let else_branch = LiveExpr::and(vec![LiveExpr::not(guard_live), else_live]);

                Ok(LiveExpr::or(vec![then_branch, else_branch]))
            }

            // Bounded Exists: \E x \in S: Body
            // Expand to disjunction over all values in S
            // Based on TLC's Liveness.java OPCODE_be handling (lines 222-251)
            Expr::Exists(bounds, body) => {
                self.convert_bounded_quantifier(ctx, bounds, body, true, original)
            }

            // Bounded Forall: \A x \in S: Body
            // Expand to conjunction over all values in S
            // Based on TLC's Liveness.java OPCODE_bf handling (lines 253-286)
            Expr::Forall(bounds, body) => {
                self.convert_bounded_quantifier(ctx, bounds, body, false, original)
            }

            // Handle WF_xxx and SF_xxx parsed as function applications
            // This happens because the lexer matches "WF_vars" as a single identifier
            // instead of "WF_" token followed by "vars" identifier.
            Expr::Apply(op, args) if args.len() == 1 => {
                if let Expr::Ident(name) = &op.node {
                    if name.starts_with("WF_") {
                        // WF_vars(Action) -> []<>(~ENABLED<<Action>>_vars \/ <<Action>>_vars)
                        // Resolve the action expression to inline operator references
                        let action = &args[0];
                        let resolved_action = self.resolve_action_expr(ctx, action);

                        // Extract subscript from "WF_xxx" and resolve it.
                        // The subscript can be a simple identifier (e.g., "vars") or a tuple
                        // (e.g., "<<coordinator, participant>>"). When it's a tuple, we need
                        // to parse it as a tuple expression, not as an identifier.
                        let subscript_name = name.strip_prefix("WF_").unwrap();
                        let subscript_expr = if subscript_name.starts_with("<<") {
                            // Tuple subscript: parse the individual variable names
                            let inner = subscript_name
                                .trim_start_matches("<<")
                                .trim_end_matches(">>");
                            let var_names: Vec<_> = inner
                                .split(',')
                                .map(|s| s.trim())
                                .filter(|s| !s.is_empty())
                                .collect();
                            let tuple_elems: Vec<_> = var_names
                                .iter()
                                .map(|vn| Spanned::new(Expr::Ident((*vn).into()), op.span))
                                .collect();
                            Spanned::new(Expr::Tuple(tuple_elems), op.span)
                        } else {
                            // Simple identifier subscript
                            Spanned::new(Expr::Ident(subscript_name.into()), op.span)
                        };
                        let resolved_subscript = self.resolve_action_expr(ctx, &subscript_expr);
                        let subscript_arc = Some(Arc::new(resolved_subscript));

                        let not_enabled = LiveExpr::not(LiveExpr::enabled_subscripted(
                            Arc::new(resolved_action.clone()),
                            subscript_arc.clone(),
                            self.alloc_tag(),
                        ));
                        let action_occurs = LiveExpr::and(vec![
                            LiveExpr::action_pred(Arc::new(resolved_action), self.alloc_tag()),
                            LiveExpr::state_changed(subscript_arc, self.alloc_tag()),
                        ]);
                        let disj = LiveExpr::or(vec![not_enabled, action_occurs]);
                        return Ok(LiveExpr::always(LiveExpr::eventually(disj)));
                    }
                    if name.starts_with("SF_") {
                        // SF_vars(Action) -> <>[]~ENABLED<<Action>>_vars \/ []<> <<Action>>_vars
                        // Resolve the action expression to inline operator references
                        let action = &args[0];
                        let resolved_action = self.resolve_action_expr(ctx, action);

                        // Extract subscript from "SF_xxx" and resolve it.
                        // The subscript can be a simple identifier (e.g., "vars") or a tuple
                        // (e.g., "<<coordinator, participant>>"). When it's a tuple, we need
                        // to parse it as a tuple expression, not as an identifier.
                        let subscript_name = name.strip_prefix("SF_").unwrap();
                        let subscript_expr = if subscript_name.starts_with("<<") {
                            // Tuple subscript: parse the individual variable names
                            let inner = subscript_name
                                .trim_start_matches("<<")
                                .trim_end_matches(">>");
                            let var_names: Vec<_> = inner
                                .split(',')
                                .map(|s| s.trim())
                                .filter(|s| !s.is_empty())
                                .collect();
                            let tuple_elems: Vec<_> = var_names
                                .iter()
                                .map(|vn| Spanned::new(Expr::Ident((*vn).into()), op.span))
                                .collect();
                            Spanned::new(Expr::Tuple(tuple_elems), op.span)
                        } else {
                            // Simple identifier subscript
                            Spanned::new(Expr::Ident(subscript_name.into()), op.span)
                        };
                        let resolved_subscript = self.resolve_action_expr(ctx, &subscript_expr);
                        let subscript_arc = Some(Arc::new(resolved_subscript));

                        let not_enabled = LiveExpr::not(LiveExpr::enabled_subscripted(
                            Arc::new(resolved_action.clone()),
                            subscript_arc.clone(),
                            self.alloc_tag(),
                        ));
                        let eventually_always_disabled =
                            LiveExpr::eventually(LiveExpr::always(not_enabled));
                        let action_occurs = LiveExpr::and(vec![
                            LiveExpr::action_pred(Arc::new(resolved_action), self.alloc_tag()),
                            LiveExpr::state_changed(subscript_arc, self.alloc_tag()),
                        ]);
                        let infinitely_often =
                            LiveExpr::always(LiveExpr::eventually(action_occurs));
                        return Ok(LiveExpr::or(vec![
                            eventually_always_disabled,
                            infinitely_often,
                        ]));
                    }
                }
                // Fall through to default handling
                let actual_level = self.get_level_with_ctx(ctx, expr);
                match actual_level {
                    ExprLevel::Constant | ExprLevel::State => {
                        let qualified = self.qualify_predicate_expr(ctx, original.as_ref());
                        let wrapped = self.wrap_in_let_defs(qualified);
                        Ok(LiveExpr::state_pred(Arc::new(wrapped), self.alloc_tag()))
                    }
                    ExprLevel::Action => {
                        let qualified = self.qualify_predicate_expr(ctx, original.as_ref());
                        let wrapped = self.wrap_in_let_defs(qualified);
                        Ok(LiveExpr::action_pred(Arc::new(wrapped), self.alloc_tag()))
                    }
                    ExprLevel::Temporal => Err(ConvertError::UnsupportedTemporal(original)),
                }
            }

            // Module reference: look up the actual operator and convert its body
            Expr::ModuleRef(instance_name, op_name, args) => {
                // For now, only inline zero-argument module references.
                if args.is_empty() {
                    if let Some(instance_ctx) = resolve_module_target_ctx(ctx, instance_name) {
                        if let Some(op_def) = instance_ctx.get_op(op_name) {
                            if op_def.params.is_empty() {
                                let substituted_body = match instance_ctx.instance_substitutions() {
                                    Some(subs) => apply_substitutions(&op_def.body, subs),
                                    None => op_def.body.clone(),
                                };
                                let _guard = self.push_target(Arc::new(instance_name.clone()));
                                return self.convert(&instance_ctx, &substituted_body);
                            }
                        }
                    }
                }
                // If we can't resolve it, fall back to predicate
                let actual_level = self.get_level_with_ctx(ctx, expr);
                match actual_level {
                    ExprLevel::Constant | ExprLevel::State => {
                        let qualified = self.qualify_predicate_expr(ctx, original.as_ref());
                        let wrapped = self.wrap_in_let_defs(qualified);
                        Ok(LiveExpr::state_pred(Arc::new(wrapped), self.alloc_tag()))
                    }
                    ExprLevel::Action => {
                        let qualified = self.qualify_predicate_expr(ctx, original.as_ref());
                        let wrapped = self.wrap_in_let_defs(qualified);
                        Ok(LiveExpr::action_pred(Arc::new(wrapped), self.alloc_tag()))
                    }
                    ExprLevel::Temporal => Err(ConvertError::UnsupportedTemporal(original)),
                }
            }

            // Identifier reference: look up the operator definition and convert its body
            Expr::Ident(name) => {
                // Look up the operator to get its body (respecting module-local operator scopes)
                if let Some(op_def) = ctx.get_op(name) {
                    // Only inline zero-argument operators
                    if op_def.params.is_empty() {
                        let substituted_body = match ctx.instance_substitutions() {
                            Some(subs) => apply_substitutions(&op_def.body, subs),
                            None => op_def.body.clone(),
                        };
                        // Convert the (possibly substituted) operator body.
                        return self.convert(ctx, &substituted_body);
                    }
                }
                // If we can't resolve it, fall back to predicate
                let actual_level = self.get_level_with_ctx(ctx, expr);
                match actual_level {
                    ExprLevel::Constant | ExprLevel::State => {
                        let qualified = self.qualify_predicate_expr(ctx, original.as_ref());
                        let wrapped = self.wrap_in_let_defs(qualified);
                        Ok(LiveExpr::state_pred(Arc::new(wrapped), self.alloc_tag()))
                    }
                    ExprLevel::Action => {
                        let qualified = self.qualify_predicate_expr(ctx, original.as_ref());
                        let wrapped = self.wrap_in_let_defs(qualified);
                        Ok(LiveExpr::action_pred(Arc::new(wrapped), self.alloc_tag()))
                    }
                    ExprLevel::Temporal => Err(ConvertError::UnsupportedTemporal(original)),
                }
            }

            // LET expressions: add definitions to context and convert body
            Expr::Let(defs, body) => {
                // Clone existing local_ops or create new
                let mut local_ops: OpEnv = match ctx.local_ops.as_ref() {
                    Some(ops) => (**ops).clone(),
                    None => OpEnv::new(),
                };
                // Add LET definitions to local_ops for resolution during conversion
                for def in defs {
                    local_ops.insert(def.name.node.clone(), def.clone());
                }
                // Create new context with merged local_ops
                let new_ctx = ctx.with_local_ops(local_ops);
                // Push LET definitions onto stack so predicates can be wrapped
                let _guard = self.push_let_defs(defs.clone());
                // Convert the body with the new context
                self.convert(&new_ctx, body)
            }

            // For other expressions that reached temporal level, they must contain
            // temporal subexpressions. If we didn't match them above, it's an error.
            // However, we should be lenient and try to treat them as predicates if possible.
            _ => {
                // Fallback: treat as predicate based on actual level
                let actual_level = self.get_level_with_ctx(ctx, expr);
                match actual_level {
                    ExprLevel::Constant | ExprLevel::State => {
                        let qualified = self.qualify_predicate_expr(ctx, original.as_ref());
                        let wrapped = self.wrap_in_let_defs(qualified);
                        Ok(LiveExpr::state_pred(Arc::new(wrapped), self.alloc_tag()))
                    }
                    ExprLevel::Action => {
                        let qualified = self.qualify_predicate_expr(ctx, original.as_ref());
                        let wrapped = self.wrap_in_let_defs(qualified);
                        Ok(LiveExpr::action_pred(Arc::new(wrapped), self.alloc_tag()))
                    }
                    ExprLevel::Temporal => Err(ConvertError::UnsupportedTemporal(original)),
                }
            }
        }
    }

    /// Convert a bounded quantifier (\E or \A) over temporal formulas
    ///
    /// This expands quantified temporal formulas by enumerating all values in
    /// the domain and building a disjunction (for \E) or conjunction (for \A).
    ///
    /// The key insight from TLC's implementation is that we must **substitute**
    /// the concrete domain values into the body expression. Simply binding them
    /// in the context is not enough because the resulting LiveExpr stores AST
    /// references that are evaluated later without the quantifier context.
    ///
    /// Based on TLC's Liveness.java handling of OPCODE_be and OPCODE_bf.
    fn convert_bounded_quantifier(
        &self,
        ctx: &EvalCtx,
        bounds: &[BoundVar],
        body: &Spanned<Expr>,
        is_exists: bool,
        original: Arc<Spanned<Expr>>,
    ) -> Result<LiveExpr, ConvertError> {
        // Handle multiple bounds recursively
        if bounds.is_empty() {
            return self.convert(ctx, body);
        }

        let first = &bounds[0];
        let rest = &bounds[1..];

        // Get the domain - it must be evaluable at conversion time (constant level)
        let domain = match &first.domain {
            Some(d) => d,
            None => {
                // Unbounded quantifier - cannot handle in liveness
                return Err(ConvertError::UnsupportedTemporal(original));
            }
        };

        // Evaluate domain to get the set of values
        let domain_value = match eval(ctx, domain) {
            Ok(v) => v,
            Err(_) => {
                // Domain couldn't be evaluated - fall back to predicate
                // This can happen if domain depends on state variables
                let level = self.get_level_with_ctx(ctx, &original.node);
                if level <= ExprLevel::Action {
                    let qualified = self.qualify_predicate_expr(ctx, original.as_ref());
                    return Ok(LiveExpr::action_pred(Arc::new(qualified), self.alloc_tag()));
                }
                return Err(ConvertError::UnsupportedTemporal(original));
            }
        };

        // Iterate over domain values
        let iter = match domain_value.iter_set() {
            Some(i) => i,
            None => {
                // Not a set - cannot enumerate
                return Err(ConvertError::UnsupportedTemporal(original));
            }
        };

        // Collect converted bodies for each domain value
        let mut parts: Vec<LiveExpr> = Vec::new();

        for elem in iter {
            // Create substitution map for this bound variable
            let mut subs = HashMap::new();
            self.add_bound_var_substitutions(&mut subs, first, &elem);

            // Also bind in context (for evaluating any constant subexpressions)
            let bound_ctx = self.bind_bound_var(ctx, first, &elem);

            // Substitute the concrete value into the body expression
            let substituted_body = substitute_values_in_expr(body, &subs);

            // Recursively convert with remaining bounds
            let converted = self.convert_bounded_quantifier(
                &bound_ctx,
                rest,
                &substituted_body,
                is_exists,
                original.clone(),
            )?;

            parts.push(converted);
        }

        // Build the result: disjunction for \E, conjunction for \A
        if parts.is_empty() {
            // Empty domain: \E over empty set is FALSE, \A over empty set is TRUE
            // This matches TLC's behavior in Liveness.java
            Ok(LiveExpr::Bool(!is_exists))
        } else {
            // Check level of result - if not temporal, can treat as predicate
            let result = if is_exists {
                LiveExpr::or(parts)
            } else {
                LiveExpr::and(parts)
            };

            // If the result is temporal level, return it
            // Otherwise, we can potentially simplify
            Ok(result)
        }
    }

    /// Add substitution mappings for a bound variable
    ///
    /// Handles simple variable binding and tuple destructuring patterns.
    fn add_bound_var_substitutions(
        &self,
        subs: &mut HashMap<String, Value>,
        bound: &BoundVar,
        elem: &Value,
    ) {
        use tla_core::ast::BoundPattern;

        match &bound.pattern {
            Some(BoundPattern::Tuple(vars)) => {
                // Destructure tuple and add substitution for each variable
                if let Some(tuple) = elem.as_tuple() {
                    for (var, val) in vars.iter().zip(tuple.iter()) {
                        subs.insert(var.node.clone(), val.clone());
                    }
                } else {
                    // Not a tuple - just substitute the whole thing
                    subs.insert(bound.name.node.clone(), elem.clone());
                }
            }
            Some(BoundPattern::Var(var)) => {
                subs.insert(var.node.clone(), elem.clone());
            }
            None => {
                subs.insert(bound.name.node.clone(), elem.clone());
            }
        }
    }

    /// Bind a bound variable to a value in the context
    ///
    /// Handles simple variable binding and tuple destructuring patterns.
    fn bind_bound_var(&self, ctx: &EvalCtx, bound: &BoundVar, elem: &Value) -> EvalCtx {
        use tla_core::ast::BoundPattern;

        match &bound.pattern {
            Some(BoundPattern::Tuple(vars)) => {
                // Destructure tuple and bind each variable
                if let Some(tuple) = elem.as_tuple() {
                    let mut new_ctx = ctx.clone();
                    for (var, val) in vars.iter().zip(tuple.iter()) {
                        new_ctx = new_ctx.bind(var.node.clone(), val.clone());
                    }
                    new_ctx
                } else {
                    // Not a tuple - just bind the whole thing to the first variable
                    ctx.bind(bound.name.node.clone(), elem.clone())
                }
            }
            Some(BoundPattern::Var(var)) => ctx.bind(var.node.clone(), elem.clone()),
            None => ctx.bind(bound.name.node.clone(), elem.clone()),
        }
    }

    /// Determine the level of an expression with context for operator resolution
    ///
    /// This version looks up operator definitions to correctly classify identifiers
    /// that refer to action-level operators (containing primed variables).
    ///
    /// Returns:
    /// - Constant: No variables, can be evaluated statically
    /// - State: Depends on current state only (no primes, no temporal)
    /// - Action: Depends on current and next state (has primes)
    /// - Temporal: Contains temporal operators ([], <>, WF, SF, ~>)
    pub fn get_level_with_ctx(&self, ctx: &EvalCtx, expr: &Expr) -> ExprLevel {
        self.get_level_with_ctx_inner(ctx, expr, &mut std::collections::HashSet::new())
    }

    #[allow(clippy::only_used_in_recursion)]
    fn get_level_with_ctx_inner(
        &self,
        ctx: &EvalCtx,
        expr: &Expr,
        visited: &mut std::collections::HashSet<String>,
    ) -> ExprLevel {
        fn resolve_instance_info(
            ctx: &EvalCtx,
            instance_name: &str,
        ) -> Option<(String, Vec<Substitution>)> {
            if let Some(info) = ctx.get_instance(instance_name) {
                return Some((info.module_name.clone(), info.substitutions.clone()));
            }
            if let Some(def) = ctx.get_op(instance_name) {
                if let Expr::InstanceExpr(module_name, substitutions) = &def.body.node {
                    return Some((module_name.clone(), substitutions.clone()));
                }
            }
            None
        }

        fn compose_instance_substitutions(
            instance_subs: &[Substitution],
            outer_subs: Option<&[Substitution]>,
        ) -> Vec<Substitution> {
            let Some(outer_subs) = outer_subs else {
                return instance_subs.to_vec();
            };
            if outer_subs.is_empty() {
                return instance_subs.to_vec();
            }

            let mut combined = Vec::with_capacity(instance_subs.len() + outer_subs.len());
            let mut overridden: std::collections::HashSet<&str> = std::collections::HashSet::new();

            for sub in instance_subs {
                overridden.insert(sub.from.node.as_str());
                combined.push(Substitution {
                    from: sub.from.clone(),
                    to: apply_substitutions(&sub.to, outer_subs),
                });
            }

            for sub in outer_subs {
                if overridden.contains(sub.from.node.as_str()) {
                    continue;
                }
                combined.push(sub.clone());
            }

            combined
        }

        match expr {
            // Literals are constant level
            Expr::Bool(_) | Expr::Int(_) | Expr::String(_) => ExprLevel::Constant,

            // For identifiers, check if they reference an operator with action-level body
            Expr::Ident(name) => {
                // Prevent infinite recursion for cyclic operator definitions
                if visited.contains(name) {
                    return ExprLevel::State;
                }
                // Try to look up the operator definition
                if let Some(op) = ctx.get_op(name) {
                    visited.insert(name.clone());
                    let body = match ctx.instance_substitutions() {
                        Some(subs) => apply_substitutions(&op.body, subs),
                        None => op.body.clone(),
                    };
                    let level = self.get_level_with_ctx_inner(ctx, &body.node, visited);
                    visited.remove(name);
                    level
                } else {
                    // Unknown identifier - treat as state variable
                    ExprLevel::State
                }
            }

            // Prime makes it action level
            Expr::Prime(_) => ExprLevel::Action,

            // Temporal operators are temporal level
            Expr::Always(_)
            | Expr::Eventually(_)
            | Expr::LeadsTo(_, _)
            | Expr::WeakFair(_, _)
            | Expr::StrongFair(_, _) => ExprLevel::Temporal,

            // ENABLED is state level (checks if action is enabled in current state)
            // The inner expression might be action-level, but ENABLED itself
            // produces a state-level result
            Expr::Enabled(_) => ExprLevel::State,

            // UNCHANGED is action level (compares x and x')
            Expr::Unchanged(_) => ExprLevel::Action,

            // Binary operators: max of operand levels
            Expr::And(a, b)
            | Expr::Or(a, b)
            | Expr::Implies(a, b)
            | Expr::Equiv(a, b)
            | Expr::Eq(a, b)
            | Expr::Neq(a, b)
            | Expr::Lt(a, b)
            | Expr::Leq(a, b)
            | Expr::Gt(a, b)
            | Expr::Geq(a, b)
            | Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::IntDiv(a, b)
            | Expr::Mod(a, b)
            | Expr::Pow(a, b)
            | Expr::In(a, b)
            | Expr::NotIn(a, b)
            | Expr::Subseteq(a, b)
            | Expr::Union(a, b)
            | Expr::Intersect(a, b)
            | Expr::SetMinus(a, b)
            | Expr::FuncApply(a, b)
            | Expr::FuncSet(a, b)
            | Expr::Range(a, b) => self
                .get_level_with_ctx_inner(ctx, &a.node, visited)
                .max(self.get_level_with_ctx_inner(ctx, &b.node, visited)),

            // Unary operators: inherit from operand
            Expr::Not(e)
            | Expr::Neg(e)
            | Expr::Powerset(e)
            | Expr::BigUnion(e)
            | Expr::Domain(e) => self.get_level_with_ctx_inner(ctx, &e.node, visited),

            // IF-THEN-ELSE: max of all three
            Expr::If(cond, then_e, else_e) => self
                .get_level_with_ctx_inner(ctx, &cond.node, visited)
                .max(self.get_level_with_ctx_inner(ctx, &then_e.node, visited))
                .max(self.get_level_with_ctx_inner(ctx, &else_e.node, visited)),

            // LET-IN: max of definitions and body
            Expr::Let(defs, body) => {
                let mut level = self.get_level_with_ctx_inner(ctx, &body.node, visited);
                for def in defs {
                    level = level.max(self.get_level_with_ctx_inner(ctx, &def.body.node, visited));
                }
                level
            }

            // Apply: check for WF_xxx/SF_xxx (parsed as function calls), otherwise max of operator and arguments
            Expr::Apply(op, args) => {
                // Check if this is WF_xxx or SF_xxx (lexer matches as single identifier)
                if let Expr::Ident(name) = &op.node {
                    if name.starts_with("WF_") || name.starts_with("SF_") {
                        return ExprLevel::Temporal;
                    }
                }
                let mut level = self.get_level_with_ctx_inner(ctx, &op.node, visited);
                for arg in args {
                    level = level.max(self.get_level_with_ctx_inner(ctx, &arg.node, visited));
                }
                level
            }

            // OpRef is constant (just a reference to an operator)
            Expr::OpRef(_) => ExprLevel::Constant,

            // Module reference: resolve the target context and look up the operator definition.
            Expr::ModuleRef(target, op_name, args) => {
                // Start with args level
                let mut level = ExprLevel::Constant;
                for arg in args {
                    level = level.max(self.get_level_with_ctx_inner(ctx, &arg.node, visited));
                }

                // Prevent infinite recursion
                let combined_name = format!("{}!{}", target, op_name);
                if visited.contains(&combined_name) {
                    return level;
                }

                fn resolve_module_target_ctx(
                    ctx: &EvalCtx,
                    target: &ModuleTarget,
                ) -> Option<EvalCtx> {
                    match target {
                        ModuleTarget::Named(name) => {
                            let (module_name, instance_subs) = resolve_instance_info(ctx, name)?;
                            let effective_subs = compose_instance_substitutions(
                                &instance_subs,
                                ctx.instance_substitutions(),
                            );
                            let instance_ops = ctx
                                .instance_ops()
                                .get(&module_name)
                                .cloned()
                                .unwrap_or_default();
                            Some(
                                ctx.with_local_ops(instance_ops)
                                    .with_instance_substitutions(effective_subs),
                            )
                        }
                        ModuleTarget::Parameterized(name, params) => {
                            let def = ctx.get_op(name)?;
                            let Expr::InstanceExpr(module_name, instance_subs) = &def.body.node
                            else {
                                return None;
                            };
                            if def.params.len() != params.len() {
                                return None;
                            }

                            // Bind instance operator parameters into the WITH substitutions.
                            let param_subs: Vec<Substitution> = def
                                .params
                                .iter()
                                .zip(params.iter())
                                .map(|(p, arg)| Substitution {
                                    from: p.name.clone(),
                                    to: arg.clone(),
                                })
                                .collect();

                            let instantiated_subs: Vec<Substitution> = instance_subs
                                .iter()
                                .map(|s| Substitution {
                                    from: s.from.clone(),
                                    to: apply_substitutions(&s.to, &param_subs),
                                })
                                .collect();

                            let effective_subs = compose_instance_substitutions(
                                &instantiated_subs,
                                ctx.instance_substitutions(),
                            );
                            let instance_ops = ctx
                                .instance_ops()
                                .get(module_name)
                                .cloned()
                                .unwrap_or_default();
                            Some(
                                ctx.with_local_ops(instance_ops)
                                    .with_instance_substitutions(effective_subs),
                            )
                        }
                        ModuleTarget::Chained(base_expr) => {
                            resolve_chained_target_ctx(ctx, base_expr)
                        }
                    }
                }

                fn resolve_chained_target_ctx(
                    ctx: &EvalCtx,
                    base_expr: &Spanned<Expr>,
                ) -> Option<EvalCtx> {
                    let Expr::ModuleRef(base_target, inst_name, inst_args) = &base_expr.node else {
                        return None;
                    };
                    if !inst_args.is_empty() {
                        return None;
                    }

                    let base_ctx = resolve_module_target_ctx(ctx, base_target)?;
                    let inst_def = base_ctx.get_op(inst_name)?;
                    let Expr::InstanceExpr(module_name, inst_subs) = &inst_def.body.node else {
                        return None;
                    };

                    let effective_subs = compose_instance_substitutions(
                        inst_subs,
                        base_ctx.instance_substitutions(),
                    );
                    let instance_ops = ctx
                        .instance_ops()
                        .get(module_name)
                        .cloned()
                        .unwrap_or_default();

                    Some(
                        base_ctx
                            .with_local_ops(instance_ops)
                            .with_instance_substitutions(effective_subs),
                    )
                }

                if let Some(instance_ctx) = resolve_module_target_ctx(ctx, target) {
                    if let Some(op_def) = instance_ctx.get_op(op_name) {
                        let substituted_body = match instance_ctx.instance_substitutions() {
                            Some(subs) => apply_substitutions(&op_def.body, subs),
                            None => op_def.body.clone(),
                        };

                        visited.insert(combined_name.clone());
                        let op_level = self.get_level_with_ctx_inner(
                            &instance_ctx,
                            &substituted_body.node,
                            visited,
                        );
                        visited.remove(&combined_name);
                        level = level.max(op_level);
                    }
                }

                level
            }

            // Lambda is constant (it's a value)
            Expr::Lambda(_, body) => self.get_level_with_ctx_inner(ctx, &body.node, visited),

            // Instance expression
            Expr::InstanceExpr(_, subs) => {
                let mut level = ExprLevel::Constant;
                for sub in subs {
                    level = level.max(self.get_level_with_ctx_inner(ctx, &sub.to.node, visited));
                }
                level
            }

            // Quantifiers: max of domain and body
            Expr::Forall(bounds, body) | Expr::Exists(bounds, body) => {
                let mut level = self.get_level_with_ctx_inner(ctx, &body.node, visited);
                for bound in bounds {
                    if let Some(domain) = &bound.domain {
                        level =
                            level.max(self.get_level_with_ctx_inner(ctx, &domain.node, visited));
                    }
                }
                level
            }

            // CHOOSE: max of domain and predicate
            Expr::Choose(bound, pred) => {
                let mut level = self.get_level_with_ctx_inner(ctx, &pred.node, visited);
                if let Some(domain) = &bound.domain {
                    level = level.max(self.get_level_with_ctx_inner(ctx, &domain.node, visited));
                }
                level
            }

            // Set enumeration: max of elements
            Expr::SetEnum(elems) => {
                let mut level = ExprLevel::Constant;
                for elem in elems {
                    level = level.max(self.get_level_with_ctx_inner(ctx, &elem.node, visited));
                }
                level
            }

            // Set builder: max of expression and bounds
            Expr::SetBuilder(expr, bounds) => {
                let mut level = self.get_level_with_ctx_inner(ctx, &expr.node, visited);
                for bound in bounds {
                    if let Some(domain) = &bound.domain {
                        level =
                            level.max(self.get_level_with_ctx_inner(ctx, &domain.node, visited));
                    }
                }
                level
            }

            // Set filter: max of domain and predicate
            Expr::SetFilter(bound, pred) => {
                let mut level = self.get_level_with_ctx_inner(ctx, &pred.node, visited);
                if let Some(domain) = &bound.domain {
                    level = level.max(self.get_level_with_ctx_inner(ctx, &domain.node, visited));
                }
                level
            }

            // Function definition: max of bounds and body
            Expr::FuncDef(bounds, body) => {
                let mut level = self.get_level_with_ctx_inner(ctx, &body.node, visited);
                for bound in bounds {
                    if let Some(domain) = &bound.domain {
                        level =
                            level.max(self.get_level_with_ctx_inner(ctx, &domain.node, visited));
                    }
                }
                level
            }

            // EXCEPT: max of base and specs
            Expr::Except(base, specs) => {
                let mut level = self.get_level_with_ctx_inner(ctx, &base.node, visited);
                for spec in specs {
                    level =
                        level.max(self.get_level_with_ctx_inner(ctx, &spec.value.node, visited));
                    for path_elem in &spec.path {
                        if let tla_core::ast::ExceptPathElement::Index(idx) = path_elem {
                            level =
                                level.max(self.get_level_with_ctx_inner(ctx, &idx.node, visited));
                        }
                    }
                }
                level
            }

            // Record: max of field values
            Expr::Record(fields) => {
                let mut level = ExprLevel::Constant;
                for (_, value) in fields {
                    level = level.max(self.get_level_with_ctx_inner(ctx, &value.node, visited));
                }
                level
            }

            // Record access: level of record
            Expr::RecordAccess(record, _) => {
                self.get_level_with_ctx_inner(ctx, &record.node, visited)
            }

            // Record set: max of field domains
            Expr::RecordSet(fields) => {
                let mut level = ExprLevel::Constant;
                for (_, domain) in fields {
                    level = level.max(self.get_level_with_ctx_inner(ctx, &domain.node, visited));
                }
                level
            }

            // Tuple: max of elements
            Expr::Tuple(elems) => {
                let mut level = ExprLevel::Constant;
                for elem in elems {
                    level = level.max(self.get_level_with_ctx_inner(ctx, &elem.node, visited));
                }
                level
            }

            // Cartesian product: max of sets
            Expr::Times(sets) => {
                let mut level = ExprLevel::Constant;
                for set in sets {
                    level = level.max(self.get_level_with_ctx_inner(ctx, &set.node, visited));
                }
                level
            }

            // CASE: max of all guards and bodies
            Expr::Case(arms, other) => {
                let mut level = ExprLevel::Constant;
                for arm in arms {
                    level = level.max(self.get_level_with_ctx_inner(ctx, &arm.guard.node, visited));
                    level = level.max(self.get_level_with_ctx_inner(ctx, &arm.body.node, visited));
                }
                if let Some(other_expr) = other {
                    level =
                        level.max(self.get_level_with_ctx_inner(ctx, &other_expr.node, visited));
                }
                level
            }
        }
    }

    /// Determine the level of an expression (syntactic only, no operator resolution)
    ///
    /// Returns:
    /// - Constant: No variables, can be evaluated statically
    /// - State: Depends on current state only (no primes, no temporal)
    /// - Action: Depends on current and next state (has primes)
    /// - Temporal: Contains temporal operators ([], <>, WF, SF, ~>)
    #[allow(clippy::only_used_in_recursion)]
    pub fn get_level(&self, expr: &Expr) -> ExprLevel {
        match expr {
            // Literals are constant level
            Expr::Bool(_) | Expr::Int(_) | Expr::String(_) => ExprLevel::Constant,

            // Identifiers are state level (depend on current state)
            // Note: This doesn't resolve operators - use get_level_with_ctx for that
            Expr::Ident(_) => ExprLevel::State,

            // Prime makes it action level
            Expr::Prime(_) => ExprLevel::Action,

            // Temporal operators are temporal level
            Expr::Always(_)
            | Expr::Eventually(_)
            | Expr::LeadsTo(_, _)
            | Expr::WeakFair(_, _)
            | Expr::StrongFair(_, _) => ExprLevel::Temporal,

            // ENABLED is state level (checks if action is enabled in current state)
            // The inner expression might be action-level, but ENABLED itself
            // produces a state-level result
            Expr::Enabled(_) => ExprLevel::State,

            // UNCHANGED is action level (compares x and x')
            Expr::Unchanged(_) => ExprLevel::Action,

            // Binary operators: max of operand levels
            Expr::And(a, b)
            | Expr::Or(a, b)
            | Expr::Implies(a, b)
            | Expr::Equiv(a, b)
            | Expr::Eq(a, b)
            | Expr::Neq(a, b)
            | Expr::Lt(a, b)
            | Expr::Leq(a, b)
            | Expr::Gt(a, b)
            | Expr::Geq(a, b)
            | Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::IntDiv(a, b)
            | Expr::Mod(a, b)
            | Expr::Pow(a, b)
            | Expr::In(a, b)
            | Expr::NotIn(a, b)
            | Expr::Subseteq(a, b)
            | Expr::Union(a, b)
            | Expr::Intersect(a, b)
            | Expr::SetMinus(a, b)
            | Expr::FuncApply(a, b)
            | Expr::FuncSet(a, b)
            | Expr::Range(a, b) => self.get_level(&a.node).max(self.get_level(&b.node)),

            // Unary operators: inherit from operand
            Expr::Not(e)
            | Expr::Neg(e)
            | Expr::Powerset(e)
            | Expr::BigUnion(e)
            | Expr::Domain(e) => self.get_level(&e.node),

            // IF-THEN-ELSE: max of all three
            Expr::If(cond, then_e, else_e) => self
                .get_level(&cond.node)
                .max(self.get_level(&then_e.node))
                .max(self.get_level(&else_e.node)),

            // LET-IN: max of definitions and body
            Expr::Let(defs, body) => {
                let mut level = self.get_level(&body.node);
                for def in defs {
                    level = level.max(self.get_level(&def.body.node));
                }
                level
            }

            // Apply: check for WF_xxx/SF_xxx (parsed as function calls), otherwise max of operator and arguments
            Expr::Apply(op, args) => {
                // Check if this is WF_xxx or SF_xxx (lexer matches as single identifier)
                if let Expr::Ident(name) = &op.node {
                    if name.starts_with("WF_") || name.starts_with("SF_") {
                        return ExprLevel::Temporal;
                    }
                }
                let mut level = self.get_level(&op.node);
                for arg in args {
                    level = level.max(self.get_level(&arg.node));
                }
                level
            }

            // OpRef is constant (just a reference to an operator)
            Expr::OpRef(_) => ExprLevel::Constant,

            // Module reference: max of arguments
            Expr::ModuleRef(_, _, args) => {
                let mut level = ExprLevel::Constant;
                for arg in args {
                    level = level.max(self.get_level(&arg.node));
                }
                level
            }

            // Lambda is constant (it's a value)
            Expr::Lambda(_, body) => self.get_level(&body.node),

            // Instance expression
            Expr::InstanceExpr(_, subs) => {
                let mut level = ExprLevel::Constant;
                for sub in subs {
                    level = level.max(self.get_level(&sub.to.node));
                }
                level
            }

            // Quantifiers: max of domain and body
            Expr::Forall(bounds, body) | Expr::Exists(bounds, body) => {
                let mut level = self.get_level(&body.node);
                for bound in bounds {
                    if let Some(domain) = &bound.domain {
                        level = level.max(self.get_level(&domain.node));
                    }
                }
                level
            }

            // CHOOSE: max of domain and predicate
            Expr::Choose(bound, pred) => {
                let mut level = self.get_level(&pred.node);
                if let Some(domain) = &bound.domain {
                    level = level.max(self.get_level(&domain.node));
                }
                level
            }

            // Set enumeration: max of elements
            Expr::SetEnum(elems) => {
                let mut level = ExprLevel::Constant;
                for elem in elems {
                    level = level.max(self.get_level(&elem.node));
                }
                level
            }

            // Set builder: max of expression and bounds
            Expr::SetBuilder(expr, bounds) => {
                let mut level = self.get_level(&expr.node);
                for bound in bounds {
                    if let Some(domain) = &bound.domain {
                        level = level.max(self.get_level(&domain.node));
                    }
                }
                level
            }

            // Set filter: max of domain and predicate
            Expr::SetFilter(bound, pred) => {
                let mut level = self.get_level(&pred.node);
                if let Some(domain) = &bound.domain {
                    level = level.max(self.get_level(&domain.node));
                }
                level
            }

            // Function definition: max of bounds and body
            Expr::FuncDef(bounds, body) => {
                let mut level = self.get_level(&body.node);
                for bound in bounds {
                    if let Some(domain) = &bound.domain {
                        level = level.max(self.get_level(&domain.node));
                    }
                }
                level
            }

            // EXCEPT: max of base and specs
            Expr::Except(base, specs) => {
                let mut level = self.get_level(&base.node);
                for spec in specs {
                    level = level.max(self.get_level(&spec.value.node));
                    for path_elem in &spec.path {
                        if let tla_core::ast::ExceptPathElement::Index(idx) = path_elem {
                            level = level.max(self.get_level(&idx.node));
                        }
                    }
                }
                level
            }

            // Record: max of field values
            Expr::Record(fields) => {
                let mut level = ExprLevel::Constant;
                for (_, value) in fields {
                    level = level.max(self.get_level(&value.node));
                }
                level
            }

            // Record access: level of record
            Expr::RecordAccess(record, _) => self.get_level(&record.node),

            // Record set: max of field domains
            Expr::RecordSet(fields) => {
                let mut level = ExprLevel::Constant;
                for (_, domain) in fields {
                    level = level.max(self.get_level(&domain.node));
                }
                level
            }

            // Tuple: max of elements
            Expr::Tuple(elems) => {
                let mut level = ExprLevel::Constant;
                for elem in elems {
                    level = level.max(self.get_level(&elem.node));
                }
                level
            }

            // Cartesian product: max of sets
            Expr::Times(sets) => {
                let mut level = ExprLevel::Constant;
                for set in sets {
                    level = level.max(self.get_level(&set.node));
                }
                level
            }

            // CASE: max of all guards and bodies
            Expr::Case(arms, other) => {
                let mut level = ExprLevel::Constant;
                for arm in arms {
                    level = level.max(self.get_level(&arm.guard.node));
                    level = level.max(self.get_level(&arm.body.node));
                }
                if let Some(other_expr) = other {
                    level = level.max(self.get_level(&other_expr.node));
                }
                level
            }
        }
    }

    /// Check if an expression contains primed variables
    pub fn contains_prime(&self, expr: &Expr) -> bool {
        self.get_level(expr) >= ExprLevel::Action
    }

    /// Check if an expression contains temporal operators
    pub fn contains_temporal(&self, expr: &Expr) -> bool {
        self.get_level(expr) >= ExprLevel::Temporal
    }

    /// Substitute parameter names with argument expressions in an expression.
    ///
    /// This is used for inlining parameterized operator definitions.
    /// It replaces each occurrence of a parameter name (as an identifier) with
    /// the corresponding argument expression.
    pub fn substitute_params_in_expr(
        &self,
        body: &Expr,
        params: &[tla_core::ast::OpParam],
        args: &[Spanned<Expr>],
    ) -> Expr {
        // Build a map from parameter name to argument expression
        let subs: HashMap<String, &Spanned<Expr>> = params
            .iter()
            .zip(args.iter())
            .map(|(p, a)| (p.name.node.clone(), a))
            .collect();
        self.substitute_params_in_expr_impl(body, &subs)
    }

    fn substitute_params_in_expr_spanned(
        &self,
        expr: &Spanned<Expr>,
        subs: &HashMap<String, &Spanned<Expr>>,
    ) -> Spanned<Expr> {
        Spanned {
            node: self.substitute_params_in_expr_impl(&expr.node, subs),
            span: expr.span,
        }
    }

    fn substitute_params_in_expr_impl(
        &self,
        expr: &Expr,
        subs: &HashMap<String, &Spanned<Expr>>,
    ) -> Expr {
        match expr {
            // Identifier - check if it should be substituted
            Expr::Ident(name) => {
                if let Some(arg) = subs.get(name) {
                    arg.node.clone()
                } else {
                    Expr::Ident(name.clone())
                }
            }

            // Literals - no substitution needed
            Expr::Bool(b) => Expr::Bool(*b),
            Expr::Int(n) => Expr::Int(n.clone()),
            Expr::String(s) => Expr::String(s.clone()),

            // Binary operators
            Expr::And(a, b) => Expr::And(
                Box::new(self.substitute_params_in_expr_spanned(a, subs)),
                Box::new(self.substitute_params_in_expr_spanned(b, subs)),
            ),
            Expr::Or(a, b) => Expr::Or(
                Box::new(self.substitute_params_in_expr_spanned(a, subs)),
                Box::new(self.substitute_params_in_expr_spanned(b, subs)),
            ),
            Expr::Implies(a, b) => Expr::Implies(
                Box::new(self.substitute_params_in_expr_spanned(a, subs)),
                Box::new(self.substitute_params_in_expr_spanned(b, subs)),
            ),
            Expr::Equiv(a, b) => Expr::Equiv(
                Box::new(self.substitute_params_in_expr_spanned(a, subs)),
                Box::new(self.substitute_params_in_expr_spanned(b, subs)),
            ),
            Expr::Eq(a, b) => Expr::Eq(
                Box::new(self.substitute_params_in_expr_spanned(a, subs)),
                Box::new(self.substitute_params_in_expr_spanned(b, subs)),
            ),
            Expr::Neq(a, b) => Expr::Neq(
                Box::new(self.substitute_params_in_expr_spanned(a, subs)),
                Box::new(self.substitute_params_in_expr_spanned(b, subs)),
            ),
            Expr::Lt(a, b) => Expr::Lt(
                Box::new(self.substitute_params_in_expr_spanned(a, subs)),
                Box::new(self.substitute_params_in_expr_spanned(b, subs)),
            ),
            Expr::Leq(a, b) => Expr::Leq(
                Box::new(self.substitute_params_in_expr_spanned(a, subs)),
                Box::new(self.substitute_params_in_expr_spanned(b, subs)),
            ),
            Expr::Gt(a, b) => Expr::Gt(
                Box::new(self.substitute_params_in_expr_spanned(a, subs)),
                Box::new(self.substitute_params_in_expr_spanned(b, subs)),
            ),
            Expr::Geq(a, b) => Expr::Geq(
                Box::new(self.substitute_params_in_expr_spanned(a, subs)),
                Box::new(self.substitute_params_in_expr_spanned(b, subs)),
            ),
            Expr::Add(a, b) => Expr::Add(
                Box::new(self.substitute_params_in_expr_spanned(a, subs)),
                Box::new(self.substitute_params_in_expr_spanned(b, subs)),
            ),
            Expr::Sub(a, b) => Expr::Sub(
                Box::new(self.substitute_params_in_expr_spanned(a, subs)),
                Box::new(self.substitute_params_in_expr_spanned(b, subs)),
            ),
            Expr::Mul(a, b) => Expr::Mul(
                Box::new(self.substitute_params_in_expr_spanned(a, subs)),
                Box::new(self.substitute_params_in_expr_spanned(b, subs)),
            ),
            Expr::Div(a, b) => Expr::Div(
                Box::new(self.substitute_params_in_expr_spanned(a, subs)),
                Box::new(self.substitute_params_in_expr_spanned(b, subs)),
            ),
            Expr::IntDiv(a, b) => Expr::IntDiv(
                Box::new(self.substitute_params_in_expr_spanned(a, subs)),
                Box::new(self.substitute_params_in_expr_spanned(b, subs)),
            ),
            Expr::Mod(a, b) => Expr::Mod(
                Box::new(self.substitute_params_in_expr_spanned(a, subs)),
                Box::new(self.substitute_params_in_expr_spanned(b, subs)),
            ),
            Expr::Pow(a, b) => Expr::Pow(
                Box::new(self.substitute_params_in_expr_spanned(a, subs)),
                Box::new(self.substitute_params_in_expr_spanned(b, subs)),
            ),
            Expr::In(a, b) => Expr::In(
                Box::new(self.substitute_params_in_expr_spanned(a, subs)),
                Box::new(self.substitute_params_in_expr_spanned(b, subs)),
            ),
            Expr::NotIn(a, b) => Expr::NotIn(
                Box::new(self.substitute_params_in_expr_spanned(a, subs)),
                Box::new(self.substitute_params_in_expr_spanned(b, subs)),
            ),
            Expr::Subseteq(a, b) => Expr::Subseteq(
                Box::new(self.substitute_params_in_expr_spanned(a, subs)),
                Box::new(self.substitute_params_in_expr_spanned(b, subs)),
            ),
            Expr::Union(a, b) => Expr::Union(
                Box::new(self.substitute_params_in_expr_spanned(a, subs)),
                Box::new(self.substitute_params_in_expr_spanned(b, subs)),
            ),
            Expr::Intersect(a, b) => Expr::Intersect(
                Box::new(self.substitute_params_in_expr_spanned(a, subs)),
                Box::new(self.substitute_params_in_expr_spanned(b, subs)),
            ),
            Expr::SetMinus(a, b) => Expr::SetMinus(
                Box::new(self.substitute_params_in_expr_spanned(a, subs)),
                Box::new(self.substitute_params_in_expr_spanned(b, subs)),
            ),
            Expr::FuncApply(a, b) => Expr::FuncApply(
                Box::new(self.substitute_params_in_expr_spanned(a, subs)),
                Box::new(self.substitute_params_in_expr_spanned(b, subs)),
            ),
            Expr::FuncSet(a, b) => Expr::FuncSet(
                Box::new(self.substitute_params_in_expr_spanned(a, subs)),
                Box::new(self.substitute_params_in_expr_spanned(b, subs)),
            ),
            Expr::Range(a, b) => Expr::Range(
                Box::new(self.substitute_params_in_expr_spanned(a, subs)),
                Box::new(self.substitute_params_in_expr_spanned(b, subs)),
            ),

            // Unary operators
            Expr::Not(e) => Expr::Not(Box::new(self.substitute_params_in_expr_spanned(e, subs))),
            Expr::Neg(e) => Expr::Neg(Box::new(self.substitute_params_in_expr_spanned(e, subs))),
            Expr::Powerset(e) => {
                Expr::Powerset(Box::new(self.substitute_params_in_expr_spanned(e, subs)))
            }
            Expr::BigUnion(e) => {
                Expr::BigUnion(Box::new(self.substitute_params_in_expr_spanned(e, subs)))
            }
            Expr::Domain(e) => {
                Expr::Domain(Box::new(self.substitute_params_in_expr_spanned(e, subs)))
            }
            Expr::Prime(e) => Expr::Prime(Box::new(self.substitute_params_in_expr_spanned(e, subs))),
            Expr::Enabled(e) => {
                Expr::Enabled(Box::new(self.substitute_params_in_expr_spanned(e, subs)))
            }
            Expr::Unchanged(e) => {
                Expr::Unchanged(Box::new(self.substitute_params_in_expr_spanned(e, subs)))
            }

            // Temporal operators
            Expr::Always(e) => {
                Expr::Always(Box::new(self.substitute_params_in_expr_spanned(e, subs)))
            }
            Expr::Eventually(e) => {
                Expr::Eventually(Box::new(self.substitute_params_in_expr_spanned(e, subs)))
            }
            Expr::LeadsTo(a, b) => Expr::LeadsTo(
                Box::new(self.substitute_params_in_expr_spanned(a, subs)),
                Box::new(self.substitute_params_in_expr_spanned(b, subs)),
            ),
            Expr::WeakFair(sub, act) => Expr::WeakFair(
                Box::new(self.substitute_params_in_expr_spanned(sub, subs)),
                Box::new(self.substitute_params_in_expr_spanned(act, subs)),
            ),
            Expr::StrongFair(sub, act) => Expr::StrongFair(
                Box::new(self.substitute_params_in_expr_spanned(sub, subs)),
                Box::new(self.substitute_params_in_expr_spanned(act, subs)),
            ),

            // IF-THEN-ELSE
            Expr::If(cond, then_e, else_e) => Expr::If(
                Box::new(self.substitute_params_in_expr_spanned(cond, subs)),
                Box::new(self.substitute_params_in_expr_spanned(then_e, subs)),
                Box::new(self.substitute_params_in_expr_spanned(else_e, subs)),
            ),

            // LET-IN: be careful with shadowing
            Expr::Let(defs, body) => {
                // Collect names defined in LET
                let shadowed: std::collections::HashSet<_> =
                    defs.iter().map(|d| d.name.node.clone()).collect();
                let filtered_subs: HashMap<_, _> = subs
                    .iter()
                    .filter(|(k, _)| !shadowed.contains(*k))
                    .map(|(k, v)| (k.clone(), *v))
                    .collect();
                let new_defs: Vec<_> = defs
                    .iter()
                    .map(|d| tla_core::ast::OperatorDef {
                        name: d.name.clone(),
                        params: d.params.clone(),
                        body: self.substitute_params_in_expr_spanned(&d.body, subs),
                        local: d.local,
                    })
                    .collect();
                Expr::Let(
                    new_defs,
                    Box::new(self.substitute_params_in_expr_spanned(body, &filtered_subs)),
                )
            }

            // Apply
            Expr::Apply(op, args) => {
                let new_args: Vec<_> = args
                    .iter()
                    .map(|a| self.substitute_params_in_expr_spanned(a, subs))
                    .collect();
                Expr::Apply(
                    Box::new(self.substitute_params_in_expr_spanned(op, subs)),
                    new_args,
                )
            }

            // OpRef - no substitution needed
            Expr::OpRef(name) => Expr::OpRef(name.clone()),

            // ModuleRef
            Expr::ModuleRef(module, op, args) => {
                let new_args: Vec<_> = args
                    .iter()
                    .map(|a| self.substitute_params_in_expr_spanned(a, subs))
                    .collect();
                Expr::ModuleRef(module.clone(), op.clone(), new_args)
            }

            // Lambda: be careful with shadowing
            Expr::Lambda(params, body) => {
                let shadowed: std::collections::HashSet<_> =
                    params.iter().map(|p| p.node.clone()).collect();
                let filtered_subs: HashMap<_, _> = subs
                    .iter()
                    .filter(|(k, _)| !shadowed.contains(*k))
                    .map(|(k, v)| (k.clone(), *v))
                    .collect();
                Expr::Lambda(
                    params.clone(),
                    Box::new(self.substitute_params_in_expr_spanned(body, &filtered_subs)),
                )
            }

            // InstanceExpr
            Expr::InstanceExpr(name, substs) => {
                let new_substs: Vec<_> = substs
                    .iter()
                    .map(|s| tla_core::ast::Substitution {
                        from: s.from.clone(),
                        to: self.substitute_params_in_expr_spanned(&s.to, subs),
                    })
                    .collect();
                Expr::InstanceExpr(name.clone(), new_substs)
            }

            // Quantifiers: be careful with shadowing
            Expr::Forall(bounds, body) => {
                let (new_bounds, filtered_subs) = self.substitute_params_in_bounds(bounds, subs);
                Expr::Forall(
                    new_bounds,
                    Box::new(self.substitute_params_in_expr_spanned(body, &filtered_subs)),
                )
            }
            Expr::Exists(bounds, body) => {
                let (new_bounds, filtered_subs) = self.substitute_params_in_bounds(bounds, subs);
                Expr::Exists(
                    new_bounds,
                    Box::new(self.substitute_params_in_expr_spanned(body, &filtered_subs)),
                )
            }

            // CHOOSE: be careful with shadowing
            Expr::Choose(bound, body) => {
                let (new_bounds, filtered_subs) =
                    self.substitute_params_in_bounds(std::slice::from_ref(bound), subs);
                Expr::Choose(
                    new_bounds.into_iter().next().unwrap(),
                    Box::new(self.substitute_params_in_expr_spanned(body, &filtered_subs)),
                )
            }

            // Set enumeration
            Expr::SetEnum(elems) => {
                let new_elems: Vec<_> = elems
                    .iter()
                    .map(|e| self.substitute_params_in_expr_spanned(e, subs))
                    .collect();
                Expr::SetEnum(new_elems)
            }

            // Set builder: be careful with shadowing
            Expr::SetBuilder(expr, bounds) => {
                let (new_bounds, filtered_subs) = self.substitute_params_in_bounds(bounds, subs);
                Expr::SetBuilder(
                    Box::new(self.substitute_params_in_expr_spanned(expr, &filtered_subs)),
                    new_bounds,
                )
            }

            // Set filter: be careful with shadowing
            Expr::SetFilter(bound, pred) => {
                let (new_bounds, filtered_subs) =
                    self.substitute_params_in_bounds(std::slice::from_ref(bound), subs);
                Expr::SetFilter(
                    new_bounds.into_iter().next().unwrap(),
                    Box::new(self.substitute_params_in_expr_spanned(pred, &filtered_subs)),
                )
            }

            // Function definition: be careful with shadowing
            Expr::FuncDef(bounds, body) => {
                let (new_bounds, filtered_subs) = self.substitute_params_in_bounds(bounds, subs);
                Expr::FuncDef(
                    new_bounds,
                    Box::new(self.substitute_params_in_expr_spanned(body, &filtered_subs)),
                )
            }

            // EXCEPT
            Expr::Except(base, specs) => {
                let new_specs: Vec<_> = specs
                    .iter()
                    .map(|s| {
                        let new_path: Vec<_> = s
                            .path
                            .iter()
                            .map(|p| match p {
                                tla_core::ast::ExceptPathElement::Index(idx) => {
                                    tla_core::ast::ExceptPathElement::Index(
                                        self.substitute_params_in_expr_spanned(idx, subs),
                                    )
                                }
                                tla_core::ast::ExceptPathElement::Field(f) => {
                                    tla_core::ast::ExceptPathElement::Field(f.clone())
                                }
                            })
                            .collect();
                        tla_core::ast::ExceptSpec {
                            path: new_path,
                            value: self.substitute_params_in_expr_spanned(&s.value, subs),
                        }
                    })
                    .collect();
                Expr::Except(
                    Box::new(self.substitute_params_in_expr_spanned(base, subs)),
                    new_specs,
                )
            }

            // Record
            Expr::Record(fields) => {
                let new_fields: Vec<_> = fields
                    .iter()
                    .map(|(k, v)| (k.clone(), self.substitute_params_in_expr_spanned(v, subs)))
                    .collect();
                Expr::Record(new_fields)
            }

            // Record access
            Expr::RecordAccess(record, field) => Expr::RecordAccess(
                Box::new(self.substitute_params_in_expr_spanned(record, subs)),
                field.clone(),
            ),

            // Record set
            Expr::RecordSet(fields) => {
                let new_fields: Vec<_> = fields
                    .iter()
                    .map(|(k, v)| (k.clone(), self.substitute_params_in_expr_spanned(v, subs)))
                    .collect();
                Expr::RecordSet(new_fields)
            }

            // Tuple
            Expr::Tuple(elems) => {
                let new_elems: Vec<_> = elems
                    .iter()
                    .map(|e| self.substitute_params_in_expr_spanned(e, subs))
                    .collect();
                Expr::Tuple(new_elems)
            }

            // Cartesian product
            Expr::Times(sets) => {
                let new_sets: Vec<_> = sets
                    .iter()
                    .map(|s| self.substitute_params_in_expr_spanned(s, subs))
                    .collect();
                Expr::Times(new_sets)
            }

            // CASE
            Expr::Case(arms, other) => {
                let new_arms: Vec<_> = arms
                    .iter()
                    .map(|arm| tla_core::ast::CaseArm {
                        guard: self.substitute_params_in_expr_spanned(&arm.guard, subs),
                        body: self.substitute_params_in_expr_spanned(&arm.body, subs),
                    })
                    .collect();
                let new_other = other
                    .as_ref()
                    .map(|e| Box::new(self.substitute_params_in_expr_spanned(e, subs)));
                Expr::Case(new_arms, new_other)
            }
        }
    }

    fn substitute_params_in_bounds<'a>(
        &self,
        bounds: &[BoundVar],
        subs: &HashMap<String, &'a Spanned<Expr>>,
    ) -> (Vec<BoundVar>, HashMap<String, &'a Spanned<Expr>>) {
        use tla_core::ast::BoundPattern;

        // Collect all bound variable names
        let mut shadowed = std::collections::HashSet::new();
        for bound in bounds {
            shadowed.insert(bound.name.node.clone());
            if let Some(BoundPattern::Tuple(vars)) = &bound.pattern {
                for var in vars {
                    shadowed.insert(var.node.clone());
                }
            } else if let Some(BoundPattern::Var(var)) = &bound.pattern {
                shadowed.insert(var.node.clone());
            }
        }

        // Filter out shadowed substitutions
        let filtered_subs: HashMap<_, _> = subs
            .iter()
            .filter(|(k, _)| !shadowed.contains(*k))
            .map(|(k, v)| (k.clone(), *v))
            .collect();

        // Substitute in domain expressions
        let new_bounds: Vec<_> = bounds
            .iter()
            .map(|b| BoundVar {
                name: b.name.clone(),
                pattern: b.pattern.clone(),
                domain: b
                    .domain
                    .as_ref()
                    .map(|d| Box::new(self.substitute_params_in_expr_spanned(d, subs))),
            })
            .collect();

        (new_bounds, filtered_subs)
    }
}

impl Default for AstToLive {
    fn default() -> Self {
        Self::new()
    }
}

/// Error during AST to LiveExpr conversion
#[derive(Debug, Clone)]
pub enum ConvertError {
    /// Expression evaluated to a non-boolean constant
    NonBooleanConstant(Arc<Spanned<Expr>>),
    /// Unsupported temporal expression
    UnsupportedTemporal(Arc<Spanned<Expr>>),
}

impl std::fmt::Display for ConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConvertError::NonBooleanConstant(e) => {
                write!(
                    f,
                    "Expected boolean constant in liveness property: {:?}",
                    e.node
                )
            }
            ConvertError::UnsupportedTemporal(e) => {
                write!(
                    f,
                    "Unsupported temporal expression in liveness property: {:?}",
                    e.node
                )
            }
        }
    }
}

impl std::error::Error for ConvertError {}

/// Substitute values for identifiers in an expression
///
/// This is used to expand quantified temporal formulas by replacing bound
/// variables with their concrete values. This is necessary because LiveExpr
/// stores AST references that are evaluated later without the quantifier context.
fn substitute_values_in_expr(expr: &Spanned<Expr>, subs: &HashMap<String, Value>) -> Spanned<Expr> {
    let new_node = substitute_values_in_expr_node(&expr.node, subs);
    Spanned {
        node: new_node,
        span: expr.span,
    }
}

fn substitute_values_in_expr_node(expr: &Expr, subs: &HashMap<String, Value>) -> Expr {
    match expr {
        // Identifier - check if it should be substituted
        Expr::Ident(name) => {
            if let Some(value) = subs.get(name) {
                value_to_expr(value)
            } else {
                Expr::Ident(name.clone())
            }
        }

        // Literals - no substitution needed
        Expr::Bool(b) => Expr::Bool(*b),
        Expr::Int(n) => Expr::Int(n.clone()),
        Expr::String(s) => Expr::String(s.clone()),

        // Binary operators
        Expr::And(a, b) => Expr::And(
            Box::new(substitute_values_in_expr(a, subs)),
            Box::new(substitute_values_in_expr(b, subs)),
        ),
        Expr::Or(a, b) => Expr::Or(
            Box::new(substitute_values_in_expr(a, subs)),
            Box::new(substitute_values_in_expr(b, subs)),
        ),
        Expr::Implies(a, b) => Expr::Implies(
            Box::new(substitute_values_in_expr(a, subs)),
            Box::new(substitute_values_in_expr(b, subs)),
        ),
        Expr::Equiv(a, b) => Expr::Equiv(
            Box::new(substitute_values_in_expr(a, subs)),
            Box::new(substitute_values_in_expr(b, subs)),
        ),
        Expr::Eq(a, b) => Expr::Eq(
            Box::new(substitute_values_in_expr(a, subs)),
            Box::new(substitute_values_in_expr(b, subs)),
        ),
        Expr::Neq(a, b) => Expr::Neq(
            Box::new(substitute_values_in_expr(a, subs)),
            Box::new(substitute_values_in_expr(b, subs)),
        ),
        Expr::Lt(a, b) => Expr::Lt(
            Box::new(substitute_values_in_expr(a, subs)),
            Box::new(substitute_values_in_expr(b, subs)),
        ),
        Expr::Leq(a, b) => Expr::Leq(
            Box::new(substitute_values_in_expr(a, subs)),
            Box::new(substitute_values_in_expr(b, subs)),
        ),
        Expr::Gt(a, b) => Expr::Gt(
            Box::new(substitute_values_in_expr(a, subs)),
            Box::new(substitute_values_in_expr(b, subs)),
        ),
        Expr::Geq(a, b) => Expr::Geq(
            Box::new(substitute_values_in_expr(a, subs)),
            Box::new(substitute_values_in_expr(b, subs)),
        ),
        Expr::Add(a, b) => Expr::Add(
            Box::new(substitute_values_in_expr(a, subs)),
            Box::new(substitute_values_in_expr(b, subs)),
        ),
        Expr::Sub(a, b) => Expr::Sub(
            Box::new(substitute_values_in_expr(a, subs)),
            Box::new(substitute_values_in_expr(b, subs)),
        ),
        Expr::Mul(a, b) => Expr::Mul(
            Box::new(substitute_values_in_expr(a, subs)),
            Box::new(substitute_values_in_expr(b, subs)),
        ),
        Expr::Div(a, b) => Expr::Div(
            Box::new(substitute_values_in_expr(a, subs)),
            Box::new(substitute_values_in_expr(b, subs)),
        ),
        Expr::IntDiv(a, b) => Expr::IntDiv(
            Box::new(substitute_values_in_expr(a, subs)),
            Box::new(substitute_values_in_expr(b, subs)),
        ),
        Expr::Mod(a, b) => Expr::Mod(
            Box::new(substitute_values_in_expr(a, subs)),
            Box::new(substitute_values_in_expr(b, subs)),
        ),
        Expr::Pow(a, b) => Expr::Pow(
            Box::new(substitute_values_in_expr(a, subs)),
            Box::new(substitute_values_in_expr(b, subs)),
        ),
        Expr::In(a, b) => Expr::In(
            Box::new(substitute_values_in_expr(a, subs)),
            Box::new(substitute_values_in_expr(b, subs)),
        ),
        Expr::NotIn(a, b) => Expr::NotIn(
            Box::new(substitute_values_in_expr(a, subs)),
            Box::new(substitute_values_in_expr(b, subs)),
        ),
        Expr::Subseteq(a, b) => Expr::Subseteq(
            Box::new(substitute_values_in_expr(a, subs)),
            Box::new(substitute_values_in_expr(b, subs)),
        ),
        Expr::Union(a, b) => Expr::Union(
            Box::new(substitute_values_in_expr(a, subs)),
            Box::new(substitute_values_in_expr(b, subs)),
        ),
        Expr::Intersect(a, b) => Expr::Intersect(
            Box::new(substitute_values_in_expr(a, subs)),
            Box::new(substitute_values_in_expr(b, subs)),
        ),
        Expr::SetMinus(a, b) => Expr::SetMinus(
            Box::new(substitute_values_in_expr(a, subs)),
            Box::new(substitute_values_in_expr(b, subs)),
        ),
        Expr::FuncApply(a, b) => Expr::FuncApply(
            Box::new(substitute_values_in_expr(a, subs)),
            Box::new(substitute_values_in_expr(b, subs)),
        ),
        Expr::FuncSet(a, b) => Expr::FuncSet(
            Box::new(substitute_values_in_expr(a, subs)),
            Box::new(substitute_values_in_expr(b, subs)),
        ),
        Expr::Range(a, b) => Expr::Range(
            Box::new(substitute_values_in_expr(a, subs)),
            Box::new(substitute_values_in_expr(b, subs)),
        ),

        // Unary operators
        Expr::Not(e) => Expr::Not(Box::new(substitute_values_in_expr(e, subs))),
        Expr::Neg(e) => Expr::Neg(Box::new(substitute_values_in_expr(e, subs))),
        Expr::Powerset(e) => Expr::Powerset(Box::new(substitute_values_in_expr(e, subs))),
        Expr::BigUnion(e) => Expr::BigUnion(Box::new(substitute_values_in_expr(e, subs))),
        Expr::Domain(e) => Expr::Domain(Box::new(substitute_values_in_expr(e, subs))),
        Expr::Prime(e) => Expr::Prime(Box::new(substitute_values_in_expr(e, subs))),
        Expr::Enabled(e) => Expr::Enabled(Box::new(substitute_values_in_expr(e, subs))),
        Expr::Unchanged(e) => Expr::Unchanged(Box::new(substitute_values_in_expr(e, subs))),

        // Temporal operators
        Expr::Always(e) => Expr::Always(Box::new(substitute_values_in_expr(e, subs))),
        Expr::Eventually(e) => Expr::Eventually(Box::new(substitute_values_in_expr(e, subs))),
        Expr::LeadsTo(a, b) => Expr::LeadsTo(
            Box::new(substitute_values_in_expr(a, subs)),
            Box::new(substitute_values_in_expr(b, subs)),
        ),
        Expr::WeakFair(sub, act) => Expr::WeakFair(
            Box::new(substitute_values_in_expr(sub, subs)),
            Box::new(substitute_values_in_expr(act, subs)),
        ),
        Expr::StrongFair(sub, act) => Expr::StrongFair(
            Box::new(substitute_values_in_expr(sub, subs)),
            Box::new(substitute_values_in_expr(act, subs)),
        ),

        // IF-THEN-ELSE
        Expr::If(cond, then_e, else_e) => Expr::If(
            Box::new(substitute_values_in_expr(cond, subs)),
            Box::new(substitute_values_in_expr(then_e, subs)),
            Box::new(substitute_values_in_expr(else_e, subs)),
        ),

        // LET-IN: be careful with shadowing
        Expr::Let(defs, body) => {
            let new_defs: Vec<_> = defs
                .iter()
                .map(|d| tla_core::ast::OperatorDef {
                    name: d.name.clone(),
                    params: d.params.clone(),
                    body: substitute_values_in_expr(&d.body, subs),
                    local: d.local,
                })
                .collect();
            Expr::Let(new_defs, Box::new(substitute_values_in_expr(body, subs)))
        }

        // Apply
        Expr::Apply(op, args) => {
            let new_args: Vec<_> = args
                .iter()
                .map(|a| substitute_values_in_expr(a, subs))
                .collect();
            Expr::Apply(Box::new(substitute_values_in_expr(op, subs)), new_args)
        }

        // OpRef - no substitution needed
        Expr::OpRef(name) => Expr::OpRef(name.clone()),

        // ModuleRef
        Expr::ModuleRef(module, op, args) => {
            let new_args: Vec<_> = args
                .iter()
                .map(|a| substitute_values_in_expr(a, subs))
                .collect();
            Expr::ModuleRef(module.clone(), op.clone(), new_args)
        }

        // Lambda: be careful with shadowing
        Expr::Lambda(params, body) => {
            // Lambda params are Vec<Spanned<String>>
            let shadowed: std::collections::HashSet<_> =
                params.iter().map(|p| p.node.clone()).collect();
            let filtered_subs: HashMap<_, _> = subs
                .iter()
                .filter(|(k, _)| !shadowed.contains(*k))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            Expr::Lambda(
                params.clone(),
                Box::new(substitute_values_in_expr(body, &filtered_subs)),
            )
        }

        // InstanceExpr
        Expr::InstanceExpr(name, substs) => {
            let new_substs: Vec<_> = substs
                .iter()
                .map(|s| tla_core::ast::Substitution {
                    from: s.from.clone(),
                    to: substitute_values_in_expr(&s.to, subs),
                })
                .collect();
            Expr::InstanceExpr(name.clone(), new_substs)
        }

        // Quantifiers: be careful with shadowing
        Expr::Forall(bounds, body) => {
            let (new_bounds, filtered_subs) = substitute_in_bounds(bounds, subs);
            Expr::Forall(
                new_bounds,
                Box::new(substitute_values_in_expr(body, &filtered_subs)),
            )
        }
        Expr::Exists(bounds, body) => {
            let (new_bounds, filtered_subs) = substitute_in_bounds(bounds, subs);
            Expr::Exists(
                new_bounds,
                Box::new(substitute_values_in_expr(body, &filtered_subs)),
            )
        }

        // CHOOSE: be careful with shadowing
        Expr::Choose(bound, body) => {
            let (new_bounds, filtered_subs) =
                substitute_in_bounds(std::slice::from_ref(bound), subs);
            Expr::Choose(
                new_bounds.into_iter().next().unwrap(),
                Box::new(substitute_values_in_expr(body, &filtered_subs)),
            )
        }

        // Set enumeration
        Expr::SetEnum(elems) => {
            let new_elems: Vec<_> = elems
                .iter()
                .map(|e| substitute_values_in_expr(e, subs))
                .collect();
            Expr::SetEnum(new_elems)
        }

        // Set builder: be careful with shadowing
        Expr::SetBuilder(expr, bounds) => {
            let (new_bounds, filtered_subs) = substitute_in_bounds(bounds, subs);
            Expr::SetBuilder(
                Box::new(substitute_values_in_expr(expr, &filtered_subs)),
                new_bounds,
            )
        }

        // Set filter: be careful with shadowing
        Expr::SetFilter(bound, pred) => {
            let (new_bounds, filtered_subs) =
                substitute_in_bounds(std::slice::from_ref(bound), subs);
            Expr::SetFilter(
                new_bounds.into_iter().next().unwrap(),
                Box::new(substitute_values_in_expr(pred, &filtered_subs)),
            )
        }

        // Function definition: be careful with shadowing
        Expr::FuncDef(bounds, body) => {
            let (new_bounds, filtered_subs) = substitute_in_bounds(bounds, subs);
            Expr::FuncDef(
                new_bounds,
                Box::new(substitute_values_in_expr(body, &filtered_subs)),
            )
        }

        // EXCEPT
        Expr::Except(base, specs) => {
            let new_specs: Vec<_> = specs
                .iter()
                .map(|s| {
                    let new_path: Vec<_> = s
                        .path
                        .iter()
                        .map(|p| match p {
                            tla_core::ast::ExceptPathElement::Index(idx) => {
                                tla_core::ast::ExceptPathElement::Index(substitute_values_in_expr(
                                    idx, subs,
                                ))
                            }
                            tla_core::ast::ExceptPathElement::Field(f) => {
                                tla_core::ast::ExceptPathElement::Field(f.clone())
                            }
                        })
                        .collect();
                    tla_core::ast::ExceptSpec {
                        path: new_path,
                        value: substitute_values_in_expr(&s.value, subs),
                    }
                })
                .collect();
            Expr::Except(Box::new(substitute_values_in_expr(base, subs)), new_specs)
        }

        // Record
        Expr::Record(fields) => {
            let new_fields: Vec<_> = fields
                .iter()
                .map(|(k, v)| (k.clone(), substitute_values_in_expr(v, subs)))
                .collect();
            Expr::Record(new_fields)
        }

        // Record access
        Expr::RecordAccess(record, field) => Expr::RecordAccess(
            Box::new(substitute_values_in_expr(record, subs)),
            field.clone(),
        ),

        // Record set
        Expr::RecordSet(fields) => {
            let new_fields: Vec<_> = fields
                .iter()
                .map(|(k, v)| (k.clone(), substitute_values_in_expr(v, subs)))
                .collect();
            Expr::RecordSet(new_fields)
        }

        // Tuple
        Expr::Tuple(elems) => {
            let new_elems: Vec<_> = elems
                .iter()
                .map(|e| substitute_values_in_expr(e, subs))
                .collect();
            Expr::Tuple(new_elems)
        }

        // Cartesian product
        Expr::Times(sets) => {
            let new_sets: Vec<_> = sets
                .iter()
                .map(|s| substitute_values_in_expr(s, subs))
                .collect();
            Expr::Times(new_sets)
        }

        // CASE
        Expr::Case(arms, other) => {
            let new_arms: Vec<_> = arms
                .iter()
                .map(|arm| tla_core::ast::CaseArm {
                    guard: substitute_values_in_expr(&arm.guard, subs),
                    body: substitute_values_in_expr(&arm.body, subs),
                })
                .collect();
            let new_other = other
                .as_ref()
                .map(|e| Box::new(substitute_values_in_expr(e, subs)));
            Expr::Case(new_arms, new_other)
        }
    }
}

/// Substitute in bounds and return the filtered substitution map
fn substitute_in_bounds(
    bounds: &[BoundVar],
    subs: &HashMap<String, Value>,
) -> (Vec<BoundVar>, HashMap<String, Value>) {
    use tla_core::ast::BoundPattern;

    // Collect all bound variable names
    let mut shadowed = std::collections::HashSet::new();
    for bound in bounds {
        shadowed.insert(bound.name.node.clone());
        if let Some(BoundPattern::Tuple(vars)) = &bound.pattern {
            for var in vars {
                shadowed.insert(var.node.clone());
            }
        } else if let Some(BoundPattern::Var(var)) = &bound.pattern {
            shadowed.insert(var.node.clone());
        }
    }

    // Filter out shadowed substitutions
    let filtered_subs: HashMap<_, _> = subs
        .iter()
        .filter(|(k, _)| !shadowed.contains(*k))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    // Substitute in domains
    let new_bounds: Vec<_> = bounds
        .iter()
        .map(|b| BoundVar {
            name: b.name.clone(),
            pattern: b.pattern.clone(),
            domain: b
                .domain
                .as_ref()
                .map(|d| Box::new(substitute_values_in_expr(d, subs))),
        })
        .collect();

    (new_bounds, filtered_subs)
}

/// Convert a Value to an Expr
///
/// This is used to substitute concrete values into expressions.
fn value_to_expr(value: &Value) -> Expr {
    match value {
        Value::Bool(b) => Expr::Bool(*b),
        Value::SmallInt(n) => Expr::Int(BigInt::from(*n)),
        Value::Int(n) => Expr::Int(n.clone()),
        Value::String(s) => Expr::String(s.to_string()),
        Value::ModelValue(name) => {
            // Model values are represented as identifiers
            Expr::Ident(name.to_string())
        }
        Value::Tuple(elems) => {
            let exprs: Vec<_> = elems
                .iter()
                .map(|e| Spanned::dummy(value_to_expr(e)))
                .collect();
            Expr::Tuple(exprs)
        }
        Value::Set(set) => {
            let exprs: Vec<_> = set
                .iter()
                .map(|e| Spanned::dummy(value_to_expr(e)))
                .collect();
            Expr::SetEnum(exprs)
        }
        Value::Seq(elems) => {
            // Sequences are tuples in TLA+
            let exprs: Vec<_> = elems
                .iter()
                .map(|e| Spanned::dummy(value_to_expr(e)))
                .collect();
            Expr::Tuple(exprs)
        }
        Value::Record(fields) => {
            let field_exprs: Vec<_> = fields
                .iter()
                .map(|(k, v)| {
                    (
                        Spanned::dummy(k.to_string()),
                        Spanned::dummy(value_to_expr(v)),
                    )
                })
                .collect();
            Expr::Record(field_exprs)
        }
        Value::Func(func) => {
            // Function literals are tricky - represent as set enumeration of pairs
            let pairs: Vec<_> = func
                .entries()
                .iter()
                .map(|(k, v)| {
                    Spanned::dummy(Expr::Tuple(vec![
                        Spanned::dummy(value_to_expr(k)),
                        Spanned::dummy(value_to_expr(v)),
                    ]))
                })
                .collect();
            Expr::SetEnum(pairs)
        }
        // For complex values that can't be easily represented as expressions,
        // create a placeholder that will be looked up at evaluation time
        _ => {
            // For intervals and other special values, we use a string representation
            // that can be parsed back. This is a fallback.
            Expr::String(format!("{:?}", value))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    fn spanned<T>(node: T) -> Spanned<T> {
        Spanned::dummy(node)
    }

    fn make_ctx() -> EvalCtx {
        EvalCtx::new()
    }

    #[test]
    fn test_level_constant() {
        let conv = AstToLive::new();
        assert_eq!(conv.get_level(&Expr::Bool(true)), ExprLevel::Constant);
        assert_eq!(
            conv.get_level(&Expr::Int(BigInt::from(42))),
            ExprLevel::Constant
        );
        assert_eq!(
            conv.get_level(&Expr::String("hello".to_string())),
            ExprLevel::Constant
        );
    }

    #[test]
    fn test_level_state() {
        let conv = AstToLive::new();
        assert_eq!(
            conv.get_level(&Expr::Ident("x".to_string())),
            ExprLevel::State
        );

        // x + 1 is state level
        let add = Expr::Add(
            Box::new(spanned(Expr::Ident("x".to_string()))),
            Box::new(spanned(Expr::Int(BigInt::from(1)))),
        );
        assert_eq!(conv.get_level(&add), ExprLevel::State);
    }

    #[test]
    fn test_level_action() {
        let conv = AstToLive::new();

        // x' is action level
        let prime = Expr::Prime(Box::new(spanned(Expr::Ident("x".to_string()))));
        assert_eq!(conv.get_level(&prime), ExprLevel::Action);

        // UNCHANGED x is action level
        let unchanged = Expr::Unchanged(Box::new(spanned(Expr::Ident("x".to_string()))));
        assert_eq!(conv.get_level(&unchanged), ExprLevel::Action);

        // x' = x + 1 is action level
        let next = Expr::Eq(
            Box::new(spanned(Expr::Prime(Box::new(spanned(Expr::Ident(
                "x".to_string(),
            )))))),
            Box::new(spanned(Expr::Add(
                Box::new(spanned(Expr::Ident("x".to_string()))),
                Box::new(spanned(Expr::Int(BigInt::from(1)))),
            ))),
        );
        assert_eq!(conv.get_level(&next), ExprLevel::Action);
    }

    #[test]
    fn test_level_temporal() {
        let conv = AstToLive::new();

        // []P is temporal
        let always = Expr::Always(Box::new(spanned(Expr::Ident("P".to_string()))));
        assert_eq!(conv.get_level(&always), ExprLevel::Temporal);

        // <>P is temporal
        let eventually = Expr::Eventually(Box::new(spanned(Expr::Ident("P".to_string()))));
        assert_eq!(conv.get_level(&eventually), ExprLevel::Temporal);

        // P ~> Q is temporal
        let leads_to = Expr::LeadsTo(
            Box::new(spanned(Expr::Ident("P".to_string()))),
            Box::new(spanned(Expr::Ident("Q".to_string()))),
        );
        assert_eq!(conv.get_level(&leads_to), ExprLevel::Temporal);
    }

    #[test]
    fn test_convert_bool_constant() {
        let conv = AstToLive::new();
        let ctx = make_ctx();

        let expr = spanned(Expr::Bool(true));
        let live = conv.convert(&ctx, &expr).unwrap();
        assert!(matches!(live, LiveExpr::Bool(true)));

        let expr = spanned(Expr::Bool(false));
        let live = conv.convert(&ctx, &expr).unwrap();
        assert!(matches!(live, LiveExpr::Bool(false)));
    }

    #[test]
    fn test_convert_always() {
        let conv = AstToLive::new();
        let ctx = make_ctx();

        // []TRUE
        let expr = spanned(Expr::Always(Box::new(spanned(Expr::Bool(true)))));
        let live = conv.convert(&ctx, &expr).unwrap();

        match live {
            LiveExpr::Always(inner) => {
                assert!(matches!(*inner, LiveExpr::Bool(true)));
            }
            _ => panic!("Expected Always"),
        }
    }

    #[test]
    fn test_convert_eventually() {
        let conv = AstToLive::new();
        let ctx = make_ctx();

        // <>TRUE
        let expr = spanned(Expr::Eventually(Box::new(spanned(Expr::Bool(true)))));
        let live = conv.convert(&ctx, &expr).unwrap();

        match live {
            LiveExpr::Eventually(inner) => {
                assert!(matches!(*inner, LiveExpr::Bool(true)));
            }
            _ => panic!("Expected Eventually"),
        }
    }

    #[test]
    fn test_convert_leads_to() {
        let conv = AstToLive::new();
        let ctx = make_ctx();

        // P ~> Q expands to [](~P \/ <>Q)
        let expr = spanned(Expr::LeadsTo(
            Box::new(spanned(Expr::Bool(true))),
            Box::new(spanned(Expr::Bool(false))),
        ));
        let live = conv.convert(&ctx, &expr).unwrap();

        // Should be [](...)
        match live {
            LiveExpr::Always(inner) => {
                // Should be ~P \/ <>Q
                match *inner {
                    LiveExpr::Or(parts) => {
                        assert_eq!(parts.len(), 2);
                        // First part is ~P (which is ~TRUE = FALSE)
                        assert!(matches!(parts[0], LiveExpr::Bool(false)));
                        // Second part is <>Q (which is <>FALSE)
                        assert!(matches!(parts[1], LiveExpr::Eventually(_)));
                    }
                    _ => panic!("Expected Or inside Always"),
                }
            }
            _ => panic!("Expected Always"),
        }
    }

    #[test]
    fn test_convert_conjunction() {
        let conv = AstToLive::new();
        let ctx = make_ctx();

        // []P /\ <>Q
        let expr = spanned(Expr::And(
            Box::new(spanned(Expr::Always(Box::new(spanned(Expr::Bool(true)))))),
            Box::new(spanned(Expr::Eventually(Box::new(spanned(Expr::Bool(
                false,
            )))))),
        ));
        let live = conv.convert(&ctx, &expr).unwrap();

        match live {
            LiveExpr::And(parts) => {
                assert_eq!(parts.len(), 2);
                assert!(matches!(parts[0], LiveExpr::Always(_)));
                assert!(matches!(parts[1], LiveExpr::Eventually(_)));
            }
            _ => panic!("Expected And"),
        }
    }

    #[test]
    fn test_convert_implication() {
        let conv = AstToLive::new();
        let ctx = make_ctx();

        // []P => <>Q expands to ~[]P \/ <>Q
        let expr = spanned(Expr::Implies(
            Box::new(spanned(Expr::Always(Box::new(spanned(Expr::Bool(true)))))),
            Box::new(spanned(Expr::Eventually(Box::new(spanned(Expr::Bool(
                false,
            )))))),
        ));
        let live = conv.convert(&ctx, &expr).unwrap();

        match live {
            LiveExpr::Or(parts) => {
                assert_eq!(parts.len(), 2);
                // First part is ~[]P
                assert!(matches!(parts[0], LiveExpr::Not(_)));
                // Second part is <>Q
                assert!(matches!(parts[1], LiveExpr::Eventually(_)));
            }
            _ => panic!("Expected Or"),
        }
    }

    #[test]
    fn test_convert_negation_in_temporal() {
        let conv = AstToLive::new();
        let ctx = make_ctx();

        // ~<>P
        let expr = spanned(Expr::Not(Box::new(spanned(Expr::Eventually(Box::new(
            spanned(Expr::Bool(true)),
        ))))));
        let live = conv.convert(&ctx, &expr).unwrap();

        match &live {
            LiveExpr::Not(inner) => {
                assert!(matches!(inner.as_ref(), LiveExpr::Eventually(_)));
            }
            _ => panic!("Expected Not"),
        }

        // After push_negation: ~<>P becomes []~P
        let normalized = live.push_negation();
        match normalized {
            LiveExpr::Always(inner) => {
                // ~TRUE = FALSE
                assert!(matches!(*inner, LiveExpr::Bool(false)));
            }
            _ => panic!("Expected Always after push_negation"),
        }
    }

    #[test]
    fn test_unique_tags() {
        let conv = AstToLive::new();
        let ctx = make_ctx();

        // Each state predicate should get a unique tag
        let expr1 = spanned(Expr::Ident("x".to_string()));
        let expr2 = spanned(Expr::Ident("y".to_string()));

        let live1 = conv.convert(&ctx, &expr1).unwrap();
        let live2 = conv.convert(&ctx, &expr2).unwrap();

        let tag1 = match live1 {
            LiveExpr::StatePred { tag, .. } => tag,
            _ => panic!("Expected StatePred"),
        };
        let tag2 = match live2 {
            LiveExpr::StatePred { tag, .. } => tag,
            _ => panic!("Expected StatePred"),
        };

        assert_ne!(tag1, tag2);
    }
}
