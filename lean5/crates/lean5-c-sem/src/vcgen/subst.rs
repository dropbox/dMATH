//! Substitution methods for verification condition generation
//!
//! This module contains all substitution-related methods for VCGen,
//! including variable substitution, result substitution, and parameter
//! substitution for interprocedural analysis.

use crate::expr::CExpr;
use crate::spec::{Location, Spec};

use super::VCGen;

impl VCGen {
    /// Substitute expression for variable in spec: spec[e/x]
    pub(crate) fn substitute(&self, spec: &Spec, var: &str, expr: &CExpr) -> Spec {
        let replacement = self.expr_to_spec(expr);
        self.subst_var(spec, var, &replacement)
    }

    /// Substitute \result with expression
    pub(crate) fn substitute_result(&self, spec: &Spec, expr: &CExpr) -> Spec {
        let replacement = self.expr_to_spec(expr);
        self.subst_result(spec, &replacement)
    }

    /// Substitute variable in spec
    pub(crate) fn subst_var(&self, spec: &Spec, var: &str, replacement: &Spec) -> Spec {
        match spec {
            Spec::Var(name) if name == var => replacement.clone(),
            Spec::Var(_) | Spec::True | Spec::False | Spec::Result | Spec::Int(_) | Spec::Null => {
                spec.clone()
            }
            Spec::Expr(e) => Spec::Expr(self.subst_expr(e, var, replacement)),
            Spec::Old(s) => Spec::old(self.subst_var(s, var, replacement)),
            Spec::At { expr, label } => Spec::At {
                expr: Box::new(self.subst_var(expr, var, replacement)),
                label: label.clone(),
            },
            Spec::Forall {
                var: bound,
                ty,
                body,
            } => {
                if bound == var {
                    spec.clone() // Shadowed
                } else {
                    Spec::forall(bound, ty.clone(), self.subst_var(body, var, replacement))
                }
            }
            Spec::Exists {
                var: bound,
                ty,
                body,
            } => {
                if bound == var {
                    spec.clone()
                } else {
                    Spec::exists(bound, ty.clone(), self.subst_var(body, var, replacement))
                }
            }
            Spec::Implies(p, q) => Spec::implies(
                self.subst_var(p, var, replacement),
                self.subst_var(q, var, replacement),
            ),
            Spec::Iff(p, q) => Spec::iff(
                self.subst_var(p, var, replacement),
                self.subst_var(q, var, replacement),
            ),
            Spec::And(specs) => Spec::and(
                specs
                    .iter()
                    .map(|s| self.subst_var(s, var, replacement))
                    .collect(),
            ),
            Spec::Or(specs) => Spec::or(
                specs
                    .iter()
                    .map(|s| self.subst_var(s, var, replacement))
                    .collect(),
            ),
            Spec::Not(s) => Spec::not(self.subst_var(s, var, replacement)),
            Spec::Valid(s) => Spec::valid(self.subst_var(s, var, replacement)),
            Spec::ValidRead(s) => Spec::valid_read(self.subst_var(s, var, replacement)),
            Spec::ValidRange { ptr, lo, hi } => Spec::ValidRange {
                ptr: Box::new(self.subst_var(ptr, var, replacement)),
                lo: Box::new(self.subst_var(lo, var, replacement)),
                hi: Box::new(self.subst_var(hi, var, replacement)),
            },
            Spec::Separated(specs) => Spec::Separated(
                specs
                    .iter()
                    .map(|s| self.subst_var(s, var, replacement))
                    .collect(),
            ),
            Spec::Fresh(s) => Spec::Fresh(Box::new(self.subst_var(s, var, replacement))),
            Spec::Freeable(s) => Spec::Freeable(Box::new(self.subst_var(s, var, replacement))),
            Spec::BlockLength(s) => {
                Spec::BlockLength(Box::new(self.subst_var(s, var, replacement)))
            }
            Spec::Offset(s) => Spec::Offset(Box::new(self.subst_var(s, var, replacement))),
            Spec::BaseAddr(s) => Spec::BaseAddr(Box::new(self.subst_var(s, var, replacement))),
            Spec::Let {
                var: bound,
                value,
                body,
            } => {
                let new_value = self.subst_var(value, var, replacement);
                if bound == var {
                    Spec::Let {
                        var: bound.clone(),
                        value: Box::new(new_value),
                        body: body.clone(), // Shadowed
                    }
                } else {
                    Spec::Let {
                        var: bound.clone(),
                        value: Box::new(new_value),
                        body: Box::new(self.subst_var(body, var, replacement)),
                    }
                }
            }
            Spec::If {
                cond,
                then_spec,
                else_spec,
            } => Spec::If {
                cond: Box::new(self.subst_var(cond, var, replacement)),
                then_spec: Box::new(self.subst_var(then_spec, var, replacement)),
                else_spec: Box::new(self.subst_var(else_spec, var, replacement)),
            },
            Spec::BinOp { op, left, right } => Spec::BinOp {
                op: *op,
                left: Box::new(self.subst_var(left, var, replacement)),
                right: Box::new(self.subst_var(right, var, replacement)),
            },
            Spec::UnaryOp { op, operand } => Spec::UnaryOp {
                op: *op,
                operand: Box::new(self.subst_var(operand, var, replacement)),
            },
            Spec::Call { func, args } => Spec::Call {
                func: func.clone(),
                args: args
                    .iter()
                    .map(|a| self.subst_var(a, var, replacement))
                    .collect(),
            },
            Spec::Index { base, index } => Spec::Index {
                base: Box::new(self.subst_var(base, var, replacement)),
                index: Box::new(self.subst_var(index, var, replacement)),
            },
            Spec::Member { object, field } => Spec::Member {
                object: Box::new(self.subst_var(object, var, replacement)),
                field: field.clone(),
            },
            Spec::Sum {
                lo,
                hi,
                var: bound,
                body,
            } => {
                let new_lo = self.subst_var(lo, var, replacement);
                let new_hi = self.subst_var(hi, var, replacement);
                if bound == var {
                    Spec::Sum {
                        lo: Box::new(new_lo),
                        hi: Box::new(new_hi),
                        var: bound.clone(),
                        body: body.clone(),
                    }
                } else {
                    Spec::Sum {
                        lo: Box::new(new_lo),
                        hi: Box::new(new_hi),
                        var: bound.clone(),
                        body: Box::new(self.subst_var(body, var, replacement)),
                    }
                }
            }
            Spec::Product {
                lo,
                hi,
                var: bound,
                body,
            } => {
                let new_lo = self.subst_var(lo, var, replacement);
                let new_hi = self.subst_var(hi, var, replacement);
                if bound == var {
                    Spec::Product {
                        lo: Box::new(new_lo),
                        hi: Box::new(new_hi),
                        var: bound.clone(),
                        body: body.clone(),
                    }
                } else {
                    Spec::Product {
                        lo: Box::new(new_lo),
                        hi: Box::new(new_hi),
                        var: bound.clone(),
                        body: Box::new(self.subst_var(body, var, replacement)),
                    }
                }
            }
            Spec::Min {
                lo,
                hi,
                var: bound,
                body,
            } => {
                let new_lo = self.subst_var(lo, var, replacement);
                let new_hi = self.subst_var(hi, var, replacement);
                if bound == var {
                    Spec::Min {
                        lo: Box::new(new_lo),
                        hi: Box::new(new_hi),
                        var: bound.clone(),
                        body: body.clone(),
                    }
                } else {
                    Spec::Min {
                        lo: Box::new(new_lo),
                        hi: Box::new(new_hi),
                        var: bound.clone(),
                        body: Box::new(self.subst_var(body, var, replacement)),
                    }
                }
            }
            Spec::Max {
                lo,
                hi,
                var: bound,
                body,
            } => {
                let new_lo = self.subst_var(lo, var, replacement);
                let new_hi = self.subst_var(hi, var, replacement);
                if bound == var {
                    Spec::Max {
                        lo: Box::new(new_lo),
                        hi: Box::new(new_hi),
                        var: bound.clone(),
                        body: body.clone(),
                    }
                } else {
                    Spec::Max {
                        lo: Box::new(new_lo),
                        hi: Box::new(new_hi),
                        var: bound.clone(),
                        body: Box::new(self.subst_var(body, var, replacement)),
                    }
                }
            }
            Spec::NumOf {
                lo,
                hi,
                var: bound,
                body,
            } => {
                let new_lo = self.subst_var(lo, var, replacement);
                let new_hi = self.subst_var(hi, var, replacement);
                if bound == var {
                    Spec::NumOf {
                        lo: Box::new(new_lo),
                        hi: Box::new(new_hi),
                        var: bound.clone(),
                        body: body.clone(),
                    }
                } else {
                    Spec::NumOf {
                        lo: Box::new(new_lo),
                        hi: Box::new(new_hi),
                        var: bound.clone(),
                        body: Box::new(self.subst_var(body, var, replacement)),
                    }
                }
            }
        }
    }

    /// Substitute \result in spec
    pub(crate) fn subst_result(&self, spec: &Spec, replacement: &Spec) -> Spec {
        match spec {
            Spec::Result => replacement.clone(),
            Spec::Var(_) | Spec::True | Spec::False | Spec::Int(_) | Spec::Null | Spec::Expr(_) => {
                spec.clone()
            }
            Spec::Old(s) => Spec::old(self.subst_result(s, replacement)),
            Spec::At { expr, label } => Spec::At {
                expr: Box::new(self.subst_result(expr, replacement)),
                label: label.clone(),
            },
            Spec::Forall { var, ty, body } => {
                Spec::forall(var, ty.clone(), self.subst_result(body, replacement))
            }
            Spec::Exists { var, ty, body } => {
                Spec::exists(var, ty.clone(), self.subst_result(body, replacement))
            }
            Spec::Implies(p, q) => Spec::implies(
                self.subst_result(p, replacement),
                self.subst_result(q, replacement),
            ),
            Spec::Iff(p, q) => Spec::iff(
                self.subst_result(p, replacement),
                self.subst_result(q, replacement),
            ),
            Spec::And(specs) => Spec::and(
                specs
                    .iter()
                    .map(|s| self.subst_result(s, replacement))
                    .collect(),
            ),
            Spec::Or(specs) => Spec::or(
                specs
                    .iter()
                    .map(|s| self.subst_result(s, replacement))
                    .collect(),
            ),
            Spec::Not(s) => Spec::not(self.subst_result(s, replacement)),
            Spec::BinOp { op, left, right } => Spec::BinOp {
                op: *op,
                left: Box::new(self.subst_result(left, replacement)),
                right: Box::new(self.subst_result(right, replacement)),
            },
            Spec::UnaryOp { op, operand } => Spec::UnaryOp {
                op: *op,
                operand: Box::new(self.subst_result(operand, replacement)),
            },
            // Recurse through other constructs
            _ => spec.clone(), // Simplified for other cases
        }
    }

    /// Substitute formal parameters with actual arguments in a specification.
    /// `params[i]` is replaced with `args[i]` for all `i < min(params.len(), args.len())`
    pub(crate) fn subst_params(&self, spec: &Spec, params: &[String], args: &[Spec]) -> Spec {
        let mut result = spec.clone();
        for (param, arg) in params.iter().zip(args.iter()) {
            result = self.subst_var(&result, param, arg);
        }
        result
    }

    /// Resolve \old() expressions for interprocedural analysis.
    ///
    /// When instantiating a callee's postcondition at a call site, the callee's
    /// \old(param) refers to the value of param at the callee's entry, which is
    /// the actual argument's value at the call site. Since we've already
    /// substituted params with actual args, \old(actual_arg) should become
    /// just `actual_arg` (the current value, before the call modifies anything).
    ///
    /// Example: callee `ensures \result == \old(x) + 1`, called as `foo(y)`:
    /// - After param subst: `\result == \old(y) + 1`
    /// - After resolve_old: `\result == y + 1` (y's value at call site)
    pub(crate) fn resolve_old_for_call(&self, spec: &Spec) -> Spec {
        match spec {
            // \old(e) becomes e at the call site
            Spec::Old(inner) => self.resolve_old_for_call(inner),

            // Recurse into sub-specifications
            Spec::Var(_) | Spec::True | Spec::False | Spec::Result | Spec::Int(_) | Spec::Null => {
                spec.clone()
            }

            Spec::Expr(e) => Spec::Expr(e.clone()),

            Spec::At { expr, label } => Spec::At {
                expr: Box::new(self.resolve_old_for_call(expr)),
                label: label.clone(),
            },

            Spec::Forall { var, ty, body } => {
                Spec::forall(var, ty.clone(), self.resolve_old_for_call(body))
            }

            Spec::Exists { var, ty, body } => {
                Spec::exists(var, ty.clone(), self.resolve_old_for_call(body))
            }

            Spec::Implies(p, q) => {
                Spec::implies(self.resolve_old_for_call(p), self.resolve_old_for_call(q))
            }

            Spec::Iff(p, q) => {
                Spec::iff(self.resolve_old_for_call(p), self.resolve_old_for_call(q))
            }

            Spec::And(specs) => {
                Spec::and(specs.iter().map(|s| self.resolve_old_for_call(s)).collect())
            }

            Spec::Or(specs) => {
                Spec::or(specs.iter().map(|s| self.resolve_old_for_call(s)).collect())
            }

            Spec::Not(s) => Spec::not(self.resolve_old_for_call(s)),

            Spec::Valid(s) => Spec::valid(self.resolve_old_for_call(s)),
            Spec::ValidRead(s) => Spec::valid_read(self.resolve_old_for_call(s)),

            Spec::ValidRange { ptr, lo, hi } => Spec::ValidRange {
                ptr: Box::new(self.resolve_old_for_call(ptr)),
                lo: Box::new(self.resolve_old_for_call(lo)),
                hi: Box::new(self.resolve_old_for_call(hi)),
            },

            Spec::BinOp { op, left, right } => Spec::binop(
                *op,
                self.resolve_old_for_call(left),
                self.resolve_old_for_call(right),
            ),

            Spec::UnaryOp { op, operand } => Spec::UnaryOp {
                op: *op,
                operand: Box::new(self.resolve_old_for_call(operand)),
            },

            Spec::Index { base, index } => Spec::Index {
                base: Box::new(self.resolve_old_for_call(base)),
                index: Box::new(self.resolve_old_for_call(index)),
            },

            Spec::Separated(specs) => {
                Spec::Separated(specs.iter().map(|s| self.resolve_old_for_call(s)).collect())
            }

            Spec::Fresh(s) => Spec::Fresh(Box::new(self.resolve_old_for_call(s))),

            Spec::Freeable(s) => Spec::Freeable(Box::new(self.resolve_old_for_call(s))),

            Spec::Call { func, args } => Spec::Call {
                func: func.clone(),
                args: args.iter().map(|a| self.resolve_old_for_call(a)).collect(),
            },

            Spec::Member { object, field } => Spec::Member {
                object: Box::new(self.resolve_old_for_call(object)),
                field: field.clone(),
            },

            Spec::BlockLength(s) => Spec::BlockLength(Box::new(self.resolve_old_for_call(s))),

            Spec::Offset(s) => Spec::Offset(Box::new(self.resolve_old_for_call(s))),

            Spec::BaseAddr(s) => Spec::BaseAddr(Box::new(self.resolve_old_for_call(s))),

            Spec::Let { var, value, body } => Spec::Let {
                var: var.clone(),
                value: Box::new(self.resolve_old_for_call(value)),
                body: Box::new(self.resolve_old_for_call(body)),
            },

            Spec::If {
                cond,
                then_spec,
                else_spec,
            } => Spec::If {
                cond: Box::new(self.resolve_old_for_call(cond)),
                then_spec: Box::new(self.resolve_old_for_call(then_spec)),
                else_spec: Box::new(self.resolve_old_for_call(else_spec)),
            },

            Spec::Sum { lo, hi, var, body } => Spec::Sum {
                lo: Box::new(self.resolve_old_for_call(lo)),
                hi: Box::new(self.resolve_old_for_call(hi)),
                var: var.clone(),
                body: Box::new(self.resolve_old_for_call(body)),
            },

            Spec::Product { lo, hi, var, body } => Spec::Product {
                lo: Box::new(self.resolve_old_for_call(lo)),
                hi: Box::new(self.resolve_old_for_call(hi)),
                var: var.clone(),
                body: Box::new(self.resolve_old_for_call(body)),
            },

            Spec::Min { lo, hi, var, body } => Spec::Min {
                lo: Box::new(self.resolve_old_for_call(lo)),
                hi: Box::new(self.resolve_old_for_call(hi)),
                var: var.clone(),
                body: Box::new(self.resolve_old_for_call(body)),
            },

            Spec::Max { lo, hi, var, body } => Spec::Max {
                lo: Box::new(self.resolve_old_for_call(lo)),
                hi: Box::new(self.resolve_old_for_call(hi)),
                var: var.clone(),
                body: Box::new(self.resolve_old_for_call(body)),
            },

            Spec::NumOf { lo, hi, var, body } => Spec::NumOf {
                lo: Box::new(self.resolve_old_for_call(lo)),
                hi: Box::new(self.resolve_old_for_call(hi)),
                var: var.clone(),
                body: Box::new(self.resolve_old_for_call(body)),
            },
        }
    }

    /// Substitute formal parameters in a Location (for assigns clauses)
    pub(crate) fn subst_params_in_location(
        &self,
        loc: &Location,
        params: &[String],
        args: &[Spec],
    ) -> Location {
        match loc {
            Location::Deref(spec) => Location::Deref(self.subst_params(spec, params, args)),
            Location::Range { base, lo, hi } => Location::Range {
                base: self.subst_params(base, params, args),
                lo: self.subst_params(lo, params, args),
                hi: self.subst_params(hi, params, args),
            },
            Location::Reachable(spec) => Location::Reachable(self.subst_params(spec, params, args)),
            // Nothing and Everything don't contain variables
            Location::Nothing | Location::Everything => loc.clone(),
        }
    }

    /// Convert a Spec to CExpr when possible
    /// Returns None for spec constructs that have no direct CExpr equivalent
    pub(crate) fn spec_to_cexpr(&self, spec: &Spec) -> Option<CExpr> {
        match spec {
            Spec::Expr(e) => Some(e.clone()),
            Spec::Int(n) => Some(CExpr::IntLit(*n)),
            Spec::Var(name) => Some(CExpr::Var(name.clone())),
            Spec::Null => Some(CExpr::IntLit(0)), // NULL is (void*)0
            Spec::BinOp { op, left, right } => {
                let l = self.spec_to_cexpr(left)?;
                let r = self.spec_to_cexpr(right)?;
                Some(CExpr::binop(*op, l, r))
            }
            Spec::UnaryOp { op, operand } => {
                let e = self.spec_to_cexpr(operand)?;
                Some(CExpr::unary(*op, e))
            }
            Spec::Index { base, index } => {
                let b = self.spec_to_cexpr(base)?;
                let i = self.spec_to_cexpr(index)?;
                Some(CExpr::index(b, i))
            }
            Spec::Member { object, field } => {
                let obj = self.spec_to_cexpr(object)?;
                Some(CExpr::Member {
                    object: Box::new(obj),
                    field: field.clone(),
                })
            }
            // Spec-only constructs that don't map to CExpr
            Spec::True
            | Spec::False
            | Spec::Result
            | Spec::Old(_)
            | Spec::At { .. }
            | Spec::Forall { .. }
            | Spec::Exists { .. }
            | Spec::Implies(_, _)
            | Spec::Iff(_, _)
            | Spec::And(_)
            | Spec::Or(_)
            | Spec::Not(_)
            | Spec::Valid(_)
            | Spec::ValidRead(_)
            | Spec::ValidRange { .. }
            | Spec::Separated(_)
            | Spec::Fresh(_)
            | Spec::Freeable(_)
            | Spec::BlockLength(_)
            | Spec::Offset(_)
            | Spec::BaseAddr(_)
            | Spec::Let { .. }
            | Spec::If { .. }
            | Spec::Call { .. }
            | Spec::Sum { .. }
            | Spec::Product { .. }
            | Spec::Min { .. }
            | Spec::Max { .. }
            | Spec::NumOf { .. } => None,
        }
    }

    /// Substitute variable in C expression (returns modified expr)
    /// Recursively traverses the expression and replaces occurrences of `var`
    /// with the CExpr equivalent of `replacement` when possible.
    pub(crate) fn subst_expr(&self, expr: &CExpr, var: &str, replacement: &Spec) -> CExpr {
        match expr {
            CExpr::Var(name) if name == var => {
                // Try to convert the Spec replacement to CExpr
                self.spec_to_cexpr(replacement)
                    .unwrap_or_else(|| expr.clone())
            }
            CExpr::BinOp { op, left, right } => CExpr::binop(
                *op,
                self.subst_expr(left, var, replacement),
                self.subst_expr(right, var, replacement),
            ),

            CExpr::UnaryOp { op, operand } => {
                CExpr::unary(*op, self.subst_expr(operand, var, replacement))
            }

            CExpr::Conditional {
                cond,
                then_expr,
                else_expr,
            } => CExpr::Conditional {
                cond: Box::new(self.subst_expr(cond, var, replacement)),
                then_expr: Box::new(self.subst_expr(then_expr, var, replacement)),
                else_expr: Box::new(self.subst_expr(else_expr, var, replacement)),
            },

            CExpr::Cast { ty, expr: e } => CExpr::Cast {
                ty: ty.clone(),
                expr: Box::new(self.subst_expr(e, var, replacement)),
            },

            CExpr::Call { func, args } => CExpr::Call {
                func: Box::new(self.subst_expr(func, var, replacement)),
                args: args
                    .iter()
                    .map(|a| self.subst_expr(a, var, replacement))
                    .collect(),
            },

            CExpr::Index { array, index } => CExpr::index(
                self.subst_expr(array, var, replacement),
                self.subst_expr(index, var, replacement),
            ),

            CExpr::Member { object, field } => CExpr::Member {
                object: Box::new(self.subst_expr(object, var, replacement)),
                field: field.clone(),
            },

            CExpr::Arrow { pointer, field } => CExpr::Arrow {
                pointer: Box::new(self.subst_expr(pointer, var, replacement)),
                field: field.clone(),
            },

            // Literals, type-only expressions, statement expressions, and unmatched vars
            // don't contain substitutable variables or are too complex
            CExpr::Var(_)
            | CExpr::StmtExpr(_)
            | CExpr::IntLit(_)
            | CExpr::UIntLit(_)
            | CExpr::FloatLit(_)
            | CExpr::CharLit(_)
            | CExpr::StringLit(_)
            | CExpr::SizeOf(_)
            | CExpr::AlignOf(_)
            | CExpr::CompoundLiteral { .. }
            | CExpr::Generic { .. } => expr.clone(),
        }
    }
}
