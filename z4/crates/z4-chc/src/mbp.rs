//! Model-Based Projection (MBP) for quantifier elimination
//!
//! This module implements model-based projection, an efficient technique for
//! quantifier elimination guided by a satisfying model. MBP is used in PDR/IC3
//! to compute more general predecessor states.
//!
//! Based on Golem's implementation (MIT license) and the paper:
//! Bjorner & Janota, "Playing with Quantified Satisfaction", LPAR-20, 2015

// Allow &self in recursive methods - needed for method syntax consistency
#![allow(clippy::only_used_in_recursion)]

use crate::{ChcExpr, ChcOp, ChcSort, ChcVar, SmtValue};
use rustc_hash::FxHashMap;
use std::sync::Arc;

/// Floor division for integers (rounds toward negative infinity)
fn div_floor(a: i64, b: i64) -> i64 {
    let d = a / b;
    let r = a % b;
    if (r > 0 && b < 0) || (r < 0 && b > 0) {
        d - 1
    } else {
        d
    }
}

/// Floor modulo for integers (result has same sign as divisor)
fn mod_floor(a: i64, b: i64) -> i64 {
    let r = a % b;
    if (r > 0 && b < 0) || (r < 0 && b > 0) {
        r + b
    } else {
        r
    }
}

/// A literal in an implicant: (atom, polarity)
/// If polarity is true, the literal is `atom`; otherwise it's `NOT atom`
#[derive(Debug, Clone)]
pub struct Literal {
    pub atom: ChcExpr,
    pub positive: bool,
}

impl Literal {
    pub fn new(atom: ChcExpr, positive: bool) -> Self {
        Self { atom, positive }
    }

    /// Convert back to a ChcExpr
    pub fn to_expr(&self) -> ChcExpr {
        if self.positive {
            self.atom.clone()
        } else {
            ChcExpr::not(self.atom.clone())
        }
    }
}

/// A bound on a variable: coeff * var <= term (or < for strict)
#[derive(Debug, Clone)]
#[allow(dead_code)] // Infrastructure for future MBP integration
pub struct Bound {
    /// Coefficient of the variable (positive for upper bound, negative for lower)
    pub coeff: i64,
    /// The term on the right side
    pub term: ChcExpr,
    /// Whether this is a strict inequality
    pub strict: bool,
}

/// Divisibility constraint: term mod constant = 0
#[derive(Debug, Clone)]
#[allow(dead_code)] // Infrastructure for future MBP integration
pub struct DivisibilityConstraint {
    pub constant: i64,
    pub term: ChcExpr,
}

/// Model-Based Projection engine
pub struct Mbp {
    // Reserved for future use (e.g., fresh variable generation)
    _reserved: (),
}

impl Default for Mbp {
    fn default() -> Self {
        Self::new()
    }
}

impl Mbp {
    pub fn new() -> Self {
        Self { _reserved: () }
    }

    /// Project away variables from a formula under a model
    ///
    /// Given a formula `phi`, variables to eliminate `vars`, and a model `M`,
    /// returns a formula `psi` such that:
    /// 1. M |= psi
    /// 2. psi implies (exists vars. phi)
    /// 3. psi does not contain any variable from `vars`
    pub fn project(
        &self,
        formula: &ChcExpr,
        vars_to_eliminate: &[ChcVar],
        model: &FxHashMap<String, SmtValue>,
    ) -> ChcExpr {
        if vars_to_eliminate.is_empty() {
            return formula.clone();
        }

        // First, substitute Boolean variables directly with model values
        let (bool_vars, arith_vars): (Vec<_>, Vec<_>) = vars_to_eliminate
            .iter()
            .partition(|v| v.sort == ChcSort::Bool);

        let mut current = formula.clone();
        for var in &bool_vars {
            if let Some(SmtValue::Bool(b)) = model.get(&var.name) {
                current = current.substitute(&[((*var).clone(), ChcExpr::Bool(*b))]);
            }
        }

        if arith_vars.is_empty() {
            return self.simplify(&current);
        }

        // Extract implicant (the literals that are true under model)
        let mut implicant = self.get_implicant(&current, model);

        // Project out each arithmetic variable
        for var in &arith_vars {
            implicant = self.project_single_var(var, implicant, model);
        }

        // Convert implicant back to formula
        self.implicant_to_formula(&implicant)
    }

    /// Extract an implicant from a formula under a model
    ///
    /// An implicant is a conjunction of literals that:
    /// 1. Is satisfied by the model
    /// 2. Implies the original formula
    fn get_implicant(
        &self,
        formula: &ChcExpr,
        model: &FxHashMap<String, SmtValue>,
    ) -> Vec<Literal> {
        let mut literals = Vec::new();
        self.collect_implicant(formula, model, &mut literals);
        literals
    }

    fn collect_implicant(
        &self,
        formula: &ChcExpr,
        model: &FxHashMap<String, SmtValue>,
        literals: &mut Vec<Literal>,
    ) {
        match formula {
            ChcExpr::Bool(true) => {}
            ChcExpr::Bool(false) => {
                // This shouldn't happen if model satisfies formula
            }
            ChcExpr::Op(ChcOp::And, args) => {
                for arg in args {
                    self.collect_implicant(arg, model, literals);
                }
            }
            ChcExpr::Op(ChcOp::Or, args) => {
                // Pick one satisfied disjunct
                for arg in args {
                    if self.eval_bool(arg, model) == Some(true) {
                        self.collect_implicant(arg, model, literals);
                        return;
                    }
                }
            }
            ChcExpr::Op(ChcOp::Not, args) if args.len() == 1 => {
                let inner = args[0].as_ref();
                if self.is_atom(inner) {
                    literals.push(Literal::new(inner.clone(), false));
                } else {
                    // Push NNF inward
                    match inner {
                        ChcExpr::Op(ChcOp::And, inner_args) => {
                            // not(a and b) = not(a) or not(b) - pick satisfied one
                            for arg in inner_args {
                                if self.eval_bool(arg, model) == Some(false) {
                                    self.collect_implicant(
                                        &ChcExpr::not(arg.as_ref().clone()),
                                        model,
                                        literals,
                                    );
                                    return;
                                }
                            }
                        }
                        ChcExpr::Op(ChcOp::Or, inner_args) => {
                            // not(a or b) = not(a) and not(b)
                            for arg in inner_args {
                                self.collect_implicant(
                                    &ChcExpr::not(arg.as_ref().clone()),
                                    model,
                                    literals,
                                );
                            }
                        }
                        ChcExpr::Op(ChcOp::Not, inner_inner) if inner_inner.len() == 1 => {
                            // not(not(a)) = a
                            self.collect_implicant(&inner_inner[0], model, literals);
                        }
                        _ => {
                            // Treat as atom
                            literals.push(Literal::new(inner.clone(), false));
                        }
                    }
                }
            }
            _ => {
                // Atomic formula
                if self.is_atom(formula) {
                    literals.push(Literal::new(formula.clone(), true));
                }
            }
        }
    }

    /// Check if an expression is an atomic formula (not And/Or/Not/Implies)
    fn is_atom(&self, expr: &ChcExpr) -> bool {
        match expr {
            ChcExpr::Bool(_) => true,
            ChcExpr::Var(v) => v.sort == ChcSort::Bool,
            ChcExpr::Op(op, _) => matches!(
                op,
                ChcOp::Eq | ChcOp::Ne | ChcOp::Lt | ChcOp::Le | ChcOp::Gt | ChcOp::Ge
            ),
            _ => false,
        }
    }

    /// Evaluate a boolean expression under a model
    fn eval_bool(&self, expr: &ChcExpr, model: &FxHashMap<String, SmtValue>) -> Option<bool> {
        match expr {
            ChcExpr::Bool(b) => Some(*b),
            ChcExpr::Var(v) if v.sort == ChcSort::Bool => {
                if let Some(SmtValue::Bool(b)) = model.get(&v.name) {
                    Some(*b)
                } else {
                    None
                }
            }
            ChcExpr::Op(ChcOp::Not, args) if args.len() == 1 => {
                self.eval_bool(&args[0], model).map(|b| !b)
            }
            ChcExpr::Op(ChcOp::And, args) => {
                let mut result = true;
                for arg in args {
                    match self.eval_bool(arg, model) {
                        Some(false) => return Some(false),
                        Some(true) => {}
                        None => result = false,
                    }
                }
                if result {
                    Some(true)
                } else {
                    None
                }
            }
            ChcExpr::Op(ChcOp::Or, args) => {
                for arg in args {
                    if self.eval_bool(arg, model) == Some(true) {
                        return Some(true);
                    }
                }
                Some(false)
            }
            ChcExpr::Op(ChcOp::Eq, args) if args.len() == 2 => {
                let v1 = self.eval_arith(&args[0], model)?;
                let v2 = self.eval_arith(&args[1], model)?;
                Some(v1 == v2)
            }
            ChcExpr::Op(ChcOp::Ne, args) if args.len() == 2 => {
                let v1 = self.eval_arith(&args[0], model)?;
                let v2 = self.eval_arith(&args[1], model)?;
                Some(v1 != v2)
            }
            ChcExpr::Op(ChcOp::Lt, args) if args.len() == 2 => {
                let v1 = self.eval_arith(&args[0], model)?;
                let v2 = self.eval_arith(&args[1], model)?;
                Some(v1 < v2)
            }
            ChcExpr::Op(ChcOp::Le, args) if args.len() == 2 => {
                let v1 = self.eval_arith(&args[0], model)?;
                let v2 = self.eval_arith(&args[1], model)?;
                Some(v1 <= v2)
            }
            ChcExpr::Op(ChcOp::Gt, args) if args.len() == 2 => {
                let v1 = self.eval_arith(&args[0], model)?;
                let v2 = self.eval_arith(&args[1], model)?;
                Some(v1 > v2)
            }
            ChcExpr::Op(ChcOp::Ge, args) if args.len() == 2 => {
                let v1 = self.eval_arith(&args[0], model)?;
                let v2 = self.eval_arith(&args[1], model)?;
                Some(v1 >= v2)
            }
            _ => None,
        }
    }

    /// Evaluate an arithmetic expression under a model
    fn eval_arith(&self, expr: &ChcExpr, model: &FxHashMap<String, SmtValue>) -> Option<i64> {
        match expr {
            ChcExpr::Int(n) => Some(*n),
            ChcExpr::Var(v) if v.sort == ChcSort::Int => {
                if let Some(SmtValue::Int(n)) = model.get(&v.name) {
                    Some(*n)
                } else {
                    None
                }
            }
            ChcExpr::Op(ChcOp::Add, args) if args.len() == 2 => {
                let v1 = self.eval_arith(&args[0], model)?;
                let v2 = self.eval_arith(&args[1], model)?;
                v1.checked_add(v2)
            }
            ChcExpr::Op(ChcOp::Sub, args) if args.len() == 2 => {
                let v1 = self.eval_arith(&args[0], model)?;
                let v2 = self.eval_arith(&args[1], model)?;
                v1.checked_sub(v2)
            }
            ChcExpr::Op(ChcOp::Mul, args) if args.len() == 2 => {
                let v1 = self.eval_arith(&args[0], model)?;
                let v2 = self.eval_arith(&args[1], model)?;
                v1.checked_mul(v2)
            }
            ChcExpr::Op(ChcOp::Neg, args) if args.len() == 1 => {
                let v = self.eval_arith(&args[0], model)?;
                v.checked_neg()
            }
            ChcExpr::Op(ChcOp::Div, args) if args.len() == 2 => {
                let v1 = self.eval_arith(&args[0], model)?;
                let v2 = self.eval_arith(&args[1], model)?;
                if v2 == 0 {
                    None
                } else {
                    Some(div_floor(v1, v2))
                }
            }
            ChcExpr::Op(ChcOp::Mod, args) if args.len() == 2 => {
                let v1 = self.eval_arith(&args[0], model)?;
                let v2 = self.eval_arith(&args[1], model)?;
                if v2 == 0 {
                    None
                } else {
                    Some(mod_floor(v1, v2))
                }
            }
            _ => None,
        }
    }

    /// Project out a single variable from the implicant
    fn project_single_var(
        &self,
        var: &ChcVar,
        implicant: Vec<Literal>,
        model: &FxHashMap<String, SmtValue>,
    ) -> Vec<Literal> {
        // Partition literals into those containing var and those not
        let (with_var, without_var): (Vec<_>, Vec<_>) = implicant
            .into_iter()
            .partition(|lit| self.contains_var(&lit.atom, var));

        if with_var.is_empty() {
            return without_var;
        }

        match var.sort {
            ChcSort::Int => self.project_integer_var(var, with_var, without_var, model),
            ChcSort::Real => self.project_real_var(var, with_var, without_var, model),
            _ => without_var, // Just drop literals for unsupported sorts
        }
    }

    /// Project out an integer variable (LIA)
    fn project_integer_var(
        &self,
        var: &ChcVar,
        with_var: Vec<Literal>,
        mut without_var: Vec<Literal>,
        model: &FxHashMap<String, SmtValue>,
    ) -> Vec<Literal> {
        // Collect bounds: lower (ax >= t), upper (ax <= t), and equalities
        let mut lower_bounds: Vec<(i64, ChcExpr, bool)> = Vec::new(); // (coeff, term, strict)
        let mut upper_bounds: Vec<(i64, ChcExpr, bool)> = Vec::new();
        let mut equalities: Vec<(i64, ChcExpr)> = Vec::new(); // ax = t

        for lit in &with_var {
            if let Some(bound) = self.extract_bound(&lit.atom, var, lit.positive) {
                match bound {
                    BoundKind::Lower(coeff, term, strict) => {
                        lower_bounds.push((coeff, term, strict));
                    }
                    BoundKind::Upper(coeff, term, strict) => {
                        upper_bounds.push((coeff, term, strict));
                    }
                    BoundKind::Equality(coeff, term) => {
                        equalities.push((coeff, term));
                    }
                }
            }
        }

        // If we have an equality, use it to substitute
        if let Some((coeff, term)) = equalities.first() {
            if *coeff == 1 || *coeff == -1 {
                // Can directly substitute: var = term or var = -term
                let subst_term = if *coeff == 1 {
                    term.clone()
                } else {
                    ChcExpr::neg(term.clone())
                };
                // Substitute in remaining literals
                for lit in &with_var {
                    let new_atom = lit.atom.substitute(&[(var.clone(), subst_term.clone())]);
                    // Simplify and check if non-trivial
                    let simplified = self.simplify(&new_atom);
                    if simplified != ChcExpr::Bool(true) {
                        without_var.push(Literal::new(simplified, lit.positive));
                    }
                }
                return without_var;
            } else {
                // Need divisibility constraint: term mod |coeff| = 0
                // For now, just use the equality directly
                let abs_coeff = coeff.abs();
                if abs_coeff > 1 {
                    // Add divisibility constraint: term mod coeff = 0
                    let div_check = ChcExpr::eq(
                        ChcExpr::Op(
                            ChcOp::Mod,
                            vec![Arc::new(term.clone()), Arc::new(ChcExpr::Int(abs_coeff))],
                        ),
                        ChcExpr::Int(0),
                    );
                    without_var.push(Literal::new(div_check, true));
                }
            }
        }

        // No equality - use bounds to eliminate variable
        if lower_bounds.is_empty() || upper_bounds.is_empty() {
            // Missing one side of bounds - variable is unconstrained
            return without_var;
        }

        // Find the tightest lower bound according to model
        let _var_val = model
            .get(&var.name)
            .and_then(|v| match v {
                SmtValue::Int(n) => Some(*n),
                _ => None,
            })
            .unwrap_or(0);

        let best_lower = lower_bounds.iter().max_by(|(c1, t1, _), (c2, t2, _)| {
            let v1 = self
                .eval_arith(t1, model)
                .map(|t| t / c1.abs())
                .unwrap_or(i64::MIN);
            let v2 = self
                .eval_arith(t2, model)
                .map(|t| t / c2.abs())
                .unwrap_or(i64::MIN);
            v1.cmp(&v2)
        });

        // Generate resolved constraints
        if let Some((lb_coeff, lb_term, lb_strict)) = best_lower {
            for (ub_coeff, ub_term, ub_strict) in &upper_bounds {
                // Resolve lower and upper: lb_coeff * x >= lb_term, ub_coeff * x <= ub_term
                // => lb_coeff * ub_term >= ub_coeff * lb_term (with adjustments for integers)
                let new_strict = *lb_strict || *ub_strict;

                // Scale terms appropriately
                let lb_coeff_abs = lb_coeff.abs();
                let ub_coeff_abs = ub_coeff.abs();

                let scaled_lb = if ub_coeff_abs == 1 {
                    lb_term.clone()
                } else {
                    ChcExpr::mul(ChcExpr::Int(ub_coeff_abs), lb_term.clone())
                };
                let scaled_ub = if lb_coeff_abs == 1 {
                    ub_term.clone()
                } else {
                    ChcExpr::mul(ChcExpr::Int(lb_coeff_abs), ub_term.clone())
                };

                // Add slack for strict integer inequality
                let final_lb = if new_strict && var.sort == ChcSort::Int {
                    ChcExpr::add(scaled_lb.clone(), ChcExpr::Int(1))
                } else {
                    scaled_lb
                };

                // lb <= ub
                let resolved = ChcExpr::le(final_lb, scaled_ub);
                let simplified = self.simplify(&resolved);
                if simplified != ChcExpr::Bool(true) {
                    without_var.push(Literal::new(simplified, true));
                }
            }

            // Also add bounds between lower bounds
            for (other_coeff, other_term, other_strict) in &lower_bounds {
                if std::ptr::eq(lb_term, other_term) {
                    continue;
                }
                // lb_term / lb_coeff <= other_term / other_coeff
                let scaled_lb = if *other_coeff == *lb_coeff {
                    lb_term.clone()
                } else {
                    ChcExpr::mul(ChcExpr::Int(other_coeff.abs()), lb_term.clone())
                };
                let scaled_other = if *lb_coeff == *other_coeff {
                    other_term.clone()
                } else {
                    ChcExpr::mul(ChcExpr::Int(lb_coeff.abs()), other_term.clone())
                };
                let new_strict = *lb_strict && !other_strict;
                let cmp = if new_strict {
                    ChcExpr::lt(scaled_other, scaled_lb)
                } else {
                    ChcExpr::le(scaled_other, scaled_lb)
                };
                let simplified = self.simplify(&cmp);
                if simplified != ChcExpr::Bool(true) {
                    without_var.push(Literal::new(simplified, true));
                }
            }
        }

        without_var
    }

    /// Project out a real variable (LRA)
    fn project_real_var(
        &self,
        var: &ChcVar,
        with_var: Vec<Literal>,
        mut without_var: Vec<Literal>,
        _model: &FxHashMap<String, SmtValue>,
    ) -> Vec<Literal> {
        // Similar to integer but without divisibility concerns
        let mut lower_bounds: Vec<(ChcExpr, bool)> = Vec::new(); // (term, strict)
        let mut upper_bounds: Vec<(ChcExpr, bool)> = Vec::new();

        for lit in &with_var {
            if let Some(bound) = self.extract_bound(&lit.atom, var, lit.positive) {
                match bound {
                    BoundKind::Lower(_, term, strict) => {
                        lower_bounds.push((term, strict));
                    }
                    BoundKind::Upper(_, term, strict) => {
                        upper_bounds.push((term, strict));
                    }
                    BoundKind::Equality(_, term) => {
                        // For equality, substitute
                        for other_lit in &with_var {
                            let new_atom =
                                other_lit.atom.substitute(&[(var.clone(), term.clone())]);
                            let simplified = self.simplify(&new_atom);
                            if simplified != ChcExpr::Bool(true) {
                                without_var.push(Literal::new(simplified, other_lit.positive));
                            }
                        }
                        return without_var;
                    }
                }
            }
        }

        if lower_bounds.is_empty() || upper_bounds.is_empty() {
            return without_var;
        }

        // Resolve each lower bound with each upper bound
        for (lb_term, lb_strict) in &lower_bounds {
            for (ub_term, ub_strict) in &upper_bounds {
                let new_strict = *lb_strict || *ub_strict;
                let cmp = if new_strict {
                    ChcExpr::lt(lb_term.clone(), ub_term.clone())
                } else {
                    ChcExpr::le(lb_term.clone(), ub_term.clone())
                };
                let simplified = self.simplify(&cmp);
                if simplified != ChcExpr::Bool(true) {
                    without_var.push(Literal::new(simplified, true));
                }
            }
        }

        without_var
    }

    /// Check if an expression contains a variable
    fn contains_var(&self, expr: &ChcExpr, var: &ChcVar) -> bool {
        match expr {
            ChcExpr::Var(v) => v == var,
            ChcExpr::Bool(_) | ChcExpr::Int(_) | ChcExpr::Real(_, _) => false,
            ChcExpr::Op(_, args) => args.iter().any(|a| self.contains_var(a, var)),
            ChcExpr::PredicateApp(_, _, args) => args.iter().any(|a| self.contains_var(a, var)),
        }
    }

    /// Extract bound information from a comparison atom
    fn extract_bound(&self, atom: &ChcExpr, var: &ChcVar, positive: bool) -> Option<BoundKind> {
        match atom {
            ChcExpr::Op(op, args) if args.len() == 2 => {
                let lhs = &args[0];
                let rhs = &args[1];

                // Try to normalize to: coeff * var op term
                if let Some((coeff, term)) = self.factor_var(lhs, rhs, var) {
                    let (effective_op, effective_coeff) = if positive {
                        (op.clone(), coeff)
                    } else {
                        // Negate the comparison
                        (self.negate_cmp(op), coeff)
                    };

                    // Determine bound type based on coefficient sign and comparison
                    match effective_op {
                        ChcOp::Eq => Some(BoundKind::Equality(effective_coeff, term)),
                        ChcOp::Le => {
                            if effective_coeff > 0 {
                                Some(BoundKind::Upper(effective_coeff, term, false))
                            } else {
                                Some(BoundKind::Lower(-effective_coeff, term, false))
                            }
                        }
                        ChcOp::Lt => {
                            if effective_coeff > 0 {
                                Some(BoundKind::Upper(effective_coeff, term, true))
                            } else {
                                Some(BoundKind::Lower(-effective_coeff, term, true))
                            }
                        }
                        ChcOp::Ge => {
                            if effective_coeff > 0 {
                                Some(BoundKind::Lower(effective_coeff, term, false))
                            } else {
                                Some(BoundKind::Upper(-effective_coeff, term, false))
                            }
                        }
                        ChcOp::Gt => {
                            if effective_coeff > 0 {
                                Some(BoundKind::Lower(effective_coeff, term, true))
                            } else {
                                Some(BoundKind::Upper(-effective_coeff, term, true))
                            }
                        }
                        _ => None,
                    }
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Try to express `lhs op rhs` as `coeff * var op term`
    /// Returns (coeff, term) where term is the expression with var factored out
    fn factor_var(&self, lhs: &ChcExpr, rhs: &ChcExpr, var: &ChcVar) -> Option<(i64, ChcExpr)> {
        // Simple case: var op term or term op var
        if let ChcExpr::Var(v) = lhs {
            if v == var {
                return Some((1, rhs.clone()));
            }
        }
        if let ChcExpr::Var(v) = rhs {
            if v == var {
                return Some((-1, ChcExpr::neg(lhs.clone())));
            }
        }

        // Handle linear expressions
        let (lhs_coeff, lhs_rest) = self.extract_var_coeff(lhs, var);
        let (rhs_coeff, rhs_rest) = self.extract_var_coeff(rhs, var);

        let total_coeff = lhs_coeff - rhs_coeff;
        if total_coeff == 0 {
            return None; // Variable cancels out
        }

        // term = rhs_rest - lhs_rest (rearranging: total_coeff * var <= rhs_rest - lhs_rest)
        let term = ChcExpr::sub(rhs_rest, lhs_rest);
        Some((total_coeff, term))
    }

    /// Extract the coefficient of a variable in a linear term
    /// Returns (coefficient, remaining term without the variable)
    fn extract_var_coeff(&self, expr: &ChcExpr, var: &ChcVar) -> (i64, ChcExpr) {
        match expr {
            ChcExpr::Var(v) if v == var => (1, ChcExpr::Int(0)),
            ChcExpr::Int(_) | ChcExpr::Real(_, _) | ChcExpr::Bool(_) | ChcExpr::Var(_) => {
                (0, expr.clone())
            }
            ChcExpr::Op(ChcOp::Neg, args) if args.len() == 1 => {
                let (c, rest) = self.extract_var_coeff(&args[0], var);
                (-c, ChcExpr::neg(rest))
            }
            ChcExpr::Op(ChcOp::Add, args) if args.len() == 2 => {
                let (c1, r1) = self.extract_var_coeff(&args[0], var);
                let (c2, r2) = self.extract_var_coeff(&args[1], var);
                (c1 + c2, ChcExpr::add(r1, r2))
            }
            ChcExpr::Op(ChcOp::Sub, args) if args.len() == 2 => {
                let (c1, r1) = self.extract_var_coeff(&args[0], var);
                let (c2, r2) = self.extract_var_coeff(&args[1], var);
                (c1 - c2, ChcExpr::sub(r1, r2))
            }
            ChcExpr::Op(ChcOp::Mul, args) if args.len() == 2 => {
                // Check for c * var or var * c
                if let ChcExpr::Int(c) = args[0].as_ref() {
                    if let ChcExpr::Var(v) = args[1].as_ref() {
                        if v == var {
                            return (*c, ChcExpr::Int(0));
                        }
                    }
                    let (inner_c, inner_r) = self.extract_var_coeff(&args[1], var);
                    return (c * inner_c, ChcExpr::mul(ChcExpr::Int(*c), inner_r));
                }
                if let ChcExpr::Int(c) = args[1].as_ref() {
                    if let ChcExpr::Var(v) = args[0].as_ref() {
                        if v == var {
                            return (*c, ChcExpr::Int(0));
                        }
                    }
                    let (inner_c, inner_r) = self.extract_var_coeff(&args[0], var);
                    return (c * inner_c, ChcExpr::mul(inner_r, ChcExpr::Int(*c)));
                }
                (0, expr.clone())
            }
            _ => (0, expr.clone()),
        }
    }

    /// Negate a comparison operator
    fn negate_cmp(&self, op: &ChcOp) -> ChcOp {
        match op {
            ChcOp::Lt => ChcOp::Ge,
            ChcOp::Le => ChcOp::Gt,
            ChcOp::Gt => ChcOp::Le,
            ChcOp::Ge => ChcOp::Lt,
            ChcOp::Eq => ChcOp::Ne,
            ChcOp::Ne => ChcOp::Eq,
            other => other.clone(),
        }
    }

    /// Convert implicant back to a formula
    fn implicant_to_formula(&self, implicant: &[Literal]) -> ChcExpr {
        if implicant.is_empty() {
            return ChcExpr::Bool(true);
        }
        if implicant.len() == 1 {
            return implicant[0].to_expr();
        }

        let exprs: Vec<_> = implicant.iter().map(|l| l.to_expr()).collect();
        let mut result = exprs[0].clone();
        for e in exprs.iter().skip(1) {
            result = ChcExpr::and(result, e.clone());
        }
        result
    }

    /// Simplify an expression
    fn simplify(&self, expr: &ChcExpr) -> ChcExpr {
        match expr {
            ChcExpr::Op(ChcOp::Le, args) if args.len() == 2 => {
                // x <= x is always true
                if args[0].as_ref() == args[1].as_ref() {
                    return ChcExpr::Bool(true);
                }
                if let (ChcExpr::Int(a), ChcExpr::Int(b)) = (args[0].as_ref(), args[1].as_ref()) {
                    return ChcExpr::Bool(*a <= *b);
                }
            }
            ChcExpr::Op(ChcOp::Lt, args) if args.len() == 2 => {
                // x < x is always false
                if args[0].as_ref() == args[1].as_ref() {
                    return ChcExpr::Bool(false);
                }
                if let (ChcExpr::Int(a), ChcExpr::Int(b)) = (args[0].as_ref(), args[1].as_ref()) {
                    return ChcExpr::Bool(*a < *b);
                }
            }
            ChcExpr::Op(ChcOp::Ge, args) if args.len() == 2 => {
                // x >= x is always true
                if args[0].as_ref() == args[1].as_ref() {
                    return ChcExpr::Bool(true);
                }
                if let (ChcExpr::Int(a), ChcExpr::Int(b)) = (args[0].as_ref(), args[1].as_ref()) {
                    return ChcExpr::Bool(*a >= *b);
                }
            }
            ChcExpr::Op(ChcOp::Gt, args) if args.len() == 2 => {
                // x > x is always false
                if args[0].as_ref() == args[1].as_ref() {
                    return ChcExpr::Bool(false);
                }
                if let (ChcExpr::Int(a), ChcExpr::Int(b)) = (args[0].as_ref(), args[1].as_ref()) {
                    return ChcExpr::Bool(*a > *b);
                }
            }
            ChcExpr::Op(ChcOp::Eq, args) if args.len() == 2 => {
                // Structurally identical expressions are equal
                if args[0].as_ref() == args[1].as_ref() {
                    return ChcExpr::Bool(true);
                }
                if let (ChcExpr::Int(a), ChcExpr::Int(b)) = (args[0].as_ref(), args[1].as_ref()) {
                    return ChcExpr::Bool(*a == *b);
                }
            }
            ChcExpr::Op(ChcOp::Add, args) if args.len() == 2 => {
                if let ChcExpr::Int(0) = args[0].as_ref() {
                    return args[1].as_ref().clone();
                }
                if let ChcExpr::Int(0) = args[1].as_ref() {
                    return args[0].as_ref().clone();
                }
                if let (ChcExpr::Int(a), ChcExpr::Int(b)) = (args[0].as_ref(), args[1].as_ref()) {
                    return ChcExpr::Int(a + b);
                }
            }
            ChcExpr::Op(ChcOp::Sub, args) if args.len() == 2 => {
                if let ChcExpr::Int(0) = args[1].as_ref() {
                    return args[0].as_ref().clone();
                }
                if let (ChcExpr::Int(a), ChcExpr::Int(b)) = (args[0].as_ref(), args[1].as_ref()) {
                    return ChcExpr::Int(a - b);
                }
            }
            ChcExpr::Op(ChcOp::Mul, args) if args.len() == 2 => {
                if let ChcExpr::Int(1) = args[0].as_ref() {
                    return args[1].as_ref().clone();
                }
                if let ChcExpr::Int(1) = args[1].as_ref() {
                    return args[0].as_ref().clone();
                }
                if let ChcExpr::Int(0) = args[0].as_ref() {
                    return ChcExpr::Int(0);
                }
                if let ChcExpr::Int(0) = args[1].as_ref() {
                    return ChcExpr::Int(0);
                }
                if let (ChcExpr::Int(a), ChcExpr::Int(b)) = (args[0].as_ref(), args[1].as_ref()) {
                    return ChcExpr::Int(a * b);
                }
            }
            ChcExpr::Op(ChcOp::Neg, args) if args.len() == 1 => {
                if let ChcExpr::Int(n) = args[0].as_ref() {
                    return ChcExpr::Int(-n);
                }
            }
            ChcExpr::Op(ChcOp::Not, args) if args.len() == 1 => {
                if let ChcExpr::Bool(b) = args[0].as_ref() {
                    return ChcExpr::Bool(!b);
                }
            }
            ChcExpr::Op(ChcOp::And, args) => {
                let mut simplified: Vec<ChcExpr> = Vec::new();
                for arg in args {
                    let s = self.simplify(arg);
                    match s {
                        ChcExpr::Bool(true) => continue,
                        ChcExpr::Bool(false) => return ChcExpr::Bool(false),
                        _ => simplified.push(s),
                    }
                }
                return match simplified.len() {
                    0 => ChcExpr::Bool(true),
                    1 => simplified.pop().unwrap(),
                    _ => {
                        let arcs: Vec<_> = simplified.into_iter().map(Arc::new).collect();
                        ChcExpr::Op(ChcOp::And, arcs)
                    }
                };
            }
            ChcExpr::Op(ChcOp::Or, args) => {
                let mut simplified: Vec<ChcExpr> = Vec::new();
                for arg in args {
                    let s = self.simplify(arg);
                    match s {
                        ChcExpr::Bool(false) => continue,
                        ChcExpr::Bool(true) => return ChcExpr::Bool(true),
                        _ => simplified.push(s),
                    }
                }
                return match simplified.len() {
                    0 => ChcExpr::Bool(false),
                    1 => simplified.pop().unwrap(),
                    _ => {
                        let arcs: Vec<_> = simplified.into_iter().map(Arc::new).collect();
                        ChcExpr::Op(ChcOp::Or, arcs)
                    }
                };
            }
            _ => {}
        }
        expr.clone()
    }
}

/// Classification of bounds for variable elimination
enum BoundKind {
    /// Lower bound: coeff * var >= term (strict if bool is true)
    Lower(i64, ChcExpr, bool),
    /// Upper bound: coeff * var <= term (strict if bool is true)
    Upper(i64, ChcExpr, bool),
    /// Equality: coeff * var = term
    Equality(i64, ChcExpr),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mbp_boolean_projection() {
        let mbp = Mbp::new();

        // Formula: x = 1 AND b
        let x = ChcVar::new("x", ChcSort::Int);
        let b = ChcVar::new("b", ChcSort::Bool);

        let formula = ChcExpr::and(
            ChcExpr::eq(ChcExpr::var(x.clone()), ChcExpr::Int(1)),
            ChcExpr::var(b.clone()),
        );

        let mut model = FxHashMap::default();
        model.insert("x".to_string(), SmtValue::Int(1));
        model.insert("b".to_string(), SmtValue::Bool(true));

        // Project out b - should get x = 1
        let result = mbp.project(&formula, &[b], &model);
        assert!(matches!(result, ChcExpr::Op(ChcOp::Eq, _)));
    }

    #[test]
    fn test_mbp_integer_equality() {
        let mbp = Mbp::new();

        // Formula: x = y + 1 AND x > 0
        let x = ChcVar::new("x", ChcSort::Int);
        let y = ChcVar::new("y", ChcSort::Int);

        let formula = ChcExpr::and(
            ChcExpr::eq(
                ChcExpr::var(x.clone()),
                ChcExpr::add(ChcExpr::var(y.clone()), ChcExpr::Int(1)),
            ),
            ChcExpr::gt(ChcExpr::var(x.clone()), ChcExpr::Int(0)),
        );

        let mut model = FxHashMap::default();
        model.insert("x".to_string(), SmtValue::Int(5));
        model.insert("y".to_string(), SmtValue::Int(4));

        // Project out x - should get constraints on y
        let result = mbp.project(&formula, &[x], &model);

        // Result should not contain x
        let vars = result.vars();
        assert!(!vars.iter().any(|v| v.name == "x"));
    }

    #[test]
    fn test_mbp_bounds_resolution() {
        let mbp = Mbp::new();

        // Formula: x >= 0 AND x <= 10 AND y = x + 1
        let x = ChcVar::new("x", ChcSort::Int);
        let y = ChcVar::new("y", ChcSort::Int);

        let formula = ChcExpr::and(
            ChcExpr::and(
                ChcExpr::ge(ChcExpr::var(x.clone()), ChcExpr::Int(0)),
                ChcExpr::le(ChcExpr::var(x.clone()), ChcExpr::Int(10)),
            ),
            ChcExpr::eq(
                ChcExpr::var(y.clone()),
                ChcExpr::add(ChcExpr::var(x.clone()), ChcExpr::Int(1)),
            ),
        );

        let mut model = FxHashMap::default();
        model.insert("x".to_string(), SmtValue::Int(5));
        model.insert("y".to_string(), SmtValue::Int(6));

        // Project out x
        let result = mbp.project(&formula, &[x], &model);

        // Result should constrain y based on the elimination
        let vars = result.vars();
        assert!(!vars.iter().any(|v| v.name == "x"));
    }

    #[test]
    fn test_implicant_extraction() {
        let mbp = Mbp::new();

        // Formula: (a AND b) OR c
        let a = ChcVar::new("a", ChcSort::Bool);
        let b = ChcVar::new("b", ChcSort::Bool);
        let _c = ChcVar::new("c", ChcSort::Bool);

        let formula = ChcExpr::or(
            ChcExpr::and(ChcExpr::var(a.clone()), ChcExpr::var(b.clone())),
            ChcExpr::var(_c.clone()),
        );

        let mut model = FxHashMap::default();
        model.insert("a".to_string(), SmtValue::Bool(true));
        model.insert("b".to_string(), SmtValue::Bool(true));
        model.insert("c".to_string(), SmtValue::Bool(false));

        let implicant = mbp.get_implicant(&formula, &model);

        // Should pick (a AND b) branch since a=true, b=true
        assert_eq!(implicant.len(), 2);
    }

    #[test]
    fn test_simplify() {
        let mbp = Mbp::new();

        // 0 + x = x
        let x = ChcVar::new("x", ChcSort::Int);
        let expr = ChcExpr::add(ChcExpr::Int(0), ChcExpr::var(x.clone()));
        let result = mbp.simplify(&expr);
        assert!(matches!(result, ChcExpr::Var(_)));

        // 5 <= 10 = true
        let expr2 = ChcExpr::le(ChcExpr::Int(5), ChcExpr::Int(10));
        let result2 = mbp.simplify(&expr2);
        assert_eq!(result2, ChcExpr::Bool(true));

        // true AND x = x (where x is boolean)
        let b = ChcVar::new("b", ChcSort::Bool);
        let expr3 = ChcExpr::and(ChcExpr::Bool(true), ChcExpr::var(b.clone()));
        let result3 = mbp.simplify(&expr3);
        assert!(matches!(result3, ChcExpr::Var(_)));
    }
}
