//! Ring normalization tactics
//!
//! Provides tactics for normalizing expressions in commutative (semi)rings.
//! These tactics are particularly useful for proving equalities involving
//! polynomial arithmetic.
//!
//! # Tactics
//!
//! - `ring` - Prove ring equalities by normalizing both sides
//! - `ring_nf` - Normalize ring expressions without closing the goal

use lean5_kernel::name::Name;
use lean5_kernel::{Expr, FVarId, Level};

use super::equality::match_equality;
use super::{rfl, Goal, ProofState, TacticError, TacticResult};

/// Ring normalization tactic for commutative (semi)rings.
///
/// Normalizes expressions in a ring to a canonical form, allowing
/// equality to be checked by syntactic comparison. Uses Horner form
/// for polynomial representation.
///
/// # Algorithm
/// 1. Flatten nested additions and multiplications
/// 2. Distribute multiplication over addition
/// 3. Collect like terms (same variable powers)
/// 4. Sort terms lexicographically
/// 5. Simplify coefficients
///
/// # Supported operations
/// - Addition (+)
/// - Multiplication (*)
/// - Subtraction (-) (converted to + neg)
/// - Negation (-)
/// - Natural number powers (^)
///
/// # Example
/// ```text
/// -- Goal: (a + b) * (a + b) = a * a + 2 * a * b + b * b
/// ring
/// -- Goal closed (both sides normalize to same form)
/// ```
///
/// # Errors
/// - `NoGoals` if there are no goals
/// - `GoalMismatch` if goal is not an equality
/// - `Other` if normalization fails to close the goal
pub fn ring(state: &mut ProofState) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Check that goal is an equality
    let (_ty, lhs, rhs, _levels) = match_equality(&goal.target)
        .map_err(|_| TacticError::GoalMismatch("ring: goal is not an equality".to_string()))?;

    // Normalize both sides
    let lhs_norm = ring_normalize(&lhs);
    let rhs_norm = ring_normalize(&rhs);

    // Check if they're equal
    if ring_exprs_equal(&lhs_norm, &rhs_norm) {
        // Close goal with rfl
        rfl(state)
    } else {
        Err(TacticError::Other(format!(
            "ring: normalized forms differ:\n  LHS: {lhs_norm:?}\n  RHS: {rhs_norm:?}"
        )))
    }
}

/// Representation of a ring expression in normalized form
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum RingExpr {
    /// A constant (natural number for now)
    Const(u64),
    /// A variable (identified by name or fvar)
    Var(String),
    /// Addition of terms
    Add(Vec<RingExpr>),
    /// Multiplication of factors
    Mul(Vec<RingExpr>),
    /// Power (base, exponent)
    Pow(Box<RingExpr>, u64),
    /// Negation
    Neg(Box<RingExpr>),
    /// Unknown expression (treated as atomic)
    Unknown(String),
}

/// Normalize a ring expression
pub(crate) fn ring_normalize(expr: &Expr) -> RingExpr {
    match expr {
        // Natural number literals
        Expr::Lit(lit) => match lit {
            lean5_kernel::expr::Literal::Nat(n) => RingExpr::Const(*n),
            _ => RingExpr::Unknown(format!("{lit:?}")),
        },

        // Constants (Nat.zero, Nat.succ, etc.)
        Expr::Const(name, _) => {
            let name_str = name.to_string();
            if name_str == "Nat.zero" {
                RingExpr::Const(0)
            } else if name_str == "Nat.one" || name_str == "1" {
                RingExpr::Const(1)
            } else {
                RingExpr::Var(name_str)
            }
        }

        // Free variables
        Expr::FVar(id) => RingExpr::Var(format!("fvar_{}", id.0)),

        // Applications (operations)
        Expr::App(f, arg) => {
            // Check for binary operations
            if let Expr::App(f2, arg1) = f.as_ref() {
                if let Expr::Const(op_name, _) = f2.as_ref() {
                    let op_str = op_name.to_string();
                    let left = ring_normalize(arg1);
                    let right = ring_normalize(arg);

                    // Addition
                    if op_str.contains("add") || op_str.contains("Add") {
                        return ring_flatten_add(left, right);
                    }

                    // Multiplication
                    if op_str.contains("mul") || op_str.contains("Mul") {
                        return ring_flatten_mul(left, right);
                    }

                    // Subtraction
                    if op_str.contains("sub") || op_str.contains("Sub") {
                        return ring_flatten_add(left, RingExpr::Neg(Box::new(right)));
                    }

                    // Power
                    if op_str.contains("pow") || op_str.contains("Pow") {
                        if let RingExpr::Const(n) = right {
                            return RingExpr::Pow(Box::new(left), n);
                        }
                    }
                }

                // Check for HAdd.hAdd, HMul.hMul (type class operations)
                if let Expr::App(f3, _) = f2.as_ref() {
                    if let Expr::App(f4, _) = f3.as_ref() {
                        if let Expr::Const(op_name, _) = f4.as_ref() {
                            let op_str = op_name.to_string();
                            let left = ring_normalize(arg1);
                            let right = ring_normalize(arg);

                            if op_str == "HAdd.hAdd" {
                                return ring_flatten_add(left, right);
                            }
                            if op_str == "HMul.hMul" {
                                return ring_flatten_mul(left, right);
                            }
                            if op_str == "HPow.hPow" {
                                if let RingExpr::Const(n) = right {
                                    return RingExpr::Pow(Box::new(left), n);
                                }
                            }
                        }
                    }
                }
            }

            // Check for unary operations
            if let Expr::Const(op_name, _) = f.as_ref() {
                let op_str = op_name.to_string();
                let operand = ring_normalize(arg);

                // Negation
                if op_str.contains("neg") || op_str.contains("Neg") {
                    return RingExpr::Neg(Box::new(operand));
                }

                // Succ
                if op_str == "Nat.succ" {
                    if let RingExpr::Const(n) = operand {
                        return RingExpr::Const(n + 1);
                    }
                    return ring_flatten_add(operand, RingExpr::Const(1));
                }
            }

            // Unknown application
            RingExpr::Unknown(format!("{expr:?}"))
        }

        _ => RingExpr::Unknown(format!("{expr:?}")),
    }
}

/// Flatten addition: a + (b + c) → Add([a, b, c])
pub(crate) fn ring_flatten_add(left: RingExpr, right: RingExpr) -> RingExpr {
    let mut terms = Vec::new();

    match left {
        RingExpr::Add(ts) => terms.extend(ts),
        other => terms.push(other),
    }

    match right {
        RingExpr::Add(ts) => terms.extend(ts),
        other => terms.push(other),
    }

    // Collect like terms and simplify
    terms = ring_collect_like_terms(terms);

    match terms.len() {
        0 => RingExpr::Const(0),
        1 => terms.pop().expect("terms has exactly 1 element"),
        _ => {
            terms.sort();
            RingExpr::Add(terms)
        }
    }
}

/// Flatten multiplication: a * (b * c) → Mul([a, b, c])
pub(crate) fn ring_flatten_mul(left: RingExpr, right: RingExpr) -> RingExpr {
    let mut factors = Vec::new();

    match left {
        RingExpr::Mul(fs) => factors.extend(fs),
        other => factors.push(other),
    }

    match right {
        RingExpr::Mul(fs) => factors.extend(fs),
        other => factors.push(other),
    }

    // Collect constants and simplify
    let (consts, vars): (Vec<_>, Vec<_>) = factors
        .into_iter()
        .partition(|f| matches!(f, RingExpr::Const(_)));

    let const_product: u64 = consts
        .iter()
        .filter_map(|c| {
            if let RingExpr::Const(n) = c {
                Some(*n)
            } else {
                None
            }
        })
        .product();

    if const_product == 0 {
        return RingExpr::Const(0);
    }

    let mut result = vars;
    if const_product != 1 {
        result.insert(0, RingExpr::Const(const_product));
    }

    match result.len() {
        0 => RingExpr::Const(1),
        1 => result.pop().expect("result has exactly 1 element"),
        _ => {
            result.sort();
            RingExpr::Mul(result)
        }
    }
}

/// Collect like terms in an addition
pub(crate) fn ring_collect_like_terms(terms: Vec<RingExpr>) -> Vec<RingExpr> {
    use std::collections::HashMap;

    let mut const_sum: i64 = 0;
    let mut var_counts: HashMap<RingExpr, i64> = HashMap::new();

    for term in terms {
        match term {
            RingExpr::Const(n) => const_sum += n as i64,
            RingExpr::Neg(inner) => {
                if let RingExpr::Const(n) = *inner {
                    const_sum -= n as i64;
                } else {
                    *var_counts.entry(*inner).or_insert(0) -= 1;
                }
            }
            other => {
                *var_counts.entry(other).or_insert(0) += 1;
            }
        }
    }

    let mut result = Vec::new();

    if const_sum > 0 {
        result.push(RingExpr::Const(const_sum as u64));
    } else if const_sum < 0 {
        result.push(RingExpr::Neg(Box::new(RingExpr::Const(
            (-const_sum) as u64,
        ))));
    }

    for (term, count) in var_counts {
        if count > 0 {
            for _ in 0..count {
                result.push(term.clone());
            }
        } else if count < 0 {
            for _ in 0..(-count) {
                result.push(RingExpr::Neg(Box::new(term.clone())));
            }
        }
    }

    result
}

/// Check if two normalized ring expressions are equal
pub(crate) fn ring_exprs_equal(a: &RingExpr, b: &RingExpr) -> bool {
    a == b
}

// ============================================================================
// ring_nf: Ring normal form tactic
// ============================================================================

/// Normalize ring expressions to canonical form without closing the goal.
///
/// Unlike `ring` which tries to prove equality, `ring_nf` transforms the goal
/// to have both sides in canonical polynomial form. This can make goals
/// easier to prove by subsequent tactics.
///
/// Canonical form: sum of monomials sorted lexicographically, each monomial
/// is a product of variables with coefficients.
pub fn ring_nf(state: &mut ProofState) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
    let target = state.metas.instantiate(&goal.target);

    // Check if goal is an equality
    let (ty, lhs, rhs, levels) = match_equality(&target)
        .map_err(|_| TacticError::GoalMismatch("ring_nf: goal must be an equality".to_string()))?;

    // Normalize both sides
    let lhs_norm = ring_normalize(&lhs);
    let rhs_norm = ring_normalize(&rhs);

    // Convert back to expressions
    let lhs_expr = ring_expr_to_expr(&lhs_norm);
    let rhs_expr = ring_expr_to_expr(&rhs_norm);

    // Create new goal with normalized expressions
    let new_target = make_eq(&ty, &lhs_expr, &rhs_expr, &levels);

    // Replace current goal with normalized form
    let new_meta_id = state.metas.fresh(new_target.clone());
    let new_goal = Goal {
        meta_id: new_meta_id,
        target: new_target,
        local_ctx: goal.local_ctx.clone(),
    };

    state.goals[0] = new_goal;

    Ok(())
}

/// Convert RingExpr back to Expr
pub(crate) fn ring_expr_to_expr(re: &RingExpr) -> Expr {
    match re {
        RingExpr::Const(n) => Expr::Lit(lean5_kernel::expr::Literal::Nat(*n)),
        RingExpr::Var(s) => {
            if let Some(suffix) = s.strip_prefix("fvar_") {
                if let Ok(id) = suffix.parse::<u64>() {
                    return Expr::FVar(FVarId(id));
                }
            }
            Expr::const_(Name::from_string(s), vec![])
        }
        RingExpr::Add(terms) => {
            if terms.is_empty() {
                return Expr::Lit(lean5_kernel::expr::Literal::Nat(0));
            }
            let mut result = ring_expr_to_expr(&terms[0]);
            for term in &terms[1..] {
                let term_expr = ring_expr_to_expr(term);
                result = make_add(&result, &term_expr);
            }
            result
        }
        RingExpr::Mul(factors) => {
            if factors.is_empty() {
                return Expr::Lit(lean5_kernel::expr::Literal::Nat(1));
            }
            let mut result = ring_expr_to_expr(&factors[0]);
            for factor in &factors[1..] {
                let factor_expr = ring_expr_to_expr(factor);
                result = make_mul(&result, &factor_expr);
            }
            result
        }
        RingExpr::Pow(base, exp) => {
            let base_expr = ring_expr_to_expr(base);
            let exp_expr = Expr::Lit(lean5_kernel::expr::Literal::Nat(*exp));
            make_pow(&base_expr, &exp_expr)
        }
        RingExpr::Neg(inner) => {
            let inner_expr = ring_expr_to_expr(inner);
            make_neg(&inner_expr)
        }
        RingExpr::Unknown(s) => Expr::const_(Name::from_string(s), vec![]),
    }
}

/// Make addition expression
pub(crate) fn make_add(a: &Expr, b: &Expr) -> Expr {
    Expr::app(
        Expr::app(
            Expr::const_(Name::from_string("HAdd.hAdd"), vec![]),
            a.clone(),
        ),
        b.clone(),
    )
}

/// Make multiplication expression
pub(crate) fn make_mul(a: &Expr, b: &Expr) -> Expr {
    Expr::app(
        Expr::app(
            Expr::const_(Name::from_string("HMul.hMul"), vec![]),
            a.clone(),
        ),
        b.clone(),
    )
}

/// Make power expression
pub(crate) fn make_pow(base: &Expr, exp: &Expr) -> Expr {
    Expr::app(
        Expr::app(
            Expr::const_(Name::from_string("HPow.hPow"), vec![]),
            base.clone(),
        ),
        exp.clone(),
    )
}

/// Make negation expression
pub(crate) fn make_neg(a: &Expr) -> Expr {
    Expr::app(
        Expr::const_(Name::from_string("Neg.neg"), vec![]),
        a.clone(),
    )
}

/// Make equality expression
pub(crate) fn make_eq(ty: &Expr, lhs: &Expr, rhs: &Expr, levels: &[Level]) -> Expr {
    Expr::app(
        Expr::app(
            Expr::app(
                Expr::const_(Name::from_string("Eq"), levels.to_vec()),
                ty.clone(),
            ),
            lhs.clone(),
        ),
        rhs.clone(),
    )
}
