//! Decidable equality tactics
//!
//! Tactics for proving goals involving decidable equality, such as
//! `Decidable (a = b)` or direct equality goals `a = b` for types
//! with `DecidableEq` instances.

use lean5_kernel::name::Name;
use lean5_kernel::{Environment, Expr, TypeChecker};

use crate::tactic::convert::make_eq_refl;
use crate::tactic::equality::match_equality;
use crate::tactic::{rfl, Goal, ProofState, TacticError, TacticResult};

/// Prove goals of the form `Decidable (a = b)` or close `a = b` when decidable.
///
/// This tactic handles decidable equality in two ways:
/// 1. If goal is `Decidable (a = b)`, construct a `Decidable` instance
/// 2. If goal is `a = b` where the type has decidable equality, use `decide`
///
/// Works for types with `DecidableEq` instances (Nat, Bool, Fin, etc.)
pub fn decide_eq(state: &mut ProofState) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
    let target = state.metas.instantiate(&goal.target);

    // Check if goal is Decidable (a = b)
    if let Some((eq_ty, lhs, rhs)) = match_decidable_eq(&target) {
        // Build DecidableEq instance check
        return decide_eq_check(state, &goal, &eq_ty, &lhs, &rhs);
    }

    // Check if goal is an equality a = b with decidable type
    if let Ok((ty, lhs, rhs, _levels)) = match_equality(&target) {
        // Try to evaluate equality decision
        return decide_eq_equality(state, &goal, &ty, &lhs, &rhs);
    }

    Err(TacticError::GoalMismatch(
        "decide_eq: goal must be `Decidable (a = b)` or an equality with decidable type"
            .to_string(),
    ))
}

/// Match `Decidable (Eq α a b)` pattern
pub(crate) fn match_decidable_eq(expr: &Expr) -> Option<(Expr, Expr, Expr)> {
    let head = expr.get_app_fn();
    let args: Vec<&Expr> = expr.get_app_args();

    if let Expr::Const(name, _) = head {
        if name == &Name::from_string("Decidable") && args.len() == 1 {
            // The argument should be an equality
            if let Ok((ty, lhs, rhs, _)) = match_equality(args[0]) {
                return Some((ty, lhs, rhs));
            }
        }
    }
    None
}

/// Handle Decidable (a = b) goal
fn decide_eq_check(
    state: &mut ProofState,
    goal: &Goal,
    eq_ty: &Expr,
    lhs: &Expr,
    rhs: &Expr,
) -> TacticResult {
    // Check if lhs and rhs are definitionally equal
    let tc = TypeChecker::new(state.env());
    if tc.is_def_eq(lhs, rhs) {
        // They're equal, construct isTrue proof
        let eq_refl = make_eq_refl(eq_ty, lhs);
        let is_true = Expr::app(
            Expr::const_(Name::from_string("Decidable.isTrue"), vec![]),
            eq_refl,
        );
        state.metas.assign(goal.meta_id, is_true);
        state.goals.remove(0);
        return Ok(());
    }

    // Try to evaluate and check
    if decidable_type_check(state.env(), eq_ty) {
        // Type has decidable equality, check values
        if exprs_definitely_not_equal(lhs, rhs) {
            // They're definitely not equal, construct isFalse
            let ne_proof = make_ne_proof(eq_ty, lhs, rhs);
            let is_false = Expr::app(
                Expr::const_(Name::from_string("Decidable.isFalse"), vec![]),
                ne_proof,
            );
            state.metas.assign(goal.meta_id, is_false);
            state.goals.remove(0);
            return Ok(());
        }
    }

    Err(TacticError::Other(
        "decide_eq: cannot decide equality".to_string(),
    ))
}

/// Handle a = b goal with decidable type
fn decide_eq_equality(
    state: &mut ProofState,
    _goal: &Goal,
    ty: &Expr,
    lhs: &Expr,
    rhs: &Expr,
) -> TacticResult {
    // Check if type has decidable equality
    if !decidable_type_check(state.env(), ty) {
        return Err(TacticError::Other(format!(
            "decide_eq: type {ty:?} does not have decidable equality"
        )));
    }

    // Check if lhs and rhs are definitionally equal
    let tc = TypeChecker::new(state.env());
    if tc.is_def_eq(lhs, rhs) {
        // Close with rfl
        return rfl(state);
    }

    // Try to evaluate both sides to literals and compare
    if let (Some(l_val), Some(r_val)) = (eval_to_nat(lhs), eval_to_nat(rhs)) {
        if l_val == r_val {
            return rfl(state);
        }
        return Err(TacticError::Other(format!("decide_eq: {l_val} ≠ {r_val}")));
    }

    Err(TacticError::Other(
        "decide_eq: cannot evaluate equality".to_string(),
    ))
}

/// Check if a type has decidable equality
pub(crate) fn decidable_type_check(_env: &Environment, ty: &Expr) -> bool {
    let head = ty.get_app_fn();
    if let Expr::Const(name, _) = head {
        let name_str = name.to_string();
        // Types with decidable equality
        matches!(
            name_str.as_str(),
            "Nat"
                | "Bool"
                | "Int"
                | "Char"
                | "String"
                | "Fin"
                | "UInt8"
                | "UInt16"
                | "UInt32"
                | "UInt64"
                | "Unit"
                | "Empty"
        )
    } else {
        false
    }
}

/// Check if two expressions are definitely not equal (by structure)
pub(crate) fn exprs_definitely_not_equal(lhs: &Expr, rhs: &Expr) -> bool {
    // Check for different constructors
    match (lhs, rhs) {
        (Expr::Lit(l1), Expr::Lit(l2)) => l1 != l2,
        (Expr::Const(n1, _), Expr::Const(n2, _)) => {
            // Different constructors like Nat.zero vs Nat.succ
            let s1 = n1.to_string();
            let s2 = n2.to_string();
            (s1.contains("zero") && s2.contains("succ"))
                || (s1.contains("succ") && s2.contains("zero"))
                || (s1 == "Bool.true" && s2 == "Bool.false")
                || (s1 == "Bool.false" && s2 == "Bool.true")
        }
        _ => false,
    }
}

/// Evaluate expression to natural number if possible
pub(crate) fn eval_to_nat(expr: &Expr) -> Option<u64> {
    match expr {
        Expr::Lit(lean5_kernel::expr::Literal::Nat(n)) => Some(*n),
        Expr::Const(name, _) => {
            let s = name.to_string();
            if s == "Nat.zero" || s == "0" {
                Some(0)
            } else {
                None
            }
        }
        Expr::App(f, arg) => {
            if let Expr::Const(name, _) = f.as_ref() {
                if name.to_string() == "Nat.succ" {
                    eval_to_nat(arg).map(|n| n + 1)
                } else {
                    None
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Make proof of a ≠ b (placeholder - in practice would need actual proof)
pub(crate) fn make_ne_proof(_ty: &Expr, _lhs: &Expr, _rhs: &Expr) -> Expr {
    // In full implementation, would construct actual proof
    // For now, return a sorry-like term
    Expr::const_(Name::from_string("sorry"), vec![])
}
