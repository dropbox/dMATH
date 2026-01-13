//! Without loss of generality and assertion utility tactics
//!
//! This module provides tactics for:
//! - `suffices_to_show`: Alias for suffices with clearer semantics
//! - `wlog`: Without loss of generality transformations
//! - `push_neg_at`: Push negations through expressions
//! - `norm_num_at`: Normalize numerals in hypotheses

use crate::tactic::{
    extract_nat_literal, suffices_tactic, Goal, LocalDecl, ProofState, TacticError, TacticResult,
};
use lean5_kernel::name::Name;
use lean5_kernel::Expr;

// ============================================================================
// Additional utility tactics
// ============================================================================

/// `suffices_to_show` - alias for suffices with clearer semantics
pub fn suffices_to_show(state: &mut ProofState, prop: Expr, cont: Option<Expr>) -> TacticResult {
    suffices_tactic(state, "this".to_string(), prop, cont)
}

/// `wlog` - without loss of generality
///
/// Transforms goal by assuming a symmetric condition holds, creating subgoals
/// to prove the symmetry and the goal under the assumption.
pub fn wlog(state: &mut ProofState, assumption_name: String, assumption: Expr) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
    let target = goal.target.clone();

    // Create two subgoals:
    // 1. Prove assumption → target (the main case)
    // 2. Prove ¬assumption → target (the symmetric case, typically dischargeable)

    // Goal 1: assumption → target
    let impl_target = Expr::arrow(assumption.clone(), target.clone());

    // Goal 2: Symmetry - prove we can reduce to the assumed case
    // For now, just require proving the goal directly in the other case
    let neg_assumption = Expr::app(
        Expr::const_(Name::from_string("Not"), vec![]),
        assumption.clone(),
    );
    let neg_impl_target = Expr::arrow(neg_assumption, target);

    state.goals.remove(0);

    // Add both goals
    let meta2 = state.metas.fresh(neg_impl_target.clone());
    state.goals.insert(
        0,
        Goal {
            meta_id: meta2,
            target: neg_impl_target,
            local_ctx: goal.local_ctx.clone(),
        },
    );

    let meta1 = state.metas.fresh(impl_target.clone());
    let mut ctx1 = goal.local_ctx.clone();
    // Add assumption to context for first goal
    let fvar = state.fresh_fvar();
    ctx1.push(LocalDecl {
        fvar,
        name: assumption_name,
        ty: assumption,
        value: None,
    });
    state.goals.insert(
        0,
        Goal {
            meta_id: meta1,
            target: impl_target,
            local_ctx: ctx1,
        },
    );

    Ok(())
}

/// `push_neg_at` - push negations at a specific hypothesis
pub fn push_neg_at(state: &mut ProofState, hyp_name: &str) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?;

    // Find hypothesis
    let hyp_idx = goal
        .local_ctx
        .iter()
        .position(|d| d.name == hyp_name)
        .ok_or_else(|| TacticError::HypothesisNotFound(hyp_name.to_string()))?;

    let hyp_ty = goal.local_ctx[hyp_idx].ty.clone();

    // Push negations in the type
    let new_ty = push_negations_in_expr(&hyp_ty);

    // Update the hypothesis type
    let goal = state.goals.get_mut(0).ok_or(TacticError::NoGoals)?;
    goal.local_ctx[hyp_idx].ty = new_ty;

    Ok(())
}

/// Push negations through an expression
pub(crate) fn push_negations_in_expr(expr: &Expr) -> Expr {
    let head = expr.get_app_fn();
    let args: Vec<&Expr> = expr.get_app_args();

    if let Expr::Const(name, _) = head {
        let name_str = name.to_string();

        // ¬¬P → P
        if name_str == "Not" && args.len() == 1 {
            let inner = args[0];
            let inner_head = inner.get_app_fn();
            if let Expr::Const(inner_name, _) = inner_head {
                if inner_name.to_string() == "Not" {
                    let inner_args: Vec<&Expr> = inner.get_app_args();
                    if inner_args.len() == 1 {
                        return push_negations_in_expr(inner_args[0]);
                    }
                }
            }
        }

        // ¬(P ∧ Q) → ¬P ∨ ¬Q
        if name_str == "Not" && args.len() == 1 {
            let inner = args[0];
            let inner_head = inner.get_app_fn();
            if let Expr::Const(inner_name, _) = inner_head {
                if inner_name.to_string() == "And" {
                    let inner_args: Vec<&Expr> = inner.get_app_args();
                    if inner_args.len() == 2 {
                        let neg_p = Expr::app(
                            Expr::const_(Name::from_string("Not"), vec![]),
                            push_negations_in_expr(inner_args[0]),
                        );
                        let neg_q = Expr::app(
                            Expr::const_(Name::from_string("Not"), vec![]),
                            push_negations_in_expr(inner_args[1]),
                        );
                        return Expr::app(
                            Expr::app(Expr::const_(Name::from_string("Or"), vec![]), neg_p),
                            neg_q,
                        );
                    }
                }
            }
        }

        // ¬(P ∨ Q) → ¬P ∧ ¬Q
        if name_str == "Not" && args.len() == 1 {
            let inner = args[0];
            let inner_head = inner.get_app_fn();
            if let Expr::Const(inner_name, _) = inner_head {
                if inner_name.to_string() == "Or" {
                    let inner_args: Vec<&Expr> = inner.get_app_args();
                    if inner_args.len() == 2 {
                        let neg_p = Expr::app(
                            Expr::const_(Name::from_string("Not"), vec![]),
                            push_negations_in_expr(inner_args[0]),
                        );
                        let neg_q = Expr::app(
                            Expr::const_(Name::from_string("Not"), vec![]),
                            push_negations_in_expr(inner_args[1]),
                        );
                        return Expr::app(
                            Expr::app(Expr::const_(Name::from_string("And"), vec![]), neg_p),
                            neg_q,
                        );
                    }
                }
            }
        }
    }

    // Recurse into structure
    match expr {
        Expr::App(f, a) => Expr::app(push_negations_in_expr(f), push_negations_in_expr(a)),
        Expr::Lam(bi, ty, body) => Expr::lam(
            *bi,
            push_negations_in_expr(ty),
            push_negations_in_expr(body),
        ),
        Expr::Pi(bi, ty, body) => Expr::pi(
            *bi,
            push_negations_in_expr(ty),
            push_negations_in_expr(body),
        ),
        _ => expr.clone(),
    }
}

/// `norm_num_at` - normalize numerals at a hypothesis
pub fn norm_num_at(state: &mut ProofState, hyp_name: &str) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?;

    let hyp_idx = goal
        .local_ctx
        .iter()
        .position(|d| d.name == hyp_name)
        .ok_or_else(|| TacticError::HypothesisNotFound(hyp_name.to_string()))?;

    let hyp_ty = goal.local_ctx[hyp_idx].ty.clone();

    // Normalize numerals in the hypothesis type
    let new_ty = normalize_numerals(&hyp_ty);

    let goal = state.goals.get_mut(0).ok_or(TacticError::NoGoals)?;
    goal.local_ctx[hyp_idx].ty = new_ty;

    Ok(())
}

/// Normalize numeral expressions
pub(crate) fn normalize_numerals(expr: &Expr) -> Expr {
    match expr {
        // Try to evaluate arithmetic
        Expr::App(f, arg) => {
            let f_norm = normalize_numerals(f);
            let arg_norm = normalize_numerals(arg);

            // Check for binary operations on literals
            if let Expr::App(f2, arg1) = &f_norm {
                if let Expr::Const(op, _) = f2.as_ref() {
                    let op_str = op.to_string();
                    if let (Some(n1), Some(n2)) =
                        (extract_nat_literal(arg1), extract_nat_literal(&arg_norm))
                    {
                        if op_str.contains("add") || op_str.contains("Add") {
                            return Expr::Lit(lean5_kernel::expr::Literal::Nat((n1 + n2) as u64));
                        }
                        if op_str.contains("mul") || op_str.contains("Mul") {
                            return Expr::Lit(lean5_kernel::expr::Literal::Nat((n1 * n2) as u64));
                        }
                        if op_str.contains("sub") || op_str.contains("Sub") {
                            return Expr::Lit(lean5_kernel::expr::Literal::Nat(
                                n1.saturating_sub(n2) as u64,
                            ));
                        }
                    }
                }
            }

            Expr::app(f_norm, arg_norm)
        }
        Expr::Lam(bi, ty, body) => Expr::lam(*bi, normalize_numerals(ty), normalize_numerals(body)),
        Expr::Pi(bi, ty, body) => Expr::pi(*bi, normalize_numerals(ty), normalize_numerals(body)),
        _ => expr.clone(),
    }
}
