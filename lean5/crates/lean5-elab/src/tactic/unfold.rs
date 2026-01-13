//! Definition unfolding tactics
//!
//! Provides tactics for unfolding definitions in goals and hypotheses.

use std::sync::Arc;

use lean5_kernel::name::Name;
use lean5_kernel::Expr;

use super::{ProofState, TacticError, TacticResult};

// ============================================================================
// Definition Tactics
// ============================================================================

/// Unfold a definition in the goal.
///
/// Replaces occurrences of a constant with its definition.
pub fn unfold(state: &mut ProofState, name: &str) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?;
    let target = goal.target.clone();

    // Try to find the definition in the environment
    let def_name = Name::from_string(name);
    let const_info = state
        .env
        .get_const(&def_name)
        .ok_or_else(|| TacticError::Other(format!("unfold: '{name}' is not a constant")))?;

    let def_val = const_info
        .value
        .as_ref()
        .ok_or_else(|| TacticError::Other(format!("unfold: '{name}' has no definition (axiom?)")))?
        .clone();

    // Substitute the definition for the constant in the target
    let unfolded = substitute_const(&target, &def_name, &def_val);

    if unfolded == target {
        return Err(TacticError::Other(format!(
            "unfold: '{name}' does not appear in the goal"
        )));
    }

    // Update the goal with the unfolded target
    state
        .current_goal_mut()
        .expect("must have at least one goal")
        .target = unfolded;
    Ok(())
}

/// Helper: substitute a constant with its definition in an expression
pub(crate) fn substitute_const(expr: &Expr, name: &Name, value: &Expr) -> Expr {
    match expr {
        Expr::Const(n, _) if n == name => value.clone(),
        Expr::App(f, arg) => Expr::app(
            substitute_const(f, name, value),
            substitute_const(arg, name, value),
        ),
        Expr::Lam(bi, ty, body) => Expr::Lam(
            *bi,
            Arc::new(substitute_const(ty, name, value)),
            Arc::new(substitute_const(body, name, value)),
        ),
        Expr::Pi(bi, ty, body) => Expr::Pi(
            *bi,
            Arc::new(substitute_const(ty, name, value)),
            Arc::new(substitute_const(body, name, value)),
        ),
        Expr::Let(ty, val, body) => Expr::Let(
            Arc::new(substitute_const(ty, name, value)),
            Arc::new(substitute_const(val, name, value)),
            Arc::new(substitute_const(body, name, value)),
        ),
        _ => expr.clone(),
    }
}

/// Unfold a definition in a hypothesis.
pub fn unfold_at(state: &mut ProofState, def_name: &str, hyp_name: &str) -> TacticResult {
    // First get the definition value from the environment
    let def_name_obj = Name::from_string(def_name);
    let const_info = state
        .env
        .get_const(&def_name_obj)
        .ok_or_else(|| TacticError::Other(format!("unfold_at: '{def_name}' is not a constant")))?;

    let def_val = const_info
        .value
        .as_ref()
        .ok_or_else(|| {
            TacticError::Other(format!(
                "unfold_at: '{def_name}' has no definition (axiom?)"
            ))
        })?
        .clone();

    let goal = state.current_goal_mut().ok_or(TacticError::NoGoals)?;

    // Find the hypothesis
    let hyp_idx = goal
        .local_ctx
        .iter()
        .position(|d| d.name == hyp_name)
        .ok_or_else(|| TacticError::HypothesisNotFound(hyp_name.to_string()))?;

    let old_ty = goal.local_ctx[hyp_idx].ty.clone();
    let new_ty = substitute_const(&old_ty, &def_name_obj, &def_val);

    if new_ty == old_ty {
        return Err(TacticError::Other(format!(
            "unfold_at: '{def_name}' does not appear in hypothesis '{hyp_name}'"
        )));
    }

    goal.local_ctx[hyp_idx].ty = new_ty;
    Ok(())
}

/// Delta-reduce (unfold) all definitions in the goal.
///
/// This iteratively unfolds definitions until no more can be unfolded.
pub fn delta(state: &mut ProofState) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?;
    let original_target = goal.target.clone();

    let mut target = original_target.clone();
    let mut changed = true;
    let mut iterations = 0;
    const MAX_ITERATIONS: usize = 100;

    while changed && iterations < MAX_ITERATIONS {
        changed = false;
        iterations += 1;

        // Collect all constants in the expression
        let consts = collect_consts(&target);

        for const_name in consts {
            if let Some(const_info) = state.env.get_const(&const_name) {
                if let Some(def_val) = &const_info.value {
                    let new_target = substitute_const(&target, &const_name, def_val);
                    if new_target != target {
                        target = new_target;
                        changed = true;
                        break; // Restart scan after substitution
                    }
                }
            }
        }
    }

    if target == original_target {
        return Err(TacticError::Other(
            "delta: no definitions to unfold".to_string(),
        ));
    }

    state
        .current_goal_mut()
        .expect("must have at least one goal")
        .target = target;
    Ok(())
}

/// Helper: collect all constant names in an expression
pub(crate) fn collect_consts(expr: &Expr) -> Vec<Name> {
    let mut consts = Vec::new();
    collect_consts_inner(expr, &mut consts);
    consts
}

fn collect_consts_inner(expr: &Expr, consts: &mut Vec<Name>) {
    match expr {
        Expr::Const(name, _) => {
            if !consts.contains(name) {
                consts.push(name.clone());
            }
        }
        Expr::App(f, arg) => {
            collect_consts_inner(f, consts);
            collect_consts_inner(arg, consts);
        }
        Expr::Lam(_, ty, body) | Expr::Pi(_, ty, body) => {
            collect_consts_inner(ty, consts);
            collect_consts_inner(body, consts);
        }
        Expr::Let(ty, val, body) => {
            collect_consts_inner(ty, consts);
            collect_consts_inner(val, consts);
            collect_consts_inner(body, consts);
        }
        _ => {}
    }
}
