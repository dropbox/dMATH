//! Hypothesis manipulation tactics
//!
//! This module provides tactics for manipulating hypotheses in the local context:
//! - `clear` - Remove a hypothesis
//! - `rename` - Rename a hypothesis
//! - `duplicate` - Duplicate a hypothesis with a new name
//! - `specialize` - Specialize a universally quantified hypothesis
//! - `clear_all_unused` - Clear all hypotheses not used in the goal
//! - `rename_all` - Rename multiple hypotheses at once
//! - `apply_fun` - Apply a function to both sides of an equality hypothesis
//! - `apply_fun_goal` - Apply a function to both sides of an equality goal
//! - `clear_except` - Clear all hypotheses except specified ones
//! - `replace` - Replace a hypothesis with a new one (creates proof obligation)
//! - `replace_hyp` - Replace a hypothesis with a new one (with explicit proof)

use lean5_kernel::name::Name;
use lean5_kernel::{Expr, FVarId, Level};

use super::{match_eq_simple, Goal, LocalDecl, ProofState, TacticError, TacticResult};
use crate::unify::MetaState;

// ============================================================================
// Clear Tactic
// ============================================================================

/// Removes a hypothesis from the local context.
///
/// # Arguments
/// * `state` - The proof state
/// * `hyp_name` - Name of the hypothesis to remove
pub fn clear(state: &mut ProofState, hyp_name: &str) -> TacticResult {
    let goal = state.current_goal_mut().ok_or(TacticError::NoGoals)?;

    let idx = goal
        .local_ctx
        .iter()
        .position(|decl| decl.name == hyp_name)
        .ok_or_else(|| TacticError::HypothesisNotFound(hyp_name.to_string()))?;

    goal.local_ctx.remove(idx);
    Ok(())
}

// ============================================================================
// Rename Tactic
// ============================================================================

/// Rename a hypothesis in the local context.
///
/// # Arguments
/// * `state` - The proof state
/// * `old_name` - Current name of the hypothesis
/// * `new_name` - New name to assign
pub fn rename(state: &mut ProofState, old_name: &str, new_name: &str) -> TacticResult {
    let goal = state.current_goal_mut().ok_or(TacticError::NoGoals)?;

    let decl = goal
        .local_ctx
        .iter_mut()
        .find(|d| d.name == old_name)
        .ok_or_else(|| TacticError::HypothesisNotFound(old_name.to_string()))?;

    decl.name = new_name.to_string();
    Ok(())
}

// ============================================================================
// Duplicate Tactic
// ============================================================================

/// Duplicate a hypothesis in the local context with a new name.
///
/// # Arguments
/// * `state` - The proof state
/// * `hyp_name` - Name of the hypothesis to duplicate
/// * `new_name` - Name for the duplicate
pub fn duplicate(state: &mut ProofState, hyp_name: &str, new_name: &str) -> TacticResult {
    let goal = state.current_goal_mut().ok_or(TacticError::NoGoals)?;

    let decl = goal
        .local_ctx
        .iter()
        .find(|d| d.name == hyp_name)
        .ok_or_else(|| TacticError::HypothesisNotFound(hyp_name.to_string()))?
        .clone();

    let new_decl = LocalDecl {
        name: new_name.to_string(),
        fvar: decl.fvar, // Share the same fvar (both refer to same value)
        ty: decl.ty,
        value: decl.value,
    };

    goal.local_ctx.push(new_decl);
    Ok(())
}

// ============================================================================
// Specialize Tactic
// ============================================================================

/// Specialize a hypothesis by applying it to an argument.
///
/// Given a hypothesis `h : ∀ x : A, P x` and a term `a : A`, this replaces
/// `h` with `h : P a`. The hypothesis is updated in-place with the specialized type.
///
/// # Arguments
/// * `state` - The proof state
/// * `hyp_name` - Name of the hypothesis to specialize (must have Pi type)
/// * `arg` - The argument to apply
///
/// # Example
/// If you have `h : ∀ n : Nat, n + 0 = n` and call `specialize(state, "h", Expr::nat_lit(5))`,
/// then `h` becomes `h : 5 + 0 = 5`.
pub fn specialize(state: &mut ProofState, hyp_name: &str, arg: Expr) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Find the hypothesis
    let (idx, decl) = goal
        .local_ctx
        .iter()
        .enumerate()
        .find(|(_, d)| d.name == hyp_name)
        .ok_or_else(|| TacticError::HypothesisNotFound(hyp_name.to_string()))?;
    let decl = decl.clone();

    // Get the type and normalize
    let hyp_ty = state.whnf(&goal, &decl.ty);

    // Check it's a Pi type
    match hyp_ty {
        Expr::Pi(_bi, domain, codomain) => {
            // Type check the argument
            let arg_ty = state.infer_type(&goal, &arg)?;

            // Check arg has the right type
            if !state.is_def_eq(&goal, &arg_ty, &domain) {
                return Err(TacticError::TypeMismatch {
                    expected: format!("{domain:?}"),
                    actual: format!("{arg_ty:?}"),
                });
            }

            // Compute the specialized type
            let specialized_ty = codomain.instantiate(&arg);

            // Create a fresh fvar for the specialized hypothesis
            let new_fvar = state.fresh_fvar();

            // Update the hypothesis with the specialized type
            let goal_mut = state.current_goal_mut().ok_or(TacticError::NoGoals)?;
            goal_mut.local_ctx[idx] = LocalDecl {
                fvar: new_fvar,
                name: hyp_name.to_string(),
                ty: specialized_ty,
                // The new value is the old hypothesis applied to the argument
                value: Some(Expr::app(Expr::fvar(decl.fvar), arg)),
            };

            Ok(())
        }
        _ => Err(TacticError::GoalMismatch(format!(
            "specialize: hypothesis '{hyp_name}' has type {hyp_ty:?}, expected ∀ or →"
        ))),
    }
}

// ============================================================================
// Clear All Unused Tactic
// ============================================================================

/// Clears all hypotheses from the local context that are not used in the goal.
/// This is useful for cleaning up the context after complex case splits.
///
/// # Example
/// ```text
/// -- h1 : P, h2 : Q, h3 : R ⊢ P
/// clear_all
/// -- h1 : P ⊢ P
/// ```
///
/// # Errors
/// - `NoGoals` if there are no goals
pub fn clear_all_unused(state: &mut ProofState) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Collect free variables used in the target
    let used_fvars = collect_fvars(&goal.target);

    // Also collect fvars used in types of hypotheses we keep
    let mut all_used: std::collections::HashSet<FVarId> = used_fvars.into_iter().collect();

    // Iteratively add dependencies
    loop {
        let mut added = false;
        for decl in &goal.local_ctx {
            if all_used.contains(&decl.fvar) {
                let ty_fvars = collect_fvars(&decl.ty);
                for fv in ty_fvars {
                    if !all_used.contains(&fv) {
                        all_used.insert(fv);
                        added = true;
                    }
                }
            }
        }
        if !added {
            break;
        }
    }

    // Remove unused hypotheses
    let goal = state.current_goal_mut().ok_or(TacticError::NoGoals)?;
    goal.local_ctx.retain(|decl| all_used.contains(&decl.fvar));

    Ok(())
}

/// Collect free variables from an expression
pub(crate) fn collect_fvars(expr: &Expr) -> Vec<FVarId> {
    let mut fvars = Vec::new();
    collect_fvars_rec(expr, &mut fvars);
    fvars
}

fn collect_fvars_rec(expr: &Expr, fvars: &mut Vec<FVarId>) {
    match expr {
        Expr::FVar(id) => {
            if !fvars.contains(id) {
                fvars.push(*id);
            }
        }
        Expr::App(f, a) => {
            collect_fvars_rec(f, fvars);
            collect_fvars_rec(a, fvars);
        }
        Expr::Lam(_, ty, body) | Expr::Pi(_, ty, body) => {
            collect_fvars_rec(ty, fvars);
            collect_fvars_rec(body, fvars);
        }
        Expr::Let(ty, val, body) => {
            collect_fvars_rec(ty, fvars);
            collect_fvars_rec(val, fvars);
            collect_fvars_rec(body, fvars);
        }
        _ => {}
    }
}

// ============================================================================
// Rename All Tactic
// ============================================================================

/// Tactic: rename_all
///
/// Renames multiple hypotheses at once according to a mapping.
///
/// # Example
/// ```text
/// -- h1 : P, h2 : Q ⊢ R
/// rename_all [h1 → hP, h2 → hQ]
/// -- hP : P, hQ : Q ⊢ R
/// ```
///
/// # Errors
/// - `NoGoals` if there are no goals
/// - `HypothesisNotFound` if any source name doesn't exist
pub fn rename_all(state: &mut ProofState, renames: Vec<(&str, &str)>) -> TacticResult {
    for (old_name, new_name) in renames {
        rename(state, old_name, new_name)?;
    }
    Ok(())
}

// ============================================================================
// Apply Fun Tactic
// ============================================================================

/// Tactic: apply_fun - Applies a function to both sides of an equality hypothesis.
pub fn apply_fun(state: &mut ProofState, func: Expr, hyp_name: &str) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?;
    let hyp_decl = goal
        .local_ctx
        .iter()
        .find(|d| d.name == hyp_name)
        .ok_or_else(|| TacticError::HypothesisNotFound(hyp_name.to_string()))?;

    let hyp_ty = state.metas.instantiate(&hyp_decl.ty);
    let hyp_fvar = hyp_decl.fvar;

    if let Some((lhs, rhs)) = match_eq_simple(&hyp_ty) {
        let new_lhs = Expr::app(func.clone(), lhs);
        let new_rhs = Expr::app(func.clone(), rhs);

        let goal_ref = state.current_goal().ok_or(TacticError::NoGoals)?;
        let result_ty = state.infer_type(goal_ref, &new_lhs)?;

        let new_eq = Expr::app(
            Expr::app(
                Expr::app(
                    Expr::const_(Name::from_string("Eq"), vec![Level::zero()]),
                    result_ty,
                ),
                new_lhs,
            ),
            new_rhs,
        );

        let goal = state.current_goal_mut().ok_or(TacticError::NoGoals)?;
        if let Some(decl) = goal.local_ctx.iter_mut().find(|d| d.fvar == hyp_fvar) {
            decl.ty = new_eq;
        }
        return Ok(());
    }

    Err(TacticError::Other(format!(
        "apply_fun: hypothesis '{hyp_name}' must be an equality"
    )))
}

/// Tactic: apply_fun_goal - Applies a function to both sides of an equality goal.
pub fn apply_fun_goal(state: &mut ProofState, func: Expr) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?;
    let target = state.metas.instantiate(&goal.target);

    let (lhs, rhs) = match_eq_simple(&target).ok_or_else(|| {
        TacticError::Other("apply_fun_goal: goal must be an equality".to_string())
    })?;

    let new_lhs = Expr::app(func.clone(), lhs);
    let new_rhs = Expr::app(func.clone(), rhs);

    let goal_ref = state.current_goal().ok_or(TacticError::NoGoals)?;
    let result_ty = state.infer_type(goal_ref, &new_lhs)?;

    let new_target = Expr::app(
        Expr::app(
            Expr::app(
                Expr::const_(Name::from_string("Eq"), vec![Level::zero()]),
                result_ty,
            ),
            new_lhs,
        ),
        new_rhs,
    );

    let goal = state.current_goal_mut().ok_or(TacticError::NoGoals)?;
    goal.target = new_target;
    Ok(())
}

// ============================================================================
// Clear Except Tactic
// ============================================================================

/// Tactic: clear_except - Clears all hypotheses except the specified ones.
pub fn clear_except(state: &mut ProofState, keep: &[&str]) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?;
    let mut fvars_to_keep: std::collections::HashSet<FVarId> = std::collections::HashSet::new();

    for name in keep {
        if let Some(decl) = goal.local_ctx.iter().find(|d| d.name == *name) {
            fvars_to_keep.insert(decl.fvar);
        }
    }

    let target = state.metas.instantiate(&goal.target);
    let target_fvars = collect_fvars(&target);
    for fv in target_fvars {
        fvars_to_keep.insert(fv);
    }

    let goal = state.current_goal().ok_or(TacticError::NoGoals)?;
    loop {
        let mut added = false;
        for decl in &goal.local_ctx {
            if fvars_to_keep.contains(&decl.fvar) {
                let ty_fvars = collect_fvars(&decl.ty);
                for fv in ty_fvars {
                    if !fvars_to_keep.contains(&fv) {
                        fvars_to_keep.insert(fv);
                        added = true;
                    }
                }
            }
        }
        if !added {
            break;
        }
    }

    let goal = state.current_goal_mut().ok_or(TacticError::NoGoals)?;
    goal.local_ctx
        .retain(|decl| fvars_to_keep.contains(&decl.fvar));
    Ok(())
}

// ============================================================================
// Replace Tactics
// ============================================================================

/// Tactic: replace - Replaces a hypothesis with a new one.
///
/// Creates a new goal to prove the replacement type.
pub fn replace(state: &mut ProofState, hyp_name: &str, new_type: Expr) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?;
    let hyp_idx = goal
        .local_ctx
        .iter()
        .position(|d| d.name == hyp_name)
        .ok_or_else(|| TacticError::HypothesisNotFound(hyp_name.to_string()))?;

    let proof_meta_id = state.metas.fresh(new_type.clone());

    let goal = state.current_goal_mut().ok_or(TacticError::NoGoals)?;
    goal.local_ctx[hyp_idx].ty = new_type.clone();
    goal.local_ctx[hyp_idx].value = Some(Expr::FVar(MetaState::to_fvar(proof_meta_id)));

    let new_goal = Goal {
        meta_id: proof_meta_id,
        target: new_type,
        local_ctx: goal.local_ctx[..hyp_idx].to_vec(),
    };

    state.goals.push(new_goal);
    Ok(())
}

/// Tactic: replace_hyp - Replaces a hypothesis with explicit proof term.
pub fn replace_hyp(
    state: &mut ProofState,
    hyp_name: &str,
    new_type: Expr,
    proof: Expr,
) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?;
    let hyp_idx = goal
        .local_ctx
        .iter()
        .position(|d| d.name == hyp_name)
        .ok_or_else(|| TacticError::HypothesisNotFound(hyp_name.to_string()))?;

    let goal = state.current_goal_mut().ok_or(TacticError::NoGoals)?;
    goal.local_ctx[hyp_idx].ty = new_type;
    goal.local_ctx[hyp_idx].value = Some(proof);
    Ok(())
}
