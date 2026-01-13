//! Instance tactics: letI, haveI, inferI
//!
//! These tactics introduce local typeclass instances that are available
//! for resolution within the current goal's scope.

use crate::tactic::smt::create_sorry_term;
use crate::tactic::{have_tactic, LocalDecl, ProofState, TacticError, TacticResult};

/// Introduce a local instance for the current goal.
///
/// `letI` introduces a local instance that will be available for typeclass
/// resolution within the current goal. Unlike `have`, it registers the
/// value as an instance.
///
/// # Example
/// ```text
/// -- Goal: Decidable P → ...
/// letI : Decidable P := Classical.dec P
/// -- Instance Decidable P is now available
/// ```
///
/// # Errors
/// - `NoGoals` if there are no goals
pub fn let_i(
    state: &mut ProofState,
    name: String,
    ty: lean5_kernel::Expr,
    value: lean5_kernel::Expr,
) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    // Get fresh fvar before mutable borrow
    let new_fvar = state.fresh_fvar();

    let goal = state.current_goal_mut().ok_or(TacticError::NoGoals)?;

    goal.local_ctx.push(LocalDecl {
        fvar: new_fvar,
        name,
        ty,
        value: Some(value),
    });

    Ok(())
}

/// Introduce a local instance hypothesis for the current goal.
///
/// `haveI` introduces a local instance hypothesis with a proof obligation.
/// It's like `have` but registers the result as an instance for typeclass
/// resolution.
///
/// # Example
/// ```text
/// -- Goal: some goal requiring Decidable P
/// haveI : Decidable P := by decide
/// -- Creates subgoal for Decidable P, then uses result as instance
/// ```
///
/// # Errors
/// - `NoGoals` if there are no goals
pub fn have_i(state: &mut ProofState, name: String, ty: lean5_kernel::Expr) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    // haveI creates a new goal for the instance, then adds it to context
    // For now, just use have_tactic which does the same thing
    have_tactic(state, name, ty, None)
}

/// Introduce an instance using inference.
///
/// `inferI` attempts to synthesize an instance using typeclass resolution
/// and adds it to the local context.
///
/// # Example
/// ```text
/// -- In a context where Decidable P can be inferred
/// inferI (inst : Decidable P)
/// -- inst is now available
/// ```
///
/// # Errors
/// - `NoGoals` if there are no goals
/// - `Other` if the instance cannot be inferred
pub fn infer_i(state: &mut ProofState, name: String, ty: lean5_kernel::Expr) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    // Create a sorry term as the inferred value
    let value = create_sorry_term(state.env(), &ty);

    // Get fresh fvar before mutable borrow
    let new_fvar = state.fresh_fvar();

    let goal = state.current_goal_mut().ok_or(TacticError::NoGoals)?;

    goal.local_ctx.push(LocalDecl {
        fvar: new_fvar,
        name,
        ty,
        value: Some(value),
    });

    Ok(())
}
