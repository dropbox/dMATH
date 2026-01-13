//! Extensionality tactics
//!
//! This module provides tactics for proving equality via extensionality principles:
//! - `funext` - Function extensionality: proves `f = g` by showing `∀ x, f x = g x`
//! - `propext` - Propositional extensionality: proves `P = Q` by showing `P ↔ Q`
//! - `set_ext` - Set extensionality: proves `s = t` by showing `∀ x, x ∈ s ↔ x ∈ t`
//! - `quot_ext` - Quotient extensionality via induction principles

use lean5_kernel::name::Name;
use lean5_kernel::{BinderInfo, Expr, Level};

use super::{
    apply, collect_consts, match_equality, Goal, LocalDecl, ProofState, TacticError, TacticResult,
};
use crate::unify::MetaState;

// ============================================================================
// Function Extensionality
// ============================================================================

/// Apply function extensionality.
///
/// For a goal of the form `f = g` where `f g : A → B`,
/// changes the goal to `∀ x, f x = g x`.
pub fn funext(state: &mut ProofState, var_name: String) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
    let target = goal.target.clone();

    // Check if target is an equality
    let (ty, lhs, rhs, _levels) = match_equality(&target)?;

    // Check if the type is a function type
    let (dom, _cod) = match &ty {
        Expr::Pi(_, binder_type, body) => (binder_type.as_ref().clone(), body.as_ref().clone()),
        _ => {
            return Err(TacticError::Other(
                "funext: equality is not between functions".to_string(),
            ))
        }
    };

    // Create a new free variable for the argument
    let fvar = state.fresh_fvar();
    let fvar_expr = Expr::FVar(fvar);

    // Create the new goal: f x = g x
    let new_lhs = Expr::app(lhs.clone(), fvar_expr.clone());
    let new_rhs = Expr::app(rhs.clone(), fvar_expr.clone());

    // Infer the result type (codomain applied to the argument)
    let result_ty = state.whnf(&goal, &Expr::app(ty.clone(), fvar_expr.clone()));

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

    // Create new goal with the variable in context
    let mut new_ctx = goal.local_ctx.clone();
    new_ctx.push(LocalDecl {
        fvar,
        name: var_name,
        ty: dom,
        value: None,
    });

    let new_meta = state.metas.fresh(new_target.clone());
    let new_goal = Goal {
        meta_id: new_meta,
        target: new_target,
        local_ctx: new_ctx,
    };

    // Build the proof term: funext (fun x => ?m)
    let meta_expr = Expr::FVar(MetaState::to_fvar(new_meta));
    let proof = Expr::app(
        Expr::app(
            Expr::app(
                Expr::app(
                    Expr::const_(
                        Name::from_string("funext"),
                        vec![Level::zero(), Level::zero()],
                    ),
                    ty.clone(),
                ),
                lhs.clone(),
            ),
            rhs.clone(),
        ),
        Expr::lam(BinderInfo::Default, Expr::type_(), meta_expr),
    );

    let old_goal = state.goals.remove(0);
    state.metas.assign(old_goal.meta_id, proof);
    state.goals.insert(0, new_goal);
    Ok(())
}

// ============================================================================
// Propositional Extensionality
// ============================================================================

/// Apply propositional extensionality.
///
/// For a goal of the form `P = Q` where `P Q : Prop`,
/// changes the goal to `P ↔ Q`.
pub fn propext(state: &mut ProofState) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
    let target = goal.target.clone();

    // Check if target is an equality of propositions
    let (ty, lhs, rhs, _levels) = match_equality(&target)?;

    // Check that the type is Prop (Sort(0))
    match &ty {
        Expr::Sort(level) if level.is_zero() => {}
        _ => {
            return Err(TacticError::Other(
                "propext: equality is not between propositions".to_string(),
            ))
        }
    }

    // Create new goal: P ↔ Q
    let iff_type = Expr::app(
        Expr::app(Expr::const_(Name::from_string("Iff"), vec![]), lhs.clone()),
        rhs.clone(),
    );

    let new_meta = state.metas.fresh(iff_type.clone());
    let new_goal = Goal {
        meta_id: new_meta,
        target: iff_type,
        local_ctx: goal.local_ctx.clone(),
    };

    // Build proof term: propext ?m
    let meta_expr = Expr::FVar(MetaState::to_fvar(new_meta));
    let proof = Expr::app(
        Expr::app(
            Expr::app(
                Expr::const_(Name::from_string("propext"), vec![]),
                lhs.clone(),
            ),
            rhs.clone(),
        ),
        meta_expr,
    );

    let old_goal = state.goals.remove(0);
    state.metas.assign(old_goal.meta_id, proof);
    state.goals.insert(0, new_goal);
    Ok(())
}

// ============================================================================
// Set Extensionality
// ============================================================================

/// Set extensionality.
///
/// For a goal of the form `s = t` where `s t : Set α`,
/// changes the goal to `∀ x, x ∈ s ↔ x ∈ t`.
pub fn set_ext(state: &mut ProofState, var_name: String) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
    let target = goal.target.clone();

    // Check if target is an equality
    let (ty, lhs, rhs, _levels) = match_equality(&target)?;

    // Check if this is a Set type (Set α = α → Prop)
    // Sets are represented as functions to Prop
    let is_set_type = match &ty {
        Expr::App(f, _) => {
            if let Expr::Const(name, _) = &**f {
                name == &Name::from_string("Set")
            } else {
                false
            }
        }
        Expr::Pi(_, _, ret) => {
            // α → Prop is represented as Pi type
            matches!(&**ret, Expr::Sort(l) if l.is_zero())
        }
        _ => false,
    };

    if !is_set_type {
        // Try to proceed anyway - the type might be definitionally equal to Set
    }

    // Extract the element type
    let elem_type = match &ty {
        Expr::App(_, arg) => (**arg).clone(),
        Expr::Pi(_, ty, _) => (**ty).clone(),
        _ => {
            return Err(TacticError::Other(
                "set_ext: cannot determine element type".to_string(),
            ))
        }
    };

    // Create x : elem_type
    let x_fvar = state.metas.fresh(elem_type.clone());

    // Build: x ∈ s ↔ x ∈ t
    // Membership is application: s x, t x
    let x_var = Expr::FVar(MetaState::to_fvar(x_fvar));
    let mem_lhs = Expr::app(lhs.clone(), x_var.clone());
    let mem_rhs = Expr::app(rhs.clone(), x_var.clone());

    let iff_type = Expr::app(
        Expr::app(Expr::const_(Name::from_string("Iff"), vec![]), mem_lhs),
        mem_rhs,
    );

    // New goal: ∀ x, x ∈ s ↔ x ∈ t
    let new_target = Expr::pi(BinderInfo::Default, elem_type.clone(), iff_type);

    // Add x to local context
    let mut new_local_ctx = goal.local_ctx.clone();
    new_local_ctx.push(LocalDecl {
        fvar: MetaState::to_fvar(x_fvar),
        name: var_name,
        ty: elem_type,
        value: None,
    });

    let new_meta = state.metas.fresh(new_target.clone());
    let new_goal = Goal {
        meta_id: new_meta,
        target: new_target,
        local_ctx: new_local_ctx,
    };

    // Build proof: funext (λ x, propext ?m)
    let meta_expr = Expr::FVar(MetaState::to_fvar(new_meta));
    let proof = Expr::app(
        Expr::const_(Name::from_string("Set.ext"), vec![]),
        meta_expr,
    );

    let old_goal = state.goals.remove(0);
    state.metas.assign(old_goal.meta_id, proof);
    state.goals.insert(0, new_goal);
    Ok(())
}

// ============================================================================
// Quotient Extensionality
// ============================================================================

/// Quotient extensionality (for quotient types).
///
/// For a goal involving quotient equality, introduces the lifting lemma.
pub fn quot_ext(state: &mut ProofState) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
    let target = goal.target.clone();

    // Check if target involves Quotient
    let has_quotient = collect_consts(&target)
        .iter()
        .any(|n| n.to_string().contains("Quotient") || n.to_string().contains("Quot"));

    if !has_quotient {
        return Err(TacticError::Other(
            "quot_ext: goal does not involve quotient types".to_string(),
        ));
    }

    // Try to apply Quotient.ind or Quot.ind
    let quot_ind = Expr::const_(
        Name::from_string("Quot.ind"),
        vec![Level::param(Name::from_string("u"))],
    );
    if state
        .env
        .get_const(&Name::from_string("Quot.ind"))
        .is_some()
    {
        // Apply Quot.ind
        apply(state, quot_ind)?;
        return Ok(());
    }

    Err(TacticError::Other(
        "quot_ext: Quot.ind not found in environment".to_string(),
    ))
}
