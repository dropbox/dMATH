//! Propositional tautology solver
//!
//! Provides the `tauto` tactic for solving classical propositional logic goals.

use lean5_kernel::Expr;

use super::{
    assumption, by_cases, cases, clear, contradiction, intro, left_tactic, rfl, right_tactic,
    solve_by_elim, split_tactic, trivial, Goal, LocalDecl, ProofState, TacticError, TacticResult,
};
use crate::tactic::arithmetic::{
    exprs_syntactically_equal, is_false, match_and, match_not, match_or,
};
use crate::tactic::simp::is_true_const;

// ============================================================================
// tauto - Classical propositional tautology solver
// ============================================================================

/// Solve propositional tautologies.
///
/// `tauto` is a decision procedure for classical propositional logic.
/// It can solve goals that are tautologies involving only propositional
/// connectives (∧, ∨, ¬, →, ↔, True, False).
///
/// # Algorithm
/// 1. Convert the goal to negation normal form (NNF)
/// 2. Apply tableau-style proof search:
///    - Decompose conjunctions in hypotheses
///    - Split on disjunctions in hypotheses
///    - Apply hypotheses to match goal
/// 3. Check for contradictions
///
/// # Supported connectives
/// - And (∧)
/// - Or (∨)
/// - Not (¬)
/// - Implies (→)
/// - Iff (↔)
/// - True, False
///
/// # Example
/// ```text
/// -- Goal: P ∨ ¬P
/// tauto
/// -- Goal closed (law of excluded middle)
/// ```
///
/// # Errors
/// - `NoGoals` if there are no goals
/// - `Other` if the goal is not a propositional tautology
pub fn tauto(state: &mut ProofState) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    // Normalize trivial context before solving
    preprocess_tauto_context(state)?;

    if state.goals.is_empty() {
        return Ok(());
    }

    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Try simple tactics first
    if rfl(state).is_ok() {
        return Ok(());
    }

    if trivial(state).is_ok() {
        return Ok(());
    }

    if assumption(state).is_ok() {
        return Ok(());
    }

    // Try to prove using propositional reasoning
    if tauto_prove(state, &goal)? {
        return Ok(());
    }

    Err(TacticError::Other(
        "tauto: not a propositional tautology".to_string(),
    ))
}

/// Simplify the local context for propositional reasoning.
///
/// Removes trivial hypotheses (True, a = a), splits conjunctions, and
/// discharges goals that already contain contradictions.
fn preprocess_tauto_context(state: &mut ProofState) -> Result<(), TacticError> {
    loop {
        if state.goals.is_empty() {
            return Ok(());
        }

        let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
        let mut progress = false;

        for decl in goal.local_ctx.clone() {
            let ty = state.metas.instantiate(&decl.ty);
            let ty_whnf = state.whnf(&goal, &ty);

            // Close immediately if we already have False
            if is_false(&ty_whnf) {
                contradiction(state)?;
                return Ok(());
            }

            // Drop trivial hypotheses to keep the search space small
            if is_true_const(&ty_whnf) || is_trivial_equality(&ty_whnf) {
                clear(state, &decl.name)?;
                progress = true;
                break;
            }

            // Break conjunctions into separate hypotheses
            if match_and(&ty_whnf).is_some() {
                cases(state, &decl.name)?;
                progress = true;
                break;
            }
        }

        if !progress {
            break;
        }
    }

    Ok(())
}

/// Find the first disjunctive hypothesis in the local context.
fn find_or_hypothesis(state: &mut ProofState, goal: &Goal) -> Option<String> {
    for decl in &goal.local_ctx {
        let ty = state.metas.instantiate(&decl.ty);
        let ty_whnf = state.whnf(goal, &ty);
        if match_or(&ty_whnf).is_some() {
            return Some(decl.name.clone());
        }
    }
    None
}

/// Main propositional tautology prover
fn tauto_prove(state: &mut ProofState, _goal: &Goal) -> Result<bool, TacticError> {
    // Simplify context (split conjunctions, remove trivial hyps)
    preprocess_tauto_context(state)?;

    if state.goals.is_empty() {
        return Ok(true);
    }

    // Refresh goal/target after preprocessing
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
    let target = &goal.target;

    // Close immediately if the context already contains a contradiction
    if contradiction(state).is_ok() {
        return Ok(true);
    }

    // Split on the first disjunctive hypothesis to explore both branches
    if let Some(hyp_name) = find_or_hypothesis(state, &goal) {
        let goals_backup = state.goals.clone();
        if cases(state, &hyp_name).is_ok() {
            let first_goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
            if tauto_prove(state, &first_goal)? {
                if state.goals.is_empty() {
                    return Ok(true);
                }
                let second_goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
                if tauto_prove(state, &second_goal)? {
                    return Ok(true);
                }
            }
        }
        state.goals = goals_backup;
    }

    // Check if goal is True
    if is_true_const(target) {
        // Close with trivial
        return Ok(trivial(state).is_ok());
    }

    // Check if goal is False and we have False in hypotheses
    if is_false(target) {
        // Try to find False in context
        if assumption(state).is_ok() {
            return Ok(true);
        }
        // Try contradiction
        if contradiction(state).is_ok() {
            return Ok(true);
        }
        return Ok(false);
    }

    // Check for P ∨ ¬P (excluded middle)
    if let Some((p, q)) = match_or(target) {
        if let Some(inner_p) = match_not(&q) {
            if exprs_syntactically_equal(&p, &inner_p) {
                // This is P ∨ ¬P - classical tautology
                // We can close this with by_cases
                let hyp_name = fresh_hyp_name(&goal.local_ctx, "h");
                // Chain the tactics using && for cleaner code
                if by_cases(state, hyp_name, p.clone()).is_ok()
                    && left_tactic(state).is_ok()
                    && assumption(state).is_ok()
                    && right_tactic(state).is_ok()
                    && assumption(state).is_ok()
                {
                    return Ok(true);
                }
            }
        }
        // Also check ¬P ∨ P
        if let Some(inner_p) = match_not(&p) {
            if exprs_syntactically_equal(&inner_p, &q) {
                let hyp_name = fresh_hyp_name(&goal.local_ctx, "h");
                // Chain the tactics using && for cleaner code
                if by_cases(state, hyp_name, q.clone()).is_ok()
                    && right_tactic(state).is_ok()
                    && assumption(state).is_ok()
                    && left_tactic(state).is_ok()
                    && assumption(state).is_ok()
                {
                    return Ok(true);
                }
            }
        }
    }

    // Check if goal is a conjunction - split and prove both
    if let Some((_p, _q)) = match_and(target) {
        if split_tactic(state).is_ok() {
            // Try to prove first conjunct
            let first_goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
            if tauto_prove(state, &first_goal)? {
                // Try to prove second conjunct
                if !state.goals.is_empty() {
                    let second_goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
                    if tauto_prove(state, &second_goal)? {
                        return Ok(true);
                    }
                }
            }
        }
        return Ok(false);
    }

    // Check if goal is a disjunction - try left, then right
    if match_or(target).is_some() {
        // Save state for backtracking
        let goals_backup = state.goals.clone();

        // Try left
        if left_tactic(state).is_ok() {
            let left_goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
            if tauto_prove(state, &left_goal)? {
                return Ok(true);
            }
        }

        // Restore and try right
        state.goals = goals_backup;
        if right_tactic(state).is_ok() {
            let right_goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
            if tauto_prove(state, &right_goal)? {
                return Ok(true);
            }
        }

        return Ok(false);
    }

    // Check if goal is an implication - intro and continue
    if let Expr::Pi(_, _, _) = target {
        let hyp_name = fresh_hyp_name(&goal.local_ctx, "h");
        if intro(state, hyp_name).is_ok() {
            let new_goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
            return tauto_prove(state, &new_goal);
        }
    }

    // Check if goal matches a hypothesis
    if assumption(state).is_ok() {
        return Ok(true);
    }

    // Try solve_by_elim with limited depth
    if solve_by_elim(state, 3).is_ok() {
        return Ok(true);
    }

    Ok(false)
}

/// Generate a fresh hypothesis name
pub(crate) fn fresh_hyp_name(local_ctx: &[LocalDecl], base: &str) -> String {
    let name = base.to_string();
    let mut counter = 0;

    loop {
        let candidate = if counter == 0 {
            name.clone()
        } else {
            format!("{name}{counter}")
        };

        if !local_ctx.iter().any(|d| d.name == candidate) {
            return candidate;
        }
        counter += 1;
    }
}

/// Check if expression is a trivial equality (a = a)
fn is_trivial_equality(expr: &Expr) -> bool {
    // Match Eq α a a pattern
    if let Expr::App(f1, rhs) = expr {
        if let Expr::App(f2, lhs) = f1.as_ref() {
            if let Expr::App(f3, _ty) = f2.as_ref() {
                if let Expr::Const(name, _) = f3.as_ref() {
                    if name.to_string() == "Eq" {
                        return exprs_syntactically_equal(lhs, rhs);
                    }
                }
            }
        }
    }
    false
}
