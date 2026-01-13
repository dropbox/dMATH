//! Debugging and utility tactics
//!
//! This module provides tactics for debugging and utility operations:
//! - `trace`: Output debug messages during proof
//! - `itauto`: Intuitionistic tautology prover
//! - `clean`: Beta-reduce let-expressions
//! - `bound`: Prove inequalities by combining bounds
//! - `substs`: Substitute all equality hypotheses

use crate::tactic::{
    assumption, constructor, exfalso, exprs_equal, intro, is_pi_expr, linarith, match_and,
    match_eq_simple, subst, ProofState, TacticError, TacticResult,
};
use lean5_kernel::Expr;

// ============================================================================
// Trace Tactic (debugging)
// ============================================================================

/// Trace output level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TraceLevel {
    /// Only critical messages
    Error,
    /// Warnings and errors
    Warn,
    /// Informational messages
    #[default]
    Info,
    /// Detailed debug output
    Debug,
    /// Very detailed trace output
    Trace,
}

/// Result of a trace call
#[derive(Debug, Clone)]
pub struct TraceOutput {
    /// The message that was traced
    pub message: String,
    /// The trace level
    pub level: TraceLevel,
    /// Current goal state summary
    pub goal_summary: String,
    /// Number of remaining goals
    pub num_goals: usize,
}

/// Tactic: trace
///
/// Outputs a debug message and current goal state without modifying
/// the proof. Useful for debugging complex tactic scripts.
///
/// # Example
/// ```text
/// -- Goal: P ∧ Q
/// trace "About to split the conjunction"
/// -- Output: "About to split the conjunction"
/// -- Goal: P ∧ Q (unchanged)
/// split
/// ```
///
/// # Arguments
/// * `message` - The message to output
///
/// # Returns
/// A `TraceOutput` containing the message and goal state summary
pub fn trace(state: &ProofState, message: &str) -> Result<TraceOutput, TacticError> {
    trace_with_level(state, message, TraceLevel::Info)
}

/// trace with explicit level
pub fn trace_with_level(
    state: &ProofState,
    message: &str,
    level: TraceLevel,
) -> Result<TraceOutput, TacticError> {
    let goal_summary = if let Some(goal) = state.current_goal() {
        format!(
            "⊢ {:?} (with {} hypotheses)",
            goal.target,
            goal.local_ctx.len()
        )
    } else {
        "no goals".to_string()
    };

    Ok(TraceOutput {
        message: message.to_string(),
        level,
        goal_summary,
        num_goals: state.goals.len(),
    })
}

/// trace_state - outputs detailed state information
pub fn trace_state(state: &ProofState) -> Result<TraceOutput, TacticError> {
    let mut lines = Vec::new();
    lines.push(format!("Goals: {}", state.goals.len()));

    for (i, goal) in state.goals.iter().enumerate() {
        lines.push(format!("Goal {}:", i + 1));
        lines.push(format!("  Target: {:?}", goal.target));
        lines.push(format!("  Context ({} items):", goal.local_ctx.len()));
        for decl in &goal.local_ctx {
            lines.push(format!("    {} : {:?}", decl.name, decl.ty));
        }
    }

    let message = lines.join("\n");
    let goal_summary = if state.goals.is_empty() {
        "no goals".to_string()
    } else {
        format!("{} goal(s)", state.goals.len())
    };

    Ok(TraceOutput {
        message,
        level: TraceLevel::Debug,
        goal_summary,
        num_goals: state.goals.len(),
    })
}

/// trace_expr - trace an expression's structure
pub fn trace_expr(state: &ProofState, expr: &Expr) -> Result<TraceOutput, TacticError> {
    let message = format!("Expression structure: {expr:?}");
    let goal_summary = if let Some(goal) = state.current_goal() {
        format!("⊢ {:?}", goal.target)
    } else {
        "no goals".to_string()
    };

    Ok(TraceOutput {
        message,
        level: TraceLevel::Debug,
        goal_summary,
        num_goals: state.goals.len(),
    })
}

// ============================================================================
// ITauto Tactic
// ============================================================================

#[derive(Debug, Clone)]
pub struct ITautoConfig {
    pub max_depth: usize,
    pub verbose: bool,
}

impl Default for ITautoConfig {
    fn default() -> Self {
        Self {
            max_depth: 20,
            verbose: false,
        }
    }
}

/// Tactic: itauto - Intuitionistic tautology prover.
pub fn itauto(state: &mut ProofState) -> TacticResult {
    itauto_with_config(state, ITautoConfig::default())
}

pub fn itauto_with_config(state: &mut ProofState, config: ITautoConfig) -> TacticResult {
    itauto_search(state, config.max_depth)
}

fn itauto_search(state: &mut ProofState, depth: usize) -> TacticResult {
    if depth == 0 {
        return Err(TacticError::Other(
            "itauto: search depth exhausted".to_string(),
        ));
    }

    if state.is_complete() {
        return Ok(());
    }

    let goal = state.current_goal().ok_or(TacticError::NoGoals)?;
    let target = state.metas.instantiate(&goal.target);

    // Rule 1: Check if goal is in hypotheses
    for decl in &goal.local_ctx {
        let ty = state.metas.instantiate(&decl.ty);
        if exprs_equal(&ty, &target) {
            return assumption(state);
        }
    }

    // Rule 2: If goal is True
    if matches!(&target, Expr::Const(name, _) if name.to_string() == "True") {
        return trivial(state);
    }

    // Rule 3: Check for False in hypotheses
    for decl in &goal.local_ctx {
        let ty = state.metas.instantiate(&decl.ty);
        if matches!(&ty, Expr::Const(name, _) if name.to_string() == "False") {
            return exfalso(state).and_then(|_| assumption(state));
        }
    }

    // Rule 4: If goal is P -> Q, use intro
    if is_pi_expr(&target) {
        let mut new_state = state.clone();
        if intro(&mut new_state, "h".to_string()).is_ok()
            && itauto_search(&mut new_state, depth - 1).is_ok()
        {
            *state = new_state;
            return Ok(());
        }
    }

    // Rule 5: If goal is P /\ Q, split
    if match_and(&target).is_some() {
        let mut new_state = state.clone();
        if constructor(&mut new_state).is_ok() && itauto_search(&mut new_state, depth - 1).is_ok() {
            *state = new_state;
            return Ok(());
        }
    }

    Err(TacticError::Other(
        "itauto: no intuitionistic proof found".to_string(),
    ))
}

/// Tactic: trivial - proves True goals
fn trivial(state: &mut ProofState) -> TacticResult {
    crate::tactic::trivial(state)
}

// ============================================================================
// Clean Tactic
// ============================================================================

/// Tactic: clean - Simplifies by beta-reducing let-expressions.
pub fn clean(state: &mut ProofState) -> TacticResult {
    let goal = state.current_goal_mut().ok_or(TacticError::NoGoals)?;
    goal.target = beta_reduce_all(&goal.target);
    for decl in &mut goal.local_ctx {
        decl.ty = beta_reduce_all(&decl.ty);
    }
    Ok(())
}

pub(crate) fn beta_reduce_all(expr: &Expr) -> Expr {
    match expr {
        Expr::App(f, a) => {
            let f_reduced = beta_reduce_all(f);
            let a_reduced = beta_reduce_all(a);
            if let Expr::Lam(_, _, body) = &f_reduced {
                beta_reduce_all(&body.instantiate(&a_reduced))
            } else {
                Expr::app(f_reduced, a_reduced)
            }
        }
        Expr::Lam(bi, ty, body) => Expr::lam(*bi, beta_reduce_all(ty), beta_reduce_all(body)),
        Expr::Pi(bi, ty, body) => Expr::pi(*bi, beta_reduce_all(ty), beta_reduce_all(body)),
        Expr::Let(_ty, val, body) => {
            let val_reduced = beta_reduce_all(val);
            let body_reduced = beta_reduce_all(body);
            beta_reduce_all(&body_reduced.instantiate(&val_reduced))
        }
        _ => expr.clone(),
    }
}

// ============================================================================
// Bound Tactic
// ============================================================================

/// Tactic: bound - Proves inequalities by combining bounds.
pub fn bound(state: &mut ProofState) -> TacticResult {
    let _goal = state.current_goal().ok_or(TacticError::NoGoals)?;
    linarith(state)
}

// ============================================================================
// Substs Tactic
// ============================================================================

/// Tactic: substs - Substitutes all equality hypotheses where lhs is a variable.
pub fn substs(state: &mut ProofState) -> TacticResult {
    let mut made_progress = true;
    let mut iterations = 0;
    let max_iterations = 100;

    while made_progress && iterations < max_iterations {
        made_progress = false;
        iterations += 1;

        let goal = state.current_goal().ok_or(TacticError::NoGoals)?;
        let local_ctx = goal.local_ctx.clone();

        let mut subst_name = None;
        for decl in &local_ctx {
            let ty = state.metas.instantiate(&decl.ty);
            if let Some((Expr::FVar(fvar_id), _rhs)) = match_eq_simple(&ty) {
                let is_in_ctx = local_ctx
                    .iter()
                    .any(|d| d.fvar == fvar_id && d.name != decl.name);
                if is_in_ctx {
                    subst_name = Some(decl.name.clone());
                    break;
                }
            }
        }

        if let Some(name) = subst_name {
            if subst(state, &name).is_ok() {
                made_progress = true;
            }
        }
    }

    Ok(())
}
