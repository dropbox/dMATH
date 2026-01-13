//! Conv (conversion) tactics for targeted rewriting
//!
//! This module provides tactics for navigating to specific subexpressions
//! and performing targeted rewrites:
//! - `ConvPosition` - Position markers for expression navigation
//! - `ConvPath` - A path through an expression tree
//! - `ConvState` - State for conv-mode rewriting
//! - `conv_rw` - Targeted rewrite using conv-style navigation
//! - `conv_lhs` - Rewrite only the left-hand side of an equality
//! - `conv_rhs` - Rewrite only the right-hand side of an equality
//! - `conv_arg` - Navigate into an argument and apply a tactic

use lean5_kernel::Expr;

use super::{contains_expr, match_equality, replace_expr, ProofState, TacticError, TacticResult};

// ============================================================================
// Conv Position and Path Types
// ============================================================================

/// Position in an expression for targeted rewriting.
///
/// Used by conv tactics to navigate to specific subexpressions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConvPosition {
    /// The root expression
    Root,
    /// Function in application (f x) - go to f
    AppFn,
    /// Argument in application (f x) - go to x
    AppArg,
    /// Body of lambda/forall (λ x, body / ∀ x, body)
    BinderBody,
    /// Type of lambda/forall (λ x : T, body / ∀ x : T, body)
    BinderType,
    /// Value in let binding (let x := v in body)
    LetValue,
    /// Body in let binding (let x := v in body)
    LetBody,
    /// Type in let binding (let x : T := v in body)
    LetType,
    /// Left-hand side of equality (a = b) - go to a
    EqLhs,
    /// Right-hand side of equality (a = b) - go to b
    EqRhs,
}

/// A path through an expression tree for conv navigation.
pub type ConvPath = Vec<ConvPosition>;

// ============================================================================
// Conv State
// ============================================================================

/// State for conv-mode rewriting
pub struct ConvState {
    /// The original expression being rewritten
    pub original: Expr,
    /// Current position in the expression tree
    pub path: ConvPath,
    /// The focused subexpression at current position
    pub focus: Expr,
}

impl ConvState {
    /// Create a new conv state focused on the given expression
    pub fn new(expr: Expr) -> Self {
        ConvState {
            original: expr.clone(),
            path: vec![],
            focus: expr,
        }
    }

    /// Replace the expression at a given position
    fn replace_at_position(expr: &Expr, path: &[ConvPosition], replacement: &Expr) -> Option<Expr> {
        if path.is_empty() {
            return Some(replacement.clone());
        }

        let (head, rest) = (&path[0], &path[1..]);
        match (head, expr) {
            (ConvPosition::Root, _) => Self::replace_at_position(expr, rest, replacement),
            (ConvPosition::AppFn, Expr::App(f, a)) => {
                let new_f = Self::replace_at_position(f, rest, replacement)?;
                Some(Expr::app(new_f, (**a).clone()))
            }
            (ConvPosition::AppArg, Expr::App(f, a)) => {
                let new_a = Self::replace_at_position(a, rest, replacement)?;
                Some(Expr::app((**f).clone(), new_a))
            }
            (ConvPosition::BinderBody, Expr::Lam(bi, ty, body)) => {
                let new_body = Self::replace_at_position(body, rest, replacement)?;
                Some(Expr::lam(*bi, (**ty).clone(), new_body))
            }
            (ConvPosition::BinderBody, Expr::Pi(bi, ty, body)) => {
                let new_body = Self::replace_at_position(body, rest, replacement)?;
                Some(Expr::pi(*bi, (**ty).clone(), new_body))
            }
            (ConvPosition::BinderType, Expr::Lam(bi, ty, body)) => {
                let new_ty = Self::replace_at_position(ty, rest, replacement)?;
                Some(Expr::lam(*bi, new_ty, (**body).clone()))
            }
            (ConvPosition::BinderType, Expr::Pi(bi, ty, body)) => {
                let new_ty = Self::replace_at_position(ty, rest, replacement)?;
                Some(Expr::pi(*bi, new_ty, (**body).clone()))
            }
            (ConvPosition::LetValue, Expr::Let(ty, _, body)) => {
                let new_val = Self::replace_at_position(&Expr::type_(), rest, replacement)?;
                Some(Expr::let_((**ty).clone(), new_val, (**body).clone()))
            }
            (ConvPosition::LetBody, Expr::Let(ty, val, body)) => {
                let new_body = Self::replace_at_position(body, rest, replacement)?;
                Some(Expr::let_((**ty).clone(), (**val).clone(), new_body))
            }
            (ConvPosition::LetType, Expr::Let(ty, val, body)) => {
                let new_ty = Self::replace_at_position(ty, rest, replacement)?;
                Some(Expr::let_(new_ty, (**val).clone(), (**body).clone()))
            }
            _ => None,
        }
    }

    /// Navigate to a subexpression
    pub fn go(&mut self, pos: ConvPosition) -> Result<(), TacticError> {
        let new_focus = match (&pos, &self.focus) {
            (ConvPosition::Root, _) => self.original.clone(),
            (ConvPosition::AppFn, Expr::App(f, _)) => (**f).clone(),
            (ConvPosition::AppArg, Expr::App(_, a)) => (**a).clone(),
            (ConvPosition::BinderBody, Expr::Lam(_, _, body))
            | (ConvPosition::BinderBody, Expr::Pi(_, _, body))
            | (ConvPosition::LetBody, Expr::Let(_, _, body)) => (**body).clone(),
            (ConvPosition::BinderType, Expr::Lam(_, ty, _))
            | (ConvPosition::BinderType, Expr::Pi(_, ty, _))
            | (ConvPosition::LetType, Expr::Let(ty, _, _)) => (**ty).clone(),
            (ConvPosition::LetValue, Expr::Let(_, val, _)) => (**val).clone(),
            (ConvPosition::EqLhs, _) => {
                let args: Vec<&Expr> = self.focus.get_app_args();
                if args.len() >= 2 {
                    args[1].clone()
                } else {
                    return Err(TacticError::Other(
                        "conv: cannot go to lhs - not an equality".to_string(),
                    ));
                }
            }
            (ConvPosition::EqRhs, _) => {
                let args: Vec<&Expr> = self.focus.get_app_args();
                if args.len() >= 3 {
                    args[2].clone()
                } else {
                    return Err(TacticError::Other(
                        "conv: cannot go to rhs - not an equality".to_string(),
                    ));
                }
            }
            _ => {
                return Err(TacticError::Other(format!(
                    "conv: cannot navigate {pos:?} at this position"
                )))
            }
        };

        self.path.push(pos);
        self.focus = new_focus;
        Ok(())
    }

    /// Apply a rewrite to the focused expression
    pub fn rewrite_focus(&mut self, from: &Expr, to: &Expr) -> bool {
        if contains_expr(&self.focus, from) {
            self.focus = replace_expr(&self.focus, from, to);
            true
        } else {
            false
        }
    }

    /// Get the final expression after all modifications
    pub fn finish(&self) -> Expr {
        if self.path.is_empty() {
            self.focus.clone()
        } else {
            ConvState::replace_at_position(&self.original, &self.path, &self.focus)
                .unwrap_or_else(|| self.original.clone())
        }
    }
}

// ============================================================================
// Conv Tactics
// ============================================================================

/// Targeted rewrite using conv-style navigation.
///
/// Allows rewriting at specific positions in the goal using a path.
///
/// # Example
/// ```text
/// -- Goal: f (a + b) = f (b + a)
/// conv_rw [AppArg, AppArg] h  -- rewrites inner (a + b) using h : a + b = b + a
/// ```
pub fn conv_rw(
    state: &mut ProofState,
    path: ConvPath,
    hyp_name: &str,
    reverse: bool,
) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Find the hypothesis
    let hyp_decl = goal
        .local_ctx
        .iter()
        .find(|d| d.name == hyp_name)
        .ok_or_else(|| TacticError::Other(format!("hypothesis '{hyp_name}' not found")))?
        .clone();

    // Check that the hypothesis is an equality
    let hyp_ty = state.whnf(&goal, &hyp_decl.ty);
    let (_eq_type, lhs, rhs, _eq_levels) = match_equality(&hyp_ty)?;

    let (from, to) = if reverse { (rhs, lhs) } else { (lhs, rhs) };

    // Create conv state and navigate to position
    let mut conv = ConvState::new(goal.target.clone());
    for pos in path {
        conv.go(pos)?;
    }

    // Apply the rewrite at the focused position
    if !conv.rewrite_focus(&from, &to) {
        return Err(TacticError::Other(
            "conv_rw: pattern not found at specified position".to_string(),
        ));
    }

    // Get the new target
    let new_target = conv.finish();

    // Update the goal
    state.goals[0].target = new_target;
    Ok(())
}

/// Rewrite the left-hand side of an equality goal.
///
/// For goal `a = b`, applies a rewrite to just the `a` part.
pub fn conv_lhs(state: &mut ProofState, hyp_name: &str, reverse: bool) -> TacticResult {
    conv_rw(state, vec![ConvPosition::EqLhs], hyp_name, reverse)
}

/// Rewrite the right-hand side of an equality goal.
///
/// For goal `a = b`, applies a rewrite to just the `b` part.
pub fn conv_rhs(state: &mut ProofState, hyp_name: &str, reverse: bool) -> TacticResult {
    conv_rw(state, vec![ConvPosition::EqRhs], hyp_name, reverse)
}

/// Navigate into an argument of the goal and apply a tactic.
///
/// For goal `f x`, `conv_arg` applies a transformation to just `x`.
pub fn conv_arg<F>(state: &mut ProofState, tactic: F) -> TacticResult
where
    F: FnOnce(&mut ProofState) -> TacticResult,
{
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Extract the argument if this is an application
    match &goal.target {
        Expr::App(_, arg) => {
            // Create a temporary state focused on the argument
            let mut temp_state = ProofState::new(state.env.clone(), (**arg).clone());
            temp_state.goals[0].local_ctx = goal.local_ctx.clone();

            // Apply the tactic
            tactic(&mut temp_state)?;

            // Get the transformed argument
            if let Some(new_goal) = temp_state.current_goal() {
                // Reconstruct with new argument
                if let Expr::App(f, _) = &goal.target {
                    state.goals[0].target = Expr::app((**f).clone(), new_goal.target.clone());
                }
            }

            Ok(())
        }
        _ => Err(TacticError::Other(
            "conv_arg: goal is not an application".to_string(),
        )),
    }
}
