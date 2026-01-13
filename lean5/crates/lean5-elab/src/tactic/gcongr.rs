//! Generalized congruence tactic for inequalities
//!
//! Provides the `gcongr` tactic which proves goals of the form
//! `f a₁ ... aₙ ≤ f b₁ ... bₙ` by creating subgoals `aᵢ ≤ bᵢ`
//! for arguments that differ.

use lean5_kernel::name::Name;
use lean5_kernel::{Expr, TypeChecker};

use super::{Goal, ProofState, TacticError, TacticResult};

/// Generalized congruence tactic for inequalities.
///
/// `gcongr` proves goals of the form `f a₁ ... aₙ ≤ f b₁ ... bₙ` by creating
/// subgoals `aᵢ ≤ bᵢ` for arguments that differ. It's particularly useful for:
/// - Monotonic functions (add, mul for non-negative)
/// - Norm bounds
/// - Integral bounds
///
/// The tactic handles:
/// - `≤` (Le), `<` (Lt), `≥` (Ge), `>` (Gt)
/// - Arithmetic operations with monotonicity
pub fn gcongr(state: &mut ProofState) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
    let target = state.metas.instantiate(&goal.target);

    // Try to match an inequality
    if let Some((rel, ty, lhs, rhs)) = match_inequality(&target) {
        return gcongr_inequality(state, &goal, rel, &ty, &lhs, &rhs);
    }

    Err(TacticError::GoalMismatch(
        "gcongr: goal must be an inequality".to_string(),
    ))
}

/// Inequality relation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IneqRel {
    Le, // ≤
    Lt, // <
    Ge, // ≥
    Gt, // >
}

/// Match inequality patterns
pub(crate) fn match_inequality(expr: &Expr) -> Option<(IneqRel, Expr, Expr, Expr)> {
    let head = expr.get_app_fn();
    let args: Vec<&Expr> = expr.get_app_args();

    if let Expr::Const(name, _) = head {
        let name_str = name.to_string();

        // LE.le or HasLe.le: expected form is LE.le α inst a b
        if (name_str == "LE.le" || name_str == "HasLe.le" || name_str == "le") && args.len() >= 2 {
            let ty = if args.len() >= 3 {
                args[0].clone()
            } else {
                Expr::const_(Name::from_string("Nat"), vec![])
            };
            let lhs = args[args.len() - 2].clone();
            let rhs = args[args.len() - 1].clone();
            return Some((IneqRel::Le, ty, lhs, rhs));
        }

        // LT.lt or HasLt.lt
        if (name_str == "LT.lt" || name_str == "HasLt.lt" || name_str == "lt") && args.len() >= 2 {
            let ty = if args.len() >= 3 {
                args[0].clone()
            } else {
                Expr::const_(Name::from_string("Nat"), vec![])
            };
            let lhs = args[args.len() - 2].clone();
            let rhs = args[args.len() - 1].clone();
            return Some((IneqRel::Lt, ty, lhs, rhs));
        }

        // GE.ge (greater or equal)
        if (name_str == "GE.ge" || name_str == "HasGe.ge" || name_str == "ge") && args.len() >= 2 {
            let ty = if args.len() >= 3 {
                args[0].clone()
            } else {
                Expr::const_(Name::from_string("Nat"), vec![])
            };
            let lhs = args[args.len() - 2].clone();
            let rhs = args[args.len() - 1].clone();
            return Some((IneqRel::Ge, ty, lhs, rhs));
        }

        // GT.gt (greater than)
        if (name_str == "GT.gt" || name_str == "HasGt.gt" || name_str == "gt") && args.len() >= 2 {
            let ty = if args.len() >= 3 {
                args[0].clone()
            } else {
                Expr::const_(Name::from_string("Nat"), vec![])
            };
            let lhs = args[args.len() - 2].clone();
            let rhs = args[args.len() - 1].clone();
            return Some((IneqRel::Gt, ty, lhs, rhs));
        }
    }
    None
}

/// Handle inequality goal with gcongr
fn gcongr_inequality(
    state: &mut ProofState,
    goal: &Goal,
    rel: IneqRel,
    _ty: &Expr,
    lhs: &Expr,
    rhs: &Expr,
) -> TacticResult {
    // Check if both sides have the same head (function application)
    let lhs_head = lhs.get_app_fn();
    let rhs_head = rhs.get_app_fn();

    // If heads are definitionally equal, decompose
    let tc = TypeChecker::new(state.env());
    if tc.is_def_eq(lhs_head, rhs_head) {
        let lhs_args: Vec<&Expr> = lhs.get_app_args();
        let rhs_args: Vec<&Expr> = rhs.get_app_args();

        if lhs_args.len() == rhs_args.len() {
            // Find differing arguments and create subgoals
            let mut subgoals = Vec::new();

            for (i, (l, r)) in lhs_args.iter().zip(rhs_args.iter()).enumerate() {
                if !tc.is_def_eq(l, r) {
                    // Create subgoal for this argument
                    let subgoal_target = make_ineq_goal(rel, l, r);
                    subgoals.push((i, subgoal_target));
                }
            }

            if subgoals.is_empty() {
                // All args equal, close with reflexivity
                let refl_proof = match rel {
                    IneqRel::Le | IneqRel::Ge => Expr::const_(Name::from_string("le_refl"), vec![]),
                    IneqRel::Lt | IneqRel::Gt => {
                        // Strict inequality cannot be reflexive
                        return Err(TacticError::Other(
                            "gcongr: strict inequality cannot hold for equal terms".to_string(),
                        ));
                    }
                };
                state.metas.assign(goal.meta_id, refl_proof);
                state.goals.remove(0);
                return Ok(());
            }

            // Create new goals for differing arguments
            state.goals.remove(0);

            for (_i, target) in subgoals.into_iter().rev() {
                let meta_id = state.metas.fresh(target.clone());
                let new_goal = Goal {
                    meta_id,
                    target,
                    local_ctx: goal.local_ctx.clone(),
                };
                state.goals.insert(0, new_goal);
            }

            return Ok(());
        }
    }

    // Try monotonicity rules for specific operations
    gcongr_monotonic(state, goal, rel, lhs, rhs)
}

/// Create inequality goal expression
pub(crate) fn make_ineq_goal(rel: IneqRel, lhs: &Expr, rhs: &Expr) -> Expr {
    let rel_name = match rel {
        IneqRel::Le => "LE.le",
        IneqRel::Lt => "LT.lt",
        IneqRel::Ge => "GE.ge",
        IneqRel::Gt => "GT.gt",
    };
    Expr::app(
        Expr::app(
            Expr::const_(Name::from_string(rel_name), vec![]),
            lhs.clone(),
        ),
        rhs.clone(),
    )
}

/// Try monotonicity rules for arithmetic operations
fn gcongr_monotonic(
    state: &mut ProofState,
    goal: &Goal,
    rel: IneqRel,
    lhs: &Expr,
    rhs: &Expr,
) -> TacticResult {
    // Match addition: a + b ≤ c + d
    if let (Some((a, b)), Some((c, d))) = (match_add(lhs), match_add(rhs)) {
        // Create two subgoals: a ≤ c and b ≤ d
        state.goals.remove(0);

        let goal1 = make_ineq_goal(rel, &a, &c);
        let goal2 = make_ineq_goal(rel, &b, &d);

        let meta1 = state.metas.fresh(goal1.clone());
        let meta2 = state.metas.fresh(goal2.clone());

        state.goals.insert(
            0,
            Goal {
                meta_id: meta2,
                target: goal2,
                local_ctx: goal.local_ctx.clone(),
            },
        );
        state.goals.insert(
            0,
            Goal {
                meta_id: meta1,
                target: goal1,
                local_ctx: goal.local_ctx.clone(),
            },
        );

        return Ok(());
    }

    Err(TacticError::Other(
        "gcongr: cannot apply congruence rules".to_string(),
    ))
}

/// Match addition pattern a + b
pub(crate) fn match_add(expr: &Expr) -> Option<(Expr, Expr)> {
    let head = expr.get_app_fn();
    let args: Vec<&Expr> = expr.get_app_args();

    if let Expr::Const(name, _) = head {
        let name_str = name.to_string();
        if (name_str.contains("add") || name_str.contains("Add") || name_str == "HAdd.hAdd")
            && args.len() >= 2
        {
            return Some((args[args.len() - 2].clone(), args[args.len() - 1].clone()));
        }
    }

    // Check for deeper application (HAdd.hAdd α β γ inst a b)
    if let Expr::App(f, b) = expr {
        if let Expr::App(f2, a) = f.as_ref() {
            let inner_head = f2.get_app_fn();
            if let Expr::Const(name, _) = inner_head {
                if name.to_string() == "HAdd.hAdd" {
                    return Some((a.as_ref().clone(), b.as_ref().clone()));
                }
            }
        }
    }

    None
}
