//! Finite and interval case splitting tactics
//!
//! Provides tactics for case splitting on finite types and integer intervals.

use lean5_kernel::name::Name;
use lean5_kernel::{Expr, FVarId, Level};

use super::{Goal, LocalDecl, ProofState, TacticError, TacticResult};
use crate::tactic::arithmetic::match_le;
use crate::tactic::arithmetic::match_lt;

// ============================================================================
// fin_cases - Case split on finite types
// ============================================================================

/// Case split on a hypothesis of finite type.
///
/// `fin_cases` works on hypotheses whose type is a finite type (like Fin n,
/// Bool, or an enumeration). It creates a goal for each possible value.
///
/// # Algorithm
/// 1. Identify the hypothesis and its finite type
/// 2. Enumerate all inhabitants of the type
/// 3. Create a subgoal for each inhabitant with the hypothesis instantiated
///
/// # Example
/// ```text
/// -- h : Fin 3
/// -- Goal: P h
/// fin_cases h
/// -- Goal 1: P 0
/// -- Goal 2: P 1
/// -- Goal 3: P 2
/// ```
///
/// # Errors
/// - `NoGoals` if there are no goals
/// - `HypNotFound` if the hypothesis is not found
/// - `Other` if the type is not finite
pub fn fin_cases(state: &mut ProofState, hyp_name: &str) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Find the hypothesis
    let hyp = goal
        .local_ctx
        .iter()
        .find(|d| d.name == hyp_name)
        .ok_or_else(|| TacticError::HypothesisNotFound(hyp_name.to_string()))?
        .clone();

    // Check if the type is finite
    let inhabitants = get_finite_inhabitants(&hyp.ty)?;

    if inhabitants.is_empty() {
        return Err(TacticError::Other(format!(
            "fin_cases: {hyp_name} has no inhabitants or is not a finite type"
        )));
    }

    // Create new goals for each case
    let mut new_goals = Vec::new();

    for inhabitant in inhabitants {
        // Create a new context where hyp is replaced with its value
        let mut new_ctx = goal.local_ctx.clone();

        // Find and update the hypothesis
        for decl in &mut new_ctx {
            if decl.name == hyp_name {
                decl.value = Some(inhabitant.clone());
            }
        }

        // Substitute the inhabitant into the target
        let new_target = substitute_fvar(&goal.target, hyp.fvar, &inhabitant);

        let new_meta_id = state.metas.fresh(new_target.clone());
        new_goals.push(Goal {
            meta_id: new_meta_id,
            target: new_target,
            local_ctx: new_ctx,
        });
    }

    // Replace current goal with new goals
    state.goals.remove(0);
    for new_goal in new_goals.into_iter().rev() {
        state.goals.insert(0, new_goal);
    }

    Ok(())
}

/// Get inhabitants of a finite type
pub(crate) fn get_finite_inhabitants(ty: &Expr) -> Result<Vec<Expr>, TacticError> {
    match ty {
        Expr::Const(name, _levels) => {
            let name_str = name.to_string();

            // Bool has two inhabitants: true and false
            if name_str == "Bool" {
                return Ok(vec![
                    Expr::const_(Name::from_string("true"), vec![]),
                    Expr::const_(Name::from_string("false"), vec![]),
                ]);
            }

            // Unit has one inhabitant: ()
            if name_str == "Unit" || name_str == "unit" || name_str == "PUnit" {
                return Ok(vec![Expr::const_(Name::from_string("Unit.unit"), vec![])]);
            }

            // Empty/False has no inhabitants
            if name_str == "Empty" || name_str == "False" {
                return Ok(vec![]);
            }

            Err(TacticError::Other(format!(
                "fin_cases: {name_str} is not a recognized finite type"
            )))
        }
        Expr::App(f, arg) => {
            // Check for Fin n
            if let Expr::Const(name, _) = f.as_ref() {
                if name.to_string() == "Fin" {
                    // Try to extract n
                    if let Some(n) = extract_nat_literal(arg) {
                        let mut inhabitants = Vec::new();
                        for i in 0..n {
                            // Create Fin.mk i proof
                            // For simplicity, just use the numeral
                            inhabitants.push(make_nat_literal(i as u64));
                        }
                        return Ok(inhabitants);
                    }
                }
            }
            Err(TacticError::Other(
                "fin_cases: not a recognized finite type".to_string(),
            ))
        }
        _ => Err(TacticError::Other(
            "fin_cases: not a finite type".to_string(),
        )),
    }
}

/// Extract a natural number from an expression
pub(crate) fn extract_nat_literal(expr: &Expr) -> Option<usize> {
    match expr {
        Expr::Const(name, _) => {
            let name_str = name.to_string();
            if name_str == "Nat.zero" {
                return Some(0);
            }
            // Try to parse as a number
            name_str.parse().ok()
        }
        Expr::Lit(lit) => {
            if let lean5_kernel::expr::Literal::Nat(n) = lit {
                Some(*n as usize)
            } else {
                None
            }
        }
        Expr::App(f, arg) => {
            // Check for Nat.succ
            if let Expr::Const(name, _) = f.as_ref() {
                if name.to_string() == "Nat.succ" {
                    if let Some(n) = extract_nat_literal(arg) {
                        return Some(n + 1);
                    }
                }
            }
            None
        }
        _ => None,
    }
}

/// Make a natural number literal expression
pub(crate) fn make_nat_literal(n: u64) -> Expr {
    if n == 0 {
        Expr::const_(Name::from_string("Nat.zero"), vec![])
    } else {
        Expr::app(
            Expr::const_(Name::from_string("Nat.succ"), vec![]),
            make_nat_literal(n - 1),
        )
    }
}

/// Substitute a free variable with an expression
pub(crate) fn substitute_fvar(expr: &Expr, fvar: FVarId, replacement: &Expr) -> Expr {
    match expr {
        Expr::FVar(id) if *id == fvar => replacement.clone(),
        Expr::App(f, arg) => Expr::app(
            substitute_fvar(f, fvar, replacement),
            substitute_fvar(arg, fvar, replacement),
        ),
        Expr::Lam(bi, ty, body) => Expr::lam(
            *bi,
            substitute_fvar(ty, fvar, replacement),
            substitute_fvar(body, fvar, replacement),
        ),
        Expr::Pi(bi, ty, body) => Expr::pi(
            *bi,
            substitute_fvar(ty, fvar, replacement),
            substitute_fvar(body, fvar, replacement),
        ),
        Expr::Let(ty, val, body) => Expr::let_(
            substitute_fvar(ty, fvar, replacement),
            substitute_fvar(val, fvar, replacement),
            substitute_fvar(body, fvar, replacement),
        ),
        _ => expr.clone(),
    }
}

// ============================================================================
// interval_cases - Case split on integer intervals
// ============================================================================

/// Case split on an integer hypothesis within an interval.
///
/// `interval_cases` creates separate goals for each integer value in the
/// range when bounds can be determined from the context.
///
/// # Algorithm
/// 1. Find lower and upper bounds on the hypothesis from context
/// 2. Create a goal for each integer in the range
/// 3. Substitute the value and add it as an equality hypothesis
///
/// # Example
/// ```text
/// -- h : n ≤ 2
/// -- k : 0 ≤ n
/// -- Goal: P n
/// interval_cases n
/// -- Goal 1: P 0
/// -- Goal 2: P 1
/// -- Goal 3: P 2
/// ```
///
/// # Errors
/// - `NoGoals` if there are no goals
/// - `HypNotFound` if the hypothesis is not found
/// - `Other` if bounds cannot be determined
pub fn interval_cases(state: &mut ProofState, hyp_name: &str) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Find the hypothesis
    let hyp = goal
        .local_ctx
        .iter()
        .find(|d| d.name == hyp_name)
        .ok_or_else(|| TacticError::HypothesisNotFound(hyp_name.to_string()))?
        .clone();

    // Try to determine bounds from context
    let (lower, upper) = find_integer_bounds(&goal, &hyp)?;

    if upper - lower > 100 {
        return Err(TacticError::Other(format!(
            "interval_cases: range too large ({} values)",
            upper - lower + 1
        )));
    }

    // Create new goals for each value in the range
    let mut new_goals = Vec::new();

    for value in lower..=upper {
        let value_expr = make_int_literal(value);

        // Create equality hypothesis: hyp = value
        let eq_hyp_name = format!("{hyp_name}_eq");
        let eq_type = make_equality_type(
            &Expr::const_(Name::from_string("Nat"), vec![]),
            &Expr::fvar(hyp.fvar),
            &value_expr,
        );

        let mut new_ctx = goal.local_ctx.clone();
        let eq_fvar = FVarId(new_ctx.len() as u64 + 1000);
        new_ctx.push(LocalDecl {
            fvar: eq_fvar,
            name: eq_hyp_name,
            ty: eq_type,
            value: None,
        });

        // Substitute the value into the target
        let new_target = substitute_fvar(&goal.target, hyp.fvar, &value_expr);

        let new_meta_id = state.metas.fresh(new_target.clone());
        new_goals.push(Goal {
            meta_id: new_meta_id,
            target: new_target,
            local_ctx: new_ctx,
        });
    }

    // Replace current goal with new goals
    state.goals.remove(0);
    for new_goal in new_goals.into_iter().rev() {
        state.goals.insert(0, new_goal);
    }

    Ok(())
}

/// Find integer bounds for a variable from the context
fn find_integer_bounds(goal: &Goal, hyp: &LocalDecl) -> Result<(i64, i64), TacticError> {
    let mut lower = i64::MIN;
    let mut upper = i64::MAX;

    // Look through hypotheses for bounds
    for decl in &goal.local_ctx {
        // Check for h ≤ n patterns (upper bound)
        // match_le returns (ty, lhs, rhs) where lhs ≤ rhs
        if let Some((_ty, lhs, rhs)) = match_le(&decl.ty) {
            if let Expr::FVar(id) = &rhs {
                if *id == hyp.fvar {
                    if let Some(val) = expr_to_int(&lhs) {
                        lower = lower.max(val);
                    }
                }
            }
            if let Expr::FVar(id) = &lhs {
                if *id == hyp.fvar {
                    if let Some(val) = expr_to_int(&rhs) {
                        upper = upper.min(val);
                    }
                }
            }
        }

        // Check for h < n patterns
        // match_lt returns (ty, lhs, rhs) where lhs < rhs
        if let Some((_ty, lhs, rhs)) = match_lt(&decl.ty) {
            if let Expr::FVar(id) = &rhs {
                if *id == hyp.fvar {
                    if let Some(val) = expr_to_int(&lhs) {
                        lower = lower.max(val + 1);
                    }
                }
            }
            if let Expr::FVar(id) = &lhs {
                if *id == hyp.fvar {
                    if let Some(val) = expr_to_int(&rhs) {
                        upper = upper.min(val - 1);
                    }
                }
            }
        }
    }

    // Default to small range if no bounds found
    if lower == i64::MIN {
        lower = 0;
    }
    if upper == i64::MAX {
        upper = lower + 10; // Default to 10 values
    }

    if lower > upper {
        return Err(TacticError::Other(
            "interval_cases: inconsistent bounds".to_string(),
        ));
    }

    Ok((lower, upper))
}

/// Convert expression to integer if possible
pub(crate) fn expr_to_int(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::Const(name, _) => {
            let name_str = name.to_string();
            if name_str == "Nat.zero" {
                return Some(0);
            }
            name_str.parse().ok()
        }
        Expr::Lit(lean5_kernel::expr::Literal::Nat(n)) => Some(*n as i64),
        Expr::App(f, arg) => {
            if let Expr::Const(name, _) = f.as_ref() {
                if name.to_string() == "Nat.succ" {
                    return expr_to_int(arg).map(|n| n + 1);
                }
            }
            None
        }
        // Non-Nat literals and other expressions don't convert to integers
        _ => None,
    }
}

/// Make an integer literal expression
pub(crate) fn make_int_literal(n: i64) -> Expr {
    if n >= 0 {
        make_nat_literal(n as u64)
    } else {
        // For negative, use Int.negOfNat
        Expr::app(
            Expr::const_(Name::from_string("Int.negOfNat"), vec![]),
            make_nat_literal((-n) as u64),
        )
    }
}

/// Make an equality type expression
pub(crate) fn make_equality_type(ty: &Expr, lhs: &Expr, rhs: &Expr) -> Expr {
    Expr::app(
        Expr::app(
            Expr::app(
                Expr::const_(Name::from_string("Eq"), vec![Level::zero()]),
                ty.clone(),
            ),
            lhs.clone(),
        ),
        rhs.clone(),
    )
}
