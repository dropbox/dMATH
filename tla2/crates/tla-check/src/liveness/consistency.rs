//! Consistency checking for liveness
//!
//! This module implements checking whether a state is consistent with a tableau node.
//! A state is consistent with a tableau node iff all state predicates in the particle
//! evaluate to TRUE when the state variables are bound.
//!
//! # TLC Reference
//!
//! This follows TLC's `TBGraphNode.isConsistent()` method which checks whether
//! a state satisfies all the state-level formulas in a tableau node.

use super::live_expr::{ExprLevel, LiveExpr};
use super::tableau::TableauNode;
use crate::error::EvalResult;
use crate::eval::{eval, Env, EvalCtx};
use crate::state::State;
use crate::Value;
use std::sync::Arc;

/// Check if a state is consistent with a tableau node
///
/// A state is consistent with a tableau node iff all state predicates
/// in the node evaluate to TRUE with the state's variable bindings.
///
/// # Arguments
///
/// * `ctx` - Base evaluation context (with operators loaded)
/// * `state` - The TLA+ state to check
/// * `node` - The tableau node to check against
/// * `get_successors` - Callback to enumerate successor states (for ENABLED)
///
/// # Returns
///
/// * `Ok(true)` if the state is consistent with the tableau node
/// * `Ok(false)` if any state predicate evaluates to FALSE
/// * `Err(...)` if evaluation fails
pub fn is_state_consistent<F>(
    ctx: &EvalCtx,
    state: &State,
    node: &TableauNode,
    get_successors: &mut F,
) -> EvalResult<bool>
where
    F: FnMut(&State) -> EvalResult<Vec<State>>,
{
    // Bind state variables to the context
    let mut state_ctx = ctx.clone();
    for (name, value) in state.vars() {
        state_ctx.bind_mut(Arc::clone(name), value.clone());
    }

    // Check each state predicate in the tableau node
    let debug = std::env::var("TLA2_DEBUG_CONSISTENCY").is_ok();
    for (i, pred) in node.state_preds().iter().enumerate() {
        match eval_live_expr(&state_ctx, pred, state, None, get_successors)? {
            Value::Bool(true) => {}
            Value::Bool(false) => {
                if debug {
                    eprintln!(
                        "[CONSISTENCY FAIL] state_fp={:016x} tableau_idx={} pred_idx={} pred={}",
                        state.fingerprint().0,
                        node.index(),
                        i,
                        pred
                    );
                }
                return Ok(false);
            }
            other => {
                if debug {
                    eprintln!(
                        "[CONSISTENCY FAIL] state_fp={:016x} tableau_idx={} pred_idx={} pred={} non-bool={:?}",
                        state.fingerprint().0,
                        node.index(),
                        i,
                        pred,
                        other
                    );
                }
                return Ok(false); // Non-boolean is treated as inconsistent
            }
        }
    }

    Ok(true)
}

/// Check if a state transition is consistent with a tableau node
///
/// This is used for action predicates that depend on both current and next state.
///
/// # Arguments
///
/// * `ctx` - Base evaluation context
/// * `current_state` - The current TLA+ state
/// * `next_state` - The next TLA+ state (for primed variables)
/// * `node` - The tableau node containing action predicates
/// * `get_successors` - Callback to enumerate successor states (for ENABLED)
///
/// # Returns
///
/// * `Ok(true)` if the transition is consistent
/// * `Ok(false)` if any action predicate evaluates to FALSE
/// * `Err(...)` if evaluation fails
pub fn is_transition_consistent<F>(
    ctx: &EvalCtx,
    current_state: &State,
    next_state: &State,
    node: &TableauNode,
    get_successors: &mut F,
) -> EvalResult<bool>
where
    F: FnMut(&State) -> EvalResult<Vec<State>>,
{
    // Bind current state variables
    let mut trans_ctx = ctx.clone();
    for (name, value) in current_state.vars() {
        trans_ctx.bind_mut(Arc::clone(name), value.clone());
    }

    // Set up next-state bindings for primed variables
    let mut next_env = Env::new();
    for (name, value) in next_state.vars() {
        next_env.insert(Arc::clone(name), value.clone());
    }
    let trans_ctx = trans_ctx.with_next_state(next_env);

    // Check all predicates in the particle
    for formula in node.particle().formulas() {
        match eval_live_expr(
            &trans_ctx,
            formula,
            current_state,
            Some(next_state),
            get_successors,
        )? {
            Value::Bool(true) => {}
            Value::Bool(false) => return Ok(false),
            other => {
                // Non-boolean at state/action level is an error
                if matches!(
                    formula.level(),
                    ExprLevel::State | ExprLevel::Action | ExprLevel::Constant
                ) {
                    return Ok(false);
                }
                // For temporal level, we don't evaluate here
                if !matches!(other, Value::Bool(_)) && formula.level() != ExprLevel::Temporal {
                    return Ok(false);
                }
            }
        }
    }

    Ok(true)
}

/// Evaluate a LiveExpr in a state context
///
/// This evaluates state predicates, action predicates, and boolean combinations.
/// Temporal operators (Always, Eventually, Next) are not evaluated here -
/// they are handled by the tableau structure itself.
///
/// # Arguments
///
/// * `ctx` - Evaluation context with state variables bound
/// * `expr` - The LiveExpr to evaluate
/// * `current_state` - Current state (for ENABLED checking)
/// * `next_state` - Next state (for action predicates, if available)
/// * `get_successors` - Callback to enumerate successor states (for ENABLED)
pub fn eval_live_expr<F>(
    ctx: &EvalCtx,
    expr: &LiveExpr,
    current_state: &State,
    next_state: Option<&State>,
    get_successors: &mut F,
) -> EvalResult<Value>
where
    F: FnMut(&State) -> EvalResult<Vec<State>>,
{
    match expr {
        LiveExpr::Bool(b) => Ok(Value::Bool(*b)),

        LiveExpr::StatePred { expr, .. } => {
            // Evaluate the AST expression in the current context
            eval(ctx, expr)
        }

        LiveExpr::ActionPred { expr, .. } => {
            // Action predicates require next state for primed variables
            // If no next state, we can't evaluate - return false
            if next_state.is_none() && contains_primed(expr) {
                // Cannot evaluate action predicate without next state
                // This is conservative - treat as not satisfied
                return Ok(Value::Bool(false));
            }
            eval(ctx, expr)
        }

        LiveExpr::Enabled {
            action,
            require_state_change,
            ..
        } => {
            // ENABLED(A) is true iff there exists a next state s' such that A(s, s') is true.
            //
            // For fairness constraints (WF_vars / SF_vars), TLC defines ENABLED in terms of
            // `ENABLED(<<A>>_vars)` which requires a *non-stuttering* successor. Using the
            // pre-computed successor set can be unsound (see #55), so prefer fresh enumeration
            // when the VarRegistry is available.
            let vars: Vec<Arc<str>> = ctx.shared.var_registry.names().to_vec();
            if !vars.is_empty() {
                let mut eval_ctx = ctx.clone();
                eval_ctx.next_state = None;

                if *require_state_change {
                    let successors = crate::enumerate::enumerate_action_successors(
                        &mut eval_ctx,
                        action,
                        current_state,
                        &vars,
                    )?;
                    let current_fp = current_state.fingerprint();
                    return Ok(Value::Bool(
                        successors
                            .iter()
                            .any(|s| s.fingerprint() != current_fp),
                    ));
                }

                return Ok(Value::Bool(crate::enumerate::eval_enabled(
                    &mut eval_ctx,
                    action,
                    &vars,
                )?));
            }

            // Fallback: var_registry is empty (synthetic tests). Use pre-computed successors.
            let successors = get_successors(current_state)?;
            for succ_state in successors {
                let mut next_env = Env::new();
                for (name, value) in succ_state.vars() {
                    next_env.insert(Arc::clone(name), value.clone());
                }
                let ctx = ctx.clone().with_next_state(next_env);
                match eval(&ctx, action)? {
                    Value::Bool(true) => return Ok(Value::Bool(true)),
                    Value::Bool(false) => {}
                    _ => {}
                }
            }
            Ok(Value::Bool(false))
        }

        LiveExpr::StateChanged { .. } => {
            // StateChanged is true iff vars' ≠ vars (non-stuttering transition)
            // For consistency checking at state level, we need a next state to evaluate
            match next_state {
                Some(ns) => Ok(Value::Bool(current_state.fingerprint() != ns.fingerprint())),
                None => Ok(Value::Bool(false)), // No next state means we can't evaluate state change
            }
        }

        LiveExpr::Not(inner) => {
            let v = eval_live_expr(ctx, inner, current_state, next_state, get_successors)?;
            match v {
                Value::Bool(b) => Ok(Value::Bool(!b)),
                _ => Ok(Value::Bool(false)), // Non-boolean negation is false
            }
        }

        LiveExpr::And(exprs) => {
            for e in exprs {
                match eval_live_expr(ctx, e, current_state, next_state, get_successors)? {
                    Value::Bool(true) => {}
                    Value::Bool(false) => return Ok(Value::Bool(false)),
                    _ => return Ok(Value::Bool(false)),
                }
            }
            Ok(Value::Bool(true))
        }

        LiveExpr::Or(exprs) => {
            for e in exprs {
                match eval_live_expr(ctx, e, current_state, next_state, get_successors)? {
                    Value::Bool(true) => return Ok(Value::Bool(true)),
                    Value::Bool(false) => {}
                    _ => {}
                }
            }
            Ok(Value::Bool(false))
        }

        // Temporal operators are not evaluated at the state level
        // They are handled by the tableau structure
        LiveExpr::Always(_) | LiveExpr::Eventually(_) | LiveExpr::Next(_) => {
            // Return true to indicate "not yet determined"
            // The actual checking happens through tableau traversal
            Ok(Value::Bool(true))
        }
    }
}

/// Check if an expression contains primed variables
fn contains_primed(expr: &tla_core::Spanned<tla_core::ast::Expr>) -> bool {
    use tla_core::ast::Expr;
    match &expr.node {
        // Primed expression
        Expr::Prime(_) => true,

        // Literals - no primed vars
        Expr::Bool(_) | Expr::Int(_) | Expr::String(_) | Expr::Ident(_) | Expr::OpRef(_) => false,

        // Binary operations
        Expr::And(l, r)
        | Expr::Or(l, r)
        | Expr::Implies(l, r)
        | Expr::Equiv(l, r)
        | Expr::In(l, r)
        | Expr::NotIn(l, r)
        | Expr::Subseteq(l, r)
        | Expr::Union(l, r)
        | Expr::Intersect(l, r)
        | Expr::SetMinus(l, r)
        | Expr::FuncApply(l, r)
        | Expr::FuncSet(l, r)
        | Expr::Eq(l, r)
        | Expr::Neq(l, r)
        | Expr::Lt(l, r)
        | Expr::Leq(l, r)
        | Expr::Gt(l, r)
        | Expr::Geq(l, r)
        | Expr::Add(l, r)
        | Expr::Sub(l, r)
        | Expr::Mul(l, r)
        | Expr::Div(l, r)
        | Expr::IntDiv(l, r)
        | Expr::Mod(l, r)
        | Expr::Pow(l, r)
        | Expr::Range(l, r)
        | Expr::LeadsTo(l, r)
        | Expr::WeakFair(l, r)
        | Expr::StrongFair(l, r) => contains_primed(l) || contains_primed(r),

        // Unary operations
        Expr::Not(e)
        | Expr::Powerset(e)
        | Expr::BigUnion(e)
        | Expr::Domain(e)
        | Expr::Always(e)
        | Expr::Eventually(e)
        | Expr::Enabled(e)
        | Expr::Unchanged(e)
        | Expr::Neg(e) => contains_primed(e),

        // Function and set operations
        Expr::Apply(func, args) => contains_primed(func) || args.iter().any(contains_primed),
        Expr::ModuleRef(_, _, args) => args.iter().any(contains_primed),
        Expr::InstanceExpr(_, _) => false, // Instance expressions are definitions
        Expr::Lambda(_, body) => contains_primed(body),

        // Quantifiers
        Expr::Forall(bounds, body) | Expr::Exists(bounds, body) => {
            bounds
                .iter()
                .any(|b| b.domain.as_ref().is_some_and(|d| contains_primed(d)))
                || contains_primed(body)
        }
        Expr::Choose(bound, body) => {
            bound.domain.as_ref().is_some_and(|d| contains_primed(d)) || contains_primed(body)
        }

        // Sets
        Expr::SetEnum(elems) => elems.iter().any(contains_primed),
        Expr::SetBuilder(body, bounds) => {
            contains_primed(body)
                || bounds
                    .iter()
                    .any(|b| b.domain.as_ref().is_some_and(|d| contains_primed(d)))
        }
        Expr::SetFilter(bound, body) => {
            bound.domain.as_ref().is_some_and(|d| contains_primed(d)) || contains_primed(body)
        }

        // Functions
        Expr::FuncDef(bounds, body) => {
            bounds
                .iter()
                .any(|b| b.domain.as_ref().is_some_and(|d| contains_primed(d)))
                || contains_primed(body)
        }
        Expr::Except(base, specs) => {
            contains_primed(base)
                || specs.iter().any(|s| {
                    contains_primed(&s.value)
                        || s.path.iter().any(|p| match p {
                            tla_core::ast::ExceptPathElement::Index(i) => contains_primed(i),
                            tla_core::ast::ExceptPathElement::Field(_) => false,
                        })
                })
        }

        // Records
        Expr::Record(fields) => fields.iter().any(|(_, e)| contains_primed(e)),
        Expr::RecordAccess(e, _) => contains_primed(e),
        Expr::RecordSet(fields) => fields.iter().any(|(_, e)| contains_primed(e)),

        // Tuples
        Expr::Tuple(elems) | Expr::Times(elems) => elems.iter().any(contains_primed),

        // Control flow
        Expr::If(cond, then_e, else_e) => {
            contains_primed(cond) || contains_primed(then_e) || contains_primed(else_e)
        }
        Expr::Case(arms, other) => {
            arms.iter()
                .any(|arm| contains_primed(&arm.guard) || contains_primed(&arm.body))
                || other.as_ref().is_some_and(|o| contains_primed(o))
        }
        Expr::Let(defs, body) => {
            defs.iter().any(|d| contains_primed(&d.body)) || contains_primed(body)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liveness::LiveExpr;
    use std::sync::Arc;
    use tla_core::ast::Expr;
    use tla_core::Spanned;

    fn empty_successors(_: &State) -> EvalResult<Vec<State>> {
        Ok(Vec::new())
    }

    #[test]
    fn test_eval_live_expr_bool() {
        let ctx = EvalCtx::new();
        let state = State::new();
        let mut get_successors = empty_successors;

        assert_eq!(
            eval_live_expr(
                &ctx,
                &LiveExpr::Bool(true),
                &state,
                None,
                &mut get_successors
            )
            .unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            eval_live_expr(
                &ctx,
                &LiveExpr::Bool(false),
                &state,
                None,
                &mut get_successors
            )
            .unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn test_eval_live_expr_not() {
        let ctx = EvalCtx::new();
        let state = State::new();
        let mut get_successors = empty_successors;

        let not_true = LiveExpr::not(LiveExpr::Bool(true));
        let not_false = LiveExpr::not(LiveExpr::Bool(false));

        assert_eq!(
            eval_live_expr(&ctx, &not_true, &state, None, &mut get_successors).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            eval_live_expr(&ctx, &not_false, &state, None, &mut get_successors).unwrap(),
            Value::Bool(true)
        );
    }

    #[test]
    fn test_eval_live_expr_and() {
        let ctx = EvalCtx::new();
        let state = State::new();
        let mut get_successors = empty_successors;

        let tt = LiveExpr::and(vec![LiveExpr::Bool(true), LiveExpr::Bool(true)]);
        let tf = LiveExpr::and(vec![LiveExpr::Bool(true), LiveExpr::Bool(false)]);
        let ff = LiveExpr::and(vec![LiveExpr::Bool(false), LiveExpr::Bool(false)]);

        assert_eq!(
            eval_live_expr(&ctx, &tt, &state, None, &mut get_successors).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            eval_live_expr(&ctx, &tf, &state, None, &mut get_successors).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            eval_live_expr(&ctx, &ff, &state, None, &mut get_successors).unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn test_eval_live_expr_or() {
        let ctx = EvalCtx::new();
        let state = State::new();
        let mut get_successors = empty_successors;

        let tt = LiveExpr::or(vec![LiveExpr::Bool(true), LiveExpr::Bool(true)]);
        let tf = LiveExpr::or(vec![LiveExpr::Bool(true), LiveExpr::Bool(false)]);
        let ff = LiveExpr::or(vec![LiveExpr::Bool(false), LiveExpr::Bool(false)]);

        assert_eq!(
            eval_live_expr(&ctx, &tt, &state, None, &mut get_successors).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            eval_live_expr(&ctx, &tf, &state, None, &mut get_successors).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            eval_live_expr(&ctx, &ff, &state, None, &mut get_successors).unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn test_eval_live_expr_state_pred_with_binding() {
        // Create a state predicate that checks x > 0
        let x_gt_0 = Expr::Gt(
            Box::new(Spanned::dummy(Expr::Ident("x".to_string()))),
            Box::new(Spanned::dummy(Expr::Int(0.into()))),
        );
        let pred = LiveExpr::StatePred {
            expr: Arc::new(Spanned::dummy(x_gt_0)),
            tag: 1,
        };

        let ctx = EvalCtx::new();
        let mut get_successors = empty_successors;

        // State where x = 5 (should be true)
        let state_pos = State::from_pairs([("x", Value::int(5))]);
        let mut ctx_pos = ctx.clone();
        ctx_pos.bind_mut("x".to_string(), Value::int(5));
        assert_eq!(
            eval_live_expr(&ctx_pos, &pred, &state_pos, None, &mut get_successors).unwrap(),
            Value::Bool(true)
        );

        // State where x = -1 (should be false)
        let state_neg = State::from_pairs([("x", Value::int(-1))]);
        let mut ctx_neg = ctx.clone();
        ctx_neg.bind_mut("x".to_string(), Value::int(-1));
        assert_eq!(
            eval_live_expr(&ctx_neg, &pred, &state_neg, None, &mut get_successors).unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn test_temporal_returns_true() {
        let ctx = EvalCtx::new();
        let state = State::new();
        let mut get_successors = empty_successors;

        // Temporal operators return true (not yet determined)
        let always = LiveExpr::always(LiveExpr::Bool(false));
        let eventually = LiveExpr::eventually(LiveExpr::Bool(false));
        let next = LiveExpr::next(LiveExpr::Bool(false));

        // These should return true because temporal checking happens via tableau
        assert_eq!(
            eval_live_expr(&ctx, &always, &state, None, &mut get_successors).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            eval_live_expr(&ctx, &eventually, &state, None, &mut get_successors).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            eval_live_expr(&ctx, &next, &state, None, &mut get_successors).unwrap(),
            Value::Bool(true)
        );
    }

    #[test]
    fn test_action_pred_with_primed_vars_no_next_state() {
        // Test that action predicates with primed variables return false when no next state
        let ctx = EvalCtx::new();
        let state = State::from_pairs([("x", Value::int(5))]);
        let mut get_successors = empty_successors;

        // Create action predicate: x' = x + 1 (contains primed variable)
        let x = Spanned::dummy(Expr::Ident("x".to_string()));
        let x_prime = Spanned::dummy(Expr::Prime(Box::new(x.clone())));
        let x_plus_1 = Spanned::dummy(Expr::Add(
            Box::new(x),
            Box::new(Spanned::dummy(Expr::Int(1.into()))),
        ));
        let action_expr = Arc::new(Spanned::dummy(Expr::Eq(
            Box::new(x_prime),
            Box::new(x_plus_1),
        )));

        let action_pred = LiveExpr::action_pred(action_expr, 1);

        // With no next state, action pred with primed vars should return false
        let mut ctx_with_x = ctx.clone();
        ctx_with_x.bind_mut("x".to_string(), Value::int(5));
        let result =
            eval_live_expr(&ctx_with_x, &action_pred, &state, None, &mut get_successors).unwrap();
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn test_action_pred_with_primed_vars_with_next_state() {
        // Test that action predicates with primed variables work when next state is provided
        let ctx = EvalCtx::new();
        let current_state = State::from_pairs([("x", Value::int(5))]);
        let next_state = State::from_pairs([("x", Value::int(6))]);
        let mut get_successors = empty_successors;

        // Create action predicate: x' = x + 1
        let x = Spanned::dummy(Expr::Ident("x".to_string()));
        let x_prime = Spanned::dummy(Expr::Prime(Box::new(x.clone())));
        let x_plus_1 = Spanned::dummy(Expr::Add(
            Box::new(x),
            Box::new(Spanned::dummy(Expr::Int(1.into()))),
        ));
        let action_expr = Arc::new(Spanned::dummy(Expr::Eq(
            Box::new(x_prime),
            Box::new(x_plus_1),
        )));

        let action_pred = LiveExpr::action_pred(action_expr, 1);

        // Set up context with current state variables
        let mut ctx_current = ctx.clone();
        ctx_current.bind_mut("x", Value::int(5));

        // Set up next state bindings
        let mut next_env = Env::new();
        next_env.insert(Arc::from("x"), Value::int(6));
        let ctx_with_next = ctx_current.with_next_state(next_env);

        // With next state context, should evaluate correctly: 6 = 5 + 1 = true
        let result = eval_live_expr(
            &ctx_with_next,
            &action_pred,
            &current_state,
            Some(&next_state),
            &mut get_successors,
        )
        .unwrap();
        assert_eq!(result, Value::Bool(true));

        // Test with non-matching next state (x' = 7, but we expect x + 1 = 6)
        let wrong_next_state = State::from_pairs([("x", Value::int(7))]);
        let mut wrong_next_env = Env::new();
        wrong_next_env.insert(Arc::from("x"), Value::int(7));
        let ctx_wrong_next = ctx_current.clone().with_next_state(wrong_next_env);

        let result = eval_live_expr(
            &ctx_wrong_next,
            &action_pred,
            &current_state,
            Some(&wrong_next_state),
            &mut get_successors,
        )
        .unwrap();
        assert_eq!(result, Value::Bool(false)); // 7 != 5 + 1
    }

    #[test]
    fn test_eval_live_expr_enabled_true_when_action_satisfiable() {
        let ctx = EvalCtx::new();
        let current_state = State::from_pairs([("x", Value::int(5))]);

        // Create action: x' = x + 1
        let x = Spanned::dummy(Expr::Ident("x".to_string()));
        let x_prime = Spanned::dummy(Expr::Prime(Box::new(x.clone())));
        let x_plus_1 = Spanned::dummy(Expr::Add(
            Box::new(x),
            Box::new(Spanned::dummy(Expr::Int(1.into()))),
        ));
        let action_expr = Arc::new(Spanned::dummy(Expr::Eq(
            Box::new(x_prime),
            Box::new(x_plus_1),
        )));

        let enabled = LiveExpr::enabled(action_expr, 1);

        // Bind current state variable in ctx.
        let mut ctx_current = ctx.clone();
        ctx_current.bind_mut("x".to_string(), Value::int(5));

        // Provide a successor where the action holds (x' = 6).
        let mut get_successors = |_s: &State| Ok(vec![State::from_pairs([("x", Value::int(6))])]);

        let result = eval_live_expr(
            &ctx_current,
            &enabled,
            &current_state,
            None,
            &mut get_successors,
        )
        .unwrap();
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn test_eval_live_expr_enabled_false_when_no_successor_satisfies_action() {
        let ctx = EvalCtx::new();
        let current_state = State::from_pairs([("x", Value::int(5))]);

        // Create action: x' = x + 1
        let x = Spanned::dummy(Expr::Ident("x".to_string()));
        let x_prime = Spanned::dummy(Expr::Prime(Box::new(x.clone())));
        let x_plus_1 = Spanned::dummy(Expr::Add(
            Box::new(x),
            Box::new(Spanned::dummy(Expr::Int(1.into()))),
        ));
        let action_expr = Arc::new(Spanned::dummy(Expr::Eq(
            Box::new(x_prime),
            Box::new(x_plus_1),
        )));

        let enabled = LiveExpr::enabled(action_expr, 1);

        // Bind current state variable in ctx.
        let mut ctx_current = ctx.clone();
        ctx_current.bind_mut("x".to_string(), Value::int(5));

        // Provide a successor where the action does NOT hold (x' = 7).
        let mut get_successors = |_s: &State| Ok(vec![State::from_pairs([("x", Value::int(7))])]);

        let result = eval_live_expr(
            &ctx_current,
            &enabled,
            &current_state,
            None,
            &mut get_successors,
        )
        .unwrap();
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn test_is_transition_consistent_with_action_pred() {
        use crate::liveness::tableau::{Particle, TableauNode};

        let ctx = EvalCtx::new();
        let current_state = State::from_pairs([("x", Value::int(5))]);
        let next_state = State::from_pairs([("x", Value::int(6))]);
        let mut get_successors = empty_successors;

        // Create action predicate: x' = x + 1
        let x = Spanned::dummy(Expr::Ident("x".to_string()));
        let x_prime = Spanned::dummy(Expr::Prime(Box::new(x.clone())));
        let x_plus_1 = Spanned::dummy(Expr::Add(
            Box::new(x),
            Box::new(Spanned::dummy(Expr::Int(1.into()))),
        ));
        let action_expr = Arc::new(Spanned::dummy(Expr::Eq(
            Box::new(x_prime),
            Box::new(x_plus_1),
        )));
        let action_pred = LiveExpr::action_pred(action_expr, 1);

        // Create a particle with this action predicate
        let particle = Particle::from_vec(vec![action_pred]);
        let node = TableauNode::new(particle, 0);

        // Transition from x=5 to x=6 should satisfy x' = x + 1
        let result = is_transition_consistent(
            &ctx,
            &current_state,
            &next_state,
            &node,
            &mut get_successors,
        )
        .unwrap();
        assert!(result);

        // Transition from x=5 to x=7 should NOT satisfy x' = x + 1
        let wrong_next_state = State::from_pairs([("x", Value::int(7))]);
        let result = is_transition_consistent(
            &ctx,
            &current_state,
            &wrong_next_state,
            &node,
            &mut get_successors,
        )
        .unwrap();
        assert!(!result);
    }
}
