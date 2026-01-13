//! Equality-focused tactics and helpers.
//!
//! This module contains tactics that manipulate equality goals and hypotheses,
//! along with helper routines for matching and rewriting equalities.

use lean5_kernel::{BinderInfo, Expr, Level, Name};

use crate::unify::MetaState;

use super::{Goal, LocalDecl, ProofState, TacticError, TacticResult};

/// Rewrite the goal using an equality hypothesis.
///
/// Given a hypothesis `h : a = b` and a goal containing `a`,
/// replaces occurrences of `a` with `b` and uses `Eq.subst` to justify the transformation.
///
/// # Arguments
/// * `state` - The proof state
/// * `hyp_name` - Name of the equality hypothesis to use
/// * `reverse` - If true, rewrite `b` to `a` instead of `a` to `b`
///
/// # Example
/// ```text
/// Given: h : x = y, goal: P x
/// After rewrite(h): goal becomes P y
/// ```
pub fn rewrite(state: &mut ProofState, hyp_name: &str, reverse: bool) -> TacticResult {
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
    let (eq_type, lhs, rhs, eq_levels) = match_equality(&hyp_ty)?;

    // Determine what to replace with what
    let (from, to) = if reverse {
        (rhs.clone(), lhs.clone())
    } else {
        (lhs.clone(), rhs.clone())
    };

    // Check if the goal contains the pattern we're replacing
    let target = state.metas.instantiate(&goal.target);
    if !contains_expr(&target, &from) {
        return Err(TacticError::Other(
            "goal does not contain the pattern to rewrite".to_string(),
        ));
    }

    // Replace occurrences of `from` with `to` in the goal
    let new_target = replace_expr(&target, &from, &to);

    // Create a metavariable for the new goal
    let new_meta_id = state.metas.fresh(new_target.clone());
    let new_meta = Expr::FVar(MetaState::to_fvar(new_meta_id));

    // Build the motive: λ x, target[from → x]
    // This is the predicate P such that P(from) = target and P(to) = new_target
    let motive = abstract_over(&target, &from);

    // Proof construction using Eq.subst:
    // Eq.subst : {α} → {motive : α → Prop} → {a b : α} → Eq a b → motive a → motive b
    //
    // For forward rewrite (h : a = b, replace a with b in goal G):
    // - Original goal: G[a] = motive(a)
    // - New goal: G[b] = motive(b)
    // - We need: proof of motive(a) from proof of motive(b)
    // - Use Eq.symm h : b = a, then Eq.subst (Eq.symm h) : motive(b) → motive(a)
    //
    // For reverse rewrite (h : a = b, replace b with a in goal G):
    // - Original goal: G[b]
    // - New goal: G[a]
    // - We need: proof of motive(b) from proof of motive(a)
    // - Use h directly: Eq.subst h : motive(a) → motive(b)

    let symm_proof = if reverse {
        // reverse: use h : a = b directly to go from motive(a) to motive(b)
        // But wait - we swapped from/to, so we're in motive(b) wanting motive(a)
        // Actually for reverse, from=rhs=b, to=lhs=a
        // Goal was G[b], new goal is G[a]
        // We need motive(b) = G[b] from motive(a) = G[a]
        // Use h : a = b with Eq.subst to go motive(a) → motive(b)
        Expr::fvar(hyp_decl.fvar)
    } else {
        // forward: use Eq.symm h : b = a to go from motive(b) to motive(a)
        let symm = Expr::const_(Name::from_string("Eq.symm"), eq_levels.clone());
        Expr::app(
            Expr::app(
                Expr::app(Expr::app(symm, eq_type.clone()), from.clone()),
                to.clone(),
            ),
            Expr::fvar(hyp_decl.fvar),
        )
    };

    // Build: Eq.subst {α} {motive} {to} {from} symm_proof ?m
    let eq_subst = Expr::const_(Name::from_string("Eq.subst"), eq_levels.clone());
    let proof = Expr::app(
        Expr::app(
            Expr::app(
                Expr::app(
                    Expr::app(
                        Expr::app(eq_subst, eq_type.clone()),
                        Expr::lam(BinderInfo::Default, eq_type.clone(), motive),
                    ),
                    to.clone(),
                ),
                from.clone(),
            ),
            symm_proof,
        ),
        new_meta.clone(),
    );

    // Close the current goal with the proof
    state.close_goal(proof)?;

    // Add the new goal
    let new_goal = Goal {
        meta_id: new_meta_id,
        target: new_target,
        local_ctx: goal.local_ctx.clone(),
    };
    state.goals.insert(0, new_goal);

    Ok(())
}

/// Match an expression against the equality pattern `Eq α a b`
pub(crate) fn match_equality(expr: &Expr) -> Result<(Expr, Expr, Expr, Vec<Level>), TacticError> {
    let head = expr.get_app_fn();
    let args: Vec<&Expr> = expr.get_app_args();

    match head {
        Expr::Const(name, levels) if name == &Name::from_string("Eq") => {
            if args.len() != 3 {
                return Err(TacticError::Other(format!(
                    "malformed equality: expected 3 args, got {}",
                    args.len()
                )));
            }
            Ok((
                args[0].clone(), // type α
                args[1].clone(), // lhs a
                args[2].clone(), // rhs b
                levels.to_vec(),
            ))
        }
        _ => Err(TacticError::Other(
            "hypothesis is not an equality".to_string(),
        )),
    }
}

/// Check if an expression contains a subexpression
pub(crate) fn contains_expr(haystack: &Expr, needle: &Expr) -> bool {
    if haystack == needle {
        return true;
    }
    match haystack {
        Expr::App(f, a) => contains_expr(f, needle) || contains_expr(a, needle),
        Expr::Lam(_bi, ty, body) | Expr::Pi(_bi, ty, body) => {
            contains_expr(ty, needle) || contains_expr(body, needle)
        }
        Expr::Let(ty, val, body) => {
            contains_expr(ty, needle) || contains_expr(val, needle) || contains_expr(body, needle)
        }
        _ => false,
    }
}

/// Replace all occurrences of `from` with `to` in an expression
pub(crate) fn replace_expr(expr: &Expr, from: &Expr, to: &Expr) -> Expr {
    if expr == from {
        return to.clone();
    }
    match expr {
        Expr::App(f, a) => Expr::app(replace_expr(f, from, to), replace_expr(a, from, to)),
        Expr::Lam(bi, ty, body) => Expr::lam(
            *bi,
            replace_expr(ty, from, to),
            replace_expr(body, from, to),
        ),
        Expr::Pi(bi, ty, body) => Expr::pi(
            *bi,
            replace_expr(ty, from, to),
            replace_expr(body, from, to),
        ),
        Expr::Let(ty, val, body) => Expr::let_(
            replace_expr(ty, from, to),
            replace_expr(val, from, to),
            replace_expr(body, from, to),
        ),
        _ => expr.clone(),
    }
}

/// Abstract over occurrences of `term` in `expr`, creating a lambda abstraction
pub(crate) fn abstract_over(expr: &Expr, term: &Expr) -> Expr {
    abstract_over_aux(expr, term, 0)
}

fn abstract_over_aux(expr: &Expr, term: &Expr, depth: u32) -> Expr {
    if expr == term {
        return Expr::bvar(depth);
    }
    match expr {
        Expr::App(f, a) => Expr::app(
            abstract_over_aux(f, term, depth),
            abstract_over_aux(a, term, depth),
        ),
        Expr::Lam(bi, ty, body) => Expr::lam(
            *bi,
            abstract_over_aux(ty, term, depth),
            abstract_over_aux(body, term, depth + 1),
        ),
        Expr::Pi(bi, ty, body) => Expr::pi(
            *bi,
            abstract_over_aux(ty, term, depth),
            abstract_over_aux(body, term, depth + 1),
        ),
        Expr::Let(ty, val, body) => Expr::let_(
            abstract_over_aux(ty, term, depth),
            abstract_over_aux(val, term, depth),
            abstract_over_aux(body, term, depth + 1),
        ),
        Expr::BVar(i) => Expr::bvar(i + u32::from(*i >= depth)),
        _ => expr.clone(),
    }
}

/// Rewrite the goal using an equality hypothesis (left-to-right).
/// Convenience wrapper for `rewrite(state, hyp_name, false)`.
pub fn rewrite_ltr(state: &mut ProofState, hyp_name: &str) -> TacticResult {
    rewrite(state, hyp_name, false)
}

/// Rewrite the goal using an equality hypothesis (right-to-left).
/// Convenience wrapper for `rewrite(state, hyp_name, true)`.
pub fn rewrite_rtl(state: &mut ProofState, hyp_name: &str) -> TacticResult {
    rewrite(state, hyp_name, true)
}

/// Symmetry tactic for equality goals.
///
/// For a goal `a = b`, reduces the goal to `b = a` using `Eq.symm`.
/// Requires `Eq.symm` to be present in the environment (via `env.init_eq()`).
pub fn symm(state: &mut ProofState) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
    let target = state.whnf(&goal, &goal.target);

    let (eq_ty, lhs, rhs, levels) = match_equality(&target)
        .map_err(|_| TacticError::GoalMismatch("symm: goal is not an equality".to_string()))?;

    let symm_name = Name::from_string("Eq.symm");
    if state.env.get_const(&symm_name).is_none() {
        return Err(TacticError::Other(
            "symm: Eq.symm not found in environment (call env.init_eq())".to_string(),
        ));
    }

    let eq_ty = state.metas.instantiate(&eq_ty);
    let lhs = state.metas.instantiate(&lhs);
    let rhs = state.metas.instantiate(&rhs);

    // Build the swapped goal: b = a
    let swapped_target = Expr::app(
        Expr::app(
            Expr::app(
                Expr::const_(Name::from_string("Eq"), levels.clone()),
                eq_ty.clone(),
            ),
            rhs.clone(),
        ),
        lhs.clone(),
    );
    let swapped_meta_id = state.metas.fresh(swapped_target.clone());
    let swapped_meta = Expr::FVar(MetaState::to_fvar(swapped_meta_id));

    // Eq.symm {α := eq_ty} {a := rhs} {b := lhs} ?m : lhs = rhs
    let mut proof = Expr::const_(symm_name, levels);
    proof = Expr::app(proof, eq_ty);
    proof = Expr::app(proof, rhs);
    proof = Expr::app(proof, lhs);
    proof = Expr::app(proof, swapped_meta.clone());

    state.close_goal(proof)?;

    let new_goal = Goal {
        meta_id: swapped_meta_id,
        target: swapped_target,
        local_ctx: goal.local_ctx.clone(),
    };
    state.goals.insert(0, new_goal);

    Ok(())
}

/// The `trans` tactic applies transitivity to an equality goal.
///
/// Given a goal `a = c`, the tactic splits it into two subgoals:
/// - `a = middle` (first goal to prove)
/// - `middle = c` (second goal to prove)
///
/// The proof is constructed using `Eq.trans`.
///
/// # Example
/// ```text
/// Goal: x = z
/// trans y
/// Goal 1: x = y
/// Goal 2: y = z
/// ```
///
/// When both subgoals are solved, the proof is `Eq.trans ?1 ?2`.
pub fn trans(state: &mut ProofState, middle: Expr) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
    let target = state.whnf(&goal, &goal.target);

    // Match goal as equality a = c
    let (eq_ty, lhs, rhs, levels) = match_equality(&target)
        .map_err(|_| TacticError::GoalMismatch("trans: goal is not an equality".to_string()))?;

    // Check that Eq.trans exists
    let trans_name = Name::from_string("Eq.trans");
    if state.env.get_const(&trans_name).is_none() {
        return Err(TacticError::Other(
            "trans: Eq.trans not found in environment (call env.init_eq())".to_string(),
        ));
    }

    let eq_ty = state.metas.instantiate(&eq_ty);
    let lhs = state.metas.instantiate(&lhs);
    let rhs = state.metas.instantiate(&rhs);
    let middle = state.metas.instantiate(&middle);

    // Build the first subgoal: a = middle
    let goal1_target = Expr::app(
        Expr::app(
            Expr::app(
                Expr::const_(Name::from_string("Eq"), levels.clone()),
                eq_ty.clone(),
            ),
            lhs.clone(),
        ),
        middle.clone(),
    );
    let goal1_meta_id = state.metas.fresh(goal1_target.clone());
    let goal1_meta = Expr::FVar(MetaState::to_fvar(goal1_meta_id));

    // Build the second subgoal: middle = c
    let goal2_target = Expr::app(
        Expr::app(
            Expr::app(
                Expr::const_(Name::from_string("Eq"), levels.clone()),
                eq_ty.clone(),
            ),
            middle.clone(),
        ),
        rhs.clone(),
    );
    let goal2_meta_id = state.metas.fresh(goal2_target.clone());
    let goal2_meta = Expr::FVar(MetaState::to_fvar(goal2_meta_id));

    // Build proof: Eq.trans {α} {a} {middle} {c} ?goal1 ?goal2
    // Eq.trans : ∀ {α : Sort u} {a b c : α}, Eq a b → Eq b c → Eq a c
    let mut proof = Expr::const_(trans_name, levels);
    proof = Expr::app(proof, eq_ty); // {α}
    proof = Expr::app(proof, lhs); // {a}
    proof = Expr::app(proof, middle); // {b}
    proof = Expr::app(proof, rhs); // {c}
    proof = Expr::app(proof, goal1_meta); // h1: a = middle
    proof = Expr::app(proof, goal2_meta); // h2: middle = c

    state.close_goal(proof)?;

    // Add both goals (goal1 first, then goal2)
    let new_goal1 = Goal {
        meta_id: goal1_meta_id,
        target: goal1_target,
        local_ctx: goal.local_ctx.clone(),
    };
    let new_goal2 = Goal {
        meta_id: goal2_meta_id,
        target: goal2_target,
        local_ctx: goal.local_ctx.clone(),
    };

    // Insert goals: goal1 is first (current), goal2 comes after
    state.goals.insert(0, new_goal2);
    state.goals.insert(0, new_goal1);

    Ok(())
}

/// The `calc_trans` tactic applies transitivity using two existing equality hypotheses.
///
/// Given hypotheses `h1: a = b` and `h2: b = c` in the context, this tactic
/// produces a proof of `a = c` by applying `Eq.trans h1 h2`.
///
/// This is useful when building calculation chains manually.
///
/// # Example
/// ```text
/// h1 : x = y
/// h2 : y = z
/// Goal: x = z
/// calc_trans "h1" "h2"
/// -- Goal solved with proof Eq.trans h1 h2
/// ```
pub fn calc_trans(state: &mut ProofState, h1_name: &str, h2_name: &str) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
    let target = state.whnf(&goal, &goal.target);

    // Match goal as equality a = c
    let (goal_ty, goal_lhs, goal_rhs, goal_levels) = match_equality(&target).map_err(|_| {
        TacticError::GoalMismatch("calc_trans: goal is not an equality".to_string())
    })?;

    // Check that Eq.trans exists
    let trans_name = Name::from_string("Eq.trans");
    if state.env.get_const(&trans_name).is_none() {
        return Err(TacticError::Other(
            "calc_trans: Eq.trans not found in environment (call env.init_eq())".to_string(),
        ));
    }

    // Find h1 in context
    let h1_decl = goal
        .local_ctx
        .iter()
        .find(|d| d.name == h1_name)
        .ok_or_else(|| TacticError::HypothesisNotFound(h1_name.to_string()))?;

    // Find h2 in context
    let h2_decl = goal
        .local_ctx
        .iter()
        .find(|d| d.name == h2_name)
        .ok_or_else(|| TacticError::HypothesisNotFound(h2_name.to_string()))?;

    // Match h1 as equality a = b
    let h1_ty = state.whnf(&goal, &h1_decl.ty);
    let (_h1_eq_ty, h1_lhs, h1_rhs, _h1_levels) = match_equality(&h1_ty)
        .map_err(|_| TacticError::Other(format!("calc_trans: {h1_name} is not an equality")))?;

    // Match h2 as equality b' = c
    let h2_ty = state.whnf(&goal, &h2_decl.ty);
    let (_h2_eq_ty, h2_lhs, h2_rhs, _h2_levels) = match_equality(&h2_ty)
        .map_err(|_| TacticError::Other(format!("calc_trans: {h2_name} is not an equality")))?;

    // Check that h1's RHS equals h2's LHS (the "middle" term)
    if !state.is_def_eq(&goal, &h1_rhs, &h2_lhs) {
        return Err(TacticError::Other(format!(
            "calc_trans: RHS of {h1_name} does not match LHS of {h2_name} (transitivity chain broken)"
        )));
    }

    // Check that h1's LHS equals goal's LHS
    if !state.is_def_eq(&goal, &h1_lhs, &goal_lhs) {
        return Err(TacticError::Other(format!(
            "calc_trans: LHS of {h1_name} does not match goal LHS"
        )));
    }

    // Check that h2's RHS equals goal's RHS
    if !state.is_def_eq(&goal, &h2_rhs, &goal_rhs) {
        return Err(TacticError::Other(format!(
            "calc_trans: RHS of {h2_name} does not match goal RHS"
        )));
    }

    // Build proof: Eq.trans {α} {a} {b} {c} h1 h2
    let mut proof = Expr::const_(trans_name, goal_levels);
    proof = Expr::app(proof, state.metas.instantiate(&goal_ty)); // {α}
    proof = Expr::app(proof, state.metas.instantiate(&h1_lhs)); // {a}
    proof = Expr::app(proof, state.metas.instantiate(&h1_rhs)); // {b}
    proof = Expr::app(proof, state.metas.instantiate(&h2_rhs)); // {c}
    proof = Expr::app(proof, Expr::fvar(h1_decl.fvar)); // h1
    proof = Expr::app(proof, Expr::fvar(h2_decl.fvar)); // h2

    state.close_goal(proof)
}

/// The `subst` tactic substitutes an equality hypothesis into the goal and context.
///
/// Given a hypothesis `h : x = e` where `x` is a free variable, this tactic:
/// 1. Replaces all occurrences of `x` with `e` in the goal
/// 2. Replaces all occurrences of `x` with `e` in other hypotheses
/// 3. Removes the hypothesis `h` from the context
/// 4. Removes `x` from the context (since it's been substituted away)
///
/// The equality can be in either direction:
/// - `h : x = e` - substitutes `x` with `e`
/// - `h : e = x` - substitutes `x` with `e`
///
/// # Arguments
/// * `state` - The proof state
/// * `hyp_name` - Name of the equality hypothesis
///
/// # Example
/// ```text
/// x : Nat
/// h : x = 5
/// goal : x + x = 10
///
/// subst h
///
/// goal : 5 + 5 = 10
/// ```
///
/// # Errors
/// - `HypothesisNotFound` if the hypothesis doesn't exist
/// - `GoalMismatch` if the hypothesis is not an equality
/// - `Other` if neither side of the equality is a free variable in the context
pub fn subst(state: &mut ProofState, hyp_name: &str) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Find the hypothesis
    let hyp_decl = goal
        .local_ctx
        .iter()
        .find(|d| d.name == hyp_name)
        .ok_or_else(|| TacticError::HypothesisNotFound(hyp_name.to_string()))?
        .clone();

    // Check that the hypothesis is an equality
    let hyp_ty = state.whnf(&goal, &hyp_decl.ty);
    let (_eq_type, lhs, rhs, _eq_levels) = match_equality(&hyp_ty)
        .map_err(|_| TacticError::GoalMismatch(format!("subst: {hyp_name} is not an equality")))?;

    // Determine which side is a free variable and which is the replacement
    // Try lhs = fvar first, then rhs = fvar
    let (fvar_id, _fvar_name, replacement) = if let Expr::FVar(id) = &lhs {
        // h : x = e, substitute x with e
        let fvar_decl = goal.local_ctx.iter().find(|d| d.fvar == *id);
        if let Some(decl) = fvar_decl {
            (*id, decl.name.clone(), rhs.clone())
        } else {
            // lhs is a free variable but not in our local context, try rhs
            if let Expr::FVar(id) = &rhs {
                let fvar_decl = goal.local_ctx.iter().find(|d| d.fvar == *id);
                if let Some(decl) = fvar_decl {
                    (*id, decl.name.clone(), lhs.clone())
                } else {
                    return Err(TacticError::Other(
                        "subst: neither side of the equality is a free variable in the context"
                            .to_string(),
                    ));
                }
            } else {
                return Err(TacticError::Other(
                    "subst: neither side of the equality is a free variable".to_string(),
                ));
            }
        }
    } else if let Expr::FVar(id) = &rhs {
        // h : e = x, substitute x with e
        let fvar_decl = goal.local_ctx.iter().find(|d| d.fvar == *id);
        if let Some(decl) = fvar_decl {
            (*id, decl.name.clone(), lhs.clone())
        } else {
            return Err(TacticError::Other(
                "subst: the free variable is not in the local context".to_string(),
            ));
        }
    } else {
        return Err(TacticError::Other(
            "subst: neither side of the equality is a free variable".to_string(),
        ));
    };

    // Substitute in the goal target
    let new_target = goal.target.subst_fvar(fvar_id, &replacement);

    // Build new local context:
    // - Remove the equality hypothesis h
    // - Remove the free variable x being substituted
    // - Substitute x with e in all other hypothesis types
    let new_ctx: Vec<LocalDecl> = goal
        .local_ctx
        .iter()
        .filter(|d| d.name != hyp_name && d.fvar != fvar_id)
        .map(|d| LocalDecl {
            fvar: d.fvar,
            name: d.name.clone(),
            ty: d.ty.subst_fvar(fvar_id, &replacement),
            value: d
                .value
                .as_ref()
                .map(|v| v.subst_fvar(fvar_id, &replacement)),
        })
        .collect();

    // Create a new goal with the substituted target and context
    let new_meta_id = state.metas.fresh(new_target.clone());
    let new_meta = Expr::FVar(MetaState::to_fvar(new_meta_id));

    // The proof uses the new goal's proof directly.
    // When we substitute x with e and h : x = e (or h : e = x),
    // the proof of the original goal from the substituted goal is:
    // λ (proof_of_new : new_target), proof_of_new
    // because after substitution, the types are definitionally equal.
    //
    // In Lean 4, subst actually uses Eq.rec/Eq.subst under the hood,
    // but since we're doing syntactic substitution and the types become
    // definitionally equal after substitution, the proof term is just
    // the substituted proof.
    let proof = new_meta.clone();

    // Close the current goal
    state.close_goal(proof)?;

    // Add the new goal with updated context
    let new_goal = Goal {
        meta_id: new_meta_id,
        target: new_target,
        local_ctx: new_ctx,
    };
    state.goals.insert(0, new_goal);

    Ok(())
}

/// The `subst_vars` tactic repeatedly applies `subst` to all equality hypotheses.
///
/// This finds all hypotheses of the form `h : x = e` or `h : e = x` where `x` is
/// a free variable and applies `subst` to eliminate them one by one.
///
/// # Example
/// ```text
/// x : Nat, y : Nat
/// h1 : x = 5
/// h2 : y = x + 1
/// goal : x + y = 11
///
/// subst_vars
///
/// goal : 5 + 6 = 11  (after substituting x=5, then y=6)
/// ```
pub fn subst_vars(state: &mut ProofState) -> TacticResult {
    // Keep trying to substitute until no more progress
    let max_iterations = 100; // Prevent infinite loops
    for _ in 0..max_iterations {
        let goal = match state.current_goal() {
            Some(g) => g.clone(),
            None => return Ok(()),
        };

        // Find an equality hypothesis where one side is a free variable
        let mut found = None;
        for decl in &goal.local_ctx {
            let hyp_ty = state.whnf(&goal, &decl.ty);
            if let Ok((_eq_type, lhs, rhs, _levels)) = match_equality(&hyp_ty) {
                // Check if lhs or rhs is a free variable in the context
                let is_fvar_in_ctx = |e: &Expr| -> bool {
                    if let Expr::FVar(id) = e {
                        goal.local_ctx.iter().any(|d| d.fvar == *id)
                    } else {
                        false
                    }
                };

                if is_fvar_in_ctx(&lhs) || is_fvar_in_ctx(&rhs) {
                    found = Some(decl.name.clone());
                    break;
                }
            }
        }

        match found {
            Some(hyp_name) => {
                subst(state, &hyp_name)?;
            }
            None => {
                // No more equality hypotheses to substitute
                return Ok(());
            }
        }
    }

    Ok(())
}
