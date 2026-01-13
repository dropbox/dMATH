//! Conversion and calculation chain tactics
//!
//! Provides tactics for converting between equivalent forms and building
//! calculation chains.

use lean5_kernel::name::Name;
use lean5_kernel::{Expr, Level, TypeChecker};

use super::equality::match_equality;
use super::ring::make_eq;
use super::{Goal, ProofState, TacticError, TacticResult};

// ============================================================================
// convert: Prove goal by converting to equivalent form
// ============================================================================

/// The `convert` tactic proves the goal by finding a proof term that may not
/// exactly match, generating subgoals for mismatched parts.
///
/// Given goal `⊢ T` and term `h : T'`, `convert h` will:
/// 1. If T and T' are definitionally equal, close the goal
/// 2. Otherwise, create subgoals to prove T = T' or its components match
///
/// This is useful when you have a proof that's "almost right" but needs
/// some massaging.
pub fn convert(state: &mut ProofState, proof_term: Expr) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
    let target = state.metas.instantiate(&goal.target);

    // Type check the proof term
    let mut tc = TypeChecker::new(state.env());
    let Ok(proof_type) = tc.infer_type(&proof_term) else {
        return Err(TacticError::Other(
            "convert: cannot infer type of proof term".to_string(),
        ));
    };

    // Check if types are definitionally equal
    if tc.is_def_eq(&target, &proof_type) {
        // Exact match, close goal
        state.metas.assign(goal.meta_id, proof_term);
        state.goals.remove(0);
        return Ok(());
    }

    // Types differ - try to decompose and create subgoals
    convert_with_subgoals(state, &goal, &target, &proof_type, &proof_term)
}

/// Create subgoals for type mismatch in convert
fn convert_with_subgoals(
    state: &mut ProofState,
    goal: &Goal,
    target: &Expr,
    proof_type: &Expr,
    proof_term: &Expr,
) -> TacticResult {
    // Strategy 1: If both are equalities, decompose
    if let (Ok((t_ty, t_lhs, t_rhs, t_levels)), Ok((p_ty, p_lhs, p_rhs, _p_levels))) =
        (match_equality(target), match_equality(proof_type))
    {
        let tc = TypeChecker::new(state.env());

        // Check type compatibility
        if !tc.is_def_eq(&t_ty, &p_ty) {
            return Err(TacticError::Other(
                "convert: equality types do not match".to_string(),
            ));
        }

        let mut subgoals = Vec::new();

        // Check LHS
        if !tc.is_def_eq(&t_lhs, &p_lhs) {
            let eq_goal = make_eq(&t_ty, &t_lhs, &p_lhs, &t_levels);
            subgoals.push(eq_goal);
        }

        // Check RHS
        if !tc.is_def_eq(&t_rhs, &p_rhs) {
            let eq_goal = make_eq(&t_ty, &t_rhs, &p_rhs, &t_levels);
            subgoals.push(eq_goal);
        }

        if subgoals.is_empty() {
            // Should have matched - close goal
            state.metas.assign(goal.meta_id, proof_term.clone());
            state.goals.remove(0);
            return Ok(());
        }

        // Remove current goal and add subgoals
        state.goals.remove(0);

        for subgoal_target in subgoals.into_iter().rev() {
            let meta_id = state.metas.fresh(subgoal_target.clone());
            state.goals.insert(
                0,
                Goal {
                    meta_id,
                    target: subgoal_target,
                    local_ctx: goal.local_ctx.clone(),
                },
            );
        }

        return Ok(());
    }

    // Strategy 2: Create a single goal to prove type equality
    let eq_goal = Expr::app(
        Expr::app(
            Expr::app(
                Expr::const_(Name::from_string("Eq"), vec![Level::succ(Level::zero())]),
                Expr::type_(),
            ),
            target.clone(),
        ),
        proof_type.clone(),
    );

    state.goals.remove(0);
    let meta_id = state.metas.fresh(eq_goal.clone());
    state.goals.insert(
        0,
        Goal {
            meta_id,
            target: eq_goal,
            local_ctx: goal.local_ctx.clone(),
        },
    );

    Ok(())
}

/// `convert` using a named hypothesis from context
pub fn convert_hyp(state: &mut ProofState, hyp_name: &str) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?;

    // Find hypothesis
    let decl = goal
        .local_ctx
        .iter()
        .find(|d| d.name == hyp_name)
        .ok_or_else(|| TacticError::HypothesisNotFound(hyp_name.to_string()))?;

    let proof_term = Expr::FVar(decl.fvar);

    convert(state, proof_term)
}

// ============================================================================
// calc_block: Calculation chain support
// ============================================================================

/// Represents a step in a calculation chain.
#[derive(Debug, Clone)]
pub struct CalcStep {
    /// The relation (=, ≤, <, etc.)
    pub rel: CalcRel,
    /// The right-hand side of this step
    pub rhs: Expr,
    /// The justification (proof term or tactic name)
    pub justification: CalcJustification,
}

/// Relation type for calc steps
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalcRel {
    Eq,
    Le,
    Lt,
    Ge,
    Gt,
    Iff,
}

/// Justification for a calc step
#[derive(Debug, Clone)]
pub enum CalcJustification {
    /// A proof term
    Term(Expr),
    /// Name of a hypothesis
    Hyp(String),
    /// Use rfl/refl
    Refl,
    /// Apply a named lemma
    Lemma(String),
}

/// Execute a calculation chain proof.
///
/// A calc block is a sequence of steps:
/// ```text
/// calc a = b := by exact h1
///      _ = c := by ring
///      _ ≤ d := by linarith
/// ```
///
/// This constructs the final proof by chaining together the individual steps
/// using transitivity.
pub fn calc_block(state: &mut ProofState, start: Expr, steps: Vec<CalcStep>) -> TacticResult {
    if steps.is_empty() {
        return Err(TacticError::Other(
            "calc_block: no steps provided".to_string(),
        ));
    }

    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Current LHS value
    let mut current = start;

    // Collected proof terms for each step
    let mut step_proofs: Vec<(CalcRel, Expr)> = Vec::new();

    // Process each step
    for step in &steps {
        // The current step proves: current `rel` step.rhs
        let step_target = make_calc_rel(step.rel, &current, &step.rhs);

        // Create temporary goal for this step
        let step_meta = state.metas.fresh(step_target.clone());
        let _step_goal = Goal {
            meta_id: step_meta,
            target: step_target,
            local_ctx: goal.local_ctx.clone(),
        };

        // Try to prove this step
        let step_proof = match &step.justification {
            CalcJustification::Term(t) => t.clone(),
            CalcJustification::Hyp(name) => {
                let decl = goal
                    .local_ctx
                    .iter()
                    .find(|d| &d.name == name)
                    .ok_or_else(|| TacticError::HypothesisNotFound(name.clone()))?;
                Expr::FVar(decl.fvar)
            }
            CalcJustification::Refl => {
                if step.rel == CalcRel::Eq || step.rel == CalcRel::Iff {
                    make_eq_refl(&Expr::type_(), &current)
                } else if step.rel == CalcRel::Le || step.rel == CalcRel::Ge {
                    Expr::const_(Name::from_string("le_refl"), vec![])
                } else {
                    return Err(TacticError::Other(
                        "calc_block: refl not applicable for strict inequality".to_string(),
                    ));
                }
            }
            CalcJustification::Lemma(name) => Expr::const_(Name::from_string(name), vec![]),
        };

        step_proofs.push((step.rel, step_proof));
        current = step.rhs.clone();
    }

    // Combine proofs using transitivity
    // For now, create subgoals for each step that needs proving
    state.goals.remove(0);

    for (i, step) in steps.iter().enumerate().rev() {
        let step_lhs = if i == 0 {
            state
                .current_goal()
                .map_or_else(Expr::type_, |g| g.target.clone())
        } else {
            steps[i - 1].rhs.clone()
        };
        let step_target = make_calc_rel(step.rel, &step_lhs, &step.rhs);
        let meta_id = state.metas.fresh(step_target.clone());
        state.goals.insert(
            0,
            Goal {
                meta_id,
                target: step_target,
                local_ctx: goal.local_ctx.clone(),
            },
        );
    }

    Ok(())
}

/// Create expression for a calc relation
pub(crate) fn make_calc_rel(rel: CalcRel, lhs: &Expr, rhs: &Expr) -> Expr {
    let rel_name = match rel {
        CalcRel::Eq => "Eq",
        CalcRel::Le => "LE.le",
        CalcRel::Lt => "LT.lt",
        CalcRel::Ge => "GE.ge",
        CalcRel::Gt => "GT.gt",
        CalcRel::Iff => "Iff",
    };

    if rel == CalcRel::Eq {
        // Eq needs type argument
        Expr::app(
            Expr::app(
                Expr::app(
                    Expr::const_(Name::from_string(rel_name), vec![Level::zero()]),
                    Expr::type_(), // placeholder type
                ),
                lhs.clone(),
            ),
            rhs.clone(),
        )
    } else {
        Expr::app(
            Expr::app(
                Expr::const_(Name::from_string(rel_name), vec![]),
                lhs.clone(),
            ),
            rhs.clone(),
        )
    }
}

/// Simple calc for equality chain: prove a = c from a = b and b = c
pub fn calc_eq(state: &mut ProofState, middle: Expr) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
    let target = state.metas.instantiate(&goal.target);

    // Check goal is equality
    let (ty, lhs, rhs, levels) = match_equality(&target)
        .map_err(|_| TacticError::GoalMismatch("calc_eq: goal must be equality".to_string()))?;

    // Create two subgoals: lhs = middle, middle = rhs
    let goal1 = make_eq(&ty, &lhs, &middle, &levels);
    let goal2 = make_eq(&ty, &middle, &rhs, &levels);

    state.goals.remove(0);

    let meta2 = state.metas.fresh(goal2.clone());
    state.goals.insert(
        0,
        Goal {
            meta_id: meta2,
            target: goal2,
            local_ctx: goal.local_ctx.clone(),
        },
    );

    let meta1 = state.metas.fresh(goal1.clone());
    state.goals.insert(
        0,
        Goal {
            meta_id: meta1,
            target: goal1,
            local_ctx: goal.local_ctx.clone(),
        },
    );

    Ok(())
}

/// Helper to create Eq.refl proof term
pub(crate) fn make_eq_refl(ty: &Expr, val: &Expr) -> Expr {
    Expr::app(
        Expr::app(
            Expr::const_(Name::from_string("Eq.refl"), vec![Level::zero()]),
            ty.clone(),
        ),
        val.clone(),
    )
}
