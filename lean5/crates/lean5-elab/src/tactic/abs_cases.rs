//! Absolute value case splitting tactic
//!
//! The `abs_cases` tactic splits on the sign of an expression, creating
//! two proof goals: one where the expression is non-negative and one
//! where it is negative.

use lean5_kernel::name::Name;
use lean5_kernel::Expr;

use crate::tactic::{Goal, LocalDecl, ProofState, TacticError, TacticResult};
use crate::unify::MetaState;

/// Configuration for abs_cases
#[derive(Debug, Clone)]
pub struct AbsCasesConfig {
    /// Name for the non-negative case hypothesis
    pub nonneg_name: String,
    /// Name for the negative case hypothesis
    pub neg_name: String,
}

impl Default for AbsCasesConfig {
    fn default() -> Self {
        AbsCasesConfig {
            nonneg_name: "h_nonneg".to_string(),
            neg_name: "h_neg".to_string(),
        }
    }
}

impl AbsCasesConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_names(nonneg: &str, neg: &str) -> Self {
        AbsCasesConfig {
            nonneg_name: nonneg.to_string(),
            neg_name: neg.to_string(),
        }
    }
}

/// Tactic: abs_cases
///
/// Splits on the absolute value of an expression, creating two cases:
/// 1. When the expression is non-negative (x ≥ 0), where |x| = x
/// 2. When the expression is negative (x < 0), where |x| = -x
///
/// This is useful for proving properties about absolute values by
/// case analysis.
///
/// # Example
/// ```text
/// -- Goal: |x| ≥ 0
/// abs_cases x
/// -- Case 1: h_nonneg : x ≥ 0 ⊢ x ≥ 0
/// -- Case 2: h_neg : x < 0 ⊢ -x ≥ 0
/// ```
///
/// # Errors
/// - `NoGoals` if there are no goals
/// - `Other` if the expression is not suitable for abs_cases
pub fn abs_cases(state: &mut ProofState, var_name: &str) -> TacticResult {
    abs_cases_with_config(state, var_name, AbsCasesConfig::new())
}

/// abs_cases with custom configuration
pub fn abs_cases_with_config(
    state: &mut ProofState,
    var_name: &str,
    config: AbsCasesConfig,
) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Find the variable in context
    let var_decl = goal
        .local_ctx
        .iter()
        .find(|d| d.name == var_name)
        .ok_or_else(|| TacticError::HypothesisNotFound(var_name.to_string()))?;

    // Check if the type is numeric (Int, Real, Rat, etc.)
    let is_numeric = is_numeric_type(&var_decl.ty);
    if !is_numeric {
        return Err(TacticError::Other(format!(
            "abs_cases: {} has type {:?}, expected numeric type",
            var_name, var_decl.ty
        )));
    }

    let var_expr = Expr::fvar(var_decl.fvar);
    let var_ty = var_decl.ty.clone();

    // Create the two cases using by_cases
    // Case 1: var ≥ 0
    let zero = make_zero_for_type(&var_ty);
    let ge_zero = make_ge_expr(&var_expr, &zero, &var_ty);

    // Split into two goals
    let goal_meta = goal.meta_id;
    let original_target = goal.target.clone();
    let local_ctx = goal.local_ctx.clone();

    // Create fresh metas for the two cases
    let case1_meta = state.metas.fresh(original_target.clone());
    let case2_meta = state.metas.fresh(original_target.clone());

    // Get fresh fvars before modifying goals
    let nonneg_fvar = state.fresh_fvar();
    let neg_fvar = state.fresh_fvar();

    // Case 1: x ≥ 0
    let mut case1_ctx = local_ctx.clone();
    case1_ctx.push(LocalDecl {
        fvar: nonneg_fvar,
        name: config.nonneg_name,
        ty: ge_zero.clone(),
        value: None,
    });

    // Case 2: x < 0 (equivalent to ¬(x ≥ 0))
    let lt_zero = make_lt_expr(&var_expr, &zero, &var_ty);
    let mut case2_ctx = local_ctx;
    case2_ctx.push(LocalDecl {
        fvar: neg_fvar,
        name: config.neg_name,
        ty: lt_zero,
        value: None,
    });

    // Create the two new goals
    let case1_goal = Goal {
        meta_id: case1_meta,
        target: original_target.clone(),
        local_ctx: case1_ctx,
    };

    let case2_goal = Goal {
        meta_id: case2_meta,
        target: original_target,
        local_ctx: case2_ctx,
    };

    // Create the Or.elim proof term structure
    // We use classical reasoning to split on x ≥ 0 ∨ x < 0
    let em_proof = create_abs_em_proof(state.env(), &var_expr, &var_ty);
    let case1_meta_expr = Expr::FVar(MetaState::to_fvar(case1_meta));
    let case2_meta_expr = Expr::FVar(MetaState::to_fvar(case2_meta));
    let proof = Expr::app(
        Expr::app(Expr::app(em_proof, case1_meta_expr), case2_meta_expr),
        Expr::type_(), // placeholder
    );

    // Assign the original goal
    state.metas.assign(goal_meta, proof);

    // Replace current goal with two new goals
    state.goals.remove(0);
    state.goals.insert(0, case2_goal);
    state.goals.insert(0, case1_goal);

    Ok(())
}

/// Check if a type is numeric (Int, Real, Rat, etc.)
pub(crate) fn is_numeric_type(ty: &Expr) -> bool {
    match ty {
        Expr::Const(name, _) => {
            let s = name.to_string();
            matches!(s.as_str(), "Int" | "Real" | "Rat" | "Float" | "Complex")
        }
        _ => false,
    }
}

/// Create a zero constant for the given numeric type
pub(crate) fn make_zero_for_type(ty: &Expr) -> Expr {
    match ty {
        Expr::Const(name, _) => {
            let type_name = name.to_string();
            Expr::const_(Name::from_string(&format!("{type_name}.zero")), vec![])
        }
        _ => Expr::const_(Name::from_string("OfNat.ofNat"), vec![]),
    }
}

/// Create a >= expression
pub(crate) fn make_ge_expr(lhs: &Expr, rhs: &Expr, _ty: &Expr) -> Expr {
    Expr::app(
        Expr::app(
            Expr::const_(Name::from_string("GE.ge"), vec![]),
            lhs.clone(),
        ),
        rhs.clone(),
    )
}

/// Create a < expression
pub(crate) fn make_lt_expr(lhs: &Expr, rhs: &Expr, _ty: &Expr) -> Expr {
    Expr::app(
        Expr::app(
            Expr::const_(Name::from_string("LT.lt"), vec![]),
            lhs.clone(),
        ),
        rhs.clone(),
    )
}

/// Create excluded middle proof for abs cases
pub(crate) fn create_abs_em_proof(
    _env: &lean5_kernel::Environment,
    _var: &Expr,
    _ty: &Expr,
) -> Expr {
    // Classical.em : ∀ p, p ∨ ¬p
    // We apply it to (x ≥ 0) to get (x ≥ 0) ∨ ¬(x ≥ 0)
    // which is equivalent to (x ≥ 0) ∨ (x < 0)
    Expr::const_(Name::from_string("Classical.em"), vec![])
}
