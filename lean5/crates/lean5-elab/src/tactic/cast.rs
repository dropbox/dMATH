//! Cast-related tactics
//!
//! Tactics for handling type coercions and casts between numeric types.
//!
//! # Tactics
//! - `push_cast` - Push coercions toward leaves of expressions
//! - `exact_mod_cast` - Close goal with exact proof, handling casts automatically
//! - `assumption_mod_cast` - Close goal with assumption, handling casts
//! - `zify` - Convert natural number goals to integer goals
//! - `qify` - Convert integer goals to rational goals
//! - `lift` - Lift values from subtypes to base types

use lean5_kernel::name::Name;
use lean5_kernel::Expr;

use super::{
    assumption, exact, is_cast_function, make_equality, norm_cast, norm_num, rfl, ring,
    try_tactic_preserving_state, LocalDecl, ProofState, TacticError, TacticResult,
};
use crate::tactic::exprs_syntactically_equal;
use crate::tactic::get_app_fn;
use crate::tactic::match_equality;

// =============================================================================
// push_cast: push coercions toward leaves
// =============================================================================

/// Push cast tactic.
///
/// Pushes coercions (casts) toward the leaves of expressions, which is the
/// opposite direction of `norm_cast`. Useful when you want casts at the
/// atomic level.
///
/// # Example
/// ```text
/// -- Goal: ↑(a + b) = ↑a + ↑b
/// push_cast
/// -- Goal: ↑a + ↑b = ↑a + ↑b (then closes with rfl)
/// ```
pub fn push_cast(state: &mut ProofState) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Check if goal is an equality
    if let Ok((ty, lhs, rhs, levels)) = match_equality(&goal.target) {
        // Push casts in both sides
        let lhs_pushed = push_casts_to_leaves(&lhs);
        let rhs_pushed = push_casts_to_leaves(&rhs);

        // If pushed forms are syntactically equal, close with rfl
        if exprs_syntactically_equal(&lhs_pushed, &rhs_pushed) {
            return rfl(state);
        }

        // Update goal with pushed expressions
        let new_goal = make_equality(&ty, &lhs_pushed, &rhs_pushed, &levels);
        if let Some(goal_mut) = state.goals.first_mut() {
            goal_mut.target = new_goal;
        }

        // Try other tactics
        if ring(state).is_ok() {
            return Ok(());
        }
    }

    if norm_num(state).is_ok() {
        return Ok(());
    }

    Ok(())
}

/// Push casts toward leaves of an expression
pub(crate) fn push_casts_to_leaves(expr: &Expr) -> Expr {
    match expr {
        Expr::App(f, arg) => {
            // Check for cast application
            if is_cast_function(f) {
                // Push cast into the argument
                return push_cast_into(arg, f);
            }

            // Regular application - recurse
            Expr::app(push_casts_to_leaves(f), push_casts_to_leaves(arg))
        }
        Expr::Lam(bind_info, ty, body) => Expr::lam(
            *bind_info,
            push_casts_to_leaves(ty),
            push_casts_to_leaves(body),
        ),
        Expr::Pi(bind_info, ty, body) => Expr::pi(
            *bind_info,
            push_casts_to_leaves(ty),
            push_casts_to_leaves(body),
        ),
        Expr::Let(ty, val, body) => Expr::let_(
            push_casts_to_leaves(ty),
            push_casts_to_leaves(val),
            push_casts_to_leaves(body),
        ),
        _ => expr.clone(),
    }
}

/// Push a cast into an expression
pub(crate) fn push_cast_into(expr: &Expr, cast_fn: &Expr) -> Expr {
    match expr {
        Expr::App(f, arg2) => {
            if let Expr::App(f2, arg1) = f.as_ref() {
                if let Expr::Const(name, _) = get_app_fn(f2) {
                    let name_str = name.to_string();

                    // Push cast over addition
                    if name_str.contains("add") || name_str.contains("Add") {
                        let cast_a = push_cast_into(arg1, cast_fn);
                        let cast_b = push_cast_into(arg2, cast_fn);
                        return Expr::app(Expr::app((**f2).clone(), cast_a), cast_b);
                    }

                    // Push cast over multiplication
                    if name_str.contains("mul") || name_str.contains("Mul") {
                        let cast_a = push_cast_into(arg1, cast_fn);
                        let cast_b = push_cast_into(arg2, cast_fn);
                        return Expr::app(Expr::app((**f2).clone(), cast_a), cast_b);
                    }

                    // Push cast over subtraction
                    if name_str.contains("sub") || name_str.contains("Sub") {
                        let cast_a = push_cast_into(arg1, cast_fn);
                        let cast_b = push_cast_into(arg2, cast_fn);
                        return Expr::app(Expr::app((**f2).clone(), cast_a), cast_b);
                    }
                }
            }
            // Can't push further - apply cast here
            Expr::app(cast_fn.clone(), expr.clone())
        }
        // Atomic - apply cast
        _ => Expr::app(cast_fn.clone(), expr.clone()),
    }
}

// ============================================================================
// Cast-related tactics: exact_mod_cast, assumption_mod_cast, zify, qify
// ============================================================================

/// Configuration for cast normalization.
#[derive(Debug, Clone)]
pub struct CastConfig {
    /// Whether to push casts inward
    pub push_inward: bool,
    /// Whether to pull casts outward
    pub pull_outward: bool,
}

impl Default for CastConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl CastConfig {
    /// Create default configuration (push inward)
    pub fn new() -> Self {
        Self {
            push_inward: true,
            pull_outward: false,
        }
    }
}

/// Close the goal with an exact proof term, automatically handling casts.
///
/// `exact_mod_cast` is like `exact`, but it first normalizes casts in both
/// the goal and the proof term before checking if they match.
///
/// # Example
/// ```text
/// -- h : (n : ℤ) = (m : ℤ)
/// -- Goal: (n : ℚ) = (m : ℚ)
/// exact_mod_cast h
/// ```
///
/// # Errors
/// - `NoGoals` if there are no goals
/// - `Other` if the proof doesn't match the goal even after cast normalization
pub fn exact_mod_cast(state: &mut ProofState, proof: Expr) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    // Try push_cast then exact, preserving state on failure
    if try_tactic_preserving_state(state, |s| {
        let _ = push_cast(s);
        exact(s, proof.clone())
    }) {
        return Ok(());
    }

    // Try norm_cast then exact
    if try_tactic_preserving_state(state, |s| {
        let _ = norm_cast(s);
        exact(s, proof.clone())
    }) {
        return Ok(());
    }

    // Try the original exact
    exact(state, proof)
}

/// Close the goal with an assumption, automatically handling casts.
///
/// `assumption_mod_cast` is like `assumption`, but it normalizes casts
/// before checking if any hypothesis matches the goal.
///
/// # Example
/// ```text
/// -- h : (n : ℤ) < (m : ℤ)
/// -- Goal: (n : ℚ) < (m : ℚ)
/// assumption_mod_cast
/// ```
///
/// # Errors
/// - `NoGoals` if there are no goals
/// - `Other` if no hypothesis matches after cast normalization
pub fn assumption_mod_cast(state: &mut ProofState) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    // Try push_cast then assumption, preserving state on failure
    if try_tactic_preserving_state(state, |s| {
        let _ = push_cast(s);
        assumption(s)
    }) {
        return Ok(());
    }

    // Try norm_cast then assumption
    if try_tactic_preserving_state(state, |s| {
        let _ = norm_cast(s);
        assumption(s)
    }) {
        return Ok(());
    }

    // Try regular assumption
    assumption(state)
}

/// Convert natural number goals to integer goals.
///
/// `zify` rewrites propositions involving natural numbers to equivalent
/// propositions involving integers, introducing appropriate casts.
///
/// # Transformations
/// - `(a : ℕ) ≤ b` → `(↑a : ℤ) ≤ ↑b`
/// - `(a : ℕ) < b` → `(↑a : ℤ) < ↑b`
/// - `(a : ℕ) = b` → `(↑a : ℤ) = ↑b`
/// - `a - b` (truncated) → `↑a - ↑b` (proper subtraction)
///
/// This is useful when natural number arithmetic would lose information
/// due to truncation.
///
/// # Example
/// ```text
/// -- Goal: a - b ≤ c (in ℕ)
/// zify
/// -- Goal: (↑a : ℤ) - ↑b ≤ ↑c
/// ```
///
/// # Errors
/// - `NoGoals` if there are no goals
pub fn zify(state: &mut ProofState) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
    let target = state.metas.instantiate(&goal.target);

    // Transform natural number expressions to integer expressions
    let new_target = zify_expr(&target);

    if new_target == target {
        return Err(TacticError::Other(
            "zify: no natural number expressions to convert".to_string(),
        ));
    }

    let goal = state.current_goal_mut().ok_or(TacticError::NoGoals)?;
    goal.target = new_target;
    Ok(())
}

/// Transform natural number expressions to integer expressions
pub(crate) fn zify_expr(expr: &Expr) -> Expr {
    // Check for Nat type annotations and convert to Int
    match expr {
        Expr::App(func, arg) => {
            let func_z = zify_expr(func);
            let arg_z = zify_expr(arg);

            // Check if this is a Nat operation that should become Int
            if let Expr::Const(name, _) = &func_z {
                let name_str = name.to_string();
                // Convert Nat.sub to Int.sub (which doesn't truncate)
                if name_str == "Nat.sub" || name_str == "HSub.hSub" {
                    // Check if operating on Nat
                    if is_nat_expr(&arg_z) {
                        // Insert Int cast
                        let int_name = Name::from_string("Int");
                        let cast = Expr::const_(Name::from_string("Int.ofNat"), vec![]);
                        return Expr::app(
                            Expr::app(
                                Expr::const_(Name::from_string("HSub.hSub"), vec![]),
                                Expr::app(cast.clone(), arg_z),
                            ),
                            Expr::const_(int_name, vec![]),
                        );
                    }
                }
            }

            Expr::app(func_z, arg_z)
        }

        Expr::Lam(bi, binder_type, body) => Expr::lam(*bi, zify_expr(binder_type), zify_expr(body)),

        Expr::Pi(bi, binder_type, body) => Expr::pi(*bi, zify_expr(binder_type), zify_expr(body)),

        _ => expr.clone(),
    }
}

/// Check if an expression has Nat type (heuristic)
pub(crate) fn is_nat_expr(expr: &Expr) -> bool {
    match expr {
        Expr::Const(name, _) => {
            let s = name.to_string();
            s == "Nat" || s.starts_with("Nat.")
        }
        Expr::App(func, _) => is_nat_expr(func),
        _ => false,
    }
}

/// Convert integer goals to rational goals.
///
/// `qify` rewrites propositions involving integers to equivalent
/// propositions involving rationals, introducing appropriate casts.
///
/// # Transformations
/// - `(a : ℤ) ≤ b` → `(↑a : ℚ) ≤ ↑b`
/// - `a / b` (truncated) → `↑a / ↑b` (exact division)
///
/// # Example
/// ```text
/// -- Goal: a / b = c (in ℤ)
/// qify
/// -- Goal: (↑a : ℚ) / ↑b = ↑c
/// ```
///
/// # Errors
/// - `NoGoals` if there are no goals
pub fn qify(state: &mut ProofState) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
    let target = state.metas.instantiate(&goal.target);

    // Transform integer expressions to rational expressions
    let new_target = qify_expr(&target);

    if new_target == target {
        return Err(TacticError::Other(
            "qify: no integer expressions to convert".to_string(),
        ));
    }

    let goal = state.current_goal_mut().ok_or(TacticError::NoGoals)?;
    goal.target = new_target;
    Ok(())
}

/// Transform integer expressions to rational expressions
pub(crate) fn qify_expr(expr: &Expr) -> Expr {
    match expr {
        Expr::App(func, arg) => {
            let func_q = qify_expr(func);
            let arg_q = qify_expr(arg);

            // Check for Int division that should become Rat
            if let Expr::Const(name, _) = &func_q {
                let name_str = name.to_string();
                if (name_str == "Int.div" || name_str == "HDiv.hDiv") && is_int_expr(&arg_q) {
                    let cast = Expr::const_(Name::from_string("Rat.ofInt"), vec![]);
                    return Expr::app(
                        Expr::app(
                            Expr::const_(Name::from_string("HDiv.hDiv"), vec![]),
                            Expr::app(cast.clone(), arg_q),
                        ),
                        Expr::const_(Name::from_string("Rat"), vec![]),
                    );
                }
            }

            Expr::app(func_q, arg_q)
        }

        Expr::Lam(bi, binder_type, body) => Expr::lam(*bi, qify_expr(binder_type), qify_expr(body)),

        Expr::Pi(bi, binder_type, body) => Expr::pi(*bi, qify_expr(binder_type), qify_expr(body)),

        _ => expr.clone(),
    }
}

/// Check if an expression has Int type (heuristic)
pub(crate) fn is_int_expr(expr: &Expr) -> bool {
    match expr {
        Expr::Const(name, _) => {
            let s = name.to_string();
            s == "Int" || s.starts_with("Int.")
        }
        Expr::App(func, _) => is_int_expr(func),
        _ => false,
    }
}

// ============================================================================
// lift - Lift values from subtypes
// ============================================================================

/// Configuration for lift tactic.
#[derive(Debug, Clone)]
pub struct LiftConfig {
    /// Name for the lifted variable
    pub new_name: Option<String>,
    /// Name for the proof that the value satisfies the predicate
    pub proof_name: Option<String>,
}

impl Default for LiftConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl LiftConfig {
    /// Create default configuration
    pub fn new() -> Self {
        Self {
            new_name: None,
            proof_name: None,
        }
    }

    /// Set the name for the lifted variable
    #[must_use]
    pub fn with_name(mut self, name: String) -> Self {
        self.new_name = Some(name);
        self
    }

    /// Set the name for the proof
    #[must_use]
    pub fn with_proof(mut self, name: String) -> Self {
        self.proof_name = Some(name);
        self
    }
}

/// Lift a value from a subtype to its base type.
///
/// Given a hypothesis `h : ↑n = m` or a subtype element `x : {n : α // P n}`,
/// `lift` introduces the underlying value and a proof of the predicate.
///
/// # Example
/// ```text
/// -- x : ℕ, h : ↑x = (y : ℤ)
/// lift x to ℤ using h
/// -- x : ℤ, h : x = y, hx : 0 ≤ x
/// ```
///
/// # Errors
/// - `NoGoals` if there are no goals
/// - `Other` if the hypothesis is not suitable for lifting
pub fn lift(state: &mut ProofState, var_name: &str, using_hyp: Option<&str>) -> TacticResult {
    lift_with_config(state, var_name, using_hyp, LiftConfig::new())
}

/// lift with custom configuration
pub fn lift_with_config(
    state: &mut ProofState,
    var_name: &str,
    using_hyp: Option<&str>,
    config: LiftConfig,
) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Find the variable to lift
    let var_decl = goal
        .local_ctx
        .iter()
        .find(|d| d.name == var_name)
        .ok_or_else(|| TacticError::Other(format!("variable '{var_name}' not found")))?
        .clone();

    // Check if there's a hypothesis we can use
    if let Some(hyp_name) = using_hyp {
        let _hyp = goal
            .local_ctx
            .iter()
            .find(|d| d.name == hyp_name)
            .ok_or_else(|| TacticError::Other(format!("hypothesis '{hyp_name}' not found")))?;
    }

    // For now, just add a new hypothesis about the lifted value
    let new_name = config
        .new_name
        .unwrap_or_else(|| format!("{var_name}_lifted"));
    let proof_name = config.proof_name.unwrap_or_else(|| format!("h{var_name}"));

    // Get fresh fvar before mutable borrow
    let new_fvar = state.fresh_fvar();

    // Add the lifted variable and proof to the context
    let goal = state.current_goal_mut().ok_or(TacticError::NoGoals)?;

    // Add a hypothesis that the lift succeeded (placeholder)
    goal.local_ctx.push(LocalDecl {
        fvar: new_fvar,
        name: proof_name,
        ty: var_decl.ty.clone(), // Placeholder - should be the predicate proof
        value: None,
    });

    // Rename the original variable
    if let Some(decl) = goal.local_ctx.iter_mut().find(|d| d.name == var_name) {
        decl.name = new_name;
    }

    Ok(())
}
