//! Simplification tactics
//!
//! This module provides the `simp` family of tactics for automatic simplification
//! of expressions using rewrite lemmas, beta/eta reduction, and other normalizations.

use lean5_kernel::name::Name;
use lean5_kernel::Expr;

use super::{
    assumption, exprs_syntactically_equal, match_equality, rfl, trivial, Goal, LocalDecl,
    ProofState, TacticError, TacticResult,
};

/// Configuration for the `simp` tactic.
#[derive(Debug, Clone, Default)]
pub struct SimpConfig {
    /// Maximum number of simplification steps
    pub max_steps: usize,
    /// Whether to apply beta reduction
    pub beta: bool,
    /// Whether to apply eta reduction
    pub eta: bool,
    /// Whether to unfold definitions
    pub unfold: bool,
    /// Additional lemmas to use for simplification
    pub extra_lemmas: Vec<String>,
    /// Lemmas to exclude from simp set
    pub exclude: Vec<String>,
    /// Only simplify to get the result, don't close the goal
    pub only_simplify: bool,
}

impl SimpConfig {
    /// Create a default configuration
    pub fn new() -> Self {
        SimpConfig {
            max_steps: 1000,
            beta: true,
            eta: true,
            unfold: false,
            extra_lemmas: vec![],
            exclude: vec![],
            only_simplify: false,
        }
    }
}

/// A simp lemma entry
#[derive(Debug, Clone)]
pub struct SimpLemma {
    /// Name of the lemma
    pub name: Name,
    /// The equality to apply (lhs = rhs)
    pub lhs: Expr,
    pub rhs: Expr,
    /// Priority (higher = try first)
    pub priority: u32,
}

/// Simplification tactic that rewrites the goal using a set of lemmas.
///
/// The `simp` tactic repeatedly applies simplification lemmas to the goal
/// until no more progress can be made. It handles:
/// - Beta reduction: (λ x => e) a → e[x := a]
/// - Eta reduction: λ x => f x → f (when x not free in f)
/// - Simp lemmas: equations marked @[simp] in the environment
/// - Custom lemmas: additional lemmas passed in config
///
/// # Arguments
/// * `state` - The proof state
/// * `config` - Configuration options for simplification
///
/// # Example
/// ```text
/// -- Goal: a + 0 = a
/// simp  -- Uses Nat.add_zero : n + 0 = n
/// -- Goal closed by reflexivity
/// ```
///
/// # Errors
/// - `NoGoals` if there are no goals
/// - `Other` if simplification makes no progress and goal not closed
pub fn simp(state: &mut ProofState, config: SimpConfig) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
    let mut current_target = goal.target.clone();
    let mut steps = 0;
    let mut made_progress = false;

    // Collect simp lemmas from environment
    let simp_lemmas = collect_simp_lemmas(state, &config);

    // Main simplification loop
    while steps < config.max_steps {
        let (new_target, changed) = simp_expr(state, &goal, &current_target, &simp_lemmas, &config);

        if changed {
            made_progress = true;
            current_target = new_target;
            steps += 1;
        } else {
            break;
        }
    }

    if made_progress {
        // Update the goal with the simplified target
        state.goals[0].target = current_target.clone();

        // Try to close with reflexivity if we simplified to a trivial equality
        if !config.only_simplify {
            if rfl(state).is_ok() {
                return Ok(());
            }
            // Try assumption in case it's now provable from context
            if assumption(state).is_ok() {
                return Ok(());
            }
        }

        Ok(())
    } else {
        // No progress made - try closing with trivial tactics anyway
        if !config.only_simplify {
            if rfl(state).is_ok() {
                return Ok(());
            }
            if assumption(state).is_ok() {
                return Ok(());
            }
        }

        Err(TacticError::Other("simp: no progress made".to_string()))
    }
}

/// Collect simp lemmas from the environment
pub(crate) fn collect_simp_lemmas(state: &ProofState, config: &SimpConfig) -> Vec<SimpLemma> {
    let mut lemmas = Vec::new();

    // Built-in simplification rules for common patterns
    // These correspond to standard Lean simp lemmas

    // Nat.add_zero: n + 0 = n
    let nat_add_zero = Name::from_string("Nat.add_zero");
    if state.env.get_const(&nat_add_zero).is_some()
        && !config.exclude.contains(&"Nat.add_zero".to_string())
    {
        lemmas.push(SimpLemma {
            name: nat_add_zero,
            lhs: make_nat_add_pattern("n", "Nat.zero"),
            rhs: Expr::bvar(0), // n
            priority: 100,
        });
    }

    // Nat.zero_add: 0 + n = n
    let nat_zero_add = Name::from_string("Nat.zero_add");
    if state.env.get_const(&nat_zero_add).is_some()
        && !config.exclude.contains(&"Nat.zero_add".to_string())
    {
        lemmas.push(SimpLemma {
            name: nat_zero_add,
            lhs: make_nat_add_pattern("Nat.zero", "n"),
            rhs: Expr::bvar(0), // n
            priority: 100,
        });
    }

    // Nat.mul_one: n * 1 = n
    let nat_mul_one = Name::from_string("Nat.mul_one");
    if state.env.get_const(&nat_mul_one).is_some()
        && !config.exclude.contains(&"Nat.mul_one".to_string())
    {
        lemmas.push(SimpLemma {
            name: nat_mul_one,
            lhs: make_nat_mul_pattern("n", "Nat.one"),
            rhs: Expr::bvar(0), // n
            priority: 100,
        });
    }

    // Nat.one_mul: 1 * n = n
    let nat_one_mul = Name::from_string("Nat.one_mul");
    if state.env.get_const(&nat_one_mul).is_some()
        && !config.exclude.contains(&"Nat.one_mul".to_string())
    {
        lemmas.push(SimpLemma {
            name: nat_one_mul,
            lhs: make_nat_mul_pattern("Nat.one", "n"),
            rhs: Expr::bvar(0), // n
            priority: 100,
        });
    }

    // Nat.mul_zero: n * 0 = 0
    let nat_mul_zero = Name::from_string("Nat.mul_zero");
    if state.env.get_const(&nat_mul_zero).is_some()
        && !config.exclude.contains(&"Nat.mul_zero".to_string())
    {
        lemmas.push(SimpLemma {
            name: nat_mul_zero,
            lhs: make_nat_mul_pattern("n", "Nat.zero"),
            rhs: Expr::const_(Name::from_string("Nat.zero"), vec![]),
            priority: 100,
        });
    }

    // Nat.zero_mul: 0 * n = 0
    let nat_zero_mul = Name::from_string("Nat.zero_mul");
    if state.env.get_const(&nat_zero_mul).is_some()
        && !config.exclude.contains(&"Nat.zero_mul".to_string())
    {
        lemmas.push(SimpLemma {
            name: nat_zero_mul,
            lhs: make_nat_mul_pattern("Nat.zero", "n"),
            rhs: Expr::const_(Name::from_string("Nat.zero"), vec![]),
            priority: 100,
        });
    }

    // Bool.not_not: !!b = b
    let bool_not_not = Name::from_string("Bool.not_not");
    if state.env.get_const(&bool_not_not).is_some()
        && !config.exclude.contains(&"Bool.not_not".to_string())
    {
        lemmas.push(SimpLemma {
            name: bool_not_not,
            lhs: Expr::app(
                Expr::const_(Name::from_string("Bool.not"), vec![]),
                Expr::app(
                    Expr::const_(Name::from_string("Bool.not"), vec![]),
                    Expr::bvar(0),
                ),
            ),
            rhs: Expr::bvar(0),
            priority: 100,
        });
    }

    // And.comm: P ∧ Q ↔ Q ∧ P (simplified for goal transformation)
    // Or.comm: P ∨ Q ↔ Q ∨ P
    // These are iff lemmas, need special handling

    // Add user-specified extra lemmas
    for lemma_name in &config.extra_lemmas {
        let name = Name::from_string(lemma_name);
        if let Some(decl) = state.env.get_const(&name) {
            // Try to extract lhs = rhs from the lemma type
            if let Some((lhs, rhs)) = extract_equality_from_type(&decl.type_) {
                lemmas.push(SimpLemma {
                    name,
                    lhs,
                    rhs,
                    priority: 50, // User lemmas have lower priority
                });
            }
        }
    }

    // Sort by priority (higher first)
    lemmas.sort_by(|a, b| b.priority.cmp(&a.priority));

    lemmas
}

/// Helper to create a Nat.add pattern
fn make_nat_add_pattern(left: &str, right: &str) -> Expr {
    let left_expr = if left == "n" {
        Expr::bvar(0)
    } else {
        Expr::const_(Name::from_string(left), vec![])
    };
    let right_expr = if right == "n" {
        Expr::bvar(0)
    } else {
        Expr::const_(Name::from_string(right), vec![])
    };

    Expr::app(
        Expr::app(
            Expr::const_(Name::from_string("Nat.add"), vec![]),
            left_expr,
        ),
        right_expr,
    )
}

/// Helper to create a Nat.mul pattern
fn make_nat_mul_pattern(left: &str, right: &str) -> Expr {
    let left_expr = if left == "n" {
        Expr::bvar(0)
    } else {
        Expr::const_(Name::from_string(left), vec![])
    };
    let right_expr = if right == "n" {
        Expr::bvar(0)
    } else {
        Expr::const_(Name::from_string(right), vec![])
    };

    Expr::app(
        Expr::app(
            Expr::const_(Name::from_string("Nat.mul"), vec![]),
            left_expr,
        ),
        right_expr,
    )
}

/// Extract lhs = rhs from a lemma type (handles forall quantifiers)
fn extract_equality_from_type(ty: &Expr) -> Option<(Expr, Expr)> {
    match ty {
        Expr::App(f, arg) => {
            // Check if this is Eq _ lhs rhs
            if let Expr::App(f2, lhs) = f.as_ref() {
                if let Expr::App(eq, _ty) = f2.as_ref() {
                    if let Expr::Const(name, _) = eq.as_ref() {
                        if name == &Name::from_string("Eq") {
                            return Some((lhs.as_ref().clone(), arg.as_ref().clone()));
                        }
                    }
                }
            }
            None
        }
        Expr::Pi(_bi, _ty, body) => {
            // Recurse into forall/pi body
            extract_equality_from_type(body)
        }
        _ => None,
    }
}

/// Simplify an expression using simp lemmas
pub(crate) fn simp_expr(
    state: &ProofState,
    goal: &Goal,
    expr: &Expr,
    lemmas: &[SimpLemma],
    config: &SimpConfig,
) -> (Expr, bool) {
    // First try beta/eta reduction
    let mut current = expr.clone();
    let mut changed = false;

    if config.beta {
        let beta_reduced = beta_reduce(&current);
        if beta_reduced != current {
            current = beta_reduced;
            changed = true;
        }
    }

    if config.eta {
        let eta_reduced = eta_reduce(&current);
        if eta_reduced != current {
            current = eta_reduced;
            changed = true;
        }
    }

    // Try to apply simp lemmas
    for lemma in lemmas {
        if let Some(result) = try_apply_simp_lemma(state, goal, &current, lemma) {
            return (result, true);
        }
    }

    // Recurse into subexpressions
    match &current {
        Expr::App(f, arg) => {
            let (new_f, f_changed) = simp_expr(state, goal, f, lemmas, config);
            let (new_arg, arg_changed) = simp_expr(state, goal, arg, lemmas, config);
            if f_changed || arg_changed {
                return (Expr::app(new_f, new_arg), true);
            }
        }
        Expr::Lam(bi, ty, body) => {
            let (new_body, body_changed) = simp_expr(state, goal, body, lemmas, config);
            if body_changed {
                return (Expr::lam(*bi, ty.as_ref().clone(), new_body), true);
            }
        }
        Expr::Pi(bi, ty, body) => {
            let (new_body, body_changed) = simp_expr(state, goal, body, lemmas, config);
            if body_changed {
                return (Expr::pi(*bi, ty.as_ref().clone(), new_body), true);
            }
        }
        Expr::Let(ty, val, body) => {
            let (new_val, val_changed) = simp_expr(state, goal, val, lemmas, config);
            let (new_body, body_changed) = simp_expr(state, goal, body, lemmas, config);
            if val_changed || body_changed {
                return (Expr::let_(ty.as_ref().clone(), new_val, new_body), true);
            }
        }
        _ => {}
    }

    (current, changed)
}

/// Try to apply a simp lemma to an expression
fn try_apply_simp_lemma(
    _state: &ProofState,
    _goal: &Goal,
    expr: &Expr,
    lemma: &SimpLemma,
) -> Option<Expr> {
    // Simple pattern matching (exact match only for now)
    // A full implementation would do unification with metavariables

    if exprs_equal_modulo_bvars(expr, &lemma.lhs) {
        return Some(lemma.rhs.clone());
    }

    None
}

/// Check if two expressions are equal, treating bvars as pattern variables
fn exprs_equal_modulo_bvars(expr: &Expr, pattern: &Expr) -> bool {
    match (expr, pattern) {
        (_, Expr::BVar(_)) => true, // bvar in pattern matches anything
        (Expr::Const(n1, _), Expr::Const(n2, _)) => n1 == n2,
        (Expr::App(f1, a1), Expr::App(f2, a2)) => {
            exprs_equal_modulo_bvars(f1, f2) && exprs_equal_modulo_bvars(a1, a2)
        }
        (Expr::Lam(_, t1, b1), Expr::Lam(_, t2, b2)) => {
            exprs_equal_modulo_bvars(t1, t2) && exprs_equal_modulo_bvars(b1, b2)
        }
        _ => expr == pattern,
    }
}

/// Perform beta reduction on an expression
pub(crate) fn beta_reduce(expr: &Expr) -> Expr {
    match expr {
        Expr::App(f, arg) => {
            let f_reduced = beta_reduce(f);
            let arg_reduced = beta_reduce(arg);

            // Check for beta redex: (λ x => body) arg
            if let Expr::Lam(_bi, _ty, body) = &f_reduced {
                // Substitute arg for bvar(0) in body
                return substitute_bvar(body, 0, &arg_reduced);
            }

            Expr::app(f_reduced, arg_reduced)
        }
        Expr::Lam(bi, ty, body) => Expr::lam(*bi, beta_reduce(ty), beta_reduce(body)),
        Expr::Pi(bi, ty, body) => Expr::pi(*bi, beta_reduce(ty), beta_reduce(body)),
        Expr::Let(_ty, val, body) => {
            // Let reduction: substitute value into body
            let val_reduced = beta_reduce(val);
            let body_reduced = beta_reduce(body);
            substitute_bvar(&body_reduced, 0, &val_reduced)
        }
        _ => expr.clone(),
    }
}

/// Substitute an expression for a bound variable
pub(crate) fn substitute_bvar(expr: &Expr, idx: u32, replacement: &Expr) -> Expr {
    substitute_bvar_aux(expr, idx, replacement, 0)
}

fn substitute_bvar_aux(expr: &Expr, target_idx: u32, replacement: &Expr, depth: u32) -> Expr {
    match expr {
        Expr::BVar(i) => {
            if *i == target_idx + depth {
                // Shift the replacement up by depth
                shift_expr(replacement, depth as i64)
            } else if *i > target_idx + depth {
                // Variable from outer scope - shift down
                Expr::bvar(*i - 1)
            } else {
                expr.clone()
            }
        }
        Expr::App(f, arg) => Expr::app(
            substitute_bvar_aux(f, target_idx, replacement, depth),
            substitute_bvar_aux(arg, target_idx, replacement, depth),
        ),
        Expr::Lam(bi, ty, body) => Expr::lam(
            *bi,
            substitute_bvar_aux(ty, target_idx, replacement, depth),
            substitute_bvar_aux(body, target_idx, replacement, depth + 1),
        ),
        Expr::Pi(bi, ty, body) => Expr::pi(
            *bi,
            substitute_bvar_aux(ty, target_idx, replacement, depth),
            substitute_bvar_aux(body, target_idx, replacement, depth + 1),
        ),
        Expr::Let(ty, val, body) => Expr::let_(
            substitute_bvar_aux(ty, target_idx, replacement, depth),
            substitute_bvar_aux(val, target_idx, replacement, depth),
            substitute_bvar_aux(body, target_idx, replacement, depth + 1),
        ),
        _ => expr.clone(),
    }
}

/// Shift free variables in an expression
pub(crate) fn shift_expr(expr: &Expr, amount: i64) -> Expr {
    shift_expr_aux(expr, amount, 0)
}

fn shift_expr_aux(expr: &Expr, amount: i64, cutoff: u32) -> Expr {
    match expr {
        Expr::BVar(i) => {
            if *i >= cutoff {
                let new_idx = (*i as i64 + amount) as u32;
                Expr::bvar(new_idx)
            } else {
                expr.clone()
            }
        }
        Expr::App(f, arg) => Expr::app(
            shift_expr_aux(f, amount, cutoff),
            shift_expr_aux(arg, amount, cutoff),
        ),
        Expr::Lam(bi, ty, body) => Expr::lam(
            *bi,
            shift_expr_aux(ty, amount, cutoff),
            shift_expr_aux(body, amount, cutoff + 1),
        ),
        Expr::Pi(bi, ty, body) => Expr::pi(
            *bi,
            shift_expr_aux(ty, amount, cutoff),
            shift_expr_aux(body, amount, cutoff + 1),
        ),
        Expr::Let(ty, val, body) => Expr::let_(
            shift_expr_aux(ty, amount, cutoff),
            shift_expr_aux(val, amount, cutoff),
            shift_expr_aux(body, amount, cutoff + 1),
        ),
        _ => expr.clone(),
    }
}

/// Perform eta reduction: λ x => f x → f (when x not free in f)
pub(crate) fn eta_reduce(expr: &Expr) -> Expr {
    match expr {
        Expr::Lam(bi, ty, body) => {
            if let Expr::App(f, arg) = body.as_ref() {
                // Check if arg is bvar(0) and f doesn't contain bvar(0)
                if let Expr::BVar(0) = arg.as_ref() {
                    if !contains_bvar(f, 0) {
                        // Eta reduce: λ x => f x → f (with shifted indices)
                        return shift_expr(f, -1);
                    }
                }
            }
            Expr::lam(*bi, eta_reduce(ty), eta_reduce(body))
        }
        Expr::App(f, arg) => Expr::app(eta_reduce(f), eta_reduce(arg)),
        Expr::Pi(bi, ty, body) => Expr::pi(*bi, eta_reduce(ty), eta_reduce(body)),
        Expr::Let(ty, val, body) => Expr::let_(eta_reduce(ty), eta_reduce(val), eta_reduce(body)),
        _ => expr.clone(),
    }
}

/// Check if an expression contains a specific bound variable
pub(crate) fn contains_bvar(expr: &Expr, idx: u32) -> bool {
    contains_bvar_aux(expr, idx, 0)
}

fn contains_bvar_aux(expr: &Expr, target_idx: u32, depth: u32) -> bool {
    match expr {
        Expr::BVar(i) => *i == target_idx + depth,
        Expr::App(f, arg) => {
            contains_bvar_aux(f, target_idx, depth) || contains_bvar_aux(arg, target_idx, depth)
        }
        Expr::Lam(_, ty, body) | Expr::Pi(_, ty, body) => {
            contains_bvar_aux(ty, target_idx, depth)
                || contains_bvar_aux(body, target_idx, depth + 1)
        }
        Expr::Let(ty, val, body) => {
            contains_bvar_aux(ty, target_idx, depth)
                || contains_bvar_aux(val, target_idx, depth)
                || contains_bvar_aux(body, target_idx, depth + 1)
        }
        _ => false,
    }
}

/// Simplified simp tactic with default config
pub fn simp_default(state: &mut ProofState) -> TacticResult {
    simp(state, SimpConfig::new())
}

/// Simp tactic with specific lemmas only
pub fn simp_only(state: &mut ProofState, lemmas: Vec<String>) -> TacticResult {
    let mut config = SimpConfig::new();
    config.extra_lemmas = lemmas;
    simp(state, config)
}

// ============================================================================
// simp_all - Simplify all hypotheses and goal
// ============================================================================

/// Simplify all hypotheses and the goal.
///
/// `simp_all` applies simplification to both the hypotheses in the local context
/// and the goal. Hypotheses can be used as rewrite lemmas for each other and
/// for the goal. Trivial hypotheses (like `True` or `a = a`) are removed.
///
/// # Example
/// ```text
/// -- h1 : n + 0 = n
/// -- h2 : m * 1 = m
/// -- Goal: n + 0 = m * 1
/// simp_all
/// -- h1 : n = n (simplified, removed as trivial)
/// -- h2 : m = m (simplified, removed as trivial)
/// -- Goal closed by rfl
/// ```
///
/// # Errors
/// - `NoGoals` if there are no goals
/// - `Other` if simplification makes no progress
pub fn simp_all(state: &mut ProofState) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
    let mut made_progress = false;

    // Build config with hypotheses as extra lemmas
    let mut config = SimpConfig::new();

    // Add hypothesis names as extra lemmas for simp to use
    for decl in &goal.local_ctx {
        // If the hypothesis has an equality type, add it as a lemma
        if match_equality(&decl.ty).is_ok() {
            config.extra_lemmas.push(decl.name.clone());
        }
    }

    // Collect simp lemmas (including from hypotheses)
    let simp_lemmas = collect_simp_lemmas(state, &config);

    // Simplify each hypothesis
    let mut new_local_ctx = Vec::new();
    let mut modified_hyps = false;

    for decl in &goal.local_ctx {
        let (simplified_ty, changed) = simp_expr(state, &goal, &decl.ty, &simp_lemmas, &config);

        if changed {
            modified_hyps = true;
            made_progress = true;
        }

        // Check if the hypothesis simplified to True (trivial)
        if is_true_const(&simplified_ty) {
            // Skip trivial hypotheses
            continue;
        }

        // Check if hypothesis is a trivial equality (a = a)
        if is_trivial_equality(&simplified_ty) {
            // Skip reflexive equalities
            continue;
        }

        new_local_ctx.push(LocalDecl {
            fvar: decl.fvar,
            name: decl.name.clone(),
            ty: simplified_ty,
            value: decl.value.clone(),
        });
    }

    // Update the local context if we modified hypotheses
    if modified_hyps {
        if let Some(goal_mut) = state.goals.first_mut() {
            goal_mut.local_ctx = new_local_ctx;
        }
    }

    // Now simplify the goal
    let current_goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
    let (simplified_target, target_changed) = simp_expr(
        state,
        &current_goal,
        &current_goal.target,
        &simp_lemmas,
        &config,
    );

    if target_changed {
        made_progress = true;
        if let Some(goal_mut) = state.goals.first_mut() {
            goal_mut.target = simplified_target;
        }
    }

    // Try to close the goal
    if rfl(state).is_ok() {
        return Ok(());
    }

    if assumption(state).is_ok() {
        return Ok(());
    }

    if trivial(state).is_ok() {
        return Ok(());
    }

    if made_progress {
        Ok(())
    } else {
        Err(TacticError::Other("simp_all: no progress made".to_string()))
    }
}

/// Check if an expression is the True constant
pub(crate) fn is_true_const(expr: &Expr) -> bool {
    if let Expr::Const(name, _) = expr {
        let name_str = name.to_string();
        name_str == "True" || name_str == "true"
    } else {
        false
    }
}

/// Check if an expression is a trivial equality (a = a)
pub(crate) fn is_trivial_equality(expr: &Expr) -> bool {
    if let Ok((_ty, lhs, rhs, _levels)) = match_equality(expr) {
        exprs_syntactically_equal(&lhs, &rhs)
    } else {
        false
    }
}

// ============================================================================
// simp_rw - Simplification with rewriting
// ============================================================================

use super::{contains_expr, replace_expr};

/// `simp_rw` applies simplification and rewriting interleaved.
/// Unlike `simp`, it applies rewrites more aggressively at all positions.
///
/// # Example
/// ```text
/// -- Goal: f (a + 0) = f a
/// simp_rw [h]  -- where h : a + 0 = a
/// ```
pub fn simp_rw(state: &mut ProofState, lemmas: Vec<String>) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
    let mut current_target = goal.target.clone();
    let mut made_progress = false;
    let max_iterations = 100;
    let mut iterations = 0;

    while iterations < max_iterations {
        let mut changed = false;

        // Try each lemma as a rewrite
        for lemma_name in &lemmas {
            // Look for the lemma in local context
            if let Some(hyp_decl) = goal.local_ctx.iter().find(|d| &d.name == lemma_name) {
                let hyp_ty = state.whnf(&goal, &hyp_decl.ty);
                if let Ok((_eq_type, lhs, rhs, _)) = match_equality(&hyp_ty) {
                    // Try forward rewrite
                    if contains_expr(&current_target, &lhs) {
                        current_target = replace_expr(&current_target, &lhs, &rhs);
                        changed = true;
                        made_progress = true;
                    }
                    // Try reverse rewrite
                    if contains_expr(&current_target, &rhs) {
                        current_target = replace_expr(&current_target, &rhs, &lhs);
                        changed = true;
                        made_progress = true;
                    }
                }
            }
        }

        // Also try simp-style simplifications
        let simp_config = SimpConfig::default();
        let simp_lemmas = collect_simp_lemmas(state, &simp_config);
        let (new_target, simp_changed) =
            simp_expr(state, &goal, &current_target, &simp_lemmas, &simp_config);
        if simp_changed {
            current_target = new_target;
            changed = true;
            made_progress = true;
        }

        if !changed {
            break;
        }
        iterations += 1;
    }

    if made_progress {
        state.goals[0].target = current_target.clone();

        // Try to close with rfl
        if rfl(state).is_ok() {
            return Ok(());
        }

        Ok(())
    } else {
        Err(TacticError::Other("simp_rw: no progress made".to_string()))
    }
}

/// Simplified version of simp_rw that uses hypotheses by name.
pub fn simp_rw_hyps(state: &mut ProofState, hyp_names: Vec<&str>) -> TacticResult {
    simp_rw(
        state,
        hyp_names.into_iter().map(ToString::to_string).collect(),
    )
}

// ============================================================================
// Squeeze Simp Tactic
// ============================================================================

/// Configuration for squeeze_simp
#[derive(Debug, Clone)]
pub struct SqueezeSimpConfig {
    /// Base simp configuration
    pub simp_config: SimpConfig,
    /// Whether to print verbose output about all attempted lemmas
    pub verbose: bool,
}

impl Default for SqueezeSimpConfig {
    fn default() -> Self {
        SqueezeSimpConfig {
            simp_config: SimpConfig::new(),
            verbose: false,
        }
    }
}

impl SqueezeSimpConfig {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Result of squeeze_simp showing which lemmas were used
#[derive(Debug, Clone)]
pub struct SqueezeSimpResult {
    /// The lemmas that were actually used during simplification
    pub used_lemmas: Vec<String>,
    /// Suggested replacement: "simp only [lemma1, lemma2, ...]"
    pub suggested_tactic: String,
    /// Whether the goal was closed
    pub closed: bool,
}

/// Tactic: squeeze_simp
///
/// Like `simp`, but tracks which lemmas were actually used during simplification.
/// Returns a `SqueezeSimpResult` with the suggested `simp only [...]` call.
///
/// This is useful for:
/// - Speeding up proofs by replacing `simp` with `simp only [...]`
/// - Making proofs more robust by explicitly listing dependencies
/// - Debugging which lemmas are being applied
///
/// # Example
/// ```text
/// -- Goal: a + 0 + 0 = a
/// squeeze_simp
/// -- Output: "Try: simp only [Nat.add_zero]"
/// -- Goal closed
/// ```
///
/// # Errors
/// - `NoGoals` if there are no goals
pub fn squeeze_simp(state: &mut ProofState) -> Result<SqueezeSimpResult, TacticError> {
    squeeze_simp_with_config(state, SqueezeSimpConfig::new())
}

/// squeeze_simp with custom configuration
pub fn squeeze_simp_with_config(
    state: &mut ProofState,
    config: SqueezeSimpConfig,
) -> Result<SqueezeSimpResult, TacticError> {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
    let mut current_target = goal.target.clone();
    let mut steps = 0;
    let mut used_lemmas: Vec<String> = Vec::new();

    // Collect simp lemmas from environment
    let simp_lemmas = collect_simp_lemmas(state, &config.simp_config);

    // Main simplification loop - track which lemmas are actually applied
    while steps < config.simp_config.max_steps {
        let (new_target, changed, applied_lemma) = simp_expr_tracking(
            state,
            &goal,
            &current_target,
            &simp_lemmas,
            &config.simp_config,
        );

        if changed {
            if let Some(lemma_name) = applied_lemma {
                if !used_lemmas.contains(&lemma_name) {
                    used_lemmas.push(lemma_name);
                }
            }
            current_target = new_target;
            steps += 1;
        } else {
            break;
        }
    }

    let made_progress = !used_lemmas.is_empty();

    // Update the goal with the simplified target
    if made_progress {
        state.goals[0].target = current_target.clone();
    }

    // Try to close with reflexivity
    let closed = if config.simp_config.only_simplify {
        false
    } else {
        rfl(state).is_ok() || assumption(state).is_ok()
    };

    // Generate suggested tactic
    let suggested_tactic = if used_lemmas.is_empty() {
        "simp only []".to_string()
    } else {
        format!("simp only [{}]", used_lemmas.join(", "))
    };

    Ok(SqueezeSimpResult {
        used_lemmas,
        suggested_tactic,
        closed,
    })
}

/// Helper: simplify expression while tracking which lemma was applied
fn simp_expr_tracking(
    state: &ProofState,
    goal: &Goal,
    expr: &Expr,
    lemmas: &[SimpLemma],
    config: &SimpConfig,
) -> (Expr, bool, Option<String>) {
    // First try beta/eta reductions
    if config.beta {
        let reduced = beta_reduce(expr);
        if reduced != *expr {
            return (reduced, true, Some("beta".to_string()));
        }
    }

    if config.eta {
        let reduced = eta_reduce(expr);
        if reduced != *expr {
            return (reduced, true, Some("eta".to_string()));
        }
    }

    // Try applying simp lemmas - track which one succeeds
    for lemma in lemmas {
        if let Some(result) = try_apply_simp_lemma(state, goal, expr, lemma) {
            return (result, true, Some(lemma.name.to_string()));
        }
    }

    // Recurse into subexpressions
    match expr {
        Expr::App(f, a) => {
            let (new_f, f_changed, f_lemma) = simp_expr_tracking(state, goal, f, lemmas, config);
            if f_changed {
                return (Expr::app(new_f, (**a).clone()), true, f_lemma);
            }
            let (new_a, a_changed, a_lemma) = simp_expr_tracking(state, goal, a, lemmas, config);
            if a_changed {
                return (Expr::app((**f).clone(), new_a), true, a_lemma);
            }
        }
        Expr::Lam(bi, ty, body) => {
            let (new_body, changed, lemma) = simp_expr_tracking(state, goal, body, lemmas, config);
            if changed {
                return (Expr::lam(*bi, ty.as_ref().clone(), new_body), true, lemma);
            }
        }
        Expr::Pi(bi, ty, body) => {
            let (new_ty, ty_changed, ty_lemma) =
                simp_expr_tracking(state, goal, ty, lemmas, config);
            if ty_changed {
                return (Expr::pi(*bi, new_ty, (**body).clone()), true, ty_lemma);
            }
            let (new_body, body_changed, body_lemma) =
                simp_expr_tracking(state, goal, body, lemmas, config);
            if body_changed {
                return (Expr::pi(*bi, (**ty).clone(), new_body), true, body_lemma);
            }
        }
        _ => {}
    }

    (expr.clone(), false, None)
}

/// squeeze_simp and apply the result
pub fn squeeze_simp_and_apply(state: &mut ProofState) -> TacticResult {
    let result = squeeze_simp(state)?;
    if result.closed {
        Ok(())
    } else {
        Err(TacticError::Other(format!(
            "squeeze_simp did not close goal. Suggestion: {}",
            result.suggested_tactic
        )))
    }
}
