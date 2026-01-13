//! Search Tactics (exact?, apply?, suggest, aesop)
//!
//! This module provides proof search tactics that explore the proof space
//! to find applicable lemmas or suggest tactics.

use super::{
    apply, assumption, cases, decide, exact, intro, left_tactic, rfl, right_tactic, simp,
    split_tactic, tauto, trivial, use_single, ProofState, SimpConfig, TacticError, TacticResult,
};
use lean5_kernel::name::Name;
use lean5_kernel::{Expr, Level, TypeChecker};

/// Result of a search tactic, containing the suggestion and its proof
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Name of the constant that was found
    pub name: Name,
    /// The expression to use (instantiated with fresh metavariables for universe params)
    pub expr: Expr,
    /// Human-readable suggestion
    pub suggestion: String,
}

/// `exact?` - search for a proof term that exactly matches the goal type
///
/// Searches through:
/// 1. Local hypotheses
/// 2. Constants in the environment
///
/// Returns a list of possible proofs.
///
/// # Example
/// ```text
/// -- goal: ∀ x y : Nat, x + y = y + x
/// exact?
/// -- suggests: Nat.add_comm
/// ```
pub fn exact_search(
    state: &mut ProofState,
    max_results: usize,
) -> Result<Vec<SearchResult>, TacticError> {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?;
    let target = goal.target.clone();
    let local_ctx = goal.local_ctx.clone();

    let mut results = Vec::new();

    // 1. Search local hypotheses
    for decl in &local_ctx {
        let tc = TypeChecker::new(state.env());
        if tc.is_def_eq(&decl.ty, &target) {
            results.push(SearchResult {
                name: Name::from_string(&decl.name),
                expr: Expr::fvar(decl.fvar),
                suggestion: format!("exact {}", decl.name),
            });
            if results.len() >= max_results {
                return Ok(results);
            }
        }
    }

    // 2. Search constants in environment
    for constant in state.env().constants() {
        // Create instance with fresh levels
        let levels: Vec<Level> = constant
            .level_params
            .iter()
            .enumerate()
            .map(|(i, _)| Level::param(Name::from_string(&format!("_u{i}"))))
            .collect();

        let const_type = if levels.is_empty() {
            constant.type_.clone()
        } else {
            // Instantiate universe parameters
            let subst: Vec<(Name, Level)> = constant
                .level_params
                .iter()
                .cloned()
                .zip(levels.iter().cloned())
                .collect();
            constant.type_.instantiate_level_params(&subst)
        };

        // Check if constant type matches target (possibly with unification)
        let _tc = TypeChecker::new(state.env());
        if types_unify(state.env(), &const_type, &target) {
            results.push(SearchResult {
                name: constant.name.clone(),
                expr: Expr::const_(constant.name.clone(), levels),
                suggestion: format!("exact {}", constant.name),
            });
            if results.len() >= max_results {
                return Ok(results);
            }
        }
    }

    Ok(results)
}

/// Check if two types can be unified (simple version using is_def_eq)
pub(crate) fn types_unify(env: &lean5_kernel::Environment, ty1: &Expr, ty2: &Expr) -> bool {
    let tc = TypeChecker::new(env);
    tc.is_def_eq(ty1, ty2)
}

/// Check if a function type can be applied to produce the target type
/// Returns Some((arg_types, result)) if the function can produce the target
pub(crate) fn can_apply_to_produce(
    env: &lean5_kernel::Environment,
    func_type: &Expr,
    target: &Expr,
    max_args: usize,
) -> Option<Vec<Expr>> {
    let tc = TypeChecker::new(env);
    let mut current = func_type.clone();
    let mut arg_types = Vec::new();

    for _ in 0..max_args {
        // Check if current type matches target
        if tc.is_def_eq(&current, target) {
            return Some(arg_types);
        }

        // If it's a Pi type, we can apply to it
        let whnf = tc.whnf(&current);
        if let Expr::Pi(_, domain, codomain) = &whnf {
            arg_types.push(domain.as_ref().clone());
            // For dependent types, we'd need to substitute - for now, just continue with codomain
            current = codomain.as_ref().clone();
        } else {
            break;
        }
    }

    // Final check
    if tc.is_def_eq(&current, target) {
        Some(arg_types)
    } else {
        None
    }
}

/// `apply?` - search for a lemma that can be applied to make progress on the goal
///
/// Searches through:
/// 1. Local hypotheses (functions that return the target type)
/// 2. Constants in the environment
///
/// Returns a list of possible applications.
///
/// # Example
/// ```text
/// -- goal: P → Q
/// apply?
/// -- suggests: apply h (if h : P → Q exists)
/// ```
pub fn apply_search(
    state: &mut ProofState,
    max_results: usize,
) -> Result<Vec<SearchResult>, TacticError> {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?;
    let target = goal.target.clone();
    let local_ctx = goal.local_ctx.clone();

    let mut results = Vec::new();

    // 1. Search local hypotheses
    for decl in &local_ctx {
        if let Some(args) = can_apply_to_produce(state.env(), &decl.ty, &target, 10) {
            let arg_count = args.len();
            results.push(SearchResult {
                name: Name::from_string(&decl.name),
                expr: Expr::fvar(decl.fvar),
                suggestion: if arg_count == 0 {
                    format!("exact {}", decl.name)
                } else {
                    format!("apply {} ({} args)", decl.name, arg_count)
                },
            });
            if results.len() >= max_results {
                return Ok(results);
            }
        }
    }

    // 2. Search constants in environment
    for constant in state.env().constants() {
        let levels: Vec<Level> = constant
            .level_params
            .iter()
            .enumerate()
            .map(|(i, _)| Level::param(Name::from_string(&format!("_u{i}"))))
            .collect();

        let const_type = if levels.is_empty() {
            constant.type_.clone()
        } else {
            let subst: Vec<(Name, Level)> = constant
                .level_params
                .iter()
                .cloned()
                .zip(levels.iter().cloned())
                .collect();
            constant.type_.instantiate_level_params(&subst)
        };

        if let Some(args) = can_apply_to_produce(state.env(), &const_type, &target, 10) {
            let arg_count = args.len();
            results.push(SearchResult {
                name: constant.name.clone(),
                expr: Expr::const_(constant.name.clone(), levels),
                suggestion: if arg_count == 0 {
                    format!("exact {}", constant.name)
                } else {
                    format!("apply {} ({} args)", constant.name, arg_count)
                },
            });
            if results.len() >= max_results {
                return Ok(results);
            }
        }
    }

    Ok(results)
}

/// A tactic suggestion with confidence score
#[derive(Debug, Clone)]
pub struct TacticSuggestion {
    /// The suggested tactic command
    pub tactic: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Explanation of why this tactic might work
    pub reason: String,
}

/// `suggest` - suggest tactics that might make progress on the goal
///
/// Analyzes the goal structure and suggests appropriate tactics.
///
/// # Example
/// ```text
/// -- goal: P ∧ Q
/// suggest
/// -- suggests: constructor, split, And.intro
/// ```
pub fn suggest(
    state: &mut ProofState,
    max_suggestions: usize,
) -> Result<Vec<TacticSuggestion>, TacticError> {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?;
    let target = goal.target.clone();
    let local_ctx = goal.local_ctx.clone();

    let mut suggestions = Vec::new();

    // Analyze goal structure
    let target_head = target.get_app_fn();
    let _target_args: Vec<&Expr> = target.get_app_args();

    // Get the head name if it's a constant
    let head_name = if let Expr::Const(name, _) = &target_head {
        Some(name.to_string())
    } else {
        None
    };

    // Check for specific goal shapes

    // 1. Equality goals
    if head_name.as_deref() == Some("Eq") {
        suggestions.push(TacticSuggestion {
            tactic: "rfl".to_string(),
            confidence: 0.9,
            reason: "Goal is an equality - try reflexivity".to_string(),
        });
        suggestions.push(TacticSuggestion {
            tactic: "simp".to_string(),
            confidence: 0.7,
            reason: "Simplification often solves equalities".to_string(),
        });
        suggestions.push(TacticSuggestion {
            tactic: "ring".to_string(),
            confidence: 0.6,
            reason: "Ring solver for algebraic equalities".to_string(),
        });
        suggestions.push(TacticSuggestion {
            tactic: "omega".to_string(),
            confidence: 0.5,
            reason: "Omega for integer arithmetic".to_string(),
        });
    }

    // 2. Conjunction goals (And)
    if head_name.as_deref() == Some("And") {
        suggestions.push(TacticSuggestion {
            tactic: "constructor".to_string(),
            confidence: 0.95,
            reason: "Goal is a conjunction - split into two goals".to_string(),
        });
        suggestions.push(TacticSuggestion {
            tactic: "split".to_string(),
            confidence: 0.95,
            reason: "Split conjunction into components".to_string(),
        });
    }

    // 3. Disjunction goals (Or)
    if head_name.as_deref() == Some("Or") {
        suggestions.push(TacticSuggestion {
            tactic: "left".to_string(),
            confidence: 0.5,
            reason: "Prove left disjunct".to_string(),
        });
        suggestions.push(TacticSuggestion {
            tactic: "right".to_string(),
            confidence: 0.5,
            reason: "Prove right disjunct".to_string(),
        });
    }

    // 4. Existential goals (Exists)
    if head_name.as_deref() == Some("Exists") {
        suggestions.push(TacticSuggestion {
            tactic: "use _".to_string(),
            confidence: 0.8,
            reason: "Provide a witness for the existential".to_string(),
        });
    }

    // 5. Universal goals (Pi/forall)
    if let Expr::Pi(_, _, _) = &target {
        suggestions.push(TacticSuggestion {
            tactic: "intro".to_string(),
            confidence: 0.95,
            reason: "Goal is a forall/implication - introduce hypothesis".to_string(),
        });
        suggestions.push(TacticSuggestion {
            tactic: "intros".to_string(),
            confidence: 0.9,
            reason: "Introduce all hypotheses at once".to_string(),
        });
    }

    // 6. False goal
    if head_name.as_deref() == Some("False") {
        suggestions.push(TacticSuggestion {
            tactic: "contradiction".to_string(),
            confidence: 0.8,
            reason: "Goal is False - look for contradiction in hypotheses".to_string(),
        });
        suggestions.push(TacticSuggestion {
            tactic: "tauto".to_string(),
            confidence: 0.6,
            reason: "Propositional tautology solver".to_string(),
        });
    }

    // 7. Negation goals (Not)
    if head_name.as_deref() == Some("Not") {
        suggestions.push(TacticSuggestion {
            tactic: "intro h".to_string(),
            confidence: 0.9,
            reason: "Goal is a negation - assume and derive contradiction".to_string(),
        });
        suggestions.push(TacticSuggestion {
            tactic: "push_neg".to_string(),
            confidence: 0.7,
            reason: "Push negations inward".to_string(),
        });
    }

    // 8. Check for applicable hypotheses
    for decl in &local_ctx {
        // If hypothesis type matches goal exactly
        let tc = TypeChecker::new(state.env());
        if tc.is_def_eq(&decl.ty, &target) {
            suggestions.push(TacticSuggestion {
                tactic: format!("exact {}", decl.name),
                confidence: 1.0,
                reason: format!("Hypothesis {} has exactly the goal type", decl.name),
            });
        }

        // If hypothesis can be applied
        if let Some(args) = can_apply_to_produce(state.env(), &decl.ty, &target, 5) {
            if !args.is_empty() {
                suggestions.push(TacticSuggestion {
                    tactic: format!("apply {}", decl.name),
                    confidence: 0.85,
                    reason: format!(
                        "Hypothesis {} can be applied ({} args needed)",
                        decl.name,
                        args.len()
                    ),
                });
            }
        }
    }

    // 9. Generic tactics that often help
    suggestions.push(TacticSuggestion {
        tactic: "simp".to_string(),
        confidence: 0.4,
        reason: "Simplification is often useful".to_string(),
    });
    suggestions.push(TacticSuggestion {
        tactic: "trivial".to_string(),
        confidence: 0.3,
        reason: "Try simple tactics".to_string(),
    });
    suggestions.push(TacticSuggestion {
        tactic: "aesop".to_string(),
        confidence: 0.5,
        reason: "General automated proof search".to_string(),
    });

    // Sort by confidence and limit results
    suggestions.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    suggestions.truncate(max_suggestions);

    Ok(suggestions)
}

/// Aesop rule kind
#[derive(Debug, Clone)]
pub enum AesopRuleKind {
    /// Safe rule - always apply without backtracking
    Safe,
    /// Norm rule - normalization that doesn't change provability
    Norm,
    /// Unsafe rule - may need backtracking (with priority)
    Unsafe(i32),
}

/// Aesop rule descriptor
#[derive(Debug, Clone)]
pub struct AesopRule {
    /// Name of the rule
    pub name: String,
    /// Kind of rule (safe, norm, unsafe)
    pub kind: AesopRuleKind,
}

/// Configuration for aesop
#[derive(Debug, Clone)]
pub struct AesopConfig {
    /// Maximum search depth
    pub max_depth: usize,
    /// Maximum number of goals to process
    pub max_goals: usize,
    /// Whether to use simp normalization
    pub use_simp: bool,
    /// Whether to try unfold tactics
    pub use_unfold: bool,
}

impl Default for AesopConfig {
    fn default() -> Self {
        AesopConfig {
            max_depth: 10,
            max_goals: 100,
            use_simp: true,
            use_unfold: true,
        }
    }
}

/// `aesop` - general automated proof search tactic
///
/// Implements a best-first search proof strategy inspired by Isabelle's auto and Lean 4's aesop.
///
/// Strategy:
/// 1. Apply safe rules (intro, split on And, etc.)
/// 2. Try normalization (simp, ring, norm_num)
/// 3. Apply unsafe rules with backtracking (apply, cases)
/// 4. Search for applicable lemmas
///
/// # Example
/// ```text
/// -- goal: P ∧ Q → Q ∧ P
/// aesop
/// -- automatically proves by intro, cases, constructor
/// ```
pub fn aesop(state: &mut ProofState) -> TacticResult {
    aesop_with_config(state, AesopConfig::default())
}

/// Aesop with custom configuration
pub fn aesop_with_config(state: &mut ProofState, config: AesopConfig) -> TacticResult {
    aesop_search(state, &config, 0)
}

/// Internal aesop search function (non-backtracking version)
fn aesop_search(state: &mut ProofState, config: &AesopConfig, depth: usize) -> TacticResult {
    if depth > config.max_depth {
        return Err(TacticError::Other(format!(
            "aesop: exceeded max depth {}",
            config.max_depth
        )));
    }

    let mut iterations = 0;
    let max_iterations = config.max_goals;

    while !state.goals().is_empty() && iterations < max_iterations {
        iterations += 1;

        // Phase 1: Apply safe rules
        if aesop_safe_rules(state, config, depth).is_ok() {
            if state.goals().is_empty() {
                return Ok(());
            }
            continue;
        }

        // Phase 2: Normalization
        if config.use_simp {
            let _ = aesop_normalize(state);
        }

        if state.goals().is_empty() {
            return Ok(());
        }

        // Phase 3: Try to close goal directly
        if aesop_try_close(state).is_ok() {
            continue;
        }

        // Phase 4: Try first applicable unsafe rule (no backtracking)
        let candidates = aesop_get_candidates(state, config);

        let mut made_progress = false;
        for candidate in candidates {
            if (candidate.apply)(state).is_ok() {
                made_progress = true;
                break;
            }
        }

        if !made_progress {
            // No progress possible
            break;
        }
    }

    if state.goals().is_empty() {
        Ok(())
    } else {
        Err(TacticError::Other("aesop: no proof found".to_string()))
    }
}

/// A candidate tactic for aesop
struct AesopCandidate {
    priority: i32,
    apply: Box<dyn Fn(&mut ProofState) -> TacticResult>,
}

/// Apply safe rules that don't require backtracking
fn aesop_safe_rules(state: &mut ProofState, _config: &AesopConfig, depth: usize) -> TacticResult {
    let mut made_progress = true;

    while made_progress && !state.goals().is_empty() {
        made_progress = false;

        let goal = state.current_goal().ok_or(TacticError::NoGoals)?;
        let target = goal.target.clone();

        // Check goal structure
        let target_head = target.get_app_fn();

        if let Expr::Const(name, _) = &target_head {
            let name_str = name.to_string();

            // Safe rule: intro for Pi types (implications/forall)
            if let Expr::Pi(_, _, _) = &target {
                if intro(state, "h".to_string()).is_ok() {
                    made_progress = true;
                    continue;
                }
            }

            // Safe rule: constructor for And
            if name_str == "And" && split_tactic(state).is_ok() {
                made_progress = true;
                continue;
            }

            // Safe rule: intro for Not (it's really an implication to False)
            if name_str == "Not" && intro(state, "h".to_string()).is_ok() {
                made_progress = true;
                continue;
            }
        }

        // If target is Pi (forall/implication)
        if let Expr::Pi(_, _, _) = &target {
            if intro(state, format!("h{depth}")).is_ok() {
                made_progress = true;
                continue;
            }
        }
    }

    Ok(())
}

/// Apply normalization tactics
fn aesop_normalize(state: &mut ProofState) -> TacticResult {
    // Try various normalization tactics
    let _ = trivial(state);

    // Try simp on current goal with default config
    if state.current_goal().is_some() {
        let _ = simp(state, SimpConfig::default());
    }

    Ok(())
}

/// Try to close the current goal
fn aesop_try_close(state: &mut ProofState) -> TacticResult {
    // Try assumption
    if assumption(state).is_ok() {
        return Ok(());
    }

    // Try rfl
    if rfl(state).is_ok() {
        return Ok(());
    }

    // Try trivial
    if trivial(state).is_ok() && state.goals().is_empty() {
        return Ok(());
    }

    // Try tauto for propositional goals
    if tauto(state).is_ok() {
        return Ok(());
    }

    // Try decide for decidable propositions
    if decide(state).is_ok() {
        return Ok(());
    }

    Err(TacticError::Other("aesop: cannot close goal".to_string()))
}

/// Get candidate tactics for backtracking search
fn aesop_get_candidates(state: &mut ProofState, _config: &AesopConfig) -> Vec<AesopCandidate> {
    let mut candidates = Vec::new();

    let Some(goal) = state.current_goal() else {
        return candidates;
    };

    let target = goal.target.clone();
    let target_head = target.get_app_fn();
    let local_ctx = goal.local_ctx.clone();

    // Candidate: left/right for Or
    if let Expr::Const(name, _) = &target_head {
        if name.to_string() == "Or" {
            candidates.push(AesopCandidate {
                priority: 50,
                apply: Box::new(left_tactic),
            });
            candidates.push(AesopCandidate {
                priority: 50,
                apply: Box::new(right_tactic),
            });
        }

        // Candidate: existsi for Exists
        if name.to_string() == "Exists" {
            // We'd need to find a witness - for now, just try common ones
            candidates.push(AesopCandidate {
                priority: 30,
                apply: Box::new(|s| use_single(s, Expr::Lit(lean5_kernel::expr::Literal::Nat(0)))),
            });
        }
    }

    // Candidate: apply local hypotheses
    for decl in &local_ctx {
        let decl_fvar = decl.fvar;
        let decl_ty = decl.ty.clone();

        // Check if it's a function that can be applied
        if let Some(args) = can_apply_to_produce(state.env(), &decl_ty, &target, 5) {
            if !args.is_empty() {
                candidates.push(AesopCandidate {
                    priority: 80,
                    apply: Box::new(move |s| apply(s, Expr::fvar(decl_fvar))),
                });
            }
        }
    }

    // Candidate: apply constants from environment (limited to avoid explosion)
    let mut const_count = 0;
    for constant in state.env().constants() {
        if const_count > 20 {
            break; // Limit to prevent explosion
        }

        let levels: Vec<Level> = constant
            .level_params
            .iter()
            .enumerate()
            .map(|(i, _)| Level::param(Name::from_string(&format!("_u{i}"))))
            .collect();

        let const_type = if levels.is_empty() {
            constant.type_.clone()
        } else {
            let subst: Vec<(Name, Level)> = constant
                .level_params
                .iter()
                .cloned()
                .zip(levels.iter().cloned())
                .collect();
            constant.type_.instantiate_level_params(&subst)
        };

        if let Some(_args) = can_apply_to_produce(state.env(), &const_type, &target, 5) {
            let const_name = constant.name.clone();
            let const_levels = levels.clone();

            candidates.push(AesopCandidate {
                priority: 40,
                apply: Box::new(move |s| {
                    apply(s, Expr::const_(const_name.clone(), const_levels.clone()))
                }),
            });
            const_count += 1;
        }
    }

    // Candidate: cases on hypotheses with inductive types
    for decl in &local_ctx {
        let decl_ty_head = decl.ty.get_app_fn();
        if let Expr::Const(name, _) = &decl_ty_head {
            let name_str = name.to_string();
            // Common inductive types
            if name_str == "And" || name_str == "Or" || name_str == "Exists" || name_str == "Sum" {
                let hyp_name = decl.name.clone();
                candidates.push(AesopCandidate {
                    priority: 70,
                    apply: Box::new(move |s| cases(s, &hyp_name)),
                });
            }
        }
    }

    // Sort by priority (higher first)
    candidates.sort_by(|a, b| b.priority.cmp(&a.priority));

    candidates
}

/// `exact?!` - search and apply the first matching proof
pub fn exact_search_and_apply(state: &mut ProofState) -> TacticResult {
    let results = exact_search(state, 1)?;

    if let Some(result) = results.first() {
        exact(state, result.expr.clone())
    } else {
        Err(TacticError::Other(
            "exact?: no matching proof found".to_string(),
        ))
    }
}

/// `apply?!` - search and apply the first matching lemma
pub fn apply_search_and_apply(state: &mut ProofState) -> TacticResult {
    let results = apply_search(state, 1)?;

    if let Some(result) = results.first() {
        apply(state, result.expr.clone())
    } else {
        Err(TacticError::Other(
            "apply?: no applicable lemma found".to_string(),
        ))
    }
}

/// `hint` - provide hints about the goal without modifying state
pub fn hint(state: &ProofState) -> Result<Vec<String>, TacticError> {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?;
    let target = goal.target.clone();

    let mut hints = Vec::new();

    let target_head = target.get_app_fn();

    if let Expr::Const(name, _) = &target_head {
        let name_str = name.to_string();

        match name_str.as_str() {
            "Eq" => {
                hints.push(
                    "This is an equality goal. Try: rfl, simp, ring, omega, or rewrite".to_string(),
                );
            }
            "And" => {
                hints.push(
                    "This is a conjunction. Use `constructor` or `split` to prove each part"
                        .to_string(),
                );
            }
            "Or" => {
                hints.push(
                    "This is a disjunction. Use `left` or `right` to choose which side to prove"
                        .to_string(),
                );
            }
            "Exists" => {
                hints.push(
                    "This is an existential. Use `use <witness>` to provide a witness".to_string(),
                );
            }
            "Not" | "False" => {
                hints.push("This is a negation/falsity goal. Try `intro` to assume the hypothesis, then derive a contradiction".to_string());
            }
            "True" => {
                hints.push("This is trivially true. Use `trivial` or `constructor`".to_string());
            }
            "Iff" => {
                hints
                    .push("This is an iff. Use `constructor` to prove both directions".to_string());
            }
            _ => {
                hints.push(format!(
                    "Goal head is `{name_str}`. Check if there's a relevant lemma or constructor"
                ));
            }
        }
    }

    if let Expr::Pi(_, _, _) = &target {
        hints.push(
            "This is a forall/implication. Use `intro` to introduce the hypothesis".to_string(),
        );
    }

    // Check local context for useful hypotheses
    let local_ctx = goal.local_ctx.clone();
    for decl in &local_ctx {
        let tc = TypeChecker::new(state.env());
        if tc.is_def_eq(&decl.ty, &target) {
            hints.push(format!(
                "Hypothesis `{}` has exactly the goal type - use `exact {}`",
                decl.name, decl.name
            ));
        }
    }

    if hints.is_empty() {
        hints.push(
            "No specific hints available. Try `simp`, `trivial`, or search with `exact?`"
                .to_string(),
        );
    }

    Ok(hints)
}
