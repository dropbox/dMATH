//! Library search tactics
//!
//! Provides tactics for searching the environment for lemmas that can close or advance goals.

use lean5_kernel::name::Name;
use lean5_kernel::{Expr, Level, TypeChecker};

use super::{apply, can_apply_to_produce, types_unify, ProofState, TacticError, TacticResult};

// =============================================================================
// Library Search Tactics
// =============================================================================

/// Configuration for library_search
#[derive(Debug, Clone)]
pub struct LibrarySearchConfig {
    /// Maximum number of results to return
    pub max_results: usize,
    /// Whether to include partial matches (lemmas that could apply with arguments)
    pub include_partial: bool,
    /// Whether to search type class instances
    pub search_instances: bool,
    /// Minimum relevance score (0.0-1.0)
    pub min_relevance: f64,
    /// Whether to search in local context first
    pub prefer_local: bool,
}

impl Default for LibrarySearchConfig {
    fn default() -> Self {
        LibrarySearchConfig {
            max_results: 20,
            include_partial: true,
            search_instances: true,
            min_relevance: 0.1,
            prefer_local: true,
        }
    }
}

/// A library search result with relevance information
#[derive(Debug, Clone)]
pub struct LibrarySearchResult {
    /// The name of the found lemma/theorem
    pub name: Name,
    /// The expression (constant with universe levels)
    pub expr: Expr,
    /// The type of the lemma
    pub type_: Expr,
    /// Relevance score (0.0-1.0, higher is better)
    pub relevance: f64,
    /// Suggested tactic to use this lemma
    pub suggestion: String,
    /// Number of arguments needed to apply
    pub args_needed: usize,
    /// Whether this is from local context
    pub is_local: bool,
    /// Category of match
    pub match_kind: LibrarySearchMatchKind,
}

/// The kind of match found by library_search
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LibrarySearchMatchKind {
    /// Exact type match - can use `exact`
    Exact,
    /// Matches after applying to some arguments - can use `apply`
    Apply,
    /// Matches goal head but different arguments - may need `convert`
    HeadMatch,
    /// Related by type similarity
    TypeSimilar,
    /// Instance that could help
    Instance,
}

/// The library_search tactic: search for lemmas that can close or advance the goal
///
/// This is an enhanced version of `exact?` and `apply?` that:
/// - Returns multiple candidates ranked by relevance
/// - Searches both local context and environment
/// - Handles partial matches and type class instances
/// - Provides confidence scores and suggested tactics
///
/// # Example
/// ```text
/// -- Goal: a + b = b + a
/// library_search  -- finds Nat.add_comm, suggests "exact Nat.add_comm a b"
/// ```
pub fn library_search(state: &mut ProofState) -> Result<Vec<LibrarySearchResult>, TacticError> {
    library_search_with_config(state, LibrarySearchConfig::default())
}

/// Library search with custom configuration
pub fn library_search_with_config(
    state: &mut ProofState,
    config: LibrarySearchConfig,
) -> Result<Vec<LibrarySearchResult>, TacticError> {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?;
    let target = state.metas.instantiate(&goal.target);
    let local_ctx = goal.local_ctx.clone();

    let mut results: Vec<LibrarySearchResult> = Vec::new();

    // Extract goal head for similarity matching
    let goal_head = extract_head_name(&target);

    // 1. Search local hypotheses (higher priority)
    if config.prefer_local {
        for decl in &local_ctx {
            if results.len() >= config.max_results {
                break;
            }

            let tc = TypeChecker::new(&state.env);

            // Exact match
            if tc.is_def_eq(&decl.ty, &target) {
                results.push(LibrarySearchResult {
                    name: Name::from_string(&decl.name),
                    expr: Expr::fvar(decl.fvar),
                    type_: decl.ty.clone(),
                    relevance: 1.0,
                    suggestion: format!("exact {}", decl.name),
                    args_needed: 0,
                    is_local: true,
                    match_kind: LibrarySearchMatchKind::Exact,
                });
                continue;
            }

            // Apply match
            if config.include_partial {
                if let Some(args) = can_apply_to_produce(&state.env, &decl.ty, &target, 10) {
                    let arg_count = args.len();
                    let relevance = 0.9 - (arg_count as f64 * 0.05);
                    if relevance >= config.min_relevance {
                        results.push(LibrarySearchResult {
                            name: Name::from_string(&decl.name),
                            expr: Expr::fvar(decl.fvar),
                            type_: decl.ty.clone(),
                            relevance,
                            suggestion: if arg_count == 0 {
                                format!("exact {}", decl.name)
                            } else {
                                format!("apply {}", decl.name)
                            },
                            args_needed: arg_count,
                            is_local: true,
                            match_kind: LibrarySearchMatchKind::Apply,
                        });
                    }
                }
            }

            // Head match
            if let Some(ref gh) = goal_head {
                if let Some(hyp_head) = extract_head_name(&decl.ty) {
                    if gh == &hyp_head {
                        let relevance = 0.5;
                        if relevance >= config.min_relevance
                            && !results.iter().any(|r| r.name.to_string() == decl.name)
                        {
                            results.push(LibrarySearchResult {
                                name: Name::from_string(&decl.name),
                                expr: Expr::fvar(decl.fvar),
                                type_: decl.ty.clone(),
                                relevance,
                                suggestion: format!("convert {}", decl.name),
                                args_needed: 0,
                                is_local: true,
                                match_kind: LibrarySearchMatchKind::HeadMatch,
                            });
                        }
                    }
                }
            }
        }
    }

    // 2. Search constants in environment
    for constant in state.env.constants() {
        if results.len() >= config.max_results * 2 {
            // Collect more, will sort and trim
            break;
        }

        // Skip internal names
        let name_str = constant.name.to_string();
        if name_str.starts_with('_') || name_str.contains("._") {
            continue;
        }

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
            let subst: Vec<(Name, Level)> = constant
                .level_params
                .iter()
                .cloned()
                .zip(levels.iter().cloned())
                .collect();
            constant.type_.instantiate_level_params(&subst)
        };

        // Exact match
        if types_unify(&state.env, &const_type, &target) {
            results.push(LibrarySearchResult {
                name: constant.name.clone(),
                expr: Expr::const_(constant.name.clone(), levels.clone()),
                type_: const_type.clone(),
                relevance: 0.95, // Slightly lower than local
                suggestion: format!("exact {}", constant.name),
                args_needed: 0,
                is_local: false,
                match_kind: LibrarySearchMatchKind::Exact,
            });
            continue;
        }

        // Apply match
        if config.include_partial {
            if let Some(args) = can_apply_to_produce(&state.env, &const_type, &target, 10) {
                let arg_count = args.len();
                let relevance = 0.85 - (arg_count as f64 * 0.05);
                if relevance >= config.min_relevance {
                    results.push(LibrarySearchResult {
                        name: constant.name.clone(),
                        expr: Expr::const_(constant.name.clone(), levels.clone()),
                        type_: const_type.clone(),
                        relevance,
                        suggestion: if arg_count == 0 {
                            format!("exact {}", constant.name)
                        } else {
                            format!("apply {}", constant.name)
                        },
                        args_needed: arg_count,
                        is_local: false,
                        match_kind: LibrarySearchMatchKind::Apply,
                    });
                    continue;
                }
            }
        }

        // Head match for type similarity
        if let Some(ref gh) = goal_head {
            if let Some(const_head) = extract_head_name(&const_type) {
                if gh == &const_head {
                    let relevance = calculate_type_similarity(&const_type, &target);
                    if relevance >= config.min_relevance {
                        results.push(LibrarySearchResult {
                            name: constant.name.clone(),
                            expr: Expr::const_(constant.name.clone(), levels.clone()),
                            type_: const_type.clone(),
                            relevance,
                            suggestion: format!("-- similar: {}", constant.name),
                            args_needed: 0,
                            is_local: false,
                            match_kind: LibrarySearchMatchKind::TypeSimilar,
                        });
                    }
                }
            }
        }

        // Instance search
        if config.search_instances
            && (name_str.contains("inst") || name_str.contains("Instance"))
            && can_apply_to_produce(&state.env, &const_type, &target, 5).is_some()
        {
            results.push(LibrarySearchResult {
                name: constant.name.clone(),
                expr: Expr::const_(constant.name.clone(), levels),
                type_: const_type,
                relevance: 0.3,
                suggestion: format!("-- instance: {}", constant.name),
                args_needed: 0,
                is_local: false,
                match_kind: LibrarySearchMatchKind::Instance,
            });
        }
    }

    // Sort by relevance (highest first)
    results.sort_by(|a, b| {
        b.relevance
            .partial_cmp(&a.relevance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Trim to max results
    results.truncate(config.max_results);

    Ok(results)
}

/// Extract the head constant name from an expression
pub(crate) fn extract_head_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Const(name, _) => Some(name.to_string()),
        Expr::App(f, _) => extract_head_name(f),
        Expr::Pi(_, _, body) => {
            // For ∀ types, look at the conclusion
            extract_head_name(body)
        }
        _ => None,
    }
}

/// Calculate a similarity score between two types
pub(crate) fn calculate_type_similarity(t1: &Expr, t2: &Expr) -> f64 {
    // Simple structural similarity
    let depth1 = expr_depth(t1);
    let depth2 = expr_depth(t2);

    let head1 = extract_head_name(t1);
    let head2 = extract_head_name(t2);

    let mut score: f64 = 0.2;

    // Same head
    if head1 == head2 && head1.is_some() {
        score += 0.3;
    }

    // Similar depth
    let depth_diff = (depth1 as i32 - depth2 as i32).abs();
    if depth_diff <= 1 {
        score += 0.2;
    } else if depth_diff <= 3 {
        score += 0.1;
    }

    // Check Pi structure similarity
    let pi_count1 = count_pis(t1);
    let pi_count2 = count_pis(t2);
    if (pi_count1 as i32 - pi_count2 as i32).abs() <= 1 {
        score += 0.1;
    }

    score.min(1.0)
}

/// Count depth of expression
pub(crate) fn expr_depth(expr: &Expr) -> usize {
    match expr {
        Expr::App(f, a) => 1 + expr_depth(f).max(expr_depth(a)),
        Expr::Lam(_, ty, body) | Expr::Pi(_, ty, body) => 1 + expr_depth(ty).max(expr_depth(body)),
        Expr::Let(ty, val, body) => 1 + expr_depth(ty).max(expr_depth(val)).max(expr_depth(body)),
        _ => 1,
    }
}

/// Count number of Pi binders
pub(crate) fn count_pis(expr: &Expr) -> usize {
    match expr {
        Expr::Pi(_, _, body) => 1 + count_pis(body),
        _ => 0,
    }
}

/// Apply library_search and use the best result
pub fn library_search_and_apply(state: &mut ProofState) -> TacticResult {
    let results = library_search(state)?;

    if results.is_empty() {
        return Err(TacticError::Other(
            "library_search: no matching lemmas found".to_string(),
        ));
    }

    // Get the best result
    let best = &results[0];

    match best.match_kind {
        LibrarySearchMatchKind::Exact => {
            // Use exact
            state.metas.assign(
                state.current_goal().ok_or(TacticError::NoGoals)?.meta_id,
                best.expr.clone(),
            );
            state.goals.remove(0);
            Ok(())
        }
        LibrarySearchMatchKind::Apply => {
            // Use apply (creates subgoals for arguments)
            apply(state, best.expr.clone())
        }
        _ => {
            // Other match kinds can't be automatically applied
            Err(TacticError::Other(format!(
                "library_search: found {} but cannot auto-apply ({:?})",
                best.name, best.match_kind
            )))
        }
    }
}

/// Show library_search results without applying
pub fn library_search_show(state: &mut ProofState) -> Result<String, TacticError> {
    let results = library_search(state)?;

    if results.is_empty() {
        return Ok("No matching lemmas found.".to_string());
    }

    use std::fmt::Write;
    let mut output = String::new();
    writeln!(output, "Found {} potential matches:", results.len()).unwrap();

    for (i, result) in results.iter().enumerate() {
        writeln!(
            output,
            "{}. {} (relevance: {:.2}, {:?})\n   {}",
            i + 1,
            result.name,
            result.relevance,
            result.match_kind,
            result.suggestion
        )
        .unwrap();
    }

    Ok(output)
}
