//! Interpolation-based lemma learning for PDR/Spacer
//!
//! This module implements interpolating SAT solving for CHC problems.
//! When A ∧ B is UNSAT, we compute an interpolant I such that:
//! - A ⊨ I (I is implied by A)
//! - I ∧ B is UNSAT (I is inconsistent with B)
//! - I uses only variables shared between A and B
//!
//! This is the core technique from Golem/Spacer for learning more general
//! blocking lemmas than point-based blocking.
//!
//! ## Implementation
//!
//! Currently uses Farkas-based interpolation for linear arithmetic.
//! Future: integrate with proof-producing SMT solver for richer interpolants.

use crate::farkas::compute_interpolant as farkas_interpolant;
use crate::{ChcExpr, ChcOp, ChcSort, ChcVar};
use rustc_hash::FxHashSet;

/// Result of interpolant computation
#[derive(Debug, Clone)]
pub enum InterpolatingSatResult {
    /// A ∧ B is unsatisfiable, with interpolant
    Unsat(ChcExpr),
    /// Unknown (could not determine)
    Unknown,
}

/// Compute an interpolant from constraint lists.
///
/// When A (transition constraints) ∧ B (bad state) is UNSAT, compute an
/// interpolant I such that:
/// - A ⊨ I
/// - I ∧ B is UNSAT
/// - I mentions only variables shared between A and B
pub fn interpolating_sat_constraints(
    a_constraints: &[ChcExpr],
    b_constraints: &[ChcExpr],
    shared_vars: &FxHashSet<String>,
) -> InterpolatingSatResult {
    if a_constraints.is_empty() || b_constraints.is_empty() {
        return InterpolatingSatResult::Unknown;
    }

    // Try Farkas-based interpolation first
    if let Some(interpolant) = farkas_interpolant(a_constraints, b_constraints, shared_vars) {
        return InterpolatingSatResult::Unsat(interpolant);
    }

    // Try bound-based interpolation
    if let Some(interpolant) = compute_bound_interpolant(a_constraints, b_constraints, shared_vars)
    {
        return InterpolatingSatResult::Unsat(interpolant);
    }

    // Try transitivity-based interpolation
    if let Some(interpolant) =
        compute_transitivity_interpolant(a_constraints, b_constraints, shared_vars)
    {
        return InterpolatingSatResult::Unsat(interpolant);
    }

    InterpolatingSatResult::Unknown
}

/// Compute a simple bound interpolant from A and B constraints
///
/// Looks for cases where A implies a bound on a shared variable
/// that contradicts B.
fn compute_bound_interpolant(
    a_constraints: &[ChcExpr],
    b_constraints: &[ChcExpr],
    shared_vars: &FxHashSet<String>,
) -> Option<ChcExpr> {
    // Extract bounds from A constraints
    let a_bounds = extract_variable_bounds(a_constraints);

    // Extract bounds from B constraints
    let b_bounds = extract_variable_bounds(b_constraints);

    // Look for contradicting bounds on shared variables
    for (var, (a_lower, a_upper)) in &a_bounds {
        if !shared_vars.contains(var) {
            continue;
        }

        if let Some((b_lower, b_upper)) = b_bounds.get(var) {
            // A says var >= a_lower, B says var < b_upper where b_upper <= a_lower
            if let (Some(a_lb), Some(b_ub)) = (a_lower, b_upper) {
                if *b_ub < *a_lb {
                    // Interpolant: var >= a_lb
                    let v = ChcVar::new(var, ChcSort::Int);
                    return Some(ChcExpr::ge(ChcExpr::var(v), ChcExpr::Int(*a_lb)));
                }
            }

            // A says var <= a_upper, B says var > b_lower where b_lower >= a_upper
            if let (Some(a_ub), Some(b_lb)) = (a_upper, b_lower) {
                if *b_lb > *a_ub {
                    // Interpolant: var <= a_ub
                    let v = ChcVar::new(var, ChcSort::Int);
                    return Some(ChcExpr::le(ChcExpr::var(v), ChcExpr::Int(*a_ub)));
                }
            }
        }
    }

    None
}

/// Extract variable bounds from a list of constraints
/// Returns map from variable name to (Option<lower_bound>, Option<upper_bound>)
fn extract_variable_bounds(
    constraints: &[ChcExpr],
) -> rustc_hash::FxHashMap<String, (Option<i64>, Option<i64>)> {
    use rustc_hash::FxHashMap;
    let mut bounds: FxHashMap<String, (Option<i64>, Option<i64>)> = FxHashMap::default();

    for c in constraints {
        if let Some((var, bound, is_upper)) = extract_simple_bound(c) {
            let entry = bounds.entry(var).or_insert((None, None));
            if is_upper {
                entry.1 = Some(entry.1.map_or(bound, |b| b.min(bound)));
            } else {
                entry.0 = Some(entry.0.map_or(bound, |b| b.max(bound)));
            }
        }
    }

    bounds
}

/// Extract a simple bound from an expression: var <= c or var >= c
/// Returns (variable_name, bound_value, is_upper_bound)
fn extract_simple_bound(expr: &ChcExpr) -> Option<(String, i64, bool)> {
    match expr {
        // var <= c
        ChcExpr::Op(ChcOp::Le, args) if args.len() == 2 => {
            match (args[0].as_ref(), args[1].as_ref()) {
                (ChcExpr::Var(v), ChcExpr::Int(c)) if matches!(v.sort, ChcSort::Int) => {
                    Some((v.name.clone(), *c, true))
                }
                _ => None,
            }
        }
        // var < c  =>  var <= c-1
        ChcExpr::Op(ChcOp::Lt, args) if args.len() == 2 => {
            match (args[0].as_ref(), args[1].as_ref()) {
                (ChcExpr::Var(v), ChcExpr::Int(c)) if matches!(v.sort, ChcSort::Int) => {
                    Some((v.name.clone(), *c - 1, true))
                }
                _ => None,
            }
        }
        // var >= c
        ChcExpr::Op(ChcOp::Ge, args) if args.len() == 2 => {
            match (args[0].as_ref(), args[1].as_ref()) {
                (ChcExpr::Var(v), ChcExpr::Int(c)) if matches!(v.sort, ChcSort::Int) => {
                    Some((v.name.clone(), *c, false))
                }
                _ => None,
            }
        }
        // var > c  =>  var >= c+1
        ChcExpr::Op(ChcOp::Gt, args) if args.len() == 2 => {
            match (args[0].as_ref(), args[1].as_ref()) {
                (ChcExpr::Var(v), ChcExpr::Int(c)) if matches!(v.sort, ChcSort::Int) => {
                    Some((v.name.clone(), *c + 1, false))
                }
                _ => None,
            }
        }
        // c <= var  =>  var >= c
        ChcExpr::Op(ChcOp::Le, args) if args.len() == 2 => {
            match (args[0].as_ref(), args[1].as_ref()) {
                (ChcExpr::Int(c), ChcExpr::Var(v)) if matches!(v.sort, ChcSort::Int) => {
                    Some((v.name.clone(), *c, false))
                }
                _ => None,
            }
        }
        // c >= var  =>  var <= c
        ChcExpr::Op(ChcOp::Ge, args) if args.len() == 2 => {
            match (args[0].as_ref(), args[1].as_ref()) {
                (ChcExpr::Int(c), ChcExpr::Var(v)) if matches!(v.sort, ChcSort::Int) => {
                    Some((v.name.clone(), *c, true))
                }
                _ => None,
            }
        }
        // var = c  =>  var >= c AND var <= c
        ChcExpr::Op(ChcOp::Eq, args) if args.len() == 2 => {
            match (args[0].as_ref(), args[1].as_ref()) {
                (ChcExpr::Var(v), ChcExpr::Int(c)) | (ChcExpr::Int(c), ChcExpr::Var(v))
                    if matches!(v.sort, ChcSort::Int) =>
                {
                    // Return as lower bound; caller should handle both
                    Some((v.name.clone(), *c, false))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// Compute a transitivity-based interpolant
///
/// For constraint chains like: A says x <= y, B says y < x
/// Derive contradiction and produce interpolant involving shared variables.
fn compute_transitivity_interpolant(
    a_constraints: &[ChcExpr],
    b_constraints: &[ChcExpr],
    shared_vars: &FxHashSet<String>,
) -> Option<ChcExpr> {
    // Extract relational constraints from A (x <= y + c, x < y + c, etc.)
    let a_relations = extract_relational_constraints(a_constraints);

    // Extract relational constraints from B
    let b_relations = extract_relational_constraints(b_constraints);

    // Look for transitivity contradictions
    // A: x - y <= c1, B: y - x <= c2 where c1 + c2 < 0 is contradiction
    for (a_vars, a_bound) in &a_relations {
        if a_vars.0 == a_vars.1 {
            continue;
        }

        // Look for opposite relation in B
        let opposite = (a_vars.1.clone(), a_vars.0.clone());
        for (b_vars, b_bound) in &b_relations {
            if *b_vars == opposite && a_bound + b_bound < 0 {
                // Found contradiction!
                // Both variables must be shared for useful interpolant
                if shared_vars.contains(&a_vars.0) && shared_vars.contains(&a_vars.1) {
                    // Interpolant: x - y <= a_bound (from A)
                    let x = ChcVar::new(&a_vars.0, ChcSort::Int);
                    let y = ChcVar::new(&a_vars.1, ChcSort::Int);
                    return Some(ChcExpr::le(
                        ChcExpr::sub(ChcExpr::var(x), ChcExpr::var(y)),
                        ChcExpr::Int(*a_bound),
                    ));
                }
            }
        }
    }

    None
}

/// Extract relational constraints of form x - y <= c
/// Returns list of ((x, y), c) tuples
fn extract_relational_constraints(constraints: &[ChcExpr]) -> Vec<((String, String), i64)> {
    let mut result = Vec::new();

    for c in constraints {
        if let Some(rel) = extract_difference_constraint(c) {
            result.push(rel);
        }
    }

    result
}

/// Extract a difference constraint: x - y <= c or x - y < c
fn extract_difference_constraint(expr: &ChcExpr) -> Option<((String, String), i64)> {
    match expr {
        ChcExpr::Op(ChcOp::Le, args) if args.len() == 2 => {
            extract_difference_lhs(&args[0], &args[1], 0)
        }
        ChcExpr::Op(ChcOp::Lt, args) if args.len() == 2 => {
            extract_difference_lhs(&args[0], &args[1], -1)
        }
        ChcExpr::Op(ChcOp::Ge, args) if args.len() == 2 => {
            // y >= x + c  =>  x - y <= -c
            extract_difference_lhs(&args[1], &args[0], 0).map(|((x, y), c)| ((y, x), -c))
        }
        ChcExpr::Op(ChcOp::Gt, args) if args.len() == 2 => {
            extract_difference_lhs(&args[1], &args[0], -1).map(|((x, y), c)| ((y, x), -c))
        }
        _ => None,
    }
}

/// Extract x - y from LHS, c from RHS
fn extract_difference_lhs(
    lhs: &ChcExpr,
    rhs: &ChcExpr,
    adjust: i64,
) -> Option<((String, String), i64)> {
    let c = match rhs {
        ChcExpr::Int(n) => *n + adjust,
        _ => return None,
    };

    match lhs {
        ChcExpr::Op(ChcOp::Sub, args) if args.len() == 2 => {
            match (args[0].as_ref(), args[1].as_ref()) {
                (ChcExpr::Var(x), ChcExpr::Var(y))
                    if matches!(x.sort, ChcSort::Int) && matches!(y.sort, ChcSort::Int) =>
                {
                    Some(((x.name.clone(), y.name.clone()), c))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolating_sat_bound_contradiction() {
        // A: x >= 10
        // B: x <= 5
        // Should be UNSAT with interpolant x >= 10
        let x = ChcVar::new("x", ChcSort::Int);
        let a = ChcExpr::ge(ChcExpr::var(x.clone()), ChcExpr::Int(10));
        let b = ChcExpr::le(ChcExpr::var(x.clone()), ChcExpr::Int(5));

        let shared: FxHashSet<String> = ["x".to_string()].into_iter().collect();

        match interpolating_sat_constraints(&[a], &[b], &shared) {
            InterpolatingSatResult::Unsat(interp) => {
                // Interpolant should be x >= 10 or equivalent
                println!("Interpolant: {}", interp);
            }
            other => panic!("Expected Unsat, got {:?}", other),
        }
    }

    #[test]
    fn test_extract_simple_bounds() {
        let x = ChcVar::new("x", ChcSort::Int);

        // x <= 5
        let le = ChcExpr::le(ChcExpr::var(x.clone()), ChcExpr::Int(5));
        assert_eq!(extract_simple_bound(&le), Some(("x".to_string(), 5, true)));

        // x >= 3
        let ge = ChcExpr::ge(ChcExpr::var(x.clone()), ChcExpr::Int(3));
        assert_eq!(extract_simple_bound(&ge), Some(("x".to_string(), 3, false)));
    }
}
