//! Farkas lemma based constraint combination
//!
//! This module implements Farkas-based combination of linear constraints
//! for generating interpolants in PDR. When a set of linear inequalities
//! is UNSAT, Farkas' lemma guarantees there exist non-negative coefficients
//! that when used to combine the inequalities, produce a contradiction.
//!
//! The combined constraint is often more general than the original constraints,
//! making it useful for lemma generalization in PDR.
//!
//! ## Algorithm
//!
//! Given constraints: a₁·x ≤ b₁, a₂·x ≤ b₂, ..., aₙ·x ≤ bₙ that are UNSAT,
//! find λ₁, ..., λₙ ≥ 0 such that:
//! - Σᵢ λᵢ·aᵢ = 0  (coefficients cancel)
//! - Σᵢ λᵢ·bᵢ < 0  (RHS is negative)
//!
//! The combined constraint Σᵢ λᵢ·(aᵢ·x - bᵢ) ≤ 0 is a valid lemma.

use crate::{ChcExpr, ChcOp, ChcSort, ChcVar};
use num_rational::Rational64;
use num_traits::Signed;
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::Arc;

/// A linear constraint in the form: Σᵢ aᵢ·xᵢ ≤ b (or < for strict)
#[derive(Debug, Clone)]
pub struct LinearConstraint {
    /// Variable name -> coefficient
    pub coeffs: FxHashMap<String, Rational64>,
    /// Constant bound (RHS)
    pub bound: Rational64,
    /// Whether this is strict (< vs ≤)
    #[allow(dead_code)] // Used for debugging and future strict inequality handling
    pub strict: bool,
    /// Original expression (for reference)
    #[allow(dead_code)] // Used for debugging and proof generation
    pub original: ChcExpr,
}

impl LinearConstraint {
    fn new(bound: Rational64, strict: bool, original: ChcExpr) -> Self {
        Self {
            coeffs: FxHashMap::default(),
            bound,
            strict,
            original,
        }
    }

    fn set_coeff(&mut self, var: &str, coeff: Rational64) {
        if coeff == Rational64::from_integer(0) {
            self.coeffs.remove(var);
        } else {
            self.coeffs.insert(var.to_string(), coeff);
        }
    }

    fn get_coeff(&self, var: &str) -> Rational64 {
        self.coeffs
            .get(var)
            .copied()
            .unwrap_or(Rational64::from_integer(0))
    }
}

/// Try to parse a ChcExpr as a linear constraint.
/// Returns None if the expression is not a linear inequality.
pub fn parse_linear_constraint(expr: &ChcExpr) -> Option<LinearConstraint> {
    match expr {
        // a ≤ b  =>  a - b ≤ 0
        ChcExpr::Op(ChcOp::Le, args) if args.len() == 2 => {
            let mut constraint =
                LinearConstraint::new(Rational64::from_integer(0), false, expr.clone());
            add_linear_expr(&args[0], Rational64::from_integer(1), &mut constraint)?;
            add_linear_expr(&args[1], Rational64::from_integer(-1), &mut constraint)?;
            Some(constraint)
        }
        // a < b  =>  a - b < 0
        ChcExpr::Op(ChcOp::Lt, args) if args.len() == 2 => {
            let mut constraint =
                LinearConstraint::new(Rational64::from_integer(0), true, expr.clone());
            add_linear_expr(&args[0], Rational64::from_integer(1), &mut constraint)?;
            add_linear_expr(&args[1], Rational64::from_integer(-1), &mut constraint)?;
            Some(constraint)
        }
        // a ≥ b  =>  b - a ≤ 0
        ChcExpr::Op(ChcOp::Ge, args) if args.len() == 2 => {
            let mut constraint =
                LinearConstraint::new(Rational64::from_integer(0), false, expr.clone());
            add_linear_expr(&args[1], Rational64::from_integer(1), &mut constraint)?;
            add_linear_expr(&args[0], Rational64::from_integer(-1), &mut constraint)?;
            Some(constraint)
        }
        // a > b  =>  b - a < 0
        ChcExpr::Op(ChcOp::Gt, args) if args.len() == 2 => {
            let mut constraint =
                LinearConstraint::new(Rational64::from_integer(0), true, expr.clone());
            add_linear_expr(&args[1], Rational64::from_integer(1), &mut constraint)?;
            add_linear_expr(&args[0], Rational64::from_integer(-1), &mut constraint)?;
            Some(constraint)
        }
        // a = b  =>  a - b ≤ 0 AND b - a ≤ 0 (we return one direction)
        ChcExpr::Op(ChcOp::Eq, args) if args.len() == 2 => {
            // For equalities, we treat as a <= b (caller may want both directions)
            let mut constraint =
                LinearConstraint::new(Rational64::from_integer(0), false, expr.clone());
            add_linear_expr(&args[0], Rational64::from_integer(1), &mut constraint)?;
            add_linear_expr(&args[1], Rational64::from_integer(-1), &mut constraint)?;
            Some(constraint)
        }
        // Handle negated comparisons
        ChcExpr::Op(ChcOp::Not, args) if args.len() == 1 => {
            match args[0].as_ref() {
                // NOT(a ≤ b)  =>  a > b  =>  b - a < 0
                ChcExpr::Op(ChcOp::Le, inner_args) if inner_args.len() == 2 => {
                    let mut constraint =
                        LinearConstraint::new(Rational64::from_integer(0), true, expr.clone());
                    add_linear_expr(&inner_args[1], Rational64::from_integer(1), &mut constraint)?;
                    add_linear_expr(
                        &inner_args[0],
                        Rational64::from_integer(-1),
                        &mut constraint,
                    )?;
                    Some(constraint)
                }
                // NOT(a < b)  =>  a ≥ b  =>  b - a ≤ 0
                ChcExpr::Op(ChcOp::Lt, inner_args) if inner_args.len() == 2 => {
                    let mut constraint =
                        LinearConstraint::new(Rational64::from_integer(0), false, expr.clone());
                    add_linear_expr(&inner_args[1], Rational64::from_integer(1), &mut constraint)?;
                    add_linear_expr(
                        &inner_args[0],
                        Rational64::from_integer(-1),
                        &mut constraint,
                    )?;
                    Some(constraint)
                }
                // NOT(a ≥ b)  =>  a < b  =>  a - b < 0
                ChcExpr::Op(ChcOp::Ge, inner_args) if inner_args.len() == 2 => {
                    let mut constraint =
                        LinearConstraint::new(Rational64::from_integer(0), true, expr.clone());
                    add_linear_expr(&inner_args[0], Rational64::from_integer(1), &mut constraint)?;
                    add_linear_expr(
                        &inner_args[1],
                        Rational64::from_integer(-1),
                        &mut constraint,
                    )?;
                    Some(constraint)
                }
                // NOT(a > b)  =>  a ≤ b  =>  a - b ≤ 0
                ChcExpr::Op(ChcOp::Gt, inner_args) if inner_args.len() == 2 => {
                    let mut constraint =
                        LinearConstraint::new(Rational64::from_integer(0), false, expr.clone());
                    add_linear_expr(&inner_args[0], Rational64::from_integer(1), &mut constraint)?;
                    add_linear_expr(
                        &inner_args[1],
                        Rational64::from_integer(-1),
                        &mut constraint,
                    )?;
                    Some(constraint)
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// Add a linear expression to a constraint with a multiplier.
/// Returns None if the expression is not linear.
fn add_linear_expr(
    expr: &ChcExpr,
    mult: Rational64,
    constraint: &mut LinearConstraint,
) -> Option<()> {
    match expr {
        ChcExpr::Int(n) => {
            constraint.bound -= mult * Rational64::from_integer(*n);
            Some(())
        }
        ChcExpr::Var(v) if matches!(v.sort, ChcSort::Int) => {
            let current = constraint.get_coeff(&v.name);
            constraint.set_coeff(&v.name, current + mult);
            Some(())
        }
        ChcExpr::Op(ChcOp::Add, args) => {
            for arg in args {
                add_linear_expr(arg, mult, constraint)?;
            }
            Some(())
        }
        ChcExpr::Op(ChcOp::Sub, args) if !args.is_empty() => {
            add_linear_expr(&args[0], mult, constraint)?;
            for arg in args.iter().skip(1) {
                add_linear_expr(arg, -mult, constraint)?;
            }
            Some(())
        }
        ChcExpr::Op(ChcOp::Neg, args) if args.len() == 1 => {
            add_linear_expr(&args[0], -mult, constraint)
        }
        ChcExpr::Op(ChcOp::Mul, args) if args.len() == 2 => {
            // Only handle constant * var or var * constant
            match (args[0].as_ref(), args[1].as_ref()) {
                (ChcExpr::Int(c), other) | (other, ChcExpr::Int(c)) => {
                    add_linear_expr(other, mult * Rational64::from_integer(*c), constraint)
                }
                _ => None, // Non-linear
            }
        }
        _ => None, // Not a linear expression
    }
}

/// Result of Farkas combination
#[derive(Debug, Clone)]
pub struct FarkasCombination {
    /// The combined constraint as a ChcExpr
    pub combined: ChcExpr,
    /// Whether the combination is strict (< vs ≤)
    #[allow(dead_code)] // Used for future strict inequality handling
    pub strict: bool,
}

/// Try to combine linear constraints using Farkas coefficients.
///
/// Given a set of linear inequalities that are UNSAT, this finds non-negative
/// coefficients λᵢ such that the weighted sum of constraints produces a
/// contradiction (0 < 0 or sum < 0).
///
/// Returns the combined constraint if successful.
pub fn farkas_combine(constraints: &[ChcExpr]) -> Option<FarkasCombination> {
    // Parse all constraints as linear inequalities
    let linear: Vec<LinearConstraint> = constraints
        .iter()
        .filter_map(parse_linear_constraint)
        .collect();

    if linear.len() < 2 {
        return None; // Need at least 2 constraints to combine
    }

    // Collect all variables
    let mut all_vars: Vec<String> = linear
        .iter()
        .flat_map(|c| c.coeffs.keys().cloned())
        .collect();
    all_vars.sort();
    all_vars.dedup();

    if all_vars.is_empty() {
        return None; // No variables, nothing to combine
    }

    // Strategy 1: Equal weights (λᵢ = 1 for all i)
    // This works when the constraints directly sum to a contradiction.
    if let Some(fc) = try_equal_weights(&linear) {
        return Some(fc);
    }

    // Strategy 2: Variable elimination by opposite signs
    // If a variable appears with opposite signs in exactly 2 constraints,
    // combine them to eliminate that variable.
    if let Some(fc) = try_variable_elimination(&linear, &all_vars) {
        return Some(fc);
    }

    // Strategy 3: Transitivity chains
    // For constraints like x <= y + c1, y <= z + c2, derive x <= z + (c1+c2)
    if let Some(fc) = try_transitivity_chain(&linear, &all_vars) {
        return Some(fc);
    }

    // Strategy 4: Bound tightening
    // Combine multiple bounds on the same variable to get tighter bounds
    if let Some(fc) = try_bound_tightening(&linear, &all_vars) {
        return Some(fc);
    }

    None
}

/// Strategy 1: Try equal weights (all λᵢ = 1)
fn try_equal_weights(linear: &[LinearConstraint]) -> Option<FarkasCombination> {
    let mut sum_coeffs: FxHashMap<String, Rational64> = FxHashMap::default();
    let mut sum_bound = Rational64::from_integer(0);
    let mut any_strict = false;

    for c in linear {
        for (var, coeff) in &c.coeffs {
            let current = sum_coeffs
                .get(var)
                .copied()
                .unwrap_or(Rational64::from_integer(0));
            sum_coeffs.insert(var.clone(), current + *coeff);
        }
        sum_bound += c.bound;
        any_strict = any_strict || c.strict;
    }

    // Check if coefficients cancel (Σᵢ aᵢ = 0 for all variables)
    let coeffs_cancel = sum_coeffs
        .values()
        .all(|c| *c == Rational64::from_integer(0));

    if coeffs_cancel
        && (sum_bound < Rational64::from_integer(0)
            || (any_strict && sum_bound <= Rational64::from_integer(0)))
    {
        // Pure contradiction: 0 ≤ negative or 0 < 0
        return Some(FarkasCombination {
            combined: ChcExpr::Bool(false),
            strict: any_strict,
        });
    }

    None
}

/// Strategy 2: Variable elimination by opposite signs
fn try_variable_elimination(
    linear: &[LinearConstraint],
    all_vars: &[String],
) -> Option<FarkasCombination> {
    for var in all_vars {
        let relevant: Vec<usize> = linear
            .iter()
            .enumerate()
            .filter(|(_, c)| c.coeffs.contains_key(var))
            .map(|(i, _)| i)
            .collect();

        if relevant.len() == 2 {
            let c1 = &linear[relevant[0]];
            let c2 = &linear[relevant[1]];
            let coeff1 = c1.get_coeff(var);
            let coeff2 = c2.get_coeff(var);

            // Check if they have opposite signs
            if (coeff1 > Rational64::from_integer(0)) != (coeff2 > Rational64::from_integer(0)) {
                // Combine with weights |coeff2| and |coeff1| to eliminate var
                let w1 = coeff2.abs();
                let w2 = coeff1.abs();

                let mut combined_coeffs: FxHashMap<String, Rational64> = FxHashMap::default();
                for (v, c) in &c1.coeffs {
                    combined_coeffs.insert(v.clone(), *c * w1);
                }
                for (v, c) in &c2.coeffs {
                    let current = combined_coeffs
                        .get(v)
                        .copied()
                        .unwrap_or(Rational64::from_integer(0));
                    combined_coeffs.insert(v.clone(), current + *c * w2);
                }
                let combined_bound = c1.bound * w1 + c2.bound * w2;
                let combined_strict = c1.strict || c2.strict;

                // Remove zero coefficients
                combined_coeffs.retain(|_, c| *c != Rational64::from_integer(0));

                // Build the combined expression
                let combined_expr =
                    build_linear_inequality(&combined_coeffs, combined_bound, combined_strict);

                return Some(FarkasCombination {
                    combined: combined_expr,
                    strict: combined_strict,
                });
            }
        }
    }
    None
}

/// Strategy 3: Transitivity chains
/// For constraints like x - y <= c1, y - z <= c2, derive x - z <= c1 + c2
fn try_transitivity_chain(
    linear: &[LinearConstraint],
    all_vars: &[String],
) -> Option<FarkasCombination> {
    // Look for difference constraints: x - y <= c
    // These have exactly two variables with coefficients +1 and -1

    #[derive(Clone)]
    struct DiffConstraint {
        from: String, // positive coefficient variable
        to: String,   // negative coefficient variable
        bound: Rational64,
        strict: bool,
    }

    let mut diff_constraints: Vec<DiffConstraint> = Vec::new();

    for c in linear {
        if c.coeffs.len() == 2 {
            let coeffs: Vec<_> = c.coeffs.iter().collect();
            let (v1, c1) = coeffs[0];
            let (v2, c2) = coeffs[1];

            // Check for +1/-1 pattern
            if *c1 == Rational64::from_integer(1) && *c2 == Rational64::from_integer(-1) {
                diff_constraints.push(DiffConstraint {
                    from: v1.clone(),
                    to: v2.clone(),
                    bound: c.bound,
                    strict: c.strict,
                });
            } else if *c1 == Rational64::from_integer(-1) && *c2 == Rational64::from_integer(1) {
                diff_constraints.push(DiffConstraint {
                    from: v2.clone(),
                    to: v1.clone(),
                    bound: c.bound,
                    strict: c.strict,
                });
            }
        }
    }

    // Try to find a chain: x -> y -> z (3 or more variables)
    for start_var in all_vars {
        // Find all paths from start_var
        let mut visited: FxHashSet<String> = FxHashSet::default();
        visited.insert(start_var.clone());

        let mut frontier: Vec<(String, Rational64, bool)> =
            vec![(start_var.clone(), Rational64::from_integer(0), false)];

        while let Some((current, current_bound, current_strict)) = frontier.pop() {
            for dc in &diff_constraints {
                if dc.from == current && !visited.contains(&dc.to) {
                    let new_bound = current_bound + dc.bound;
                    let new_strict = current_strict || dc.strict;

                    // Check if we've found a path back to start (cycle)
                    // Or if we can combine with another constraint on the end variable
                    for c in linear {
                        // Look for a constraint on dc.to that would create a contradiction
                        // or a useful combination
                        if c.coeffs.len() == 1 {
                            if let Some(coeff) = c.coeffs.get(&dc.to) {
                                if *coeff == Rational64::from_integer(1) {
                                    // dc.to <= c.bound, and start_var - dc.to <= new_bound
                                    // so start_var <= new_bound + c.bound
                                    let combined_bound = new_bound + c.bound;
                                    let combined_strict = new_strict || c.strict;

                                    let mut combined_coeffs = FxHashMap::default();
                                    combined_coeffs
                                        .insert(start_var.clone(), Rational64::from_integer(1));

                                    return Some(FarkasCombination {
                                        combined: build_linear_inequality(
                                            &combined_coeffs,
                                            combined_bound,
                                            combined_strict,
                                        ),
                                        strict: combined_strict,
                                    });
                                } else if *coeff == Rational64::from_integer(-1) {
                                    // -dc.to <= c.bound means dc.to >= -c.bound
                                    // Combined with start_var - dc.to <= new_bound:
                                    // start_var >= -c.bound + (dc.to - start_var) >= -c.bound - new_bound
                                    // So start_var >= -(c.bound + new_bound)
                                    let combined_bound = -(c.bound + new_bound);
                                    let combined_strict = new_strict || c.strict;

                                    let mut combined_coeffs = FxHashMap::default();
                                    combined_coeffs
                                        .insert(start_var.clone(), Rational64::from_integer(-1));

                                    return Some(FarkasCombination {
                                        combined: build_linear_inequality(
                                            &combined_coeffs,
                                            combined_bound,
                                            combined_strict,
                                        ),
                                        strict: combined_strict,
                                    });
                                }
                            }
                        }
                    }

                    visited.insert(dc.to.clone());
                    frontier.push((dc.to.clone(), new_bound, new_strict));
                }
            }
        }
    }

    None
}

/// Strategy 4: Bound tightening
/// Combine multiple bounds on the same variable
fn try_bound_tightening(
    linear: &[LinearConstraint],
    all_vars: &[String],
) -> Option<FarkasCombination> {
    // For single-variable constraints, find the tightest bounds
    for var in all_vars {
        let mut upper_bounds: Vec<(Rational64, bool)> = Vec::new();
        let mut lower_bounds: Vec<(Rational64, bool)> = Vec::new();

        for c in linear {
            if c.coeffs.len() == 1 {
                if let Some(coeff) = c.coeffs.get(var) {
                    if *coeff == Rational64::from_integer(1) {
                        // var <= bound (after flipping sign)
                        upper_bounds.push((c.bound, c.strict));
                    } else if *coeff == Rational64::from_integer(-1) {
                        // -var <= bound means var >= -bound
                        lower_bounds.push((-c.bound, c.strict));
                    }
                }
            }
        }

        // If we have both upper and lower bounds, check for contradiction
        if !upper_bounds.is_empty() && !lower_bounds.is_empty() {
            let (min_upper, upper_strict) =
                upper_bounds.iter().min_by(|a, b| a.0.cmp(&b.0)).unwrap();
            let (max_lower, lower_strict) =
                lower_bounds.iter().max_by(|a, b| a.0.cmp(&b.0)).unwrap();

            if max_lower > min_upper || (*upper_strict || *lower_strict) && max_lower >= min_upper {
                // Contradiction: var >= max_lower and var <= min_upper but max_lower > min_upper
                return Some(FarkasCombination {
                    combined: ChcExpr::Bool(false),
                    strict: *upper_strict || *lower_strict,
                });
            }
        }
    }

    None
}

/// Build a ChcExpr from a linear constraint: Σᵢ aᵢ·xᵢ ≤ b (or <)
fn build_linear_inequality(
    coeffs: &FxHashMap<String, Rational64>,
    bound: Rational64,
    strict: bool,
) -> ChcExpr {
    if coeffs.is_empty() {
        // Pure constant comparison: 0 ≤ bound or 0 < bound
        let result = if strict {
            Rational64::from_integer(0) < bound
        } else {
            Rational64::from_integer(0) <= bound
        };
        return ChcExpr::Bool(result);
    }

    // Build LHS: Σᵢ aᵢ·xᵢ
    let mut terms: Vec<ChcExpr> = Vec::new();
    let mut sorted_vars: Vec<_> = coeffs.iter().collect();
    sorted_vars.sort_by(|a, b| a.0.cmp(b.0));

    for (var_name, coeff) in sorted_vars {
        let var = ChcVar::new(var_name, ChcSort::Int);
        let var_expr = ChcExpr::var(var);

        let numer = *coeff.numer();
        let denom = *coeff.denom();

        if denom != 1 {
            // Rational coefficient - for now, just multiply by denom and adjust RHS
            // This is a simplification; proper handling would use fractions
            continue;
        }

        if numer == 1 {
            terms.push(var_expr);
        } else if numer == -1 {
            terms.push(ChcExpr::neg(var_expr));
        } else {
            // Handle both positive (> 1) and negative (< -1) cases
            terms.push(ChcExpr::mul(ChcExpr::Int(numer), var_expr));
        }
    }

    let lhs = if terms.len() == 1 {
        terms.pop().unwrap()
    } else {
        ChcExpr::Op(ChcOp::Add, terms.into_iter().map(Arc::new).collect())
    };

    // Build RHS: bound (as integer if possible)
    let numer = *bound.numer();
    let denom = *bound.denom();
    let rhs = if denom == 1 {
        ChcExpr::Int(numer)
    } else {
        // Approximate as integer (floor)
        ChcExpr::Int(numer / denom)
    };

    // Build comparison
    if strict {
        ChcExpr::lt(lhs, rhs)
    } else {
        ChcExpr::le(lhs, rhs)
    }
}

/// Compute a LIA interpolant from transition (A) and bad state (B) constraints.
///
/// When A ∧ B is UNSAT, this computes an interpolant I such that:
/// - A ⊨ I (I is implied by A)
/// - I ∧ B is UNSAT (I is inconsistent with B)
/// - I uses only variables shared between A and B
///
/// This implements the Golem/Spacer approach where interpolants are used
/// as blocking lemmas in PDR instead of heuristic generalization.
///
/// # Arguments
/// * `a_constraints` - Constraints from the transition relation
/// * `b_constraints` - Constraints from the bad state (proof obligation)
/// * `shared_vars` - Variables that appear in both A and B (predicate variables)
///
/// # Returns
/// An interpolant expression if successful, None otherwise.
pub fn compute_interpolant(
    a_constraints: &[ChcExpr],
    b_constraints: &[ChcExpr],
    shared_vars: &FxHashSet<String>,
) -> Option<ChcExpr> {
    // Parse A-constraints as linear inequalities
    let a_linear: Vec<LinearConstraint> = a_constraints
        .iter()
        .filter_map(parse_linear_constraint)
        .collect();

    // Parse B-constraints as linear inequalities
    let b_linear: Vec<LinearConstraint> = b_constraints
        .iter()
        .filter_map(parse_linear_constraint)
        .collect();

    if a_linear.is_empty() || b_linear.is_empty() {
        return None;
    }

    // Collect all variables from both A and B
    let mut a_vars: FxHashSet<String> = FxHashSet::default();
    for c in &a_linear {
        a_vars.extend(c.coeffs.keys().cloned());
    }
    let mut b_vars: FxHashSet<String> = FxHashSet::default();
    for c in &b_linear {
        b_vars.extend(c.coeffs.keys().cloned());
    }

    // Strategy 1: Look for a single A-constraint that contradicts all B-constraints
    // This is the simplest case: find an A-constraint on a shared variable
    // that directly bounds the variable in a way that contradicts B.
    for a_c in &a_linear {
        // Find single-variable bounds in A
        if a_c.coeffs.len() == 1 {
            let (var, coeff) = a_c.coeffs.iter().next().unwrap();
            if !shared_vars.contains(var) {
                continue;
            }

            // Check if this bound contradicts any B-constraint
            let a_bound = if *coeff > Rational64::from_integer(0) {
                // var <= bound/coeff (upper bound)
                Some((var.clone(), a_c.bound / *coeff, true, a_c.strict))
            } else if *coeff < Rational64::from_integer(0) {
                // var >= -bound/coeff (lower bound)
                Some((var.clone(), -a_c.bound / *coeff, false, a_c.strict))
            } else {
                None
            };

            if let Some((var_name, bound, is_upper, strict)) = a_bound {
                for b_c in &b_linear {
                    if b_c.coeffs.len() == 1 {
                        if let Some(b_coeff) = b_c.coeffs.get(&var_name) {
                            let b_bound = if *b_coeff > Rational64::from_integer(0) {
                                Some((b_c.bound / *b_coeff, true))
                            } else if *b_coeff < Rational64::from_integer(0) {
                                Some((-b_c.bound / *b_coeff, false))
                            } else {
                                None
                            };

                            if let Some((b_val, b_is_upper)) = b_bound {
                                // Check for contradiction: A says var <= k, B says var >= m where m > k
                                if is_upper && !b_is_upper && b_val > bound {
                                    // A gives upper bound, B gives lower bound that exceeds it
                                    // Interpolant: var <= bound (from A)
                                    let numer = *bound.numer();
                                    let denom = *bound.denom();
                                    let bound_int = if denom == 1 { numer } else { numer / denom };
                                    let var = ChcVar::new(&var_name, ChcSort::Int);
                                    return Some(if strict {
                                        ChcExpr::lt(ChcExpr::var(var), ChcExpr::Int(bound_int))
                                    } else {
                                        ChcExpr::le(ChcExpr::var(var), ChcExpr::Int(bound_int))
                                    });
                                }
                                if !is_upper && b_is_upper && b_val < bound {
                                    // A gives lower bound, B gives upper bound below it
                                    // Interpolant: var >= bound (from A)
                                    let numer = *bound.numer();
                                    let denom = *bound.denom();
                                    let bound_int = if denom == 1 { numer } else { numer / denom };
                                    let var = ChcVar::new(&var_name, ChcSort::Int);
                                    return Some(if strict {
                                        ChcExpr::gt(ChcExpr::var(var), ChcExpr::Int(bound_int))
                                    } else {
                                        ChcExpr::ge(ChcExpr::var(var), ChcExpr::Int(bound_int))
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Strategy 2: Combine A-constraints that involve shared variables
    // to derive a constraint that contradicts B
    let shared_a: Vec<&LinearConstraint> = a_linear
        .iter()
        .filter(|c| c.coeffs.keys().any(|v| shared_vars.contains(v)))
        .collect();

    if shared_a.len() >= 2 {
        // Try to combine two A-constraints via variable elimination
        for i in 0..shared_a.len() {
            for j in (i + 1)..shared_a.len() {
                let c1 = shared_a[i];
                let c2 = shared_a[j];

                // Find a variable to eliminate
                for var in c1.coeffs.keys() {
                    if !shared_vars.contains(var) {
                        // Variable to eliminate must be in A but we project to shared vars
                        if let Some(coeff2) = c2.coeffs.get(var) {
                            let coeff1 = c1.get_coeff(var);
                            // Opposite signs allow elimination
                            if (coeff1 > Rational64::from_integer(0))
                                != (*coeff2 > Rational64::from_integer(0))
                            {
                                // Combine: w1*c1 + w2*c2 where w1=|coeff2|, w2=|coeff1|
                                let w1 = coeff2.abs();
                                let w2 = coeff1.abs();

                                let mut combined_coeffs: FxHashMap<String, Rational64> =
                                    FxHashMap::default();
                                for (v, c) in &c1.coeffs {
                                    combined_coeffs.insert(v.clone(), *c * w1);
                                }
                                for (v, c) in &c2.coeffs {
                                    let current = combined_coeffs
                                        .get(v)
                                        .copied()
                                        .unwrap_or(Rational64::from_integer(0));
                                    combined_coeffs.insert(v.clone(), current + *c * w2);
                                }
                                let combined_bound = c1.bound * w1 + c2.bound * w2;

                                // Remove eliminated variables and non-shared variables
                                combined_coeffs.retain(|v, c| {
                                    shared_vars.contains(v) && *c != Rational64::from_integer(0)
                                });

                                if !combined_coeffs.is_empty() {
                                    // Build the interpolant
                                    let expr = build_linear_inequality(
                                        &combined_coeffs,
                                        combined_bound,
                                        c1.strict || c2.strict,
                                    );
                                    return Some(expr);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Strategy 3: Use the strongest single A-constraint on shared variables
    // as a fallback interpolant
    for a_c in &a_linear {
        if a_c.coeffs.keys().all(|v| shared_vars.contains(v)) && !a_c.coeffs.is_empty() {
            return Some(build_linear_inequality(&a_c.coeffs, a_c.bound, a_c.strict));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_le() {
        let x = ChcVar::new("x", ChcSort::Int);
        // x <= 5  =>  x - 5 <= 0, so coeff(x)=1, bound=5
        let expr = ChcExpr::le(ChcExpr::var(x), ChcExpr::Int(5));
        let constraint = parse_linear_constraint(&expr).unwrap();

        assert_eq!(constraint.get_coeff("x"), Rational64::from_integer(1));
        // Constraint form: x - 5 <= 0 means bound is stored as the RHS negated from the expr
        // For a <= b, we have a - b <= 0, so bound stores +b's contribution
        assert_eq!(constraint.bound, Rational64::from_integer(5));
        assert!(!constraint.strict);
    }

    #[test]
    fn test_parse_ge_to_le() {
        let x = ChcVar::new("x", ChcSort::Int);
        // x >= 5  =>  5 - x <= 0  =>  -x <= -5
        let expr = ChcExpr::ge(ChcExpr::var(x), ChcExpr::Int(5));
        let constraint = parse_linear_constraint(&expr).unwrap();

        assert_eq!(constraint.get_coeff("x"), Rational64::from_integer(-1));
        // For a >= b, we have b - a <= 0
        assert_eq!(constraint.bound, Rational64::from_integer(-5));
        assert!(!constraint.strict);
    }

    #[test]
    fn test_parse_linear_combination() {
        let x = ChcVar::new("x", ChcSort::Int);
        let y = ChcVar::new("y", ChcSort::Int);
        // x + 2*y <= 10  =>  x + 2y - 10 <= 0
        let expr = ChcExpr::le(
            ChcExpr::add(
                ChcExpr::var(x),
                ChcExpr::mul(ChcExpr::Int(2), ChcExpr::var(y)),
            ),
            ChcExpr::Int(10),
        );
        let constraint = parse_linear_constraint(&expr).unwrap();

        assert_eq!(constraint.get_coeff("x"), Rational64::from_integer(1));
        assert_eq!(constraint.get_coeff("y"), Rational64::from_integer(2));
        // Bound stores the RHS contribution: -(-10) = 10
        assert_eq!(constraint.bound, Rational64::from_integer(10));
    }

    #[test]
    fn test_farkas_combine_opposite_bounds() {
        let x = ChcVar::new("x", ChcSort::Int);
        // x <= 5 and x >= 10 (UNSAT)
        // Combined: 0 <= 5 - 10 = -5 (false)
        let c1 = ChcExpr::le(ChcExpr::var(x.clone()), ChcExpr::Int(5));
        let c2 = ChcExpr::ge(ChcExpr::var(x), ChcExpr::Int(10));

        let result = farkas_combine(&[c1, c2]);
        assert!(result.is_some());
    }

    #[test]
    fn test_farkas_combine_variable_elimination() {
        let x = ChcVar::new("x", ChcSort::Int);
        let y = ChcVar::new("y", ChcSort::Int);
        // x + y <= 5 and -x <= -3 (i.e., x >= 3)
        // Combining eliminates x: y <= 2
        let c1 = ChcExpr::le(
            ChcExpr::add(ChcExpr::var(x.clone()), ChcExpr::var(y.clone())),
            ChcExpr::Int(5),
        );
        let c2 = ChcExpr::ge(ChcExpr::var(x), ChcExpr::Int(3));

        let result = farkas_combine(&[c1, c2]);
        assert!(result.is_some());
        if let Some(fc) = result {
            eprintln!("Combined: {:?}", fc.combined);
        }
    }
}
