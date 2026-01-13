//! USL expression analysis for ERAN
//!
//! Extracts epsilon values, neural network dimensions, and input bounds
//! from USL expressions for compilation to ERAN verification parameters.

use dashprove_usl::ast::{ComparisonOp, Expr, Property};
use dashprove_usl::typecheck::TypedSpec;
use std::collections::{HashMap, HashSet};

/// Extract numeric index from a variable name like "x1", "input_2", "output0"
pub fn extract_index(name: &str) -> Option<usize> {
    let digit_start = name.find(|c: char| c.is_ascii_digit())?;
    let num_str: String = name[digit_start..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    num_str.parse().ok()
}

/// Extract epsilon (perturbation bound) from USL expression
///
/// Looks for patterns like:
/// - `epsilon <= 0.01` or `0.01 >= epsilon`
/// - `|x - x0| <= eps` patterns
/// - Explicit epsilon parameter in robustness functions
pub fn extract_epsilon(expr: &Expr) -> Option<f64> {
    match expr {
        // Direct comparison: epsilon <= 0.01
        Expr::Compare(lhs, op, rhs) => {
            // Check if left side is epsilon variable
            if let Expr::Var(name) = lhs.as_ref() {
                let lower = name.to_lowercase();
                if (lower.contains("epsilon") || lower == "eps")
                    && matches!(op, ComparisonOp::Le | ComparisonOp::Lt)
                {
                    return extract_numeric_value(rhs);
                }
            }
            // Check if right side is epsilon variable (reversed comparison)
            if let Expr::Var(name) = rhs.as_ref() {
                let lower = name.to_lowercase();
                if (lower.contains("epsilon") || lower == "eps")
                    && matches!(op, ComparisonOp::Ge | ComparisonOp::Gt)
                {
                    return extract_numeric_value(lhs);
                }
            }
            // Check for abs comparison: |x - x0| <= eps
            if let Expr::App(func, args) = lhs.as_ref() {
                if func.to_lowercase() == "abs" && matches!(op, ComparisonOp::Le | ComparisonOp::Lt)
                {
                    return extract_numeric_value(rhs);
                }
                // Recurse into function arguments
                for arg in args {
                    if let Some(eps) = extract_epsilon(arg) {
                        return Some(eps);
                    }
                }
            }
            // Recurse into both sides
            extract_epsilon(lhs).or_else(|| extract_epsilon(rhs))
        }
        // Function application: robustness(x, eps)
        Expr::App(func, args) => {
            let lower = func.to_lowercase();
            if lower.contains("robust") || lower.contains("perturb") {
                // Look for numeric argument that could be epsilon
                for arg in args {
                    if let Some(val) = extract_numeric_value(arg) {
                        if val > 0.0 && val < 1.0 {
                            return Some(val);
                        }
                    }
                }
            }
            // Recurse into arguments
            for arg in args {
                if let Some(eps) = extract_epsilon(arg) {
                    return Some(eps);
                }
            }
            None
        }
        // Logical operators - recurse
        Expr::And(lhs, rhs) | Expr::Or(lhs, rhs) | Expr::Implies(lhs, rhs) => {
            extract_epsilon(lhs).or_else(|| extract_epsilon(rhs))
        }
        Expr::Not(inner) | Expr::Neg(inner) => extract_epsilon(inner),
        Expr::Binary(lhs, _, rhs) => extract_epsilon(lhs).or_else(|| extract_epsilon(rhs)),
        Expr::ForAll { body, .. }
        | Expr::Exists { body, .. }
        | Expr::ForAllIn { body, .. }
        | Expr::ExistsIn { body, .. } => extract_epsilon(body),
        Expr::FieldAccess(obj, _) => extract_epsilon(obj),
        Expr::MethodCall { receiver, args, .. } => {
            extract_epsilon(receiver).or_else(|| args.iter().find_map(extract_epsilon))
        }
        // Literals don't contain epsilon
        Expr::Var(_) | Expr::Int(_) | Expr::Float(_) | Expr::String(_) | Expr::Bool(_) => None,
    }
}

/// Extract numeric value from expression
pub fn extract_numeric_value(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::Float(f) => Some(*f),
        Expr::Int(i) => Some(*i as f64),
        Expr::Neg(inner) => extract_numeric_value(inner).map(|v| -v),
        _ => None,
    }
}

/// Extract neural network input/output dimensions from USL expressions
///
/// Identifies input and output variables based on naming conventions:
/// - Inputs: x0, x1, input_0, in0, etc.
/// - Outputs: y0, y1, output_0, out0, etc.
pub fn extract_neural_dimensions(
    expr: &Expr,
    inputs: &mut HashSet<usize>,
    outputs: &mut HashSet<usize>,
) {
    match expr {
        Expr::Var(name) => {
            let lower = name.to_lowercase();
            if lower.starts_with("x") || lower.contains("input") || lower.starts_with("in") {
                if let Some(idx) = extract_index(name) {
                    inputs.insert(idx);
                }
            } else if lower.starts_with("y") || lower.contains("output") || lower.starts_with("out")
            {
                if let Some(idx) = extract_index(name) {
                    outputs.insert(idx);
                }
            }
        }
        Expr::Compare(lhs, _, rhs) => {
            extract_neural_dimensions(lhs, inputs, outputs);
            extract_neural_dimensions(rhs, inputs, outputs);
        }
        Expr::And(lhs, rhs) | Expr::Or(lhs, rhs) | Expr::Implies(lhs, rhs) => {
            extract_neural_dimensions(lhs, inputs, outputs);
            extract_neural_dimensions(rhs, inputs, outputs);
        }
        Expr::Not(inner) | Expr::Neg(inner) => {
            extract_neural_dimensions(inner, inputs, outputs);
        }
        Expr::Binary(lhs, _, rhs) => {
            extract_neural_dimensions(lhs, inputs, outputs);
            extract_neural_dimensions(rhs, inputs, outputs);
        }
        Expr::ForAll { body, .. }
        | Expr::Exists { body, .. }
        | Expr::ForAllIn { body, .. }
        | Expr::ExistsIn { body, .. } => {
            extract_neural_dimensions(body, inputs, outputs);
        }
        Expr::App(_, args) => {
            for arg in args {
                extract_neural_dimensions(arg, inputs, outputs);
            }
        }
        Expr::FieldAccess(obj, field) => {
            let lower = field.to_lowercase();
            if lower.starts_with("x") || lower.contains("input") {
                if let Some(idx) = extract_index(field) {
                    inputs.insert(idx);
                }
            } else if lower.starts_with("y") || lower.contains("output") {
                if let Some(idx) = extract_index(field) {
                    outputs.insert(idx);
                }
            }
            extract_neural_dimensions(obj, inputs, outputs);
        }
        Expr::MethodCall { receiver, args, .. } => {
            extract_neural_dimensions(receiver, inputs, outputs);
            for arg in args {
                extract_neural_dimensions(arg, inputs, outputs);
            }
        }
        // Literals don't contribute dimensions
        Expr::Int(_) | Expr::Float(_) | Expr::String(_) | Expr::Bool(_) => {}
    }
}

/// Extract input bounds from USL expressions
///
/// Returns a vector of (index, lower_bound, upper_bound) tuples
pub fn extract_input_bounds(expr: &Expr) -> Vec<(usize, f64, f64)> {
    let mut bounds = Vec::new();
    extract_bounds_recursive(expr, &mut bounds);
    consolidate_bounds(&bounds)
}

/// Recursively extract bounds from expression
fn extract_bounds_recursive(expr: &Expr, bounds: &mut Vec<(usize, f64, f64)>) {
    match expr {
        Expr::Compare(lhs, op, rhs) => {
            // Check for Var on left side: x0 >= 0, x0 <= 1
            if let Expr::Var(var) = lhs.as_ref() {
                let lower = var.to_lowercase();
                if lower.starts_with("x") || lower.contains("input") {
                    if let Some(idx) = extract_index(var) {
                        if let Some(val) = extract_numeric_value(rhs) {
                            match op {
                                ComparisonOp::Ge | ComparisonOp::Gt => {
                                    bounds.push((idx, val, f64::INFINITY));
                                }
                                ComparisonOp::Le | ComparisonOp::Lt => {
                                    bounds.push((idx, f64::NEG_INFINITY, val));
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            // Check for Var on right side: 0 <= x0, 1 >= x0
            if let Expr::Var(var) = rhs.as_ref() {
                let lower = var.to_lowercase();
                if lower.starts_with("x") || lower.contains("input") {
                    if let Some(idx) = extract_index(var) {
                        if let Some(val) = extract_numeric_value(lhs) {
                            match op {
                                ComparisonOp::Le | ComparisonOp::Lt => {
                                    bounds.push((idx, val, f64::INFINITY));
                                }
                                ComparisonOp::Ge | ComparisonOp::Gt => {
                                    bounds.push((idx, f64::NEG_INFINITY, val));
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            // Recurse
            extract_bounds_recursive(lhs, bounds);
            extract_bounds_recursive(rhs, bounds);
        }
        Expr::And(lhs, rhs) | Expr::Or(lhs, rhs) | Expr::Implies(lhs, rhs) => {
            extract_bounds_recursive(lhs, bounds);
            extract_bounds_recursive(rhs, bounds);
        }
        Expr::Not(inner) | Expr::Neg(inner) => {
            extract_bounds_recursive(inner, bounds);
        }
        Expr::Binary(lhs, _, rhs) => {
            extract_bounds_recursive(lhs, bounds);
            extract_bounds_recursive(rhs, bounds);
        }
        Expr::ForAll { body, .. }
        | Expr::Exists { body, .. }
        | Expr::ForAllIn { body, .. }
        | Expr::ExistsIn { body, .. } => {
            extract_bounds_recursive(body, bounds);
        }
        Expr::App(_, args) => {
            for arg in args {
                extract_bounds_recursive(arg, bounds);
            }
        }
        Expr::FieldAccess(obj, _) => {
            extract_bounds_recursive(obj, bounds);
        }
        Expr::MethodCall { receiver, args, .. } => {
            extract_bounds_recursive(receiver, bounds);
            for arg in args {
                extract_bounds_recursive(arg, bounds);
            }
        }
        // Literals don't contribute bounds
        Expr::Var(_) | Expr::Int(_) | Expr::Float(_) | Expr::String(_) | Expr::Bool(_) => {}
    }
}

/// Consolidate multiple bounds for the same variable into (min, max) pairs
pub fn consolidate_bounds(bounds: &[(usize, f64, f64)]) -> Vec<(usize, f64, f64)> {
    let mut consolidated: HashMap<usize, (f64, f64)> = HashMap::new();

    for (idx, lower, upper) in bounds {
        let entry = consolidated
            .entry(*idx)
            .or_insert((f64::NEG_INFINITY, f64::INFINITY));
        if lower.is_finite() {
            entry.0 = entry.0.max(*lower);
        }
        if upper.is_finite() {
            entry.1 = entry.1.min(*upper);
        }
    }

    let mut result: Vec<_> = consolidated
        .into_iter()
        .map(|(idx, (lower, upper))| {
            let l = if lower.is_finite() { lower } else { 0.0 };
            let u = if upper.is_finite() { upper } else { 1.0 };
            (idx, l, u)
        })
        .collect();
    result.sort_by_key(|(idx, _, _)| *idx);
    result
}

/// Extract epsilon from spec properties
pub fn extract_epsilon_from_spec(spec: &TypedSpec, default_epsilon: f64) -> f64 {
    for prop in &spec.spec.properties {
        let expr = match prop {
            Property::Invariant(inv) => Some(&inv.body),
            Property::Theorem(thm) => Some(&thm.body),
            Property::Security(sec) => Some(&sec.body),
            Property::Probabilistic(prob) => Some(&prob.condition),
            _ => None,
        };
        if let Some(e) = expr {
            if let Some(eps) = extract_epsilon(e) {
                return eps;
            }
        }
    }
    // Return default epsilon
    default_epsilon
}

/// Determine if zonotope spec is needed
///
/// Zonotope specs are useful when we have specific input bounds
pub fn needs_zonotope_spec(spec: &TypedSpec) -> bool {
    for prop in &spec.spec.properties {
        let expr = match prop {
            Property::Invariant(inv) => Some(&inv.body),
            Property::Theorem(thm) => Some(&thm.body),
            Property::Security(sec) => Some(&sec.body),
            Property::Probabilistic(prob) => Some(&prob.condition),
            _ => None,
        };
        if let Some(e) = expr {
            let bounds = extract_input_bounds(e);
            if !bounds.is_empty() {
                return true;
            }
        }
    }
    false
}
