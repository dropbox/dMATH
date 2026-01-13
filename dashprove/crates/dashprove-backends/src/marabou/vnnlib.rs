//! VNNLIB generation and SMT-LIB 2 compilation
//!
//! Compiles USL neural network properties to VNNLIB format for Marabou.
//! VNNLIB is SMT-LIB 2 based format for neural network verification.

use crate::traits::BackendError;
use dashprove_usl::ast::{BinaryOp, ComparisonOp, Expr, Property};
use dashprove_usl::typecheck::TypedSpec;
use std::collections::{HashMap, HashSet};

/// Convert a variable name to a valid VNNLIB identifier
///
/// VNNLIB uses X_i for inputs and Y_i for outputs.
pub fn to_vnnlib_ident(name: &str) -> String {
    let clean = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect::<String>();

    // Convert common patterns to VNNLIB conventions
    let lower = clean.to_lowercase();
    if lower.starts_with("input") || lower.starts_with("x") || lower.starts_with("in") {
        // Try to extract index from name like "input_0", "x1", "in0"
        if let Some(idx) = extract_index(&clean) {
            return format!("X_{}", idx);
        }
        clean
    } else if lower.starts_with("output") || lower.starts_with("y") || lower.starts_with("out") {
        if let Some(idx) = extract_index(&clean) {
            return format!("Y_{}", idx);
        }
        clean
    } else {
        clean
    }
}

/// Extract numeric index from a variable name like "x1", "input_2", "output0"
pub fn extract_index(name: &str) -> Option<usize> {
    // Find where digits start
    let digit_start = name.find(|c: char| c.is_ascii_digit())?;
    let num_str: String = name[digit_start..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    num_str.parse().ok()
}

/// Extract neural network patterns from USL expression
///
/// Identifies:
/// - Input variables and their bounds
/// - Output variables and constraints
/// - Robustness epsilon bounds
pub fn extract_neural_patterns(
    expr: &Expr,
    inputs: &mut HashSet<usize>,
    outputs: &mut HashSet<usize>,
    input_bounds: &mut Vec<(usize, f64, f64)>,
    output_constraints: &mut Vec<String>,
) {
    match expr {
        Expr::Var(name) => {
            let lower = name.to_lowercase();
            if lower.starts_with("x") || lower.contains("input") {
                if let Some(idx) = extract_index(name) {
                    inputs.insert(idx);
                }
            } else if lower.starts_with("y") || lower.contains("output") {
                if let Some(idx) = extract_index(name) {
                    outputs.insert(idx);
                }
            }
        }
        Expr::Compare(lhs, op, rhs) => {
            // Look for input bounds: x >= 0, x <= 1, etc.
            // Check for Var on left side
            if let Expr::Var(var) = lhs.as_ref() {
                let val_f64 = match rhs.as_ref() {
                    Expr::Float(f) => Some(*f),
                    Expr::Int(i) => Some(*i as f64),
                    _ => None,
                };
                if let Some(val_f64) = val_f64 {
                    let lower = var.to_lowercase();
                    if lower.starts_with("x") || lower.contains("input") {
                        if let Some(idx) = extract_index(var) {
                            inputs.insert(idx);
                            match op {
                                ComparisonOp::Ge | ComparisonOp::Gt => {
                                    // x >= val means lower bound
                                    input_bounds.push((idx, val_f64, f64::INFINITY));
                                }
                                ComparisonOp::Le | ComparisonOp::Lt => {
                                    // x <= val means upper bound
                                    input_bounds.push((idx, f64::NEG_INFINITY, val_f64));
                                }
                                _ => {}
                            }
                        }
                    } else if lower.starts_with("y") || lower.contains("output") {
                        if let Some(idx) = extract_index(var) {
                            outputs.insert(idx);
                            // Output constraints are compiled to assertions
                            let constraint = compile_comparison_to_smt(lhs, *op, rhs);
                            output_constraints.push(constraint);
                        }
                    }
                }
            }
            // Also handle reversed comparisons: 0 <= x (value on left, var on right)
            if let Expr::Var(var) = rhs.as_ref() {
                let val_f64 = match lhs.as_ref() {
                    Expr::Float(f) => Some(*f),
                    Expr::Int(i) => Some(*i as f64),
                    _ => None,
                };
                if let Some(val_f64) = val_f64 {
                    let lower = var.to_lowercase();
                    if lower.starts_with("x") || lower.contains("input") {
                        if let Some(idx) = extract_index(var) {
                            inputs.insert(idx);
                            match op {
                                ComparisonOp::Le | ComparisonOp::Lt => {
                                    // val <= x means lower bound
                                    input_bounds.push((idx, val_f64, f64::INFINITY));
                                }
                                ComparisonOp::Ge | ComparisonOp::Gt => {
                                    // val >= x means upper bound
                                    input_bounds.push((idx, f64::NEG_INFINITY, val_f64));
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            // Recurse into comparison operands
            extract_neural_patterns(lhs, inputs, outputs, input_bounds, output_constraints);
            extract_neural_patterns(rhs, inputs, outputs, input_bounds, output_constraints);
        }
        Expr::And(lhs, rhs) | Expr::Or(lhs, rhs) | Expr::Implies(lhs, rhs) => {
            extract_neural_patterns(lhs, inputs, outputs, input_bounds, output_constraints);
            extract_neural_patterns(rhs, inputs, outputs, input_bounds, output_constraints);
        }
        Expr::Not(inner) => {
            extract_neural_patterns(inner, inputs, outputs, input_bounds, output_constraints);
        }
        Expr::ForAll { body, .. }
        | Expr::Exists { body, .. }
        | Expr::ForAllIn { body, .. }
        | Expr::ExistsIn { body, .. } => {
            extract_neural_patterns(body, inputs, outputs, input_bounds, output_constraints);
        }
        Expr::App(func, args) => {
            // Look for functions like abs(), robustness(), etc.
            let lower = func.to_lowercase();
            if lower == "abs" && args.len() == 1 {
                // Could be robustness property: abs(x - x0) <= epsilon
                extract_neural_patterns(
                    &args[0],
                    inputs,
                    outputs,
                    input_bounds,
                    output_constraints,
                );
            } else if lower.contains("robust") || lower.contains("perturb") {
                // Robustness function - extract its arguments
                for arg in args {
                    extract_neural_patterns(arg, inputs, outputs, input_bounds, output_constraints);
                }
            }
            // Recurse into all arguments
            for arg in args {
                extract_neural_patterns(arg, inputs, outputs, input_bounds, output_constraints);
            }
        }
        Expr::Binary(lhs, _op, rhs) => {
            extract_neural_patterns(lhs, inputs, outputs, input_bounds, output_constraints);
            extract_neural_patterns(rhs, inputs, outputs, input_bounds, output_constraints);
        }
        Expr::Neg(inner) => {
            extract_neural_patterns(inner, inputs, outputs, input_bounds, output_constraints);
        }
        Expr::FieldAccess(obj, field) => {
            // Field names like "input[0]" or "output[1]"
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
            extract_neural_patterns(obj, inputs, outputs, input_bounds, output_constraints);
        }
        Expr::MethodCall { receiver, args, .. } => {
            extract_neural_patterns(receiver, inputs, outputs, input_bounds, output_constraints);
            for arg in args {
                extract_neural_patterns(arg, inputs, outputs, input_bounds, output_constraints);
            }
        }
        // Literals don't contribute variables
        Expr::Int(_) | Expr::Float(_) | Expr::String(_) | Expr::Bool(_) => {}
    }
}

/// Compile a comparison expression to SMT-LIB 2 format
pub fn compile_comparison_to_smt(lhs: &Expr, op: ComparisonOp, rhs: &Expr) -> String {
    let lhs_smt = compile_expr_to_smt(lhs);
    let rhs_smt = compile_expr_to_smt(rhs);
    let op_str = match op {
        ComparisonOp::Eq => "=",
        ComparisonOp::Ne => "distinct",
        ComparisonOp::Lt => "<",
        ComparisonOp::Le => "<=",
        ComparisonOp::Gt => ">",
        ComparisonOp::Ge => ">=",
    };
    format!("({} {} {})", op_str, lhs_smt, rhs_smt)
}

/// Compile a USL expression to SMT-LIB 2 format
pub fn compile_expr_to_smt(expr: &Expr) -> String {
    match expr {
        Expr::Var(name) => to_vnnlib_ident(name),
        Expr::Int(i) => {
            if *i < 0 {
                format!("(- {})", -i)
            } else {
                format!("{}.0", i)
            }
        }
        Expr::Float(f) => {
            if *f < 0.0 {
                format!("(- {})", -f)
            } else {
                format!("{}", f)
            }
        }
        Expr::Bool(b) => if *b { "true" } else { "false" }.to_string(),
        Expr::String(s) => format!("\"{}\"", s),
        Expr::And(lhs, rhs) => {
            format!(
                "(and {} {})",
                compile_expr_to_smt(lhs),
                compile_expr_to_smt(rhs)
            )
        }
        Expr::Or(lhs, rhs) => {
            format!(
                "(or {} {})",
                compile_expr_to_smt(lhs),
                compile_expr_to_smt(rhs)
            )
        }
        Expr::Not(inner) => {
            format!("(not {})", compile_expr_to_smt(inner))
        }
        Expr::Implies(lhs, rhs) => {
            format!(
                "(=> {} {})",
                compile_expr_to_smt(lhs),
                compile_expr_to_smt(rhs)
            )
        }
        Expr::Compare(lhs, op, rhs) => compile_comparison_to_smt(lhs, *op, rhs),
        Expr::Binary(lhs, op, rhs) => {
            let op_str = match op {
                BinaryOp::Add => "+",
                BinaryOp::Sub => "-",
                BinaryOp::Mul => "*",
                BinaryOp::Div => "/",
                BinaryOp::Mod => "mod",
            };
            format!(
                "({} {} {})",
                op_str,
                compile_expr_to_smt(lhs),
                compile_expr_to_smt(rhs)
            )
        }
        Expr::Neg(inner) => {
            format!("(- {})", compile_expr_to_smt(inner))
        }
        Expr::App(func, args) => {
            if args.is_empty() {
                func.clone()
            } else {
                let args_smt: Vec<String> = args.iter().map(compile_expr_to_smt).collect();
                format!("({} {})", func, args_smt.join(" "))
            }
        }
        Expr::FieldAccess(obj, field) => {
            format!("({} {})", field, compile_expr_to_smt(obj))
        }
        Expr::MethodCall {
            receiver,
            method,
            args,
        } => {
            let mut all_args = vec![compile_expr_to_smt(receiver)];
            all_args.extend(args.iter().map(compile_expr_to_smt));
            format!("({} {})", method, all_args.join(" "))
        }
        Expr::ForAll { var, body, .. } => {
            format!("(forall (({} Real)) {})", var, compile_expr_to_smt(body))
        }
        Expr::Exists { var, body, .. } => {
            format!("(exists (({} Real)) {})", var, compile_expr_to_smt(body))
        }
        // For set-based quantifiers, treat as regular quantifiers
        Expr::ForAllIn { var, body, .. } => {
            format!("(forall (({} Real)) {})", var, compile_expr_to_smt(body))
        }
        Expr::ExistsIn { var, body, .. } => {
            format!("(exists (({} Real)) {})", var, compile_expr_to_smt(body))
        }
    }
}

/// Consolidate input bounds into (min, max) pairs
pub fn consolidate_bounds(bounds: &[(usize, f64, f64)]) -> HashMap<usize, (f64, f64)> {
    let mut result: HashMap<usize, (f64, f64)> = HashMap::new();
    for (idx, low, high) in bounds {
        let entry = result
            .entry(*idx)
            .or_insert((f64::NEG_INFINITY, f64::INFINITY));
        if *low > entry.0 {
            entry.0 = *low;
        }
        if *high < entry.1 {
            entry.1 = *high;
        }
    }
    result
}

/// Generate VNNLIB property from USL spec
///
/// Compiles USL neural network properties to VNNLIB format.
/// VNNLIB is SMT-LIB 2 based format for neural network verification.
pub fn generate_vnnlib(spec: &TypedSpec) -> Result<String, BackendError> {
    let mut vnnlib = String::new();
    vnnlib.push_str("; Generated by DashProve\n");
    vnnlib.push_str("; USL to VNNLIB compilation\n\n");

    // Collect all inputs, outputs, and constraints from properties
    let mut inputs: HashSet<usize> = HashSet::new();
    let mut outputs: HashSet<usize> = HashSet::new();
    let mut input_bounds: Vec<(usize, f64, f64)> = Vec::new();
    let mut output_constraints: Vec<String> = Vec::new();

    // Extract patterns from all properties
    for property in &spec.spec.properties {
        vnnlib.push_str(&format!("; Property: {}\n", property.name()));

        match property {
            Property::Theorem(t) => {
                extract_neural_patterns(
                    &t.body,
                    &mut inputs,
                    &mut outputs,
                    &mut input_bounds,
                    &mut output_constraints,
                );
            }
            Property::Invariant(inv) => {
                extract_neural_patterns(
                    &inv.body,
                    &mut inputs,
                    &mut outputs,
                    &mut input_bounds,
                    &mut output_constraints,
                );
            }
            Property::Contract(c) => {
                for req in &c.requires {
                    extract_neural_patterns(
                        req,
                        &mut inputs,
                        &mut outputs,
                        &mut input_bounds,
                        &mut output_constraints,
                    );
                }
                for ens in &c.ensures {
                    extract_neural_patterns(
                        ens,
                        &mut inputs,
                        &mut outputs,
                        &mut input_bounds,
                        &mut output_constraints,
                    );
                }
            }
            _ => {
                // Other property types not directly applicable to neural networks
            }
        }
    }

    // Also extract from type definitions
    for typedef in &spec.spec.types {
        for (idx, field) in typedef.fields.iter().enumerate() {
            let lower = field.name.to_lowercase();
            if lower.contains("input") || lower.starts_with("x") {
                inputs.insert(idx);
            } else if lower.contains("output") || lower.starts_with("y") {
                outputs.insert(idx);
            }
        }
    }

    // Ensure we have at least one input and output
    if inputs.is_empty() {
        inputs.insert(0);
    }
    if outputs.is_empty() {
        outputs.insert(0);
    }

    // Declare input variables
    vnnlib.push_str("; Input variables\n");
    let mut sorted_inputs: Vec<_> = inputs.iter().collect();
    sorted_inputs.sort();
    for idx in sorted_inputs {
        vnnlib.push_str(&format!("(declare-const X_{} Real)\n", idx));
    }
    vnnlib.push('\n');

    // Declare output variables
    vnnlib.push_str("; Output variables\n");
    let mut sorted_outputs: Vec<_> = outputs.iter().collect();
    sorted_outputs.sort();
    for idx in sorted_outputs {
        vnnlib.push_str(&format!("(declare-const Y_{} Real)\n", idx));
    }
    vnnlib.push('\n');

    // Generate input bounds
    let consolidated = consolidate_bounds(&input_bounds);
    if !consolidated.is_empty() {
        vnnlib.push_str("; Input constraints\n");
        let mut sorted_bounds: Vec<_> = consolidated.iter().collect();
        sorted_bounds.sort_by_key(|(idx, _)| *idx);
        for (idx, (low, high)) in sorted_bounds {
            if low.is_finite() {
                vnnlib.push_str(&format!("(assert (>= X_{} {}))\n", idx, low));
            }
            if high.is_finite() {
                vnnlib.push_str(&format!("(assert (<= X_{} {}))\n", idx, high));
            }
        }
        vnnlib.push('\n');
    } else {
        // Default bounds if none specified
        vnnlib.push_str("; Default input constraints (normalized inputs)\n");
        for idx in &inputs {
            vnnlib.push_str(&format!("(assert (>= X_{} 0.0))\n", idx));
            vnnlib.push_str(&format!("(assert (<= X_{} 1.0))\n", idx));
        }
        vnnlib.push('\n');
    }

    // Generate output constraints
    if !output_constraints.is_empty() {
        vnnlib.push_str("; Output constraints (from property)\n");
        for constraint in &output_constraints {
            // Note: VNNLIB searches for counterexamples, so we assert the negation
            // of what we want to prove. The constraint is already in SMT format.
            vnnlib.push_str(&format!("(assert {})\n", constraint));
        }
    } else {
        // If no explicit constraints, generate a default safety property
        vnnlib.push_str("; Output constraints (default safety check)\n");
        for idx in &outputs {
            // Default: output should be non-negative (typical for classification)
            vnnlib.push_str(&format!("(assert (< Y_{} 0.0))\n", idx));
        }
    }

    Ok(vnnlib)
}
