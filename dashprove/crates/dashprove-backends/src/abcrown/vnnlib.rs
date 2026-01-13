//! VNNLIB generation and identifier handling

use crate::traits::BackendError;
use dashprove_usl::ast::{BinaryOp, ComparisonOp, Expr, Property};
use dashprove_usl::typecheck::TypedSpec;
use std::collections::{HashMap, HashSet};

/// Convert a variable name to a valid VNNLIB identifier
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

    let lower = clean.to_lowercase();
    if lower.starts_with("input") || lower.starts_with('x') || lower.starts_with("in") {
        if let Some(idx) = extract_index(&clean) {
            return format!("X_{}", idx);
        }
        clean
    } else if lower.starts_with("output") || lower.starts_with('y') || lower.starts_with("out") {
        if let Some(idx) = extract_index(&clean) {
            return format!("Y_{}", idx);
        }
        clean
    } else {
        clean
    }
}

/// Extract numeric index from variable names like x0, input_3, output5
pub fn extract_index(name: &str) -> Option<usize> {
    let digit_start = name.find(|c: char| c.is_ascii_digit())?;
    let num_str: String = name[digit_start..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    num_str.parse().ok()
}

/// Extract neural patterns from USL expressions
///
/// Tracks:
/// - Inputs/outputs encountered
/// - Input bounds from comparisons
/// - Output constraints to assert in VNNLIB
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
            if lower.starts_with('x') || lower.contains("input") || lower.starts_with("in") {
                if let Some(idx) = extract_index(name) {
                    inputs.insert(idx);
                }
            } else if lower.starts_with('y') || lower.contains("output") || lower.starts_with("out")
            {
                if let Some(idx) = extract_index(name) {
                    outputs.insert(idx);
                }
            }
        }
        Expr::Compare(lhs, op, rhs) => {
            if let Expr::Var(var) = lhs.as_ref() {
                let val = match rhs.as_ref() {
                    Expr::Float(f) => Some(*f),
                    Expr::Int(i) => Some(*i as f64),
                    _ => None,
                };
                if let Some(value) = val {
                    let lower = var.to_lowercase();
                    if lower.starts_with('x') || lower.contains("input") || lower.starts_with("in")
                    {
                        if let Some(idx) = extract_index(var) {
                            inputs.insert(idx);
                            match op {
                                ComparisonOp::Ge | ComparisonOp::Gt => {
                                    input_bounds.push((idx, value, f64::INFINITY));
                                }
                                ComparisonOp::Le | ComparisonOp::Lt => {
                                    input_bounds.push((idx, f64::NEG_INFINITY, value));
                                }
                                _ => {}
                            }
                        }
                    } else if lower.starts_with('y')
                        || lower.contains("output")
                        || lower.starts_with("out")
                    {
                        if let Some(idx) = extract_index(var) {
                            outputs.insert(idx);
                            let constraint = compile_comparison_to_smt(lhs, *op, rhs);
                            output_constraints.push(constraint);
                        }
                    }
                }
            }

            if let Expr::Var(var) = rhs.as_ref() {
                let val = match lhs.as_ref() {
                    Expr::Float(f) => Some(*f),
                    Expr::Int(i) => Some(*i as f64),
                    _ => None,
                };
                if let Some(value) = val {
                    let lower = var.to_lowercase();
                    if lower.starts_with('x') || lower.contains("input") || lower.starts_with("in")
                    {
                        if let Some(idx) = extract_index(var) {
                            inputs.insert(idx);
                            match op {
                                ComparisonOp::Le | ComparisonOp::Lt => {
                                    input_bounds.push((idx, value, f64::INFINITY));
                                }
                                ComparisonOp::Ge | ComparisonOp::Gt => {
                                    input_bounds.push((idx, f64::NEG_INFINITY, value));
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }

            extract_neural_patterns(lhs, inputs, outputs, input_bounds, output_constraints);
            extract_neural_patterns(rhs, inputs, outputs, input_bounds, output_constraints);
        }
        Expr::And(lhs, rhs) | Expr::Or(lhs, rhs) | Expr::Implies(lhs, rhs) => {
            extract_neural_patterns(lhs, inputs, outputs, input_bounds, output_constraints);
            extract_neural_patterns(rhs, inputs, outputs, input_bounds, output_constraints);
        }
        Expr::Not(inner) | Expr::Neg(inner) => {
            extract_neural_patterns(inner, inputs, outputs, input_bounds, output_constraints);
        }
        Expr::ForAll { body, .. }
        | Expr::Exists { body, .. }
        | Expr::ForAllIn { body, .. }
        | Expr::ExistsIn { body, .. } => {
            extract_neural_patterns(body, inputs, outputs, input_bounds, output_constraints);
        }
        Expr::App(_func, args) => {
            for arg in args {
                extract_neural_patterns(arg, inputs, outputs, input_bounds, output_constraints);
            }
        }
        Expr::Binary(lhs, _op, rhs) => {
            extract_neural_patterns(lhs, inputs, outputs, input_bounds, output_constraints);
            extract_neural_patterns(rhs, inputs, outputs, input_bounds, output_constraints);
        }
        Expr::FieldAccess(obj, field) => {
            let lower = field.to_lowercase();
            if lower.starts_with('x') || lower.contains("input") || lower.starts_with("in") {
                if let Some(idx) = extract_index(field) {
                    inputs.insert(idx);
                }
            } else if lower.starts_with('y') || lower.contains("output") || lower.starts_with("out")
            {
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
        Expr::Bool(_) | Expr::Float(_) | Expr::Int(_) | Expr::String(_) => {}
    }
}

/// Compile a comparison to SMT-LIB 2
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
    format!("({op_str} {lhs_smt} {rhs_smt})")
}

/// Compile USL expressions to SMT-LIB 2 for VNNLIB
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
                f.to_string()
            }
        }
        Expr::Bool(b) => {
            if *b {
                "true".to_string()
            } else {
                "false".to_string()
            }
        }
        Expr::String(s) => format!("\"{s}\""),
        Expr::And(lhs, rhs) => format!(
            "(and {} {})",
            compile_expr_to_smt(lhs),
            compile_expr_to_smt(rhs)
        ),
        Expr::Or(lhs, rhs) => format!(
            "(or {} {})",
            compile_expr_to_smt(lhs),
            compile_expr_to_smt(rhs)
        ),
        Expr::Not(inner) => format!("(not {})", compile_expr_to_smt(inner)),
        Expr::Implies(lhs, rhs) => format!(
            "(=> {} {})",
            compile_expr_to_smt(lhs),
            compile_expr_to_smt(rhs)
        ),
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
                "({op_str} {} {})",
                compile_expr_to_smt(lhs),
                compile_expr_to_smt(rhs)
            )
        }
        Expr::Neg(inner) => format!("(- {})", compile_expr_to_smt(inner)),
        Expr::App(func, args) => {
            if args.is_empty() {
                func.clone()
            } else {
                let args_smt: Vec<String> = args.iter().map(compile_expr_to_smt).collect();
                format!("({} {})", func, args_smt.join(" "))
            }
        }
        Expr::FieldAccess(obj, field) => {
            format!("({field} {})", compile_expr_to_smt(obj))
        }
        Expr::MethodCall {
            receiver,
            method,
            args,
        } => {
            let mut all_args = vec![compile_expr_to_smt(receiver)];
            all_args.extend(args.iter().map(compile_expr_to_smt));
            format!("({method} {})", all_args.join(" "))
        }
        Expr::ForAll { var, body, .. } | Expr::ForAllIn { var, body, .. } => {
            format!("(forall (({} Real)) {})", var, compile_expr_to_smt(body))
        }
        Expr::Exists { var, body, .. } | Expr::ExistsIn { var, body, .. } => {
            format!("(exists (({} Real)) {})", var, compile_expr_to_smt(body))
        }
    }
}

/// Consolidate multiple bounds for the same variable into tight (min, max) pairs
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

/// Generate VNNLIB property from USL specification
pub fn generate_vnnlib(spec: &TypedSpec) -> Result<String, BackendError> {
    let mut vnnlib = String::new();
    vnnlib.push_str("; Generated by DashProve\n");
    vnnlib.push_str("; USL to VNNLIB compilation\n\n");

    let mut inputs: HashSet<usize> = HashSet::new();
    let mut outputs: HashSet<usize> = HashSet::new();
    let mut input_bounds: Vec<(usize, f64, f64)> = Vec::new();
    let mut output_constraints: Vec<String> = Vec::new();

    for property in &spec.spec.properties {
        vnnlib.push_str(&format!("; Property: {}\n", property.name()));
        match property {
            Property::Theorem(t) => extract_neural_patterns(
                &t.body,
                &mut inputs,
                &mut outputs,
                &mut input_bounds,
                &mut output_constraints,
            ),
            Property::Invariant(inv) => extract_neural_patterns(
                &inv.body,
                &mut inputs,
                &mut outputs,
                &mut input_bounds,
                &mut output_constraints,
            ),
            Property::Contract(contract) => {
                for req in &contract.requires {
                    extract_neural_patterns(
                        req,
                        &mut inputs,
                        &mut outputs,
                        &mut input_bounds,
                        &mut output_constraints,
                    );
                }
                for ens in &contract.ensures {
                    extract_neural_patterns(
                        ens,
                        &mut inputs,
                        &mut outputs,
                        &mut input_bounds,
                        &mut output_constraints,
                    );
                }
            }
            _ => {}
        }
    }

    // Use type definitions to infer IO dimensions
    for typedef in &spec.spec.types {
        for (idx, field) in typedef.fields.iter().enumerate() {
            let lower = field.name.to_lowercase();
            if lower.contains("input") || lower.starts_with('x') || lower.starts_with("in") {
                inputs.insert(idx);
            } else if lower.contains("output") || lower.starts_with('y') || lower.starts_with("out")
            {
                outputs.insert(idx);
            }
        }
    }

    if inputs.is_empty() {
        inputs.insert(0);
    }
    if outputs.is_empty() {
        outputs.insert(0);
    }

    vnnlib.push_str("; Input variables\n");
    let mut sorted_inputs: Vec<_> = inputs.iter().collect();
    sorted_inputs.sort();
    for idx in sorted_inputs {
        vnnlib.push_str(&format!("(declare-const X_{} Real)\n", idx));
    }
    vnnlib.push('\n');

    vnnlib.push_str("; Output variables\n");
    let mut sorted_outputs: Vec<_> = outputs.iter().collect();
    sorted_outputs.sort();
    for idx in sorted_outputs {
        vnnlib.push_str(&format!("(declare-const Y_{} Real)\n", idx));
    }
    vnnlib.push('\n');

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
        vnnlib.push_str("; Default input constraints (normalized)\n");
        for idx in &inputs {
            vnnlib.push_str(&format!("(assert (>= X_{} 0.0))\n", idx));
            vnnlib.push_str(&format!("(assert (<= X_{} 1.0))\n", idx));
        }
        vnnlib.push('\n');
    }

    if !output_constraints.is_empty() {
        vnnlib.push_str("; Output constraints (from property)\n");
        for constraint in &output_constraints {
            vnnlib.push_str(&format!("(assert {})\n", constraint));
        }
    } else {
        vnnlib.push_str("; Output constraints (default safety)\n");
        for idx in &outputs {
            vnnlib.push_str(&format!("(assert (< Y_{} 0.0))\n", idx));
        }
    }

    Ok(vnnlib)
}

/// Generate YAML configuration for alpha-beta-CROWN
pub fn generate_config(
    config: &super::AbCrownConfig,
    model_path: &str,
    property_path: &str,
) -> String {
    let mut yaml = String::new();
    yaml.push_str("# Generated by DashProve\n");
    yaml.push_str("general:\n");
    yaml.push_str(&format!(
        "  device: {}\n",
        if config.use_gpu { "cuda" } else { "cpu" }
    ));
    yaml.push_str("  seed: 100\n\n");

    yaml.push_str("model:\n");
    yaml.push_str(&format!("  path: \"{}\"\n\n", model_path));

    yaml.push_str("specification:\n");
    yaml.push_str(&format!("  vnnlib_path: \"{}\"\n\n", property_path));

    yaml.push_str("solver:\n");
    yaml.push_str("  batch_size: ");
    if let Some(batch) = config.batch_size {
        yaml.push_str(&format!("{}\n", batch));
    } else {
        yaml.push_str("1\n");
    }

    yaml.push_str("  alpha-crown:\n");
    yaml.push_str("    iteration: 100\n");
    yaml.push_str("  beta-crown:\n");
    yaml.push_str("    iteration: 20\n");

    yaml
}
