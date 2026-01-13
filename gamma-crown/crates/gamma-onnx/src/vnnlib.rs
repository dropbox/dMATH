//! VNN-LIB property specification format parser.
//!
//! VNN-LIB is the standard property specification language for VNN-COMP
//! (Verification of Neural Networks Competition). It uses SMT-LIB v2 syntax
//! to specify input constraints and output properties.
//!
//! # Format Specification
//!
//! - Variable declarations: `(declare-const X_0 Real)` for inputs, `(declare-const Y_0 Real)` for outputs
//! - Input bounds: `(assert (<= X_0 upper))` and `(assert (>= X_0 lower))`
//! - Output constraints: `(assert (<= Y_0 Y_1))` means property holds if Y_0 ≤ Y_1
//!
//! # Safety Properties
//!
//! VNN-LIB specifies **unsafe regions** - the property is verified (safe) if
//! the neural network output CANNOT satisfy the output constraints for any
//! input in the specified region.
//!
//! # Usage
//!
//! ```ignore
//! use gamma_onnx::vnnlib::{load_vnnlib, VnnLibSpec};
//!
//! let spec = load_vnnlib("property.vnnlib")?;
//! println!("Inputs: {}, Outputs: {}", spec.num_inputs, spec.num_outputs);
//! for (i, (lower, upper)) in spec.input_bounds.iter().enumerate() {
//!     println!("  X_{}: [{}, {}]", i, lower, upper);
//! }
//! ```

use gamma_core::{GammaError, Result};
use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, info, warn};

/// A single output constraint (relational property).
#[derive(Debug, Clone, PartialEq)]
pub enum OutputConstraint {
    /// Y_i <= Y_j
    LessEq(usize, usize),
    /// Y_i >= Y_j (equivalent to Y_j <= Y_i)
    GreaterEq(usize, usize),
    /// Y_i < Y_j
    LessThan(usize, usize),
    /// Y_i > Y_j
    GreaterThan(usize, usize),
    /// Y_i <= constant
    LessEqConst(usize, f64),
    /// Y_i >= constant
    GreaterEqConst(usize, f64),
    /// Y_i < constant
    LessThanConst(usize, f64),
    /// Y_i > constant
    GreaterThanConst(usize, f64),
}

/// A parsed VNN-LIB specification.
#[derive(Debug, Clone)]
pub struct VnnLibSpec {
    /// Number of input variables (X_0, X_1, ..., X_{n-1}).
    pub num_inputs: usize,
    /// Number of output variables (Y_0, Y_1, ..., Y_{m-1}).
    pub num_outputs: usize,
    /// Input bounds as (lower, upper) for each X_i.
    pub input_bounds: Vec<(f64, f64)>,
    /// Output constraints.
    pub output_constraints: Vec<OutputConstraint>,
    /// Whether output constraints form a disjunction (OR) at the top level.
    /// If true, unsafe region is (C1 OR C2 OR ...), so SAFE requires ALL violated.
    /// If false, unsafe region is (C1 AND C2 AND ...), so SAFE requires ANY violated.
    pub is_disjunction: bool,
}

impl VnnLibSpec {
    /// Create a new empty VNN-LIB specification.
    pub fn new() -> Self {
        Self {
            num_inputs: 0,
            num_outputs: 0,
            input_bounds: Vec::new(),
            output_constraints: Vec::new(),
            is_disjunction: false,
        }
    }

    /// Check if the specification has valid input bounds.
    pub fn has_valid_bounds(&self) -> bool {
        self.input_bounds
            .iter()
            .all(|(lower, upper)| lower <= upper)
    }

    /// Get input bounds as separate lower/upper vectors.
    pub fn get_input_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        let lower: Vec<f64> = self.input_bounds.iter().map(|(l, _)| *l).collect();
        let upper: Vec<f64> = self.input_bounds.iter().map(|(_, u)| *u).collect();
        (lower, upper)
    }

    /// Get input bounds as f32 vectors.
    pub fn get_input_bounds_f32(&self) -> (Vec<f32>, Vec<f32>) {
        let lower: Vec<f32> = self.input_bounds.iter().map(|(l, _)| *l as f32).collect();
        let upper: Vec<f32> = self.input_bounds.iter().map(|(_, u)| *u as f32).collect();
        (lower, upper)
    }

    /// Check if output constraints are satisfied by given output values.
    ///
    /// Returns true if the output satisfies ALL constraints (i.e., in unsafe region).
    pub fn check_unsafe(&self, outputs: &[f64]) -> bool {
        self.output_constraints.iter().all(|c| match c {
            OutputConstraint::LessEq(i, j) => outputs[*i] <= outputs[*j],
            OutputConstraint::GreaterEq(i, j) => outputs[*i] >= outputs[*j],
            OutputConstraint::LessThan(i, j) => outputs[*i] < outputs[*j],
            OutputConstraint::GreaterThan(i, j) => outputs[*i] > outputs[*j],
            OutputConstraint::LessEqConst(i, c) => outputs[*i] <= *c,
            OutputConstraint::GreaterEqConst(i, c) => outputs[*i] >= *c,
            OutputConstraint::LessThanConst(i, c) => outputs[*i] < *c,
            OutputConstraint::GreaterThanConst(i, c) => outputs[*i] > *c,
        })
    }

    /// Describe the property in human-readable form.
    pub fn describe(&self) -> String {
        let mut desc = format!(
            "VNN-LIB Property: {} inputs, {} outputs\n",
            self.num_inputs, self.num_outputs
        );

        desc.push_str("Input bounds:\n");
        for (i, (lower, upper)) in self.input_bounds.iter().enumerate() {
            desc.push_str(&format!("  X_{}: [{:.6}, {:.6}]\n", i, lower, upper));
        }

        desc.push_str("Output constraints (unsafe if ALL satisfied):\n");
        for c in &self.output_constraints {
            match c {
                OutputConstraint::LessEq(i, j) => desc.push_str(&format!("  Y_{} <= Y_{}\n", i, j)),
                OutputConstraint::GreaterEq(i, j) => {
                    desc.push_str(&format!("  Y_{} >= Y_{}\n", i, j))
                }
                OutputConstraint::LessThan(i, j) => {
                    desc.push_str(&format!("  Y_{} < Y_{}\n", i, j))
                }
                OutputConstraint::GreaterThan(i, j) => {
                    desc.push_str(&format!("  Y_{} > Y_{}\n", i, j))
                }
                OutputConstraint::LessEqConst(i, c) => {
                    desc.push_str(&format!("  Y_{} <= {:.6}\n", i, c))
                }
                OutputConstraint::GreaterEqConst(i, c) => {
                    desc.push_str(&format!("  Y_{} >= {:.6}\n", i, c))
                }
                OutputConstraint::LessThanConst(i, c) => {
                    desc.push_str(&format!("  Y_{} < {:.6}\n", i, c))
                }
                OutputConstraint::GreaterThanConst(i, c) => {
                    desc.push_str(&format!("  Y_{} > {:.6}\n", i, c))
                }
            }
        }

        desc
    }
}

impl Default for VnnLibSpec {
    fn default() -> Self {
        Self::new()
    }
}

/// Load a VNN-LIB property specification from a file.
///
/// # Arguments
///
/// * `path` - Path to the .vnnlib file
///
/// # Returns
///
/// A parsed `VnnLibSpec` containing input bounds and output constraints.
pub fn load_vnnlib<P: AsRef<Path>>(path: P) -> Result<VnnLibSpec> {
    let path = path.as_ref();
    info!("Loading VNN-LIB from: {}", path.display());

    let content = crate::io::read_string_maybe_gzip(path)?;

    parse_vnnlib(&content)
}

#[cfg(test)]
mod gz_tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn load_vnnlib_supports_gzip() {
        let dir = tempdir().unwrap();
        let vnnlib_gz_path = dir.path().join("prop.vnnlib.gz");

        let vnnlib_content = r#"
; Property with label: 0.
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (<= X_0 0.5))
(assert (>= X_0 -0.5))
(assert (<= Y_0 0.0))
"#;

        let mut enc = GzEncoder::new(Vec::new(), Compression::default());
        enc.write_all(vnnlib_content.as_bytes()).unwrap();
        let compressed = enc.finish().unwrap();
        std::fs::write(&vnnlib_gz_path, compressed).unwrap();

        let spec = load_vnnlib(&vnnlib_gz_path).unwrap();
        assert_eq!(spec.num_inputs, 1);
        assert_eq!(spec.num_outputs, 1);
        assert_eq!(spec.output_constraints.len(), 1);
    }
}

/// Parse VNN-LIB format from string content.
pub fn parse_vnnlib(content: &str) -> Result<VnnLibSpec> {
    let mut spec = VnnLibSpec::new();
    let mut input_lower: HashMap<usize, f64> = HashMap::new();
    let mut input_upper: HashMap<usize, f64> = HashMap::new();
    let mut max_input_idx = 0;
    let mut max_output_idx = 0;

    // Remove comments (lines starting with ;)
    let content: String = content
        .lines()
        .filter(|line| !line.trim().starts_with(';'))
        .collect::<Vec<_>>()
        .join(" ");

    // Parse S-expressions
    let tokens = tokenize(&content)?;
    let exprs = parse_expressions(&tokens)?;

    for expr in exprs {
        if let Expr::List(items) = expr {
            if items.is_empty() {
                continue;
            }

            match items.first() {
                Some(Expr::Symbol(s)) if s == "declare-const" => {
                    // (declare-const X_0 Real)
                    if items.len() >= 3 {
                        if let Some(Expr::Symbol(var_name)) = items.get(1) {
                            if let Some(idx) = parse_var_index(var_name, "X_") {
                                max_input_idx = max_input_idx.max(idx + 1);
                            } else if let Some(idx) = parse_var_index(var_name, "Y_") {
                                max_output_idx = max_output_idx.max(idx + 1);
                            }
                        }
                    }
                }
                Some(Expr::Symbol(s)) if s == "assert" => {
                    // (assert (<= X_0 0.5))
                    if items.len() >= 2 {
                        parse_assert(
                            &items[1],
                            &mut input_lower,
                            &mut input_upper,
                            &mut spec,
                            true,
                        )?;
                    }
                }
                _ => {
                    // Skip unknown expressions
                    debug!("Skipping unknown expression: {:?}", items.first());
                }
            }
        }
    }

    // Build input bounds from collected constraints
    spec.num_inputs = max_input_idx;
    spec.num_outputs = max_output_idx;
    spec.input_bounds = Vec::with_capacity(max_input_idx);

    for i in 0..max_input_idx {
        let lower = input_lower.get(&i).copied().unwrap_or(f64::NEG_INFINITY);
        let upper = input_upper.get(&i).copied().unwrap_or(f64::INFINITY);
        spec.input_bounds.push((lower, upper));
    }

    info!(
        "Parsed VNN-LIB: {} inputs, {} outputs, {} constraints",
        spec.num_inputs,
        spec.num_outputs,
        spec.output_constraints.len()
    );

    Ok(spec)
}

/// Simple S-expression representation.
#[derive(Debug, Clone)]
enum Expr {
    Symbol(String),
    Number(f64),
    List(Vec<Expr>),
}

/// Tokenize VNN-LIB content.
fn tokenize(content: &str) -> Result<Vec<String>> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut in_string = false;

    for c in content.chars() {
        if in_string {
            current.push(c);
            if c == '"' {
                tokens.push(current.clone());
                current.clear();
                in_string = false;
            }
        } else {
            match c {
                '(' | ')' => {
                    if !current.is_empty() {
                        tokens.push(current.clone());
                        current.clear();
                    }
                    tokens.push(c.to_string());
                }
                ' ' | '\t' | '\n' | '\r' => {
                    if !current.is_empty() {
                        tokens.push(current.clone());
                        current.clear();
                    }
                }
                '"' => {
                    if !current.is_empty() {
                        tokens.push(current.clone());
                        current.clear();
                    }
                    current.push(c);
                    in_string = true;
                }
                _ => {
                    current.push(c);
                }
            }
        }
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    Ok(tokens)
}

/// Parse tokens into S-expressions.
fn parse_expressions(tokens: &[String]) -> Result<Vec<Expr>> {
    let mut exprs = Vec::new();
    let mut pos = 0;

    while pos < tokens.len() {
        let (expr, new_pos) = parse_expr(tokens, pos)?;
        exprs.push(expr);
        pos = new_pos;
    }

    Ok(exprs)
}

/// Parse a single S-expression starting at position.
fn parse_expr(tokens: &[String], pos: usize) -> Result<(Expr, usize)> {
    if pos >= tokens.len() {
        return Err(GammaError::ModelLoad("Unexpected end of input".to_string()));
    }

    let token = &tokens[pos];

    if token == "(" {
        // Parse list
        let mut items = Vec::new();
        let mut i = pos + 1;

        while i < tokens.len() && tokens[i] != ")" {
            let (expr, new_pos) = parse_expr(tokens, i)?;
            items.push(expr);
            i = new_pos;
        }

        if i >= tokens.len() {
            return Err(GammaError::ModelLoad(
                "Unmatched opening parenthesis".to_string(),
            ));
        }

        Ok((Expr::List(items), i + 1))
    } else if token == ")" {
        Err(GammaError::ModelLoad(
            "Unexpected closing parenthesis".to_string(),
        ))
    } else if let Ok(num) = token.parse::<f64>() {
        Ok((Expr::Number(num), pos + 1))
    } else {
        Ok((Expr::Symbol(token.clone()), pos + 1))
    }
}

/// Parse variable index from name like "X_0" or "Y_1".
fn parse_var_index(name: &str, prefix: &str) -> Option<usize> {
    name.strip_prefix(prefix).and_then(|s| s.parse().ok())
}

/// Check if an expression contains output variable constraints (Y_i comparisons).
fn contains_output_constraint(expr: &Expr) -> bool {
    match expr {
        Expr::Symbol(s) => s.starts_with('Y') || s.starts_with("Y_"),
        Expr::List(items) => items.iter().any(contains_output_constraint),
        _ => false,
    }
}

/// Parse an assert expression and update bounds/constraints.
/// `is_top_level` indicates whether this is the first level of output constraints
/// (used to detect disjunctive vs conjunctive property structure).
fn parse_assert(
    expr: &Expr,
    input_lower: &mut HashMap<usize, f64>,
    input_upper: &mut HashMap<usize, f64>,
    spec: &mut VnnLibSpec,
    is_top_level: bool,
) -> Result<()> {
    if let Expr::List(items) = expr {
        if items.is_empty() {
            return Ok(());
        }

        let op = match items.first() {
            Some(Expr::Symbol(s)) => s.as_str(),
            _ => return Ok(()),
        };

        // Handle OR expressions: (or C1 C2 ... Cn)
        // For OR semantics, unsafe if ANY holds, so SAFE requires ALL violated.
        if op == "or" {
            // Check if this OR contains output constraints (Y_i comparisons)
            // If at top level and contains output constraints, mark as disjunction
            if is_top_level && contains_output_constraint(expr) {
                spec.is_disjunction = true;
            }
            for child in items.iter().skip(1) {
                parse_assert(child, input_lower, input_upper, spec, false)?;
            }
            return Ok(());
        }

        // Handle AND expressions: (and C1 C2 ... Cn)
        // For conjunctive properties, all constraints must hold for unsafe.
        if op == "and" {
            for child in items.iter().skip(1) {
                parse_assert(child, input_lower, input_upper, spec, false)?;
            }
            return Ok(());
        }

        // For comparison operators, we need exactly 2 arguments
        if items.len() < 3 {
            return Ok(());
        }

        // Get the arguments
        let arg1 = &items[1];
        let arg2 = &items[2];

        // Try to parse as input bound
        if let Some((var_idx, is_input)) = get_var_info(arg1) {
            if is_input {
                // Input constraint: X_i op constant
                if let Some(val) = get_number(arg2) {
                    match op {
                        "<=" => {
                            // X_i <= val means upper bound
                            input_upper
                                .entry(var_idx)
                                .and_modify(|u| *u = u.min(val))
                                .or_insert(val);
                        }
                        ">=" => {
                            // X_i >= val means lower bound
                            input_lower
                                .entry(var_idx)
                                .and_modify(|l| *l = l.max(val))
                                .or_insert(val);
                        }
                        "<" => {
                            // X_i < val means upper bound (exclusive)
                            input_upper
                                .entry(var_idx)
                                .and_modify(|u| *u = u.min(val))
                                .or_insert(val);
                        }
                        ">" => {
                            // X_i > val means lower bound (exclusive)
                            input_lower
                                .entry(var_idx)
                                .and_modify(|l| *l = l.max(val))
                                .or_insert(val);
                        }
                        _ => {
                            warn!("Unknown operator in input constraint: {}", op);
                        }
                    }
                    return Ok(());
                }
            } else {
                // Output constraint involving Y_i
                if let Some((other_idx, other_is_input)) = get_var_info(arg2) {
                    if !other_is_input {
                        // Y_i op Y_j
                        let constraint = match op {
                            "<=" => OutputConstraint::LessEq(var_idx, other_idx),
                            ">=" => OutputConstraint::GreaterEq(var_idx, other_idx),
                            "<" => OutputConstraint::LessThan(var_idx, other_idx),
                            ">" => OutputConstraint::GreaterThan(var_idx, other_idx),
                            _ => {
                                warn!("Unknown operator in output constraint: {}", op);
                                return Ok(());
                            }
                        };
                        spec.output_constraints.push(constraint);
                        return Ok(());
                    }
                } else if let Some(val) = get_number(arg2) {
                    // Y_i op constant
                    let constraint = match op {
                        "<=" => OutputConstraint::LessEqConst(var_idx, val),
                        ">=" => OutputConstraint::GreaterEqConst(var_idx, val),
                        "<" => OutputConstraint::LessThanConst(var_idx, val),
                        ">" => OutputConstraint::GreaterThanConst(var_idx, val),
                        _ => {
                            warn!("Unknown operator in output constraint: {}", op);
                            return Ok(());
                        }
                    };
                    spec.output_constraints.push(constraint);
                    return Ok(());
                }
            }
        }

        // Try reversed form: constant op var
        if let Some((var_idx, is_input)) = get_var_info(arg2) {
            if let Some(val) = get_number(arg1) {
                if is_input {
                    // constant op X_i
                    match op {
                        "<=" => {
                            // val <= X_i means X_i >= val (lower bound)
                            input_lower
                                .entry(var_idx)
                                .and_modify(|l| *l = l.max(val))
                                .or_insert(val);
                        }
                        ">=" => {
                            // val >= X_i means X_i <= val (upper bound)
                            input_upper
                                .entry(var_idx)
                                .and_modify(|u| *u = u.min(val))
                                .or_insert(val);
                        }
                        "<" => {
                            // val < X_i means X_i > val (lower bound exclusive)
                            input_lower
                                .entry(var_idx)
                                .and_modify(|l| *l = l.max(val))
                                .or_insert(val);
                        }
                        ">" => {
                            // val > X_i means X_i < val (upper bound exclusive)
                            input_upper
                                .entry(var_idx)
                                .and_modify(|u| *u = u.min(val))
                                .or_insert(val);
                        }
                        _ => {}
                    }
                } else {
                    // constant op Y_i
                    let constraint = match op {
                        "<=" => OutputConstraint::GreaterEqConst(var_idx, val),
                        ">=" => OutputConstraint::LessEqConst(var_idx, val),
                        "<" => OutputConstraint::GreaterThanConst(var_idx, val),
                        ">" => OutputConstraint::LessThanConst(var_idx, val),
                        _ => return Ok(()),
                    };
                    spec.output_constraints.push(constraint);
                }
            }
        }
    }

    Ok(())
}

/// Get variable info from expression (index, is_input).
fn get_var_info(expr: &Expr) -> Option<(usize, bool)> {
    if let Expr::Symbol(name) = expr {
        if let Some(idx) = parse_var_index(name, "X_") {
            return Some((idx, true));
        }
        if let Some(idx) = parse_var_index(name, "Y_") {
            return Some((idx, false));
        }
    }
    None
}

/// Get number from expression.
fn get_number(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::Number(n) => Some(*n),
        Expr::Symbol(s) => s.parse().ok(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_vnnlib() {
        let content = r#"
; Simple test
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (>= X_0 -1))
(assert (<= X_0 1))

(assert (<= Y_0 -1))
"#;

        let spec = parse_vnnlib(content).unwrap();
        assert_eq!(spec.num_inputs, 1);
        assert_eq!(spec.num_outputs, 1);
        assert_eq!(spec.input_bounds.len(), 1);
        assert_eq!(spec.input_bounds[0], (-1.0, 1.0));
        assert_eq!(spec.output_constraints.len(), 1);
        assert!(matches!(
            spec.output_constraints[0],
            OutputConstraint::LessEqConst(0, c) if (c - (-1.0)).abs() < 1e-10
        ));
    }

    #[test]
    fn test_parse_acasxu_property() {
        let content = r#"
; ACAS Xu property 2
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const X_2 Real)
(declare-const X_3 Real)
(declare-const X_4 Real)

(declare-const Y_0 Real)
(declare-const Y_1 Real)
(declare-const Y_2 Real)
(declare-const Y_3 Real)
(declare-const Y_4 Real)

(assert (<= X_0 0.679857769))
(assert (>= X_0 0.6))

(assert (<= X_1 0.5))
(assert (>= X_1 -0.5))

(assert (<= X_2 0.5))
(assert (>= X_2 -0.5))

(assert (<= X_3 0.5))
(assert (>= X_3 0.45))

(assert (<= X_4 -0.45))
(assert (>= X_4 -0.5))

; Unsafe if COC is maximal
(assert (<= Y_1 Y_0))
(assert (<= Y_2 Y_0))
(assert (<= Y_3 Y_0))
(assert (<= Y_4 Y_0))
"#;

        let spec = parse_vnnlib(content).unwrap();
        assert_eq!(spec.num_inputs, 5);
        assert_eq!(spec.num_outputs, 5);
        assert_eq!(spec.input_bounds.len(), 5);

        // Check input bounds
        assert!((spec.input_bounds[0].0 - 0.6).abs() < 1e-10);
        assert!((spec.input_bounds[0].1 - 0.679857769).abs() < 1e-10);
        assert!((spec.input_bounds[1].0 - (-0.5)).abs() < 1e-10);
        assert!((spec.input_bounds[1].1 - 0.5).abs() < 1e-10);

        // Check output constraints (Y_1 <= Y_0, Y_2 <= Y_0, Y_3 <= Y_0, Y_4 <= Y_0)
        assert_eq!(spec.output_constraints.len(), 4);
        assert!(matches!(
            spec.output_constraints[0],
            OutputConstraint::LessEq(1, 0)
        ));
        assert!(matches!(
            spec.output_constraints[1],
            OutputConstraint::LessEq(2, 0)
        ));
        assert!(matches!(
            spec.output_constraints[2],
            OutputConstraint::LessEq(3, 0)
        ));
        assert!(matches!(
            spec.output_constraints[3],
            OutputConstraint::LessEq(4, 0)
        ));
    }

    #[test]
    fn test_check_unsafe() {
        let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))

(assert (<= Y_0 Y_1))
"#;

        let spec = parse_vnnlib(content).unwrap();

        // Y_0 <= Y_1 is satisfied (unsafe region)
        assert!(spec.check_unsafe(&[0.5, 1.0]));

        // Y_0 > Y_1 is not satisfied (safe)
        assert!(!spec.check_unsafe(&[1.5, 1.0]));
    }

    #[test]
    fn test_describe() {
        let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (>= X_0 -1))
(assert (<= X_0 1))
(assert (<= Y_0 0))
"#;

        let spec = parse_vnnlib(content).unwrap();
        let desc = spec.describe();

        assert!(desc.contains("1 inputs"));
        assert!(desc.contains("1 outputs"));
        assert!(desc.contains("X_0"));
        assert!(desc.contains("Y_0"));
    }

    #[test]
    fn test_get_input_bounds_f32() {
        let content = r#"
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)

(assert (>= X_0 -1.5))
(assert (<= X_0 1.5))
(assert (>= X_1 0.0))
(assert (<= X_1 2.0))
"#;

        let spec = parse_vnnlib(content).unwrap();
        let (lower, upper) = spec.get_input_bounds_f32();

        assert_eq!(lower.len(), 2);
        assert_eq!(upper.len(), 2);
        assert!((lower[0] - (-1.5f32)).abs() < 1e-6);
        assert!((upper[0] - 1.5f32).abs() < 1e-6);
        assert!((lower[1] - 0.0f32).abs() < 1e-6);
        assert!((upper[1] - 2.0f32).abs() < 1e-6);
    }

    #[test]
    fn test_load_real_vnnlib() {
        // Try to load an actual VNN-LIB file if available
        let test_paths = [
            "../../research/repos/nnenum/examples/test/test_prop.vnnlib",
            "../../research/repos/nnenum/examples/acasxu/data/prop_2.vnnlib",
        ];

        for path in test_paths {
            let full_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(path);
            if full_path.exists() {
                let spec = load_vnnlib(&full_path).unwrap();
                assert!(spec.num_inputs > 0);
                assert!(spec.num_outputs > 0);
                assert!(spec.has_valid_bounds());
                println!("Loaded {}: {}", path, spec.describe());
            }
        }
    }

    // ==================== NEW COMPREHENSIVE TESTS ====================

    #[test]
    fn test_vnnlib_spec_new() {
        let spec = VnnLibSpec::new();
        assert_eq!(spec.num_inputs, 0);
        assert_eq!(spec.num_outputs, 0);
        assert!(spec.input_bounds.is_empty());
        assert!(spec.output_constraints.is_empty());
        assert!(!spec.is_disjunction);
    }

    #[test]
    fn test_vnnlib_spec_default() {
        let spec = VnnLibSpec::default();
        assert_eq!(spec.num_inputs, 0);
        assert_eq!(spec.num_outputs, 0);
        assert!(spec.input_bounds.is_empty());
        assert!(spec.output_constraints.is_empty());
        assert!(!spec.is_disjunction);
    }

    #[test]
    fn test_has_valid_bounds_invalid() {
        let mut spec = VnnLibSpec::new();
        spec.input_bounds.push((5.0, 1.0)); // lower > upper = invalid
        assert!(!spec.has_valid_bounds());
    }

    #[test]
    fn test_has_valid_bounds_empty() {
        let spec = VnnLibSpec::new();
        assert!(spec.has_valid_bounds()); // Empty bounds are valid
    }

    #[test]
    fn test_has_valid_bounds_equal() {
        let mut spec = VnnLibSpec::new();
        spec.input_bounds.push((1.0, 1.0)); // lower == upper is valid
        assert!(spec.has_valid_bounds());
    }

    #[test]
    fn test_get_input_bounds_f64() {
        let content = r#"
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)

(assert (>= X_0 -1.5))
(assert (<= X_0 1.5))
(assert (>= X_1 0.25))
(assert (<= X_1 2.75))
"#;

        let spec = parse_vnnlib(content).unwrap();
        let (lower, upper) = spec.get_input_bounds();

        assert_eq!(lower.len(), 2);
        assert_eq!(upper.len(), 2);
        assert!((lower[0] - (-1.5)).abs() < 1e-10);
        assert!((upper[0] - 1.5).abs() < 1e-10);
        assert!((lower[1] - 0.25).abs() < 1e-10);
        assert!((upper[1] - 2.75).abs() < 1e-10);
    }

    #[test]
    fn test_check_unsafe_greater_eq_constraint() {
        let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))

(assert (>= Y_0 Y_1))
"#;

        let spec = parse_vnnlib(content).unwrap();

        // Y_0 >= Y_1 is satisfied (unsafe)
        assert!(spec.check_unsafe(&[2.0, 1.0]));

        // Y_0 >= Y_1 not satisfied (safe)
        assert!(!spec.check_unsafe(&[0.5, 1.0]));
    }

    #[test]
    fn test_check_unsafe_less_than_constraint() {
        let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))

(assert (< Y_0 Y_1))
"#;

        let spec = parse_vnnlib(content).unwrap();

        // Y_0 < Y_1 is satisfied (unsafe)
        assert!(spec.check_unsafe(&[0.5, 1.0]));

        // Y_0 < Y_1 not satisfied when equal (safe)
        assert!(!spec.check_unsafe(&[1.0, 1.0]));
    }

    #[test]
    fn test_check_unsafe_greater_than_constraint() {
        let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))

(assert (> Y_0 Y_1))
"#;

        let spec = parse_vnnlib(content).unwrap();

        // Y_0 > Y_1 is satisfied (unsafe)
        assert!(spec.check_unsafe(&[1.5, 1.0]));

        // Y_0 > Y_1 not satisfied when equal (safe)
        assert!(!spec.check_unsafe(&[1.0, 1.0]));
    }

    #[test]
    fn test_check_unsafe_less_eq_const() {
        let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))

(assert (<= Y_0 0.5))
"#;

        let spec = parse_vnnlib(content).unwrap();

        // Y_0 <= 0.5 is satisfied
        assert!(spec.check_unsafe(&[0.5]));
        assert!(spec.check_unsafe(&[0.3]));

        // Y_0 <= 0.5 not satisfied
        assert!(!spec.check_unsafe(&[0.6]));
    }

    #[test]
    fn test_check_unsafe_greater_eq_const() {
        let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))

(assert (>= Y_0 0.5))
"#;

        let spec = parse_vnnlib(content).unwrap();

        // Y_0 >= 0.5 is satisfied
        assert!(spec.check_unsafe(&[0.5]));
        assert!(spec.check_unsafe(&[0.7]));

        // Y_0 >= 0.5 not satisfied
        assert!(!spec.check_unsafe(&[0.4]));
    }

    #[test]
    fn test_check_unsafe_less_than_const() {
        let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))

(assert (< Y_0 0.5))
"#;

        let spec = parse_vnnlib(content).unwrap();

        // Y_0 < 0.5 is satisfied
        assert!(spec.check_unsafe(&[0.4]));

        // Y_0 < 0.5 not satisfied (equal)
        assert!(!spec.check_unsafe(&[0.5]));
    }

    #[test]
    fn test_check_unsafe_greater_than_const() {
        let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))

(assert (> Y_0 0.5))
"#;

        let spec = parse_vnnlib(content).unwrap();

        // Y_0 > 0.5 is satisfied
        assert!(spec.check_unsafe(&[0.6]));

        // Y_0 > 0.5 not satisfied (equal)
        assert!(!spec.check_unsafe(&[0.5]));
    }

    #[test]
    fn test_describe_all_constraint_types() {
        let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (>= X_0 -1))
(assert (<= X_0 1))

(assert (<= Y_0 Y_1))
(assert (>= Y_0 Y_1))
(assert (< Y_0 Y_1))
(assert (> Y_0 Y_1))
(assert (<= Y_0 0.5))
(assert (>= Y_0 -0.5))
(assert (< Y_0 1.0))
(assert (> Y_0 -1.0))
"#;

        let spec = parse_vnnlib(content).unwrap();
        let desc = spec.describe();

        // Check all constraint types appear in description
        assert!(desc.contains("Y_0 <= Y_1"));
        assert!(desc.contains("Y_0 >= Y_1"));
        assert!(desc.contains("Y_0 < Y_1"));
        assert!(desc.contains("Y_0 > Y_1"));
        assert!(desc.contains("Y_0 <= 0.5"));
        assert!(desc.contains("Y_0 >= -0.5"));
        assert!(desc.contains("Y_0 < 1.0"));
        assert!(desc.contains("Y_0 > -1.0"));
    }

    #[test]
    fn test_tokenize_basic() {
        let tokens = tokenize("(assert (<= X_0 1))").unwrap();
        assert_eq!(tokens, vec!["(", "assert", "(", "<=", "X_0", "1", ")", ")"]);
    }

    #[test]
    fn test_tokenize_with_whitespace() {
        let tokens = tokenize("  (  assert   X_0  )  ").unwrap();
        assert_eq!(tokens, vec!["(", "assert", "X_0", ")"]);
    }

    #[test]
    fn test_tokenize_with_newlines() {
        let tokens = tokenize("(assert\n  X_0\n)").unwrap();
        assert_eq!(tokens, vec!["(", "assert", "X_0", ")"]);
    }

    #[test]
    fn test_tokenize_with_string() {
        let tokens = tokenize("(set-info \"test string\")").unwrap();
        assert_eq!(tokens, vec!["(", "set-info", "\"test string\"", ")"]);
    }

    #[test]
    fn test_tokenize_empty() {
        let tokens = tokenize("").unwrap();
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_parse_expr_unexpected_end() {
        let tokens: Vec<String> = vec![];
        let result = parse_expr(&tokens, 0);
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("Unexpected end"));
    }

    #[test]
    fn test_parse_expr_unmatched_open_paren() {
        let tokens = vec!["(".to_string(), "assert".to_string()];
        let result = parse_expr(&tokens, 0);
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("Unmatched opening"));
    }

    #[test]
    fn test_parse_expr_unexpected_close_paren() {
        let tokens = vec![")".to_string()];
        let result = parse_expr(&tokens, 0);
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("Unexpected closing"));
    }

    #[test]
    fn test_parse_var_index_valid() {
        assert_eq!(parse_var_index("X_0", "X_"), Some(0));
        assert_eq!(parse_var_index("X_42", "X_"), Some(42));
        assert_eq!(parse_var_index("Y_0", "Y_"), Some(0));
        assert_eq!(parse_var_index("Y_123", "Y_"), Some(123));
    }

    #[test]
    fn test_parse_var_index_invalid() {
        assert_eq!(parse_var_index("Z_0", "X_"), None);
        assert_eq!(parse_var_index("X_abc", "X_"), None);
        assert_eq!(parse_var_index("X_", "X_"), None);
        assert_eq!(parse_var_index("", "X_"), None);
    }

    #[test]
    fn test_contains_output_constraint_positive() {
        let tokens = tokenize("(<= Y_0 Y_1)").unwrap();
        let exprs = parse_expressions(&tokens).unwrap();
        assert!(contains_output_constraint(&exprs[0]));
    }

    #[test]
    fn test_contains_output_constraint_negative() {
        let tokens = tokenize("(<= X_0 1.0)").unwrap();
        let exprs = parse_expressions(&tokens).unwrap();
        assert!(!contains_output_constraint(&exprs[0]));
    }

    #[test]
    fn test_reversed_input_bounds() {
        // Test constant op variable form
        let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

; Reversed form: constant <= variable
(assert (<= -1.0 X_0))
(assert (>= 1.0 X_0))
"#;

        let spec = parse_vnnlib(content).unwrap();
        assert_eq!(spec.num_inputs, 1);
        // -1.0 <= X_0 means X_0 >= -1.0 (lower bound)
        // 1.0 >= X_0 means X_0 <= 1.0 (upper bound)
        assert!((spec.input_bounds[0].0 - (-1.0)).abs() < 1e-10);
        assert!((spec.input_bounds[0].1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_reversed_output_constraints() {
        // Test constant op output variable
        let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))

; Reversed form: 0.5 <= Y_0 means Y_0 >= 0.5
(assert (<= 0.5 Y_0))
"#;

        let spec = parse_vnnlib(content).unwrap();
        assert_eq!(spec.output_constraints.len(), 1);
        // 0.5 <= Y_0 means Y_0 >= 0.5
        assert!(matches!(
            spec.output_constraints[0],
            OutputConstraint::GreaterEqConst(0, c) if (c - 0.5).abs() < 1e-10
        ));
    }

    #[test]
    fn test_and_expression() {
        let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))

(assert (and (<= Y_0 Y_1) (<= Y_1 1.0)))
"#;

        let spec = parse_vnnlib(content).unwrap();
        // AND should parse both constraints
        assert_eq!(spec.output_constraints.len(), 2);
    }

    #[test]
    fn test_or_expression_sets_disjunction() {
        let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))

(assert (or (<= Y_0 0.0) (<= Y_1 0.0)))
"#;

        let spec = parse_vnnlib(content).unwrap();
        // OR with output constraints should set is_disjunction
        assert!(spec.is_disjunction);
        assert_eq!(spec.output_constraints.len(), 2);
    }

    #[test]
    fn test_partial_bounds_lower_only() {
        let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (>= X_0 -1.0))
; No upper bound specified
"#;

        let spec = parse_vnnlib(content).unwrap();
        assert_eq!(spec.num_inputs, 1);
        assert!((spec.input_bounds[0].0 - (-1.0)).abs() < 1e-10);
        assert!(spec.input_bounds[0].1.is_infinite()); // Default infinity
    }

    #[test]
    fn test_partial_bounds_upper_only() {
        let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (<= X_0 1.0))
; No lower bound specified
"#;

        let spec = parse_vnnlib(content).unwrap();
        assert_eq!(spec.num_inputs, 1);
        assert!(spec.input_bounds[0].0.is_infinite()); // Default -infinity
        assert!((spec.input_bounds[0].1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_multiple_bounds_same_variable_takes_tightest() {
        let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (>= X_0 -2.0))
(assert (>= X_0 -1.0))
(assert (<= X_0 2.0))
(assert (<= X_0 1.0))
"#;

        let spec = parse_vnnlib(content).unwrap();
        // Lower bound: max(-2.0, -1.0) = -1.0
        // Upper bound: min(2.0, 1.0) = 1.0
        assert!((spec.input_bounds[0].0 - (-1.0)).abs() < 1e-10);
        assert!((spec.input_bounds[0].1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_strict_inequality_input_bounds() {
        let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (< X_0 1.0))
(assert (> X_0 -1.0))
"#;

        let spec = parse_vnnlib(content).unwrap();
        // Strict inequalities should still set bounds
        assert!((spec.input_bounds[0].0 - (-1.0)).abs() < 1e-10);
        assert!((spec.input_bounds[0].1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_reversed_strict_input_bounds() {
        let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

; val < X_0 means X_0 > val (lower bound exclusive)
(assert (< -1.0 X_0))
; val > X_0 means X_0 < val (upper bound exclusive)
(assert (> 1.0 X_0))
"#;

        let spec = parse_vnnlib(content).unwrap();
        assert!((spec.input_bounds[0].0 - (-1.0)).abs() < 1e-10);
        assert!((spec.input_bounds[0].1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_reversed_strict_output_constraints() {
        let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))

; 0.5 < Y_0 means Y_0 > 0.5
(assert (< 0.5 Y_0))
; 1.0 > Y_0 means Y_0 < 1.0
(assert (> 1.0 Y_0))
"#;

        let spec = parse_vnnlib(content).unwrap();
        assert_eq!(spec.output_constraints.len(), 2);
        assert!(matches!(
            spec.output_constraints[0],
            OutputConstraint::GreaterThanConst(0, c) if (c - 0.5).abs() < 1e-10
        ));
        assert!(matches!(
            spec.output_constraints[1],
            OutputConstraint::LessThanConst(0, c) if (c - 1.0).abs() < 1e-10
        ));
    }

    #[test]
    fn test_scientific_notation_in_bounds() {
        let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (>= X_0 -1.5e-3))
(assert (<= X_0 2.0E+2))
"#;

        let spec = parse_vnnlib(content).unwrap();
        assert!((spec.input_bounds[0].0 - (-0.0015)).abs() < 1e-10);
        assert!((spec.input_bounds[0].1 - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty_assert() {
        let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert ())
"#;

        // Empty assert should be ignored gracefully
        let spec = parse_vnnlib(content).unwrap();
        assert_eq!(spec.num_inputs, 1);
    }

    #[test]
    fn test_unknown_expression_ignored() {
        let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)

(set-info :status unknown)
(set-logic QF_LRA)

(assert (>= X_0 0))
(assert (<= X_0 1))
"#;

        // Unknown top-level expressions should be ignored
        let spec = parse_vnnlib(content).unwrap();
        assert_eq!(spec.num_inputs, 1);
        assert!((spec.input_bounds[0].0 - 0.0).abs() < 1e-10);
        assert!((spec.input_bounds[0].1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_output_constraint_equality() {
        // Test that OutputConstraint derives PartialEq correctly
        let c1 = OutputConstraint::LessEq(0, 1);
        let c2 = OutputConstraint::LessEq(0, 1);
        let c3 = OutputConstraint::LessEq(1, 0);
        assert_eq!(c1, c2);
        assert_ne!(c1, c3);
    }

    #[test]
    fn test_nested_and_in_or() {
        let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))

(assert (or (and (<= Y_0 0.0) (<= Y_1 0.0)) (and (>= Y_0 1.0) (>= Y_1 1.0))))
"#;

        let spec = parse_vnnlib(content).unwrap();
        // All constraints should be parsed
        assert_eq!(spec.output_constraints.len(), 4);
        assert!(spec.is_disjunction);
    }

    #[test]
    fn test_get_number_from_symbol() {
        // Test that get_number can parse number strings
        let expr = Expr::Symbol("3.125".to_string());
        assert!((get_number(&expr).unwrap() - 3.125).abs() < 1e-10);
    }

    #[test]
    fn test_get_number_from_number() {
        let expr = Expr::Number(2.75);
        assert!((get_number(&expr).unwrap() - 2.75).abs() < 1e-10);
    }

    #[test]
    fn test_get_number_from_list() {
        let expr = Expr::List(vec![]);
        assert!(get_number(&expr).is_none());
    }

    #[test]
    fn test_get_var_info_x() {
        let expr = Expr::Symbol("X_5".to_string());
        assert_eq!(get_var_info(&expr), Some((5, true)));
    }

    #[test]
    fn test_get_var_info_y() {
        let expr = Expr::Symbol("Y_10".to_string());
        assert_eq!(get_var_info(&expr), Some((10, false)));
    }

    #[test]
    fn test_get_var_info_invalid() {
        let expr = Expr::Symbol("Z_0".to_string());
        assert!(get_var_info(&expr).is_none());

        let expr2 = Expr::Number(1.0);
        assert!(get_var_info(&expr2).is_none());
    }

    #[test]
    fn test_sparse_variable_indices() {
        // Test that sparse indices work (X_0, X_5, X_2)
        let content = r#"
(declare-const X_0 Real)
(declare-const X_5 Real)
(declare-const X_2 Real)
(declare-const Y_0 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))
(assert (>= X_5 5))
(assert (<= X_5 6))
(assert (>= X_2 2))
(assert (<= X_2 3))
"#;

        let spec = parse_vnnlib(content).unwrap();
        // num_inputs should be max_idx + 1 = 6
        assert_eq!(spec.num_inputs, 6);
        // Bounds for undefined indices should be (-inf, +inf)
        assert!(spec.input_bounds[1].0.is_infinite());
        assert!(spec.input_bounds[3].0.is_infinite());
        assert!(spec.input_bounds[4].0.is_infinite());
    }
}
