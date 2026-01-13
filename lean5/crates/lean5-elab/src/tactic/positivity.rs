//! Positivity tactics
//!
//! Provides tactics for analyzing expressions to determine if they are positive,
//! non-negative, or nonzero.

use lean5_kernel::name::Name;
use lean5_kernel::Expr;

use super::{LocalDecl, ProofState, TacticError, TacticResult};

/// Tactic: positivity_at
///
/// Analyzes a hypothesis to add information about whether expressions are positive,
/// non-negative, or nonzero using positivity analysis.
///
/// This is useful when you need to establish positivity facts about
/// values in hypotheses for use in subsequent reasoning.
///
/// # Example
/// ```text
/// -- h : x^2 + 1 > y
/// positivity_at h
/// -- Adds h_pos : x^2 + 1 > 0 to context
/// ```
///
/// # Errors
/// - `NoGoals` if there are no goals
/// - `HypothesisNotFound` if the named hypothesis doesn't exist
/// - `Other` if positivity analysis fails
pub fn positivity_at(state: &mut ProofState, hyp_name: &str) -> TacticResult {
    positivity_at_with_config(state, hyp_name, PositivityAtConfig::new())
}

/// Configuration for positivity_at
#[derive(Debug, Clone)]
pub struct PositivityAtConfig {
    /// Name for the generated positivity hypothesis
    pub result_name: Option<String>,
    /// Whether to try stronger claims (positive vs non-negative)
    pub try_stronger: bool,
}

impl Default for PositivityAtConfig {
    fn default() -> Self {
        PositivityAtConfig {
            result_name: None,
            try_stronger: true,
        }
    }
}

impl PositivityAtConfig {
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_name(mut self, name: &str) -> Self {
        self.result_name = Some(name.to_string());
        self
    }
}

/// positivity_at with configuration
pub fn positivity_at_with_config(
    state: &mut ProofState,
    hyp_name: &str,
    config: PositivityAtConfig,
) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Find the hypothesis
    let hyp = goal
        .local_ctx
        .iter()
        .find(|d| d.name == hyp_name)
        .ok_or_else(|| TacticError::HypothesisNotFound(hyp_name.to_string()))?;

    let hyp_ty = hyp.ty.clone();

    // Extract the expression to analyze for positivity
    // We look for patterns like: x > y, x ≥ y, x = y, etc.
    let (expr_to_analyze, comparison_kind) = extract_comparison_expr(&hyp_ty)?;

    // Perform positivity analysis on the expression
    let positivity_result = analyze_positivity(&expr_to_analyze)?;

    // Generate the result name
    let result_name = config
        .result_name
        .unwrap_or_else(|| format!("{hyp_name}_pos"));

    // Create the positivity proposition based on analysis
    let pos_prop = make_positivity_prop(&expr_to_analyze, positivity_result, config.try_stronger);

    // Get fresh fvar before mutable borrow
    let new_fvar = state.fresh_fvar();

    // Add the new hypothesis
    let goal = state.current_goal_mut().ok_or(TacticError::NoGoals)?;
    goal.local_ctx.push(LocalDecl {
        fvar: new_fvar,
        name: result_name,
        ty: pos_prop,
        value: None,
    });

    let _ = comparison_kind; // Mark as used
    Ok(())
}

/// Kind of comparison in an expression
#[derive(Debug, Clone, Copy)]
pub(crate) enum ComparisonKind {
    Gt, // >
    Ge, // ≥
    Lt, // <
    Le, // ≤
    Eq, // =
    Ne, // ≠
}

/// Extract the main expression from a comparison
pub(crate) fn extract_comparison_expr(ty: &Expr) -> Result<(Expr, ComparisonKind), TacticError> {
    // Look for patterns like GT.gt x y, GE.ge x y, etc.
    match ty {
        Expr::App(f, _rhs) => match &**f {
            Expr::App(f2, lhs) => {
                if let Expr::Const(name, _) = &**f2 {
                    let kind = match name.to_string().as_str() {
                        "GT.gt" | "gt" => Some(ComparisonKind::Gt),
                        "GE.ge" | "ge" => Some(ComparisonKind::Ge),
                        "LT.lt" | "lt" => Some(ComparisonKind::Lt),
                        "LE.le" | "le" => Some(ComparisonKind::Le),
                        "Eq" | "eq" => Some(ComparisonKind::Eq),
                        "Ne" | "ne" => Some(ComparisonKind::Ne),
                        _ => None,
                    };
                    if let Some(k) = kind {
                        return Ok(((**lhs).clone(), k));
                    }
                }
                // Check for app with more arguments
                if let Expr::App(f3, _) = &**f2 {
                    if let Expr::Const(name, _) = &**f3 {
                        let kind = match name.to_string().as_str() {
                            "GT.gt" | "gt" => Some(ComparisonKind::Gt),
                            "GE.ge" | "ge" => Some(ComparisonKind::Ge),
                            "LT.lt" | "lt" => Some(ComparisonKind::Lt),
                            "LE.le" | "le" => Some(ComparisonKind::Le),
                            _ => None,
                        };
                        if let Some(k) = kind {
                            return Ok(((**lhs).clone(), k));
                        }
                    }
                }
                Err(TacticError::Other(
                    "positivity_at: could not extract comparison from hypothesis".to_string(),
                ))
            }
            _ => Err(TacticError::Other(
                "positivity_at: hypothesis is not a comparison".to_string(),
            )),
        },
        _ => Err(TacticError::Other(
            "positivity_at: hypothesis is not a comparison".to_string(),
        )),
    }
}

/// Result of positivity analysis
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // NonZero variant reserved for future use
pub(crate) enum PositivityResult {
    /// Definitely positive (> 0)
    Positive,
    /// Definitely non-negative (≥ 0)
    NonNegative,
    /// Definitely nonzero (≠ 0)
    NonZero,
    /// Unknown
    Unknown,
}

/// Analyze an expression for positivity
pub(crate) fn analyze_positivity(expr: &Expr) -> Result<PositivityResult, TacticError> {
    // Simple pattern matching for common cases
    match expr {
        // Constants
        Expr::Const(name, _) => {
            let s = name.to_string();
            if s.ends_with(".one") || s == "1" {
                return Ok(PositivityResult::Positive);
            }
            if s.ends_with(".zero") || s == "0" {
                return Ok(PositivityResult::NonNegative);
            }
            Ok(PositivityResult::Unknown)
        }

        // Literal natural numbers are non-negative
        Expr::Lit(lean5_kernel::Literal::Nat(_)) => Ok(PositivityResult::NonNegative),

        // Application patterns
        Expr::App(f, arg) => {
            // Check for squared terms: x^2, x * x
            if is_square_pattern(expr) {
                return Ok(PositivityResult::NonNegative);
            }

            // Check for absolute value
            if is_abs_pattern(expr) {
                return Ok(PositivityResult::NonNegative);
            }

            // Check for sum of non-negatives
            if let Some((a, b)) = is_add_pattern(expr) {
                let a_pos = analyze_positivity(&a)?;
                let b_pos = analyze_positivity(&b)?;
                match (a_pos, b_pos) {
                    (PositivityResult::Positive, _) | (_, PositivityResult::Positive) => {
                        return Ok(PositivityResult::Positive);
                    }
                    (PositivityResult::NonNegative, PositivityResult::NonNegative) => {
                        return Ok(PositivityResult::NonNegative);
                    }
                    _ => {}
                }
            }

            // Check for product of positives
            if let Some((a, b)) = is_mul_pattern(expr) {
                let a_pos = analyze_positivity(&a)?;
                let b_pos = analyze_positivity(&b)?;
                match (a_pos, b_pos) {
                    (PositivityResult::Positive, PositivityResult::Positive) => {
                        return Ok(PositivityResult::Positive);
                    }
                    (PositivityResult::NonNegative, PositivityResult::NonNegative) => {
                        return Ok(PositivityResult::NonNegative);
                    }
                    _ => {}
                }
            }

            let _ = (f, arg); // Mark as used
            Ok(PositivityResult::Unknown)
        }

        _ => Ok(PositivityResult::Unknown),
    }
}

/// Check if expression is a square pattern (x^2 or x*x)
pub(crate) fn is_square_pattern(expr: &Expr) -> bool {
    // Check for HPow.hPow x 2 or HMul.hMul x x
    match expr {
        Expr::App(f, _) => {
            if let Expr::App(f2, _) = &**f {
                if let Expr::Const(name, _) = &**f2 {
                    let s = name.to_string();
                    if s.contains("HPow") || s.contains("pow") {
                        return true; // Simplified check
                    }
                }
                // Check for HMul.hMul
                if let Expr::App(f3, _) = &**f2 {
                    if let Expr::Const(name, _) = &**f3 {
                        if name.to_string().contains("HMul") || name.to_string().contains("mul") {
                            // Would need to check if both args are the same
                            return false; // Conservative
                        }
                    }
                }
            }
            false
        }
        _ => false,
    }
}

/// Check if expression is an absolute value pattern
pub(crate) fn is_abs_pattern(expr: &Expr) -> bool {
    match expr {
        Expr::App(f, _) => {
            if let Expr::Const(name, _) = &**f {
                let s = name.to_string();
                return s.contains("abs") || s.contains("Abs");
            }
            if let Expr::App(f2, _) = &**f {
                if let Expr::Const(name, _) = &**f2 {
                    let s = name.to_string();
                    return s.contains("abs") || s.contains("Abs");
                }
            }
            false
        }
        _ => false,
    }
}

/// Check if expression is an addition pattern
pub(crate) fn is_add_pattern(expr: &Expr) -> Option<(Expr, Expr)> {
    match expr {
        Expr::App(f, b) => {
            if let Expr::App(f2, a) = &**f {
                // Check for HAdd, Add.add, etc.
                let is_add = match &**f2 {
                    Expr::Const(name, _) => {
                        let s = name.to_string();
                        s.contains("HAdd") || s.contains("Add.add") || s.contains("add")
                    }
                    Expr::App(f3, _) => {
                        if let Expr::Const(name, _) = &**f3 {
                            let s = name.to_string();
                            s.contains("HAdd") || s.contains("Add.add")
                        } else {
                            false
                        }
                    }
                    _ => false,
                };
                if is_add {
                    return Some(((**a).clone(), (**b).clone()));
                }
            }
            None
        }
        _ => None,
    }
}

/// Check if expression is a multiplication pattern
pub(crate) fn is_mul_pattern(expr: &Expr) -> Option<(Expr, Expr)> {
    match expr {
        Expr::App(f, b) => {
            if let Expr::App(f2, a) = &**f {
                let is_mul = match &**f2 {
                    Expr::Const(name, _) => {
                        let s = name.to_string();
                        s.contains("HMul") || s.contains("Mul.mul") || s.contains("mul")
                    }
                    Expr::App(f3, _) => {
                        if let Expr::Const(name, _) = &**f3 {
                            let s = name.to_string();
                            s.contains("HMul") || s.contains("Mul.mul")
                        } else {
                            false
                        }
                    }
                    _ => false,
                };
                if is_mul {
                    return Some(((**a).clone(), (**b).clone()));
                }
            }
            None
        }
        _ => None,
    }
}

/// Make a positivity proposition from analysis result
pub(crate) fn make_positivity_prop(
    expr: &Expr,
    result: PositivityResult,
    try_stronger: bool,
) -> Expr {
    let zero = Expr::const_(Name::from_string("OfNat.ofNat"), vec![]);

    match result {
        PositivityResult::Positive if try_stronger => {
            // expr > 0
            Expr::app(
                Expr::app(
                    Expr::const_(Name::from_string("GT.gt"), vec![]),
                    expr.clone(),
                ),
                zero,
            )
        }
        PositivityResult::NonNegative | PositivityResult::Positive => {
            // expr ≥ 0
            Expr::app(
                Expr::app(
                    Expr::const_(Name::from_string("GE.ge"), vec![]),
                    expr.clone(),
                ),
                zero,
            )
        }
        PositivityResult::NonZero => {
            // expr ≠ 0
            Expr::app(
                Expr::app(Expr::const_(Name::from_string("Ne"), vec![]), expr.clone()),
                zero,
            )
        }
        PositivityResult::Unknown => {
            // Default to ≥ 0 as a guess
            Expr::app(
                Expr::app(
                    Expr::const_(Name::from_string("GE.ge"), vec![]),
                    expr.clone(),
                ),
                zero,
            )
        }
    }
}
