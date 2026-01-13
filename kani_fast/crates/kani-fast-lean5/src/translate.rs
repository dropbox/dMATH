//! SMT to Lean5 translation
//!
//! Translates SMT-LIB2 formulas to Lean5 expressions.

use crate::expr::{Lean5Expr, Lean5Name, Lean5Type};
use crate::smt_parser::{SmtAst, SmtSort};
use kani_fast_chc::result::{InvariantModel, SolvedPredicate};
use kani_fast_kinduction::SmtType;
use std::collections::HashMap;
use thiserror::Error;

/// Translation errors
#[derive(Debug, Error)]
pub enum TranslationError {
    #[error("unsupported SMT operator: {0}")]
    UnsupportedOperator(String),

    #[error("unsupported SMT sort: {0}")]
    UnsupportedSort(String),

    #[error("unknown variable: {0}")]
    UnknownVariable(String),

    #[error("invalid formula structure: {0}")]
    InvalidStructure(String),

    #[error("parse error: {0}")]
    ParseError(String),
}

/// Translation context with variable bindings
#[derive(Debug, Clone, Default)]
pub struct TranslationContext {
    /// Variable name to type mapping
    pub var_types: HashMap<String, Lean5Type>,
    /// Let-bound variables
    pub let_bindings: HashMap<String, Lean5Expr>,
}

impl TranslationContext {
    /// Create a new empty context
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a variable with its type
    pub fn add_var(&mut self, name: impl Into<String>, ty: Lean5Type) {
        self.var_types.insert(name.into(), ty);
    }

    /// Add a let binding
    pub fn add_let(&mut self, name: impl Into<String>, value: Lean5Expr) {
        self.let_bindings.insert(name.into(), value);
    }

    /// Get variable type
    pub fn get_var_type(&self, name: &str) -> Option<&Lean5Type> {
        self.var_types.get(name)
    }

    /// Check if a name is a let binding
    pub fn is_let_bound(&self, name: &str) -> bool {
        self.let_bindings.contains_key(name)
    }
}

/// Translate an invariant model to Lean5 expressions
pub fn translate_invariant(model: &InvariantModel) -> Result<Vec<Lean5Expr>, TranslationError> {
    model.predicates.iter().map(translate_predicate).collect()
}

/// Translate a single solved predicate to a Lean5 expression
pub fn translate_predicate(pred: &SolvedPredicate) -> Result<Lean5Expr, TranslationError> {
    // Build context from predicate parameters
    let mut ctx = TranslationContext::new();
    for (name, ty) in &pred.params {
        ctx.add_var(name.clone(), smt_type_to_lean5(ty));
    }

    // Parse the SMT formula
    let ast = crate::smt_parser::parse_smt_formula(&pred.formula.smt_formula)
        .map_err(|e| TranslationError::ParseError(e.to_string()))?;

    // Translate to Lean5
    let body = translate_ast(&ast, &ctx)?;

    // Wrap in forall for each parameter (clean variable names for Lean5 compatibility)
    let mut result = body;
    for (name, ty) in pred.params.iter().rev() {
        result = Lean5Expr::forall_(clean_var_name(name), smt_type_to_lean5(ty), result);
    }

    Ok(result)
}

/// Translate an SMT AST node to a Lean5 expression
pub fn translate_ast(
    ast: &SmtAst,
    ctx: &TranslationContext,
) -> Result<Lean5Expr, TranslationError> {
    match ast {
        SmtAst::Symbol(s) => {
            // Check if it's a let-bound variable first
            if let Some(binding) = ctx.let_bindings.get(s) {
                return Ok(binding.clone());
            }
            // Check if it's a known variable
            if ctx.var_types.contains_key(s) || s.contains('!') {
                // Variable reference (including Z3-generated names like x!0)
                Ok(Lean5Expr::var(clean_var_name(s)))
            } else {
                // Assume it's a constant
                Ok(Lean5Expr::const_(Lean5Name::simple(s.clone())))
            }
        }

        SmtAst::Int(n) => Ok(Lean5Expr::IntLit(*n)),

        SmtAst::Bool(b) => Ok(Lean5Expr::BoolLit(*b)),

        SmtAst::Neg(inner) => {
            let inner_expr = translate_ast(inner, ctx)?;
            // Check if inner is an integer - then negate
            if let SmtAst::Int(n) = inner.as_ref() {
                return Ok(Lean5Expr::IntLit(-n));
            }
            Ok(Lean5Expr::Neg(Box::new(inner_expr)))
        }

        SmtAst::App(op, args) => translate_app(op, args, ctx),

        SmtAst::Let(bindings, body) => {
            // Create new context with let bindings
            let mut new_ctx = ctx.clone();
            let mut lean_bindings = Vec::new();

            for (name, val) in bindings {
                let val_expr = translate_ast(val, &new_ctx)?;
                // Infer type from value (simplified)
                let ty = infer_type(&val_expr);
                new_ctx.add_var(name.clone(), ty.clone());
                new_ctx.add_let(name.clone(), val_expr.clone());
                lean_bindings.push((name.clone(), ty, val_expr));
            }

            // Translate body with updated context
            let body_expr = translate_ast(body, &new_ctx)?;

            // Build nested let expressions
            let mut result = body_expr;
            for (name, ty, val) in lean_bindings.into_iter().rev() {
                result = Lean5Expr::Let(name, ty, Box::new(val), Box::new(result));
            }

            Ok(result)
        }

        SmtAst::Forall(vars, body) => {
            let mut new_ctx = ctx.clone();
            for (name, sort) in vars {
                new_ctx.add_var(name.clone(), smt_sort_to_lean5(sort));
            }

            let body_expr = translate_ast(body, &new_ctx)?;

            let mut result = body_expr;
            for (name, sort) in vars.iter().rev() {
                result = Lean5Expr::forall_(clean_var_name(name), smt_sort_to_lean5(sort), result);
            }

            Ok(result)
        }

        SmtAst::Exists(vars, body) => {
            let mut new_ctx = ctx.clone();
            for (name, sort) in vars {
                new_ctx.add_var(name.clone(), smt_sort_to_lean5(sort));
            }

            let body_expr = translate_ast(body, &new_ctx)?;

            let mut result = body_expr;
            for (name, sort) in vars.iter().rev() {
                result = Lean5Expr::exists_(clean_var_name(name), smt_sort_to_lean5(sort), result);
            }

            Ok(result)
        }
    }
}

/// Translate an SMT application to Lean5
fn translate_app(
    op: &str,
    args: &[SmtAst],
    ctx: &TranslationContext,
) -> Result<Lean5Expr, TranslationError> {
    // Translate arguments
    let lean_args: Result<Vec<Lean5Expr>, _> = args.iter().map(|a| translate_ast(a, ctx)).collect();
    let lean_args = lean_args?;

    match op {
        // Logical operators
        "and" => {
            if lean_args.is_empty() {
                return Ok(Lean5Expr::BoolLit(true));
            }
            let mut result = lean_args[0].clone();
            for arg in &lean_args[1..] {
                result = Lean5Expr::and(result, arg.clone());
            }
            Ok(result)
        }

        "or" => {
            if lean_args.is_empty() {
                return Ok(Lean5Expr::BoolLit(false));
            }
            let mut result = lean_args[0].clone();
            for arg in &lean_args[1..] {
                result = Lean5Expr::or(result, arg.clone());
            }
            Ok(result)
        }

        "not" => {
            if lean_args.len() != 1 {
                return Err(TranslationError::InvalidStructure(
                    "not requires exactly 1 argument".to_string(),
                ));
            }
            Ok(Lean5Expr::not(lean_args[0].clone()))
        }

        "=>" => {
            if lean_args.len() != 2 {
                return Err(TranslationError::InvalidStructure(
                    "=> requires exactly 2 arguments".to_string(),
                ));
            }
            Ok(Lean5Expr::implies(
                lean_args[0].clone(),
                lean_args[1].clone(),
            ))
        }

        // Equality and comparison
        "=" => {
            if lean_args.len() != 2 {
                return Err(TranslationError::InvalidStructure(
                    "= requires exactly 2 arguments".to_string(),
                ));
            }
            Ok(Lean5Expr::eq(lean_args[0].clone(), lean_args[1].clone()))
        }

        "<" => {
            if lean_args.len() != 2 {
                return Err(TranslationError::InvalidStructure(
                    "< requires exactly 2 arguments".to_string(),
                ));
            }
            Ok(Lean5Expr::lt(lean_args[0].clone(), lean_args[1].clone()))
        }

        "<=" => {
            if lean_args.len() != 2 {
                return Err(TranslationError::InvalidStructure(
                    "<= requires exactly 2 arguments".to_string(),
                ));
            }
            Ok(Lean5Expr::le(lean_args[0].clone(), lean_args[1].clone()))
        }

        ">" => {
            if lean_args.len() != 2 {
                return Err(TranslationError::InvalidStructure(
                    "> requires exactly 2 arguments".to_string(),
                ));
            }
            Ok(Lean5Expr::gt(lean_args[0].clone(), lean_args[1].clone()))
        }

        ">=" => {
            if lean_args.len() != 2 {
                return Err(TranslationError::InvalidStructure(
                    ">= requires exactly 2 arguments".to_string(),
                ));
            }
            Ok(Lean5Expr::ge(lean_args[0].clone(), lean_args[1].clone()))
        }

        // Arithmetic
        "+" => {
            if lean_args.is_empty() {
                return Ok(Lean5Expr::IntLit(0));
            }
            let mut result = lean_args[0].clone();
            for arg in &lean_args[1..] {
                result = Lean5Expr::add(result, arg.clone());
            }
            Ok(result)
        }

        "-" => {
            if lean_args.is_empty() {
                return Err(TranslationError::InvalidStructure(
                    "- requires at least 1 argument".to_string(),
                ));
            }
            if lean_args.len() == 1 {
                // Unary minus
                return Ok(Lean5Expr::Neg(Box::new(lean_args[0].clone())));
            }
            // Binary minus
            let mut result = lean_args[0].clone();
            for arg in &lean_args[1..] {
                result = Lean5Expr::sub(result, arg.clone());
            }
            Ok(result)
        }

        "*" => {
            if lean_args.is_empty() {
                return Ok(Lean5Expr::IntLit(1));
            }
            let mut result = lean_args[0].clone();
            for arg in &lean_args[1..] {
                result = Lean5Expr::mul(result, arg.clone());
            }
            Ok(result)
        }

        "/" | "div" => {
            if lean_args.len() != 2 {
                return Err(TranslationError::InvalidStructure(
                    "div requires exactly 2 arguments".to_string(),
                ));
            }
            Ok(Lean5Expr::Div(
                Box::new(lean_args[0].clone()),
                Box::new(lean_args[1].clone()),
            ))
        }

        "mod" => {
            if lean_args.len() != 2 {
                return Err(TranslationError::InvalidStructure(
                    "mod requires exactly 2 arguments".to_string(),
                ));
            }
            Ok(Lean5Expr::Mod(
                Box::new(lean_args[0].clone()),
                Box::new(lean_args[1].clone()),
            ))
        }

        // If-then-else
        "ite" => {
            if lean_args.len() != 3 {
                return Err(TranslationError::InvalidStructure(
                    "ite requires exactly 3 arguments".to_string(),
                ));
            }
            Ok(Lean5Expr::ite(
                lean_args[0].clone(),
                lean_args[1].clone(),
                lean_args[2].clone(),
            ))
        }

        // Unknown operator - try as function application
        _ => {
            let func = Lean5Expr::const_(Lean5Name::simple(op.to_string()));
            Ok(Lean5Expr::apps(func, lean_args))
        }
    }
}

/// Convert an SMT type to a Lean5 type
fn smt_type_to_lean5(ty: &SmtType) -> Lean5Type {
    match ty {
        SmtType::Int => Lean5Type::Int,
        SmtType::Bool => Lean5Type::Bool,
        SmtType::Real => Lean5Type::Const(Lean5Name::simple("Real")),
        SmtType::BitVec(_) => Lean5Type::Const(Lean5Name::simple("BitVec")),
        SmtType::Array { .. } => Lean5Type::Const(Lean5Name::simple("Array")),
    }
}

/// Convert an SMT sort to a Lean5 type
fn smt_sort_to_lean5(sort: &SmtSort) -> Lean5Type {
    match sort {
        SmtSort::Int => Lean5Type::Int,
        SmtSort::Bool => Lean5Type::Bool,
        SmtSort::Real => Lean5Type::Const(Lean5Name::simple("Real")),
        SmtSort::BitVec(_) => Lean5Type::Const(Lean5Name::simple("BitVec")),
        SmtSort::Array(_, _) => Lean5Type::Const(Lean5Name::simple("Array")),
        SmtSort::Unknown(s) => Lean5Type::Const(Lean5Name::simple(s.clone())),
    }
}

/// Infer the type of a Lean5 expression (simplified)
fn infer_type(expr: &Lean5Expr) -> Lean5Type {
    match expr {
        Lean5Expr::IntLit(_) => Lean5Type::Int,
        Lean5Expr::NatLit(_) => Lean5Type::Nat,
        Lean5Expr::BoolLit(_) => Lean5Type::Bool,
        Lean5Expr::And(_, _)
        | Lean5Expr::Or(_, _)
        | Lean5Expr::Not(_)
        | Lean5Expr::Implies(_, _)
        | Lean5Expr::Eq(_, _)
        | Lean5Expr::Lt(_, _)
        | Lean5Expr::Le(_, _)
        | Lean5Expr::Gt(_, _)
        | Lean5Expr::Ge(_, _) => Lean5Type::Prop,
        Lean5Expr::Add(_, _)
        | Lean5Expr::Sub(_, _)
        | Lean5Expr::Mul(_, _)
        | Lean5Expr::Div(_, _)
        | Lean5Expr::Mod(_, _)
        | Lean5Expr::Neg(_) => Lean5Type::Int,
        Lean5Expr::Forall(_, _, _) | Lean5Expr::Exists(_, _, _) => Lean5Type::Prop,
        Lean5Expr::Ite(_, t, _) => infer_type(t),
        Lean5Expr::Ascribe(_, ty) => ty.clone(),
        _ => Lean5Type::Type, // Default fallback
    }
}

/// Clean variable names (remove Z3-specific suffixes)
fn clean_var_name(name: &str) -> String {
    // Keep the variable name but make it valid Lean5
    // Z3 uses names like x!0, x!1, etc.
    // Replace ! with _ for Lean5 compatibility
    name.replace('!', "_")
}

#[cfg(test)]
mod tests {
    use super::*;
    use kani_fast_kinduction::StateFormula;

    #[test]
    fn test_translate_simple_comparison() {
        let ctx = TranslationContext::new();
        let ast = crate::smt_parser::parse_smt_formula("(>= x 0)").unwrap();
        let expr = translate_ast(&ast, &ctx).unwrap();

        // Should produce (x ≥ 0)
        assert!(matches!(expr, Lean5Expr::Ge(_, _)));
    }

    #[test]
    fn test_translate_conjunction() {
        let ctx = TranslationContext::new();
        let ast = crate::smt_parser::parse_smt_formula("(and (>= x 0) (< x 10))").unwrap();
        let expr = translate_ast(&ast, &ctx).unwrap();

        assert!(matches!(expr, Lean5Expr::And(_, _)));
    }

    #[test]
    fn test_translate_negation() {
        let ctx = TranslationContext::new();
        let ast = crate::smt_parser::parse_smt_formula("(not (< x 0))").unwrap();
        let expr = translate_ast(&ast, &ctx).unwrap();

        assert!(matches!(expr, Lean5Expr::Not(_)));
    }

    #[test]
    fn test_translate_let() {
        let ctx = TranslationContext::new();
        let ast = crate::smt_parser::parse_smt_formula("(let ((a 5)) (+ a 1))").unwrap();
        let expr = translate_ast(&ast, &ctx).unwrap();

        assert!(matches!(expr, Lean5Expr::Let(_, _, _, _)));
    }

    #[test]
    fn test_translate_forall() {
        let ctx = TranslationContext::new();
        let ast = crate::smt_parser::parse_smt_formula("(forall ((x Int)) (>= x 0))").unwrap();
        let expr = translate_ast(&ast, &ctx).unwrap();

        assert!(matches!(expr, Lean5Expr::Forall(_, _, _)));
    }

    #[test]
    fn test_translate_predicate() {
        let pred = SolvedPredicate {
            name: "Inv".to_string(),
            params: vec![("x".to_string(), SmtType::Int)],
            formula: StateFormula::new("(>= x 0)"),
        };

        let expr = translate_predicate(&pred).unwrap();

        // Should be ∀ (x : Int), (x ≥ 0)
        match expr {
            Lean5Expr::Forall(name, ty, body) => {
                assert_eq!(name, "x");
                assert_eq!(ty, Lean5Type::Int);
                assert!(matches!(*body, Lean5Expr::Ge(_, _)));
            }
            _ => panic!("Expected Forall, got {:?}", expr),
        }
    }

    #[test]
    fn test_translate_arithmetic() {
        let ctx = TranslationContext::new();

        let ast = crate::smt_parser::parse_smt_formula("(+ x (* 2 y))").unwrap();
        let expr = translate_ast(&ast, &ctx).unwrap();
        assert!(matches!(expr, Lean5Expr::Add(_, _)));

        let ast = crate::smt_parser::parse_smt_formula("(- x 1)").unwrap();
        let expr = translate_ast(&ast, &ctx).unwrap();
        assert!(matches!(expr, Lean5Expr::Sub(_, _)));
    }

    #[test]
    fn test_clean_var_name() {
        assert_eq!(clean_var_name("x!0"), "x_0");
        assert_eq!(clean_var_name("x"), "x");
        assert_eq!(clean_var_name("foo!bar!baz"), "foo_bar_baz");
    }

    #[test]
    fn test_translate_ite() {
        let ctx = TranslationContext::new();
        let ast = crate::smt_parser::parse_smt_formula("(ite (> x 0) 1 0)").unwrap();
        let expr = translate_ast(&ast, &ctx).unwrap();

        assert!(matches!(expr, Lean5Expr::Ite(_, _, _)));
    }

    #[test]
    fn test_translate_let_binding_structure() {
        let ctx = TranslationContext::new();
        let ast = crate::smt_parser::parse_smt_formula("(let ((a 5)) (+ a 1))").unwrap();
        let expr = translate_ast(&ast, &ctx).unwrap();

        match expr {
            Lean5Expr::Let(name, ty, val, body) => {
                assert_eq!(name, "a");
                assert_eq!(ty, Lean5Type::Int);
                assert!(matches!(*val, Lean5Expr::IntLit(5)));
                match *body {
                    Lean5Expr::Add(lhs, rhs) => {
                        // The let binding is substituted into the body, so we expect the literal value
                        assert!(matches!(*lhs, Lean5Expr::IntLit(5)));
                        assert!(matches!(*rhs, Lean5Expr::IntLit(1)));
                    }
                    other => panic!("expected add in let body, got {other:?}"),
                }
            }
            other => panic!("expected Let expression, got {other:?}"),
        }
    }

    #[test]
    fn test_translate_unknown_operator_as_function_application() {
        let ctx = TranslationContext::new();
        let ast = crate::smt_parser::parse_smt_formula("(custom_op x 1)").unwrap();
        let expr = translate_ast(&ast, &ctx).unwrap();

        match expr {
            Lean5Expr::App(func, arg) => {
                assert!(matches!(*func, Lean5Expr::App(_, _)));
                assert!(matches!(*arg, Lean5Expr::IntLit(1)));
                assert!(format!("{}", func).contains("custom_op"));
            }
            other => panic!("expected nested application, got {other:?}"),
        }
    }

    #[test]
    fn test_translate_ite_invalid_arity() {
        let ctx = TranslationContext::new();
        let ast = crate::smt_parser::parse_smt_formula("(ite x 1)").unwrap();
        let err = translate_ast(&ast, &ctx).unwrap_err();
        assert!(matches!(err, TranslationError::InvalidStructure(_)));
    }

    #[test]
    fn test_translate_bitvector_parameter() {
        let pred = SolvedPredicate {
            name: "Inv".to_string(),
            params: vec![("bv".to_string(), SmtType::BitVec(8))],
            formula: StateFormula::new("(= bv bv)"),
        };

        let expr = translate_predicate(&pred).unwrap();
        match expr {
            Lean5Expr::Forall(name, ty, body) => {
                assert_eq!(name, "bv");
                assert_eq!(ty, Lean5Type::Const(Lean5Name::simple("BitVec")));
                assert!(matches!(*body, Lean5Expr::Eq(_, _)));
            }
            other => panic!("expected forall, got {other:?}"),
        }
    }

    // ========================================================================
    // Mutation coverage tests
    // ========================================================================

    #[test]
    fn test_translate_app_empty_and() {
        // Mutation: delete if lean_args.is_empty() check for "and"
        let ctx = TranslationContext::new();
        // Empty and returns true
        let ast = SmtAst::App("and".to_string(), vec![]);
        let expr = translate_ast(&ast, &ctx).unwrap();
        assert!(matches!(expr, Lean5Expr::BoolLit(true)));
    }

    #[test]
    fn test_translate_app_empty_or() {
        // Mutation: delete if lean_args.is_empty() check for "or"
        let ctx = TranslationContext::new();
        // Empty or returns false
        let ast = SmtAst::App("or".to_string(), vec![]);
        let expr = translate_ast(&ast, &ctx).unwrap();
        assert!(matches!(expr, Lean5Expr::BoolLit(false)));
    }

    #[test]
    fn test_translate_app_empty_plus() {
        // Mutation: delete if lean_args.is_empty() check for "+"
        let ctx = TranslationContext::new();
        let ast = SmtAst::App("+".to_string(), vec![]);
        let expr = translate_ast(&ast, &ctx).unwrap();
        assert!(matches!(expr, Lean5Expr::IntLit(0)));
    }

    #[test]
    fn test_translate_app_empty_mul() {
        // Mutation: delete if lean_args.is_empty() check for "*"
        let ctx = TranslationContext::new();
        let ast = SmtAst::App("*".to_string(), vec![]);
        let expr = translate_ast(&ast, &ctx).unwrap();
        assert!(matches!(expr, Lean5Expr::IntLit(1)));
    }

    #[test]
    fn test_translate_app_unary_minus() {
        // Mutation: delete if lean_args.len() == 1 check for "-"
        let ctx = TranslationContext::new();
        let ast = crate::smt_parser::parse_smt_formula("(- x)").unwrap();
        let expr = translate_ast(&ast, &ctx).unwrap();
        assert!(matches!(expr, Lean5Expr::Neg(_)));
    }

    #[test]
    fn test_translate_app_binary_minus() {
        // Different from unary minus
        let ctx = TranslationContext::new();
        let ast = crate::smt_parser::parse_smt_formula("(- x y)").unwrap();
        let expr = translate_ast(&ast, &ctx).unwrap();
        assert!(matches!(expr, Lean5Expr::Sub(_, _)));
    }

    #[test]
    fn test_translate_app_not_arity_check() {
        // Mutation: delete if lean_args.len() != 1 check for "not"
        let ctx = TranslationContext::new();
        let ast = SmtAst::App(
            "not".to_string(),
            vec![SmtAst::Bool(true), SmtAst::Bool(false)],
        );
        let err = translate_ast(&ast, &ctx).unwrap_err();
        assert!(matches!(err, TranslationError::InvalidStructure(_)));
    }

    #[test]
    fn test_translate_app_implies_arity_check() {
        // Mutation: delete if lean_args.len() != 2 check for "=>"
        let ctx = TranslationContext::new();
        let ast = SmtAst::App("=>".to_string(), vec![SmtAst::Bool(true)]);
        let err = translate_ast(&ast, &ctx).unwrap_err();
        assert!(matches!(err, TranslationError::InvalidStructure(_)));
    }

    #[test]
    fn test_translate_app_eq_arity_check() {
        // Mutation: delete if lean_args.len() != 2 check for "="
        let ctx = TranslationContext::new();
        let ast = SmtAst::App("=".to_string(), vec![SmtAst::Int(1)]);
        let err = translate_ast(&ast, &ctx).unwrap_err();
        assert!(matches!(err, TranslationError::InvalidStructure(_)));
    }

    #[test]
    fn test_translate_app_lt_arity_check() {
        // Mutation: delete if lean_args.len() != 2 check for "<"
        let ctx = TranslationContext::new();
        let ast = SmtAst::App("<".to_string(), vec![SmtAst::Int(1)]);
        let err = translate_ast(&ast, &ctx).unwrap_err();
        assert!(matches!(err, TranslationError::InvalidStructure(_)));
    }

    #[test]
    fn test_translate_app_le_arity_check() {
        // Mutation: delete if lean_args.len() != 2 check for "<="
        let ctx = TranslationContext::new();
        let ast = SmtAst::App("<=".to_string(), vec![SmtAst::Int(1)]);
        let err = translate_ast(&ast, &ctx).unwrap_err();
        assert!(matches!(err, TranslationError::InvalidStructure(_)));
    }

    #[test]
    fn test_translate_app_gt_arity_check() {
        // Mutation: delete if lean_args.len() != 2 check for ">"
        let ctx = TranslationContext::new();
        let ast = SmtAst::App(">".to_string(), vec![SmtAst::Int(1)]);
        let err = translate_ast(&ast, &ctx).unwrap_err();
        assert!(matches!(err, TranslationError::InvalidStructure(_)));
    }

    #[test]
    fn test_translate_app_ge_arity_check() {
        // Mutation: delete if lean_args.len() != 2 check for ">="
        let ctx = TranslationContext::new();
        let ast = SmtAst::App(">=".to_string(), vec![SmtAst::Int(1)]);
        let err = translate_ast(&ast, &ctx).unwrap_err();
        assert!(matches!(err, TranslationError::InvalidStructure(_)));
    }

    #[test]
    fn test_translate_app_minus_empty_check() {
        // Mutation: delete if lean_args.is_empty() check for "-"
        let ctx = TranslationContext::new();
        let ast = SmtAst::App("-".to_string(), vec![]);
        let err = translate_ast(&ast, &ctx).unwrap_err();
        assert!(matches!(err, TranslationError::InvalidStructure(_)));
    }

    #[test]
    fn test_translate_app_div_arity_check() {
        // Mutation: delete if lean_args.len() != 2 check for "/" and "div"
        let ctx = TranslationContext::new();
        let ast = SmtAst::App("div".to_string(), vec![SmtAst::Int(1)]);
        let err = translate_ast(&ast, &ctx).unwrap_err();
        assert!(matches!(err, TranslationError::InvalidStructure(_)));
    }

    #[test]
    fn test_translate_app_mod_arity_check() {
        // Mutation: delete if lean_args.len() != 2 check for "mod"
        let ctx = TranslationContext::new();
        let ast = SmtAst::App("mod".to_string(), vec![SmtAst::Int(1)]);
        let err = translate_ast(&ast, &ctx).unwrap_err();
        assert!(matches!(err, TranslationError::InvalidStructure(_)));
    }

    #[test]
    fn test_translate_ast_neg_with_inner_int() {
        // Mutation: delete if let SmtAst::Int(n) check in Neg handling
        let ctx = TranslationContext::new();
        // (- 5) should produce IntLit(-5)
        let ast = SmtAst::Neg(Box::new(SmtAst::Int(5)));
        let expr = translate_ast(&ast, &ctx).unwrap();
        assert!(matches!(expr, Lean5Expr::IntLit(-5)));
    }

    #[test]
    fn test_translate_ast_symbol_let_bound() {
        // Mutation: delete if let Some(binding) check in Symbol handling
        let mut ctx = TranslationContext::new();
        ctx.add_let("x", Lean5Expr::IntLit(42));
        let ast = SmtAst::Symbol("x".to_string());
        let expr = translate_ast(&ast, &ctx).unwrap();
        assert!(matches!(expr, Lean5Expr::IntLit(42)));
    }

    #[test]
    fn test_translate_ast_symbol_variable() {
        // Mutation: delete ctx.var_types.contains_key check or s.contains('!') check
        let mut ctx = TranslationContext::new();
        ctx.add_var("x", Lean5Type::Int);
        let ast = SmtAst::Symbol("x".to_string());
        let expr = translate_ast(&ast, &ctx).unwrap();
        assert!(matches!(expr, Lean5Expr::Var(_)));
    }

    #[test]
    fn test_translate_ast_symbol_z3_generated_name() {
        // Mutation: delete s.contains('!') check
        let ctx = TranslationContext::new();
        // Z3-generated names like x!0 should be treated as variables
        let ast = SmtAst::Symbol("x!0".to_string());
        let expr = translate_ast(&ast, &ctx).unwrap();
        // Should produce variable (with ! replaced by _)
        assert!(matches!(expr, Lean5Expr::Var(_)));
        if let Lean5Expr::Var(name) = expr {
            assert_eq!(name, "x_0"); // ! replaced by _
        }
    }

    #[test]
    fn test_translate_ast_symbol_constant() {
        // If not let-bound and not a variable, treat as constant
        let ctx = TranslationContext::new();
        let ast = SmtAst::Symbol("MY_CONSTANT".to_string());
        let expr = translate_ast(&ast, &ctx).unwrap();
        assert!(matches!(expr, Lean5Expr::Const(_)));
    }

    #[test]
    fn test_smt_type_to_lean5_all_types() {
        // Mutation: delete match arms for different SmtType variants
        assert_eq!(smt_type_to_lean5(&SmtType::Int), Lean5Type::Int);
        assert_eq!(smt_type_to_lean5(&SmtType::Bool), Lean5Type::Bool);
        assert_eq!(
            smt_type_to_lean5(&SmtType::Real),
            Lean5Type::Const(Lean5Name::simple("Real"))
        );
        assert_eq!(
            smt_type_to_lean5(&SmtType::BitVec(32)),
            Lean5Type::Const(Lean5Name::simple("BitVec"))
        );
        assert_eq!(
            smt_type_to_lean5(&SmtType::Array {
                index: Box::new(SmtType::Int),
                element: Box::new(SmtType::Int)
            }),
            Lean5Type::Const(Lean5Name::simple("Array"))
        );
    }

    #[test]
    fn test_smt_sort_to_lean5_all_sorts() {
        // Mutation: delete match arms for different SmtSort variants
        use crate::smt_parser::SmtSort;
        assert_eq!(smt_sort_to_lean5(&SmtSort::Int), Lean5Type::Int);
        assert_eq!(smt_sort_to_lean5(&SmtSort::Bool), Lean5Type::Bool);
        assert_eq!(
            smt_sort_to_lean5(&SmtSort::Real),
            Lean5Type::Const(Lean5Name::simple("Real"))
        );
        assert_eq!(
            smt_sort_to_lean5(&SmtSort::BitVec(64)),
            Lean5Type::Const(Lean5Name::simple("BitVec"))
        );
        assert_eq!(
            smt_sort_to_lean5(&SmtSort::Array(
                Box::new(SmtSort::Int),
                Box::new(SmtSort::Int)
            )),
            Lean5Type::Const(Lean5Name::simple("Array"))
        );
        assert_eq!(
            smt_sort_to_lean5(&SmtSort::Unknown("CustomSort".to_string())),
            Lean5Type::Const(Lean5Name::simple("CustomSort"))
        );
    }

    #[test]
    fn test_infer_type_all_branches() {
        // Mutation: delete match arms for different Lean5Expr variants

        // Int literal
        assert_eq!(infer_type(&Lean5Expr::IntLit(5)), Lean5Type::Int);

        // Nat literal
        assert_eq!(infer_type(&Lean5Expr::NatLit(5)), Lean5Type::Nat);

        // Bool literal
        assert_eq!(infer_type(&Lean5Expr::BoolLit(true)), Lean5Type::Bool);

        // Logical operators return Prop
        assert_eq!(
            infer_type(&Lean5Expr::and(
                Lean5Expr::BoolLit(true),
                Lean5Expr::BoolLit(false)
            )),
            Lean5Type::Prop
        );
        assert_eq!(
            infer_type(&Lean5Expr::or(
                Lean5Expr::BoolLit(true),
                Lean5Expr::BoolLit(false)
            )),
            Lean5Type::Prop
        );
        assert_eq!(
            infer_type(&Lean5Expr::not(Lean5Expr::BoolLit(true))),
            Lean5Type::Prop
        );
        assert_eq!(
            infer_type(&Lean5Expr::implies(
                Lean5Expr::BoolLit(true),
                Lean5Expr::BoolLit(false)
            )),
            Lean5Type::Prop
        );

        // Comparisons return Prop
        assert_eq!(
            infer_type(&Lean5Expr::eq(Lean5Expr::IntLit(1), Lean5Expr::IntLit(2))),
            Lean5Type::Prop
        );
        assert_eq!(
            infer_type(&Lean5Expr::lt(Lean5Expr::IntLit(1), Lean5Expr::IntLit(2))),
            Lean5Type::Prop
        );
        assert_eq!(
            infer_type(&Lean5Expr::le(Lean5Expr::IntLit(1), Lean5Expr::IntLit(2))),
            Lean5Type::Prop
        );
        assert_eq!(
            infer_type(&Lean5Expr::gt(Lean5Expr::IntLit(1), Lean5Expr::IntLit(2))),
            Lean5Type::Prop
        );
        assert_eq!(
            infer_type(&Lean5Expr::ge(Lean5Expr::IntLit(1), Lean5Expr::IntLit(2))),
            Lean5Type::Prop
        );

        // Arithmetic operations return Int
        assert_eq!(
            infer_type(&Lean5Expr::add(Lean5Expr::IntLit(1), Lean5Expr::IntLit(2))),
            Lean5Type::Int
        );
        assert_eq!(
            infer_type(&Lean5Expr::sub(Lean5Expr::IntLit(1), Lean5Expr::IntLit(2))),
            Lean5Type::Int
        );
        assert_eq!(
            infer_type(&Lean5Expr::mul(Lean5Expr::IntLit(1), Lean5Expr::IntLit(2))),
            Lean5Type::Int
        );
        assert_eq!(
            infer_type(&Lean5Expr::Div(
                Box::new(Lean5Expr::IntLit(1)),
                Box::new(Lean5Expr::IntLit(2))
            )),
            Lean5Type::Int
        );
        assert_eq!(
            infer_type(&Lean5Expr::Mod(
                Box::new(Lean5Expr::IntLit(1)),
                Box::new(Lean5Expr::IntLit(2))
            )),
            Lean5Type::Int
        );
        assert_eq!(
            infer_type(&Lean5Expr::Neg(Box::new(Lean5Expr::IntLit(1)))),
            Lean5Type::Int
        );

        // Quantifiers return Prop
        assert_eq!(
            infer_type(&Lean5Expr::forall_(
                "x",
                Lean5Type::Int,
                Lean5Expr::BoolLit(true)
            )),
            Lean5Type::Prop
        );
        assert_eq!(
            infer_type(&Lean5Expr::exists_(
                "x",
                Lean5Type::Int,
                Lean5Expr::BoolLit(true)
            )),
            Lean5Type::Prop
        );

        // Ite infers type from then branch
        assert_eq!(
            infer_type(&Lean5Expr::ite(
                Lean5Expr::BoolLit(true),
                Lean5Expr::IntLit(1),
                Lean5Expr::IntLit(0)
            )),
            Lean5Type::Int
        );

        // Ascribe returns the ascribed type
        assert_eq!(
            infer_type(&Lean5Expr::Ascribe(
                Box::new(Lean5Expr::var("x")),
                Lean5Type::Nat
            )),
            Lean5Type::Nat
        );

        // Default fallback
        assert_eq!(infer_type(&Lean5Expr::var("x")), Lean5Type::Type);
    }

    #[test]
    fn test_translate_ast_forall() {
        // Mutation: delete match arm for SmtAst::Forall
        let ctx = TranslationContext::new();
        let ast = SmtAst::Forall(
            vec![("x".to_string(), crate::smt_parser::SmtSort::Int)],
            Box::new(SmtAst::App(
                ">=".to_string(),
                vec![SmtAst::Symbol("x".to_string()), SmtAst::Int(0)],
            )),
        );
        let expr = translate_ast(&ast, &ctx).unwrap();
        assert!(matches!(expr, Lean5Expr::Forall(_, _, _)));
    }

    #[test]
    fn test_translate_ast_exists() {
        // Mutation: delete match arm for SmtAst::Exists
        let ctx = TranslationContext::new();
        let ast = SmtAst::Exists(
            vec![("x".to_string(), crate::smt_parser::SmtSort::Int)],
            Box::new(SmtAst::App(
                ">=".to_string(),
                vec![SmtAst::Symbol("x".to_string()), SmtAst::Int(0)],
            )),
        );
        let expr = translate_ast(&ast, &ctx).unwrap();
        assert!(matches!(expr, Lean5Expr::Exists(_, _, _)));
    }

    // ========================================================================
    // TranslationContext mutation coverage tests
    // ========================================================================

    #[test]
    fn test_translation_context_get_var_type_found() {
        // Mutation: replace get_var_type -> Option<&Lean5Type> with None
        let mut ctx = TranslationContext::new();
        ctx.add_var("x", Lean5Type::Int);
        ctx.add_var("flag", Lean5Type::Bool);

        // Must return Some for existing variables
        let x_type = ctx.get_var_type("x");
        assert!(x_type.is_some(), "get_var_type should return Some for 'x'");
        assert_eq!(*x_type.unwrap(), Lean5Type::Int);

        let flag_type = ctx.get_var_type("flag");
        assert!(
            flag_type.is_some(),
            "get_var_type should return Some for 'flag'"
        );
        assert_eq!(*flag_type.unwrap(), Lean5Type::Bool);
    }

    #[test]
    fn test_translation_context_get_var_type_not_found() {
        // Complementary test: None for non-existent variables
        let ctx = TranslationContext::new();
        assert!(
            ctx.get_var_type("nonexistent").is_none(),
            "get_var_type should return None for non-existent variable"
        );
    }

    #[test]
    fn test_translation_context_is_let_bound_true() {
        // Mutation: replace is_let_bound -> bool with false
        let mut ctx = TranslationContext::new();
        ctx.add_let("bound_var", Lean5Expr::IntLit(42));

        assert!(
            ctx.is_let_bound("bound_var"),
            "is_let_bound should return true for 'bound_var'"
        );
    }

    #[test]
    fn test_translation_context_is_let_bound_false() {
        // Mutation: replace is_let_bound -> bool with true
        let ctx = TranslationContext::new();

        assert!(
            !ctx.is_let_bound("unbound_var"),
            "is_let_bound should return false for non-existent variable"
        );
    }

    #[test]
    fn test_translation_context_is_let_bound_multiple() {
        // Test with multiple let bindings
        let mut ctx = TranslationContext::new();
        ctx.add_let("a", Lean5Expr::IntLit(1));
        ctx.add_let("b", Lean5Expr::BoolLit(true));

        assert!(ctx.is_let_bound("a"));
        assert!(ctx.is_let_bound("b"));
        assert!(!ctx.is_let_bound("c"));
    }
}
