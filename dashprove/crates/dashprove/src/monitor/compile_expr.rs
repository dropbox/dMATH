//! Expression compilation for different target languages

use crate::monitor::operators::{
    binary_op_to_python, binary_op_to_rust, binary_op_to_ts, comparison_op_to_python,
    comparison_op_to_rust, comparison_op_to_ts,
};
use crate::monitor::types::{type_to_python_iter, type_to_rust_iter, type_to_ts_iter};
use dashprove_usl::ast::{Expr, TemporalExpr};
use std::collections::HashMap;

// =============================================================================
// Rust Expression Compilation
// =============================================================================

/// Compile a USL expression to Rust code
pub fn compile_expr_to_rust(expr: &Expr) -> String {
    compile_expr_to_rust_with_env(expr, &HashMap::new())
}

pub fn compile_expr_to_rust_with_env(expr: &Expr, env: &HashMap<String, String>) -> String {
    match expr {
        Expr::Var(name) => env.get(name).cloned().unwrap_or_else(|| name.clone()),
        Expr::Int(n) => n.to_string(),
        Expr::Float(f) => format!("{f:?}"),
        Expr::String(s) => format!("{s:?}"),
        Expr::Bool(b) => b.to_string(),
        Expr::ForAll { var, ty, body } => {
            let domain = type_to_rust_iter(ty.as_ref());
            let body_code = compile_expr_to_rust_with_env(body, env);
            format!("{domain}.all(|{var}| {body_code})")
        }
        Expr::Exists { var, ty, body } => {
            let domain = type_to_rust_iter(ty.as_ref());
            let body_code = compile_expr_to_rust_with_env(body, env);
            format!("{domain}.any(|{var}| {body_code})")
        }
        Expr::ForAllIn {
            var,
            collection,
            body,
        } => {
            let coll = compile_expr_to_rust_with_env(collection, env);
            let body_code = compile_expr_to_rust_with_env(body, env);
            format!("{coll}.iter().all(|{var}| {body_code})")
        }
        Expr::ExistsIn {
            var,
            collection,
            body,
        } => {
            let coll = compile_expr_to_rust_with_env(collection, env);
            let body_code = compile_expr_to_rust_with_env(body, env);
            format!("{coll}.iter().any(|{var}| {body_code})")
        }
        Expr::Implies(a, b) => {
            let a_code = compile_expr_to_rust_with_env(a, env);
            let b_code = compile_expr_to_rust_with_env(b, env);
            format!("(!({a_code}) || ({b_code}))")
        }
        Expr::And(a, b) => {
            let a_code = compile_expr_to_rust_with_env(a, env);
            let b_code = compile_expr_to_rust_with_env(b, env);
            format!("({a_code} && {b_code})")
        }
        Expr::Or(a, b) => {
            let a_code = compile_expr_to_rust_with_env(a, env);
            let b_code = compile_expr_to_rust_with_env(b, env);
            format!("({a_code} || {b_code})")
        }
        Expr::Not(a) => {
            let a_code = compile_expr_to_rust_with_env(a, env);
            format!("!({a_code})")
        }
        Expr::Compare(a, op, b) => {
            let a_code = compile_expr_to_rust_with_env(a, env);
            let b_code = compile_expr_to_rust_with_env(b, env);
            let op_str = comparison_op_to_rust(op);
            format!("({a_code} {op_str} {b_code})")
        }
        Expr::Binary(a, op, b) => {
            let a_code = compile_expr_to_rust_with_env(a, env);
            let b_code = compile_expr_to_rust_with_env(b, env);
            let op_str = binary_op_to_rust(op);
            format!("({a_code} {op_str} {b_code})")
        }
        Expr::Neg(a) => {
            let a_code = compile_expr_to_rust_with_env(a, env);
            format!("-({a_code})")
        }
        Expr::App(name, args) => {
            let args_code: Vec<_> = args
                .iter()
                .map(|arg| compile_expr_to_rust_with_env(arg, env))
                .collect();
            format!("{}({})", name, args_code.join(", "))
        }
        Expr::MethodCall {
            receiver,
            method,
            args,
        } => {
            let recv = compile_expr_to_rust_with_env(receiver, env);
            let args_code: Vec<_> = args
                .iter()
                .map(|arg| compile_expr_to_rust_with_env(arg, env))
                .collect();
            format!("{recv}.{method}({})", args_code.join(", "))
        }
        Expr::FieldAccess(obj, field) => {
            let obj_code = compile_expr_to_rust_with_env(obj, env);
            format!("{obj_code}.{field}")
        }
    }
}

// =============================================================================
// TypeScript Expression Compilation
// =============================================================================

/// Compile a USL expression to TypeScript code
pub fn compile_expr_to_typescript(expr: &Expr) -> String {
    compile_expr_to_typescript_with_env(expr, &HashMap::new())
}

pub fn compile_expr_to_typescript_with_env(expr: &Expr, env: &HashMap<String, String>) -> String {
    match expr {
        Expr::Var(name) => env.get(name).cloned().unwrap_or_else(|| name.clone()),
        Expr::Int(n) => n.to_string(),
        Expr::Float(f) => format!("{f:?}"),
        Expr::String(s) => format!("{s:?}"),
        Expr::Bool(b) => b.to_string(),
        Expr::ForAll { var, ty, body } => {
            let domain = type_to_ts_iter(ty.as_ref());
            let body_code = compile_expr_to_typescript_with_env(body, env);
            format!("{domain}.every(({var}) => {body_code})")
        }
        Expr::Exists { var, ty, body } => {
            let domain = type_to_ts_iter(ty.as_ref());
            let body_code = compile_expr_to_typescript_with_env(body, env);
            format!("{domain}.some(({var}) => {body_code})")
        }
        Expr::ForAllIn {
            var,
            collection,
            body,
        } => {
            let coll = compile_expr_to_typescript_with_env(collection, env);
            let body_code = compile_expr_to_typescript_with_env(body, env);
            format!("{coll}.every(({var}) => {body_code})")
        }
        Expr::ExistsIn {
            var,
            collection,
            body,
        } => {
            let coll = compile_expr_to_typescript_with_env(collection, env);
            let body_code = compile_expr_to_typescript_with_env(body, env);
            format!("{coll}.some(({var}) => {body_code})")
        }
        Expr::Implies(a, b) => {
            let a_code = compile_expr_to_typescript_with_env(a, env);
            let b_code = compile_expr_to_typescript_with_env(b, env);
            format!("(!({a_code}) || ({b_code}))")
        }
        Expr::And(a, b) => {
            let a_code = compile_expr_to_typescript_with_env(a, env);
            let b_code = compile_expr_to_typescript_with_env(b, env);
            format!("({a_code} && {b_code})")
        }
        Expr::Or(a, b) => {
            let a_code = compile_expr_to_typescript_with_env(a, env);
            let b_code = compile_expr_to_typescript_with_env(b, env);
            format!("({a_code} || {b_code})")
        }
        Expr::Not(a) => {
            let a_code = compile_expr_to_typescript_with_env(a, env);
            format!("!({a_code})")
        }
        Expr::Compare(a, op, b) => {
            let a_code = compile_expr_to_typescript_with_env(a, env);
            let b_code = compile_expr_to_typescript_with_env(b, env);
            let op_str = comparison_op_to_ts(op);
            format!("({a_code} {op_str} {b_code})")
        }
        Expr::Binary(a, op, b) => {
            let a_code = compile_expr_to_typescript_with_env(a, env);
            let b_code = compile_expr_to_typescript_with_env(b, env);
            let op_str = binary_op_to_ts(op);
            format!("({a_code} {op_str} {b_code})")
        }
        Expr::Neg(a) => {
            let a_code = compile_expr_to_typescript_with_env(a, env);
            format!("-({a_code})")
        }
        Expr::App(name, args) => {
            let args_code: Vec<_> = args
                .iter()
                .map(|arg| compile_expr_to_typescript_with_env(arg, env))
                .collect();
            format!("{}({})", name, args_code.join(", "))
        }
        Expr::MethodCall {
            receiver,
            method,
            args,
        } => {
            let recv = compile_expr_to_typescript_with_env(receiver, env);
            let args_code: Vec<_> = args
                .iter()
                .map(|arg| compile_expr_to_typescript_with_env(arg, env))
                .collect();
            format!("{recv}.{method}({})", args_code.join(", "))
        }
        Expr::FieldAccess(obj, field) => {
            let obj_code = compile_expr_to_typescript_with_env(obj, env);
            format!("{obj_code}.{field}")
        }
    }
}

// =============================================================================
// Python Expression Compilation
// =============================================================================

/// Compile a USL expression to Python code
pub fn compile_expr_to_python(expr: &Expr) -> String {
    compile_expr_to_python_with_env(expr, &HashMap::new())
}

pub fn compile_expr_to_python_with_env(expr: &Expr, env: &HashMap<String, String>) -> String {
    match expr {
        Expr::Var(name) => env.get(name).cloned().unwrap_or_else(|| name.clone()),
        Expr::Int(n) => n.to_string(),
        Expr::Float(f) => format!("{f:?}"),
        Expr::String(s) => format!("{s:?}"),
        Expr::Bool(b) => {
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        Expr::ForAll { var, ty, body } => {
            let domain = type_to_python_iter(ty.as_ref());
            let body_code = compile_expr_to_python_with_env(body, env);
            format!("all(({body_code} for {var} in {domain}))")
        }
        Expr::Exists { var, ty, body } => {
            let domain = type_to_python_iter(ty.as_ref());
            let body_code = compile_expr_to_python_with_env(body, env);
            format!("any(({body_code} for {var} in {domain}))")
        }
        Expr::ForAllIn {
            var,
            collection,
            body,
        } => {
            let coll = compile_expr_to_python_with_env(collection, env);
            let body_code = compile_expr_to_python_with_env(body, env);
            format!("all(({body_code} for {var} in {coll}))")
        }
        Expr::ExistsIn {
            var,
            collection,
            body,
        } => {
            let coll = compile_expr_to_python_with_env(collection, env);
            let body_code = compile_expr_to_python_with_env(body, env);
            format!("any(({body_code} for {var} in {coll}))")
        }
        Expr::Implies(a, b) => {
            let a_code = compile_expr_to_python_with_env(a, env);
            let b_code = compile_expr_to_python_with_env(b, env);
            format!("(not ({a_code}) or ({b_code}))")
        }
        Expr::And(a, b) => {
            let a_code = compile_expr_to_python_with_env(a, env);
            let b_code = compile_expr_to_python_with_env(b, env);
            format!("({a_code} and {b_code})")
        }
        Expr::Or(a, b) => {
            let a_code = compile_expr_to_python_with_env(a, env);
            let b_code = compile_expr_to_python_with_env(b, env);
            format!("({a_code} or {b_code})")
        }
        Expr::Not(a) => {
            let a_code = compile_expr_to_python_with_env(a, env);
            format!("not ({a_code})")
        }
        Expr::Compare(a, op, b) => {
            let a_code = compile_expr_to_python_with_env(a, env);
            let b_code = compile_expr_to_python_with_env(b, env);
            let op_str = comparison_op_to_python(op);
            format!("({a_code} {op_str} {b_code})")
        }
        Expr::Binary(a, op, b) => {
            let a_code = compile_expr_to_python_with_env(a, env);
            let b_code = compile_expr_to_python_with_env(b, env);
            let op_str = binary_op_to_python(op);
            format!("({a_code} {op_str} {b_code})")
        }
        Expr::Neg(a) => {
            let a_code = compile_expr_to_python_with_env(a, env);
            format!("-({a_code})")
        }
        Expr::App(name, args) => {
            let args_code: Vec<_> = args
                .iter()
                .map(|arg| compile_expr_to_python_with_env(arg, env))
                .collect();
            format!("{}({})", name, args_code.join(", "))
        }
        Expr::MethodCall {
            receiver,
            method,
            args,
        } => {
            let recv = compile_expr_to_python_with_env(receiver, env);
            let args_code: Vec<_> = args
                .iter()
                .map(|arg| compile_expr_to_python_with_env(arg, env))
                .collect();
            format!("{recv}.{method}({})", args_code.join(", "))
        }
        Expr::FieldAccess(obj, field) => {
            let obj_code = compile_expr_to_python_with_env(obj, env);
            format!("{obj_code}.{field}")
        }
    }
}

// =============================================================================
// Temporal Expression Compilation
// =============================================================================

/// Compile a temporal expression to Rust (for runtime checks, temporal ops become comments)
pub fn compile_temporal_to_rust(expr: &TemporalExpr) -> String {
    match expr {
        TemporalExpr::Atom(e) => compile_expr_to_rust(e),
        TemporalExpr::Always(inner) => {
            let inner_code = compile_temporal_to_rust(inner);
            format!("/* always */ {inner_code}")
        }
        TemporalExpr::Eventually(inner) => {
            let inner_code = compile_temporal_to_rust(inner);
            format!("/* eventually */ {inner_code}")
        }
        TemporalExpr::LeadsTo(a, b) => {
            let a_code = compile_temporal_to_rust(a);
            let b_code = compile_temporal_to_rust(b);
            format!("/* {a_code} ~> {b_code} */ true")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dashprove_usl::ast::{BinaryOp, ComparisonOp, Type};

    #[test]
    fn test_compile_simple_bool_rust() {
        let expr = Expr::Bool(true);
        assert_eq!(compile_expr_to_rust(&expr), "true");

        let expr = Expr::Bool(false);
        assert_eq!(compile_expr_to_rust(&expr), "false");
    }

    #[test]
    fn test_compile_arithmetic_rust() {
        let expr = Expr::Binary(
            Box::new(Expr::Int(2)),
            BinaryOp::Add,
            Box::new(Expr::Int(3)),
        );
        assert_eq!(compile_expr_to_rust(&expr), "(2 + 3)");

        let expr = Expr::Binary(
            Box::new(Expr::Int(10)),
            BinaryOp::Mul,
            Box::new(Expr::Int(5)),
        );
        assert_eq!(compile_expr_to_rust(&expr), "(10 * 5)");
    }

    #[test]
    fn test_compile_comparison_rust() {
        let expr = Expr::Compare(
            Box::new(Expr::Var("x".to_string())),
            ComparisonOp::Gt,
            Box::new(Expr::Int(0)),
        );
        assert_eq!(compile_expr_to_rust(&expr), "(x > 0)");
    }

    #[test]
    fn test_compile_logical_rust() {
        let expr = Expr::And(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(false)));
        assert_eq!(compile_expr_to_rust(&expr), "(true && false)");

        let expr = Expr::Or(
            Box::new(Expr::Var("a".to_string())),
            Box::new(Expr::Var("b".to_string())),
        );
        assert_eq!(compile_expr_to_rust(&expr), "(a || b)");

        let expr = Expr::Not(Box::new(Expr::Bool(true)));
        assert_eq!(compile_expr_to_rust(&expr), "!(true)");
    }

    #[test]
    fn test_compile_implies_rust() {
        let expr = Expr::Implies(
            Box::new(Expr::Var("p".to_string())),
            Box::new(Expr::Var("q".to_string())),
        );
        assert_eq!(compile_expr_to_rust(&expr), "(!(p) || (q))");
    }

    #[test]
    fn test_compile_forall_bool_rust() {
        let expr = Expr::ForAll {
            var: "x".to_string(),
            ty: Some(Type::Named("Bool".to_string())),
            body: Box::new(Expr::Var("x".to_string())),
        };
        let code = compile_expr_to_rust(&expr);
        assert!(code.contains("[false, true]"));
        assert!(code.contains(".all(|x|"));
    }

    #[test]
    fn test_compile_forall_int_rust() {
        let expr = Expr::ForAll {
            var: "n".to_string(),
            ty: Some(Type::Named("Int".to_string())),
            body: Box::new(Expr::Compare(
                Box::new(Expr::Var("n".to_string())),
                ComparisonOp::Ge,
                Box::new(Expr::Int(0)),
            )),
        };
        let code = compile_expr_to_rust(&expr);
        assert!(code.contains("(-1000..=1000)"));
        assert!(code.contains(".all(|n|"));
    }

    #[test]
    fn test_compile_python_bool() {
        let expr = Expr::Bool(true);
        assert_eq!(compile_expr_to_python(&expr), "True");

        let expr = Expr::Bool(false);
        assert_eq!(compile_expr_to_python(&expr), "False");
    }

    #[test]
    fn test_compile_python_logical() {
        let expr = Expr::And(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(false)));
        assert_eq!(compile_expr_to_python(&expr), "(True and False)");

        let expr = Expr::Or(
            Box::new(Expr::Var("a".to_string())),
            Box::new(Expr::Var("b".to_string())),
        );
        assert_eq!(compile_expr_to_python(&expr), "(a or b)");

        let expr = Expr::Not(Box::new(Expr::Bool(true)));
        assert_eq!(compile_expr_to_python(&expr), "not (True)");
    }

    #[test]
    fn test_compile_python_forall() {
        let expr = Expr::ForAll {
            var: "x".to_string(),
            ty: Some(Type::Named("Bool".to_string())),
            body: Box::new(Expr::Var("x".to_string())),
        };
        let code = compile_expr_to_python(&expr);
        assert!(code.contains("all("));
        assert!(code.contains("[False, True]"));
    }

    #[test]
    fn test_compile_typescript_logical() {
        let expr = Expr::And(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(false)));
        assert_eq!(compile_expr_to_typescript(&expr), "(true && false)");
    }

    #[test]
    fn test_compile_typescript_comparison() {
        // TypeScript uses === for equality
        let expr = Expr::Compare(
            Box::new(Expr::Var("x".to_string())),
            ComparisonOp::Eq,
            Box::new(Expr::Int(5)),
        );
        assert_eq!(compile_expr_to_typescript(&expr), "(x === 5)");
    }
}
