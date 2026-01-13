//! Model path extraction from USL specifications

use dashprove_usl::ast::{Expr, Property};
use dashprove_usl::typecheck::TypedSpec;

/// Extract ONNX model path from USL specification
///
/// Searches for model paths in:
/// 1. String literals ending in `.onnx` or `.pb`
/// 2. Type fields named "model", "network", or "onnx"
/// 3. Function applications like `model("path.onnx")`
pub fn extract_model_path(spec: &TypedSpec) -> Option<String> {
    // Search in properties
    for prop in &spec.spec.properties {
        let expr = match prop {
            Property::Theorem(t) => Some(&t.body),
            Property::Invariant(inv) => Some(&inv.body),
            Property::Contract(c) => {
                // Check requires and ensures
                for req in &c.requires {
                    if let Some(path) = find_model_path_in_expr(req) {
                        return Some(path);
                    }
                }
                for ens in &c.ensures {
                    if let Some(path) = find_model_path_in_expr(ens) {
                        return Some(path);
                    }
                }
                None
            }
            Property::Security(s) => Some(&s.body),
            _ => None,
        };

        if let Some(e) = expr {
            if let Some(path) = find_model_path_in_expr(e) {
                return Some(path);
            }
        }
    }

    // Search in type definitions for fields named "model" or similar
    for typedef in &spec.spec.types {
        for field in &typedef.fields {
            let lower = field.name.to_lowercase();
            if lower.contains("model") || lower.contains("network") || lower.contains("onnx") {
                // Check if the type name looks like a path
                if let dashprove_usl::ast::Type::Named(type_name) = &field.ty {
                    if type_name.ends_with(".onnx") || type_name.ends_with(".pb") {
                        return Some(type_name.clone());
                    }
                }
            }
        }
    }

    None
}

/// Recursively search for model paths in an expression
fn find_model_path_in_expr(expr: &Expr) -> Option<String> {
    match expr {
        Expr::String(s) => {
            if s.ends_with(".onnx") || s.ends_with(".pb") {
                return Some(s.clone());
            }
            None
        }
        Expr::App(func, args) => {
            let lower = func.to_lowercase();
            if lower.contains("model") || lower.contains("network") || lower.contains("load") {
                // First string arg might be path
                for arg in args {
                    if let Expr::String(s) = arg {
                        if s.ends_with(".onnx") || s.ends_with(".pb") || s.contains('/') {
                            return Some(s.clone());
                        }
                    }
                }
            }
            // Recurse into arguments
            for arg in args {
                if let Some(path) = find_model_path_in_expr(arg) {
                    return Some(path);
                }
            }
            None
        }
        Expr::Compare(lhs, _, rhs) => {
            // Check for assignments like `model == "path.onnx"`
            if let Expr::Var(var) = lhs.as_ref() {
                let lower = var.to_lowercase();
                if lower.contains("model") || lower.contains("network") {
                    if let Expr::String(s) = rhs.as_ref() {
                        if s.ends_with(".onnx") || s.ends_with(".pb") || s.contains('/') {
                            return Some(s.clone());
                        }
                    }
                }
            }
            find_model_path_in_expr(lhs).or_else(|| find_model_path_in_expr(rhs))
        }
        Expr::And(lhs, rhs) | Expr::Or(lhs, rhs) | Expr::Implies(lhs, rhs) => {
            find_model_path_in_expr(lhs).or_else(|| find_model_path_in_expr(rhs))
        }
        Expr::Not(inner) | Expr::Neg(inner) => find_model_path_in_expr(inner),
        Expr::Binary(lhs, _, rhs) => {
            find_model_path_in_expr(lhs).or_else(|| find_model_path_in_expr(rhs))
        }
        Expr::ForAll { body, .. }
        | Expr::Exists { body, .. }
        | Expr::ForAllIn { body, .. }
        | Expr::ExistsIn { body, .. } => find_model_path_in_expr(body),
        Expr::FieldAccess(obj, _) => find_model_path_in_expr(obj),
        Expr::MethodCall { receiver, args, .. } => {
            find_model_path_in_expr(receiver).or_else(|| {
                for arg in args {
                    if let Some(path) = find_model_path_in_expr(arg) {
                        return Some(path);
                    }
                }
                None
            })
        }
        _ => None,
    }
}
