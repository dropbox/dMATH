//! Type conversion utilities for monitor code generation

use crate::monitor::utils::{to_camel_case, to_snake_case};
use dashprove_usl::ast::Type;

/// Convert USL type to Rust type name
pub fn rust_type_name(ty: &Type) -> String {
    match ty {
        Type::Named(name) => match name.as_str() {
            "Bool" => "bool".to_string(),
            "Int" => "i64".to_string(),
            "Nat" => "u64".to_string(),
            "Float" => "f64".to_string(),
            "String" => "String".to_string(),
            other => other.to_string(),
        },
        Type::Set(inner) => format!("std::collections::BTreeSet<{}>", rust_type_name(inner)),
        Type::List(inner) => format!("Vec<{}>", rust_type_name(inner)),
        Type::Map(k, v) => format!(
            "std::collections::BTreeMap<{}, {}>",
            rust_type_name(k),
            rust_type_name(v)
        ),
        Type::Relation(a, b) => format!("Vec<({}, {})>", rust_type_name(a), rust_type_name(b)),
        Type::Function(arg, ret) => format!(
            "Box<dyn Fn({}) -> {}>",
            rust_type_name(arg),
            rust_type_name(ret)
        ),
        Type::Result(inner) => rust_type_name(inner),
        Type::Graph(n, e) => format!("Graph<{}, {}>", rust_type_name(n), rust_type_name(e)),
        Type::Path(n) => format!("Vec<{}>", rust_type_name(n)),
        Type::Unit => "()".to_string(),
    }
}

/// Convert USL type to TypeScript type name
pub fn ts_type_name(ty: &Type) -> String {
    match ty {
        Type::Named(name) => match name.as_str() {
            "Bool" => "boolean".to_string(),
            "Int" | "Nat" | "Float" => "number".to_string(),
            "String" => "string".to_string(),
            other => other.to_string(),
        },
        Type::Set(inner) => format!("Set<{}>", ts_type_name(inner)),
        Type::List(inner) => format!("Array<{}>", ts_type_name(inner)),
        Type::Map(k, v) => format!("Map<{}, {}>", ts_type_name(k), ts_type_name(v)),
        Type::Relation(a, b) => format!("Array<[{}, {}]>", ts_type_name(a), ts_type_name(b)),
        Type::Function(arg, ret) => format!("({}) => {}", ts_type_name(arg), ts_type_name(ret)),
        Type::Result(inner) => ts_type_name(inner),
        Type::Graph(n, e) => format!("Graph<{}, {}>", ts_type_name(n), ts_type_name(e)),
        Type::Path(n) => format!("Array<{}>", ts_type_name(n)),
        Type::Unit => "void".to_string(),
    }
}

/// Convert type annotation to Rust iterator (for bounded quantifiers)
pub fn type_to_rust_iter(ty: Option<&Type>) -> String {
    match ty {
        Some(Type::Named(name)) => match name.as_str() {
            "Bool" => "[false, true].iter().copied()".to_string(),
            "Int" => "(-1000..=1000)".to_string(), // bounded for runtime checks
            "Nat" => "(0..=1000)".to_string(),
            _ => format!("self.{}.iter()", to_snake_case(name)),
        },
        Some(Type::Set(inner)) => {
            format!("self.{}_set.iter()", type_name(inner))
        }
        _ => "std::iter::empty()".to_string(),
    }
}

/// Convert type annotation to TypeScript iterator
pub fn type_to_ts_iter(ty: Option<&Type>) -> String {
    match ty {
        Some(Type::Named(name)) => match name.as_str() {
            "Bool" => "[false, true]".to_string(),
            "Int" => "Array.from({ length: 2001 }, (_, i) => i - 1000)".to_string(),
            "Nat" => "Array.from({ length: 1001 }, (_, i) => i)".to_string(),
            _ => format!("this.{}", to_camel_case(name)),
        },
        Some(Type::Set(inner)) => {
            format!("[...this.{}Set]", type_name(inner))
        }
        _ => "[]".to_string(),
    }
}

/// Convert type annotation to Python iterator
pub fn type_to_python_iter(ty: Option<&Type>) -> String {
    match ty {
        Some(Type::Named(name)) => match name.as_str() {
            "Bool" => "[False, True]".to_string(),
            "Int" => "range(-1000, 1001)".to_string(),
            "Nat" => "range(0, 1001)".to_string(),
            _ => format!("self.{}", to_snake_case(name)),
        },
        Some(Type::Set(inner)) => {
            format!("self.{}_set", type_name(inner))
        }
        _ => "[]".to_string(),
    }
}

/// Get simple type name
pub fn type_name(ty: &Type) -> String {
    match ty {
        Type::Named(n) => n.to_lowercase(),
        Type::Set(inner) => format!("{}_set", type_name(inner)),
        Type::List(inner) => format!("{}_list", type_name(inner)),
        _ => "unknown".to_string(),
    }
}
