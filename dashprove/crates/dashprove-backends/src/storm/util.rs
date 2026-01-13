//! Utility functions for Storm backend

use dashprove_usl::ast::Expr;

/// Convert USL identifier to valid PRISM identifier
pub fn to_prism_ident(name: &str) -> String {
    // PRISM identifiers must start with letter or underscore
    let mut result = String::new();
    for (i, c) in name.chars().enumerate() {
        if c.is_alphanumeric() || c == '_' {
            if i == 0 && c.is_numeric() {
                result.push('_');
            }
            result.push(c);
        } else {
            result.push('_');
        }
    }
    // Avoid PRISM keywords
    if [
        "module",
        "endmodule",
        "init",
        "true",
        "false",
        "const",
        "global",
    ]
    .contains(&result.as_str())
    {
        result.push_str("_var");
    }
    result
}

/// Extract numeric value from expression
pub fn extract_numeric_value(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::Int(n) => Some(*n as f64),
        Expr::Float(f) => Some(*f),
        _ => None,
    }
}
