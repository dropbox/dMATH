//! Algebraic Rewriting for Bitwise Operations
//!
//! This module provides algebraic rewrites that transform common bitwise patterns
//! to equivalent arithmetic expressions. This enables Z3/Spacer to reason about
//! bitwise operations on the Int theory without requiring BitVec support.
//!
//! # Strategy
//!
//! Bitwise operations like `&`, `|`, `^`, `<<`, `>>` are typically handled as
//! uninterpreted functions on Int, which Z3 Spacer cannot reason about.
//! This module detects common patterns and rewrites them to equivalent arithmetic:
//!
//! | Pattern | Arithmetic Equivalent |
//! |---------|----------------------|
//! | `x & (2^n - 1)` | `x mod 2^n` |
//! | `x << n` (const n) | `x * 2^n` |
//! | `x >> n` (const n, x >= 0) | `x / 2^n` |
//! | `x ^ x` | `0` |
//! | `x \| 0` | `x` |
//! | `x & 0` | `0` |
//! | `x & x` | `x` |
//! | `x \| x` | `x` |
//!
//! Complex patterns (e.g., `x & y` with both symbolic) cannot be rewritten
//! and require delegation to Kani/CBMC.
//!
//! # Example
//!
//! ```ignore
//! use kani_fast_chc::algebraic_rewrite::{try_rewrite, BitwiseOp, RewriteResult};
//!
//! // x & 0xFF -> x mod 256
//! let result = try_rewrite(BitwiseOp::And, "x", "255");
//! assert!(matches!(result, RewriteResult::Rewritten(_)));
//! ```

use std::collections::HashSet;

/// Result of attempting an algebraic rewrite
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RewriteResult {
    /// Successfully transformed to arithmetic expression
    Rewritten(String),
    /// Pattern cannot be rewritten to arithmetic
    CannotRewrite,
}

/// Bitwise operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BitwiseOp {
    /// Bitwise AND
    And,
    /// Bitwise OR
    Or,
    /// Bitwise XOR
    Xor,
    /// Left shift
    Shl,
    /// Logical right shift (unsigned)
    ShrLogical,
    /// Arithmetic right shift (signed)
    ShrArithmetic,
}

impl BitwiseOp {
    /// Get the SMT function name for this operation
    pub fn smt_function_name(&self) -> &'static str {
        match self {
            BitwiseOp::And => "bitand",
            BitwiseOp::Or => "bitor",
            BitwiseOp::Xor => "bitxor",
            BitwiseOp::Shl => "bitshl",
            BitwiseOp::ShrLogical => "bitshr_logical",
            BitwiseOp::ShrArithmetic => "bitshr_arithmetic",
        }
    }
}

/// Main entry point: attempt to rewrite a bitwise operation to arithmetic
///
/// # Arguments
/// * `op` - The bitwise operation
/// * `lhs` - Left operand (SMT expression)
/// * `rhs` - Right operand (SMT expression)
///
/// # Returns
/// `RewriteResult::Rewritten` with the arithmetic equivalent, or
/// `RewriteResult::CannotRewrite` if no rewrite is possible.
pub fn try_rewrite(op: BitwiseOp, lhs: &str, rhs: &str) -> RewriteResult {
    match op {
        BitwiseOp::And => rewrite_and(lhs, rhs),
        BitwiseOp::Or => rewrite_or(lhs, rhs),
        BitwiseOp::Xor => rewrite_xor(lhs, rhs),
        BitwiseOp::Shl => rewrite_shl(lhs, rhs),
        BitwiseOp::ShrLogical => rewrite_shr_logical(lhs, rhs),
        BitwiseOp::ShrArithmetic => rewrite_shr_arithmetic(lhs, rhs),
    }
}

/// Check if an expression is a constant integer
fn parse_constant(expr: &str) -> Option<i128> {
    let trimmed = expr.trim();

    // Handle negative numbers in SMT format: (- N)
    if trimmed.starts_with("(- ") && trimmed.ends_with(')') {
        let inner = &trimmed[3..trimmed.len() - 1];
        return inner.parse::<i128>().ok().map(|n| -n);
    }

    // Handle negative numbers in regular format: -N
    if trimmed.starts_with('-') {
        return trimmed.parse::<i128>().ok();
    }

    // Try parsing as positive integer
    trimmed.parse::<i128>().ok()
}

/// Check if a number is a power of 2
fn is_power_of_2(n: i128) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

/// Get the log base 2 of a power of 2
fn log2(n: i128) -> u32 {
    debug_assert!(is_power_of_2(n));
    let mut result = 0;
    let mut val = n;
    while val > 1 {
        val >>= 1;
        result += 1;
    }
    result
}

/// Check if a number is one less than a power of 2 (mask pattern)
fn is_mask(n: i128) -> bool {
    n > 0 && (n & (n + 1)) == 0
}

/// Get the number of bits in a mask (test-only helper)
#[cfg(test)]
fn mask_bits(n: i128) -> u32 {
    debug_assert!(is_mask(n));
    log2(n + 1)
}

/// Check if two expressions are identical (syntactically equal)
fn exprs_equal(a: &str, b: &str) -> bool {
    a.trim() == b.trim()
}

/// Check if an expression is symbolic (not a constant) (test-only helper)
#[cfg(test)]
fn is_symbolic(expr: &str) -> bool {
    parse_constant(expr).is_none()
}

// ============================================================
// AND rewrites
// ============================================================

/// Rewrite bitwise AND operations to arithmetic
///
/// Patterns:
/// - `x & 0` → `0`
/// - `x & x` → `x`
/// - `x & (2^n - 1)` → `x mod 2^n` (mask pattern)
/// - `x & 2^n` → `(x / 2^n mod 2) * 2^n` (single bit check)
fn rewrite_and(lhs: &str, rhs: &str) -> RewriteResult {
    let lhs_const = parse_constant(lhs);
    let rhs_const = parse_constant(rhs);

    // x & 0 → 0
    if rhs_const == Some(0) || lhs_const == Some(0) {
        return RewriteResult::Rewritten("0".to_string());
    }

    // x & x → x
    if exprs_equal(lhs, rhs) {
        return RewriteResult::Rewritten(lhs.trim().to_string());
    }

    // x & (2^n - 1) → x mod 2^n (mask pattern)
    if let Some(mask) = rhs_const {
        if is_mask(mask) {
            let modulus = mask + 1;
            return RewriteResult::Rewritten(format!("(mod {} {})", lhs, modulus));
        }
    }
    if let Some(mask) = lhs_const {
        if is_mask(mask) {
            let modulus = mask + 1;
            return RewriteResult::Rewritten(format!("(mod {} {})", rhs, modulus));
        }
    }

    // x & 2^n → (x / 2^n mod 2) * 2^n (single bit extraction)
    if let Some(val) = rhs_const {
        if is_power_of_2(val) {
            let shift = log2(val);
            // This extracts bit `shift` from x and returns it at its position
            // (x / 2^shift mod 2) * 2^shift
            let divisor = 1i128 << shift;
            return RewriteResult::Rewritten(format!(
                "(* (mod (div {} {}) 2) {})",
                lhs, divisor, divisor
            ));
        }
    }
    if let Some(val) = lhs_const {
        if is_power_of_2(val) {
            let shift = log2(val);
            let divisor = 1i128 << shift;
            return RewriteResult::Rewritten(format!(
                "(* (mod (div {} {}) 2) {})",
                rhs, divisor, divisor
            ));
        }
    }

    // All bits set (-1 in two's complement, but on Int this is conceptual)
    // x & -1 → x (identity)
    if rhs_const == Some(-1) {
        return RewriteResult::Rewritten(lhs.trim().to_string());
    }
    if lhs_const == Some(-1) {
        return RewriteResult::Rewritten(rhs.trim().to_string());
    }

    // Both symbolic - cannot rewrite
    RewriteResult::CannotRewrite
}

// ============================================================
// OR rewrites
// ============================================================

/// Rewrite bitwise OR operations to arithmetic
///
/// Patterns:
/// - `x | 0` → `x`
/// - `x | x` → `x`
fn rewrite_or(lhs: &str, rhs: &str) -> RewriteResult {
    let lhs_const = parse_constant(lhs);
    let rhs_const = parse_constant(rhs);

    // x | 0 → x
    if rhs_const == Some(0) {
        return RewriteResult::Rewritten(lhs.trim().to_string());
    }
    if lhs_const == Some(0) {
        return RewriteResult::Rewritten(rhs.trim().to_string());
    }

    // x | x → x
    if exprs_equal(lhs, rhs) {
        return RewriteResult::Rewritten(lhs.trim().to_string());
    }

    // Both symbolic - cannot rewrite
    RewriteResult::CannotRewrite
}

// ============================================================
// XOR rewrites
// ============================================================

/// Rewrite bitwise XOR operations to arithmetic
///
/// Patterns:
/// - `x ^ x` → `0`
/// - `x ^ 0` → `x`
fn rewrite_xor(lhs: &str, rhs: &str) -> RewriteResult {
    let lhs_const = parse_constant(lhs);
    let rhs_const = parse_constant(rhs);

    // x ^ x → 0
    if exprs_equal(lhs, rhs) {
        return RewriteResult::Rewritten("0".to_string());
    }

    // x ^ 0 → x
    if rhs_const == Some(0) {
        return RewriteResult::Rewritten(lhs.trim().to_string());
    }
    if lhs_const == Some(0) {
        return RewriteResult::Rewritten(rhs.trim().to_string());
    }

    // Both symbolic - cannot rewrite
    RewriteResult::CannotRewrite
}

// ============================================================
// Shift rewrites
// ============================================================

/// Rewrite left shift to multiplication
///
/// Pattern: `x << n` (const n) → `x * 2^n`
fn rewrite_shl(lhs: &str, rhs: &str) -> RewriteResult {
    if let Some(shift) = parse_constant(rhs) {
        if (0..128).contains(&shift) {
            let multiplier = 1i128 << shift;
            return RewriteResult::Rewritten(format!("(* {} {})", lhs, multiplier));
        }
    }

    // Variable shift - cannot rewrite without BitVec
    RewriteResult::CannotRewrite
}

/// Rewrite logical right shift to division (for unsigned/non-negative values)
///
/// Pattern: `x >> n` (const n, x >= 0) → `x / 2^n`
fn rewrite_shr_logical(lhs: &str, rhs: &str) -> RewriteResult {
    if let Some(shift) = parse_constant(rhs) {
        if (0..128).contains(&shift) {
            let divisor = 1i128 << shift;
            return RewriteResult::Rewritten(format!("(div {} {})", lhs, divisor));
        }
    }

    // Variable shift - cannot rewrite
    RewriteResult::CannotRewrite
}

/// Rewrite arithmetic right shift
///
/// For signed values, this is more complex as it preserves the sign bit.
/// For now, we handle the case where x >= 0, which behaves like logical shift.
fn rewrite_shr_arithmetic(lhs: &str, rhs: &str) -> RewriteResult {
    // For now, treat arithmetic shift as logical shift
    // This is correct when x >= 0
    // A more complete solution would require knowing the sign of lhs
    rewrite_shr_logical(lhs, rhs)
}

// ============================================================
// Expression rewriting for full SMT expressions
// ============================================================

/// Rewrite an SMT expression, replacing bitwise function calls with arithmetic
///
/// This recursively processes the expression and applies rewrites where possible.
pub fn rewrite_expression(expr: &str) -> (String, bool) {
    let trimmed = expr.trim();

    // Not an S-expression
    if !trimmed.starts_with('(') {
        return (trimmed.to_string(), false);
    }

    // Parse the function call
    let inner = &trimmed[1..trimmed.len() - 1];
    let parts = parse_sexp_parts(inner);

    if parts.is_empty() {
        return (trimmed.to_string(), false);
    }

    let func = &parts[0];
    let mut any_rewritten = false;

    // First, recursively process arguments
    let rewritten_args: Vec<String> = parts[1..]
        .iter()
        .map(|arg| {
            let (rewritten, was_rewritten) = rewrite_expression(arg);
            if was_rewritten {
                any_rewritten = true;
            }
            rewritten
        })
        .collect();

    // Check if this is a bitwise operation we can rewrite
    let op = match func.as_str() {
        "bitand" => Some(BitwiseOp::And),
        "bitor" => Some(BitwiseOp::Or),
        "bitxor" => Some(BitwiseOp::Xor),
        "bitshl" => Some(BitwiseOp::Shl),
        "bitshr" => Some(BitwiseOp::ShrLogical),
        _ => None,
    };

    if let Some(op) = op {
        if rewritten_args.len() >= 2 {
            match try_rewrite(op, &rewritten_args[0], &rewritten_args[1]) {
                RewriteResult::Rewritten(result) => {
                    return (result, true);
                }
                RewriteResult::CannotRewrite => {
                    // Keep as uninterpreted function call
                }
            }
        }
    }

    // Rebuild the expression with rewritten args
    if any_rewritten || func.as_str() != parts[0] {
        let new_expr = format!("({} {})", func, rewritten_args.join(" "));
        (new_expr, any_rewritten)
    } else {
        (trimmed.to_string(), false)
    }
}

/// Parse parts of an S-expression, handling nested parentheses
fn parse_sexp_parts(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut depth = 0;

    for c in s.chars() {
        match c {
            '(' => {
                depth += 1;
                current.push(c);
            }
            ')' => {
                depth -= 1;
                current.push(c);
            }
            ' ' | '\t' | '\n' if depth == 0 => {
                if !current.is_empty() {
                    parts.push(std::mem::take(&mut current));
                }
            }
            _ => {
                current.push(c);
            }
        }
    }

    if !current.is_empty() {
        parts.push(current);
    }

    parts
}

/// Check if an expression contains any bitwise operations
pub fn contains_bitwise(expr: &str) -> bool {
    expr.contains("bitand")
        || expr.contains("bitor")
        || expr.contains("bitxor")
        || expr.contains("bitshl")
        || expr.contains("bitshr")
}

/// Get the set of bitwise operations used in an expression
pub fn collect_bitwise_ops(expr: &str) -> HashSet<BitwiseOp> {
    let mut ops = HashSet::new();

    if expr.contains("bitand") {
        ops.insert(BitwiseOp::And);
    }
    if expr.contains("bitor") {
        ops.insert(BitwiseOp::Or);
    }
    if expr.contains("bitxor") {
        ops.insert(BitwiseOp::Xor);
    }
    if expr.contains("bitshl") {
        ops.insert(BitwiseOp::Shl);
    }
    if expr.contains("bitshr") {
        ops.insert(BitwiseOp::ShrLogical);
    }

    ops
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================
    // Constant parsing tests
    // ============================================================

    #[test]
    fn test_parse_constant_positive() {
        assert_eq!(parse_constant("42"), Some(42));
        assert_eq!(parse_constant("0"), Some(0));
        assert_eq!(parse_constant("255"), Some(255));
    }

    #[test]
    fn test_parse_constant_negative() {
        assert_eq!(parse_constant("-1"), Some(-1));
        assert_eq!(parse_constant("-128"), Some(-128));
    }

    #[test]
    fn test_parse_constant_smt_negative() {
        assert_eq!(parse_constant("(- 42)"), Some(-42));
        assert_eq!(parse_constant("(- 1)"), Some(-1));
    }

    #[test]
    fn test_parse_constant_symbolic() {
        assert_eq!(parse_constant("x"), None);
        assert_eq!(parse_constant("(+ x 1)"), None);
        assert_eq!(parse_constant("_0"), None);
    }

    #[test]
    fn test_parse_constant_whitespace() {
        assert_eq!(parse_constant("  42  "), Some(42));
        assert_eq!(parse_constant("\t255\n"), Some(255));
    }

    // ============================================================
    // Power of 2 tests
    // ============================================================

    #[test]
    fn test_is_power_of_2() {
        assert!(is_power_of_2(1));
        assert!(is_power_of_2(2));
        assert!(is_power_of_2(4));
        assert!(is_power_of_2(256));
        assert!(is_power_of_2(1024));
        assert!(!is_power_of_2(0));
        assert!(!is_power_of_2(3));
        assert!(!is_power_of_2(255));
        assert!(!is_power_of_2(-1));
    }

    #[test]
    fn test_log2() {
        assert_eq!(log2(1), 0);
        assert_eq!(log2(2), 1);
        assert_eq!(log2(4), 2);
        assert_eq!(log2(256), 8);
        assert_eq!(log2(1024), 10);
    }

    #[test]
    fn test_is_mask() {
        assert!(is_mask(1)); // 0b1
        assert!(is_mask(3)); // 0b11
        assert!(is_mask(7)); // 0b111
        assert!(is_mask(255)); // 0xFF
        assert!(is_mask(65535)); // 0xFFFF
        assert!(!is_mask(0));
        assert!(!is_mask(2));
        assert!(!is_mask(4));
        assert!(!is_mask(-1));
    }

    #[test]
    fn test_mask_bits() {
        assert_eq!(mask_bits(1), 1);
        assert_eq!(mask_bits(3), 2);
        assert_eq!(mask_bits(7), 3);
        assert_eq!(mask_bits(255), 8);
        assert_eq!(mask_bits(65535), 16);
    }

    // ============================================================
    // AND rewrite tests
    // ============================================================

    #[test]
    fn test_and_with_zero() {
        assert_eq!(
            try_rewrite(BitwiseOp::And, "x", "0"),
            RewriteResult::Rewritten("0".to_string())
        );
        assert_eq!(
            try_rewrite(BitwiseOp::And, "0", "x"),
            RewriteResult::Rewritten("0".to_string())
        );
    }

    #[test]
    fn test_and_with_self() {
        assert_eq!(
            try_rewrite(BitwiseOp::And, "x", "x"),
            RewriteResult::Rewritten("x".to_string())
        );
        assert_eq!(
            try_rewrite(BitwiseOp::And, "(+ y 1)", "(+ y 1)"),
            RewriteResult::Rewritten("(+ y 1)".to_string())
        );
    }

    #[test]
    fn test_and_with_mask() {
        // x & 255 → x mod 256
        assert_eq!(
            try_rewrite(BitwiseOp::And, "x", "255"),
            RewriteResult::Rewritten("(mod x 256)".to_string())
        );
        // x & 65535 → x mod 65536
        assert_eq!(
            try_rewrite(BitwiseOp::And, "x", "65535"),
            RewriteResult::Rewritten("(mod x 65536)".to_string())
        );
        // Commutative
        assert_eq!(
            try_rewrite(BitwiseOp::And, "255", "x"),
            RewriteResult::Rewritten("(mod x 256)".to_string())
        );
    }

    #[test]
    fn test_and_with_power_of_2() {
        // x & 4 → (* (mod (div x 4) 2) 4)
        let result = try_rewrite(BitwiseOp::And, "x", "4");
        assert!(matches!(result, RewriteResult::Rewritten(_)));
        if let RewriteResult::Rewritten(expr) = result {
            assert!(expr.contains("div"));
            assert!(expr.contains("mod"));
        }
    }

    #[test]
    fn test_and_symbolic_symbolic() {
        assert_eq!(
            try_rewrite(BitwiseOp::And, "x", "y"),
            RewriteResult::CannotRewrite
        );
    }

    // ============================================================
    // OR rewrite tests
    // ============================================================

    #[test]
    fn test_or_with_zero() {
        assert_eq!(
            try_rewrite(BitwiseOp::Or, "x", "0"),
            RewriteResult::Rewritten("x".to_string())
        );
        assert_eq!(
            try_rewrite(BitwiseOp::Or, "0", "x"),
            RewriteResult::Rewritten("x".to_string())
        );
    }

    #[test]
    fn test_or_with_self() {
        assert_eq!(
            try_rewrite(BitwiseOp::Or, "x", "x"),
            RewriteResult::Rewritten("x".to_string())
        );
    }

    #[test]
    fn test_or_symbolic_symbolic() {
        assert_eq!(
            try_rewrite(BitwiseOp::Or, "x", "y"),
            RewriteResult::CannotRewrite
        );
    }

    // ============================================================
    // XOR rewrite tests
    // ============================================================

    #[test]
    fn test_xor_with_self() {
        assert_eq!(
            try_rewrite(BitwiseOp::Xor, "x", "x"),
            RewriteResult::Rewritten("0".to_string())
        );
        assert_eq!(
            try_rewrite(BitwiseOp::Xor, "(+ y 1)", "(+ y 1)"),
            RewriteResult::Rewritten("0".to_string())
        );
    }

    #[test]
    fn test_xor_with_zero() {
        assert_eq!(
            try_rewrite(BitwiseOp::Xor, "x", "0"),
            RewriteResult::Rewritten("x".to_string())
        );
        assert_eq!(
            try_rewrite(BitwiseOp::Xor, "0", "x"),
            RewriteResult::Rewritten("x".to_string())
        );
    }

    #[test]
    fn test_xor_symbolic_symbolic() {
        assert_eq!(
            try_rewrite(BitwiseOp::Xor, "x", "y"),
            RewriteResult::CannotRewrite
        );
    }

    // ============================================================
    // Shift rewrite tests
    // ============================================================

    #[test]
    fn test_shl_constant() {
        // x << 0 → x * 1
        assert_eq!(
            try_rewrite(BitwiseOp::Shl, "x", "0"),
            RewriteResult::Rewritten("(* x 1)".to_string())
        );
        // x << 1 → x * 2
        assert_eq!(
            try_rewrite(BitwiseOp::Shl, "x", "1"),
            RewriteResult::Rewritten("(* x 2)".to_string())
        );
        // x << 4 → x * 16
        assert_eq!(
            try_rewrite(BitwiseOp::Shl, "x", "4"),
            RewriteResult::Rewritten("(* x 16)".to_string())
        );
    }

    #[test]
    fn test_shl_variable() {
        assert_eq!(
            try_rewrite(BitwiseOp::Shl, "x", "y"),
            RewriteResult::CannotRewrite
        );
    }

    #[test]
    fn test_shr_logical_constant() {
        // x >> 0 → x / 1
        assert_eq!(
            try_rewrite(BitwiseOp::ShrLogical, "x", "0"),
            RewriteResult::Rewritten("(div x 1)".to_string())
        );
        // x >> 1 → x / 2
        assert_eq!(
            try_rewrite(BitwiseOp::ShrLogical, "x", "1"),
            RewriteResult::Rewritten("(div x 2)".to_string())
        );
        // x >> 4 → x / 16
        assert_eq!(
            try_rewrite(BitwiseOp::ShrLogical, "x", "4"),
            RewriteResult::Rewritten("(div x 16)".to_string())
        );
    }

    #[test]
    fn test_shr_variable() {
        assert_eq!(
            try_rewrite(BitwiseOp::ShrLogical, "x", "y"),
            RewriteResult::CannotRewrite
        );
    }

    // ============================================================
    // Expression rewriting tests
    // ============================================================

    #[test]
    fn test_rewrite_expression_simple() {
        let (result, rewritten) = rewrite_expression("(bitand x 255)");
        assert!(rewritten);
        assert!(result.contains("mod"));
        assert!(result.contains("256"));
    }

    #[test]
    fn test_rewrite_expression_nested() {
        // (+ (bitand x 255) 1) should rewrite the inner bitand
        let (result, rewritten) = rewrite_expression("(+ (bitand x 255) 1)");
        assert!(rewritten);
        assert!(result.contains("mod"));
    }

    #[test]
    fn test_rewrite_expression_no_bitwise() {
        let (result, rewritten) = rewrite_expression("(+ x 1)");
        assert!(!rewritten);
        assert_eq!(result, "(+ x 1)");
    }

    #[test]
    fn test_rewrite_expression_cannot_rewrite() {
        // (bitand x y) cannot be rewritten
        let (result, rewritten) = rewrite_expression("(bitand x y)");
        assert!(!rewritten);
        assert_eq!(result, "(bitand x y)");
    }

    // ============================================================
    // Collection tests
    // ============================================================

    #[test]
    fn test_contains_bitwise() {
        assert!(contains_bitwise("(bitand x 255)"));
        assert!(contains_bitwise("(bitor x y)"));
        assert!(contains_bitwise("(+ (bitxor a b) 1)"));
        assert!(!contains_bitwise("(+ x 1)"));
        assert!(!contains_bitwise("(* y z)"));
    }

    #[test]
    fn test_collect_bitwise_ops() {
        let ops = collect_bitwise_ops("(+ (bitand x 255) (bitor y z))");
        assert!(ops.contains(&BitwiseOp::And));
        assert!(ops.contains(&BitwiseOp::Or));
        assert!(!ops.contains(&BitwiseOp::Xor));
    }

    // ============================================================
    // S-expression parsing tests
    // ============================================================

    #[test]
    fn test_parse_sexp_parts_simple() {
        let parts = parse_sexp_parts("+ x y");
        assert_eq!(parts, vec!["+", "x", "y"]);
    }

    #[test]
    fn test_parse_sexp_parts_nested() {
        let parts = parse_sexp_parts("+ (- a b) c");
        assert_eq!(parts, vec!["+", "(- a b)", "c"]);
    }

    #[test]
    fn test_parse_sexp_parts_deeply_nested() {
        let parts = parse_sexp_parts("foo (bar (baz x)) y");
        assert_eq!(parts, vec!["foo", "(bar (baz x))", "y"]);
    }

    // ============================================================
    // Edge case tests
    // ============================================================

    #[test]
    fn test_and_with_negative_one() {
        // x & -1 → x (all bits set)
        assert_eq!(
            try_rewrite(BitwiseOp::And, "x", "-1"),
            RewriteResult::Rewritten("x".to_string())
        );
    }

    #[test]
    fn test_large_shift() {
        // Very large shifts should still work
        assert_eq!(
            try_rewrite(BitwiseOp::Shl, "x", "64"),
            RewriteResult::Rewritten(format!("(* x {})", 1u128 << 64))
        );
    }

    #[test]
    fn test_shift_too_large() {
        // Shift >= 128 cannot be computed
        assert_eq!(
            try_rewrite(BitwiseOp::Shl, "x", "128"),
            RewriteResult::CannotRewrite
        );
    }

    #[test]
    fn test_whitespace_handling() {
        assert_eq!(
            try_rewrite(BitwiseOp::And, "  x  ", "  0  "),
            RewriteResult::Rewritten("0".to_string())
        );
    }

    // ============================================================
    // BitwiseOp tests
    // ============================================================

    #[test]
    fn test_bitwise_op_smt_names() {
        assert_eq!(BitwiseOp::And.smt_function_name(), "bitand");
        assert_eq!(BitwiseOp::Or.smt_function_name(), "bitor");
        assert_eq!(BitwiseOp::Xor.smt_function_name(), "bitxor");
        assert_eq!(BitwiseOp::Shl.smt_function_name(), "bitshl");
        assert_eq!(BitwiseOp::ShrLogical.smt_function_name(), "bitshr_logical");
    }

    // ============================================================
    // Mutation coverage tests
    // ============================================================

    /// Test parse_constant rejects malformed SMT negative format
    /// Catches: algebraic_rewrite.rs:105:35 replace && with ||
    #[test]
    fn test_parse_constant_smt_negative_requires_both_conditions() {
        // Starts with "(- " but does NOT end with ")"
        // If && becomes ||, this would incorrectly try to parse as SMT negative
        assert_eq!(parse_constant("(- 42"), None);
        // Ends with ")" but does NOT start with "(- "
        // If && becomes ||, this would incorrectly try to parse
        assert_eq!(parse_constant("42)"), None);
        // Neither condition met
        assert_eq!(parse_constant("abc"), None);
        // Both conditions met - should parse correctly
        assert_eq!(parse_constant("(- 42)"), Some(-42));
    }

    /// Test is_symbolic correctly distinguishes constants from symbols
    /// Catches: algebraic_rewrite.rs:156:5 replace is_symbolic -> bool with true/false
    #[test]
    fn test_is_symbolic_distinguishes_constants_and_symbols() {
        // Constants should NOT be symbolic
        assert!(!is_symbolic("42"));
        assert!(!is_symbolic("-1"));
        assert!(!is_symbolic("0"));
        assert!(!is_symbolic("(- 123)"));
        // Symbols SHOULD be symbolic
        assert!(is_symbolic("x"));
        assert!(is_symbolic("my_var"));
        assert!(is_symbolic("(+ x 1)"));
    }

    /// Test that rewrite_and produces correct shifts for power-of-2 masks
    /// Catches: algebraic_rewrite.rs:204:33 replace << with >>
    /// Catches: algebraic_rewrite.rs:214:33 replace << with >>
    #[test]
    fn test_rewrite_and_shift_direction_matters() {
        // x & 4 should produce divisor = 4 (1 << 2 = 4, not 1 >> 2 = 0)
        // Result: (* (mod (div x 4) 2) 4)
        let result = try_rewrite(BitwiseOp::And, "x", "4");
        match result {
            RewriteResult::Rewritten(s) => {
                assert!(s.contains('4'), "Should contain divisor 4, got: {s}");
                assert!(
                    !s.contains(" 0)"),
                    "Should not contain 0 as divisor, got: {}",
                    s
                );
            }
            RewriteResult::CannotRewrite => panic!("Expected rewrite for x & 4"),
        }

        // 8 & y should also produce divisor = 8 (1 << 3 = 8, not 1 >> 3 = 0)
        let result2 = try_rewrite(BitwiseOp::And, "8", "y");
        match result2 {
            RewriteResult::Rewritten(s) => {
                assert!(s.contains('8'), "Should contain divisor 8, got: {s}");
            }
            RewriteResult::CannotRewrite => panic!("Expected rewrite for 8 & y"),
        }
    }

    /// Test that -1 identity in rewrite_and preserves the other operand
    /// Catches: algebraic_rewrite.rs:227:26 delete - in rewrite_and
    #[test]
    fn test_rewrite_and_minus_one_identity() {
        // x & -1 should return x (identity)
        let result = try_rewrite(BitwiseOp::And, "x", "-1");
        assert_eq!(result, RewriteResult::Rewritten("x".to_string()));

        // -1 & y should return y (identity)
        let result2 = try_rewrite(BitwiseOp::And, "-1", "y");
        assert_eq!(result2, RewriteResult::Rewritten("y".to_string()));
    }

    /// Test rewrite_expression comparison for changed expressions
    /// Catches: algebraic_rewrite.rs:402:39 replace != with ==
    #[test]
    fn test_rewrite_expression_reports_changes_correctly() {
        // An expression that WILL be rewritten
        let (result, changed) = rewrite_expression("(bitand x 0)");
        assert_eq!(result, "0");
        assert!(changed, "Should report changed=true when rewrite happens");

        // An expression that won't be rewritten (no bitwise ops)
        let (result2, changed2) = rewrite_expression("(+ x 1)");
        assert_eq!(result2, "(+ x 1)");
        assert!(
            !changed2,
            "Should report changed=false when no rewrite happens"
        );
    }

    /// Test contains_bitwise requires any one of the operations
    /// Catches: algebraic_rewrite.rs:451:9 replace || with &&
    #[test]
    fn test_contains_bitwise_is_disjunction_not_conjunction() {
        // Single bitwise operation should be detected
        assert!(contains_bitwise("(bitand x y)"));
        assert!(contains_bitwise("(bitor x y)"));
        assert!(contains_bitwise("(bitxor x y)"));
        assert!(contains_bitwise("(bitshl x 2)"));
        assert!(contains_bitwise("(bitshr x 2)"));

        // No bitwise operations
        assert!(!contains_bitwise("(+ x y)"));
        assert!(!contains_bitwise("(* x 2)"));
    }
}
