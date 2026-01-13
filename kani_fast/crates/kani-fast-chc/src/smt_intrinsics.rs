//! Mapping from Rust intrinsics to SMT bitvector theory.
//!
//! Z4/Z3 implements the actual bitvector semantics - we just connect the names.
//! For intrinsics not native to SMT-LIB2, we provide manual encodings.
//!
//! # Philosophy
//!
//! Don't reimplement what Z4 already knows. Map intrinsics to native SMT-LIB2
//! bitvector operations and let the solver handle the semantics. Only provide
//! manual encodings when absolutely necessary.
//!
//! # Supported Intrinsics
//!
//! | Rust Intrinsic | SMT-LIB2 Operation |
//! |----------------|-------------------|
//! | wrapping_add | bvadd |
//! | wrapping_sub | bvsub |
//! | wrapping_mul | bvmul |
//! | rotate_left | rotate_left |
//! | rotate_right | rotate_right |
//! | bitand (&) | bvand |
//! | bitor (\|) | bvor |
//! | bitxor (^) | bvxor |
//! | shl (<<) | bvshl |
//! | shr (>>) | bvlshr/bvashr |
//! | ctpop | manual encoding |
//! | ctlz | manual encoding |
//! | cttz | manual encoding |
//! | bswap | extract+concat |

use crate::bitvec::BitvecConfig;

/// Map a Rust intrinsic name to SMT-LIB2 bitvector expression.
///
/// Returns `Ok(smt_expr)` if the intrinsic is supported, `Err(reason)` otherwise.
/// The returned expression uses native SMT-LIB2 bitvector operations.
pub fn intrinsic_to_smt(
    name: &str,
    args: &[String],
    config: &BitvecConfig,
) -> Result<String, String> {
    // Extract operation name from mangled MIR names like "core_num_impl_u32_wrapping_add"
    let op = extract_operation(name);

    match op {
        // ===== Wrapping arithmetic (native bitvector behavior) =====
        "wrapping_add" | "unchecked_add" => {
            require_args(args, 2)?;
            Ok(format!("(bvadd {} {})", args[0], args[1]))
        }
        "wrapping_sub" | "unchecked_sub" => {
            require_args(args, 2)?;
            Ok(format!("(bvsub {} {})", args[0], args[1]))
        }
        "wrapping_mul" | "unchecked_mul" => {
            require_args(args, 2)?;
            Ok(format!("(bvmul {} {})", args[0], args[1]))
        }
        "wrapping_neg" => {
            require_args(args, 1)?;
            Ok(format!("(bvneg {})", args[0]))
        }

        // ===== Division/remainder =====
        "unchecked_div" | "wrapping_div" => {
            require_args(args, 2)?;
            if config.use_signed {
                Ok(format!("(bvsdiv {} {})", args[0], args[1]))
            } else {
                Ok(format!("(bvudiv {} {})", args[0], args[1]))
            }
        }
        "unchecked_rem" | "wrapping_rem" => {
            require_args(args, 2)?;
            if config.use_signed {
                Ok(format!("(bvsrem {} {})", args[0], args[1]))
            } else {
                Ok(format!("(bvurem {} {})", args[0], args[1]))
            }
        }

        // ===== Rotation =====
        // SMT-LIB2 rotate_left/rotate_right require constant amounts.
        // For variable amounts, we encode as: (x << n) | (x >> (width - n))
        "rotate_left" => {
            require_args(args, 2)?;
            Ok(encode_rotate_left(&args[0], &args[1], config.default_width))
        }
        "rotate_right" => {
            require_args(args, 2)?;
            Ok(encode_rotate_right(
                &args[0],
                &args[1],
                config.default_width,
            ))
        }

        // ===== Shifts =====
        "unchecked_shl" | "wrapping_shl" => {
            require_args(args, 2)?;
            Ok(format!("(bvshl {} {})", args[0], args[1]))
        }
        "unchecked_shr" | "wrapping_shr" => {
            require_args(args, 2)?;
            if config.use_signed {
                Ok(format!("(bvashr {} {})", args[0], args[1]))
            } else {
                Ok(format!("(bvlshr {} {})", args[0], args[1]))
            }
        }

        // ===== Bitwise operations =====
        "bitand" => {
            require_args(args, 2)?;
            Ok(format!("(bvand {} {})", args[0], args[1]))
        }
        "bitor" => {
            require_args(args, 2)?;
            Ok(format!("(bvor {} {})", args[0], args[1]))
        }
        "bitxor" => {
            require_args(args, 2)?;
            Ok(format!("(bvxor {} {})", args[0], args[1]))
        }
        "bitnot" => {
            require_args(args, 1)?;
            Ok(format!("(bvnot {})", args[0]))
        }

        // ===== Comparison =====
        "ptr_guaranteed_cmp" => {
            require_args(args, 2)?;
            Ok(format!("(= {} {})", args[0], args[1]))
        }

        // ===== Bit manipulation (manual encodings) =====
        // These don't have native SMT-LIB2 operations, so we provide manual encodings.
        "ctlz" | "ctlz_nonzero" | "leading_zeros" => {
            require_args(args, 1)?;
            Ok(encode_ctlz(&args[0], config.default_width))
        }
        "cttz" | "cttz_nonzero" | "trailing_zeros" => {
            require_args(args, 1)?;
            Ok(encode_cttz(&args[0], config.default_width))
        }
        "ctpop" | "count_ones" => {
            require_args(args, 1)?;
            Ok(encode_ctpop(&args[0], config.default_width))
        }
        "bitreverse" | "reverse_bits" => {
            require_args(args, 1)?;
            Ok(encode_bitreverse(&args[0], config.default_width))
        }
        "bswap" | "swap_bytes" => {
            require_args(args, 1)?;
            Ok(encode_bswap(&args[0], config.default_width))
        }

        // ===== Control flow hints (identity) =====
        "assume" => {
            require_args(args, 1)?;
            Ok(args[0].clone())
        }
        "likely" | "unlikely" => {
            require_args(args, 1)?;
            Ok(args[0].clone())
        }
        "black_box" => {
            require_args(args, 1)?;
            Ok(args[0].clone())
        }

        // ===== Type intrinsics (should be const-evaluated) =====
        "size_of" | "align_of" | "type_id" | "type_name" | "needs_drop" => {
            Err(format!("Type intrinsic {} should be const-evaluated", op))
        }

        // ===== Memory intrinsics (need array theory) =====
        "copy" | "copy_nonoverlapping" | "write_bytes" => {
            Err(format!("Memory intrinsic {} needs array theory", op))
        }

        // ===== Unknown =====
        _ => Err(format!("Unknown intrinsic: {}", name)),
    }
}

/// Extract operation name from mangled MIR function names.
///
/// Handles patterns like:
/// - `core_num_impl_u32_wrapping_add` -> `wrapping_add`
/// - `std_intrinsics_rotate_left` -> `rotate_left`
fn extract_operation(name: &str) -> &str {
    // Common patterns: find the last known operation
    let ops = [
        "wrapping_add",
        "wrapping_sub",
        "wrapping_mul",
        "wrapping_neg",
        "wrapping_div",
        "wrapping_rem",
        "wrapping_shl",
        "wrapping_shr",
        "unchecked_add",
        "unchecked_sub",
        "unchecked_mul",
        "unchecked_div",
        "unchecked_rem",
        "unchecked_shl",
        "unchecked_shr",
        "rotate_left",
        "rotate_right",
        "leading_zeros",
        "trailing_zeros",
        "count_ones",
        "swap_bytes",
        "reverse_bits",
        "ctlz",
        "cttz",
        "ctpop",
        "bitreverse",
        "bswap",
        "assume",
        "likely",
        "unlikely",
        "black_box",
        "size_of",
        "align_of",
        "type_id",
        "type_name",
        "needs_drop",
        "copy",
        "copy_nonoverlapping",
        "write_bytes",
        "ptr_guaranteed_cmp",
    ];

    for op in ops {
        // Check if name ends with the operation
        if name.ends_with(op) {
            return op;
        }
        // Check if name contains _op (underscore prefix pattern)
        // Avoid format! allocation by searching for op and checking preceding char
        if let Some(pos) = name.find(op) {
            if pos > 0 && name.as_bytes()[pos - 1] == b'_' {
                return op;
            }
        }
    }

    // Fallback: try the raw name
    name.rsplit('_').next().unwrap_or(name)
}

/// Check argument count matches expected.
fn require_args(args: &[String], expected: usize) -> Result<(), String> {
    if args.len() != expected {
        Err(format!(
            "Expected {} arguments, got {}",
            expected,
            args.len()
        ))
    } else {
        Ok(())
    }
}

// =============================================================================
// Manual bit manipulation encodings
// =============================================================================

/// Encode popcount (count ones) for a bitvector.
///
/// Strategy: Sum each bit individually. For n-bit value:
/// (+ (ite (= ((_ extract i i) x) #b1) 1 0) ...)
///
/// This generates O(n) terms but SMT solvers handle it efficiently.
fn encode_ctpop(arg: &str, width: u32) -> String {
    let mut terms = Vec::new();
    for i in 0..width {
        // Extract bit i and convert to 0 or 1
        terms.push(format!(
            "(ite (= ((_ extract {i} {i}) {arg}) #b1) (_ bv1 {width}) (_ bv0 {width}))"
        ));
    }
    // Sum all the bit values
    if terms.is_empty() {
        return format!("(_ bv0 {})", width);
    }
    let mut result = terms[0].clone();
    for term in terms.iter().skip(1) {
        result = format!("(bvadd {} {})", result, term);
    }
    result
}

/// Encode count leading zeros for a bitvector.
///
/// Strategy: Cascade of conditionals from MSB down:
/// (ite (= ((_ extract n-1 n-1) x) #b1) 0
///   (ite (= ((_ extract n-2 n-2) x) #b1) 1
///     ...
///     n))
fn encode_ctlz(arg: &str, width: u32) -> String {
    if width == 0 {
        return format!("(_ bv0 {})", width);
    }
    // Build from inside out: if all bits are 0, result is width
    let mut result = format!("(_ bv{} {})", width, width);
    for i in 0..width {
        let bit_pos = width - 1 - i;
        // If bit at bit_pos is 1, we found the first set bit, so clz = i
        result = format!(
            "(ite (= ((_ extract {bit_pos} {bit_pos}) {arg}) #b1) (_ bv{i} {width}) {result})"
        );
    }
    result
}

/// Encode count trailing zeros for a bitvector.
///
/// Strategy: Cascade of conditionals from LSB up:
/// (ite (= ((_ extract 0 0) x) #b1) 0
///   (ite (= ((_ extract 1 1) x) #b1) 1
///     ...
///     n))
fn encode_cttz(arg: &str, width: u32) -> String {
    if width == 0 {
        return format!("(_ bv0 {})", width);
    }
    // Build from inside out: if all bits are 0, result is width
    let mut result = format!("(_ bv{} {})", width, width);
    for i in (0..width).rev() {
        // If bit at position i is 1, we found the first set bit from LSB, so ctz = i
        result = format!("(ite (= ((_ extract {i} {i}) {arg}) #b1) (_ bv{i} {width}) {result})");
    }
    result
}

/// Encode byte swap for a bitvector.
///
/// Strategy: Extract each byte and concatenate in reverse order.
/// For 32-bit: concat(byte0, byte1, byte2, byte3) from original (byte3, byte2, byte1, byte0)
fn encode_bswap(arg: &str, width: u32) -> String {
    let bytes = width / 8;
    if bytes <= 1 {
        // 8-bit or less: no swap needed
        return arg.to_string();
    }
    // Extract bytes in reverse order
    let mut parts = Vec::new();
    for i in 0..bytes {
        let low = i * 8;
        let high = low + 7;
        parts.push(format!("((_ extract {} {}) {})", high, low, arg));
    }
    // Concatenate in order (first extracted byte becomes MSB)
    let mut result = parts[0].clone();
    for part in parts.iter().skip(1) {
        result = format!("(concat {} {})", result, part);
    }
    result
}

/// Encode bit reverse for a bitvector.
///
/// Strategy: Extract each bit and concatenate in reverse order.
fn encode_bitreverse(arg: &str, width: u32) -> String {
    if width == 0 {
        return arg.to_string();
    }
    // Extract bits in order (bit 0 first, becomes MSB after concat)
    let mut parts = Vec::new();
    for i in 0..width {
        parts.push(format!("((_ extract {} {}) {})", i, i, arg));
    }
    // Concatenate: bit 0 becomes MSB, bit n-1 becomes LSB
    let mut result = parts[0].clone();
    for part in parts.iter().skip(1) {
        result = format!("(concat {} {})", result, part);
    }
    result
}

/// Encode rotate left with variable amount.
///
/// rotate_left(x, n) = (x << n) | (x >> (width - n))
///
/// In SMT-LIB2: (bvor (bvshl x n) (bvlshr x (bvsub width n)))
fn encode_rotate_left(val: &str, amount: &str, width: u32) -> String {
    // Mask amount to valid range (n mod width) to handle amounts >= width
    let width_bv = format!("(_ bv{} {})", width, width);
    let masked_amount = format!("(bvurem {} {})", amount, width_bv);
    let complement = format!("(bvsub {} {})", width_bv, masked_amount);

    format!(
        "(bvor (bvshl {} {}) (bvlshr {} {}))",
        val, masked_amount, val, complement
    )
}

/// Encode rotate right with variable amount.
///
/// rotate_right(x, n) = (x >> n) | (x << (width - n))
///
/// In SMT-LIB2: (bvor (bvlshr x n) (bvshl x (bvsub width n)))
fn encode_rotate_right(val: &str, amount: &str, width: u32) -> String {
    // Mask amount to valid range (n mod width) to handle amounts >= width
    let width_bv = format!("(_ bv{} {})", width, width);
    let masked_amount = format!("(bvurem {} {})", amount, width_bv);
    let complement = format!("(bvsub {} {})", width_bv, masked_amount);

    format!(
        "(bvor (bvlshr {} {}) (bvshl {} {}))",
        val, masked_amount, val, complement
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> BitvecConfig {
        BitvecConfig::new(32)
    }

    #[test]
    fn test_wrapping_add() {
        let result = intrinsic_to_smt(
            "core_num_impl_u32_wrapping_add",
            &["x".to_string(), "y".to_string()],
            &config(),
        );
        assert_eq!(result.unwrap(), "(bvadd x y)");
    }

    #[test]
    fn test_wrapping_sub() {
        let result = intrinsic_to_smt(
            "wrapping_sub",
            &["a".to_string(), "b".to_string()],
            &config(),
        );
        assert_eq!(result.unwrap(), "(bvsub a b)");
    }

    #[test]
    fn test_wrapping_neg() {
        let result = intrinsic_to_smt("wrapping_neg", &["x".to_string()], &config());
        assert_eq!(result.unwrap(), "(bvneg x)");
    }

    #[test]
    fn test_shift_signed() {
        let config = BitvecConfig::new(32);
        let result = intrinsic_to_smt(
            "unchecked_shr",
            &["x".to_string(), "n".to_string()],
            &config,
        );
        assert_eq!(result.unwrap(), "(bvashr x n)");
    }

    #[test]
    fn test_shift_unsigned() {
        let config = BitvecConfig::unsigned(32);
        let result = intrinsic_to_smt(
            "unchecked_shr",
            &["x".to_string(), "n".to_string()],
            &config,
        );
        assert_eq!(result.unwrap(), "(bvlshr x n)");
    }

    #[test]
    fn test_ctlz() {
        let cfg = BitvecConfig::new(8);
        let result = intrinsic_to_smt("ctlz", &["x".to_string()], &cfg);
        assert!(result.is_ok());
        let smt = result.unwrap();
        // Should contain ite cascade
        assert!(smt.contains("ite"));
        assert!(smt.contains("extract"));
    }

    #[test]
    fn test_cttz() {
        let cfg = BitvecConfig::new(8);
        let result = intrinsic_to_smt("cttz", &["x".to_string()], &cfg);
        assert!(result.is_ok());
        let smt = result.unwrap();
        assert!(smt.contains("ite"));
        assert!(smt.contains("extract"));
    }

    #[test]
    fn test_ctpop() {
        let cfg = BitvecConfig::new(8);
        let result = intrinsic_to_smt("ctpop", &["x".to_string()], &cfg);
        assert!(result.is_ok());
        let smt = result.unwrap();
        // Should sum bits with bvadd
        assert!(smt.contains("bvadd"));
        assert!(smt.contains("extract"));
    }

    #[test]
    fn test_bswap_32() {
        let cfg = BitvecConfig::new(32);
        let result = intrinsic_to_smt("bswap", &["x".to_string()], &cfg);
        assert!(result.is_ok());
        let smt = result.unwrap();
        // Should extract 4 bytes and concat
        assert!(smt.contains("extract"));
        assert!(smt.contains("concat"));
    }

    #[test]
    fn test_bitreverse() {
        let cfg = BitvecConfig::new(8);
        let result = intrinsic_to_smt("bitreverse", &["x".to_string()], &cfg);
        assert!(result.is_ok());
        let smt = result.unwrap();
        // Should extract bits and concat
        assert!(smt.contains("extract"));
        assert!(smt.contains("concat"));
    }

    #[test]
    fn test_identity_black_box() {
        let result = intrinsic_to_smt("black_box", &["expr".to_string()], &config());
        assert_eq!(result.unwrap(), "expr");
    }

    #[test]
    fn test_wrong_arity() {
        let result = intrinsic_to_smt("wrapping_add", &["x".to_string()], &config());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Expected 2"));
    }

    #[test]
    fn test_extract_operation() {
        assert_eq!(
            extract_operation("core_num_impl_u32_wrapping_add"),
            "wrapping_add"
        );
        assert_eq!(
            extract_operation("std_intrinsics_rotate_left"),
            "rotate_left"
        );
        assert_eq!(extract_operation("wrapping_mul"), "wrapping_mul");
    }

    #[test]
    fn test_rotate_left() {
        let cfg = BitvecConfig::new(32);
        let result = intrinsic_to_smt("rotate_left", &["x".to_string(), "n".to_string()], &cfg);
        assert!(result.is_ok());
        let smt = result.unwrap();
        // Should use bvor, bvshl, bvlshr for variable rotation
        assert!(smt.contains("bvor"));
        assert!(smt.contains("bvshl"));
        assert!(smt.contains("bvlshr"));
        // Should mask amount to valid range
        assert!(smt.contains("bvurem"));
    }

    #[test]
    fn test_rotate_right() {
        let cfg = BitvecConfig::new(32);
        let result = intrinsic_to_smt("rotate_right", &["x".to_string(), "n".to_string()], &cfg);
        assert!(result.is_ok());
        let smt = result.unwrap();
        // Should use bvor, bvshl, bvlshr for variable rotation
        assert!(smt.contains("bvor"));
        assert!(smt.contains("bvshl"));
        assert!(smt.contains("bvlshr"));
        // Should mask amount to valid range
        assert!(smt.contains("bvurem"));
    }

    #[test]
    fn test_rotate_left_encoding_correctness() {
        // Verify the encoding produces valid SMT-LIB2
        let result = encode_rotate_left("x", "n", 32);
        // Should be: (bvor (bvshl x (bvurem n width)) (bvlshr x (bvsub width (bvurem n width))))
        assert!(result.starts_with("(bvor"));
        assert!(result.contains("(_ bv32 32)")); // width constant
    }
}
