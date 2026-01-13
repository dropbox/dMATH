//! Axiomatic definitions for Rust intrinsic functions
//!
//! This module provides SMT-LIB2 axioms for common Rust intrinsic functions
//! like `wrapping_add`, `saturating_add`, `checked_add`, etc.
//!
//! Without axioms, these functions are treated as uninterpreted (arbitrary behavior),
//! which is sound but prevents verification of programs using them.
//! With axioms, Z3 can reason precisely about their behavior.
//!
//! # Supported Functions
//!
//! - `wrapping_add`, `wrapping_sub`, `wrapping_mul` - Modular arithmetic
//! - `saturating_add`, `saturating_sub`, `saturating_mul` - Clamped arithmetic
//! - `checked_add`, `checked_sub`, `checked_mul`, `checked_div`, `checked_rem` - Option-returning arithmetic
//! - `overflowing_add`, `overflowing_sub`, `overflowing_mul` - Tuple-returning arithmetic
//!
//! # Example
//!
//! For u8::wrapping_add, the axiom is:
//! ```text
//! (assert (forall ((a Int) (b Int))
//!   (= (core_num_impl_u8_wrapping_add a b)
//!      (mod (+ a b) 256))))
//! ```

use std::collections::HashMap;

/// Generate SMT expression for 2^n where n is bounded by max_shift.
/// Uses nested ite to compute powers of 2 without SMT exponentiation.
/// For shift amounts outside [0, max_shift), returns 0.
fn power_of_2_expr(shift_var: &str, max_shift: u32) -> String {
    // Build from inside out: (ite (= n max_shift-1) 2^(max_shift-1) 0)
    let mut expr = "0".to_string();
    for i in (0..max_shift).rev() {
        let power = 1u128 << i;
        expr = format!("(ite (= {} {}) {} {})", shift_var, i, power, expr);
    }
    expr
}

/// Recognized Rust intrinsic function patterns
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntrinsicKind {
    /// wrapping_add - modular addition
    WrappingAdd { bits: u32, signed: bool },
    /// wrapping_sub - modular subtraction
    WrappingSub { bits: u32, signed: bool },
    /// wrapping_mul - modular multiplication
    WrappingMul { bits: u32, signed: bool },
    /// saturating_add - clamped addition
    SaturatingAdd { bits: u32, signed: bool },
    /// saturating_sub - clamped subtraction
    SaturatingSub { bits: u32, signed: bool },
    /// saturating_mul - clamped multiplication
    SaturatingMul { bits: u32, signed: bool },
    /// saturating_div - clamped division (signed MIN / -1 = MAX)
    SaturatingDiv { bits: u32, signed: bool },
    /// checked_add - returns `Option<T>`, None on overflow
    CheckedAdd { bits: u32, signed: bool },
    /// checked_sub - returns `Option<T>`, None on underflow
    CheckedSub { bits: u32, signed: bool },
    /// checked_mul - returns `Option<T>`, None on overflow
    CheckedMul { bits: u32, signed: bool },
    /// checked_div - returns `Option<T>`, None on div-by-zero or overflow
    CheckedDiv { bits: u32, signed: bool },
    /// checked_rem - returns `Option<T>`, None on div-by-zero or overflow
    CheckedRem { bits: u32, signed: bool },
    /// overflowing_add - returns (T, bool), wrapping value + overflow flag
    OverflowingAdd { bits: u32, signed: bool },
    /// overflowing_sub - returns (T, bool), wrapping value + underflow flag
    OverflowingSub { bits: u32, signed: bool },
    /// overflowing_mul - returns (T, bool), wrapping value + overflow flag
    OverflowingMul { bits: u32, signed: bool },
    /// abs - absolute value (signed types only)
    Abs { bits: u32 },
    /// signum - sign of the number (-1, 0, or 1)
    Signum,
    /// min - minimum of two values
    Min { signed: bool },
    /// max - maximum of two values
    Max { signed: bool },
    /// rotate_left - bitwise left rotation
    RotateLeft { bits: u32 },
    /// rotate_right - bitwise right rotation
    RotateRight { bits: u32 },
    /// count_ones - population count (number of 1 bits)
    CountOnes { bits: u32 },
    /// leading_zeros - count of leading zero bits
    LeadingZeros { bits: u32 },
    /// trailing_zeros - count of trailing zero bits
    TrailingZeros { bits: u32 },
    /// swap_bytes - reverse byte order
    SwapBytes { bits: u32 },
    /// reverse_bits - reverse bit order
    ReverseBits { bits: u32 },
    /// wrapping_div - division with wrapping for signed MIN / -1
    WrappingDiv { bits: u32, signed: bool },
    /// wrapping_rem - remainder with wrapping for signed MIN % -1
    WrappingRem { bits: u32, signed: bool },
    /// wrapping_neg - negation with wrapping (0 - x mod 2^n for unsigned)
    /// For signed types, just negate (overflow at MIN is handled separately)
    WrappingNeg { bits: u32, signed: bool },
    /// checked_neg - returns `Option<T>`, None if negation would overflow
    CheckedNeg { bits: u32, signed: bool },
    /// overflowing_neg - returns (T, bool), wrapping value + overflow flag
    OverflowingNeg { bits: u32, signed: bool },
    /// wrapping_shl - left shift with wrapping (shift amount mod bits)
    WrappingShl { bits: u32 },
    /// wrapping_shr - right shift with wrapping (shift amount mod bits)
    WrappingShr { bits: u32, signed: bool },
    /// checked_shl - returns `Option<T>`, None if shift >= bits
    CheckedShl { bits: u32 },
    /// checked_shr - returns `Option<T>`, None if shift >= bits
    CheckedShr { bits: u32, signed: bool },
    /// overflowing_shl - returns (T, bool), value + overflow flag if shift >= bits
    OverflowingShl { bits: u32 },
    /// overflowing_shr - returns (T, bool), value + overflow flag if shift >= bits
    OverflowingShr { bits: u32, signed: bool },
}

impl IntrinsicKind {
    /// Try to recognize an intrinsic from a sanitized SMT function name
    ///
    /// Sanitized names look like:
    /// - `core_num_impl_u8_wrapping_add`
    /// - `core_num_impl_i32_saturating_sub`
    pub fn from_smt_name(name: &str) -> Option<Self> {
        // Pattern: core_num_impl_{type}_{operation}
        // or: core_num_{type}_{operation} (alternative MIR format)
        let parts: Vec<&str> = name.split('_').collect();

        // Find the type indicator (u8, i32, etc.)
        let mut type_idx = None;
        let mut bits = 0u32;
        let mut signed = false;

        for (i, part) in parts.iter().enumerate() {
            if let Some((s, b)) = parse_int_type(part) {
                type_idx = Some(i);
                bits = b;
                signed = s;
                break;
            }
        }

        let type_idx = type_idx?;

        // Get the operation name (everything after the type)
        // No intermediate Vec allocation - join works directly on slices
        let op_name = parts[type_idx + 1..].join("_");

        match op_name.as_str() {
            "wrapping_add" => Some(IntrinsicKind::WrappingAdd { bits, signed }),
            "wrapping_sub" => Some(IntrinsicKind::WrappingSub { bits, signed }),
            "wrapping_mul" => Some(IntrinsicKind::WrappingMul { bits, signed }),
            "saturating_add" => Some(IntrinsicKind::SaturatingAdd { bits, signed }),
            "saturating_sub" => Some(IntrinsicKind::SaturatingSub { bits, signed }),
            "saturating_mul" => Some(IntrinsicKind::SaturatingMul { bits, signed }),
            "saturating_div" => Some(IntrinsicKind::SaturatingDiv { bits, signed }),
            "checked_add" => Some(IntrinsicKind::CheckedAdd { bits, signed }),
            "checked_sub" => Some(IntrinsicKind::CheckedSub { bits, signed }),
            "checked_mul" => Some(IntrinsicKind::CheckedMul { bits, signed }),
            "checked_div" => Some(IntrinsicKind::CheckedDiv { bits, signed }),
            "checked_rem" => Some(IntrinsicKind::CheckedRem { bits, signed }),
            "overflowing_add" => Some(IntrinsicKind::OverflowingAdd { bits, signed }),
            "overflowing_sub" => Some(IntrinsicKind::OverflowingSub { bits, signed }),
            "overflowing_mul" => Some(IntrinsicKind::OverflowingMul { bits, signed }),
            "abs" => Some(IntrinsicKind::Abs { bits }),
            "signum" => Some(IntrinsicKind::Signum),
            "min" => Some(IntrinsicKind::Min { signed }),
            "max" => Some(IntrinsicKind::Max { signed }),
            "rotate_left" => Some(IntrinsicKind::RotateLeft { bits }),
            "rotate_right" => Some(IntrinsicKind::RotateRight { bits }),
            "count_ones" => Some(IntrinsicKind::CountOnes { bits }),
            "leading_zeros" => Some(IntrinsicKind::LeadingZeros { bits }),
            "trailing_zeros" => Some(IntrinsicKind::TrailingZeros { bits }),
            "swap_bytes" => Some(IntrinsicKind::SwapBytes { bits }),
            "reverse_bits" => Some(IntrinsicKind::ReverseBits { bits }),
            "wrapping_div" => Some(IntrinsicKind::WrappingDiv { bits, signed }),
            "wrapping_rem" => Some(IntrinsicKind::WrappingRem { bits, signed }),
            "wrapping_neg" => Some(IntrinsicKind::WrappingNeg { bits, signed }),
            "checked_neg" => Some(IntrinsicKind::CheckedNeg { bits, signed }),
            "overflowing_neg" => Some(IntrinsicKind::OverflowingNeg { bits, signed }),
            "wrapping_shl" => Some(IntrinsicKind::WrappingShl { bits }),
            "wrapping_shr" => Some(IntrinsicKind::WrappingShr { bits, signed }),
            "checked_shl" => Some(IntrinsicKind::CheckedShl { bits }),
            "checked_shr" => Some(IntrinsicKind::CheckedShr { bits, signed }),
            "overflowing_shl" => Some(IntrinsicKind::OverflowingShl { bits }),
            "overflowing_shr" => Some(IntrinsicKind::OverflowingShr { bits, signed }),
            _ => None,
        }
    }

    /// Return the expected number of arguments for this intrinsic
    pub fn arity(&self) -> usize {
        match self {
            // Unary operations
            IntrinsicKind::Abs { .. }
            | IntrinsicKind::Signum
            | IntrinsicKind::CountOnes { .. }
            | IntrinsicKind::LeadingZeros { .. }
            | IntrinsicKind::TrailingZeros { .. }
            | IntrinsicKind::SwapBytes { .. }
            | IntrinsicKind::ReverseBits { .. }
            | IntrinsicKind::WrappingNeg { .. }
            | IntrinsicKind::CheckedNeg { .. }
            | IntrinsicKind::OverflowingNeg { .. } => 1,
            // Binary operations
            IntrinsicKind::WrappingAdd { .. }
            | IntrinsicKind::WrappingSub { .. }
            | IntrinsicKind::WrappingMul { .. }
            | IntrinsicKind::WrappingDiv { .. }
            | IntrinsicKind::WrappingRem { .. }
            | IntrinsicKind::SaturatingAdd { .. }
            | IntrinsicKind::SaturatingSub { .. }
            | IntrinsicKind::SaturatingMul { .. }
            | IntrinsicKind::SaturatingDiv { .. }
            | IntrinsicKind::CheckedAdd { .. }
            | IntrinsicKind::CheckedSub { .. }
            | IntrinsicKind::CheckedMul { .. }
            | IntrinsicKind::CheckedDiv { .. }
            | IntrinsicKind::CheckedRem { .. }
            | IntrinsicKind::OverflowingAdd { .. }
            | IntrinsicKind::OverflowingSub { .. }
            | IntrinsicKind::OverflowingMul { .. }
            | IntrinsicKind::Min { .. }
            | IntrinsicKind::Max { .. }
            | IntrinsicKind::RotateLeft { .. }
            | IntrinsicKind::RotateRight { .. }
            | IntrinsicKind::WrappingShl { .. }
            | IntrinsicKind::WrappingShr { .. }
            | IntrinsicKind::CheckedShl { .. }
            | IntrinsicKind::CheckedShr { .. }
            | IntrinsicKind::OverflowingShl { .. }
            | IntrinsicKind::OverflowingShr { .. } => 2,
        }
    }

    /// Generate an inline SMT expression for this intrinsic
    ///
    /// This is preferred over axioms because Spacer can reason about inline
    /// expressions directly, whereas axioms on uninterpreted functions cause
    /// it to return UNKNOWN.
    ///
    /// # Arguments
    /// * `args` - SMT expressions for the arguments
    ///
    /// # Returns
    /// An SMT-LIB2 expression string, or None if arity doesn't match
    pub fn inline_expression(&self, args: &[&str]) -> Option<String> {
        if args.len() != self.arity() {
            return None;
        }

        Some(match self {
            IntrinsicKind::WrappingAdd { bits, signed } => {
                let modulus = 1u128 << bits;
                if *signed {
                    // For signed integers, we need to wrap to signed range [-2^(n-1), 2^(n-1))
                    // Formula: ((a + b + half_range) mod range) - half_range
                    // This converts the result to two's complement representation
                    let half_range = 1u128 << (bits - 1);
                    format!(
                        "(- (mod (+ (+ {} {}) {}) {}) {})",
                        args[0], args[1], half_range, modulus, half_range
                    )
                } else {
                    format!("(mod (+ {} {}) {})", args[0], args[1], modulus)
                }
            }
            IntrinsicKind::WrappingSub { bits, signed } => {
                let modulus = 1u128 << bits;
                if *signed {
                    // For signed integers, wrap to signed range
                    let half_range = 1u128 << (bits - 1);
                    format!(
                        "(- (mod (+ (- {} {}) {}) {}) {})",
                        args[0], args[1], half_range, modulus, half_range
                    )
                } else {
                    // wrapping_sub: (a - b) mod 2^n
                    // In SMT-LIB2, mod always returns non-negative
                    format!(
                        "(mod (+ (- {} {}) {}) {})",
                        args[0], args[1], modulus, modulus
                    )
                }
            }
            IntrinsicKind::WrappingMul { bits, signed } => {
                let modulus = 1u128 << bits;
                if *signed {
                    // For signed integers, wrap to signed range
                    let half_range = 1u128 << (bits - 1);
                    format!(
                        "(- (mod (+ (* {} {}) {}) {}) {})",
                        args[0], args[1], half_range, modulus, half_range
                    )
                } else {
                    format!("(mod (* {} {}) {})", args[0], args[1], modulus)
                }
            }
            IntrinsicKind::SaturatingAdd { bits, signed } => {
                if *signed {
                    let max = (1i128 << (bits - 1)) - 1;
                    let min = -(1i128 << (bits - 1));
                    format!(
                        "(ite (> (+ {} {}) {}) {} \
                              (ite (< (+ {} {}) {}) {} (+ {} {})))",
                        args[0], args[1], max, max, args[0], args[1], min, min, args[0], args[1]
                    )
                } else {
                    let max = (1u128 << bits) - 1;
                    format!(
                        "(ite (> (+ {} {}) {}) {} \
                              (ite (< (+ {} {}) 0) 0 (+ {} {})))",
                        args[0], args[1], max, max, args[0], args[1], args[0], args[1]
                    )
                }
            }
            IntrinsicKind::SaturatingSub { bits, signed } => {
                if *signed {
                    let max = (1i128 << (bits - 1)) - 1;
                    let min = -(1i128 << (bits - 1));
                    format!(
                        "(ite (> (- {} {}) {}) {} \
                              (ite (< (- {} {}) {}) {} (- {} {})))",
                        args[0], args[1], max, max, args[0], args[1], min, min, args[0], args[1]
                    )
                } else {
                    // Unsigned saturating_sub: max(0, a - b)
                    format!(
                        "(ite (< (- {} {}) 0) 0 (- {} {}))",
                        args[0], args[1], args[0], args[1]
                    )
                }
            }
            IntrinsicKind::SaturatingMul { bits, signed } => {
                let product = format!("(* {} {})", args[0], args[1]);
                if *signed {
                    let max = (1i128 << (bits - 1)) - 1;
                    let min = -(1i128 << (bits - 1));
                    format!(
                        "(ite (> {} {}) {} (ite (< {} {}) {} {}))",
                        product, max, max, product, min, min, product
                    )
                } else {
                    let max = (1u128 << bits) - 1;
                    format!(
                        "(ite (> {} {}) {} (ite (< {} 0) 0 {}))",
                        product, max, max, product, product
                    )
                }
            }
            IntrinsicKind::SaturatingDiv { bits, signed } => {
                // saturating_div: For unsigned, just regular div (can't overflow).
                // For signed, MIN / -1 would be -MIN = MAX+1, so saturate to MAX.
                if *signed {
                    let max = (1i128 << (bits - 1)) - 1;
                    let min = -(1i128 << (bits - 1));
                    format!(
                        "(ite (and (= {} {}) (= {} (- 1))) {} (div {} {}))",
                        args[0], min, args[1], max, args[0], args[1]
                    )
                } else {
                    // Unsigned division can't overflow
                    format!("(div {} {})", args[0], args[1])
                }
            }
            IntrinsicKind::Abs { bits } => {
                // abs(x) = if x < 0 then -x else x
                // Note: For signed types, abs(MIN) can overflow (e.g., abs(-128i8) = 128 which doesn't fit in i8)
                // We encode the mathematical absolute value here; overflow checking is separate
                let min = -(1i128 << (bits - 1));
                format!(
                    "(ite (< {} 0) (ite (= {} {}) {} (- {})) {})",
                    args[0], args[0], min, min, args[0], args[0]
                )
            }
            IntrinsicKind::Signum => {
                // signum(x) = -1 if x < 0, 0 if x == 0, 1 if x > 0
                format!("(ite (< {} 0) (- 1) (ite (> {} 0) 1 0))", args[0], args[0])
            }
            IntrinsicKind::Min { signed: _ } => {
                // min(a, b) = if a <= b then a else b
                format!("(ite (<= {} {}) {} {})", args[0], args[1], args[0], args[1])
            }
            IntrinsicKind::Max { signed: _ } => {
                // max(a, b) = if a >= b then a else b
                format!("(ite (>= {} {}) {} {})", args[0], args[1], args[0], args[1])
            }
            IntrinsicKind::RotateLeft { bits } => {
                // rotate_left(x, n) = ((x << n) | (x >> (bits - n))) mod 2^bits
                // For SMT-LIB2 with integers, we compute this arithmetically:
                // x * 2^n mod 2^bits + x div 2^(bits-n)
                let modulus = 1u128 << bits;
                format!(
                    "(mod (+ (* {} (^ 2 {})) (div {} (^ 2 (- {} {})))) {})",
                    args[0], args[1], args[0], bits, args[1], modulus
                )
            }
            IntrinsicKind::RotateRight { bits } => {
                // rotate_right(x, n) = ((x >> n) | (x << (bits - n))) mod 2^bits
                let modulus = 1u128 << bits;
                format!(
                    "(mod (+ (div {} (^ 2 {})) (* {} (^ 2 (- {} {})))) {})",
                    args[0], args[1], args[0], bits, args[1], modulus
                )
            }
            IntrinsicKind::CountOnes { bits: _ } => {
                // count_ones is hard to express in pure SMT-LIB2 without bitvectors
                // For now, we use an uninterpreted function placeholder
                // This will be handled by axioms or left abstract
                format!("(popcount {})", args[0])
            }
            IntrinsicKind::LeadingZeros { bits: _ } => {
                // Also hard to express; use uninterpreted placeholder
                format!("(leading_zeros {})", args[0])
            }
            IntrinsicKind::TrailingZeros { bits: _ } => {
                format!("(trailing_zeros {})", args[0])
            }
            IntrinsicKind::SwapBytes { bits: _ } => {
                format!("(swap_bytes {})", args[0])
            }
            IntrinsicKind::ReverseBits { bits: _ } => {
                format!("(reverse_bits {})", args[0])
            }
            IntrinsicKind::WrappingDiv { bits, signed } => {
                // wrapping_div: For unsigned, just regular div.
                // For signed, the only wrapping case is MIN / -1 = MIN (not representable, wraps)
                // Since we use unbounded integers, we model the wrapping behavior:
                // result = if (a == MIN and b == -1) then MIN else (a / b)
                if *signed {
                    let min = -(1i128 << (bits - 1));
                    format!(
                        "(ite (and (= {} {}) (= {} (- 1))) {} (div {} {}))",
                        args[0], min, args[1], min, args[0], args[1]
                    )
                } else {
                    // Unsigned division doesn't wrap
                    format!("(div {} {})", args[0], args[1])
                }
            }
            IntrinsicKind::WrappingRem { bits, signed } => {
                // wrapping_rem: For unsigned, just regular mod.
                // For signed, MIN % -1 = 0 (special case)
                if *signed {
                    let min = -(1i128 << (bits - 1));
                    format!(
                        "(ite (and (= {} {}) (= {} (- 1))) 0 (mod {} {}))",
                        args[0], min, args[1], args[0], args[1]
                    )
                } else {
                    format!("(mod {} {})", args[0], args[1])
                }
            }
            IntrinsicKind::WrappingNeg { bits, signed } => {
                if *signed {
                    // For signed types, just negate. The CHC encoding uses Int (unbounded),
                    // so -42 is valid. Overflow at MIN is a soundness limitation we document.
                    format!("(- 0 {})", args[0])
                } else {
                    // For unsigned: wrapping_neg is (0 - x) mod 2^bits
                    let modulus = 1u128 << bits;
                    format!("(mod (- 0 {}) {})", args[0], modulus)
                }
            }
            IntrinsicKind::WrappingShl { bits } => {
                // wrapping_shl: (x << (n mod bits))
                // In SMT-LIB2: x * 2^(n mod bits) mod 2^bits
                // Note: SMT-LIB2 doesn't have integer exponentiation, so we use ite chain
                let modulus = 1u128 << bits;
                let shift_mod = format!("(mod {} {})", args[1], bits);
                let power_expr = power_of_2_expr(&shift_mod, *bits);
                format!("(mod (* {} {}) {})", args[0], power_expr, modulus)
            }
            IntrinsicKind::WrappingShr { bits, signed } => {
                // wrapping_shr: (x >> (n mod bits))
                // In SMT-LIB2: x div 2^(n mod bits)
                // Note: SMT-LIB2 doesn't have integer exponentiation, so we use ite chain
                let shift_mod = format!("(mod {} {})", args[1], bits);
                let power_expr = power_of_2_expr(&shift_mod, *bits);
                if *signed {
                    // For signed, arithmetic right shift preserves sign
                    // We use integer division which rounds toward negative infinity
                    format!("(div {} {})", args[0], power_expr)
                } else {
                    format!("(div {} {})", args[0], power_expr)
                }
            }
            // Checked operations return Option<T>, which is a compound type.
            // They cannot be inlined as simple expressions. Instead, they should
            // be handled at the MIR parsing level similar to XWithOverflow operations.
            // Return None to indicate they cannot be inlined.
            IntrinsicKind::CheckedAdd { .. }
            | IntrinsicKind::CheckedSub { .. }
            | IntrinsicKind::CheckedMul { .. }
            | IntrinsicKind::CheckedDiv { .. }
            | IntrinsicKind::CheckedRem { .. }
            | IntrinsicKind::CheckedNeg { .. }
            | IntrinsicKind::CheckedShl { .. }
            | IntrinsicKind::CheckedShr { .. } => {
                return None;
            }
            // Overflowing operations return (T, bool), also a compound type.
            // They should be handled at the MIR parsing level to split into
            // separate value and overflow flag variables.
            IntrinsicKind::OverflowingAdd { .. }
            | IntrinsicKind::OverflowingSub { .. }
            | IntrinsicKind::OverflowingMul { .. }
            | IntrinsicKind::OverflowingNeg { .. }
            | IntrinsicKind::OverflowingShl { .. }
            | IntrinsicKind::OverflowingShr { .. } => {
                return None;
            }
        })
    }

    /// Returns true if this intrinsic returns a compound type (like `Option<T>` or `(T, bool)`)
    /// that cannot be inlined as a simple expression and requires special handling.
    pub fn returns_compound_type(&self) -> bool {
        matches!(
            self,
            IntrinsicKind::CheckedAdd { .. }
                | IntrinsicKind::CheckedSub { .. }
                | IntrinsicKind::CheckedMul { .. }
                | IntrinsicKind::CheckedDiv { .. }
                | IntrinsicKind::CheckedRem { .. }
                | IntrinsicKind::CheckedNeg { .. }
                | IntrinsicKind::CheckedShl { .. }
                | IntrinsicKind::CheckedShr { .. }
                | IntrinsicKind::OverflowingAdd { .. }
                | IntrinsicKind::OverflowingSub { .. }
                | IntrinsicKind::OverflowingMul { .. }
                | IntrinsicKind::OverflowingNeg { .. }
                | IntrinsicKind::OverflowingShl { .. }
                | IntrinsicKind::OverflowingShr { .. }
        )
    }

    /// Truncating division that matches Rust semantics (round toward zero).
    /// Assumes denominator is non-zero; callers should guard divide-by-zero explicitly.
    fn div_toward_zero(a: &str, b: &str) -> String {
        format!("(ite (>= {a} 0) (div {a} {b}) (- (div (- {a}) {b})))")
    }

    /// Remainder consistent with Rust semantics (same sign as dividend).
    /// Assumes denominator is non-zero; callers should guard divide-by-zero explicitly.
    fn rem_toward_zero(a: &str, b: &str) -> String {
        let quotient = Self::div_toward_zero(a, b);
        format!("(- {a} (* {quotient} {b}))")
    }

    /// For checked operations, generate the overflow condition and value expressions
    /// separately. Returns (overflow_condition, value_expression) where:
    /// - overflow_condition is true when overflow would occur (None case)
    /// - value_expression is the computed value (only meaningful when no overflow)
    ///
    /// Returns None if this is not a checked operation or arity doesn't match.
    pub fn checked_components(&self, args: &[&str]) -> Option<(String, String)> {
        if args.len() != self.arity() {
            return None;
        }

        match self {
            IntrinsicKind::CheckedAdd { bits, signed } => {
                let a = args[0];
                let b = args[1];
                let value = format!("(+ {} {})", a, b);
                let overflow = if *signed {
                    let max = (1i128 << (bits - 1)) - 1;
                    let min = -(1i128 << (bits - 1));
                    format!("(or (> {} {}) (< {} {}))", value, max, value, min)
                } else {
                    let max = (1u128 << bits) - 1;
                    format!("(or (> {} {}) (< {} 0))", value, max, value)
                };
                Some((overflow, value))
            }
            IntrinsicKind::CheckedSub { bits, signed } => {
                let a = args[0];
                let b = args[1];
                let value = format!("(- {} {})", a, b);
                let overflow = if *signed {
                    let max = (1i128 << (bits - 1)) - 1;
                    let min = -(1i128 << (bits - 1));
                    format!("(or (> {} {}) (< {} {}))", value, max, value, min)
                } else {
                    // Unsigned: underflow when a < b
                    format!("(< {} 0)", value)
                };
                Some((overflow, value))
            }
            IntrinsicKind::CheckedMul { bits, signed } => {
                let a = args[0];
                let b = args[1];
                let value = format!("(* {} {})", a, b);
                let overflow = if *signed {
                    let max = (1i128 << (bits - 1)) - 1;
                    let min = -(1i128 << (bits - 1));
                    format!("(or (> {} {}) (< {} {}))", value, max, value, min)
                } else {
                    let max = (1u128 << bits) - 1;
                    format!("(or (> {} {}) (< {} 0))", value, max, value)
                };
                Some((overflow, value))
            }
            IntrinsicKind::CheckedDiv { bits, signed } => {
                let a = args[0];
                let b = args[1];
                // Guard value to avoid division by zero in the SMT term
                let value = format!("(ite (= {} 0) 0 {})", b, Self::div_toward_zero(a, b));
                let overflow = if *signed {
                    let min = -(1i128 << (bits - 1));
                    format!("(or (= {} 0) (and (= {} {}) (= {} (- 1))))", b, a, min, b)
                } else {
                    format!("(= {} 0)", b)
                };
                Some((overflow, value))
            }
            IntrinsicKind::CheckedRem { bits, signed } => {
                let a = args[0];
                let b = args[1];
                // Guard value to avoid division by zero in the SMT term
                let value = format!("(ite (= {} 0) 0 {})", b, Self::rem_toward_zero(a, b));
                let overflow = if *signed {
                    let min = -(1i128 << (bits - 1));
                    format!("(or (= {} 0) (and (= {} {}) (= {} (- 1))))", b, a, min, b)
                } else {
                    format!("(= {} 0)", b)
                };
                Some((overflow, value))
            }
            IntrinsicKind::CheckedNeg { bits, signed } => {
                let x = args[0];
                // Negation: -x
                let value = format!("(- 0 {})", x);
                let overflow = if *signed {
                    // Signed negation overflows only when x == MIN (because -MIN > MAX)
                    let min = -(1i128 << (bits - 1));
                    format!("(= {} {})", x, min)
                } else {
                    // Unsigned negation overflows when x != 0 (since -x < 0)
                    format!("(not (= {} 0))", x)
                };
                Some((overflow, value))
            }
            IntrinsicKind::CheckedShl { bits } => {
                let x = args[0];
                let n = args[1];
                // Shift: x << n (only valid if n < bits)
                let modulus = 1u128 << bits;
                let value = format!("(mod (* {} (^ 2 {})) {})", x, n, modulus);
                // Overflow when shift amount >= bit width
                let overflow = format!("(>= {} {})", n, bits);
                Some((overflow, value))
            }
            IntrinsicKind::CheckedShr { bits, signed } => {
                let x = args[0];
                let n = args[1];
                // Shift: x >> n (only valid if n < bits)
                let value = if *signed {
                    // Arithmetic right shift (preserves sign)
                    format!("(div {} (^ 2 {}))", x, n)
                } else {
                    format!("(div {} (^ 2 {}))", x, n)
                };
                // Overflow when shift amount >= bit width
                let overflow = format!("(>= {} {})", n, bits);
                Some((overflow, value))
            }
            _ => None,
        }
    }

    /// For overflowing operations, generate the wrapped value and overflow flag expressions.
    /// Returns (wrapped_value, overflow_flag) where:
    /// - wrapped_value is the result with wrapping semantics (always valid)
    /// - overflow_flag is true when overflow/underflow occurred
    ///
    /// Returns None if this is not an overflowing operation or arity doesn't match.
    pub fn overflowing_components(&self, args: &[&str]) -> Option<(String, String)> {
        if args.len() != self.arity() {
            return None;
        }

        match self {
            IntrinsicKind::OverflowingAdd { bits, signed } => {
                let a = args[0];
                let b = args[1];
                let modulus = 1u128 << bits;
                // Wrapped value: (a + b) mod 2^bits
                let wrapped = format!("(mod (+ {} {}) {})", a, b, modulus);
                // Overflow: mathematical sum exceeds type range
                let unwrapped = format!("(+ {} {})", a, b);
                let overflow = if *signed {
                    let max = (1i128 << (bits - 1)) - 1;
                    let min = -(1i128 << (bits - 1));
                    format!("(or (> {} {}) (< {} {}))", unwrapped, max, unwrapped, min)
                } else {
                    let max = (1u128 << bits) - 1;
                    format!("(> {} {})", unwrapped, max)
                };
                Some((wrapped, overflow))
            }
            IntrinsicKind::OverflowingSub { bits, signed } => {
                let a = args[0];
                let b = args[1];
                let modulus = 1u128 << bits;
                // Wrapped value: (a - b + modulus) mod modulus (to handle negative results)
                let wrapped = format!("(mod (+ (- {} {}) {}) {})", a, b, modulus, modulus);
                // Overflow/underflow detection
                let unwrapped = format!("(- {} {})", a, b);
                let overflow = if *signed {
                    let max = (1i128 << (bits - 1)) - 1;
                    let min = -(1i128 << (bits - 1));
                    format!("(or (> {} {}) (< {} {}))", unwrapped, max, unwrapped, min)
                } else {
                    // Unsigned: underflow when a < b
                    format!("(< {} 0)", unwrapped)
                };
                Some((wrapped, overflow))
            }
            IntrinsicKind::OverflowingMul { bits, signed } => {
                let a = args[0];
                let b = args[1];
                let modulus = 1u128 << bits;
                // Wrapped value: (a * b) mod 2^bits
                let wrapped = format!("(mod (* {} {}) {})", a, b, modulus);
                // Overflow detection
                let unwrapped = format!("(* {} {})", a, b);
                let overflow = if *signed {
                    let max = (1i128 << (bits - 1)) - 1;
                    let min = -(1i128 << (bits - 1));
                    format!("(or (> {} {}) (< {} {}))", unwrapped, max, unwrapped, min)
                } else {
                    let max = (1u128 << bits) - 1;
                    format!("(> {} {})", unwrapped, max)
                };
                Some((wrapped, overflow))
            }
            IntrinsicKind::OverflowingNeg { bits, signed } => {
                let x = args[0];
                let modulus = 1u128 << bits;
                // Wrapped value: (0 - x) mod 2^bits
                let wrapped = format!("(mod (- 0 {}) {})", x, modulus);
                // Overflow detection
                let overflow = if *signed {
                    // Signed negation overflows only when x == MIN
                    let min = -(1i128 << (bits - 1));
                    format!("(= {} {})", x, min)
                } else {
                    // Unsigned negation overflows when x != 0
                    format!("(not (= {} 0))", x)
                };
                Some((wrapped, overflow))
            }
            IntrinsicKind::OverflowingShl { bits } => {
                let x = args[0];
                let n = args[1];
                let modulus = 1u128 << bits;
                // Wrapped value: (x << (n mod bits)) mod 2^bits
                let wrapped = format!("(mod (* {} (^ 2 (mod {} {}))) {})", x, n, bits, modulus);
                // Overflow when shift amount >= bit width
                let overflow = format!("(>= {} {})", n, bits);
                Some((wrapped, overflow))
            }
            IntrinsicKind::OverflowingShr { bits, signed: _ } => {
                let x = args[0];
                let n = args[1];
                // Wrapped value: x >> (n mod bits)
                // Note: SMT-LIB2's div rounds toward negative infinity, which gives
                // correct arithmetic shift semantics for both signed and unsigned.
                // The signedness affects interpretation but not the formula in Z3's Int theory.
                let wrapped = format!("(div {} (^ 2 (mod {} {})))", x, n, bits);
                // Overflow when shift amount >= bit width
                let overflow = format!("(>= {} {})", n, bits);
                Some((wrapped, overflow))
            }
            _ => None,
        }
    }

    /// Generate SMT-LIB2 axiom for this intrinsic
    ///
    /// Returns an assertion string that defines the function's behavior
    pub fn generate_axiom(&self, func_name: &str) -> String {
        match self {
            IntrinsicKind::WrappingAdd { bits, signed } => {
                let modulus = 1u128 << bits;
                if *signed {
                    let half_range = 1u128 << (bits - 1);
                    format!(
                        "(assert (forall ((a Int) (b Int)) (= ({func_name} a b) (- (mod (+ (+ a b) {half_range}) {modulus}) {half_range}))))"
                    )
                } else {
                    format!(
                        "(assert (forall ((a Int) (b Int)) (= ({func_name} a b) (mod (+ a b) {modulus}))))"
                    )
                }
            }
            IntrinsicKind::WrappingSub { bits, signed } => {
                let modulus = 1u128 << bits;
                if *signed {
                    let half_range = 1u128 << (bits - 1);
                    format!(
                        "(assert (forall ((a Int) (b Int)) (= ({func_name} a b) (- (mod (+ (- a b) {half_range}) {modulus}) {half_range}))))"
                    )
                } else {
                    // wrapping_sub: (a - b) mod 2^n, but need to handle negative results
                    // In SMT-LIB2, mod always returns non-negative, so:
                    // wrapping_sub(a, b) = mod(a - b + modulus, modulus)
                    format!(
                        "(assert (forall ((a Int) (b Int)) (= ({func_name} a b) (mod (+ (- a b) {modulus}) {modulus}))))"
                    )
                }
            }
            IntrinsicKind::WrappingMul { bits, signed } => {
                let modulus = 1u128 << bits;
                if *signed {
                    let half_range = 1u128 << (bits - 1);
                    format!(
                        "(assert (forall ((a Int) (b Int)) (= ({func_name} a b) (- (mod (+ (* a b) {half_range}) {modulus}) {half_range}))))"
                    )
                } else {
                    format!(
                        "(assert (forall ((a Int) (b Int)) (= ({func_name} a b) (mod (* a b) {modulus}))))"
                    )
                }
            }
            IntrinsicKind::SaturatingAdd { bits, signed } => {
                if *signed {
                    let max = (1i128 << (bits - 1)) - 1;
                    let min = -(1i128 << (bits - 1));
                    format!(
                        "(assert (forall ((a Int) (b Int)) \
                         (= ({func_name} a b) \
                            (ite (> (+ a b) {max}) {max} \
                                 (ite (< (+ a b) {min}) {min} (+ a b))))))"
                    )
                } else {
                    let max = (1u128 << bits) - 1;
                    format!(
                        "(assert (forall ((a Int) (b Int)) \
                         (= ({func_name} a b) \
                            (ite (> (+ a b) {max}) {max} \
                                 (ite (< (+ a b) 0) 0 (+ a b))))))"
                    )
                }
            }
            IntrinsicKind::SaturatingSub { bits, signed } => {
                if *signed {
                    let max = (1i128 << (bits - 1)) - 1;
                    let min = -(1i128 << (bits - 1));
                    format!(
                        "(assert (forall ((a Int) (b Int)) \
                         (= ({func_name} a b) \
                            (ite (> (- a b) {max}) {max} \
                                 (ite (< (- a b) {min}) {min} (- a b))))))"
                    )
                } else {
                    // Unsigned saturating_sub: max(0, a - b)
                    format!(
                        "(assert (forall ((a Int) (b Int)) \
                         (= ({func_name} a b) (ite (< (- a b) 0) 0 (- a b)))))"
                    )
                }
            }
            IntrinsicKind::SaturatingMul { bits, signed } => {
                let product = "(* a b)";
                if *signed {
                    let max = (1i128 << (bits - 1)) - 1;
                    let min = -(1i128 << (bits - 1));
                    format!(
                        "(assert (forall ((a Int) (b Int)) \
                         (= ({func_name} a b) \
                            (ite (> {product} {max}) {max} \
                                 (ite (< {product} {min}) {min} {product})))))"
                    )
                } else {
                    let max = (1u128 << bits) - 1;
                    format!(
                        "(assert (forall ((a Int) (b Int)) \
                         (= ({func_name} a b) \
                            (ite (> {product} {max}) {max} \
                                 (ite (< {product} 0) 0 {product})))))"
                    )
                }
            }
            IntrinsicKind::SaturatingDiv { bits, signed } => {
                // saturating_div: For unsigned, regular division.
                // For signed, MIN / -1 = MAX (saturates instead of wrapping).
                if *signed {
                    let max = (1i128 << (bits - 1)) - 1;
                    let min = -(1i128 << (bits - 1));
                    format!(
                        "(assert (forall ((a Int) (b Int)) \
                         (= ({func_name} a b) \
                            (ite (and (= a {min}) (= b (- 1))) {max} (div a b)))))"
                    )
                } else {
                    format!(
                        "(assert (forall ((a Int) (b Int)) \
                         (= ({func_name} a b) (div a b))))"
                    )
                }
            }
            IntrinsicKind::Abs { bits } => {
                let min = -(1i128 << (bits - 1));
                format!(
                    "(assert (forall ((a Int)) \
                     (= ({func_name} a) \
                        (ite (< a 0) (ite (= a {min}) {min} (- a)) a))))"
                )
            }
            IntrinsicKind::Signum => {
                format!(
                    "(assert (forall ((a Int)) \
                     (= ({func_name} a) \
                        (ite (< a 0) (- 1) (ite (> a 0) 1 0)))))"
                )
            }
            IntrinsicKind::Min { .. } => {
                format!(
                    "(assert (forall ((a Int) (b Int)) \
                     (= ({func_name} a b) (ite (<= a b) a b))))"
                )
            }
            IntrinsicKind::Max { .. } => {
                format!(
                    "(assert (forall ((a Int) (b Int)) \
                     (= ({func_name} a b) (ite (>= a b) a b))))"
                )
            }
            IntrinsicKind::RotateLeft { bits } => {
                let modulus = 1u128 << bits;
                format!(
                    "(assert (forall ((x Int) (n Int)) \
                     (= ({func_name} x n) \
                        (mod (+ (* x (^ 2 n)) (div x (^ 2 (- {bits} n)))) {modulus}))))"
                )
            }
            IntrinsicKind::RotateRight { bits } => {
                let modulus = 1u128 << bits;
                format!(
                    "(assert (forall ((x Int) (n Int)) \
                     (= ({func_name} x n) \
                        (mod (+ (div x (^ 2 n)) (* x (^ 2 (- {bits} n)))) {modulus}))))"
                )
            }
            // These are hard to axiomatize without bitvectors; use uninterpreted
            IntrinsicKind::CountOnes { .. }
            | IntrinsicKind::LeadingZeros { .. }
            | IntrinsicKind::TrailingZeros { .. }
            | IntrinsicKind::SwapBytes { .. }
            | IntrinsicKind::ReverseBits { .. } => {
                // No axiom - left as uninterpreted function
                String::new()
            }
            // Wrapping division/remainder/negation
            IntrinsicKind::WrappingDiv { bits, signed } => {
                if *signed {
                    let min = -(1i128 << (bits - 1));
                    format!(
                        "(assert (forall ((a Int) (b Int)) \
                         (= ({func_name} a b) \
                            (ite (and (= a {min}) (= b (- 1))) {min} (div a b)))))"
                    )
                } else {
                    format!("(assert (forall ((a Int) (b Int)) (= ({func_name} a b) (div a b))))")
                }
            }
            IntrinsicKind::WrappingRem { bits, signed } => {
                if *signed {
                    let min = -(1i128 << (bits - 1));
                    format!(
                        "(assert (forall ((a Int) (b Int)) \
                         (= ({func_name} a b) \
                            (ite (and (= a {min}) (= b (- 1))) 0 (mod a b)))))"
                    )
                } else {
                    format!("(assert (forall ((a Int) (b Int)) (= ({func_name} a b) (mod a b))))")
                }
            }
            IntrinsicKind::WrappingNeg { bits, signed } => {
                if *signed {
                    // For signed types, just negate (overflow at MIN is a soundness limitation)
                    format!("(assert (forall ((a Int)) (= ({func_name} a) (- 0 a))))")
                } else {
                    // For unsigned: (0 - x) mod 2^bits
                    let modulus = 1u128 << bits;
                    format!(
                        "(assert (forall ((a Int)) (= ({func_name} a) (mod (- 0 a) {modulus}))))"
                    )
                }
            }
            // Wrapping shifts
            IntrinsicKind::WrappingShl { bits } => {
                let modulus = 1u128 << bits;
                format!(
                    "(assert (forall ((x Int) (n Int)) \
                     (= ({func_name} x n) (mod (* x (^ 2 (mod n {bits}))) {modulus}))))"
                )
            }
            IntrinsicKind::WrappingShr { bits, .. } => {
                format!(
                    "(assert (forall ((x Int) (n Int)) \
                     (= ({func_name} x n) (div x (^ 2 (mod n {bits}))))))"
                )
            }
            // Checked operations return Option<T>, which requires special handling.
            // They should be expanded at the MIR level into separate discriminant and
            // value components. When appearing as uninterpreted functions, no useful
            // axiom can be provided since they return compound types.
            IntrinsicKind::CheckedAdd { .. }
            | IntrinsicKind::CheckedSub { .. }
            | IntrinsicKind::CheckedMul { .. }
            | IntrinsicKind::CheckedDiv { .. }
            | IntrinsicKind::CheckedRem { .. }
            | IntrinsicKind::CheckedNeg { .. }
            | IntrinsicKind::CheckedShl { .. }
            | IntrinsicKind::CheckedShr { .. } => {
                // No axiom - requires MIR-level expansion
                String::new()
            }
            // Overflowing operations return (T, bool), which requires special handling.
            // They should be expanded at the MIR level into separate value and flag components.
            IntrinsicKind::OverflowingAdd { .. }
            | IntrinsicKind::OverflowingSub { .. }
            | IntrinsicKind::OverflowingMul { .. }
            | IntrinsicKind::OverflowingNeg { .. }
            | IntrinsicKind::OverflowingShl { .. }
            | IntrinsicKind::OverflowingShr { .. } => {
                // No axiom - requires MIR-level expansion
                String::new()
            }
        }
    }
}

/// Parse a Rust integer type string to extract signedness and bit width.
///
/// Returns `Some((signed, bits))` for recognized integer types, `None` otherwise.
/// Note: `usize` and `isize` are assumed to be 64-bit (common for modern systems).
///
/// # Examples
/// - `"u8"` → `Some((false, 8))`
/// - `"i32"` → `Some((true, 32))`
/// - `"usize"` → `Some((false, 64))`
fn parse_int_type(s: &str) -> Option<(bool, u32)> {
    match s {
        "u8" => Some((false, 8)),
        "u16" => Some((false, 16)),
        "u32" => Some((false, 32)),
        "u64" | "usize" => Some((false, 64)), // usize assumes 64-bit platform
        "u128" => Some((false, 128)),
        "i8" => Some((true, 8)),
        "i16" => Some((true, 16)),
        "i32" => Some((true, 32)),
        "i64" | "isize" => Some((true, 64)), // isize assumes 64-bit platform
        "i128" => Some((true, 128)),
        _ => None,
    }
}

/// Generate all applicable axioms for a set of function declarations
///
/// Takes a map of function names to their SMT signatures and returns
/// axiom assertions for any recognized intrinsics.
pub fn generate_intrinsic_axioms(function_names: &[&str]) -> Vec<String> {
    let mut axioms = Vec::new();

    for name in function_names {
        if let Some(kind) = IntrinsicKind::from_smt_name(name) {
            axioms.push(kind.generate_axiom(name));
        }
    }

    axioms
}

/// Check if a function name corresponds to a known intrinsic
pub fn is_known_intrinsic(name: &str) -> bool {
    IntrinsicKind::from_smt_name(name).is_some()
}

/// Try to inline a function call as an SMT expression
///
/// If the function is a recognized intrinsic (wrapping_add, etc.), returns
/// the inlined expression. Otherwise returns None and the function should
/// be treated as uninterpreted.
///
/// # Arguments
/// * `func_name` - The sanitized SMT function name
/// * `args` - The arguments to the function call
///
/// # Returns
/// * `Some(expr)` - Inlined SMT expression if the function is recognized
/// * `None` - Function is not a known intrinsic or wrong arity
pub fn try_inline_call(func_name: &str, args: &[String]) -> Option<String> {
    let kind = IntrinsicKind::from_smt_name(func_name)?;

    // Check if arity matches
    if args.len() != kind.arity() {
        return None;
    }

    // Convert owned Strings to &str references
    let args_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    kind.inline_expression(&args_refs)
}

/// Collect axioms for all recognized intrinsics in a CHC system's function declarations
pub fn collect_axioms_for_functions(
    functions: &HashMap<String, crate::UninterpretedFunction>,
) -> Vec<String> {
    let names: Vec<&str> = functions.keys().map(|s| s.as_str()).collect();
    generate_intrinsic_axioms(&names)
}

/// Try to expand a checked operation call into its components.
///
/// For checked operations (checked_add, checked_sub, checked_mul), this returns
/// the overflow condition and value expression that can be used to synthesize
/// the `Option<T>` result at the MIR level.
///
/// # Arguments
/// * `func_name` - The sanitized SMT function name
/// * `args` - The arguments to the function call
///
/// # Returns
/// * `Some((overflow_cond, value_expr))` - The overflow condition (true when None)
///   and the computed value (meaningful when Some)
/// * `None` - Function is not a checked operation or wrong arity
pub fn try_expand_checked_call(func_name: &str, args: &[String]) -> Option<(String, String)> {
    let kind = IntrinsicKind::from_smt_name(func_name)?;

    // Check if arity matches
    if args.len() != kind.arity() {
        return None;
    }

    // Convert owned Strings to &str references
    let args_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    kind.checked_components(&args_refs)
}

/// Try to expand an overflowing operation call into its components.
///
/// For overflowing operations (overflowing_add, overflowing_sub, overflowing_mul),
/// this returns the wrapped value and overflow flag that can be used to synthesize
/// the (T, bool) tuple result at the MIR level.
///
/// # Arguments
/// * `func_name` - The sanitized SMT function name
/// * `args` - The arguments to the function call
///
/// # Returns
/// * `Some((wrapped_value, overflow_flag))` - The wrapped result (always valid)
///   and the overflow flag (true when overflow occurred)
/// * `None` - Function is not an overflowing operation or wrong arity
pub fn try_expand_overflowing_call(func_name: &str, args: &[String]) -> Option<(String, String)> {
    let kind = IntrinsicKind::from_smt_name(func_name)?;

    // Check if arity matches
    if args.len() != kind.arity() {
        return None;
    }

    // Convert owned Strings to &str references
    let args_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    kind.overflowing_components(&args_refs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_wrapping_add_u8() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_u8_wrapping_add");
        assert_eq!(
            kind,
            Some(IntrinsicKind::WrappingAdd {
                bits: 8,
                signed: false
            })
        );
    }

    #[test]
    fn test_parse_wrapping_add_u32() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_u32_wrapping_add");
        assert_eq!(
            kind,
            Some(IntrinsicKind::WrappingAdd {
                bits: 32,
                signed: false
            })
        );
    }

    #[test]
    fn test_parse_saturating_add_i32() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_i32_saturating_add");
        assert_eq!(
            kind,
            Some(IntrinsicKind::SaturatingAdd {
                bits: 32,
                signed: true
            })
        );
    }

    #[test]
    fn test_parse_saturating_sub_u8() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_u8_saturating_sub");
        assert_eq!(
            kind,
            Some(IntrinsicKind::SaturatingSub {
                bits: 8,
                signed: false
            })
        );
    }

    #[test]
    fn test_parse_saturating_mul_i16() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_i16_saturating_mul");
        assert_eq!(
            kind,
            Some(IntrinsicKind::SaturatingMul {
                bits: 16,
                signed: true
            })
        );
    }

    #[test]
    fn test_parse_saturating_div_i32() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_i32_saturating_div");
        assert_eq!(
            kind,
            Some(IntrinsicKind::SaturatingDiv {
                bits: 32,
                signed: true
            })
        );
    }

    #[test]
    fn test_parse_saturating_div_u8() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_u8_saturating_div");
        assert_eq!(
            kind,
            Some(IntrinsicKind::SaturatingDiv {
                bits: 8,
                signed: false
            })
        );
    }

    #[test]
    fn test_parse_checked_div_i32() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_i32_checked_div");
        assert_eq!(
            kind,
            Some(IntrinsicKind::CheckedDiv {
                bits: 32,
                signed: true
            })
        );
    }

    #[test]
    fn test_parse_checked_rem_u64() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_u64_checked_rem");
        assert_eq!(
            kind,
            Some(IntrinsicKind::CheckedRem {
                bits: 64,
                signed: false
            })
        );
    }

    #[test]
    fn test_unknown_function() {
        let kind = IntrinsicKind::from_smt_name("unknown_function");
        assert_eq!(kind, None);

        let kind = IntrinsicKind::from_smt_name("core_num_impl_u8_some_other_op");
        assert_eq!(kind, None);
    }

    #[test]
    fn test_generate_wrapping_add_axiom_u8() {
        let kind = IntrinsicKind::WrappingAdd {
            bits: 8,
            signed: false,
        };
        let axiom = kind.generate_axiom("wrapping_add_u8");
        assert!(axiom.contains("(mod (+ a b) 256)"));
    }

    #[test]
    fn test_generate_wrapping_sub_axiom_u8() {
        let kind = IntrinsicKind::WrappingSub {
            bits: 8,
            signed: false,
        };
        let axiom = kind.generate_axiom("wrapping_sub_u8");
        assert!(axiom.contains("(mod (+ (- a b) 256) 256)"));
    }

    #[test]
    fn test_generate_saturating_add_axiom_unsigned() {
        let kind = IntrinsicKind::SaturatingAdd {
            bits: 8,
            signed: false,
        };
        let axiom = kind.generate_axiom("saturating_add_u8");
        assert!(axiom.contains("(ite (> (+ a b) 255) 255"));
        assert!(axiom.contains("(ite (< (+ a b) 0) 0"));
    }

    #[test]
    fn test_generate_saturating_add_axiom_signed() {
        let kind = IntrinsicKind::SaturatingAdd {
            bits: 8,
            signed: true,
        };
        let axiom = kind.generate_axiom("saturating_add_i8");
        assert!(axiom.contains("127")); // max for i8
        assert!(axiom.contains("-128")); // min for i8
    }

    #[test]
    fn test_generate_saturating_sub_axiom_unsigned() {
        let kind = IntrinsicKind::SaturatingSub {
            bits: 8,
            signed: false,
        };
        let axiom = kind.generate_axiom("saturating_sub_u8");
        // Unsigned saturating_sub clamps to 0
        assert!(axiom.contains("(ite (< (- a b) 0) 0 (- a b))"));
    }

    #[test]
    fn test_generate_saturating_mul_axiom_signed() {
        let kind = IntrinsicKind::SaturatingMul {
            bits: 8,
            signed: true,
        };
        let axiom = kind.generate_axiom("saturating_mul_i8");
        assert!(axiom.contains("(* a b)"));
        assert!(axiom.contains("127")); // max for i8
        assert!(axiom.contains("-128")); // min for i8
    }

    #[test]
    fn test_generate_saturating_div_axiom_unsigned() {
        let kind = IntrinsicKind::SaturatingDiv {
            bits: 8,
            signed: false,
        };
        let axiom = kind.generate_axiom("saturating_div_u8");
        // Unsigned saturating_div is just regular division
        assert!(axiom.contains("(div a b)"));
        // Should not have min/max clamping
        assert!(!axiom.contains("255"));
    }

    #[test]
    fn test_generate_saturating_div_axiom_signed() {
        let kind = IntrinsicKind::SaturatingDiv {
            bits: 8,
            signed: true,
        };
        let axiom = kind.generate_axiom("saturating_div_i8");
        // Signed saturating_div: MIN / -1 = MAX
        assert!(axiom.contains("-128")); // i8 MIN
        assert!(axiom.contains("127")); // i8 MAX (saturates to)
        assert!(axiom.contains("(- 1)")); // divisor -1
    }

    #[test]
    fn test_generate_intrinsic_axioms() {
        let names = vec![
            "core_num_impl_u8_wrapping_add",
            "some_unknown_func",
            "core_num_impl_i32_saturating_sub",
        ];
        let axioms = generate_intrinsic_axioms(&names);
        assert_eq!(axioms.len(), 2);
    }

    #[test]
    fn test_is_known_intrinsic() {
        assert!(is_known_intrinsic("core_num_impl_u8_wrapping_add"));
        assert!(is_known_intrinsic("core_num_impl_u64_wrapping_mul"));
        assert!(!is_known_intrinsic("unknown_function"));
        assert!(!is_known_intrinsic("core_num_impl_u8_unknown_op"));
    }

    #[test]
    fn test_inline_wrapping_add_u8() {
        let result = try_inline_call(
            "core_num_impl_u8_wrapping_add",
            &["a".to_string(), "b".to_string()],
        );
        assert_eq!(result, Some("(mod (+ a b) 256)".to_string()));
    }

    #[test]
    fn test_inline_wrapping_sub_u8() {
        let result = try_inline_call(
            "core_num_impl_u8_wrapping_sub",
            &["a".to_string(), "b".to_string()],
        );
        assert_eq!(result, Some("(mod (+ (- a b) 256) 256)".to_string()));
    }

    #[test]
    fn test_inline_wrapping_mul_u16() {
        let result = try_inline_call(
            "core_num_impl_u16_wrapping_mul",
            &["x".to_string(), "y".to_string()],
        );
        assert_eq!(result, Some("(mod (* x y) 65536)".to_string()));
    }

    #[test]
    fn test_inline_saturating_add_u8() {
        let result = try_inline_call(
            "core_num_impl_u8_saturating_add",
            &["x".to_string(), "y".to_string()],
        );
        assert!(result.is_some());
        let expr = result.unwrap();
        assert!(expr.contains("ite"));
        assert!(expr.contains("255")); // u8 max
    }

    #[test]
    fn test_inline_saturating_sub_u8() {
        let result = try_inline_call(
            "core_num_impl_u8_saturating_sub",
            &["a".to_string(), "b".to_string()],
        );
        assert_eq!(result, Some("(ite (< (- a b) 0) 0 (- a b))".to_string()));
    }

    #[test]
    fn test_inline_saturating_mul_u8() {
        let result = try_inline_call(
            "core_num_impl_u8_saturating_mul",
            &["a".to_string(), "b".to_string()],
        );
        assert!(result.is_some());
        let expr = result.unwrap();
        assert!(expr.contains("(* a b)"));
        assert!(expr.contains("255")); // u8 max
    }

    #[test]
    fn test_inline_saturating_div_u32() {
        // Unsigned saturating_div is just regular division
        let result = try_inline_call(
            "core_num_impl_u32_saturating_div",
            &["a".to_string(), "b".to_string()],
        );
        assert_eq!(result, Some("(div a b)".to_string()));
    }

    #[test]
    fn test_inline_saturating_div_i8() {
        // Signed saturating_div: MIN / -1 = MAX
        let result = try_inline_call(
            "core_num_impl_i8_saturating_div",
            &["a".to_string(), "b".to_string()],
        );
        assert!(result.is_some());
        let expr = result.unwrap();
        assert!(expr.contains("-128")); // i8 MIN
        assert!(expr.contains("127")); // i8 MAX (saturates to this)
        assert!(expr.contains("div a b")); // regular division for non-special case
    }

    #[test]
    fn test_inline_unknown_function() {
        let result = try_inline_call("unknown_function", &["a".to_string(), "b".to_string()]);
        assert_eq!(result, None);
    }

    #[test]
    fn test_inline_wrong_arity() {
        // wrapping_add expects 2 args, but we give 3
        let result = try_inline_call(
            "core_num_impl_u8_wrapping_add",
            &["a".to_string(), "b".to_string(), "c".to_string()],
        );
        assert_eq!(result, None);
    }

    #[test]
    fn test_inline_expression_direct() {
        let kind = IntrinsicKind::WrappingAdd {
            bits: 8,
            signed: false,
        };
        let expr = kind.inline_expression(&["_1", "1"]);
        assert_eq!(expr, Some("(mod (+ _1 1) 256)".to_string()));
    }

    // New intrinsic tests
    #[test]
    fn test_parse_abs_i32() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_i32_abs");
        assert_eq!(kind, Some(IntrinsicKind::Abs { bits: 32 }));
    }

    #[test]
    fn test_parse_signum_i64() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_i64_signum");
        assert_eq!(kind, Some(IntrinsicKind::Signum));
    }

    #[test]
    fn test_parse_min_i32() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_i32_min");
        assert_eq!(kind, Some(IntrinsicKind::Min { signed: true }));
    }

    #[test]
    fn test_parse_max_u64() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_u64_max");
        assert_eq!(kind, Some(IntrinsicKind::Max { signed: false }));
    }

    #[test]
    fn test_parse_rotate_left_u32() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_u32_rotate_left");
        assert_eq!(kind, Some(IntrinsicKind::RotateLeft { bits: 32 }));
    }

    #[test]
    fn test_parse_count_ones_u8() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_u8_count_ones");
        assert_eq!(kind, Some(IntrinsicKind::CountOnes { bits: 8 }));
    }

    #[test]
    fn test_inline_abs_i32() {
        let result = try_inline_call("core_num_impl_i32_abs", &["x".to_string()]);
        assert!(result.is_some());
        let expr = result.unwrap();
        assert!(expr.contains("ite"));
        assert!(expr.contains("< x 0"));
    }

    #[test]
    fn test_inline_signum() {
        let result = try_inline_call("core_num_impl_i32_signum", &["x".to_string()]);
        assert!(result.is_some());
        let expr = result.unwrap();
        // signum returns -1, 0, or 1
        assert!(expr.contains("(- 1)")); // -1
        assert!(expr.contains('1'));
        assert!(expr.contains('0'));
    }

    #[test]
    fn test_inline_min() {
        let result = try_inline_call("core_num_impl_i32_min", &["a".to_string(), "b".to_string()]);
        assert_eq!(result, Some("(ite (<= a b) a b)".to_string()));
    }

    #[test]
    fn test_inline_max() {
        let result = try_inline_call("core_num_impl_u32_max", &["a".to_string(), "b".to_string()]);
        assert_eq!(result, Some("(ite (>= a b) a b)".to_string()));
    }

    #[test]
    fn test_inline_unary_wrong_arity() {
        // abs expects 1 arg, but we give 2
        let result = try_inline_call("core_num_impl_i32_abs", &["a".to_string(), "b".to_string()]);
        assert_eq!(result, None);
    }

    #[test]
    fn test_generate_abs_axiom() {
        let kind = IntrinsicKind::Abs { bits: 8 };
        let axiom = kind.generate_axiom("abs_i8");
        assert!(axiom.contains("forall ((a Int))"));
        assert!(axiom.contains("-128")); // i8 MIN
    }

    #[test]
    fn test_generate_signum_axiom() {
        let kind = IntrinsicKind::Signum;
        let axiom = kind.generate_axiom("signum");
        assert!(axiom.contains("(- 1)")); // returns -1 for negative
        assert!(axiom.contains('1'));
        assert!(axiom.contains('0'));
    }

    #[test]
    fn test_generate_min_axiom() {
        let kind = IntrinsicKind::Min { signed: true };
        let axiom = kind.generate_axiom("min");
        assert!(axiom.contains("(ite (<= a b) a b)"));
    }

    #[test]
    fn test_intrinsic_arity() {
        assert_eq!(IntrinsicKind::Abs { bits: 32 }.arity(), 1);
        assert_eq!(IntrinsicKind::Signum.arity(), 1);
        assert_eq!(IntrinsicKind::Min { signed: true }.arity(), 2);
        assert_eq!(
            IntrinsicKind::WrappingAdd {
                bits: 8,
                signed: false
            }
            .arity(),
            2
        );
        assert_eq!(IntrinsicKind::CountOnes { bits: 32 }.arity(), 1);
    }

    #[test]
    fn test_is_known_intrinsic_new() {
        assert!(is_known_intrinsic("core_num_impl_i32_abs"));
        assert!(is_known_intrinsic("core_num_impl_i64_signum"));
        assert!(is_known_intrinsic("core_num_impl_u32_min"));
        assert!(is_known_intrinsic("core_num_impl_i8_max"));
        assert!(is_known_intrinsic("core_num_impl_u64_rotate_left"));
        assert!(is_known_intrinsic("core_num_impl_u8_count_ones"));
    }

    // Checked intrinsic tests
    #[test]
    fn test_parse_checked_add_u8() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_u8_checked_add");
        assert_eq!(
            kind,
            Some(IntrinsicKind::CheckedAdd {
                bits: 8,
                signed: false
            })
        );
    }

    #[test]
    fn test_parse_checked_sub_i32() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_i32_checked_sub");
        assert_eq!(
            kind,
            Some(IntrinsicKind::CheckedSub {
                bits: 32,
                signed: true
            })
        );
    }

    #[test]
    fn test_parse_checked_mul_u64() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_u64_checked_mul");
        assert_eq!(
            kind,
            Some(IntrinsicKind::CheckedMul {
                bits: 64,
                signed: false
            })
        );
    }

    #[test]
    fn test_checked_arity() {
        assert_eq!(
            IntrinsicKind::CheckedAdd {
                bits: 8,
                signed: false
            }
            .arity(),
            2
        );
        assert_eq!(
            IntrinsicKind::CheckedSub {
                bits: 32,
                signed: true
            }
            .arity(),
            2
        );
        assert_eq!(
            IntrinsicKind::CheckedMul {
                bits: 64,
                signed: false
            }
            .arity(),
            2
        );
        assert_eq!(
            IntrinsicKind::CheckedDiv {
                bits: 16,
                signed: true
            }
            .arity(),
            2
        );
        assert_eq!(
            IntrinsicKind::CheckedRem {
                bits: 32,
                signed: false
            }
            .arity(),
            2
        );
    }

    #[test]
    fn test_checked_returns_compound_type() {
        assert!(IntrinsicKind::CheckedAdd {
            bits: 8,
            signed: false
        }
        .returns_compound_type());
        assert!(IntrinsicKind::CheckedSub {
            bits: 32,
            signed: true
        }
        .returns_compound_type());
        assert!(IntrinsicKind::CheckedMul {
            bits: 64,
            signed: false
        }
        .returns_compound_type());
        assert!(IntrinsicKind::CheckedDiv {
            bits: 16,
            signed: true
        }
        .returns_compound_type());
        assert!(IntrinsicKind::CheckedRem {
            bits: 32,
            signed: false
        }
        .returns_compound_type());

        // Non-checked should return false
        assert!(!IntrinsicKind::WrappingAdd {
            bits: 8,
            signed: false
        }
        .returns_compound_type());
        assert!(!IntrinsicKind::SaturatingAdd {
            bits: 8,
            signed: false
        }
        .returns_compound_type());
    }

    #[test]
    fn test_checked_inline_returns_none() {
        // Checked operations cannot be inlined as simple expressions
        let result = try_inline_call(
            "core_num_impl_u8_checked_add",
            &["a".to_string(), "b".to_string()],
        );
        assert_eq!(result, None);

        let result = try_inline_call(
            "core_num_impl_i32_checked_sub",
            &["x".to_string(), "y".to_string()],
        );
        assert_eq!(result, None);

        let result = try_inline_call(
            "core_num_impl_i16_checked_div",
            &["x".to_string(), "y".to_string()],
        );
        assert_eq!(result, None);

        let result = try_inline_call(
            "core_num_impl_u64_checked_rem",
            &["x".to_string(), "y".to_string()],
        );
        assert_eq!(result, None);
    }

    #[test]
    fn test_checked_add_components_u8() {
        let kind = IntrinsicKind::CheckedAdd {
            bits: 8,
            signed: false,
        };
        let (overflow, value) = kind.checked_components(&["a", "b"]).unwrap();

        // Value should be (+ a b)
        assert_eq!(value, "(+ a b)");

        // Overflow for u8: result > 255 or result < 0
        assert!(overflow.contains("255")); // u8 max
        assert!(overflow.contains("< (+ a b) 0")); // underflow check
    }

    #[test]
    fn test_checked_add_components_i8() {
        let kind = IntrinsicKind::CheckedAdd {
            bits: 8,
            signed: true,
        };
        let (overflow, value) = kind.checked_components(&["a", "b"]).unwrap();

        // Value should be (+ a b)
        assert_eq!(value, "(+ a b)");

        // Overflow for i8: result > 127 or result < -128
        assert!(overflow.contains("127")); // i8 max
        assert!(overflow.contains("-128")); // i8 min
    }

    #[test]
    fn test_checked_sub_components_u8() {
        let kind = IntrinsicKind::CheckedSub {
            bits: 8,
            signed: false,
        };
        let (overflow, value) = kind.checked_components(&["a", "b"]).unwrap();

        // Value should be (- a b)
        assert_eq!(value, "(- a b)");

        // Unsigned underflow: result < 0
        assert!(overflow.contains("< (- a b) 0"));
    }

    #[test]
    fn test_checked_mul_components_u8() {
        let kind = IntrinsicKind::CheckedMul {
            bits: 8,
            signed: false,
        };
        let (overflow, value) = kind.checked_components(&["a", "b"]).unwrap();

        // Value should be (* a b)
        assert_eq!(value, "(* a b)");

        // Overflow for u8: result > 255 or result < 0
        assert!(overflow.contains("255")); // u8 max
    }

    #[test]
    fn test_checked_div_components_i32() {
        let kind = IntrinsicKind::CheckedDiv {
            bits: 32,
            signed: true,
        };
        let (overflow, value) = kind.checked_components(&["a", "b"]).unwrap();

        assert!(overflow.contains("(= b 0)"));
        assert!(overflow.contains("-2147483648")); // i32 min
        assert!(overflow.contains("(- 1)")); // divisor -1 overflow check
        assert!(value.starts_with("(ite (= b 0) 0"));
        assert!(value.contains("(div a b)"));
    }

    #[test]
    fn test_checked_rem_components_u8() {
        let kind = IntrinsicKind::CheckedRem {
            bits: 8,
            signed: false,
        };
        let (overflow, value) = kind.checked_components(&["a", "b"]).unwrap();

        assert_eq!(overflow, "(= b 0)");
        assert!(value.starts_with("(ite (= b 0) 0"));
        assert!(value.contains("(- a (* (ite (>= a 0) (div a b) (- (div (- a) b))) b))"));
    }

    #[test]
    fn test_try_expand_checked_call() {
        let result = try_expand_checked_call(
            "core_num_impl_u8_checked_add",
            &["a".to_string(), "b".to_string()],
        );
        assert!(result.is_some());
        let (overflow, value) = result.unwrap();
        assert_eq!(value, "(+ a b)");
        assert!(overflow.contains("255"));

        let div_result = try_expand_checked_call(
            "core_num_impl_i32_checked_div",
            &["a".to_string(), "b".to_string()],
        );
        assert!(div_result.is_some());
        let (div_overflow, div_value) = div_result.unwrap();
        assert!(div_overflow.contains("(= b 0)"));
        assert!(div_value.contains("(div a b)"));
    }

    #[test]
    fn test_try_expand_non_checked_returns_none() {
        // Non-checked operations should return None
        let result = try_expand_checked_call(
            "core_num_impl_u8_wrapping_add",
            &["a".to_string(), "b".to_string()],
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_is_known_intrinsic_checked() {
        assert!(is_known_intrinsic("core_num_impl_u8_checked_add"));
        assert!(is_known_intrinsic("core_num_impl_i32_checked_sub"));
        assert!(is_known_intrinsic("core_num_impl_u64_checked_mul"));
        assert!(is_known_intrinsic("core_num_impl_i16_checked_div"));
        assert!(is_known_intrinsic("core_num_impl_u32_checked_rem"));
    }

    #[test]
    fn test_checked_axiom_is_empty() {
        // Checked operations don't have useful axioms
        let kind = IntrinsicKind::CheckedAdd {
            bits: 8,
            signed: false,
        };
        let axiom = kind.generate_axiom("checked_add_u8");
        assert!(axiom.is_empty());

        let div_axiom = IntrinsicKind::CheckedDiv {
            bits: 32,
            signed: true,
        }
        .generate_axiom("checked_div_i32");
        assert!(div_axiom.is_empty());
    }

    // Overflowing intrinsic tests
    #[test]
    fn test_parse_overflowing_add_u8() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_u8_overflowing_add");
        assert_eq!(
            kind,
            Some(IntrinsicKind::OverflowingAdd {
                bits: 8,
                signed: false
            })
        );
    }

    #[test]
    fn test_parse_overflowing_sub_i32() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_i32_overflowing_sub");
        assert_eq!(
            kind,
            Some(IntrinsicKind::OverflowingSub {
                bits: 32,
                signed: true
            })
        );
    }

    #[test]
    fn test_parse_overflowing_mul_u64() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_u64_overflowing_mul");
        assert_eq!(
            kind,
            Some(IntrinsicKind::OverflowingMul {
                bits: 64,
                signed: false
            })
        );
    }

    #[test]
    fn test_overflowing_arity() {
        assert_eq!(
            IntrinsicKind::OverflowingAdd {
                bits: 8,
                signed: false
            }
            .arity(),
            2
        );
        assert_eq!(
            IntrinsicKind::OverflowingSub {
                bits: 32,
                signed: true
            }
            .arity(),
            2
        );
        assert_eq!(
            IntrinsicKind::OverflowingMul {
                bits: 64,
                signed: false
            }
            .arity(),
            2
        );
    }

    #[test]
    fn test_overflowing_returns_compound_type() {
        assert!(IntrinsicKind::OverflowingAdd {
            bits: 8,
            signed: false
        }
        .returns_compound_type());
        assert!(IntrinsicKind::OverflowingSub {
            bits: 32,
            signed: true
        }
        .returns_compound_type());
        assert!(IntrinsicKind::OverflowingMul {
            bits: 64,
            signed: false
        }
        .returns_compound_type());
    }

    #[test]
    fn test_overflowing_inline_returns_none() {
        // Overflowing operations cannot be inlined as simple expressions
        let result = try_inline_call(
            "core_num_impl_u8_overflowing_add",
            &["a".to_string(), "b".to_string()],
        );
        assert_eq!(result, None);

        let result = try_inline_call(
            "core_num_impl_i32_overflowing_sub",
            &["x".to_string(), "y".to_string()],
        );
        assert_eq!(result, None);
    }

    #[test]
    fn test_overflowing_add_components_u8() {
        let kind = IntrinsicKind::OverflowingAdd {
            bits: 8,
            signed: false,
        };
        let (wrapped, overflow) = kind.overflowing_components(&["a", "b"]).unwrap();

        // Wrapped value should use mod 256 for u8
        assert!(wrapped.contains("mod"));
        assert!(wrapped.contains("256"));

        // Overflow for u8: result > 255
        assert!(overflow.contains("255")); // u8 max
    }

    #[test]
    fn test_overflowing_add_components_i8() {
        let kind = IntrinsicKind::OverflowingAdd {
            bits: 8,
            signed: true,
        };
        let (wrapped, overflow) = kind.overflowing_components(&["a", "b"]).unwrap();

        // Wrapped value should use mod 256 for i8
        assert!(wrapped.contains("mod"));
        assert!(wrapped.contains("256"));

        // Overflow for i8: result > 127 or result < -128
        assert!(overflow.contains("127")); // i8 max
        assert!(overflow.contains("-128")); // i8 min
    }

    #[test]
    fn test_overflowing_sub_components_u8() {
        let kind = IntrinsicKind::OverflowingSub {
            bits: 8,
            signed: false,
        };
        let (wrapped, overflow) = kind.overflowing_components(&["a", "b"]).unwrap();

        // Wrapped value should use modular arithmetic
        assert!(wrapped.contains("mod"));
        assert!(wrapped.contains("256"));

        // Unsigned underflow: result < 0
        assert!(overflow.contains("< (- a b) 0"));
    }

    #[test]
    fn test_overflowing_mul_components_u8() {
        let kind = IntrinsicKind::OverflowingMul {
            bits: 8,
            signed: false,
        };
        let (wrapped, overflow) = kind.overflowing_components(&["a", "b"]).unwrap();

        // Wrapped value should use mod 256
        assert!(wrapped.contains("mod"));
        assert!(wrapped.contains("256"));

        // Overflow for u8: result > 255
        assert!(overflow.contains("255")); // u8 max
    }

    #[test]
    fn test_try_expand_overflowing_call() {
        let result = try_expand_overflowing_call(
            "core_num_impl_u8_overflowing_add",
            &["a".to_string(), "b".to_string()],
        );
        assert!(result.is_some());
        let (wrapped, overflow) = result.unwrap();
        assert!(wrapped.contains("mod"));
        assert!(overflow.contains("255"));
    }

    #[test]
    fn test_try_expand_non_overflowing_returns_none() {
        // Non-overflowing operations should return None from try_expand_overflowing_call
        let result = try_expand_overflowing_call(
            "core_num_impl_u8_wrapping_add",
            &["a".to_string(), "b".to_string()],
        );
        assert!(result.is_none());

        // Checked operations should also return None (different method)
        let result = try_expand_overflowing_call(
            "core_num_impl_u8_checked_add",
            &["a".to_string(), "b".to_string()],
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_is_known_intrinsic_overflowing() {
        assert!(is_known_intrinsic("core_num_impl_u8_overflowing_add"));
        assert!(is_known_intrinsic("core_num_impl_i32_overflowing_sub"));
        assert!(is_known_intrinsic("core_num_impl_u64_overflowing_mul"));
    }

    #[test]
    fn test_overflowing_axiom_is_empty() {
        // Overflowing operations don't have useful axioms
        let kind = IntrinsicKind::OverflowingAdd {
            bits: 8,
            signed: false,
        };
        let axiom = kind.generate_axiom("overflowing_add_u8");
        assert!(axiom.is_empty());

        let kind = IntrinsicKind::OverflowingSub {
            bits: 32,
            signed: true,
        };
        let axiom = kind.generate_axiom("overflowing_sub_i32");
        assert!(axiom.is_empty());
    }

    // =========================================================================
    // Tests for new intrinsics: wrapping_div, wrapping_rem, wrapping_neg
    // =========================================================================

    #[test]
    fn test_parse_wrapping_div_u32() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_u32_wrapping_div");
        assert_eq!(
            kind,
            Some(IntrinsicKind::WrappingDiv {
                bits: 32,
                signed: false
            })
        );
    }

    #[test]
    fn test_parse_wrapping_div_i8() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_i8_wrapping_div");
        assert_eq!(
            kind,
            Some(IntrinsicKind::WrappingDiv {
                bits: 8,
                signed: true
            })
        );
    }

    #[test]
    fn test_parse_wrapping_rem_u64() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_u64_wrapping_rem");
        assert_eq!(
            kind,
            Some(IntrinsicKind::WrappingRem {
                bits: 64,
                signed: false
            })
        );
    }

    #[test]
    fn test_parse_wrapping_neg_u8() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_u8_wrapping_neg");
        assert_eq!(
            kind,
            Some(IntrinsicKind::WrappingNeg {
                bits: 8,
                signed: false
            })
        );
    }

    #[test]
    fn test_parse_wrapping_neg_i32() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_i32_wrapping_neg");
        assert_eq!(
            kind,
            Some(IntrinsicKind::WrappingNeg {
                bits: 32,
                signed: true
            })
        );
    }

    #[test]
    fn test_inline_wrapping_div_unsigned() {
        let result = try_inline_call(
            "core_num_impl_u32_wrapping_div",
            &["a".to_string(), "b".to_string()],
        );
        assert!(result.is_some());
        let expr = result.unwrap();
        assert!(expr.contains("div"));
    }

    #[test]
    fn test_inline_wrapping_div_signed() {
        let result = try_inline_call(
            "core_num_impl_i8_wrapping_div",
            &["a".to_string(), "b".to_string()],
        );
        assert!(result.is_some());
        let expr = result.unwrap();
        // Should handle MIN / -1 case
        assert!(expr.contains("-128")); // i8::MIN
    }

    #[test]
    fn test_inline_wrapping_neg_u8() {
        let result = try_inline_call("core_num_impl_u8_wrapping_neg", &["x".to_string()]);
        assert!(result.is_some());
        let expr = result.unwrap();
        assert!(expr.contains("mod"));
        assert!(expr.contains("256"));
    }

    // =========================================================================
    // Tests for shift intrinsics
    // =========================================================================

    #[test]
    fn test_parse_wrapping_shl_u32() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_u32_wrapping_shl");
        assert_eq!(kind, Some(IntrinsicKind::WrappingShl { bits: 32 }));
    }

    #[test]
    fn test_parse_wrapping_shr_i16() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_i16_wrapping_shr");
        assert_eq!(
            kind,
            Some(IntrinsicKind::WrappingShr {
                bits: 16,
                signed: true
            })
        );
    }

    #[test]
    fn test_parse_checked_shl_u64() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_u64_checked_shl");
        assert_eq!(kind, Some(IntrinsicKind::CheckedShl { bits: 64 }));
    }

    #[test]
    fn test_parse_checked_shr_i32() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_i32_checked_shr");
        assert_eq!(
            kind,
            Some(IntrinsicKind::CheckedShr {
                bits: 32,
                signed: true
            })
        );
    }

    #[test]
    fn test_parse_overflowing_shl_u8() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_u8_overflowing_shl");
        assert_eq!(kind, Some(IntrinsicKind::OverflowingShl { bits: 8 }));
    }

    #[test]
    fn test_parse_overflowing_shr_u32() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_u32_overflowing_shr");
        assert_eq!(
            kind,
            Some(IntrinsicKind::OverflowingShr {
                bits: 32,
                signed: false
            })
        );
    }

    #[test]
    fn test_inline_wrapping_shl_u8() {
        let result = try_inline_call(
            "core_num_impl_u8_wrapping_shl",
            &["x".to_string(), "n".to_string()],
        );
        assert!(result.is_some());
        let expr = result.unwrap();
        assert!(expr.contains("mod"));
        assert!(expr.contains("256")); // u8 modulus
    }

    #[test]
    fn test_inline_wrapping_shr_u32() {
        let result = try_inline_call(
            "core_num_impl_u32_wrapping_shr",
            &["x".to_string(), "n".to_string()],
        );
        assert!(result.is_some());
        let expr = result.unwrap();
        assert!(expr.contains("div"));
        assert!(expr.contains("mod")); // n mod 32
    }

    // =========================================================================
    // Tests for checked_neg and overflowing_neg
    // =========================================================================

    #[test]
    fn test_parse_checked_neg_i8() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_i8_checked_neg");
        assert_eq!(
            kind,
            Some(IntrinsicKind::CheckedNeg {
                bits: 8,
                signed: true
            })
        );
    }

    #[test]
    fn test_parse_checked_neg_u32() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_u32_checked_neg");
        assert_eq!(
            kind,
            Some(IntrinsicKind::CheckedNeg {
                bits: 32,
                signed: false
            })
        );
    }

    #[test]
    fn test_parse_overflowing_neg_i32() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_i32_overflowing_neg");
        assert_eq!(
            kind,
            Some(IntrinsicKind::OverflowingNeg {
                bits: 32,
                signed: true
            })
        );
    }

    #[test]
    fn test_checked_neg_components_signed() {
        let kind = IntrinsicKind::CheckedNeg {
            bits: 8,
            signed: true,
        };
        let result = kind.checked_components(&["x"]);
        assert!(result.is_some());
        let (overflow, value) = result.unwrap();
        // Overflow when x == MIN
        assert!(overflow.contains("-128")); // i8::MIN
                                            // Value is -x
        assert!(value.contains("(- 0 x)"));
    }

    #[test]
    fn test_checked_neg_components_unsigned() {
        let kind = IntrinsicKind::CheckedNeg {
            bits: 8,
            signed: false,
        };
        let result = kind.checked_components(&["x"]);
        assert!(result.is_some());
        let (overflow, _value) = result.unwrap();
        // Overflow when x != 0
        assert!(overflow.contains("not"));
        assert!(overflow.contains('0'));
    }

    #[test]
    fn test_overflowing_neg_components_signed() {
        let kind = IntrinsicKind::OverflowingNeg {
            bits: 8,
            signed: true,
        };
        let result = kind.overflowing_components(&["x"]);
        assert!(result.is_some());
        let (wrapped, overflow) = result.unwrap();
        // Wrapped value with mod
        assert!(wrapped.contains("mod"));
        assert!(wrapped.contains("256"));
        // Overflow when x == MIN
        assert!(overflow.contains("-128"));
    }

    #[test]
    fn test_overflowing_neg_components_unsigned() {
        let kind = IntrinsicKind::OverflowingNeg {
            bits: 8,
            signed: false,
        };
        let result = kind.overflowing_components(&["x"]);
        assert!(result.is_some());
        let (wrapped, overflow) = result.unwrap();
        assert!(wrapped.contains("mod"));
        // Overflow when x != 0
        assert!(overflow.contains("not"));
    }

    #[test]
    fn test_checked_shl_components() {
        let kind = IntrinsicKind::CheckedShl { bits: 8 };
        let result = kind.checked_components(&["x", "n"]);
        assert!(result.is_some());
        let (overflow, value) = result.unwrap();
        // Overflow when n >= 8
        assert!(overflow.contains(">= n 8"));
        // Value is x << n
        assert!(value.contains('^'));
    }

    #[test]
    fn test_checked_shr_components() {
        let kind = IntrinsicKind::CheckedShr {
            bits: 32,
            signed: false,
        };
        let result = kind.checked_components(&["x", "n"]);
        assert!(result.is_some());
        let (overflow, value) = result.unwrap();
        // Overflow when n >= 32
        assert!(overflow.contains(">= n 32"));
        assert!(value.contains("div"));
    }

    #[test]
    fn test_overflowing_shl_components() {
        let kind = IntrinsicKind::OverflowingShl { bits: 8 };
        let result = kind.overflowing_components(&["x", "n"]);
        assert!(result.is_some());
        let (wrapped, overflow) = result.unwrap();
        // Wrapped uses mod bits for shift amount
        assert!(wrapped.contains("mod n 8"));
        assert!(wrapped.contains("256")); // result mod 2^8
                                          // Overflow when n >= 8
        assert!(overflow.contains(">= n 8"));
    }

    #[test]
    fn test_overflowing_shr_components() {
        let kind = IntrinsicKind::OverflowingShr {
            bits: 32,
            signed: false,
        };
        let result = kind.overflowing_components(&["x", "n"]);
        assert!(result.is_some());
        let (wrapped, overflow) = result.unwrap();
        assert!(wrapped.contains("div"));
        assert!(wrapped.contains("mod n 32"));
        // Overflow when n >= 32
        assert!(overflow.contains(">= n 32"));
    }

    #[test]
    fn test_arity_new_intrinsics() {
        // Unary
        assert_eq!(
            IntrinsicKind::WrappingNeg {
                bits: 8,
                signed: false
            }
            .arity(),
            1
        );
        assert_eq!(
            IntrinsicKind::CheckedNeg {
                bits: 8,
                signed: true
            }
            .arity(),
            1
        );
        assert_eq!(
            IntrinsicKind::OverflowingNeg {
                bits: 8,
                signed: true
            }
            .arity(),
            1
        );

        // Binary
        assert_eq!(
            IntrinsicKind::WrappingDiv {
                bits: 32,
                signed: false
            }
            .arity(),
            2
        );
        assert_eq!(
            IntrinsicKind::WrappingRem {
                bits: 32,
                signed: false
            }
            .arity(),
            2
        );
        assert_eq!(IntrinsicKind::WrappingShl { bits: 8 }.arity(), 2);
        assert_eq!(
            IntrinsicKind::WrappingShr {
                bits: 8,
                signed: true
            }
            .arity(),
            2
        );
        assert_eq!(IntrinsicKind::CheckedShl { bits: 32 }.arity(), 2);
        assert_eq!(
            IntrinsicKind::CheckedShr {
                bits: 32,
                signed: true
            }
            .arity(),
            2
        );
        assert_eq!(IntrinsicKind::OverflowingShl { bits: 8 }.arity(), 2);
        assert_eq!(
            IntrinsicKind::OverflowingShr {
                bits: 8,
                signed: false
            }
            .arity(),
            2
        );
    }

    #[test]
    fn test_returns_compound_type_new_intrinsics() {
        // CheckedNeg, CheckedShl, CheckedShr return Option<T>
        assert!(IntrinsicKind::CheckedNeg {
            bits: 8,
            signed: true
        }
        .returns_compound_type());
        assert!(IntrinsicKind::CheckedShl { bits: 32 }.returns_compound_type());
        assert!(IntrinsicKind::CheckedShr {
            bits: 32,
            signed: false
        }
        .returns_compound_type());

        // OverflowingNeg, OverflowingShl, OverflowingShr return (T, bool)
        assert!(IntrinsicKind::OverflowingNeg {
            bits: 8,
            signed: true
        }
        .returns_compound_type());
        assert!(IntrinsicKind::OverflowingShl { bits: 32 }.returns_compound_type());
        assert!(IntrinsicKind::OverflowingShr {
            bits: 32,
            signed: true
        }
        .returns_compound_type());

        // Wrapping operations do NOT return compound types
        assert!(!IntrinsicKind::WrappingNeg {
            bits: 8,
            signed: false
        }
        .returns_compound_type());
        assert!(!IntrinsicKind::WrappingDiv {
            bits: 32,
            signed: false
        }
        .returns_compound_type());
        assert!(!IntrinsicKind::WrappingRem {
            bits: 32,
            signed: true
        }
        .returns_compound_type());
        assert!(!IntrinsicKind::WrappingShl { bits: 8 }.returns_compound_type());
        assert!(!IntrinsicKind::WrappingShr {
            bits: 8,
            signed: false
        }
        .returns_compound_type());
    }

    #[test]
    fn test_is_known_intrinsic_div_rem_neg_shift() {
        assert!(is_known_intrinsic("core_num_impl_u8_wrapping_div"));
        assert!(is_known_intrinsic("core_num_impl_i32_wrapping_rem"));
        assert!(is_known_intrinsic("core_num_impl_u16_wrapping_neg"));
        assert!(is_known_intrinsic("core_num_impl_i8_checked_neg"));
        assert!(is_known_intrinsic("core_num_impl_u32_overflowing_neg"));
        assert!(is_known_intrinsic("core_num_impl_u64_wrapping_shl"));
        assert!(is_known_intrinsic("core_num_impl_i16_wrapping_shr"));
        assert!(is_known_intrinsic("core_num_impl_u8_checked_shl"));
        assert!(is_known_intrinsic("core_num_impl_i32_checked_shr"));
        assert!(is_known_intrinsic("core_num_impl_u64_overflowing_shl"));
        assert!(is_known_intrinsic("core_num_impl_i8_overflowing_shr"));
    }

    // ============================================================
    // Tests for bit rotation intrinsics (catches mutation testing gaps)
    // ============================================================

    #[test]
    fn test_parse_rotate_right_u32() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_u32_rotate_right");
        assert_eq!(kind, Some(IntrinsicKind::RotateRight { bits: 32 }));
    }

    #[test]
    fn test_parse_leading_zeros_u32() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_u32_leading_zeros");
        assert_eq!(kind, Some(IntrinsicKind::LeadingZeros { bits: 32 }));
    }

    #[test]
    fn test_parse_trailing_zeros_u64() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_u64_trailing_zeros");
        assert_eq!(kind, Some(IntrinsicKind::TrailingZeros { bits: 64 }));
    }

    #[test]
    fn test_parse_swap_bytes_u16() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_u16_swap_bytes");
        assert_eq!(kind, Some(IntrinsicKind::SwapBytes { bits: 16 }));
    }

    #[test]
    fn test_parse_reverse_bits_u8() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_u8_reverse_bits");
        assert_eq!(kind, Some(IntrinsicKind::ReverseBits { bits: 8 }));
    }

    #[test]
    fn test_inline_rotate_left_u8() {
        let result = try_inline_call(
            "core_num_impl_u8_rotate_left",
            &["x".to_string(), "n".to_string()],
        );
        assert!(result.is_some());
        let expr = result.unwrap();
        // rotate_left: (mod (+ (* x (^ 2 n)) (div x (^ 2 (- 8 n)))) 256)
        assert!(expr.contains("mod"));
        assert!(expr.contains("256")); // 2^8 modulus
        assert!(expr.contains("(^ 2 n)")); // left shift factor
        assert!(expr.contains("(- 8 n)")); // bits - n for right shift
    }

    #[test]
    fn test_inline_rotate_right_u8() {
        let result = try_inline_call(
            "core_num_impl_u8_rotate_right",
            &["x".to_string(), "n".to_string()],
        );
        assert!(result.is_some());
        let expr = result.unwrap();
        // rotate_right: (mod (+ (div x (^ 2 n)) (* x (^ 2 (- 8 n)))) 256)
        assert!(expr.contains("mod"));
        assert!(expr.contains("256")); // 2^8 modulus
        assert!(expr.contains("(div x (^ 2 n))")); // right shift
        assert!(expr.contains("(- 8 n)")); // bits - n for left shift
    }

    #[test]
    fn test_inline_rotate_left_u32() {
        let result = try_inline_call(
            "core_num_impl_u32_rotate_left",
            &["val".to_string(), "amt".to_string()],
        );
        assert!(result.is_some());
        let expr = result.unwrap();
        // Should use 2^32 = 4294967296 as modulus
        assert!(expr.contains("4294967296"));
        assert!(expr.contains("(- 32 amt)")); // bits - n
    }

    #[test]
    fn test_inline_rotate_right_u64() {
        let result = try_inline_call(
            "core_num_impl_u64_rotate_right",
            &["val".to_string(), "amt".to_string()],
        );
        assert!(result.is_some());
        let expr = result.unwrap();
        // Should use 2^64 as modulus
        assert!(expr.contains("18446744073709551616"));
        assert!(expr.contains("(- 64 amt)")); // bits - n
    }

    // ============================================================
    // Tests for saturating intrinsics with specific boundary values
    // ============================================================

    #[test]
    fn test_saturating_add_signed_boundary_values() {
        // i8: max = 127, min = -128
        let kind = IntrinsicKind::SaturatingAdd {
            bits: 8,
            signed: true,
        };
        let expr = kind.inline_expression(&["a", "b"]).expect("should inline");
        assert!(expr.contains("127")); // (1 << (8-1)) - 1 = 127
        assert!(expr.contains("-128")); // -(1 << (8-1)) = -128

        // i16: max = 32767, min = -32768
        let kind = IntrinsicKind::SaturatingAdd {
            bits: 16,
            signed: true,
        };
        let expr = kind.inline_expression(&["a", "b"]).expect("should inline");
        assert!(expr.contains("32767")); // (1 << 15) - 1
        assert!(expr.contains("-32768")); // -(1 << 15)

        // i32: max = 2147483647, min = -2147483648
        let kind = IntrinsicKind::SaturatingAdd {
            bits: 32,
            signed: true,
        };
        let expr = kind.inline_expression(&["a", "b"]).expect("should inline");
        assert!(expr.contains("2147483647")); // (1 << 31) - 1
        assert!(expr.contains("-2147483648")); // -(1 << 31)
    }

    #[test]
    fn test_saturating_sub_signed_boundary_values() {
        // i8: max = 127, min = -128
        let kind = IntrinsicKind::SaturatingSub {
            bits: 8,
            signed: true,
        };
        let expr = kind.inline_expression(&["a", "b"]).expect("should inline");
        assert!(expr.contains("127"));
        assert!(expr.contains("-128"));
    }

    #[test]
    fn test_saturating_mul_signed_boundary_values() {
        // i8: max = 127, min = -128
        let kind = IntrinsicKind::SaturatingMul {
            bits: 8,
            signed: true,
        };
        let expr = kind.inline_expression(&["a", "b"]).expect("should inline");
        assert!(expr.contains("127"));
        assert!(expr.contains("-128"));
    }

    // ============================================================
    // Tests for checked components with specific boundary values
    // ============================================================

    #[test]
    fn test_checked_add_components_i8_boundaries() {
        let kind = IntrinsicKind::CheckedAdd {
            bits: 8,
            signed: true,
        };
        let (overflow, _value) = kind.checked_components(&["a", "b"]).unwrap();
        // Overflow for i8: (+ a b) > 127 or (+ a b) < -128
        assert!(overflow.contains("127"));
        assert!(overflow.contains("-128"));
    }

    #[test]
    fn test_checked_add_components_i16_boundaries() {
        let kind = IntrinsicKind::CheckedAdd {
            bits: 16,
            signed: true,
        };
        let (overflow, _value) = kind.checked_components(&["a", "b"]).unwrap();
        assert!(overflow.contains("32767"));
        assert!(overflow.contains("-32768"));
    }

    #[test]
    fn test_checked_add_components_i32_boundaries() {
        let kind = IntrinsicKind::CheckedAdd {
            bits: 32,
            signed: true,
        };
        let (overflow, _value) = kind.checked_components(&["a", "b"]).unwrap();
        assert!(overflow.contains("2147483647"));
        assert!(overflow.contains("-2147483648"));
    }

    #[test]
    fn test_checked_sub_components_i8_boundaries() {
        let kind = IntrinsicKind::CheckedSub {
            bits: 8,
            signed: true,
        };
        let (overflow, _value) = kind.checked_components(&["a", "b"]).unwrap();
        assert!(overflow.contains("127"));
        assert!(overflow.contains("-128"));
    }

    #[test]
    fn test_checked_mul_components_i8_boundaries() {
        let kind = IntrinsicKind::CheckedMul {
            bits: 8,
            signed: true,
        };
        let (overflow, _value) = kind.checked_components(&["a", "b"]).unwrap();
        assert!(overflow.contains("127"));
        assert!(overflow.contains("-128"));
    }

    // ============================================================
    // Tests for overflowing components with specific boundary values
    // ============================================================

    #[test]
    fn test_overflowing_add_components_i8_boundaries() {
        let kind = IntrinsicKind::OverflowingAdd {
            bits: 8,
            signed: true,
        };
        let (_wrapped, overflow) = kind.overflowing_components(&["a", "b"]).unwrap();
        assert!(overflow.contains("127"));
        assert!(overflow.contains("-128"));
    }

    #[test]
    fn test_overflowing_add_components_i16_boundaries() {
        let kind = IntrinsicKind::OverflowingAdd {
            bits: 16,
            signed: true,
        };
        let (_wrapped, overflow) = kind.overflowing_components(&["a", "b"]).unwrap();
        assert!(overflow.contains("32767"));
        assert!(overflow.contains("-32768"));
    }

    #[test]
    fn test_overflowing_sub_components_i8_boundaries() {
        let kind = IntrinsicKind::OverflowingSub {
            bits: 8,
            signed: true,
        };
        let (_wrapped, overflow) = kind.overflowing_components(&["a", "b"]).unwrap();
        assert!(overflow.contains("127"));
        assert!(overflow.contains("-128"));
    }

    #[test]
    fn test_overflowing_mul_components_i8_boundaries() {
        let kind = IntrinsicKind::OverflowingMul {
            bits: 8,
            signed: true,
        };
        let (_wrapped, overflow) = kind.overflowing_components(&["a", "b"]).unwrap();
        assert!(overflow.contains("127"));
        assert!(overflow.contains("-128"));
    }

    // ============================================================
    // Tests for axiom generation with specific boundaries
    // ============================================================

    #[test]
    fn test_generate_saturating_add_axiom_i8_boundaries() {
        let kind = IntrinsicKind::SaturatingAdd {
            bits: 8,
            signed: true,
        };
        let axiom = kind.generate_axiom("sat_add_i8");
        assert!(axiom.contains("127"));
        assert!(axiom.contains("-128"));
    }

    #[test]
    fn test_generate_saturating_sub_axiom_i16_boundaries() {
        let kind = IntrinsicKind::SaturatingSub {
            bits: 16,
            signed: true,
        };
        let axiom = kind.generate_axiom("sat_sub_i16");
        assert!(axiom.contains("32767"));
        assert!(axiom.contains("-32768"));
    }

    #[test]
    fn test_generate_rotate_left_axiom_u8() {
        let kind = IntrinsicKind::RotateLeft { bits: 8 };
        let axiom = kind.generate_axiom("rotate_left_u8");
        assert!(axiom.contains("256")); // 2^8 modulus
        assert!(axiom.contains("(- 8")); // bits - n calculation
    }

    #[test]
    fn test_generate_rotate_right_axiom_u16() {
        let kind = IntrinsicKind::RotateRight { bits: 16 };
        let axiom = kind.generate_axiom("rotate_right_u16");
        assert!(axiom.contains("65536")); // 2^16 modulus
        assert!(axiom.contains("(- 16")); // bits - n calculation
    }

    // ============================================================
    // Tests to verify correct shift direction in rotation expressions
    // ============================================================

    #[test]
    fn test_rotate_left_shift_direction() {
        // rotate_left should: (* x (^ 2 n)) for left shift part
        let kind = IntrinsicKind::RotateLeft { bits: 8 };
        let expr = kind.inline_expression(&["x", "n"]).expect("should inline");
        // The expression should have multiplication before division (left shift dominates)
        let mul_pos = expr.find("(* x").expect("should have multiplication");
        let div_pos = expr.find("(div x").expect("should have division");
        // In rotate_left, the multiplication (left shift) comes first in the addition
        assert!(
            mul_pos < div_pos,
            "rotate_left: left shift should come before right shift in expression"
        );
    }

    #[test]
    fn test_rotate_right_shift_direction() {
        // rotate_right should: (div x (^ 2 n)) for right shift part
        let kind = IntrinsicKind::RotateRight { bits: 8 };
        let expr = kind.inline_expression(&["x", "n"]).expect("should inline");
        // The expression should have division before multiplication (right shift dominates)
        let div_pos = expr.find("(div x").expect("should have division");
        let mul_pos = expr.find("(* x").expect("should have multiplication");
        // In rotate_right, the division (right shift) comes first in the addition
        assert!(
            div_pos < mul_pos,
            "rotate_right: right shift should come before left shift in expression"
        );
    }

    // ============================================================
    // Tests for wrapping shift intrinsics expression structure
    // ============================================================

    #[test]
    fn test_wrapping_shl_u8_structure() {
        let result = try_inline_call(
            "core_num_impl_u8_wrapping_shl",
            &["x".to_string(), "n".to_string()],
        );
        assert!(result.is_some());
        let expr = result.unwrap();
        // wrapping_shl: (mod (* x (^ 2 (mod n 8))) 256)
        assert!(expr.contains("mod"));
        assert!(expr.contains("256"));
        assert!(expr.contains("(mod n 8)")); // shift amount mod bits
    }

    #[test]
    fn test_wrapping_shr_u8_structure() {
        let result = try_inline_call(
            "core_num_impl_u8_wrapping_shr",
            &["x".to_string(), "n".to_string()],
        );
        assert!(result.is_some());
        let expr = result.unwrap();
        // wrapping_shr unsigned: (div x (^ 2 (mod n 8)))
        assert!(expr.contains("div"));
        assert!(expr.contains("(mod n 8)")); // shift amount mod bits
    }

    #[test]
    fn test_wrapping_shr_i8_signed_structure() {
        let result = try_inline_call(
            "core_num_impl_i8_wrapping_shr",
            &["x".to_string(), "n".to_string()],
        );
        assert!(result.is_some());
        let expr = result.unwrap();
        // Signed right shift has floor division semantics
        assert!(expr.contains("(mod n 8)")); // shift amount mod bits
    }

    // ============================================================
    // Tests for checked shift intrinsics
    // ============================================================

    #[test]
    fn test_checked_shl_components_u8() {
        let kind = IntrinsicKind::CheckedShl { bits: 8 };
        let (overflow, value) = kind.checked_components(&["x", "n"]).unwrap();
        // Overflow if n >= 8
        assert!(overflow.contains(">= n 8"));
        // Value uses shift expression
        assert!(value.contains("(^ 2 n)"));
    }

    #[test]
    fn test_checked_shr_components_u32() {
        let kind = IntrinsicKind::CheckedShr {
            bits: 32,
            signed: false,
        };
        let (overflow, _value) = kind.checked_components(&["x", "n"]).unwrap();
        // Overflow if n >= 32
        assert!(overflow.contains(">= n 32"));
    }

    // ============================================================
    // Tests for abs intrinsic with MIN boundary
    // ============================================================

    #[test]
    fn test_inline_abs_i8_min_boundary() {
        let result = try_inline_call("core_num_impl_i8_abs", &["x".to_string()]);
        assert!(result.is_some());
        let expr = result.unwrap();
        // abs should handle MIN specially: abs(MIN) = MIN (wraps)
        assert!(expr.contains("-128")); // i8 MIN
    }

    #[test]
    fn test_inline_abs_i16_min_boundary() {
        let result = try_inline_call("core_num_impl_i16_abs", &["x".to_string()]);
        assert!(result.is_some());
        let expr = result.unwrap();
        assert!(expr.contains("-32768")); // i16 MIN
    }

    #[test]
    fn test_inline_abs_i32_min_boundary() {
        let result = try_inline_call("core_num_impl_i32_abs", &["x".to_string()]);
        assert!(result.is_some());
        let expr = result.unwrap();
        assert!(expr.contains("-2147483648")); // i32 MIN
    }

    // ============================================================
    // Tests for signed MIN value calculations in wrapping/checked operations
    // These tests kill mutants for -(1i128 << (bits - 1)) formula
    // ============================================================

    #[test]
    fn test_wrapping_rem_signed_i8_min_value() {
        // WrappingRem signed: MIN % -1 = 0 special case
        // For i8: MIN = -128 = -(1 << 7) = -(1 << (8-1))
        let kind = IntrinsicKind::WrappingRem {
            bits: 8,
            signed: true,
        };
        let expr = kind.inline_expression(&["a", "b"]).expect("Should inline");
        // Must contain -128 (the exact MIN for i8)
        assert!(
            expr.contains("-128"),
            "i8 MIN should be -128, got: {}",
            expr
        );
        // Must NOT contain -64 (wrong if bits+1 used) or -256 (wrong if bits-1 not used)
        assert!(
            !expr.contains("-64") || expr.contains("-128"),
            "Should use bits-1 in shift"
        );
    }

    #[test]
    fn test_wrapping_rem_signed_i16_min_value() {
        let kind = IntrinsicKind::WrappingRem {
            bits: 16,
            signed: true,
        };
        let expr = kind.inline_expression(&["a", "b"]).expect("Should inline");
        // i16 MIN = -32768 = -(1 << 15)
        assert!(
            expr.contains("-32768"),
            "i16 MIN should be -32768, got: {}",
            expr
        );
    }

    #[test]
    fn test_wrapping_rem_signed_i32_min_value() {
        let kind = IntrinsicKind::WrappingRem {
            bits: 32,
            signed: true,
        };
        let expr = kind.inline_expression(&["a", "b"]).expect("Should inline");
        // i32 MIN = -2147483648 = -(1 << 31)
        assert!(
            expr.contains("-2147483648"),
            "i32 MIN should be -2147483648, got: {}",
            expr
        );
    }

    #[test]
    fn test_wrapping_rem_unsigned_no_min_handling() {
        // Unsigned should NOT have the MIN special case
        let kind = IntrinsicKind::WrappingRem {
            bits: 8,
            signed: false,
        };
        let expr = kind.inline_expression(&["a", "b"]).expect("Should inline");
        // Unsigned is just mod
        assert!(expr.contains("(mod a b)"), "Unsigned should be simple mod");
        assert!(
            !expr.contains("-128"),
            "Unsigned should not have signed MIN"
        );
    }

    #[test]
    fn test_checked_rem_signed_i8_overflow_includes_min() {
        // CheckedRem signed: overflow when b=0 OR (a=MIN AND b=-1)
        let kind = IntrinsicKind::CheckedRem {
            bits: 8,
            signed: true,
        };
        let (overflow, _value) = kind.checked_components(&["a", "b"]).unwrap();
        // Must contain -128 for the MIN check
        assert!(
            overflow.contains("-128"),
            "i8 overflow should check MIN=-128, got: {}",
            overflow
        );
        // Must be an OR of two conditions
        assert!(
            overflow.contains("or"),
            "Should have OR for div-by-zero and MIN/-1"
        );
    }

    #[test]
    fn test_checked_rem_signed_i16_overflow_includes_min() {
        let kind = IntrinsicKind::CheckedRem {
            bits: 16,
            signed: true,
        };
        let (overflow, _value) = kind.checked_components(&["a", "b"]).unwrap();
        assert!(
            overflow.contains("-32768"),
            "i16 overflow should check MIN=-32768, got: {}",
            overflow
        );
    }

    #[test]
    fn test_checked_rem_signed_i32_overflow_includes_min() {
        let kind = IntrinsicKind::CheckedRem {
            bits: 32,
            signed: true,
        };
        let (overflow, _value) = kind.checked_components(&["a", "b"]).unwrap();
        assert!(
            overflow.contains("-2147483648"),
            "i32 overflow should check MIN=-2147483648, got: {}",
            overflow
        );
    }

    #[test]
    fn test_checked_rem_unsigned_no_min_check() {
        let kind = IntrinsicKind::CheckedRem {
            bits: 8,
            signed: false,
        };
        let (overflow, _value) = kind.checked_components(&["a", "b"]).unwrap();
        // Unsigned only checks div-by-zero
        assert_eq!(
            overflow, "(= b 0)",
            "Unsigned should only check div-by-zero"
        );
    }

    #[test]
    fn test_wrapping_div_signed_i8_min_value() {
        let kind = IntrinsicKind::WrappingDiv {
            bits: 8,
            signed: true,
        };
        let expr = kind.inline_expression(&["a", "b"]).expect("Should inline");
        // i8 MIN/-1 = MIN (wrapping overflow)
        assert!(
            expr.contains("-128"),
            "i8 MIN should be -128 in wrapping_div, got: {}",
            expr
        );
    }

    #[test]
    fn test_wrapping_div_signed_i32_min_value() {
        let kind = IntrinsicKind::WrappingDiv {
            bits: 32,
            signed: true,
        };
        let expr = kind.inline_expression(&["a", "b"]).expect("Should inline");
        assert!(
            expr.contains("-2147483648"),
            "i32 MIN should be -2147483648, got: {}",
            expr
        );
    }

    #[test]
    fn test_checked_shl_shift_direction() {
        // CheckedShl: x << n (multiply by power of 2)
        let kind = IntrinsicKind::CheckedShl { bits: 8 };
        let (overflow, value) = kind.checked_components(&["x", "n"]).unwrap();

        // Overflow when n >= bits
        assert!(overflow.contains(">= n 8"), "Overflow at n >= 8");

        // Value should be x * 2^n mod 2^8
        // Must use multiplication (left shift), not division (right shift)
        assert!(
            value.contains("(* x"),
            "Left shift should multiply by power of 2, got: {}",
            value
        );
        assert!(
            value.contains("(^ 2 n)"),
            "Should raise 2 to power n, got: {}",
            value
        );
        // Modulus for overflow wrapping
        assert!(
            value.contains("256"),
            "Should mod by 2^8=256, got: {}",
            value
        );
    }

    #[test]
    fn test_wrapping_shl_u8_shift_direction() {
        let kind = IntrinsicKind::WrappingShl { bits: 8 };
        let expr = kind.inline_expression(&["x", "n"]).expect("Should inline");

        // Must multiply, not divide
        assert!(
            expr.contains("(* x"),
            "Left shift should multiply, got: {}",
            expr
        );
        // Wrap shift amount by bits
        assert!(
            expr.contains("(mod n 8)"),
            "Should wrap shift amount mod 8, got: {}",
            expr
        );
        // Wrap result by modulus
        assert!(expr.contains("256"), "Should mod by 2^8=256, got: {}", expr);
    }

    // ============================================================
    // Tests for generate_axiom signed MIN values
    // ============================================================

    #[test]
    fn test_generate_axiom_saturating_mul_signed_i8_min_max() {
        let kind = IntrinsicKind::SaturatingMul {
            bits: 8,
            signed: true,
        };
        let axiom = kind.generate_axiom("saturating_mul_i8");
        // i8: MAX=127, MIN=-128
        assert!(
            axiom.contains("127"),
            "i8 MAX should be 127, got: {}",
            axiom
        );
        assert!(
            axiom.contains("-128"),
            "i8 MIN should be -128, got: {}",
            axiom
        );
    }

    #[test]
    fn test_generate_axiom_saturating_mul_unsigned_u8_max_only() {
        let kind = IntrinsicKind::SaturatingMul {
            bits: 8,
            signed: false,
        };
        let axiom = kind.generate_axiom("saturating_mul_u8");
        // u8: MAX=255, MIN=0
        assert!(
            axiom.contains("255"),
            "u8 MAX should be 255, got: {}",
            axiom
        );
        // Unsigned doesn't use negative MIN
        assert!(!axiom.contains("-128"), "u8 should not have -128");
    }

    #[test]
    fn test_generate_axiom_wrapping_div_signed_i8_min() {
        let kind = IntrinsicKind::WrappingDiv {
            bits: 8,
            signed: true,
        };
        let axiom = kind.generate_axiom("wrapping_div_i8");
        // MIN/-1 = MIN special case
        assert!(
            axiom.contains("-128"),
            "Axiom should mention i8 MIN=-128, got: {}",
            axiom
        );
    }

    #[test]
    fn test_generate_axiom_wrapping_div_signed_i32_min() {
        let kind = IntrinsicKind::WrappingDiv {
            bits: 32,
            signed: true,
        };
        let axiom = kind.generate_axiom("wrapping_div_i32");
        assert!(
            axiom.contains("-2147483648"),
            "Axiom should mention i32 MIN, got: {}",
            axiom
        );
    }

    #[test]
    fn test_generate_axiom_wrapping_rem_signed_i8_min() {
        let kind = IntrinsicKind::WrappingRem {
            bits: 8,
            signed: true,
        };
        let axiom = kind.generate_axiom("wrapping_rem_i8");
        // MIN % -1 = 0 special case
        assert!(
            axiom.contains("-128"),
            "Axiom should mention i8 MIN=-128, got: {}",
            axiom
        );
    }

    #[test]
    fn test_generate_axiom_wrapping_rem_signed_i16_min() {
        let kind = IntrinsicKind::WrappingRem {
            bits: 16,
            signed: true,
        };
        let axiom = kind.generate_axiom("wrapping_rem_i16");
        assert!(
            axiom.contains("-32768"),
            "Axiom should mention i16 MIN=-32768, got: {}",
            axiom
        );
    }

    #[test]
    fn test_generate_axiom_wrapping_div_unsigned_no_min() {
        let kind = IntrinsicKind::WrappingDiv {
            bits: 8,
            signed: false,
        };
        let axiom = kind.generate_axiom("wrapping_div_u8");
        // Unsigned doesn't have MIN special case
        assert!(!axiom.contains("-128"), "u8 should not mention -128");
        // Just regular division
        assert!(axiom.contains("(div a b)"), "Should be simple div");
    }

    #[test]
    fn test_generate_axiom_abs_i8_min() {
        let kind = IntrinsicKind::Abs { bits: 8 };
        let axiom = kind.generate_axiom("abs_i8");
        // abs(MIN) = MIN (wraps)
        assert!(
            axiom.contains("-128"),
            "Axiom should handle MIN=-128, got: {}",
            axiom
        );
    }

    #[test]
    fn test_generate_axiom_abs_i32_min() {
        let kind = IntrinsicKind::Abs { bits: 32 };
        let axiom = kind.generate_axiom("abs_i32");
        assert!(
            axiom.contains("-2147483648"),
            "Axiom should handle MIN=-2147483648, got: {}",
            axiom
        );
    }

    // ============================================================
    // Tests verifying shift direction is correct (catches << vs >>)
    // ============================================================

    #[test]
    fn test_checked_shl_value_increases_with_shift() {
        // Left shift: 1 << 3 = 8, not 1 >> 3 = 0
        let kind = IntrinsicKind::CheckedShl { bits: 32 };
        let (_overflow, value) = kind.checked_components(&["1", "3"]).unwrap();
        // The expression uses (^ 2 n), so for n=3: 1 * 2^3 = 8
        // This is encoded, not evaluated, but structure should show multiplication
        assert!(
            value.contains("(*"),
            "Left shift should multiply, got: {}",
            value
        );
    }

    #[test]
    fn test_wrapping_shr_unsigned_uses_div() {
        // Right shift: x >> n = x / 2^n
        let kind = IntrinsicKind::WrappingShr {
            bits: 8,
            signed: false,
        };
        let expr = kind.inline_expression(&["x", "n"]).expect("Should inline");
        // Should divide, not multiply
        assert!(
            expr.contains("(div x"),
            "Right shift should divide, got: {}",
            expr
        );
    }

    #[test]
    fn test_wrapping_shr_signed_uses_div() {
        let kind = IntrinsicKind::WrappingShr {
            bits: 32,
            signed: true,
        };
        let expr = kind.inline_expression(&["x", "n"]).expect("Should inline");
        // Signed right shift should also divide (arithmetic shift)
        assert!(
            expr.contains("div"),
            "Signed right shift should divide, got: {}",
            expr
        );
    }

    // ============================================================
    // Tests for generate_axiom modulus calculations
    // These kill mutants for 1u128 << bits formula
    // ============================================================

    #[test]
    fn test_generate_axiom_wrapping_mul_u8_modulus() {
        let kind = IntrinsicKind::WrappingMul {
            bits: 8,
            signed: false,
        };
        let axiom = kind.generate_axiom("wrapping_mul_u8");
        // Modulus should be 2^8 = 256
        assert!(
            axiom.contains("256"),
            "u8 wrapping_mul should mod by 256, got: {}",
            axiom
        );
    }

    #[test]
    fn test_generate_axiom_wrapping_mul_u16_modulus() {
        let kind = IntrinsicKind::WrappingMul {
            bits: 16,
            signed: false,
        };
        let axiom = kind.generate_axiom("wrapping_mul_u16");
        // Modulus should be 2^16 = 65536
        assert!(
            axiom.contains("65536"),
            "u16 wrapping_mul should mod by 65536, got: {}",
            axiom
        );
    }

    #[test]
    fn test_generate_axiom_wrapping_neg_u8_modulus() {
        let kind = IntrinsicKind::WrappingNeg {
            bits: 8,
            signed: false,
        };
        let axiom = kind.generate_axiom("wrapping_neg_u8");
        // Modulus should be 2^8 = 256
        assert!(
            axiom.contains("256"),
            "u8 wrapping_neg should mod by 256, got: {}",
            axiom
        );
    }

    #[test]
    fn test_generate_axiom_wrapping_neg_u32_modulus() {
        let kind = IntrinsicKind::WrappingNeg {
            bits: 32,
            signed: false,
        };
        let axiom = kind.generate_axiom("wrapping_neg_u32");
        // Modulus should be 2^32 = 4294967296
        assert!(
            axiom.contains("4294967296"),
            "u32 wrapping_neg should mod by 4294967296, got: {}",
            axiom
        );
    }

    #[test]
    fn test_generate_axiom_wrapping_neg_i32_signed() {
        let kind = IntrinsicKind::WrappingNeg {
            bits: 32,
            signed: true,
        };
        let axiom = kind.generate_axiom("wrapping_neg_i32");
        // For signed, should just be negation, no modulus
        assert!(
            axiom.contains("(- 0 a)"),
            "i32 wrapping_neg should just negate, got: {}",
            axiom
        );
        assert!(
            !axiom.contains("mod"),
            "i32 wrapping_neg should NOT use mod, got: {}",
            axiom
        );
    }

    #[test]
    fn test_generate_axiom_wrapping_shl_u8_modulus() {
        let kind = IntrinsicKind::WrappingShl { bits: 8 };
        let axiom = kind.generate_axiom("wrapping_shl_u8");
        // Modulus should be 2^8 = 256
        assert!(
            axiom.contains("256"),
            "u8 wrapping_shl should mod by 256, got: {}",
            axiom
        );
        // Should also use mod n 8 for shift amount wrapping
        assert!(
            axiom.contains("(mod n 8)"),
            "Should wrap shift amount mod 8, got: {}",
            axiom
        );
    }

    #[test]
    fn test_generate_axiom_wrapping_shl_u32_modulus() {
        let kind = IntrinsicKind::WrappingShl { bits: 32 };
        let axiom = kind.generate_axiom("wrapping_shl_u32");
        // Modulus should be 2^32 = 4294967296
        assert!(
            axiom.contains("4294967296"),
            "u32 wrapping_shl should mod by 4294967296, got: {}",
            axiom
        );
    }

    // ============================================================
    // Tests for parse_int_type coverage (u128, usize, i128, isize)
    // ============================================================

    #[test]
    fn test_parse_u128_intrinsic() {
        // Test that u128 type is recognized
        let kind = IntrinsicKind::from_smt_name("core_num_impl_u128_wrapping_add");
        assert!(kind.is_some(), "Should recognize u128 intrinsic");
        let kind = kind.unwrap();
        match kind {
            IntrinsicKind::WrappingAdd { bits, signed } => {
                assert_eq!(bits, 128, "u128 should have 128 bits");
                assert!(!signed, "u128 should be unsigned");
            }
            _ => panic!("Should be WrappingAdd"),
        }
    }

    #[test]
    fn test_parse_usize_intrinsic() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_usize_wrapping_add");
        assert!(kind.is_some(), "Should recognize usize intrinsic");
        let kind = kind.unwrap();
        match kind {
            IntrinsicKind::WrappingAdd { bits, signed } => {
                assert_eq!(bits, 64, "usize should have 64 bits on 64-bit platform");
                assert!(!signed, "usize should be unsigned");
            }
            _ => panic!("Should be WrappingAdd"),
        }
    }

    #[test]
    fn test_parse_i128_intrinsic() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_i128_wrapping_add");
        assert!(kind.is_some(), "Should recognize i128 intrinsic");
        let kind = kind.unwrap();
        match kind {
            IntrinsicKind::WrappingAdd { bits, signed } => {
                assert_eq!(bits, 128, "i128 should have 128 bits");
                assert!(signed, "i128 should be signed");
            }
            _ => panic!("Should be WrappingAdd"),
        }
    }

    #[test]
    fn test_parse_isize_intrinsic() {
        let kind = IntrinsicKind::from_smt_name("core_num_impl_isize_wrapping_add");
        assert!(kind.is_some(), "Should recognize isize intrinsic");
        let kind = kind.unwrap();
        match kind {
            IntrinsicKind::WrappingAdd { bits, signed } => {
                assert_eq!(bits, 64, "isize should have 64 bits on 64-bit platform");
                assert!(signed, "isize should be signed");
            }
            _ => panic!("Should be WrappingAdd"),
        }
    }

    // ============================================================
    // Tests for collect_axioms_for_functions
    // ============================================================

    #[test]
    fn test_collect_axioms_for_functions_returns_axioms() {
        use crate::UninterpretedFunction;
        use kani_fast_kinduction::SmtType;
        use std::collections::HashMap;

        let mut functions = HashMap::new();
        functions.insert(
            "core_num_impl_u8_wrapping_add".to_string(),
            UninterpretedFunction {
                name: "core_num_impl_u8_wrapping_add".to_string(),
                param_types: vec![],
                return_type: SmtType::Int,
            },
        );

        let axioms = collect_axioms_for_functions(&functions);
        assert!(!axioms.is_empty(), "Should return at least one axiom");
        assert!(
            axioms[0].contains("wrapping_add") || axioms[0].contains("256"),
            "Axiom should be for wrapping_add, got: {}",
            axioms[0]
        );
    }

    #[test]
    fn test_collect_axioms_for_functions_empty_for_unknown() {
        use crate::UninterpretedFunction;
        use kani_fast_kinduction::SmtType;
        use std::collections::HashMap;

        let mut functions = HashMap::new();
        functions.insert(
            "unknown_function".to_string(),
            UninterpretedFunction {
                name: "unknown_function".to_string(),
                param_types: vec![],
                return_type: SmtType::Int,
            },
        );

        let axioms = collect_axioms_for_functions(&functions);
        assert!(
            axioms.is_empty(),
            "Should return empty for unknown functions"
        );
    }

    #[test]
    fn test_collect_axioms_for_functions_multiple() {
        use crate::UninterpretedFunction;
        use kani_fast_kinduction::SmtType;
        use std::collections::HashMap;

        let mut functions = HashMap::new();
        functions.insert(
            "core_num_impl_u8_wrapping_add".to_string(),
            UninterpretedFunction {
                name: "core_num_impl_u8_wrapping_add".to_string(),
                param_types: vec![],
                return_type: SmtType::Int,
            },
        );
        functions.insert(
            "core_num_impl_u16_wrapping_mul".to_string(),
            UninterpretedFunction {
                name: "core_num_impl_u16_wrapping_mul".to_string(),
                param_types: vec![],
                return_type: SmtType::Int,
            },
        );

        let axioms = collect_axioms_for_functions(&functions);
        assert_eq!(axioms.len(), 2, "Should return two axioms");
    }
}

// ============================================================================
// Kani Verification Harnesses
// ============================================================================
//
// These harnesses verify overflow detection semantics match Rust's behavior.
// Run with: cargo kani -p kani-fast-chc
//
// Background: The intrinsics module provides SMT axioms for checked/overflowing
// arithmetic. These harnesses verify our understanding of Rust's overflow semantics.

#[cfg(kani)]
mod kani_proofs {
    /// Verify overflowing_add correctly detects overflow for i8
    ///
    /// Rust's overflowing_add returns (wrapping_result, did_overflow).
    /// Overflow occurs when adding two positives gives negative, or two negatives gives positive.
    #[kani::proof]
    fn verify_overflowing_add_i8() {
        let a: i8 = kani::any();
        let b: i8 = kani::any();
        let (result, overflow) = a.overflowing_add(b);

        // If no overflow, result should equal mathematical sum
        if !overflow {
            // Safe to compute - use wider type to avoid overflow
            let wide_sum = (a as i16) + (b as i16);
            assert!(wide_sum >= i8::MIN as i16 && wide_sum <= i8::MAX as i16);
            assert!(result as i16 == wide_sum);
        }

        // Verify wrapping behavior: result is always (a + b) mod 2^8
        let expected_wrapped = a.wrapping_add(b);
        assert_eq!(result, expected_wrapped);
    }

    /// Verify overflowing_sub correctly detects underflow for i8
    #[kani::proof]
    fn verify_overflowing_sub_i8() {
        let a: i8 = kani::any();
        let b: i8 = kani::any();
        let (result, overflow) = a.overflowing_sub(b);

        // Verify wrapping behavior
        let expected_wrapped = a.wrapping_sub(b);
        assert_eq!(result, expected_wrapped);

        // If no overflow, result should equal mathematical difference
        if !overflow {
            let wide_diff = (a as i16) - (b as i16);
            assert!(wide_diff >= i8::MIN as i16 && wide_diff <= i8::MAX as i16);
        }
    }

    /// Verify checked_add returns None on overflow
    #[kani::proof]
    fn verify_checked_add_i8() {
        let a: i8 = kani::any();
        let b: i8 = kani::any();

        match a.checked_add(b) {
            Some(result) => {
                // No overflow occurred - verify mathematical correctness
                let wide_sum = (a as i16) + (b as i16);
                assert!(wide_sum >= i8::MIN as i16 && wide_sum <= i8::MAX as i16);
                assert!(result as i16 == wide_sum);
            }
            None => {
                // Overflow occurred - verify sum is outside range
                let wide_sum = (a as i16) + (b as i16);
                assert!(wide_sum < i8::MIN as i16 || wide_sum > i8::MAX as i16);
            }
        }
    }

    /// Verify saturating_add clamps at bounds
    #[kani::proof]
    fn verify_saturating_add_i8() {
        let a: i8 = kani::any();
        let b: i8 = kani::any();
        let result = a.saturating_add(b);

        // Result is always within valid range
        assert!(result >= i8::MIN && result <= i8::MAX);

        let wide_sum = (a as i16) + (b as i16);
        if wide_sum > i8::MAX as i16 {
            assert!(result == i8::MAX);
        } else if wide_sum < i8::MIN as i16 {
            assert!(result == i8::MIN);
        } else {
            assert!(result as i16 == wide_sum);
        }
    }

    /// Verify wrapping_add is equivalent to modular arithmetic
    #[kani::proof]
    fn verify_wrapping_add_u8() {
        let a: u8 = kani::any();
        let b: u8 = kani::any();
        let result = a.wrapping_add(b);

        // wrapping_add is (a + b) mod 256
        let wide_sum = (a as u16) + (b as u16);
        let expected = (wide_sum % 256) as u8;
        assert_eq!(result, expected);
    }

    /// Verify overflow detection for positive + positive
    #[kani::proof]
    fn verify_positive_overflow_detection() {
        let a: i8 = kani::any();
        let b: i8 = kani::any();
        kani::assume(a > 0 && b > 0);

        let (result, overflow) = a.overflowing_add(b);

        // Positive + positive overflows iff result is negative
        if overflow {
            assert!(result < 0);
        } else {
            assert!(result >= a && result >= b);
        }
    }

    /// Verify overflow detection for negative + negative
    #[kani::proof]
    fn verify_negative_overflow_detection() {
        let a: i8 = kani::any();
        let b: i8 = kani::any();
        kani::assume(a < 0 && b < 0);

        let (result, overflow) = a.overflowing_add(b);

        // Negative + negative underflows iff result is non-negative
        if overflow {
            assert!(result >= 0);
        } else {
            assert!(result < 0);
        }
    }

    /// Verify checked_mul handles multiplication overflow
    #[kani::proof]
    fn verify_checked_mul_i8() {
        let a: i8 = kani::any();
        let b: i8 = kani::any();

        match a.checked_mul(b) {
            Some(result) => {
                // No overflow - verify mathematical correctness
                let wide_product = (a as i16) * (b as i16);
                assert!(wide_product >= i8::MIN as i16 && wide_product <= i8::MAX as i16);
                assert!(result as i16 == wide_product);
            }
            None => {
                // Overflow - verify product is outside range
                let wide_product = (a as i16) * (b as i16);
                assert!(wide_product < i8::MIN as i16 || wide_product > i8::MAX as i16);
            }
        }
    }

    /// Verify the MIN / -1 overflow case for division
    #[kani::proof]
    fn verify_min_div_minus_one_overflow() {
        // i8::MIN / -1 would be 128, which overflows i8
        let result = i8::MIN.overflowing_div(-1);
        assert!(result.1); // Overflow flag should be true
        assert!(result.0 == i8::MIN); // Wraps to MIN
    }

    /// Verify abs overflow at MIN
    #[kani::proof]
    fn verify_abs_min_overflow() {
        // |i8::MIN| = 128, which overflows to i8::MIN
        let result = i8::MIN.wrapping_abs();
        assert!(result == i8::MIN);

        // checked_abs should return None for MIN
        assert!(i8::MIN.checked_abs().is_none());
    }
}
