//! C Value Representation
//!
//! This module defines how C values are represented, including:
//!
//! - Integer values (various widths, signed/unsigned)
//! - Floating-point values
//! - Pointer values
//! - Aggregate values (structs, arrays, unions)
//! - Undefined values (uninitialized memory)
//!
//! ## Key Concepts
//!
//! ### Undefined Values
//!
//! Reading uninitialized memory yields an undefined value. Using
//! undefined values in most operations is undefined behavior.
//! We track this explicitly to detect UB.
//!
//! ### Value Representation
//!
//! Values can be:
//! - **Concrete**: Known, valid values
//! - **Undefined**: Not yet initialized
//! - **Poison**: Result of UB (for future use in optimizations)

use crate::memory::Pointer;
use crate::types::{CType, IntKind, Signedness};
use crate::ub::{UBKind, UBResult};
use serde::{Deserialize, Serialize};

/// A C value
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CValue {
    /// Undefined/uninitialized value
    Undef,

    /// Boolean value (C99 _Bool)
    Bool(bool),

    /// Signed integer value (any width, stored as i128)
    Int(i128),

    /// Unsigned integer value (any width, stored as u128)
    UInt(u128),

    /// Float value
    Float(f32),

    /// Double value
    Double(f64),

    /// Pointer value
    Pointer(Pointer),

    /// Struct value (ordered list of field values)
    Struct(Vec<CValue>),

    /// Union value (which field is active + value)
    Union {
        active_field: usize,
        value: Box<CValue>,
    },

    /// Array value
    Array(Vec<CValue>),
}

impl CValue {
    /// Create a zero value of the given type
    pub fn zero(ty: &CType) -> Self {
        match ty {
            // void and function types have no value
            CType::Void | CType::Function { .. } => CValue::Undef,

            CType::Int(IntKind::Bool, _) => CValue::Bool(false),

            CType::Int(_, Signedness::Signed) | CType::Enum { .. } => CValue::Int(0),

            CType::Int(_, Signedness::Unsigned) => CValue::UInt(0),

            CType::Float(crate::types::FloatKind::Float) => CValue::Float(0.0),

            CType::Float(crate::types::FloatKind::Double | crate::types::FloatKind::LongDouble) => {
                CValue::Double(0.0)
            }

            CType::Pointer(_) => CValue::Pointer(Pointer::null()),

            CType::Struct { fields, .. } => {
                CValue::Struct(fields.iter().map(|f| CValue::zero(&f.ty)).collect())
            }

            CType::Union { fields, .. } => {
                let first_field = fields
                    .first()
                    .map_or(CValue::Undef, |f| CValue::zero(&f.ty));
                CValue::Union {
                    active_field: 0,
                    value: Box::new(first_field),
                }
            }

            CType::Array(elem, count) => CValue::Array(vec![CValue::zero(elem); *count]),

            CType::TypeDef(_) => panic!("typedef should be resolved"),

            CType::Qualified { ty, .. } => CValue::zero(ty),
        }
    }

    /// Check if this value is undefined
    pub fn is_undef(&self) -> bool {
        matches!(self, CValue::Undef)
    }

    /// Check if this value is a null pointer
    pub fn is_null_ptr(&self) -> bool {
        matches!(self, CValue::Pointer(p) if p.is_null())
    }

    /// Convert to bool (for conditions)
    ///
    /// In C, 0/null is false, everything else is true.
    pub fn to_bool(&self) -> UBResult<bool> {
        match self {
            CValue::Undef => Err(UBKind::UninitializedRead),
            CValue::Bool(b) => Ok(*b),
            CValue::Int(i) => Ok(*i != 0),
            CValue::UInt(u) => Ok(*u != 0),
            CValue::Float(f) => Ok(*f != 0.0),
            CValue::Double(d) => Ok(*d != 0.0),
            CValue::Pointer(p) => Ok(!p.is_null()),
            CValue::Struct(_) | CValue::Union { .. } | CValue::Array(_) => Err(UBKind::Other(
                "aggregate value in boolean context".to_string(),
            )),
        }
    }

    /// Convert to i128 (for integer operations)
    pub fn to_int(&self) -> UBResult<i128> {
        match self {
            CValue::Undef => Err(UBKind::UninitializedRead),
            CValue::Bool(b) => Ok(if *b { 1 } else { 0 }),
            CValue::Int(i) => Ok(*i),
            CValue::UInt(u) => Ok(*u as i128),
            CValue::Float(f) => float_to_int_checked(f64::from(*f), i128::MIN, i128::MAX),
            CValue::Double(d) => float_to_int_checked(*d, i128::MIN, i128::MAX),
            _ => Err(UBKind::Other("cannot convert to integer".to_string())),
        }
    }

    /// Convert to u128 (for unsigned integer operations)
    pub fn to_uint(&self) -> UBResult<u128> {
        match self {
            CValue::Undef => Err(UBKind::UninitializedRead),
            CValue::Bool(b) => Ok(if *b { 1 } else { 0 }),
            CValue::Int(i) => Ok(*i as u128),
            CValue::UInt(u) => Ok(*u),
            CValue::Float(f) => float_to_uint_checked(f64::from(*f), u128::MAX),
            CValue::Double(d) => float_to_uint_checked(*d, u128::MAX),
            _ => Err(UBKind::Other(
                "cannot convert to unsigned integer".to_string(),
            )),
        }
    }

    /// Convert to f64 (for floating-point operations)
    pub fn to_double(&self) -> UBResult<f64> {
        match self {
            CValue::Undef => Err(UBKind::UninitializedRead),
            CValue::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
            CValue::Int(i) => Ok(*i as f64),
            CValue::UInt(u) => Ok(*u as f64),
            CValue::Float(f) => Ok(*f as f64),
            CValue::Double(d) => Ok(*d),
            _ => Err(UBKind::Other("cannot convert to double".to_string())),
        }
    }

    /// Convert to pointer
    pub fn to_ptr(&self) -> UBResult<Pointer> {
        match self {
            CValue::Undef => Err(UBKind::UninitializedRead),
            CValue::Pointer(p) => Ok(*p),
            CValue::Int(0) | CValue::UInt(0) => Ok(Pointer::null()),
            _ => Err(UBKind::Other("cannot convert to pointer".to_string())),
        }
    }

    /// Get size in bytes when stored
    pub fn size(&self, ty: &CType) -> usize {
        ty.size()
    }

    /// Cast to a different type
    pub fn cast(&self, from: &CType, to: &CType) -> UBResult<CValue> {
        if self.is_undef() {
            return Err(UBKind::UninitializedRead);
        }

        match (from.unqualified(), to.unqualified()) {
            // Same type: no-op
            (a, b) if a.is_compatible(b) => Ok(self.clone()),

            // Bool conversions (must come before general Int patterns)
            (_, CType::Int(IntKind::Bool, _)) => Ok(CValue::Bool(self.to_bool()?)),

            (CType::Int(IntKind::Bool, _), CType::Int(kind, sign)) => {
                let b = match self {
                    CValue::Bool(b) => *b,
                    _ => self.to_bool()?,
                };
                Ok(truncate_int(if b { 1 } else { 0 }, *kind, *sign))
            }

            // Integer to integer
            (CType::Int(_, _) | CType::Enum { .. }, CType::Int(to_kind, to_sign)) => {
                let val = self.to_int()?;
                Ok(truncate_int(val, *to_kind, *to_sign))
            }

            // Integer to float
            (CType::Int(_, _) | CType::Enum { .. }, CType::Float(_)) => {
                Ok(CValue::Double(self.to_int()? as f64))
            }

            // Float to integer
            (CType::Float(_), CType::Int(to_kind, to_sign)) => {
                let f = self.to_double()?;
                let truncated = match to_sign {
                    Signedness::Signed => {
                        float_to_int_checked(f, to_kind.signed_min(), to_kind.signed_max())?
                    }
                    Signedness::Unsigned => {
                        float_to_uint_checked(f, to_kind.unsigned_max())? as i128
                    }
                };
                Ok(truncate_int(truncated, *to_kind, *to_sign))
            }

            // Float to float
            (CType::Float(_), CType::Float(to_kind)) => {
                let f = self.to_double()?;
                match to_kind {
                    crate::types::FloatKind::Float => Ok(CValue::Float(f as f32)),
                    _ => Ok(CValue::Double(f)),
                }
            }

            // Pointer to pointer
            (CType::Pointer(_), CType::Pointer(_)) => Ok(self.clone()),

            // Integer to pointer
            (CType::Int(_, _), CType::Pointer(_)) => {
                // This is implementation-defined, but we allow 0 -> null
                let val = self.to_int()?;
                if val == 0 {
                    Ok(CValue::Pointer(Pointer::null()))
                } else {
                    Err(UBKind::Other(
                        "non-zero integer to pointer cast".to_string(),
                    ))
                }
            }

            // Pointer to integer
            (CType::Pointer(_), CType::Int(_, _)) => {
                // Allowed but implementation-defined
                let ptr = self.to_ptr()?;
                if ptr.is_null() {
                    Ok(CValue::Int(0))
                } else {
                    // Return some encoding - block_id + offset
                    Ok(CValue::Int(
                        ptr.block.0 as i128 * 1_000_000 + ptr.offset as i128,
                    ))
                }
            }

            _ => Err(UBKind::Other(format!(
                "unsupported cast from {from:?} to {to:?}"
            ))),
        }
    }
}

/// Truncate/extend integer to the specified width and signedness
fn truncate_int(val: i128, kind: IntKind, sign: Signedness) -> CValue {
    let bits = kind.size() * 8;
    let mask = if bits >= 128 {
        u128::MAX
    } else {
        (1u128 << bits) - 1
    };

    match sign {
        Signedness::Unsigned => CValue::UInt((val as u128) & mask),
        Signedness::Signed => {
            let unsigned = (val as u128) & mask;
            let sign_bit = 1u128 << (bits - 1);
            if unsigned & sign_bit != 0 {
                // Sign extend
                let sign_extended = unsigned | !mask;
                CValue::Int(sign_extended as i128)
            } else {
                CValue::Int(unsigned as i128)
            }
        }
    }
}

fn float_to_int_checked(value: f64, min: i128, max: i128) -> UBResult<i128> {
    if !value.is_finite() || value < min as f64 || value > max as f64 {
        return Err(UBKind::FloatToIntOverflow);
    }
    Ok(value.trunc() as i128)
}

fn float_to_uint_checked(value: f64, max: u128) -> UBResult<u128> {
    if !value.is_finite() || value < 0.0 || value > max as f64 {
        return Err(UBKind::FloatToIntOverflow);
    }
    Ok(value.trunc() as u128)
}

/// Binary operations on C values
impl CValue {
    /// Addition
    pub fn add(&self, other: &CValue, ty: &CType) -> UBResult<CValue> {
        match (self, other) {
            (CValue::Int(a), CValue::Int(b)) => {
                // Check for overflow in the target type, not i128
                if let CType::Int(kind, Signedness::Signed) = ty.unqualified() {
                    if check_signed_add_overflow(*a, *b, *kind) {
                        return Err(UBKind::SignedOverflow);
                    }
                }
                let result = a.wrapping_add(*b);
                Ok(truncate_to_type(result, ty))
            }
            (CValue::UInt(a), CValue::UInt(b)) => {
                // Unsigned wraps
                let result = a.wrapping_add(*b);
                Ok(truncate_to_type_unsigned(result, ty))
            }
            (CValue::Float(a), CValue::Float(b)) => Ok(CValue::Float(a + b)),
            (CValue::Double(a), CValue::Double(b)) => Ok(CValue::Double(a + b)),
            (CValue::Pointer(p), CValue::Int(i)) => {
                // Pointer arithmetic with signed offset
                let offset = to_pointer_offset(*i)?;
                let new_ptr = p.offset(offset).ok_or(UBKind::PointerOverflow)?;
                Ok(CValue::Pointer(new_ptr))
            }
            (CValue::Pointer(p), CValue::UInt(u)) => {
                // Pointer arithmetic with unsigned offset
                let offset = to_pointer_offset_unsigned(*u)?;
                let new_ptr = p.offset(offset).ok_or(UBKind::PointerOverflow)?;
                Ok(CValue::Pointer(new_ptr))
            }
            (CValue::Int(i), CValue::Pointer(p)) => {
                let offset = to_pointer_offset(*i)?;
                let new_ptr = p.offset(offset).ok_or(UBKind::PointerOverflow)?;
                Ok(CValue::Pointer(new_ptr))
            }
            (CValue::UInt(u), CValue::Pointer(p)) => {
                let offset = to_pointer_offset_unsigned(*u)?;
                let new_ptr = p.offset(offset).ok_or(UBKind::PointerOverflow)?;
                Ok(CValue::Pointer(new_ptr))
            }
            (CValue::Undef, _) | (_, CValue::Undef) => Err(UBKind::UninitializedRead),
            _ => Err(UBKind::Other("invalid operands for addition".to_string())),
        }
    }

    /// Subtraction
    pub fn sub(&self, other: &CValue, ty: &CType) -> UBResult<CValue> {
        match (self, other) {
            (CValue::Int(a), CValue::Int(b)) => {
                // Check for overflow in the target type, not i128
                if let CType::Int(kind, Signedness::Signed) = ty.unqualified() {
                    if check_signed_sub_overflow(*a, *b, *kind) {
                        return Err(UBKind::SignedOverflow);
                    }
                }
                let result = a.wrapping_sub(*b);
                Ok(truncate_to_type(result, ty))
            }
            (CValue::UInt(a), CValue::UInt(b)) => {
                let result = a.wrapping_sub(*b);
                Ok(truncate_to_type_unsigned(result, ty))
            }
            (CValue::Float(a), CValue::Float(b)) => Ok(CValue::Float(a - b)),
            (CValue::Double(a), CValue::Double(b)) => Ok(CValue::Double(a - b)),
            (CValue::Pointer(p), CValue::Int(i)) => {
                let offset = to_pointer_offset(*i)?;
                let neg_offset = negate_offset(offset)?;
                let new_ptr = p.offset(neg_offset).ok_or(UBKind::PointerOverflow)?;
                Ok(CValue::Pointer(new_ptr))
            }
            (CValue::Pointer(p), CValue::UInt(u)) => {
                let offset = to_pointer_offset_unsigned(*u)?;
                let neg_offset = negate_offset(offset)?;
                let new_ptr = p.offset(neg_offset).ok_or(UBKind::PointerOverflow)?;
                Ok(CValue::Pointer(new_ptr))
            }
            (CValue::Pointer(p1), CValue::Pointer(p2)) => {
                // Pointer difference
                let diff = p1.diff(*p2).ok_or(UBKind::InvalidPointerSubtraction)?;
                Ok(CValue::Int(diff as i128))
            }
            (CValue::Undef, _) | (_, CValue::Undef) => Err(UBKind::UninitializedRead),
            _ => Err(UBKind::Other(
                "invalid operands for subtraction".to_string(),
            )),
        }
    }

    /// Multiplication
    pub fn mul(&self, other: &CValue, ty: &CType) -> UBResult<CValue> {
        match (self, other) {
            (CValue::Int(a), CValue::Int(b)) => {
                // Check for overflow in the target type, not i128
                if let CType::Int(kind, Signedness::Signed) = ty.unqualified() {
                    if check_signed_mul_overflow(*a, *b, *kind) {
                        return Err(UBKind::SignedOverflow);
                    }
                }
                let result = a.wrapping_mul(*b);
                Ok(truncate_to_type(result, ty))
            }
            (CValue::UInt(a), CValue::UInt(b)) => {
                let result = a.wrapping_mul(*b);
                Ok(truncate_to_type_unsigned(result, ty))
            }
            (CValue::Float(a), CValue::Float(b)) => Ok(CValue::Float(a * b)),
            (CValue::Double(a), CValue::Double(b)) => Ok(CValue::Double(a * b)),
            (CValue::Undef, _) | (_, CValue::Undef) => Err(UBKind::UninitializedRead),
            _ => Err(UBKind::Other(
                "invalid operands for multiplication".to_string(),
            )),
        }
    }

    /// Division
    pub fn div(&self, other: &CValue, ty: &CType) -> UBResult<CValue> {
        match (self, other) {
            (CValue::Int(a), CValue::Int(b)) => {
                if *b == 0 {
                    return Err(UBKind::DivisionByZero);
                }
                // Check for overflow: INT_MIN / -1
                if *a == i64::MIN as i128 && *b == -1 {
                    return Err(UBKind::DivisionOverflow);
                }
                Ok(truncate_to_type(a / b, ty))
            }
            (CValue::UInt(a), CValue::UInt(b)) => {
                if *b == 0 {
                    return Err(UBKind::DivisionByZero);
                }
                Ok(truncate_to_type_unsigned(a / b, ty))
            }
            (CValue::Float(a), CValue::Float(b)) => {
                Ok(CValue::Float(a / b)) // Float division by zero is defined (inf/nan)
            }
            (CValue::Double(a), CValue::Double(b)) => Ok(CValue::Double(a / b)),
            (CValue::Undef, _) | (_, CValue::Undef) => Err(UBKind::UninitializedRead),
            _ => Err(UBKind::Other("invalid operands for division".to_string())),
        }
    }

    /// Modulo
    pub fn rem(&self, other: &CValue, ty: &CType) -> UBResult<CValue> {
        match (self, other) {
            (CValue::Int(a), CValue::Int(b)) => {
                if *b == 0 {
                    return Err(UBKind::DivisionByZero);
                }
                if *a == i64::MIN as i128 && *b == -1 {
                    return Err(UBKind::DivisionOverflow);
                }
                Ok(truncate_to_type(a % b, ty))
            }
            (CValue::UInt(a), CValue::UInt(b)) => {
                if *b == 0 {
                    return Err(UBKind::DivisionByZero);
                }
                Ok(truncate_to_type_unsigned(a % b, ty))
            }
            (CValue::Undef, _) | (_, CValue::Undef) => Err(UBKind::UninitializedRead),
            _ => Err(UBKind::Other("invalid operands for modulo".to_string())),
        }
    }

    /// Bitwise AND
    pub fn bit_and(&self, other: &CValue, ty: &CType) -> UBResult<CValue> {
        match (self, other) {
            (CValue::Int(a), CValue::Int(b)) => Ok(truncate_to_type(a & b, ty)),
            (CValue::UInt(a), CValue::UInt(b)) => Ok(truncate_to_type_unsigned(a & b, ty)),
            (CValue::Undef, _) | (_, CValue::Undef) => Err(UBKind::UninitializedRead),
            _ => Err(UBKind::Other(
                "invalid operands for bitwise AND".to_string(),
            )),
        }
    }

    /// Bitwise OR
    pub fn bit_or(&self, other: &CValue, ty: &CType) -> UBResult<CValue> {
        match (self, other) {
            (CValue::Int(a), CValue::Int(b)) => Ok(truncate_to_type(a | b, ty)),
            (CValue::UInt(a), CValue::UInt(b)) => Ok(truncate_to_type_unsigned(a | b, ty)),
            (CValue::Undef, _) | (_, CValue::Undef) => Err(UBKind::UninitializedRead),
            _ => Err(UBKind::Other("invalid operands for bitwise OR".to_string())),
        }
    }

    /// Bitwise XOR
    pub fn bit_xor(&self, other: &CValue, ty: &CType) -> UBResult<CValue> {
        match (self, other) {
            (CValue::Int(a), CValue::Int(b)) => Ok(truncate_to_type(a ^ b, ty)),
            (CValue::UInt(a), CValue::UInt(b)) => Ok(truncate_to_type_unsigned(a ^ b, ty)),
            (CValue::Undef, _) | (_, CValue::Undef) => Err(UBKind::UninitializedRead),
            _ => Err(UBKind::Other(
                "invalid operands for bitwise XOR".to_string(),
            )),
        }
    }

    /// Left shift
    pub fn shl(&self, other: &CValue, ty: &CType) -> UBResult<CValue> {
        let shift_i128 = other.to_int()?;
        // SAFETY: C type sizes bounded at ~16 bytes max, so bit_width <= 128 fits in u32
        let bit_width = (ty.size() * 8) as u32;

        // Negative shift is UB in C
        if shift_i128 < 0 {
            return Err(UBKind::InvalidShift(format!(
                "negative shift amount {shift_i128}"
            )));
        }

        // Shift amount must fit in u32 for Rust shift operations
        let shift = u32::try_from(shift_i128).map_err(|_| {
            UBKind::InvalidShift(format!("shift amount {shift_i128} exceeds u32::MAX"))
        })?;

        if shift >= bit_width {
            return Err(UBKind::InvalidShift(format!(
                "shift {shift} >= bit width {bit_width}"
            )));
        }

        match self {
            CValue::Int(a) => {
                // Signed left shift: UB if shifts into sign bit
                if *a < 0 {
                    return Err(UBKind::SignedOverflow);
                }
                let result = (*a as u128).wrapping_shl(shift) as i128;
                Ok(truncate_to_type(result, ty))
            }
            CValue::UInt(a) => {
                let result = a.wrapping_shl(shift);
                Ok(truncate_to_type_unsigned(result, ty))
            }
            CValue::Undef => Err(UBKind::UninitializedRead),
            _ => Err(UBKind::Other("invalid operand for left shift".to_string())),
        }
    }

    /// Right shift
    pub fn shr(&self, other: &CValue, ty: &CType) -> UBResult<CValue> {
        let shift_i128 = other.to_int()?;
        // SAFETY: C type sizes bounded at ~16 bytes max, so bit_width <= 128 fits in u32
        let bit_width = (ty.size() * 8) as u32;

        // Negative shift is UB in C
        if shift_i128 < 0 {
            return Err(UBKind::InvalidShift(format!(
                "negative shift amount {shift_i128}"
            )));
        }

        // Shift amount must fit in u32 for Rust shift operations
        let shift = u32::try_from(shift_i128).map_err(|_| {
            UBKind::InvalidShift(format!("shift amount {shift_i128} exceeds u32::MAX"))
        })?;

        if shift >= bit_width {
            return Err(UBKind::InvalidShift(format!(
                "shift {shift} >= bit width {bit_width}"
            )));
        }

        match self {
            CValue::Int(a) => {
                // Arithmetic shift for signed (preserves sign)
                let result = a >> shift;
                Ok(truncate_to_type(result, ty))
            }
            CValue::UInt(a) => {
                // Logical shift for unsigned
                let result = a >> shift;
                Ok(truncate_to_type_unsigned(result, ty))
            }
            CValue::Undef => Err(UBKind::UninitializedRead),
            _ => Err(UBKind::Other("invalid operand for right shift".to_string())),
        }
    }

    /// Unary negation
    pub fn neg(&self, ty: &CType) -> UBResult<CValue> {
        match self {
            CValue::Int(a) => {
                let result = a.checked_neg().ok_or(UBKind::SignedOverflow)?;
                Ok(truncate_to_type(result, ty))
            }
            CValue::UInt(a) => {
                // Unsigned negation wraps
                let result = (0u128).wrapping_sub(*a);
                Ok(truncate_to_type_unsigned(result, ty))
            }
            CValue::Float(f) => Ok(CValue::Float(-f)),
            CValue::Double(d) => Ok(CValue::Double(-d)),
            CValue::Undef => Err(UBKind::UninitializedRead),
            _ => Err(UBKind::Other("invalid operand for negation".to_string())),
        }
    }

    /// Bitwise NOT
    pub fn bit_not(&self, ty: &CType) -> UBResult<CValue> {
        match self {
            CValue::Int(a) => Ok(truncate_to_type(!a, ty)),
            CValue::UInt(a) => Ok(truncate_to_type_unsigned(!a, ty)),
            CValue::Undef => Err(UBKind::UninitializedRead),
            _ => Err(UBKind::Other("invalid operand for bitwise NOT".to_string())),
        }
    }

    /// Logical NOT
    pub fn log_not(&self) -> UBResult<CValue> {
        Ok(CValue::Int(if self.to_bool()? { 0 } else { 1 }))
    }

    /// Equality comparison
    pub fn eq(&self, other: &CValue) -> UBResult<CValue> {
        let result = match (self, other) {
            (CValue::Int(a), CValue::Int(b)) => a == b,
            (CValue::UInt(a), CValue::UInt(b)) => a == b,
            (CValue::Float(a), CValue::Float(b)) => a == b,
            (CValue::Double(a), CValue::Double(b)) => a == b,
            (CValue::Pointer(a), CValue::Pointer(b)) => a == b,
            (CValue::Undef, _) | (_, CValue::Undef) => {
                return Err(UBKind::UninitializedRead);
            }
            _ => return Err(UBKind::Other("invalid operands for equality".to_string())),
        };
        Ok(CValue::Int(if result { 1 } else { 0 }))
    }

    /// Inequality comparison
    pub fn ne(&self, other: &CValue) -> UBResult<CValue> {
        let eq_result = self.eq(other)?;
        match eq_result {
            CValue::Int(1) => Ok(CValue::Int(0)),
            CValue::Int(0) => Ok(CValue::Int(1)),
            _ => unreachable!(),
        }
    }

    /// Less than comparison
    pub fn lt(&self, other: &CValue) -> UBResult<CValue> {
        let result = match (self, other) {
            (CValue::Int(a), CValue::Int(b)) => a < b,
            (CValue::UInt(a), CValue::UInt(b)) => a < b,
            (CValue::Float(a), CValue::Float(b)) => a < b,
            (CValue::Double(a), CValue::Double(b)) => a < b,
            (CValue::Pointer(a), CValue::Pointer(b)) => {
                if a.block != b.block {
                    return Err(UBKind::InvalidPointerComparison);
                }
                a.offset < b.offset
            }
            (CValue::Undef, _) | (_, CValue::Undef) => {
                return Err(UBKind::UninitializedRead);
            }
            _ => return Err(UBKind::Other("invalid operands for comparison".to_string())),
        };
        Ok(CValue::Int(if result { 1 } else { 0 }))
    }

    /// Less than or equal comparison
    pub fn le(&self, other: &CValue) -> UBResult<CValue> {
        let lt = self.lt(other)?;
        let eq = self.eq(other)?;
        match (lt, eq) {
            (CValue::Int(1), _) | (_, CValue::Int(1)) => Ok(CValue::Int(1)),
            _ => Ok(CValue::Int(0)),
        }
    }

    /// Greater than comparison
    pub fn gt(&self, other: &CValue) -> UBResult<CValue> {
        other.lt(self)
    }

    /// Greater than or equal comparison
    pub fn ge(&self, other: &CValue) -> UBResult<CValue> {
        other.le(self)
    }
}

/// Truncate i128 to fit the specified type
fn truncate_to_type(val: i128, ty: &CType) -> CValue {
    match ty.unqualified() {
        CType::Int(kind, sign) => truncate_int(val, *kind, *sign),
        CType::Enum { .. } => truncate_int(val, IntKind::Int, Signedness::Signed),
        _ => CValue::Int(val),
    }
}

/// Check if adding two i128 values would overflow a signed type
fn check_signed_add_overflow(a: i128, b: i128, kind: IntKind) -> bool {
    let min = kind.signed_min();
    let max = kind.signed_max();
    // Check if result would be out of range for the target type
    if b > 0 && a > max - b {
        true
    } else {
        b < 0 && a < min - b
    }
}

/// Check if subtracting two i128 values would overflow a signed type
fn check_signed_sub_overflow(a: i128, b: i128, kind: IntKind) -> bool {
    let min = kind.signed_min();
    let max = kind.signed_max();
    if b < 0 && a > max + b {
        true
    } else {
        b > 0 && a < min + b
    }
}

/// Check if multiplying two i128 values would overflow a signed type
fn check_signed_mul_overflow(a: i128, b: i128, kind: IntKind) -> bool {
    if a == 0 || b == 0 {
        return false;
    }
    let min = kind.signed_min();
    let max = kind.signed_max();

    // Check all combinations of signs
    if a > 0 {
        if b > 0 {
            a > max / b
        } else {
            b < min / a
        }
    } else if b > 0 {
        a < min / b
    } else {
        // Both negative: result is positive
        // a * b = |a| * |b|, check if this overflows max
        let pos_a = -a;
        let pos_b = -b;
        pos_a > max / pos_b
    }
}

/// Truncate u128 to fit the specified type
fn truncate_to_type_unsigned(val: u128, ty: &CType) -> CValue {
    match ty.unqualified() {
        CType::Int(kind, _) => {
            let bits = kind.size() * 8;
            let mask = if bits >= 128 {
                u128::MAX
            } else {
                (1u128 << bits) - 1
            };
            CValue::UInt(val & mask)
        }
        _ => CValue::UInt(val),
    }
}

/// Convert i128 to i64 for pointer offset, returning error if out of range.
/// Pointer offsets that exceed i64 bounds cause undefined behavior in C.
pub fn to_pointer_offset(val: i128) -> UBResult<i64> {
    i64::try_from(val).map_err(|_| UBKind::PointerOverflow)
}

/// Convert u128 to i64 for pointer offset, returning error if out of range.
/// Pointer offsets that exceed i64 bounds cause undefined behavior in C.
pub fn to_pointer_offset_unsigned(val: u128) -> UBResult<i64> {
    if val > i64::MAX as u128 {
        Err(UBKind::PointerOverflow)
    } else {
        Ok(val as i64)
    }
}

/// Compute negative offset safely, checking for overflow when negating.
/// Returns error if offset is i64::MIN (negation would overflow).
pub fn negate_offset(offset: i64) -> UBResult<i64> {
    offset.checked_neg().ok_or(UBKind::PointerOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_values() {
        let int_ty = CType::int();
        let ptr_ty = CType::ptr(CType::int());
        let arr_ty = CType::array(CType::int(), 3);

        assert_eq!(CValue::zero(&int_ty), CValue::Int(0));
        assert!(matches!(CValue::zero(&ptr_ty), CValue::Pointer(p) if p.is_null()));
        assert!(matches!(CValue::zero(&arr_ty), CValue::Array(v) if v.len() == 3));
    }

    #[test]
    fn test_to_bool() {
        assert!(!CValue::Int(0).to_bool().unwrap());
        assert!(CValue::Int(1).to_bool().unwrap());
        assert!(CValue::Int(-1).to_bool().unwrap());
        assert!(!CValue::UInt(0).to_bool().unwrap());
        assert!(CValue::UInt(42).to_bool().unwrap());
        assert!(!CValue::Pointer(Pointer::null()).to_bool().unwrap());
        assert!(CValue::Undef.to_bool().is_err());
    }

    #[test]
    fn test_float_to_int_checks_overflow_and_nan() {
        let large = CValue::Float(f32::MAX);
        assert!(matches!(large.to_int(), Err(UBKind::FloatToIntOverflow)));

        let nan_val = CValue::Double(f64::NAN);
        assert!(matches!(nan_val.to_int(), Err(UBKind::FloatToIntOverflow)));

        let small = CValue::Float(12.75);
        assert_eq!(small.to_int().unwrap(), 12);
    }

    #[test]
    fn test_float_to_uint_rejects_invalid_values() {
        let negative = CValue::Double(-1.0);
        assert!(matches!(
            negative.to_uint(),
            Err(UBKind::FloatToIntOverflow)
        ));

        let huge = CValue::Double((u128::MAX as f64) * 2.0);
        assert!(matches!(huge.to_uint(), Err(UBKind::FloatToIntOverflow)));

        let ok = CValue::Double(42.9);
        assert_eq!(ok.to_uint().unwrap(), 42);
    }

    #[test]
    fn test_arithmetic() {
        let ty = CType::int();

        let a = CValue::Int(10);
        let b = CValue::Int(3);

        assert_eq!(a.add(&b, &ty).unwrap(), CValue::Int(13));
        assert_eq!(a.sub(&b, &ty).unwrap(), CValue::Int(7));
        assert_eq!(a.mul(&b, &ty).unwrap(), CValue::Int(30));
        assert_eq!(a.div(&b, &ty).unwrap(), CValue::Int(3));
        assert_eq!(a.rem(&b, &ty).unwrap(), CValue::Int(1));
    }

    #[test]
    fn test_division_by_zero() {
        let ty = CType::int();
        let a = CValue::Int(10);
        let zero = CValue::Int(0);

        assert!(matches!(a.div(&zero, &ty), Err(UBKind::DivisionByZero)));
        assert!(matches!(a.rem(&zero, &ty), Err(UBKind::DivisionByZero)));
    }

    #[test]
    fn test_signed_overflow() {
        let ty = CType::int();
        // Use i32::MAX since int is 32-bit in C
        let max = CValue::Int(i32::MAX as i128);
        let one = CValue::Int(1);

        assert!(matches!(max.add(&one, &ty), Err(UBKind::SignedOverflow)));
    }

    #[test]
    fn test_signed_sub_overflow() {
        let ty = CType::int();
        // INT_MIN - 1 should overflow
        let min = CValue::Int(i32::MIN as i128);
        let one = CValue::Int(1);

        assert!(matches!(min.sub(&one, &ty), Err(UBKind::SignedOverflow)));

        // INT_MAX - (-1) should overflow (equivalent to INT_MAX + 1)
        let max = CValue::Int(i32::MAX as i128);
        let neg_one = CValue::Int(-1);
        assert!(matches!(
            max.sub(&neg_one, &ty),
            Err(UBKind::SignedOverflow)
        ));
    }

    #[test]
    fn test_signed_mul_overflow() {
        let ty = CType::int();
        // INT_MAX * 2 should overflow
        let max = CValue::Int(i32::MAX as i128);
        let two = CValue::Int(2);

        assert!(matches!(max.mul(&two, &ty), Err(UBKind::SignedOverflow)));

        // INT_MIN * -1 should overflow (result is INT_MAX + 1)
        let min = CValue::Int(i32::MIN as i128);
        let neg_one = CValue::Int(-1);
        assert!(matches!(
            min.mul(&neg_one, &ty),
            Err(UBKind::SignedOverflow)
        ));
    }

    #[test]
    fn test_bitwise_ops() {
        let ty = CType::uint();
        let a = CValue::UInt(0b1100);
        let b = CValue::UInt(0b1010);

        assert_eq!(a.bit_and(&b, &ty).unwrap(), CValue::UInt(0b1000));
        assert_eq!(a.bit_or(&b, &ty).unwrap(), CValue::UInt(0b1110));
        assert_eq!(a.bit_xor(&b, &ty).unwrap(), CValue::UInt(0b0110));
    }

    #[test]
    fn test_shift_ops() {
        let ty = CType::uint();
        let a = CValue::UInt(1);
        let shift = CValue::Int(4);

        assert_eq!(a.shl(&shift, &ty).unwrap(), CValue::UInt(16));

        let b = CValue::UInt(16);
        assert_eq!(b.shr(&shift, &ty).unwrap(), CValue::UInt(1));
    }

    #[test]
    fn test_invalid_shift() {
        let ty = CType::Int(IntKind::Int, Signedness::Signed);
        let a = CValue::Int(1);
        let shift = CValue::Int(32); // >= 32 bits for int

        assert!(matches!(a.shl(&shift, &ty), Err(UBKind::InvalidShift(_))));
    }

    #[test]
    fn test_negative_shift_is_ub() {
        let ty = CType::Int(IntKind::Int, Signedness::Signed);
        let a = CValue::Int(1);

        // Negative shift amounts are undefined behavior in C
        let neg_shift = CValue::Int(-1);
        let result = a.shl(&neg_shift, &ty);
        assert!(matches!(result, Err(UBKind::InvalidShift(msg)) if msg.contains("negative")));

        let result = a.shr(&neg_shift, &ty);
        assert!(matches!(result, Err(UBKind::InvalidShift(msg)) if msg.contains("negative")));

        // Large negative value should also be caught
        let large_neg = CValue::Int(i128::MIN);
        let result = a.shl(&large_neg, &ty);
        assert!(matches!(result, Err(UBKind::InvalidShift(msg)) if msg.contains("negative")));
    }

    #[test]
    fn test_shift_amount_overflow() {
        let ty = CType::Int(IntKind::Int, Signedness::Signed);
        let a = CValue::Int(1);

        // Shift amount exceeding u32::MAX should return error, not silently truncate
        let huge_shift = CValue::UInt((u32::MAX as u128) + 1);
        let result = a.shl(&huge_shift, &ty);
        assert!(matches!(result, Err(UBKind::InvalidShift(msg)) if msg.contains("exceeds")));

        let result = a.shr(&huge_shift, &ty);
        assert!(matches!(result, Err(UBKind::InvalidShift(msg)) if msg.contains("exceeds")));
    }

    #[test]
    fn test_comparisons() {
        let a = CValue::Int(5);
        let b = CValue::Int(10);

        assert_eq!(a.lt(&b).unwrap(), CValue::Int(1));
        assert_eq!(a.le(&b).unwrap(), CValue::Int(1));
        assert_eq!(a.gt(&b).unwrap(), CValue::Int(0));
        assert_eq!(a.ge(&b).unwrap(), CValue::Int(0));
        assert_eq!(a.eq(&b).unwrap(), CValue::Int(0));
        assert_eq!(a.ne(&b).unwrap(), CValue::Int(1));
        assert_eq!(a.eq(&a).unwrap(), CValue::Int(1));
    }

    #[test]
    fn test_pointer_arithmetic() {
        let ty = CType::ptr(CType::int());
        let ptr = CValue::Pointer(Pointer::new(crate::memory::BlockId(1)));
        let offset = CValue::Int(10);

        let result = ptr.add(&offset, &ty).unwrap();
        if let CValue::Pointer(p) = result {
            assert_eq!(p.offset, 10);
        } else {
            panic!("Expected pointer result");
        }
    }

    #[test]
    fn test_cast() {
        let int_ty = CType::int();
        let uint_ty = CType::uint();
        let double_ty = CType::Float(crate::types::FloatKind::Double);

        // int -> uint
        let neg = CValue::Int(-1);
        let result = neg.cast(&int_ty, &uint_ty).unwrap();
        assert!(matches!(result, CValue::UInt(_)));

        // int -> double
        let int_val = CValue::Int(42);
        let result = int_val.cast(&int_ty, &double_ty).unwrap();
        assert!(matches!(result, CValue::Double(d) if (d - 42.0).abs() < 0.001));
    }

    #[test]
    fn test_float_to_int_cast_validates_range_and_nan() {
        let float_ty = CType::Float(crate::types::FloatKind::Double);
        let int_ty = CType::int();
        let uint_ty = CType::uint();

        let nan_val = CValue::Double(f64::NAN);
        assert!(matches!(
            nan_val.cast(&float_ty, &int_ty),
            Err(UBKind::FloatToIntOverflow)
        ));

        let negative = CValue::Double(-1.0);
        assert!(matches!(
            negative.cast(&float_ty, &uint_ty),
            Err(UBKind::FloatToIntOverflow)
        ));

        let too_big = CValue::Double((crate::types::IntKind::Int.unsigned_max() as f64) + 1024.0);
        assert!(matches!(
            too_big.cast(&float_ty, &uint_ty),
            Err(UBKind::FloatToIntOverflow)
        ));
    }

    #[test]
    fn test_pointer_arithmetic_offset_overflow() {
        let ty = CType::ptr(CType::int());
        let ptr = CValue::Pointer(Pointer::new(crate::memory::BlockId(1)));

        // Offset exceeding i64::MAX should be detected
        let huge_offset = CValue::Int((i64::MAX as i128) + 1);
        let result = ptr.add(&huge_offset, &ty);
        assert!(matches!(result, Err(UBKind::PointerOverflow)));

        // Unsigned offset exceeding i64::MAX should also be detected
        let huge_unsigned = CValue::UInt((i64::MAX as u128) + 1);
        let result = ptr.add(&huge_unsigned, &ty);
        assert!(matches!(result, Err(UBKind::PointerOverflow)));

        // Negative offset more negative than i64::MIN
        let very_negative = CValue::Int((i64::MIN as i128) - 1);
        let result = ptr.add(&very_negative, &ty);
        assert!(matches!(result, Err(UBKind::PointerOverflow)));
    }

    #[test]
    fn test_pointer_subtraction_offset_overflow() {
        let ty = CType::ptr(CType::int());
        let ptr = CValue::Pointer(Pointer::new(crate::memory::BlockId(1)));

        // Subtracting a huge positive should detect overflow during negation
        let huge_offset = CValue::UInt((i64::MAX as u128) + 1);
        let result = ptr.sub(&huge_offset, &ty);
        assert!(matches!(result, Err(UBKind::PointerOverflow)));

        // Subtracting i64::MIN as i128 should detect negation overflow
        // (because -i64::MIN doesn't fit in i64)
        let min_offset = CValue::Int(i64::MIN as i128);
        let result = ptr.sub(&min_offset, &ty);
        assert!(matches!(result, Err(UBKind::PointerOverflow)));
    }
}
