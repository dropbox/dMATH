//! Rust Value Representation
//!
//! This module defines how Rust values are represented in the semantic model.
//! Values are the runtime representations of data.
//!
//! ## Value Categories
//!
//! - **Scalars**: Integers, floats, booleans, chars
//! - **Aggregates**: Tuples, arrays, structs
//! - **References**: Pointers with provenance
//! - **Functions**: Function pointers and closures

use crate::memory::Address;
use crate::types::{FloatType, IntType, Lifetime, Mutability, RustType, UintType};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A Rust value
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    /// Unit value ()
    Unit,

    /// Boolean
    Bool(bool),

    /// Character (Unicode scalar value)
    Char(char),

    /// Unsigned integer with size
    Uint { value: u128, ty: UintType },

    /// Signed integer with size
    Int { value: i128, ty: IntType },

    /// Floating point
    Float {
        /// Stored as bits to preserve NaN payloads
        bits: u64,
        ty: FloatType,
    },

    /// Reference (pointer with provenance)
    Reference {
        addr: Address,
        mutability: Mutability,
        lifetime: Lifetime,
    },

    /// Raw pointer (no provenance tracking)
    RawPtr {
        addr: Address,
        mutability: Mutability,
    },

    /// Tuple of values
    Tuple(Vec<Value>),

    /// Array of values (fixed size)
    Array(Vec<Value>),

    /// Struct value
    Struct {
        name: String,
        fields: BTreeMap<String, Value>,
    },

    /// Enum variant
    Enum {
        name: String,
        variant: String,
        payload: Box<EnumPayload>,
    },

    /// Function pointer
    FnPtr { name: String },

    /// Closure (captured environment)
    Closure {
        fn_id: String,
        captures: Vec<(String, Value)>,
    },

    /// The "never" value (unreachable)
    Never,

    /// Uninitialized memory (poison)
    Uninit,
}

/// Enum variant payload
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EnumPayload {
    /// Unit variant: Foo
    Unit,
    /// Tuple variant: Foo(x, y)
    Tuple(Vec<Value>),
    /// Struct variant: Foo { a, b }
    Struct(BTreeMap<String, Value>),
}

impl Value {
    /// Create a u8 value
    pub fn u8(v: u8) -> Self {
        Value::Uint {
            value: v as u128,
            ty: UintType::U8,
        }
    }

    /// Create a u16 value
    pub fn u16(v: u16) -> Self {
        Value::Uint {
            value: v as u128,
            ty: UintType::U16,
        }
    }

    /// Create a u32 value
    pub fn u32(v: u32) -> Self {
        Value::Uint {
            value: v as u128,
            ty: UintType::U32,
        }
    }

    /// Create a u64 value
    pub fn u64(v: u64) -> Self {
        Value::Uint {
            value: v as u128,
            ty: UintType::U64,
        }
    }

    /// Create a usize value
    pub fn usize(v: usize) -> Self {
        Value::Uint {
            value: v as u128,
            ty: UintType::Usize,
        }
    }

    /// Create an i32 value
    pub fn i32(v: i32) -> Self {
        Value::Int {
            value: v as i128,
            ty: IntType::I32,
        }
    }

    /// Create an i64 value
    pub fn i64(v: i64) -> Self {
        Value::Int {
            value: v as i128,
            ty: IntType::I64,
        }
    }

    /// Create an f64 value
    pub fn f64(v: f64) -> Self {
        Value::Float {
            bits: v.to_bits(),
            ty: FloatType::F64,
        }
    }

    /// Create an f32 value
    pub fn f32(v: f32) -> Self {
        Value::Float {
            bits: u64::from(v.to_bits()),
            ty: FloatType::F32,
        }
    }

    /// Check if value is uninitialized
    pub fn is_uninit(&self) -> bool {
        matches!(self, Value::Uninit)
    }

    /// Check if value is zero/default
    pub fn is_zero(&self) -> bool {
        matches!(
            self,
            Value::Bool(false)
                | Value::Uint { value: 0, .. }
                | Value::Int { value: 0, .. }
                | Value::Float { bits: 0, .. }
        )
    }

    /// Get the type of this value
    pub fn get_type(&self) -> RustType {
        match self {
            Value::Bool(_) => RustType::Bool,
            Value::Char(_) => RustType::Char,
            Value::Uint { ty, .. } => RustType::Uint(*ty),
            Value::Int { ty, .. } => RustType::Int(*ty),
            Value::Float { ty, .. } => RustType::Float(*ty),
            Value::Reference {
                mutability,
                lifetime,
                ..
            } => RustType::Reference {
                lifetime: lifetime.clone(),
                mutability: *mutability,
                inner: Box::new(RustType::Unit), // Type would need to be tracked
            },
            Value::RawPtr { mutability, .. } => RustType::RawPtr {
                mutability: *mutability,
                inner: Box::new(RustType::Unit),
            },
            Value::Tuple(elems) => RustType::Tuple(elems.iter().map(Value::get_type).collect()),
            Value::Array(elems) => {
                let elem_ty = elems.first().map_or(RustType::Unit, Value::get_type);
                RustType::Array {
                    element: Box::new(elem_ty),
                    len: elems.len(),
                }
            }
            Value::Struct { name, .. } | Value::Enum { name, .. } => RustType::Named {
                name: name.clone(),
                type_args: vec![],
                lifetime_args: vec![],
            },
            Value::FnPtr { .. } => RustType::Function {
                params: vec![],
                ret: Box::new(RustType::Unit),
            },
            Value::Closure { .. } => RustType::Closure {
                params: vec![],
                ret: Box::new(RustType::Unit),
                captures: vec![],
            },
            Value::Never => RustType::Never,
            // Unit and Uninit both map to Unit type
            Value::Unit | Value::Uninit => RustType::Unit,
        }
    }

    /// Try to convert to bool
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Try to convert to u64, returning None if the value doesn't fit
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Uint { value, .. } => u64::try_from(*value).ok(),
            _ => None,
        }
    }

    /// Try to convert to i64, returning None if the value doesn't fit
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int { value, .. } => i64::try_from(*value).ok(),
            _ => None,
        }
    }

    /// Try to convert to f64
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Float { bits, ty } => match ty {
                FloatType::F64 => Some(f64::from_bits(*bits)),
                // SAFETY: For F32, bits stores only 32 significant bits; truncation is intentional
                #[allow(clippy::cast_possible_truncation)]
                FloatType::F32 => Some(f32::from_bits(*bits as u32) as f64),
            },
            _ => None,
        }
    }
}

/// Binary arithmetic operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BinOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Rem,

    // Bitwise
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,

    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Unary operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum UnOp {
    /// Negation (-)
    Neg,
    /// Bitwise not (!)
    Not,
}

/// Evaluate a binary operation
pub fn eval_binop(op: BinOp, left: &Value, right: &Value) -> Option<Value> {
    match (left, right) {
        // Integer operations
        (Value::Uint { value: l, ty: ty_l }, Value::Uint { value: r, ty: ty_r })
            if ty_l == ty_r =>
        {
            // Mask for the type's bit width
            let mask: u128 = match ty_l {
                UintType::U8 => 0xFF,
                UintType::U16 => 0xFFFF,
                UintType::U32 => 0xFFFF_FFFF,
                UintType::U64 | UintType::Usize => 0xFFFF_FFFF_FFFF_FFFF,
                UintType::U128 => u128::MAX,
            };
            let result = match op {
                BinOp::Add => l.wrapping_add(*r) & mask,
                BinOp::Sub => l.wrapping_sub(*r) & mask,
                BinOp::Mul => l.wrapping_mul(*r) & mask,
                BinOp::Div => l.checked_div(*r)?,
                BinOp::Rem => l.checked_rem(*r)?,
                BinOp::BitAnd => l & r,
                BinOp::BitOr => l | r,
                BinOp::BitXor => l ^ r,
                BinOp::Shl => (l << (r & 0x7F)) & mask,
                BinOp::Shr => l >> (r & 0x7F),
                BinOp::Eq => return Some(Value::Bool(l == r)),
                BinOp::Ne => return Some(Value::Bool(l != r)),
                BinOp::Lt => return Some(Value::Bool(l < r)),
                BinOp::Le => return Some(Value::Bool(l <= r)),
                BinOp::Gt => return Some(Value::Bool(l > r)),
                BinOp::Ge => return Some(Value::Bool(l >= r)),
            };
            Some(Value::Uint {
                value: result,
                ty: *ty_l,
            })
        }

        // Signed integer operations
        (Value::Int { value: l, ty: ty_l }, Value::Int { value: r, ty: ty_r }) if ty_l == ty_r => {
            let result = match op {
                BinOp::Add => l.wrapping_add(*r),
                BinOp::Sub => l.wrapping_sub(*r),
                BinOp::Mul => l.wrapping_mul(*r),
                BinOp::Div => l.checked_div(*r)?,
                BinOp::Rem => l.checked_rem(*r)?,
                BinOp::BitAnd => l & r,
                BinOp::BitOr => l | r,
                BinOp::BitXor => l ^ r,
                BinOp::Shl => l << (r & 0x7F),
                BinOp::Shr => l >> (r & 0x7F),
                BinOp::Eq => return Some(Value::Bool(l == r)),
                BinOp::Ne => return Some(Value::Bool(l != r)),
                BinOp::Lt => return Some(Value::Bool(l < r)),
                BinOp::Le => return Some(Value::Bool(l <= r)),
                BinOp::Gt => return Some(Value::Bool(l > r)),
                BinOp::Ge => return Some(Value::Bool(l >= r)),
            };
            Some(Value::Int {
                value: result,
                ty: *ty_l,
            })
        }

        // Boolean operations
        (Value::Bool(l), Value::Bool(r)) => match op {
            BinOp::BitAnd => Some(Value::Bool(*l && *r)),
            BinOp::BitOr => Some(Value::Bool(*l || *r)),
            BinOp::BitXor => Some(Value::Bool(*l ^ *r)),
            BinOp::Eq => Some(Value::Bool(l == r)),
            BinOp::Ne => Some(Value::Bool(l != r)),
            _ => None,
        },

        // Float operations
        (Value::Float { bits: l, ty: ty_l }, Value::Float { bits: r, ty: ty_r })
            if ty_l == ty_r =>
        {
            let lf = f64::from_bits(*l);
            let rf = f64::from_bits(*r);
            let result = match op {
                BinOp::Add => Value::Float {
                    bits: (lf + rf).to_bits(),
                    ty: *ty_l,
                },
                BinOp::Sub => Value::Float {
                    bits: (lf - rf).to_bits(),
                    ty: *ty_l,
                },
                BinOp::Mul => Value::Float {
                    bits: (lf * rf).to_bits(),
                    ty: *ty_l,
                },
                BinOp::Div => Value::Float {
                    bits: (lf / rf).to_bits(),
                    ty: *ty_l,
                },
                BinOp::Rem => Value::Float {
                    bits: (lf % rf).to_bits(),
                    ty: *ty_l,
                },
                BinOp::Eq => Value::Bool(lf == rf),
                BinOp::Ne => Value::Bool(lf != rf),
                BinOp::Lt => Value::Bool(lf < rf),
                BinOp::Le => Value::Bool(lf <= rf),
                BinOp::Gt => Value::Bool(lf > rf),
                BinOp::Ge => Value::Bool(lf >= rf),
                _ => return None,
            };
            Some(result)
        }

        _ => None,
    }
}

/// Evaluate a unary operation
pub fn eval_unop(op: UnOp, val: &Value) -> Option<Value> {
    match (op, val) {
        (UnOp::Not, Value::Bool(b)) => Some(Value::Bool(!b)),
        (UnOp::Not, Value::Uint { value, ty }) => Some(Value::Uint {
            value: !value,
            ty: *ty,
        }),
        (UnOp::Neg, Value::Int { value, ty }) => Some(Value::Int {
            value: value.wrapping_neg(),
            ty: *ty,
        }),
        (UnOp::Neg, Value::Float { bits, ty }) => {
            let f = f64::from_bits(*bits);
            Some(Value::Float {
                bits: (-f).to_bits(),
                ty: *ty,
            })
        }
        _ => None,
    }
}

/// Type cast operations
pub fn cast_value(val: &Value, target: &RustType) -> Option<Value> {
    match (val, target) {
        // Bool to int
        (Value::Bool(b), RustType::Uint(ty)) => Some(Value::Uint {
            value: u128::from(*b),
            ty: *ty,
        }),
        (Value::Bool(b), RustType::Int(ty)) => Some(Value::Int {
            value: i128::from(*b),
            ty: *ty,
        }),

        // Int to int (truncation/extension)
        (Value::Uint { value, .. }, RustType::Uint(ty)) => {
            let mask = match ty {
                UintType::U8 => 0xFF,
                UintType::U16 => 0xFFFF,
                UintType::U32 => 0xFFFF_FFFF,
                UintType::U64 | UintType::Usize => 0xFFFF_FFFF_FFFF_FFFF,
                UintType::U128 => u128::MAX,
            };
            Some(Value::Uint {
                value: value & mask,
                ty: *ty,
            })
        }

        (Value::Uint { value, .. }, RustType::Int(ty)) => Some(Value::Int {
            value: *value as i128,
            ty: *ty,
        }),

        (Value::Int { value, .. }, RustType::Uint(ty)) => Some(Value::Uint {
            value: *value as u128,
            ty: *ty,
        }),

        (Value::Int { value, .. }, RustType::Int(ty)) => Some(Value::Int {
            value: *value,
            ty: *ty,
        }),

        // Int to float
        (Value::Uint { value, .. }, RustType::Float(ty)) => Some(Value::Float {
            bits: (*value as f64).to_bits(),
            ty: *ty,
        }),

        (Value::Int { value, .. }, RustType::Float(ty)) => Some(Value::Float {
            bits: (*value as f64).to_bits(),
            ty: *ty,
        }),

        // Float to int
        (Value::Float { bits, .. }, RustType::Uint(ty)) => {
            let f = f64::from_bits(*bits);
            Some(Value::Uint {
                value: f as u128,
                ty: *ty,
            })
        }

        (Value::Float { bits, .. }, RustType::Int(ty)) => {
            let f = f64::from_bits(*bits);
            Some(Value::Int {
                value: f as i128,
                ty: *ty,
            })
        }

        // Pointer casts
        (Value::Reference { addr, .. }, RustType::RawPtr { mutability, .. }) => {
            Some(Value::RawPtr {
                addr: *addr,
                mutability: *mutability,
            })
        }

        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_constructors() {
        assert_eq!(
            Value::u32(42),
            Value::Uint {
                value: 42,
                ty: UintType::U32,
            }
        );

        assert_eq!(
            Value::i64(-1),
            Value::Int {
                value: -1,
                ty: IntType::I64,
            }
        );
    }

    #[test]
    fn test_binop_add() {
        let left = Value::u32(10);
        let right = Value::u32(20);

        let result = eval_binop(BinOp::Add, &left, &right);
        assert_eq!(result, Some(Value::u32(30)));
    }

    #[test]
    fn test_binop_compare() {
        let left = Value::i32(5);
        let right = Value::i32(10);

        assert_eq!(
            eval_binop(BinOp::Lt, &left, &right),
            Some(Value::Bool(true))
        );
        assert_eq!(
            eval_binop(BinOp::Gt, &left, &right),
            Some(Value::Bool(false))
        );
        assert_eq!(
            eval_binop(BinOp::Eq, &left, &right),
            Some(Value::Bool(false))
        );
    }

    #[test]
    fn test_unop_neg() {
        let val = Value::i32(42);
        let result = eval_unop(UnOp::Neg, &val);
        assert_eq!(result, Some(Value::i32(-42)));
    }

    #[test]
    fn test_unop_not() {
        let val = Value::Bool(true);
        let result = eval_unop(UnOp::Not, &val);
        assert_eq!(result, Some(Value::Bool(false)));
    }

    #[test]
    fn test_cast_int_to_int() {
        let val = Value::u32(256);
        let result = cast_value(&val, &RustType::Uint(UintType::U8));
        assert_eq!(result, Some(Value::u8(0))); // Truncation
    }

    #[test]
    fn test_cast_bool_to_int() {
        let val = Value::Bool(true);
        let result = cast_value(&val, &RustType::Uint(UintType::U32));
        assert_eq!(result, Some(Value::u32(1)));
    }

    #[test]
    fn test_overflow_wrapping() {
        let left = Value::u8(255);
        let right = Value::u8(1);

        let result = eval_binop(BinOp::Add, &left, &right);
        // Should wrap to 0
        assert_eq!(result, Some(Value::u8(0)));
    }

    #[test]
    fn test_division_by_zero() {
        let left = Value::u32(10);
        let right = Value::u32(0);

        let result = eval_binop(BinOp::Div, &left, &right);
        assert_eq!(result, None); // Division by zero returns None
    }

    #[test]
    fn test_float_operations() {
        let left = Value::f64(3.0);
        let right = Value::f64(2.0);

        let result = eval_binop(BinOp::Add, &left, &right);
        assert!(result.is_some());

        let result_f64 = result.unwrap().as_f64().unwrap();
        assert!((result_f64 - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_struct_value() {
        let mut fields = BTreeMap::new();
        fields.insert("x".to_string(), Value::f64(1.0));
        fields.insert("y".to_string(), Value::f64(2.0));

        let point = Value::Struct {
            name: "Point".to_string(),
            fields,
        };

        match point.get_type() {
            RustType::Named { name, .. } => assert_eq!(name, "Point"),
            _ => panic!("Expected named type"),
        }
    }

    #[test]
    fn test_enum_value() {
        let some_val = Value::Enum {
            name: "Option".to_string(),
            variant: "Some".to_string(),
            payload: Box::new(EnumPayload::Tuple(vec![Value::u32(42)])),
        };

        let none_val = Value::Enum {
            name: "Option".to_string(),
            variant: "None".to_string(),
            payload: Box::new(EnumPayload::Unit),
        };

        assert!(matches!(some_val.get_type(), RustType::Named { .. }));
        assert!(matches!(none_val.get_type(), RustType::Named { .. }));
    }
}
