//! C Type System Formalization
//!
//! This module defines the C type system following the C11 standard.
//! Types are fundamental for memory layout, alignment, and value representation.
//!
//! ## Type Categories
//!
//! 1. **Integer Types**: char, short, int, long, long long (signed/unsigned)
//! 2. **Floating Types**: float, double, long double
//! 3. **Pointer Types**: T* for any complete type T
//! 4. **Array Types**: `T[N]` for fixed-size arrays
//! 5. **Struct Types**: struct { ... }
//! 6. **Union Types**: union { ... }
//! 7. **Enum Types**: enum { ... }
//! 8. **Function Types**: T(T1, T2, ...) -> T
//! 9. **Void**: void (incomplete type)
//!
//! ## Data Model
//!
//! We use the LP64 data model (common on 64-bit Unix):
//! - char: 1 byte
//! - short: 2 bytes
//! - int: 4 bytes
//! - long: 8 bytes
//! - long long: 8 bytes
//! - pointer: 8 bytes
//!
//! ## Alignment
//!
//! Alignment follows C11 rules:
//! - Natural alignment for primitives
//! - Struct alignment is max of member alignments
//! - Arrays have element alignment

use serde::{Deserialize, Serialize};

/// Integer kinds in C
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IntKind {
    /// char (1 byte)
    Char,
    /// short (2 bytes)
    Short,
    /// int (4 bytes)
    Int,
    /// long (8 bytes on LP64)
    Long,
    /// long long (8 bytes)
    LongLong,
    /// _Bool / bool (1 byte, C99+)
    Bool,
}

impl IntKind {
    /// Size in bytes (LP64 model)
    pub fn size(&self) -> usize {
        match self {
            IntKind::Bool | IntKind::Char => 1,
            IntKind::Short => 2,
            IntKind::Int => 4,
            IntKind::Long | IntKind::LongLong => 8,
        }
    }

    /// Alignment in bytes
    pub fn align(&self) -> usize {
        self.size()
    }

    /// Minimum value for signed variant
    pub fn signed_min(&self) -> i128 {
        match self {
            IntKind::Bool => 0,
            IntKind::Char => i8::MIN as i128,
            IntKind::Short => i16::MIN as i128,
            IntKind::Int => i32::MIN as i128,
            IntKind::Long | IntKind::LongLong => i64::MIN as i128,
        }
    }

    /// Maximum value for signed variant
    pub fn signed_max(&self) -> i128 {
        match self {
            IntKind::Bool => 1,
            IntKind::Char => i8::MAX as i128,
            IntKind::Short => i16::MAX as i128,
            IntKind::Int => i32::MAX as i128,
            IntKind::Long | IntKind::LongLong => i64::MAX as i128,
        }
    }

    /// Maximum value for unsigned variant
    pub fn unsigned_max(&self) -> u128 {
        match self {
            IntKind::Bool => 1,
            IntKind::Char => u8::MAX as u128,
            IntKind::Short => u16::MAX as u128,
            IntKind::Int => u32::MAX as u128,
            IntKind::Long | IntKind::LongLong => u64::MAX as u128,
        }
    }
}

/// Floating-point kinds in C
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FloatKind {
    /// float (4 bytes)
    Float,
    /// double (8 bytes)
    Double,
    /// long double (16 bytes on most platforms)
    LongDouble,
}

impl FloatKind {
    /// Size in bytes
    pub fn size(&self) -> usize {
        match self {
            FloatKind::Float => 4,
            FloatKind::Double => 8,
            FloatKind::LongDouble => 16,
        }
    }

    /// Alignment in bytes
    pub fn align(&self) -> usize {
        self.size().min(16) // Max alignment is typically 16
    }
}

/// Signedness of integer types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Signedness {
    Signed,
    Unsigned,
}

/// A field in a struct or union
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StructField {
    pub name: String,
    pub ty: CType,
}

/// A function parameter
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FuncParam {
    pub name: Option<String>,
    pub ty: CType,
}

/// C Type representation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CType {
    /// void (incomplete type)
    Void,

    /// Integer types: char, short, int, long, long long, _Bool
    Int(IntKind, Signedness),

    /// Floating-point types: float, double, long double
    Float(FloatKind),

    /// Pointer to another type: T*
    Pointer(Box<CType>),

    /// Fixed-size array: `T[N]`
    Array(Box<CType>, usize),

    /// Struct type: struct { ... }
    Struct {
        name: Option<String>,
        fields: Vec<StructField>,
    },

    /// Union type: union { ... }
    Union {
        name: Option<String>,
        fields: Vec<StructField>,
    },

    /// Enum type: enum { ... }
    Enum {
        name: Option<String>,
        /// (name, value)
        variants: Vec<(String, i64)>,
    },

    /// Function type: T(T1, T2, ...) -> RetT
    Function {
        return_type: Box<CType>,
        params: Vec<FuncParam>,
        variadic: bool,
    },

    /// typedef reference (resolved during type checking)
    TypeDef(String),

    /// Qualified type (const, volatile, restrict)
    Qualified {
        ty: Box<CType>,
        is_const: bool,
        is_volatile: bool,
        is_restrict: bool,
    },
}

impl CType {
    /// Create a void type
    pub fn void() -> Self {
        CType::Void
    }

    /// Create a signed int type
    pub fn int() -> Self {
        CType::Int(IntKind::Int, Signedness::Signed)
    }

    /// Create an unsigned int type
    pub fn uint() -> Self {
        CType::Int(IntKind::Int, Signedness::Unsigned)
    }

    /// Create a signed char type
    pub fn char() -> Self {
        CType::Int(IntKind::Char, Signedness::Signed)
    }

    /// Create an unsigned char type
    pub fn unsigned_char() -> Self {
        CType::Int(IntKind::Char, Signedness::Unsigned)
    }

    /// Create a size_t type (unsigned long on LP64)
    pub fn size_t() -> Self {
        CType::Int(IntKind::Long, Signedness::Unsigned)
    }

    /// Create a pointer type
    pub fn ptr(inner: CType) -> Self {
        CType::Pointer(Box::new(inner))
    }

    /// Create an array type
    pub fn array(elem: CType, size: usize) -> Self {
        CType::Array(Box::new(elem), size)
    }

    /// Create a const-qualified type
    pub fn const_ty(ty: CType) -> Self {
        CType::Qualified {
            ty: Box::new(ty),
            is_const: true,
            is_volatile: false,
            is_restrict: false,
        }
    }

    /// Size in bytes
    ///
    /// Returns None for incomplete types (void, unsized arrays, forward decls)
    pub fn size(&self) -> usize {
        match self {
            // Incomplete types and function types have no size
            CType::Void | CType::Function { .. } => 0,

            CType::Int(kind, _) => kind.size(),

            CType::Float(kind) => kind.size(),

            CType::Pointer(_) => 8, // LP64: all pointers are 8 bytes

            CType::Array(elem, count) => elem.size() * count,

            CType::Struct { fields, .. } => {
                // Calculate size with padding for alignment
                let mut offset = 0usize;
                for field in fields {
                    let field_align = field.ty.align();
                    // Add padding for alignment
                    let padding = (field_align - (offset % field_align)) % field_align;
                    offset += padding;
                    offset += field.ty.size();
                }
                // Add trailing padding to align to struct's overall alignment
                let struct_align = self.align();
                let trailing_padding = (struct_align - (offset % struct_align)) % struct_align;
                offset + trailing_padding
            }

            CType::Union { fields, .. } => {
                // Union size is max of all field sizes
                fields.iter().map(|f| f.ty.size()).max().unwrap_or(0)
            }

            CType::Enum { .. } => 4, // Enums are int-sized by default

            CType::TypeDef(_) => panic!("typedef should be resolved before size calculation"),

            CType::Qualified { ty, .. } => ty.size(),
        }
    }

    /// Alignment in bytes
    pub fn align(&self) -> usize {
        match self {
            CType::Void | CType::Function { .. } => 1,

            CType::Int(kind, _) => kind.align(),

            CType::Float(kind) => kind.align(),

            CType::Pointer(_) => 8,

            CType::Array(elem, _) => elem.align(),

            CType::Struct { fields, .. } => {
                // Struct alignment is max of all field alignments
                fields.iter().map(|f| f.ty.align()).max().unwrap_or(1)
            }

            CType::Union { fields, .. } => {
                // Union alignment is max of all field alignments
                fields.iter().map(|f| f.ty.align()).max().unwrap_or(1)
            }

            CType::Enum { .. } => 4,

            CType::TypeDef(_) => panic!("typedef should be resolved before alignment calculation"),

            CType::Qualified { ty, .. } => ty.align(),
        }
    }

    /// Check if type is complete (has known size)
    pub fn is_complete(&self) -> bool {
        match self {
            CType::Void | CType::Function { .. } | CType::TypeDef(_) => false,
            CType::Qualified { ty, .. } => ty.is_complete(),
            CType::Array(elem, _) => elem.is_complete(),
            _ => true,
        }
    }

    /// Check if type is an integer type
    pub fn is_integer(&self) -> bool {
        match self {
            CType::Int(_, _) | CType::Enum { .. } => true,
            CType::Qualified { ty, .. } => ty.is_integer(),
            _ => false,
        }
    }

    /// Check if type is a floating-point type
    pub fn is_float(&self) -> bool {
        match self {
            CType::Float(_) => true,
            CType::Qualified { ty, .. } => ty.is_float(),
            _ => false,
        }
    }

    /// Check if type is an arithmetic type (integer or float)
    pub fn is_arithmetic(&self) -> bool {
        self.is_integer() || self.is_float()
    }

    /// Check if type is a scalar type (arithmetic or pointer)
    pub fn is_scalar(&self) -> bool {
        self.is_arithmetic() || self.is_pointer()
    }

    /// Check if type is a pointer type
    pub fn is_pointer(&self) -> bool {
        match self {
            CType::Pointer(_) => true,
            CType::Qualified { ty, .. } => ty.is_pointer(),
            _ => false,
        }
    }

    /// Check if type is an array type
    pub fn is_array(&self) -> bool {
        match self {
            CType::Array(_, _) => true,
            CType::Qualified { ty, .. } => ty.is_array(),
            _ => false,
        }
    }

    /// Check if type is a struct type
    pub fn is_struct(&self) -> bool {
        matches!(self, CType::Struct { .. })
    }

    /// Check if type is a union type
    pub fn is_union(&self) -> bool {
        matches!(self, CType::Union { .. })
    }

    /// Check if type is a function type
    pub fn is_function(&self) -> bool {
        matches!(self, CType::Function { .. })
    }

    /// Get the pointee type if this is a pointer
    pub fn pointee(&self) -> Option<&CType> {
        match self {
            CType::Pointer(inner) => Some(inner),
            CType::Qualified { ty, .. } => ty.pointee(),
            _ => None,
        }
    }

    /// Get the element type if this is an array
    pub fn element(&self) -> Option<&CType> {
        match self {
            CType::Array(elem, _) => Some(elem),
            CType::Qualified { ty, .. } => ty.element(),
            _ => None,
        }
    }

    /// Get struct field by name
    pub fn get_field(&self, name: &str) -> Option<(usize, &StructField)> {
        match self {
            CType::Struct { fields, .. } | CType::Union { fields, .. } => {
                fields.iter().enumerate().find(|(_, f)| f.name == name)
            }
            CType::Qualified { ty, .. } => ty.get_field(name),
            _ => None,
        }
    }

    /// Get field offset in a struct
    pub fn field_offset(&self, name: &str) -> Option<usize> {
        match self {
            CType::Struct { fields, .. } => {
                let mut offset = 0usize;
                for field in fields {
                    let field_align = field.ty.align();
                    let padding = (field_align - (offset % field_align)) % field_align;
                    offset += padding;

                    if field.name == name {
                        return Some(offset);
                    }
                    offset += field.ty.size();
                }
                None
            }
            CType::Union { fields, .. } => {
                // All union fields are at offset 0
                if fields.iter().any(|f| f.name == name) {
                    Some(0)
                } else {
                    None
                }
            }
            CType::Qualified { ty, .. } => ty.field_offset(name),
            _ => None,
        }
    }

    /// Remove qualifiers from type
    pub fn unqualified(&self) -> &CType {
        match self {
            CType::Qualified { ty, .. } => ty.unqualified(),
            _ => self,
        }
    }

    /// Check if types are compatible (C11 6.2.7)
    pub fn is_compatible(&self, other: &CType) -> bool {
        match (self.unqualified(), other.unqualified()) {
            (CType::Void, CType::Void) => true,

            (CType::Int(k1, s1), CType::Int(k2, s2)) => k1 == k2 && s1 == s2,

            (CType::Float(k1), CType::Float(k2)) => k1 == k2,

            (CType::Pointer(p1), CType::Pointer(p2)) => p1.is_compatible(p2),

            (CType::Array(e1, n1), CType::Array(e2, n2)) => n1 == n2 && e1.is_compatible(e2),

            (
                CType::Struct {
                    name: n1,
                    fields: f1,
                },
                CType::Struct {
                    name: n2,
                    fields: f2,
                },
            ) => {
                if n1 != n2 {
                    return false;
                }
                if f1.len() != f2.len() {
                    return false;
                }
                f1.iter()
                    .zip(f2.iter())
                    .all(|(a, b)| a.name == b.name && a.ty.is_compatible(&b.ty))
            }

            (
                CType::Union {
                    name: n1,
                    fields: f1,
                },
                CType::Union {
                    name: n2,
                    fields: f2,
                },
            ) => {
                if n1 != n2 {
                    return false;
                }
                if f1.len() != f2.len() {
                    return false;
                }
                f1.iter()
                    .zip(f2.iter())
                    .all(|(a, b)| a.name == b.name && a.ty.is_compatible(&b.ty))
            }

            (
                CType::Enum {
                    name: n1,
                    variants: v1,
                },
                CType::Enum {
                    name: n2,
                    variants: v2,
                },
            ) => n1 == n2 && v1 == v2,

            (
                CType::Function {
                    return_type: r1,
                    params: p1,
                    variadic: v1,
                },
                CType::Function {
                    return_type: r2,
                    params: p2,
                    variadic: v2,
                },
            ) => {
                if v1 != v2 {
                    return false;
                }
                if !r1.is_compatible(r2) {
                    return false;
                }
                if p1.len() != p2.len() {
                    return false;
                }
                p1.iter()
                    .zip(p2.iter())
                    .all(|(a, b)| a.ty.is_compatible(&b.ty))
            }

            _ => false,
        }
    }

    /// Integer promotion (C11 6.3.1.1)
    ///
    /// Small integer types are promoted to int or unsigned int
    #[must_use]
    pub fn integer_promotion(&self) -> CType {
        match self {
            CType::Int(IntKind::Bool | IntKind::Char | IntKind::Short, _) => {
                CType::Int(IntKind::Int, Signedness::Signed)
            }

            CType::Int(IntKind::Int, Signedness::Unsigned) => {
                CType::Int(IntKind::Int, Signedness::Unsigned)
            }

            ty => ty.clone(),
        }
    }

    /// Usual arithmetic conversions (C11 6.3.1.8)
    ///
    /// Returns the common type for arithmetic operations
    #[must_use]
    pub fn usual_arithmetic_conversion(&self, other: &CType) -> CType {
        let a = self.integer_promotion();
        let b = other.integer_promotion();

        // If either is long double
        if matches!(a, CType::Float(FloatKind::LongDouble))
            || matches!(b, CType::Float(FloatKind::LongDouble))
        {
            return CType::Float(FloatKind::LongDouble);
        }

        // If either is double
        if matches!(a, CType::Float(FloatKind::Double))
            || matches!(b, CType::Float(FloatKind::Double))
        {
            return CType::Float(FloatKind::Double);
        }

        // If either is float
        if matches!(a, CType::Float(FloatKind::Float))
            || matches!(b, CType::Float(FloatKind::Float))
        {
            return CType::Float(FloatKind::Float);
        }

        // Both are integers after promotion
        match (&a, &b) {
            (CType::Int(k1, s1), CType::Int(k2, s2)) => {
                // If same signedness, use larger rank
                if s1 == s2 {
                    if k1.size() >= k2.size() {
                        return a;
                    }
                    return b;
                }

                // Different signedness: complex rules
                let (unsigned, signed, unsigned_kind, signed_kind) = if *s1 == Signedness::Unsigned
                {
                    (&a, &b, k1, k2)
                } else {
                    (&b, &a, k2, k1)
                };

                // If unsigned has rank >= signed, use unsigned
                if unsigned_kind.size() >= signed_kind.size() {
                    return unsigned.clone();
                }

                // If signed can represent all unsigned values
                if signed_kind.size() > unsigned_kind.size() {
                    return signed.clone();
                }

                // Otherwise, use unsigned version of signed type
                CType::Int(*signed_kind, Signedness::Unsigned)
            }

            _ => a, // Shouldn't happen after promotion
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int_kind_sizes() {
        assert_eq!(IntKind::Char.size(), 1);
        assert_eq!(IntKind::Short.size(), 2);
        assert_eq!(IntKind::Int.size(), 4);
        assert_eq!(IntKind::Long.size(), 8);
        assert_eq!(IntKind::LongLong.size(), 8);
        assert_eq!(IntKind::Bool.size(), 1);
    }

    #[test]
    fn test_int_kind_ranges() {
        assert_eq!(IntKind::Char.signed_min(), -128);
        assert_eq!(IntKind::Char.signed_max(), 127);
        assert_eq!(IntKind::Char.unsigned_max(), 255);

        assert_eq!(IntKind::Int.signed_min(), -2_147_483_648);
        assert_eq!(IntKind::Int.signed_max(), 2_147_483_647);
        assert_eq!(IntKind::Int.unsigned_max(), 4_294_967_295);
    }

    #[test]
    fn test_struct_layout() {
        // struct { char a; int b; char c; }
        let fields = vec![
            StructField {
                name: "a".to_string(),
                ty: CType::Int(IntKind::Char, Signedness::Signed),
            },
            StructField {
                name: "b".to_string(),
                ty: CType::Int(IntKind::Int, Signedness::Signed),
            },
            StructField {
                name: "c".to_string(),
                ty: CType::Int(IntKind::Char, Signedness::Signed),
            },
        ];
        let struct_ty = CType::Struct { name: None, fields };

        // Layout: a(1) + pad(3) + b(4) + c(1) + pad(3) = 12
        assert_eq!(struct_ty.size(), 12);
        assert_eq!(struct_ty.align(), 4);

        // Field offsets
        assert_eq!(struct_ty.field_offset("a"), Some(0));
        assert_eq!(struct_ty.field_offset("b"), Some(4));
        assert_eq!(struct_ty.field_offset("c"), Some(8));
    }

    #[test]
    fn test_union_layout() {
        // union { int i; double d; char c; }
        let fields = vec![
            StructField {
                name: "i".to_string(),
                ty: CType::Int(IntKind::Int, Signedness::Signed),
            },
            StructField {
                name: "d".to_string(),
                ty: CType::Float(FloatKind::Double),
            },
            StructField {
                name: "c".to_string(),
                ty: CType::Int(IntKind::Char, Signedness::Signed),
            },
        ];
        let union_ty = CType::Union { name: None, fields };

        // Size is max(4, 8, 1) = 8
        assert_eq!(union_ty.size(), 8);
        // Align is max(4, 8, 1) = 8
        assert_eq!(union_ty.align(), 8);

        // All fields at offset 0
        assert_eq!(union_ty.field_offset("i"), Some(0));
        assert_eq!(union_ty.field_offset("d"), Some(0));
        assert_eq!(union_ty.field_offset("c"), Some(0));
    }

    #[test]
    fn test_integer_promotion() {
        let char_ty = CType::Int(IntKind::Char, Signedness::Signed);
        let short_ty = CType::Int(IntKind::Short, Signedness::Unsigned);
        let int_ty = CType::Int(IntKind::Int, Signedness::Signed);
        let long_ty = CType::Int(IntKind::Long, Signedness::Signed);

        // char -> int
        assert_eq!(
            char_ty.integer_promotion(),
            CType::Int(IntKind::Int, Signedness::Signed)
        );

        // unsigned short -> int (fits)
        assert_eq!(
            short_ty.integer_promotion(),
            CType::Int(IntKind::Int, Signedness::Signed)
        );

        // int stays int
        assert_eq!(int_ty.integer_promotion(), int_ty);

        // long stays long
        assert_eq!(long_ty.integer_promotion(), long_ty);
    }

    #[test]
    fn test_usual_arithmetic_conversions() {
        let int_ty = CType::Int(IntKind::Int, Signedness::Signed);
        let uint_ty = CType::Int(IntKind::Int, Signedness::Unsigned);
        let long_ty = CType::Int(IntKind::Long, Signedness::Signed);
        let double_ty = CType::Float(FloatKind::Double);

        // int + unsigned int -> unsigned int
        assert_eq!(
            int_ty.usual_arithmetic_conversion(&uint_ty),
            CType::Int(IntKind::Int, Signedness::Unsigned)
        );

        // int + long -> long
        assert_eq!(int_ty.usual_arithmetic_conversion(&long_ty), long_ty);

        // int + double -> double
        assert_eq!(int_ty.usual_arithmetic_conversion(&double_ty), double_ty);
    }

    #[test]
    fn test_type_predicates() {
        let int_ty = CType::int();
        let ptr_ty = CType::ptr(CType::int());
        let arr_ty = CType::array(CType::int(), 10);
        let void_ty = CType::void();

        assert!(int_ty.is_integer());
        assert!(int_ty.is_arithmetic());
        assert!(int_ty.is_scalar());
        assert!(int_ty.is_complete());

        assert!(ptr_ty.is_pointer());
        assert!(ptr_ty.is_scalar());
        assert!(!ptr_ty.is_arithmetic());

        assert!(arr_ty.is_array());
        assert!(!arr_ty.is_scalar());

        assert!(!void_ty.is_complete());
    }

    #[test]
    fn test_pointer_pointee() {
        let int_ty = CType::int();
        let ptr_ty = CType::ptr(int_ty.clone());
        let ptr_ptr_ty = CType::ptr(ptr_ty.clone());

        assert_eq!(ptr_ty.pointee(), Some(&int_ty));
        assert_eq!(ptr_ptr_ty.pointee(), Some(&ptr_ty));
        assert_eq!(int_ty.pointee(), None);
    }

    #[test]
    fn test_qualified_types() {
        let int_ty = CType::int();
        let const_int = CType::const_ty(int_ty.clone());

        assert_eq!(const_int.size(), 4);
        assert_eq!(const_int.unqualified(), &int_ty);
        assert!(const_int.is_integer());
    }
}
