//! Rust Type System Formalization
//!
//! This module defines the Rust type system as it relates to
//! ownership and borrowing semantics.
//!
//! ## Type Categories
//!
//! - **Primitive Types**: bool, integers, floats, char
//! - **Compound Types**: tuples, arrays, structs, enums
//! - **Reference Types**: &T, &mut T with lifetimes
//! - **Pointer Types**: *const T, *mut T
//! - **Function Types**: fn(A) -> B
//! - **Trait Objects**: dyn Trait
//!
//! ## Ownership Properties
//!
//! Types have several ownership-related properties:
//!
//! - **Copy**: Type can be bitwise copied (no ownership transfer)
//! - **Clone**: Type can be explicitly cloned
//! - **Drop**: Type has custom destructor
//! - **Send**: Type can be sent between threads
//! - **Sync**: Type can be shared between threads via references

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Mutability qualifier for references and pointers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Mutability {
    /// Shared/immutable access
    Shared,
    /// Exclusive/mutable access
    Mutable,
}

/// Lifetime in the Rust type system
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Lifetime {
    /// Static lifetime (lives for entire program)
    Static,
    /// Named lifetime parameter (e.g., 'a)
    Named(String),
    /// Anonymous/elided lifetime
    Anonymous(u32),
    /// Existential lifetime (for type inference)
    Existential(u32),
}

impl Lifetime {
    /// Check if this lifetime outlives another
    pub fn outlives(&self, other: &Lifetime) -> bool {
        match (self, other) {
            (Lifetime::Static, _) => true,
            (Lifetime::Named(a), Lifetime::Named(b)) => a == b,
            // Conservative: unknown lifetimes don't outlive each other
            (_, Lifetime::Static)
            | (Lifetime::Named(_), _)
            | (_, Lifetime::Named(_))
            | (Lifetime::Anonymous(_), _)
            | (Lifetime::Existential(_), _) => false,
        }
    }
}

/// Unsigned integer types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UintType {
    U8,
    U16,
    U32,
    U64,
    U128,
    Usize,
}

impl UintType {
    /// Size in bytes (assumes 64-bit platform for `usize`)
    pub fn size(&self) -> usize {
        match self {
            UintType::U8 => 1,
            UintType::U16 => 2,
            UintType::U32 => 4,
            UintType::U64 | UintType::Usize => 8,
            UintType::U128 => 16,
        }
    }
}

/// Signed integer types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IntType {
    I8,
    I16,
    I32,
    I64,
    I128,
    Isize,
}

impl IntType {
    /// Size in bytes (assumes 64-bit platform for `isize`)
    pub fn size(&self) -> usize {
        match self {
            IntType::I8 => 1,
            IntType::I16 => 2,
            IntType::I32 => 4,
            IntType::I64 | IntType::Isize => 8,
            IntType::I128 => 16,
        }
    }
}

/// Floating point types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FloatType {
    F32,
    F64,
}

impl FloatType {
    pub fn size(&self) -> usize {
        match self {
            FloatType::F32 => 4,
            FloatType::F64 => 8,
        }
    }
}

/// Type variable for generic types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TypeVar {
    pub id: u32,
    pub name: Option<String>,
}

/// Rust type representation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RustType {
    /// Unit type ()
    Unit,

    /// Boolean
    Bool,

    /// Character (Unicode scalar)
    Char,

    /// Unsigned integer
    Uint(UintType),

    /// Signed integer
    Int(IntType),

    /// Floating point
    Float(FloatType),

    /// Reference with lifetime
    Reference {
        lifetime: Lifetime,
        mutability: Mutability,
        inner: Box<RustType>,
    },

    /// Raw pointer
    RawPtr {
        mutability: Mutability,
        inner: Box<RustType>,
    },

    /// Fixed-size array [T; N]
    Array { element: Box<RustType>, len: usize },

    /// Slice type `[T]`
    Slice { element: Box<RustType> },

    /// String slice str
    Str,

    /// Tuple (T1, T2, ...)
    Tuple(Vec<RustType>),

    /// Function type fn(Args) -> Ret
    Function {
        params: Vec<RustType>,
        ret: Box<RustType>,
    },

    /// Named type (struct, enum, etc.)
    Named {
        name: String,
        /// Generic type arguments
        type_args: Vec<RustType>,
        /// Generic lifetime arguments
        lifetime_args: Vec<Lifetime>,
    },

    /// Type parameter (generic T)
    TypeParam(TypeVar),

    /// The "never" type !
    Never,

    /// `Box<T>` - owned heap allocation
    Box { inner: Box<RustType> },

    /// `Option<T>`
    Option { inner: Box<RustType> },

    /// Result<T, E>
    Result {
        ok: Box<RustType>,
        err: Box<RustType>,
    },

    /// `Vec<T>`
    Vec { element: Box<RustType> },

    /// Dynamic trait object dyn Trait
    DynTrait {
        traits: Vec<String>,
        lifetime: Lifetime,
    },

    /// Impl trait (impl Trait)
    ImplTrait { traits: Vec<String> },

    /// Closure type (with captured environment)
    Closure {
        params: Vec<RustType>,
        ret: Box<RustType>,
        captures: Vec<(String, RustType, Mutability)>,
    },
}

impl RustType {
    /// Size of the type in bytes (None for unsized types)
    pub fn size(&self) -> Option<usize> {
        match self {
            RustType::Unit | RustType::Never => Some(0),
            RustType::Bool => Some(1),
            RustType::Char => Some(4),
            RustType::Uint(u) => Some(u.size()),
            RustType::Int(i) => Some(i.size()),
            RustType::Float(f) => Some(f.size()),
            // Pointer-sized types (8 bytes on 64-bit)
            RustType::Reference { .. }
            | RustType::RawPtr { .. }
            | RustType::Function { .. }
            | RustType::Box { .. } => Some(8),
            RustType::Array { element, len } => element.size().map(|s| s * len),
            RustType::Tuple(elems) => {
                let mut size = 0;
                for elem in elems {
                    size += elem.size()?;
                }
                Some(size)
            }
            RustType::Option { inner } => {
                // Option<T> has size of T + discriminant (usually 1)
                // But Option<&T> uses null pointer optimization
                if matches!(**inner, RustType::Reference { .. }) {
                    Some(8)
                } else {
                    inner.size().map(|s| s + 1)
                }
            }
            RustType::Result { ok, err } => {
                // Size is max(ok, err) + discriminant
                let ok_size = ok.size()?;
                let err_size = err.size()?;
                Some(ok_size.max(err_size) + 1)
            }
            RustType::Vec { .. } => Some(24), // ptr + len + cap
            // Unsized types
            RustType::Slice { .. }
            | RustType::Str
            | RustType::DynTrait { .. }
            | RustType::ImplTrait { .. }
            | RustType::Closure { .. }
            | RustType::Named { .. }
            | RustType::TypeParam(_) => None,
        }
    }

    /// Check if type is sized (has known size at compile time)
    pub fn is_sized(&self) -> bool {
        self.size().is_some()
    }

    /// Check if type implements Copy
    pub fn is_copy(&self) -> bool {
        match self {
            RustType::Unit
            | RustType::Bool
            | RustType::Char
            | RustType::Uint(_)
            | RustType::Int(_)
            | RustType::Float(_)
            | RustType::RawPtr { .. }
            | RustType::Function { .. }
            | RustType::Never
            | RustType::Reference {
                mutability: Mutability::Shared,
                ..
            } => true,
            RustType::Array { element, .. } => element.is_copy(),
            RustType::Tuple(elems) => elems.iter().all(RustType::is_copy),
            // &mut T is not Copy, nor are heap allocations, unsized types, etc.
            RustType::Reference {
                mutability: Mutability::Mutable,
                ..
            }
            | RustType::Slice { .. }
            | RustType::Str
            | RustType::Named { .. }
            | RustType::TypeParam(_)
            | RustType::Box { .. }
            | RustType::Option { .. }
            | RustType::Result { .. }
            | RustType::Vec { .. }
            | RustType::DynTrait { .. }
            | RustType::ImplTrait { .. }
            | RustType::Closure { .. } => false,
        }
    }

    /// Check if type is compatible (structurally equal) with another
    pub fn is_compatible(&self, other: &RustType) -> bool {
        match (self, other) {
            (RustType::Unit, RustType::Unit)
            | (RustType::Bool, RustType::Bool)
            | (RustType::Char, RustType::Char)
            | (RustType::Never, RustType::Never) => true,
            (RustType::Uint(a), RustType::Uint(b)) => a == b,
            (RustType::Int(a), RustType::Int(b)) => a == b,
            (RustType::Float(a), RustType::Float(b)) => a == b,
            (
                RustType::Reference {
                    lifetime: l1,
                    mutability: m1,
                    inner: i1,
                },
                RustType::Reference {
                    lifetime: l2,
                    mutability: m2,
                    inner: i2,
                },
            ) => l1 == l2 && m1 == m2 && i1.is_compatible(i2),
            (
                RustType::Array {
                    element: e1,
                    len: l1,
                },
                RustType::Array {
                    element: e2,
                    len: l2,
                },
            ) => l1 == l2 && e1.is_compatible(e2),
            (RustType::Tuple(e1), RustType::Tuple(e2)) => {
                e1.len() == e2.len() && e1.iter().zip(e2).all(|(a, b)| a.is_compatible(b))
            }
            _ => false,
        }
    }

    /// Check if this type has interior mutability (UnsafeCell-like)
    pub fn has_interior_mutability(&self) -> bool {
        match self {
            // These would need to check for UnsafeCell wrapper
            RustType::Named { name, .. } => {
                matches!(
                    name.as_str(),
                    "Cell"
                        | "RefCell"
                        | "UnsafeCell"
                        | "Mutex"
                        | "RwLock"
                        | "AtomicBool"
                        | "AtomicI8"
                        | "AtomicI16"
                        | "AtomicI32"
                        | "AtomicI64"
                        | "AtomicU8"
                        | "AtomicU16"
                        | "AtomicU32"
                        | "AtomicU64"
                        | "AtomicUsize"
                        | "AtomicIsize"
                        | "AtomicPtr"
                )
            }
            _ => false,
        }
    }

    /// Check if type is Send (can be transferred between threads)
    pub fn is_send(&self) -> bool {
        match self {
            RustType::Unit
            | RustType::Bool
            | RustType::Char
            | RustType::Uint(_)
            | RustType::Int(_)
            | RustType::Float(_)
            | RustType::Never => true,
            RustType::Reference { inner, .. } => inner.is_sync(),
            RustType::Array { element, .. } | RustType::Vec { element } => element.is_send(),
            RustType::Tuple(elems) => elems.iter().all(RustType::is_send),
            RustType::Box { inner } | RustType::Option { inner } => inner.is_send(),
            RustType::Result { ok, err } => ok.is_send() && err.is_send(),
            // Raw pointers are not Send by default, nor are unsized types, etc.
            RustType::RawPtr { .. }
            | RustType::Slice { .. }
            | RustType::Str
            | RustType::Function { .. }
            | RustType::Named { .. }
            | RustType::TypeParam(_)
            | RustType::DynTrait { .. }
            | RustType::ImplTrait { .. }
            | RustType::Closure { .. } => false,
        }
    }

    /// Check if type is Sync (can be shared between threads via &T)
    pub fn is_sync(&self) -> bool {
        match self {
            RustType::Unit
            | RustType::Bool
            | RustType::Char
            | RustType::Uint(_)
            | RustType::Int(_)
            | RustType::Float(_)
            | RustType::Never => true,
            RustType::RawPtr { .. } => false,
            // Types with interior mutability need Sync wrapper
            t if t.has_interior_mutability() => false,
            RustType::Reference { inner, .. } | RustType::Box { inner } | RustType::Option { inner } => inner.is_sync(),
            RustType::Array { element, .. } | RustType::Vec { element } => element.is_sync(),
            RustType::Tuple(elems) => elems.iter().all(RustType::is_sync),
            RustType::Result { ok, err } => ok.is_sync() && err.is_sync(),
            _ => false,
        }
    }
}

/// Struct field definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructField {
    pub name: String,
    pub ty: RustType,
    pub visibility: Visibility,
}

/// Visibility modifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Visibility {
    /// Private (default)
    Private,
    /// pub
    Public,
    /// pub(crate)
    Crate,
    /// pub(super)
    Super,
}

/// Struct definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructDef {
    pub name: String,
    pub type_params: Vec<TypeVar>,
    pub lifetime_params: Vec<String>,
    pub fields: Vec<StructField>,
    /// Derived traits (Copy, Clone, Debug, etc.)
    pub derives: Vec<String>,
}

impl StructDef {
    /// Check if struct is Copy
    pub fn is_copy(&self) -> bool {
        self.derives.contains(&"Copy".to_string()) && self.fields.iter().all(|f| f.ty.is_copy())
    }

    /// Calculate struct size (None if contains unsized field)
    pub fn size(&self) -> Option<usize> {
        let mut total = 0;
        for field in &self.fields {
            total += field.ty.size()?;
        }
        Some(total)
    }
}

/// Enum variant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnumVariant {
    /// Unit variant: Foo
    Unit { name: String },
    /// Tuple variant: Foo(T1, T2)
    Tuple { name: String, fields: Vec<RustType> },
    /// Struct variant: Foo { x: T1, y: T2 }
    Struct {
        name: String,
        fields: Vec<StructField>,
    },
}

/// Enum definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnumDef {
    pub name: String,
    pub type_params: Vec<TypeVar>,
    pub lifetime_params: Vec<String>,
    pub variants: Vec<EnumVariant>,
    pub derives: Vec<String>,
}

/// Type context for name resolution
#[derive(Debug, Clone, Default)]
pub struct TypeContext {
    pub structs: HashMap<String, StructDef>,
    pub enums: HashMap<String, EnumDef>,
    pub type_aliases: HashMap<String, RustType>,
}

impl TypeContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Resolve a named type to its definition
    pub fn resolve_type(&self, name: &str) -> Option<&StructDef> {
        self.structs.get(name)
    }

    /// Get size of a named type
    pub fn named_type_size(&self, name: &str) -> Option<usize> {
        self.structs.get(name).and_then(StructDef::size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primitive_sizes() {
        assert_eq!(RustType::Unit.size(), Some(0));
        assert_eq!(RustType::Bool.size(), Some(1));
        assert_eq!(RustType::Char.size(), Some(4));
        assert_eq!(RustType::Uint(UintType::U8).size(), Some(1));
        assert_eq!(RustType::Uint(UintType::U64).size(), Some(8));
        assert_eq!(RustType::Int(IntType::I32).size(), Some(4));
        assert_eq!(RustType::Float(FloatType::F64).size(), Some(8));
    }

    #[test]
    fn test_copy_types() {
        assert!(RustType::Bool.is_copy());
        assert!(RustType::Uint(UintType::U32).is_copy());
        assert!(RustType::Int(IntType::I64).is_copy());

        let shared_ref = RustType::Reference {
            lifetime: Lifetime::Static,
            mutability: Mutability::Shared,
            inner: Box::new(RustType::Bool),
        };
        assert!(shared_ref.is_copy());

        let mut_ref = RustType::Reference {
            lifetime: Lifetime::Static,
            mutability: Mutability::Mutable,
            inner: Box::new(RustType::Bool),
        };
        assert!(!mut_ref.is_copy());
    }

    #[test]
    fn test_lifetime_outlives() {
        let static_lt = Lifetime::Static;
        let named_a = Lifetime::Named("a".to_string());
        let named_b = Lifetime::Named("b".to_string());

        assert!(static_lt.outlives(&named_a));
        assert!(!named_a.outlives(&static_lt));
        assert!(named_a.outlives(&named_a));
        assert!(!named_a.outlives(&named_b));
    }

    #[test]
    fn test_struct_definition() {
        let point_struct = StructDef {
            name: "Point".to_string(),
            type_params: vec![],
            lifetime_params: vec![],
            fields: vec![
                StructField {
                    name: "x".to_string(),
                    ty: RustType::Float(FloatType::F64),
                    visibility: Visibility::Public,
                },
                StructField {
                    name: "y".to_string(),
                    ty: RustType::Float(FloatType::F64),
                    visibility: Visibility::Public,
                },
            ],
            derives: vec!["Copy".to_string(), "Clone".to_string()],
        };

        assert_eq!(point_struct.size(), Some(16));
        assert!(point_struct.is_copy());
    }
}
