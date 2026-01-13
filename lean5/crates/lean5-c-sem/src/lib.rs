//! # Lean5 C Semantics (lean5-c-sem)
//!
//! This crate provides a formal model of C semantics that can be used
//! to verify C programs in Lean5. The formalization is based on:
//!
//! - **CompCert memory model**: Proven correct C memory semantics
//! - **Cerberus**: Rigorous C memory object model
//! - **ACSL/Frama-C**: Specification language concepts
//! - **VST**: Separation logic for C verification
//!
//! ## Architecture
//!
//! The formalization models C11 semantics with the following modules:
//!
//! 1. **Types** (`types.rs`): C type system (primitives, pointers, structs, unions)
//! 2. **Memory** (`memory.rs`): Memory model with blocks, provenance, and UB detection
//! 3. **Values** (`values.rs`): C value representation (including undefined values)
//! 4. **Expr** (`expr.rs`): Expression semantics and evaluation
//! 5. **Stmt** (`stmt.rs`): Statement execution and control flow
//! 6. **UB** (`ub.rs`): Undefined behavior detection and categorization
//! 7. **Eval** (`eval.rs`): Operational semantics interpreter
//! 8. **Spec** (`spec.rs`): ACSL-style specification language
//! 9. **Translate** (`translate.rs`): C → Lean5 kernel translation
//!
//! ## Memory Model
//!
//! The memory model is block-based (like CompCert):
//!
//! - Memory is a collection of blocks, each with a unique block ID
//! - Pointers are (block_id, offset) pairs
//! - Provenance is tracked to detect use-after-free
//! - Undefined behavior is modeled explicitly
//!
//! ## Undefined Behavior
//!
//! C has extensive undefined behavior (UB). We model it by:
//!
//! 1. **Detecting UB**: Operations that trigger UB return `Err(UB::Kind)`
//! 2. **Categorizing UB**: Different UB kinds (null deref, overflow, etc.)
//! 3. **Proving absence**: Verification goals include showing no UB paths
//!
//! ## Example
//!
//! ```ignore
//! use lean5_c_sem::*;
//!
//! // Create a C type for int
//! let int_ty = CType::Int(IntKind::Int, Signedness::Signed);
//!
//! // Allocate memory
//! let mut mem = Memory::new();
//! let ptr = mem.alloc(int_ty.size()).unwrap();
//!
//! // Write a value
//! mem.store_int(ptr, 42i32).unwrap();
//!
//! // Read it back
//! let val = mem.load_int::<i32>(ptr).unwrap();
//! assert_eq!(val, 42);
//! ```

pub mod auto;
pub mod eval;
pub mod examples;
pub mod expr;
pub mod memory;
pub mod parser;
pub mod sep;
pub mod spec;
pub mod stmt;
pub mod translate;
pub mod types;
pub mod ub;
pub mod values;
pub mod vcgen;
pub mod verified;

pub use memory::*;
pub use types::*;
pub use ub::*;
pub use values::*;
pub use verified::VerifiedFunction;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_int_types() {
        let int_ty = CType::Int(IntKind::Int, Signedness::Signed);
        let uint_ty = CType::Int(IntKind::Int, Signedness::Unsigned);
        let char_ty = CType::Int(IntKind::Char, Signedness::Signed);

        // Size assumptions (LP64 model)
        assert_eq!(int_ty.size(), 4);
        assert_eq!(uint_ty.size(), 4);
        assert_eq!(char_ty.size(), 1);
    }

    #[test]
    fn test_pointer_types() {
        let int_ty = CType::Int(IntKind::Int, Signedness::Signed);
        let ptr_ty = CType::Pointer(Box::new(int_ty.clone()));
        let ptr_ptr_ty = CType::Pointer(Box::new(ptr_ty.clone()));

        // Pointers are 8 bytes on 64-bit
        assert_eq!(ptr_ty.size(), 8);
        assert_eq!(ptr_ptr_ty.size(), 8);
    }

    #[test]
    fn test_memory_allocation() {
        let mut mem = Memory::new();

        // Allocate 4 bytes
        let ptr = mem.alloc(4, 4).expect("allocation should succeed");
        assert!(mem.is_valid(ptr));

        // Write and read
        mem.store_bytes(ptr, &[1, 2, 3, 4])
            .expect("store should succeed");
        let bytes = mem.load_bytes(ptr, 4).expect("load should succeed");
        assert_eq!(bytes, vec![1, 2, 3, 4]);

        // Free
        mem.free(ptr).expect("free should succeed");
        assert!(!mem.is_valid(ptr));
    }

    #[test]
    fn test_memory_use_after_free() {
        let mut mem = Memory::new();
        let ptr = mem.alloc(4, 4).expect("allocation should succeed");
        mem.free(ptr).expect("free should succeed");

        // Use after free should fail
        let result = mem.load_bytes(ptr, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_null_pointer() {
        let mem = Memory::new();
        let null_ptr = Pointer::null();

        assert!(null_ptr.is_null());
        assert!(!mem.is_valid(null_ptr));
    }

    #[test]
    fn test_pointer_arithmetic() {
        let mut mem = Memory::new();
        let ptr = mem.alloc(16, 4).expect("allocation should succeed");

        // Offset pointer
        let ptr_plus_4 = ptr.offset(4).expect("offset should succeed");
        let ptr_plus_8 = ptr.offset(8).expect("offset should succeed");

        // Write at different offsets
        mem.store_bytes(ptr, &[1, 0, 0, 0])
            .expect("store should succeed");
        mem.store_bytes(ptr_plus_4, &[2, 0, 0, 0])
            .expect("store should succeed");
        mem.store_bytes(ptr_plus_8, &[3, 0, 0, 0])
            .expect("store should succeed");

        // Read back
        assert_eq!(mem.load_bytes(ptr, 1).unwrap(), vec![1]);
        assert_eq!(mem.load_bytes(ptr_plus_4, 1).unwrap(), vec![2]);
        assert_eq!(mem.load_bytes(ptr_plus_8, 1).unwrap(), vec![3]);
    }

    #[test]
    fn test_out_of_bounds_access() {
        let mut mem = Memory::new();
        let ptr = mem.alloc(4, 4).expect("allocation should succeed");

        // Out of bounds access should fail
        let result = mem.load_bytes(ptr.offset(8).unwrap(), 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_cvalue_construction() {
        let int_val = CValue::Int(42);
        let uint_val = CValue::UInt(100);
        let ptr_val = CValue::Pointer(Pointer::null());
        let undef = CValue::Undef;

        assert!(!int_val.is_undef());
        assert!(!uint_val.is_undef());
        assert!(!ptr_val.is_undef());
        assert!(undef.is_undef());
    }

    #[test]
    fn test_struct_type() {
        // struct { int x; char c; int y; }
        let fields = vec![
            StructField {
                name: "x".to_string(),
                ty: CType::Int(IntKind::Int, Signedness::Signed),
            },
            StructField {
                name: "c".to_string(),
                ty: CType::Int(IntKind::Char, Signedness::Signed),
            },
            StructField {
                name: "y".to_string(),
                ty: CType::Int(IntKind::Int, Signedness::Signed),
            },
        ];
        let struct_ty = CType::Struct {
            name: Some("Point".to_string()),
            fields,
        };

        // Size includes padding: 4 + 1 + 3(pad) + 4 = 12
        assert_eq!(struct_ty.size(), 12);
        assert_eq!(struct_ty.align(), 4);
    }

    #[test]
    fn test_array_type() {
        let int_ty = CType::Int(IntKind::Int, Signedness::Signed);
        let arr_ty = CType::Array(Box::new(int_ty), 10);

        // Array of 10 ints = 40 bytes
        assert_eq!(arr_ty.size(), 40);
        assert_eq!(arr_ty.align(), 4);
    }
}
