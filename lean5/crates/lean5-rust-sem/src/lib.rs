//! # Lean5 Rust Semantics (lean5-rust-sem)
//!
//! This crate provides a formal model of Rust semantics that can be used
//! to verify Rust programs in Lean5. The ultimate goal is self-verification:
//! Lean5 verifying its own Rust implementation.
//!
//! ## Architecture
//!
//! The formalization follows the Rust memory model and type system:
//!
//! 1. **Types** (`types.rs`): Rust type system including ownership types
//! 2. **Memory** (`memory.rs`): Memory model with regions and lifetimes
//! 3. **Ownership** (`ownership.rs`): Ownership and borrowing model
//! 4. **Values** (`values.rs`): Value representation and operations
//! 5. **Expressions** (`expr.rs`): Rust expression semantics
//! 6. **Statements** (`stmt.rs`): Statement semantics and control flow
//! 7. **Translation** (`translate.rs`): Rust → Lean5 kernel translation
//!
//! ## Verification Approach
//!
//! We formalize Rust semantics using Lean5's kernel terms:
//!
//! - Rust types map to Lean5 types with ownership predicates
//! - Memory operations map to state-passing functions
//! - Ownership rules are encoded as proof obligations
//!
//! This allows Lean5 to verify properties of Rust programs,
//! including the Lean5 kernel itself.

pub mod eval;
pub mod expr;
pub mod memory;
pub mod ownership;
pub mod stmt;
pub mod translate;
pub mod types;
pub mod values;

pub use memory::*;
pub use ownership::*;
pub use types::*;
pub use values::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_type_construction() {
        let unit_ty = RustType::Unit;
        let bool_ty = RustType::Bool;
        let u32_ty = RustType::Uint(UintType::U32);
        let i64_ty = RustType::Int(IntType::I64);

        assert_eq!(unit_ty.size(), Some(0));
        assert_eq!(bool_ty.size(), Some(1));
        assert_eq!(u32_ty.size(), Some(4));
        assert_eq!(i64_ty.size(), Some(8));
    }

    #[test]
    fn test_reference_types() {
        let lifetime = Lifetime::Named("a".to_string());
        let inner = RustType::Bool;

        let shared_ref = RustType::Reference {
            lifetime: lifetime.clone(),
            mutability: Mutability::Shared,
            inner: Box::new(inner.clone()),
        };

        let mutable_ref = RustType::Reference {
            lifetime,
            mutability: Mutability::Mutable,
            inner: Box::new(inner),
        };

        // References have pointer size (8 bytes on 64-bit)
        assert_eq!(shared_ref.size(), Some(8));
        assert_eq!(mutable_ref.size(), Some(8));
    }

    #[test]
    fn test_ownership_state() {
        let place = Place::local(0);

        let mut state = OwnershipState::new();
        state.mark_owned(place.clone());

        assert!(state.is_owned(&place));
        assert!(!state.is_borrowed(&place));
        assert!(!state.is_moved(&place));

        state.mark_moved(place.clone());
        assert!(!state.is_owned(&place));
        assert!(state.is_moved(&place));
    }

    #[test]
    fn test_memory_model() {
        let mut mem = Memory::new();

        // Allocate a value
        let ptr = mem.allocate(4).expect("allocation failed");
        assert!(mem.is_valid(ptr));

        // Write and read
        mem.write_u32(ptr, 42).expect("write failed");
        let val = mem.read_u32(ptr).expect("read failed");
        assert_eq!(val, 42);

        // Deallocate
        mem.deallocate(ptr).expect("deallocation failed");
        assert!(!mem.is_valid(ptr));
    }

    #[test]
    fn test_borrow_checker_rules() {
        let checker = BorrowChecker::new();
        let place = Place::local(0);
        let lifetime = Lifetime::Named("a".to_string());

        // Start with owned value
        let mut state = OwnershipState::new();
        state.mark_owned(place.clone());

        // Can create shared borrow
        let result = checker.check_borrow(&state, &place, Mutability::Shared, &lifetime);
        assert!(result.is_ok());

        // Can create multiple shared borrows (would be checked in full impl)
    }

    #[test]
    fn test_type_compatibility() {
        let u32_ty = RustType::Uint(UintType::U32);
        let i32_ty = RustType::Int(IntType::I32);
        let bool_ty = RustType::Bool;

        // Same types are compatible
        assert!(u32_ty.is_compatible(&u32_ty));
        assert!(bool_ty.is_compatible(&bool_ty));

        // Different numeric types are not compatible
        assert!(!u32_ty.is_compatible(&i32_ty));
        assert!(!u32_ty.is_compatible(&bool_ty));
    }
}
