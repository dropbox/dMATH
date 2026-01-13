// Copyright Kani Fast Contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Compatibility shims for types from cprover_bindings that we don't want to depend on.
//!
//! These provide the same interfaces but without the full CBMC codegen dependency.

/// A simple interned string type that wraps a String.
/// In cbmc/cprover_bindings, this is a thread-local interned string pool.
/// For our purposes (CHC backend), we don't need interning - just the type compatibility.
pub type InternedString = String;

/// Trait for interning strings.
/// In the real cbmc crate, this interns into a thread-local pool.
/// Here we just convert to String.
pub trait InternString {
    fn intern(&self) -> InternedString;
}

impl InternString for str {
    fn intern(&self) -> InternedString {
        self.to_string()
    }
}

impl InternString for String {
    fn intern(&self) -> InternedString {
        self.clone()
    }
}

impl InternString for &str {
    fn intern(&self) -> InternedString {
        (*self).to_string()
    }
}
