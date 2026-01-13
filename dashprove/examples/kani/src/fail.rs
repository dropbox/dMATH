//! Kani examples that FAIL verification - they find bugs.

/// Buggy function: doesn't handle division by zero
pub fn unsafe_div(a: u32, b: u32) -> u32 {
    a / b  // BUG: panics when b == 0
}

/// Buggy function: can overflow
pub fn unsafe_add(a: u8, b: u8) -> u8 {
    a + b  // BUG: overflows in debug mode
}

/// Function with wrong assertion
pub fn wrong_assertion(x: u32) -> u32 {
    x * 2
}

#[cfg(kani)]
mod verification_fail {
    use super::*;

    /// This proof FAILS - finds division by zero bug
    #[kani::proof]
    fn verify_unsafe_div_fails() {
        let a: u32 = kani::any();
        let b: u32 = kani::any();
        let _result = unsafe_div(a, b);  // Will find counterexample when b == 0
    }

    /// This proof FAILS - finds overflow bug
    #[kani::proof]
    fn verify_unsafe_add_fails() {
        let a: u8 = kani::any();
        let b: u8 = kani::any();
        let _result = unsafe_add(a, b);  // Will find counterexample when a + b > 255
    }

    /// This proof FAILS - wrong assertion
    #[kani::proof]
    fn verify_wrong_assertion() {
        let x: u32 = kani::any();
        let result = wrong_assertion(x);
        assert!(result == x);  // WRONG: should be result == x * 2
    }
}
