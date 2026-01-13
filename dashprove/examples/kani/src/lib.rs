//! Minimal Kani examples demonstrating bounded model checking for Rust.

mod fail;

/// Example function: safe add that won't overflow
pub fn safe_add(a: u8, b: u8) -> Option<u8> {
    a.checked_add(b)
}

/// Example function: unsafe add that CAN overflow
pub fn unsafe_add(a: u8, b: u8) -> u8 {
    a + b  // Will panic on overflow in debug mode
}

/// Example function: division
pub fn safe_div(a: u32, b: u32) -> Option<u32> {
    if b == 0 {
        None
    } else {
        Some(a / b)
    }
}

/// Example function: unsafe division (division by zero possible)
pub fn unsafe_div(a: u32, b: u32) -> u32 {
    a / b  // Will panic if b == 0
}

// ============== KANI PROOF HARNESSES ==============

#[cfg(kani)]
mod verification {
    use super::*;

    /// This proof PASSES - safe_add handles overflow
    #[kani::proof]
    fn verify_safe_add_no_panic() {
        let a: u8 = kani::any();
        let b: u8 = kani::any();
        let result = safe_add(a, b);
        // safe_add should never panic - it returns None on overflow
        if let Some(sum) = result {
            assert!(sum >= a || sum >= b);
        }
    }

    /// This proof PASSES - safe_div handles division by zero
    #[kani::proof]
    fn verify_safe_div_no_panic() {
        let a: u32 = kani::any();
        let b: u32 = kani::any();
        let result = safe_div(a, b);
        // safe_div should never panic
        if let Some(quotient) = result {
            assert!(b != 0);
            assert!(quotient <= a);
        }
    }

    /// This proof PASSES - basic arithmetic property
    #[kani::proof]
    fn verify_add_commutative() {
        let a: u32 = kani::any();
        let b: u32 = kani::any();
        // Assume no overflow
        kani::assume(a.checked_add(b).is_some());
        assert!(a.wrapping_add(b) == b.wrapping_add(a));
    }
}
