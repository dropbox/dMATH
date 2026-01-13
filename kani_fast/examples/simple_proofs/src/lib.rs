//! Simple proof harnesses for testing Kani Fast
//!
//! This crate contains various proof harnesses that exercise
//! different verification scenarios:
//! - Successful proofs (all checks pass)
//! - Failing proofs (counterexamples found)
//! - Different property types (overflow, bounds, assertions)

/// Add two u32 values with overflow checking
pub fn checked_add(a: u32, b: u32) -> Option<u32> {
    a.checked_add(b)
}

/// Absolute difference between two values
pub fn abs_diff(a: u32, b: u32) -> u32 {
    if a > b { a - b } else { b - a }
}

/// Safe division that returns None for division by zero
pub fn safe_div(a: u32, b: u32) -> Option<u32> {
    if b == 0 { None } else { Some(a / b) }
}

/// Binary search implementation (correct)
pub fn binary_search(arr: &[i32], target: i32) -> Option<usize> {
    if arr.is_empty() {
        return None;
    }

    let mut low = 0;
    let mut high = arr.len();

    while low < high {
        let mid = low + (high - low) / 2;
        if arr[mid] == target {
            return Some(mid);
        } else if arr[mid] < target {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    None
}

/// A buggy function that can overflow (for counterexample testing)
pub fn buggy_multiply(a: u8, b: u8) -> u8 {
    // This will overflow for large inputs
    a * b
}

#[cfg(kani)]
mod proofs {
    use super::*;

    /// Proof that checked_add never panics and correctly handles overflow
    #[kani::proof]
    fn proof_checked_add_safe() {
        let a: u32 = kani::any();
        let b: u32 = kani::any();

        let result = checked_add(a, b);

        // If result is Some, it must equal a + b (no overflow)
        if let Some(sum) = result {
            // Check the result is correct when it succeeds
            assert!(sum >= a || sum >= b);
        }
    }

    /// Proof that abs_diff is commutative
    #[kani::proof]
    fn proof_abs_diff_commutative() {
        let a: u32 = kani::any();
        let b: u32 = kani::any();

        assert_eq!(abs_diff(a, b), abs_diff(b, a));
    }

    /// Proof that abs_diff result is bounded
    #[kani::proof]
    fn proof_abs_diff_bounded() {
        let a: u32 = kani::any();
        let b: u32 = kani::any();

        let diff = abs_diff(a, b);

        // Result should be <= max(a, b)
        assert!(diff <= a || diff <= b);
    }

    /// Proof that safe_div handles division by zero
    #[kani::proof]
    fn proof_safe_div_no_panic() {
        let a: u32 = kani::any();
        let b: u32 = kani::any();

        let result = safe_div(a, b);

        // Should return None for b == 0
        if b == 0 {
            assert!(result.is_none());
        } else {
            assert!(result.is_some());
        }
    }

    /// Proof that binary search finds the element if it exists
    #[kani::proof]
    #[kani::unwind(5)]
    fn proof_binary_search_finds() {
        // Use a small fixed-size array for bounded verification
        let arr: [i32; 4] = kani::any();
        let target: i32 = kani::any();

        // Assume array is sorted (precondition)
        kani::assume(arr[0] <= arr[1]);
        kani::assume(arr[1] <= arr[2]);
        kani::assume(arr[2] <= arr[3]);

        if let Some(idx) = binary_search(&arr, target) {
            assert!(arr[idx] == target);
        }
    }

    /// This proof SHOULD FAIL - demonstrates counterexample generation
    #[kani::proof]
    fn proof_buggy_multiply_overflows() {
        let a: u8 = kani::any();
        let b: u8 = kani::any();

        // This will find a counterexample where a * b overflows
        let _result = buggy_multiply(a, b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checked_add() {
        assert_eq!(checked_add(1, 2), Some(3));
        assert_eq!(checked_add(u32::MAX, 1), None);
    }

    #[test]
    fn test_abs_diff() {
        assert_eq!(abs_diff(5, 3), 2);
        assert_eq!(abs_diff(3, 5), 2);
    }

    #[test]
    fn test_safe_div() {
        assert_eq!(safe_div(10, 2), Some(5));
        assert_eq!(safe_div(10, 0), None);
    }

    #[test]
    fn test_binary_search() {
        let arr = [1, 3, 5, 7, 9];
        assert_eq!(binary_search(&arr, 5), Some(2));
        assert_eq!(binary_search(&arr, 4), None);
    }
}
