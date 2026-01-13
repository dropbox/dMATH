//! Test project for DashProve Kani e2e verification tests
//!
//! This crate provides simple functions with known properties that can be
//! verified via DashProve's Kani backend integration.

/// Adds two bounded u32 values (guaranteed no overflow when inputs < 1000)
pub fn bounded_add(x: u32, y: u32) -> u32 {
    x + y
}

/// Always returns a value greater than or equal to input
pub fn increment(x: u32) -> u32 {
    x.saturating_add(1)
}

/// Identity function - always returns input
pub fn identity(x: u32) -> u32 {
    x
}

/// Returns double the input (can overflow for large inputs)
pub fn double(x: u32) -> u32 {
    x.saturating_mul(2)
}

/// Returns true if x is even
pub fn is_even(x: u32) -> bool {
    x % 2 == 0
}

/// Returns the maximum of two values
pub fn max_of(a: u32, b: u32) -> u32 {
    if a >= b { a } else { b }
}

/// Returns the minimum of two values
pub fn min_of(a: u32, b: u32) -> u32 {
    if a <= b { a } else { b }
}

/// Absolute difference between two values
pub fn abs_diff(a: u32, b: u32) -> u32 {
    if a >= b { a - b } else { b - a }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounded_add() {
        assert_eq!(bounded_add(100, 200), 300);
    }

    #[test]
    fn test_increment() {
        assert_eq!(increment(5), 6);
        assert_eq!(increment(u32::MAX), u32::MAX); // saturating
    }

    #[test]
    fn test_identity() {
        assert_eq!(identity(42), 42);
    }

    #[test]
    fn test_double() {
        assert_eq!(double(5), 10);
    }

    #[test]
    fn test_is_even() {
        assert!(is_even(4));
        assert!(!is_even(5));
    }

    #[test]
    fn test_max_min() {
        assert_eq!(max_of(3, 7), 7);
        assert_eq!(min_of(3, 7), 3);
    }

    #[test]
    fn test_abs_diff() {
        assert_eq!(abs_diff(10, 3), 7);
        assert_eq!(abs_diff(3, 10), 7);
    }
}
