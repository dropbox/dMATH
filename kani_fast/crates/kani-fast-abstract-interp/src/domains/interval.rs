//! Interval domain for tracking numeric bounds.
//!
//! An interval [lo, hi] represents all integers x where lo ≤ x ≤ hi.
//! This is one of the most fundamental abstract domains.

use std::cmp::Ordering;

use crate::lattice::Lattice;

/// An interval representing a range of integers.
/// Uses i128 to handle all Rust integer types.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Interval {
    /// Bottom: no possible values (unreachable/empty set)
    Bottom,
    /// A bounded interval [lo, hi]
    Range { lo: i128, hi: i128 },
    /// Top: all possible values
    Top,
}

impl Interval {
    /// Create a new bounded interval.
    #[inline]
    pub fn new(lo: i128, hi: i128) -> Self {
        if lo > hi {
            Interval::Bottom
        } else {
            Interval::Range { lo, hi }
        }
    }

    /// Create a singleton interval containing just one value.
    #[inline]
    pub fn singleton(value: i128) -> Self {
        Interval::Range {
            lo: value,
            hi: value,
        }
    }

    /// Check if the interval contains a specific value.
    #[inline]
    pub fn contains(&self, value: i128) -> bool {
        match self {
            Interval::Bottom => false,
            Interval::Range { lo, hi } => *lo <= value && value <= *hi,
            Interval::Top => true,
        }
    }

    /// Get the lower bound, if bounded.
    pub fn lower_bound(&self) -> Option<i128> {
        match self {
            Interval::Range { lo, .. } => Some(*lo),
            Interval::Bottom | Interval::Top => None,
        }
    }

    /// Get the upper bound, if bounded.
    pub fn upper_bound(&self) -> Option<i128> {
        match self {
            Interval::Range { hi, .. } => Some(*hi),
            Interval::Bottom | Interval::Top => None,
        }
    }

    /// Check if all values in the interval are non-negative.
    pub fn is_non_negative(&self) -> bool {
        match self {
            Interval::Bottom => true, // vacuously true
            Interval::Range { lo, .. } => *lo >= 0,
            Interval::Top => false,
        }
    }

    /// Check if all values in the interval are strictly positive.
    pub fn is_positive(&self) -> bool {
        match self {
            Interval::Bottom => true, // vacuously true
            Interval::Range { lo, .. } => *lo > 0,
            Interval::Top => false,
        }
    }

    /// Check if all values are strictly less than a bound.
    pub fn all_less_than(&self, bound: i128) -> bool {
        match self {
            Interval::Bottom => true,
            Interval::Range { hi, .. } => *hi < bound,
            Interval::Top => false,
        }
    }

    /// Check if all values are less than or equal to a bound.
    pub fn all_leq(&self, bound: i128) -> bool {
        match self {
            Interval::Bottom => true,
            Interval::Range { hi, .. } => *hi <= bound,
            Interval::Top => false,
        }
    }

    /// Addition: [a, b] + [c, d] = [a+c, b+d]
    pub fn add(&self, other: &Self) -> Self {
        match (self, other) {
            (Interval::Bottom, _) | (_, Interval::Bottom) => Interval::Bottom,
            (Interval::Top, _) | (_, Interval::Top) => Interval::Top,
            (Interval::Range { lo: a, hi: b }, Interval::Range { lo: c, hi: d }) => {
                // Check for overflow
                let lo = a.checked_add(*c);
                let hi = b.checked_add(*d);
                match (lo, hi) {
                    (Some(l), Some(h)) => Interval::Range { lo: l, hi: h },
                    _ => Interval::Top, // Overflow → Top
                }
            }
        }
    }

    /// Subtraction: [a, b] - [c, d] = [a-d, b-c]
    pub fn sub(&self, other: &Self) -> Self {
        match (self, other) {
            (Interval::Bottom, _) | (_, Interval::Bottom) => Interval::Bottom,
            (Interval::Top, _) | (_, Interval::Top) => Interval::Top,
            (Interval::Range { lo: a, hi: b }, Interval::Range { lo: c, hi: d }) => {
                let lo = a.checked_sub(*d);
                let hi = b.checked_sub(*c);
                match (lo, hi) {
                    (Some(l), Some(h)) => Interval::Range { lo: l, hi: h },
                    _ => Interval::Top,
                }
            }
        }
    }

    /// Multiplication (sound but imprecise for mixed-sign intervals)
    pub fn mul(&self, other: &Self) -> Self {
        match (self, other) {
            (Interval::Bottom, _) | (_, Interval::Bottom) => Interval::Bottom,
            (Interval::Top, _) | (_, Interval::Top) => Interval::Top,
            (Interval::Range { lo: a, hi: b }, Interval::Range { lo: c, hi: d }) => {
                // Compute all four products on stack (no heap allocation)
                let products = [
                    a.checked_mul(*c),
                    a.checked_mul(*d),
                    b.checked_mul(*c),
                    b.checked_mul(*d),
                ];

                // Find min/max without intermediate Vec allocation
                let mut lo = i128::MAX;
                let mut hi = i128::MIN;
                let mut valid_count = 0;
                for v in products.iter().flatten() {
                    lo = lo.min(*v);
                    hi = hi.max(*v);
                    valid_count += 1;
                }

                if valid_count < 4 {
                    // Overflow occurred
                    return Interval::Top;
                }

                Interval::Range { lo, hi }
            }
        }
    }

    /// Negation: -[a, b] = [-b, -a]
    pub fn neg(&self) -> Self {
        match self {
            Interval::Bottom => Interval::Bottom,
            Interval::Top => Interval::Top,
            Interval::Range { lo, hi } => {
                let new_lo = hi.checked_neg();
                let new_hi = lo.checked_neg();
                match (new_lo, new_hi) {
                    (Some(l), Some(h)) => Interval::Range { lo: l, hi: h },
                    _ => Interval::Top,
                }
            }
        }
    }

    /// Restrict interval to values less than bound: [a, b] ∩ (-∞, bound) = [a, min(b, bound-1)]
    pub fn restrict_lt(&self, bound: i128) -> Self {
        match self {
            Interval::Bottom => Interval::Bottom,
            Interval::Top => {
                if let Some(b) = bound.checked_sub(1) {
                    Interval::Range {
                        lo: i128::MIN,
                        hi: b,
                    }
                } else {
                    Interval::Bottom // bound is i128::MIN, nothing less
                }
            }
            Interval::Range { lo, hi } => {
                if let Some(b) = bound.checked_sub(1) {
                    if *lo > b {
                        Interval::Bottom
                    } else {
                        Interval::Range {
                            lo: *lo,
                            hi: (*hi).min(b),
                        }
                    }
                } else {
                    Interval::Bottom
                }
            }
        }
    }

    /// Restrict interval to values >= bound: [a, b] ∩ [bound, ∞) = [max(a, bound), b]
    pub fn restrict_geq(&self, bound: i128) -> Self {
        match self {
            Interval::Bottom => Interval::Bottom,
            Interval::Top => Interval::Range {
                lo: bound,
                hi: i128::MAX,
            },
            Interval::Range { lo, hi } => {
                if *hi < bound {
                    Interval::Bottom
                } else {
                    Interval::Range {
                        lo: (*lo).max(bound),
                        hi: *hi,
                    }
                }
            }
        }
    }
}

impl Lattice for Interval {
    #[inline]
    fn bottom() -> Self {
        Interval::Bottom
    }

    #[inline]
    fn top() -> Self {
        Interval::Top
    }

    #[inline]
    fn is_bottom(&self) -> bool {
        matches!(self, Interval::Bottom)
    }

    #[inline]
    fn is_top(&self) -> bool {
        matches!(self, Interval::Top)
    }

    #[inline]
    fn join(&self, other: &Self) -> Self {
        match (self, other) {
            (Interval::Bottom, x) | (x, Interval::Bottom) => x.clone(),
            (Interval::Top, _) | (_, Interval::Top) => Interval::Top,
            (Interval::Range { lo: a, hi: b }, Interval::Range { lo: c, hi: d }) => {
                Interval::Range {
                    lo: (*a).min(*c),
                    hi: (*b).max(*d),
                }
            }
        }
    }

    #[inline]
    fn meet(&self, other: &Self) -> Self {
        match (self, other) {
            (Interval::Top, x) | (x, Interval::Top) => x.clone(),
            (Interval::Bottom, _) | (_, Interval::Bottom) => Interval::Bottom,
            (Interval::Range { lo: a, hi: b }, Interval::Range { lo: c, hi: d }) => {
                let lo = (*a).max(*c);
                let hi = (*b).min(*d);
                if lo > hi {
                    Interval::Bottom
                } else {
                    Interval::Range { lo, hi }
                }
            }
        }
    }

    #[inline]
    fn partial_cmp_lattice(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Interval::Bottom, Interval::Bottom) | (Interval::Top, Interval::Top) => {
                Some(Ordering::Equal)
            }
            (Interval::Bottom, _) | (_, Interval::Top) => Some(Ordering::Less),
            (_, Interval::Bottom) | (Interval::Top, _) => Some(Ordering::Greater),
            (Interval::Range { lo: a, hi: b }, Interval::Range { lo: c, hi: d }) => {
                if a == c && b == d {
                    Some(Ordering::Equal)
                } else if a >= c && b <= d {
                    Some(Ordering::Less) // [a,b] ⊆ [c,d]
                } else if a <= c && b >= d {
                    Some(Ordering::Greater) // [c,d] ⊆ [a,b]
                } else {
                    None // Incomparable (overlapping but neither contains the other)
                }
            }
        }
    }

    /// Widening: if bounds grow, jump to infinity.
    /// This ensures termination for loops.
    #[inline]
    fn widen(&self, other: &Self) -> Self {
        match (self, other) {
            (Interval::Bottom, x) | (x, Interval::Bottom) => x.clone(),
            (Interval::Top, _) | (_, Interval::Top) => Interval::Top,
            (Interval::Range { lo: a, hi: b }, Interval::Range { lo: c, hi: d }) => {
                // If lower bound decreased, go to -∞
                let lo = if c < a { i128::MIN } else { *a };
                // If upper bound increased, go to +∞
                let hi = if d > b { i128::MAX } else { *b };

                if lo == i128::MIN && hi == i128::MAX {
                    Interval::Top
                } else {
                    Interval::Range { lo, hi }
                }
            }
        }
    }

    /// Narrowing: improve precision after widening.
    #[inline]
    fn narrow(&self, other: &Self) -> Self {
        match (self, other) {
            (Interval::Bottom, _) | (_, Interval::Bottom) => Interval::Bottom,
            (x, Interval::Top) | (Interval::Top, x) => x.clone(),
            (Interval::Range { lo: a, hi: b }, Interval::Range { lo: c, hi: d }) => {
                // If we went to -∞, use the other bound
                let lo = if *a == i128::MIN { *c } else { *a };
                // If we went to +∞, use the other bound
                let hi = if *b == i128::MAX { *d } else { *b };
                Interval::Range { lo, hi }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_creation() {
        let i = Interval::new(0, 10);
        assert!(i.contains(0));
        assert!(i.contains(5));
        assert!(i.contains(10));
        assert!(!i.contains(-1));
        assert!(!i.contains(11));
    }

    #[test]
    fn test_empty_interval() {
        let i = Interval::new(10, 5); // Invalid range
        assert!(i.is_bottom());
    }

    #[test]
    fn test_singleton() {
        let i = Interval::singleton(42);
        assert!(i.contains(42));
        assert!(!i.contains(41));
        assert!(!i.contains(43));
    }

    #[test]
    fn test_join() {
        let a = Interval::new(0, 5);
        let b = Interval::new(3, 10);
        let joined = a.join(&b);

        assert_eq!(joined, Interval::new(0, 10));
    }

    #[test]
    fn test_meet() {
        let a = Interval::new(0, 10);
        let b = Interval::new(5, 15);
        let met = a.meet(&b);

        assert_eq!(met, Interval::new(5, 10));
    }

    #[test]
    fn test_meet_empty() {
        let a = Interval::new(0, 5);
        let b = Interval::new(10, 15);
        let met = a.meet(&b);

        assert!(met.is_bottom());
    }

    #[test]
    fn test_addition() {
        let a = Interval::new(1, 5);
        let b = Interval::new(10, 20);
        let sum = a.add(&b);

        assert_eq!(sum, Interval::new(11, 25));
    }

    #[test]
    fn test_subtraction() {
        let a = Interval::new(10, 20);
        let b = Interval::new(1, 5);
        let diff = a.sub(&b);

        assert_eq!(diff, Interval::new(5, 19));
    }

    #[test]
    fn test_multiplication() {
        let a = Interval::new(2, 3);
        let b = Interval::new(4, 5);
        let prod = a.mul(&b);

        assert_eq!(prod, Interval::new(8, 15));
    }

    #[test]
    fn test_multiplication_mixed_signs() {
        let a = Interval::new(-2, 3);
        let b = Interval::new(-1, 2);
        let prod = a.mul(&b);

        // Products: 2, -4, -3, 6 → [-4, 6]
        assert_eq!(prod, Interval::new(-4, 6));
    }

    #[test]
    fn test_widening() {
        let a = Interval::new(0, 10);
        let b = Interval::new(0, 20); // Upper bound increased
        let widened = a.widen(&b);

        // Widening should push upper bound to infinity
        assert_eq!(
            widened,
            Interval::Range {
                lo: 0,
                hi: i128::MAX
            }
        );
    }

    #[test]
    fn test_widening_both_bounds() {
        let a = Interval::new(0, 10);
        let b = Interval::new(-5, 20);
        let widened = a.widen(&b);

        assert!(widened.is_top());
    }

    #[test]
    fn test_restrict_lt() {
        let i = Interval::new(0, 10);
        let restricted = i.restrict_lt(5);
        assert_eq!(restricted, Interval::new(0, 4));
    }

    #[test]
    fn test_restrict_geq() {
        let i = Interval::new(0, 10);
        let restricted = i.restrict_geq(5);
        assert_eq!(restricted, Interval::new(5, 10));
    }

    #[test]
    fn test_all_less_than() {
        let i = Interval::new(0, 9);
        assert!(i.all_less_than(10));
        assert!(!i.all_less_than(9));
    }

    #[test]
    fn test_ordering() {
        let a = Interval::new(2, 8);
        let b = Interval::new(0, 10);

        assert!(a.leq(&b)); // [2,8] ⊆ [0,10]
        assert!(!b.leq(&a));
    }
}
