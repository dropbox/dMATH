//! Nullability domain for tracking Option/Result types.
//!
//! This domain tracks whether a value is definitely Some, definitely None,
//! or possibly either. Essential for proving .unwrap() calls are safe.

use std::cmp::Ordering;

use crate::lattice::Lattice;

/// Nullability state for Option-like types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Nullability {
    /// Bottom: unreachable/no information
    Bottom,
    /// Definitely None/null
    None,
    /// Definitely Some/non-null
    Some,
    /// Could be either (unknown)
    Maybe,
}

impl Nullability {
    /// Check if unwrap() is definitely safe.
    #[inline]
    pub fn is_safe_to_unwrap(&self) -> bool {
        matches!(self, Nullability::Some)
    }

    /// Check if unwrap() might panic.
    #[inline]
    pub fn might_panic_on_unwrap(&self) -> bool {
        matches!(self, Nullability::None | Nullability::Maybe)
    }

    /// Check if definitely None.
    #[inline]
    pub fn is_definitely_none(&self) -> bool {
        matches!(self, Nullability::None)
    }

    /// Check if definitely Some.
    #[inline]
    pub fn is_definitely_some(&self) -> bool {
        matches!(self, Nullability::Some)
    }
}

impl Lattice for Nullability {
    #[inline]
    fn bottom() -> Self {
        Nullability::Bottom
    }

    #[inline]
    fn top() -> Self {
        Nullability::Maybe
    }

    #[inline]
    fn is_bottom(&self) -> bool {
        matches!(self, Nullability::Bottom)
    }

    #[inline]
    fn is_top(&self) -> bool {
        matches!(self, Nullability::Maybe)
    }

    /// Join: merge information from different control flow paths.
    /// ```text
    ///       Maybe
    ///      /     \
    ///   Some     None
    ///      \     /
    ///       Bottom
    /// ```
    #[inline]
    fn join(&self, other: &Self) -> Self {
        match (self, other) {
            (Nullability::Bottom, x) | (x, Nullability::Bottom) => *x,
            (Nullability::Maybe, _) | (_, Nullability::Maybe) => Nullability::Maybe,
            (Nullability::Some, Nullability::Some) => Nullability::Some,
            (Nullability::None, Nullability::None) => Nullability::None,
            (Nullability::Some, Nullability::None) | (Nullability::None, Nullability::Some) => {
                Nullability::Maybe
            }
        }
    }

    /// Meet: intersection of constraints.
    #[inline]
    fn meet(&self, other: &Self) -> Self {
        match (self, other) {
            (Nullability::Maybe, x) | (x, Nullability::Maybe) => *x,
            (Nullability::Bottom, _) | (_, Nullability::Bottom) => Nullability::Bottom,
            (Nullability::Some, Nullability::Some) => Nullability::Some,
            (Nullability::None, Nullability::None) => Nullability::None,
            (Nullability::Some, Nullability::None) | (Nullability::None, Nullability::Some) => {
                Nullability::Bottom
            }
        }
    }

    #[inline]
    fn partial_cmp_lattice(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            // Equal cases
            (Nullability::Bottom, Nullability::Bottom)
            | (Nullability::Some, Nullability::Some)
            | (Nullability::None, Nullability::None)
            | (Nullability::Maybe, Nullability::Maybe) => Some(Ordering::Equal),
            // Bottom is less than everything, everything is less than Maybe
            (Nullability::Bottom, _) | (_, Nullability::Maybe) => Some(Ordering::Less),
            // Everything is greater than Bottom, Maybe is greater than everything
            (_, Nullability::Bottom) | (Nullability::Maybe, _) => Some(Ordering::Greater),
            // Some and None are incomparable
            (Nullability::Some, Nullability::None) | (Nullability::None, Nullability::Some) => None,
        }
    }
}

/// Extended nullability with tracking for Result types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResultState {
    /// Bottom: unreachable
    Bottom,
    /// Definitely Ok
    Ok,
    /// Definitely Err
    Err,
    /// Could be either
    Maybe,
}

impl ResultState {
    /// Check if .unwrap() is safe.
    #[inline]
    pub fn is_safe_to_unwrap(&self) -> bool {
        matches!(self, ResultState::Ok)
    }

    /// Check if .unwrap_err() is safe.
    #[inline]
    pub fn is_safe_to_unwrap_err(&self) -> bool {
        matches!(self, ResultState::Err)
    }
}

impl Lattice for ResultState {
    #[inline]
    fn bottom() -> Self {
        ResultState::Bottom
    }

    #[inline]
    fn top() -> Self {
        ResultState::Maybe
    }

    #[inline]
    fn is_bottom(&self) -> bool {
        matches!(self, ResultState::Bottom)
    }

    #[inline]
    fn is_top(&self) -> bool {
        matches!(self, ResultState::Maybe)
    }

    #[inline]
    fn join(&self, other: &Self) -> Self {
        match (self, other) {
            (ResultState::Bottom, x) | (x, ResultState::Bottom) => *x,
            (ResultState::Maybe, _) | (_, ResultState::Maybe) => ResultState::Maybe,
            (ResultState::Ok, ResultState::Ok) => ResultState::Ok,
            (ResultState::Err, ResultState::Err) => ResultState::Err,
            (ResultState::Ok, ResultState::Err) | (ResultState::Err, ResultState::Ok) => {
                ResultState::Maybe
            }
        }
    }

    #[inline]
    fn meet(&self, other: &Self) -> Self {
        match (self, other) {
            (ResultState::Maybe, x) | (x, ResultState::Maybe) => *x,
            (ResultState::Bottom, _) | (_, ResultState::Bottom) => ResultState::Bottom,
            (ResultState::Ok, ResultState::Ok) => ResultState::Ok,
            (ResultState::Err, ResultState::Err) => ResultState::Err,
            (ResultState::Ok, ResultState::Err) | (ResultState::Err, ResultState::Ok) => {
                ResultState::Bottom
            }
        }
    }

    #[inline]
    fn partial_cmp_lattice(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            // Equal cases
            (ResultState::Bottom, ResultState::Bottom)
            | (ResultState::Ok, ResultState::Ok)
            | (ResultState::Err, ResultState::Err)
            | (ResultState::Maybe, ResultState::Maybe) => Some(Ordering::Equal),
            // Bottom is less than everything, everything is less than Maybe
            (ResultState::Bottom, _) | (_, ResultState::Maybe) => Some(Ordering::Less),
            // Everything is greater than Bottom, Maybe is greater than everything
            (_, ResultState::Bottom) | (ResultState::Maybe, _) => Some(Ordering::Greater),
            // Ok and Err are incomparable
            (ResultState::Ok, ResultState::Err) | (ResultState::Err, ResultState::Ok) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nullability_join() {
        // Bottom is identity
        assert_eq!(
            Nullability::Bottom.join(&Nullability::Some),
            Nullability::Some
        );
        assert_eq!(
            Nullability::Some.join(&Nullability::Bottom),
            Nullability::Some
        );

        // Same values stay the same
        assert_eq!(
            Nullability::Some.join(&Nullability::Some),
            Nullability::Some
        );
        assert_eq!(
            Nullability::None.join(&Nullability::None),
            Nullability::None
        );

        // Different values become Maybe
        assert_eq!(
            Nullability::Some.join(&Nullability::None),
            Nullability::Maybe
        );

        // Maybe absorbs everything
        assert_eq!(
            Nullability::Maybe.join(&Nullability::Some),
            Nullability::Maybe
        );
    }

    #[test]
    fn test_nullability_meet() {
        // Maybe is identity for meet
        assert_eq!(
            Nullability::Maybe.meet(&Nullability::Some),
            Nullability::Some
        );

        // Same values stay the same
        assert_eq!(
            Nullability::Some.meet(&Nullability::Some),
            Nullability::Some
        );

        // Contradictions become Bottom
        assert_eq!(
            Nullability::Some.meet(&Nullability::None),
            Nullability::Bottom
        );
    }

    #[test]
    fn test_safe_to_unwrap() {
        assert!(Nullability::Some.is_safe_to_unwrap());
        assert!(!Nullability::None.is_safe_to_unwrap());
        assert!(!Nullability::Maybe.is_safe_to_unwrap());
        assert!(!Nullability::Bottom.is_safe_to_unwrap());
    }

    #[test]
    fn test_might_panic() {
        assert!(!Nullability::Some.might_panic_on_unwrap());
        assert!(Nullability::None.might_panic_on_unwrap());
        assert!(Nullability::Maybe.might_panic_on_unwrap());
        assert!(!Nullability::Bottom.might_panic_on_unwrap()); // unreachable
    }

    #[test]
    fn test_ordering() {
        assert!(Nullability::Bottom.leq(&Nullability::Some));
        assert!(Nullability::Bottom.leq(&Nullability::None));
        assert!(Nullability::Bottom.leq(&Nullability::Maybe));

        assert!(Nullability::Some.leq(&Nullability::Maybe));
        assert!(Nullability::None.leq(&Nullability::Maybe));

        // Some and None are incomparable
        assert!(Nullability::Some
            .partial_cmp_lattice(&Nullability::None)
            .is_none());
    }

    #[test]
    fn test_result_state() {
        assert!(ResultState::Ok.is_safe_to_unwrap());
        assert!(!ResultState::Err.is_safe_to_unwrap());
        assert!(!ResultState::Maybe.is_safe_to_unwrap());

        assert!(ResultState::Err.is_safe_to_unwrap_err());
        assert!(!ResultState::Ok.is_safe_to_unwrap_err());
    }
}
