//! Core lattice operations for abstract interpretation.
//!
//! A lattice is a partially ordered set where every pair of elements has:
//! - A least upper bound (join, ⊔) - represents combining information from branches
//! - A greatest lower bound (meet, ⊓) - represents intersection of constraints
//!
//! Special elements:
//! - Bottom (⊥) - no information, unreachable code
//! - Top (⊤) - all possible values, unknown

use std::cmp::Ordering;

/// A lattice element with join, meet, and ordering operations.
pub trait Lattice: Clone + PartialEq + Sized {
    /// The bottom element (⊥) - represents unreachable/no information.
    fn bottom() -> Self;

    /// The top element (⊤) - represents all possible values.
    fn top() -> Self;

    /// Check if this is the bottom element.
    fn is_bottom(&self) -> bool;

    /// Check if this is the top element.
    fn is_top(&self) -> bool;

    /// Least upper bound (join, ⊔).
    /// Used when merging information from different control flow paths.
    fn join(&self, other: &Self) -> Self;

    /// Greatest lower bound (meet, ⊓).
    /// Used when intersecting constraints.
    fn meet(&self, other: &Self) -> Self;

    /// Partial ordering: self ⊑ other means self is less than or equal to other.
    /// More precise ⊑ Less precise (bottom ⊑ everything ⊑ top)
    fn partial_cmp_lattice(&self, other: &Self) -> Option<Ordering>;

    /// Check if self ⊑ other
    #[inline]
    fn leq(&self, other: &Self) -> bool {
        matches!(
            self.partial_cmp_lattice(other),
            Some(Ordering::Less | Ordering::Equal)
        )
    }

    /// Widening operator for accelerating fixed-point computation.
    /// Default implementation is just join, but domains should override
    /// this to ensure termination on infinite ascending chains.
    #[inline]
    fn widen(&self, other: &Self) -> Self {
        self.join(other)
    }

    /// Narrowing operator for improving precision after widening.
    /// Default implementation is just meet.
    #[inline]
    fn narrow(&self, other: &Self) -> Self {
        self.meet(other)
    }
}

/// A flat lattice lifts any set into a lattice with three levels:
/// - Bottom (⊥): undefined/unreachable
/// - Middle: the actual values
/// - Top (⊤): unknown/all values
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FlatLattice<T> {
    Bottom,
    Value(T),
    Top,
}

impl<T: Clone + PartialEq> Lattice for FlatLattice<T> {
    #[inline]
    fn bottom() -> Self {
        FlatLattice::Bottom
    }

    #[inline]
    fn top() -> Self {
        FlatLattice::Top
    }

    #[inline]
    fn is_bottom(&self) -> bool {
        matches!(self, FlatLattice::Bottom)
    }

    #[inline]
    fn is_top(&self) -> bool {
        matches!(self, FlatLattice::Top)
    }

    #[inline]
    fn join(&self, other: &Self) -> Self {
        match (self, other) {
            (FlatLattice::Bottom, x) | (x, FlatLattice::Bottom) => x.clone(),
            (FlatLattice::Top, _) | (_, FlatLattice::Top) => FlatLattice::Top,
            (FlatLattice::Value(a), FlatLattice::Value(b)) => {
                if a == b {
                    FlatLattice::Value(a.clone())
                } else {
                    FlatLattice::Top
                }
            }
        }
    }

    #[inline]
    fn meet(&self, other: &Self) -> Self {
        match (self, other) {
            (FlatLattice::Top, x) | (x, FlatLattice::Top) => x.clone(),
            (FlatLattice::Bottom, _) | (_, FlatLattice::Bottom) => FlatLattice::Bottom,
            (FlatLattice::Value(a), FlatLattice::Value(b)) => {
                if a == b {
                    FlatLattice::Value(a.clone())
                } else {
                    FlatLattice::Bottom
                }
            }
        }
    }

    #[inline]
    fn partial_cmp_lattice(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (FlatLattice::Bottom, FlatLattice::Bottom) | (FlatLattice::Top, FlatLattice::Top) => {
                Some(Ordering::Equal)
            }
            (FlatLattice::Bottom, _) | (_, FlatLattice::Top) => Some(Ordering::Less),
            (_, FlatLattice::Bottom) | (FlatLattice::Top, _) => Some(Ordering::Greater),
            (FlatLattice::Value(a), FlatLattice::Value(b)) => {
                if a == b {
                    Some(Ordering::Equal)
                } else {
                    None // Incomparable values
                }
            }
        }
    }
}

/// Product of two lattices - combines two domains.
#[derive(Debug, Clone, PartialEq)]
pub struct ProductLattice<A, B> {
    pub first: A,
    pub second: B,
}

impl<A: Lattice, B: Lattice> Lattice for ProductLattice<A, B> {
    #[inline]
    fn bottom() -> Self {
        ProductLattice {
            first: A::bottom(),
            second: B::bottom(),
        }
    }

    #[inline]
    fn top() -> Self {
        ProductLattice {
            first: A::top(),
            second: B::top(),
        }
    }

    #[inline]
    fn is_bottom(&self) -> bool {
        self.first.is_bottom() && self.second.is_bottom()
    }

    #[inline]
    fn is_top(&self) -> bool {
        self.first.is_top() && self.second.is_top()
    }

    #[inline]
    fn join(&self, other: &Self) -> Self {
        ProductLattice {
            first: self.first.join(&other.first),
            second: self.second.join(&other.second),
        }
    }

    #[inline]
    fn meet(&self, other: &Self) -> Self {
        ProductLattice {
            first: self.first.meet(&other.first),
            second: self.second.meet(&other.second),
        }
    }

    #[inline]
    fn partial_cmp_lattice(&self, other: &Self) -> Option<Ordering> {
        match (
            self.first.partial_cmp_lattice(&other.first),
            self.second.partial_cmp_lattice(&other.second),
        ) {
            (Some(Ordering::Equal), Some(Ordering::Equal)) => Some(Ordering::Equal),
            (Some(Ordering::Less | Ordering::Equal), Some(Ordering::Less | Ordering::Equal)) => {
                Some(Ordering::Less)
            }
            (
                Some(Ordering::Greater | Ordering::Equal),
                Some(Ordering::Greater | Ordering::Equal),
            ) => Some(Ordering::Greater),
            _ => None,
        }
    }

    #[inline]
    fn widen(&self, other: &Self) -> Self {
        ProductLattice {
            first: self.first.widen(&other.first),
            second: self.second.widen(&other.second),
        }
    }

    #[inline]
    fn narrow(&self, other: &Self) -> Self {
        ProductLattice {
            first: self.first.narrow(&other.first),
            second: self.second.narrow(&other.second),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_lattice_join() {
        let bot: FlatLattice<i32> = FlatLattice::Bottom;
        let val5 = FlatLattice::Value(5);
        let val10 = FlatLattice::Value(10);
        let top: FlatLattice<i32> = FlatLattice::Top;

        // Bottom is identity for join
        assert_eq!(bot.join(&val5), val5);
        assert_eq!(val5.join(&bot), val5);

        // Same values stay the same
        assert_eq!(val5.join(&val5.clone()), val5);

        // Different values go to top
        assert_eq!(val5.join(&val10), top);

        // Top absorbs everything
        assert_eq!(top.join(&val5), top);
        assert_eq!(val5.join(&top), top);
    }

    #[test]
    fn test_flat_lattice_meet() {
        let bot: FlatLattice<i32> = FlatLattice::Bottom;
        let val5 = FlatLattice::Value(5);
        let val10 = FlatLattice::Value(10);
        let top: FlatLattice<i32> = FlatLattice::Top;

        // Top is identity for meet
        assert_eq!(top.meet(&val5), val5);
        assert_eq!(val5.meet(&top), val5);

        // Same values stay the same
        assert_eq!(val5.meet(&val5.clone()), val5);

        // Different values go to bottom
        assert_eq!(val5.meet(&val10), bot);

        // Bottom absorbs everything
        assert_eq!(bot.meet(&val5), bot);
        assert_eq!(val5.meet(&bot), bot);
    }

    #[test]
    fn test_flat_lattice_ordering() {
        let bot: FlatLattice<i32> = FlatLattice::Bottom;
        let val5 = FlatLattice::Value(5);
        let val10 = FlatLattice::Value(10);
        let top: FlatLattice<i32> = FlatLattice::Top;

        // Bottom ⊑ everything
        assert!(bot.leq(&bot));
        assert!(bot.leq(&val5));
        assert!(bot.leq(&top));

        // Everything ⊑ top
        assert!(bot.leq(&top));
        assert!(val5.leq(&top));
        assert!(top.leq(&top));

        // Values: equal to themselves, incomparable otherwise
        assert!(val5.leq(&val5));
        assert!(!val5.leq(&val10));
        assert!(!val10.leq(&val5));
    }

    #[test]
    fn test_product_lattice() {
        type P = ProductLattice<FlatLattice<i32>, FlatLattice<bool>>;

        let bot = P::bottom();
        let top = P::top();

        let p1 = ProductLattice {
            first: FlatLattice::Value(5),
            second: FlatLattice::Value(true),
        };

        let p2 = ProductLattice {
            first: FlatLattice::Value(5),
            second: FlatLattice::Value(false),
        };

        // Bottom and top
        assert!(bot.is_bottom());
        assert!(top.is_top());

        // Join: different second components → second becomes top
        let joined = p1.join(&p2);
        assert_eq!(joined.first, FlatLattice::Value(5));
        assert_eq!(joined.second, FlatLattice::Top);
    }
}
