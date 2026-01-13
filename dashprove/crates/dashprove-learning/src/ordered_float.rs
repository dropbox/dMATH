//! Ordered floating-point helpers with total ordering semantics
//!
//! Provides `Ord` wrappers around `f32` and `f64` that treat NaN as less than
//! all other values, enabling use in data structures like `BinaryHeap`.

use std::cmp::Ordering;

/// A totally ordered wrapper for `f32`
///
/// NaN values compare less than all finite values and equal to other NaNs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrderedF32(pub f32);

impl Eq for OrderedF32 {}

impl PartialOrd for OrderedF32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF32 {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.0.partial_cmp(&other.0) {
            Some(ordering) => ordering,
            None => {
                if self.0.is_nan() {
                    if other.0.is_nan() {
                        Ordering::Equal
                    } else {
                        Ordering::Less
                    }
                } else {
                    Ordering::Greater
                }
            }
        }
    }
}

impl OrderedF32 {
    /// Extract the inner `f32` value.
    pub fn into_inner(self) -> f32 {
        self.0
    }
}

/// A totally ordered wrapper for `f64`
///
/// NaN values compare less than all finite values and equal to other NaNs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrderedF64(pub f64);

impl Eq for OrderedF64 {}

impl PartialOrd for OrderedF64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.0.partial_cmp(&other.0) {
            Some(ordering) => ordering,
            None => {
                if self.0.is_nan() {
                    if other.0.is_nan() {
                        Ordering::Equal
                    } else {
                        Ordering::Less
                    }
                } else {
                    Ordering::Greater
                }
            }
        }
    }
}

impl OrderedF64 {
    /// Extract the inner `f64` value.
    pub fn into_inner(self) -> f64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BinaryHeap;

    #[test]
    fn test_ordered_f64_normal_comparison() {
        let a = OrderedF64(0.5);
        let b = OrderedF64(0.8);

        assert_eq!(a.cmp(&b), Ordering::Less);
        assert_eq!(b.cmp(&a), Ordering::Greater);
        assert_eq!(a.cmp(&a), Ordering::Equal);
    }

    #[test]
    fn test_ordered_f64_nan_comparison() {
        let nan = OrderedF64(f64::NAN);
        let normal = OrderedF64(0.5);
        let zero = OrderedF64(0.0);
        let neg = OrderedF64(-1.0);

        // NaN is less than all finite values
        assert_eq!(nan.cmp(&normal), Ordering::Less);
        assert_eq!(nan.cmp(&zero), Ordering::Less);
        assert_eq!(nan.cmp(&neg), Ordering::Less);

        // All finite values are greater than NaN
        assert_eq!(normal.cmp(&nan), Ordering::Greater);
        assert_eq!(zero.cmp(&nan), Ordering::Greater);
        assert_eq!(neg.cmp(&nan), Ordering::Greater);

        // NaN equals NaN
        assert_eq!(nan.cmp(&nan), Ordering::Equal);
    }

    #[test]
    fn test_ordered_f64_infinity_comparison() {
        let pos_inf = OrderedF64(f64::INFINITY);
        let neg_inf = OrderedF64(f64::NEG_INFINITY);
        let normal = OrderedF64(0.5);
        let nan = OrderedF64(f64::NAN);

        assert_eq!(pos_inf.cmp(&normal), Ordering::Greater);
        assert_eq!(neg_inf.cmp(&normal), Ordering::Less);
        assert_eq!(pos_inf.cmp(&neg_inf), Ordering::Greater);

        // NaN is still less than infinities
        assert_eq!(nan.cmp(&pos_inf), Ordering::Less);
        assert_eq!(nan.cmp(&neg_inf), Ordering::Less);
    }

    #[test]
    fn test_ordered_f64_partial_ord_consistency() {
        let a = OrderedF64(0.5);
        let b = OrderedF64(0.8);
        let nan = OrderedF64(f64::NAN);

        // partial_cmp should return Some(cmp result)
        assert_eq!(a.partial_cmp(&b), Some(Ordering::Less));
        assert_eq!(b.partial_cmp(&a), Some(Ordering::Greater));
        assert_eq!(a.partial_cmp(&a), Some(Ordering::Equal));
        assert_eq!(nan.partial_cmp(&a), Some(Ordering::Less));
        assert_eq!(nan.partial_cmp(&nan), Some(Ordering::Equal));
    }

    #[test]
    fn test_ordered_f64_into_inner() {
        let x = OrderedF64(1.234);
        assert!((x.into_inner() - 1.234).abs() < f64::EPSILON);

        let nan = OrderedF64(f64::NAN);
        assert!(nan.into_inner().is_nan());
    }

    #[test]
    fn test_ordered_f64_eq() {
        let a = OrderedF64(0.5);
        let b = OrderedF64(0.5);
        let c = OrderedF64(0.8);

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_ordered_f64_clone_copy() {
        let a = OrderedF64(0.5);
        let b = a; // Copy

        assert_eq!(a, b);
    }

    #[test]
    fn test_ordered_f64_debug() {
        let a = OrderedF64(0.5);
        let debug_str = format!("{:?}", a);
        assert!(debug_str.contains("0.5"));
    }

    #[test]
    fn test_ordered_f64_in_binary_heap() {
        // Max-heap by default
        let mut heap: BinaryHeap<OrderedF64> = BinaryHeap::new();

        heap.push(OrderedF64(0.3));
        heap.push(OrderedF64(0.1));
        heap.push(OrderedF64(0.9));
        heap.push(OrderedF64(f64::NAN));
        heap.push(OrderedF64(0.5));

        // Pop in descending order (max-heap)
        let first = heap.pop().unwrap();
        assert!((first.into_inner() - 0.9).abs() < f64::EPSILON);

        let second = heap.pop().unwrap();
        assert!((second.into_inner() - 0.5).abs() < f64::EPSILON);

        // NaN should come last (it's the minimum)
        let mut remaining: Vec<_> = heap.into_iter().map(|x| x.into_inner()).collect();
        remaining.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less));
        assert!(!remaining.last().map(|x| x.is_nan()).unwrap_or(false));
    }

    #[test]
    fn test_ordered_f64_min_heap_with_reverse() {
        use std::cmp::Reverse;

        // Min-heap using Reverse
        let mut heap: BinaryHeap<Reverse<OrderedF64>> = BinaryHeap::new();

        heap.push(Reverse(OrderedF64(0.3)));
        heap.push(Reverse(OrderedF64(0.1)));
        heap.push(Reverse(OrderedF64(0.9)));
        heap.push(Reverse(OrderedF64(f64::NAN)));
        heap.push(Reverse(OrderedF64(0.5)));

        // Pop in ascending order (min-heap)
        // NaN is less than all, so it comes first
        let first = heap.pop().unwrap().0;
        assert!(first.into_inner().is_nan());

        let second = heap.pop().unwrap().0;
        assert!((second.into_inner() - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ordered_f64_sorting() {
        let mut values = [
            OrderedF64(0.5),
            OrderedF64(f64::NAN),
            OrderedF64(-1.0),
            OrderedF64(f64::INFINITY),
            OrderedF64(0.0),
            OrderedF64(f64::NEG_INFINITY),
        ];

        values.sort();

        // NaN should be first (smallest)
        assert!(values[0].into_inner().is_nan());
        // NEG_INFINITY second
        assert_eq!(values[1].into_inner(), f64::NEG_INFINITY);
        // Then -1.0, 0.0, 0.5
        assert!((values[2].into_inner() - (-1.0)).abs() < f64::EPSILON);
        assert!((values[3].into_inner() - 0.0).abs() < f64::EPSILON);
        assert!((values[4].into_inner() - 0.5).abs() < f64::EPSILON);
        // INFINITY last
        assert_eq!(values[5].into_inner(), f64::INFINITY);
    }

    #[test]
    fn test_ordered_f64_top_k_pattern() {
        use std::cmp::Reverse;

        // Simulate top-k selection pattern used in similarity search
        let data = [0.2, 0.8, 0.1, 0.9, 0.5, 0.3, f64::NAN, 0.7];
        let k = 3;

        let mut heap: BinaryHeap<Reverse<OrderedF64>> = BinaryHeap::with_capacity(k + 1);

        for &score in &data {
            let ordered = OrderedF64(score);
            if heap.len() < k {
                heap.push(Reverse(ordered));
            } else if let Some(Reverse(min)) = heap.peek() {
                if ordered > *min {
                    heap.pop();
                    heap.push(Reverse(ordered));
                }
            }
        }

        // Should have top 3: 0.9, 0.8, 0.7
        let mut results: Vec<f64> = heap.into_iter().map(|Reverse(x)| x.into_inner()).collect();
        results.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));

        assert_eq!(results.len(), 3);
        assert!((results[0] - 0.9).abs() < f64::EPSILON);
        assert!((results[1] - 0.8).abs() < f64::EPSILON);
        assert!((results[2] - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ordered_f64_negative_zero() {
        let pos_zero = OrderedF64(0.0);
        let neg_zero = OrderedF64(-0.0);

        // Per IEEE 754, -0.0 == 0.0
        assert_eq!(pos_zero.cmp(&neg_zero), Ordering::Equal);
    }

    #[test]
    fn test_ordered_f64_subnormal() {
        let tiny = OrderedF64(f64::MIN_POSITIVE / 2.0); // Subnormal
        let normal = OrderedF64(f64::MIN_POSITIVE);
        let zero = OrderedF64(0.0);

        assert_eq!(tiny.cmp(&normal), Ordering::Less);
        assert_eq!(tiny.cmp(&zero), Ordering::Greater);
    }
}
