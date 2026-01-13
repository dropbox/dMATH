//! Model Minimization for Counterexample Generation
//!
//! This module provides model minimization to produce readable, debugging-friendly
//! counterexamples for tRust and other verification tools.
//!
//! ## Problem
//!
//! SMT solvers often return arbitrary satisfying values like `x = 847293847` when
//! a simpler value like `x = 0` would also satisfy the constraints. This makes
//! debugging verification failures difficult.
//!
//! ## Solution
//!
//! Post-SAT model minimization with preference ordering:
//! - Integers: 0 > ±1 > powers of 2 > small values > arbitrary
//! - Bitvectors: 0 > 1 > MAX > powers of 2 > small values > arbitrary
//! - Rationals: 0 > ±1 > simple fractions > arbitrary
//!
//! ## Usage
//!
//! ```ignore
//! use z4_dpll::minimize::{CounterexampleStyle, ModelMinimizer};
//!
//! let minimizer = ModelMinimizer::new(CounterexampleStyle::Minimal);
//! let minimal_model = minimizer.minimize(&original_model, &constraints);
//! ```

use hashbrown::HashMap;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use z4_core::TermId;

/// Style of counterexample to generate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CounterexampleStyle {
    /// Fast, current behavior - return any satisfying value
    Any,
    /// Prefer minimal values: 0, ±1, powers of 2, MIN, MAX
    #[default]
    Minimal,
    /// Prefer human-readable values: round numbers, simple fractions
    Readable,
}

/// Candidate values for integer minimization, in preference order
const INT_CANDIDATES: &[i64] = &[
    0,
    1,
    -1,
    2,
    -2,
    4,
    -4,
    8,
    -8,
    16,
    -16,
    32,
    -32,
    64,
    -64,
    128,
    -128,
    256,
    -256,
    i64::MAX,
    i64::MIN,
];

/// Model minimizer for producing readable counterexamples
pub struct ModelMinimizer {
    style: CounterexampleStyle,
}

impl ModelMinimizer {
    /// Create a new model minimizer with the specified style
    #[must_use]
    pub fn new(style: CounterexampleStyle) -> Self {
        ModelMinimizer { style }
    }

    /// Try to minimize an integer value while satisfying bounds
    ///
    /// # Arguments
    /// * `current` - The current value from the solver
    /// * `lower` - Optional lower bound (inclusive)
    /// * `upper` - Optional upper bound (inclusive)
    ///
    /// # Returns
    /// A simpler value that satisfies the bounds, or the original if no simpler value works
    #[must_use]
    pub fn minimize_int(
        &self,
        current: &BigInt,
        lower: Option<&BigInt>,
        upper: Option<&BigInt>,
    ) -> BigInt {
        if self.style == CounterexampleStyle::Any {
            return current.clone();
        }

        // Check if a value satisfies the bounds
        let satisfies = |v: &BigInt| -> bool {
            if let Some(lo) = lower {
                if v < lo {
                    return false;
                }
            }
            if let Some(hi) = upper {
                if v > hi {
                    return false;
                }
            }
            true
        };

        // Try candidates in preference order
        for &candidate in INT_CANDIDATES {
            let big_candidate = BigInt::from(candidate);
            if satisfies(&big_candidate) {
                return big_candidate;
            }
        }

        // For Readable style, try to round to nearest power of 10
        if self.style == CounterexampleStyle::Readable {
            if let Some(rounded) = round_to_readable(current) {
                if satisfies(&rounded) {
                    return rounded;
                }
            }
        }

        // Return original if no better candidate found
        current.clone()
    }

    /// Try to minimize a rational value while satisfying bounds
    #[must_use]
    pub fn minimize_rational(
        &self,
        current: &BigRational,
        lower: Option<&BigRational>,
        upper: Option<&BigRational>,
    ) -> BigRational {
        if self.style == CounterexampleStyle::Any {
            return current.clone();
        }

        let satisfies = |v: &BigRational| -> bool {
            if let Some(lo) = lower {
                if v < lo {
                    return false;
                }
            }
            if let Some(hi) = upper {
                if v > hi {
                    return false;
                }
            }
            true
        };

        // Try simple values first
        let candidates: Vec<BigRational> = vec![
            BigRational::zero(),
            BigRational::one(),
            -BigRational::one(),
            BigRational::new(1.into(), 2.into()),
            BigRational::new((-1).into(), 2.into()),
            BigRational::new(2.into(), 1.into()),
            BigRational::new((-2).into(), 1.into()),
        ];

        for candidate in &candidates {
            if satisfies(candidate) {
                return candidate.clone();
            }
        }

        // For Readable style, try to simplify to integer if close
        if self.style == CounterexampleStyle::Readable && current.is_integer() {
            let int_part = current.numer().clone();
            if let Some(rounded) = round_to_readable(&int_part) {
                let rational = BigRational::from(rounded);
                if satisfies(&rational) {
                    return rational;
                }
            }
        }

        current.clone()
    }

    /// Try to minimize a bitvector value while satisfying constraints
    ///
    /// # Arguments
    /// * `current` - The current value from the solver
    /// * `width` - Bitvector width in bits
    /// * `lower` - Optional lower bound (unsigned comparison)
    /// * `upper` - Optional upper bound (unsigned comparison)
    #[must_use]
    pub fn minimize_bitvec(
        &self,
        current: &BigInt,
        width: u32,
        lower: Option<&BigInt>,
        upper: Option<&BigInt>,
    ) -> BigInt {
        if self.style == CounterexampleStyle::Any {
            return current.clone();
        }

        let max_val = (BigInt::one() << width) - 1;

        let satisfies = |v: &BigInt| -> bool {
            if v.is_negative() || v > &max_val {
                return false;
            }
            if let Some(lo) = lower {
                if v < lo {
                    return false;
                }
            }
            if let Some(hi) = upper {
                if v > hi {
                    return false;
                }
            }
            true
        };

        // Try candidates in preference order for bitvectors
        let bv_candidates: Vec<BigInt> = vec![
            BigInt::zero(),
            BigInt::one(),
            max_val.clone(),              // all 1s
            BigInt::one() << (width - 1), // sign bit only (for signed interpretation)
            BigInt::from(2),
            BigInt::from(4),
            BigInt::from(8),
            BigInt::from(16),
            BigInt::from(255),             // 0xFF
            BigInt::from(256),             // 0x100
            BigInt::from(0xFFFF_u32),      // 16-bit max
            BigInt::from(0xFFFF_FFFF_u64), // 32-bit max
        ];

        for candidate in &bv_candidates {
            if satisfies(candidate) {
                return candidate.clone();
            }
        }

        // Try powers of 2 up to width
        for i in 0..width {
            let pow2 = BigInt::one() << i;
            if satisfies(&pow2) {
                return pow2;
            }
            // Also try (power of 2) - 1 (all 1s up to bit i)
            let pow2_minus_1 = &pow2 - 1;
            if satisfies(&pow2_minus_1) {
                return pow2_minus_1;
            }
        }

        current.clone()
    }

    /// Minimize a model with bounds collected from constraints
    ///
    /// This takes a model and tries to replace each value with a simpler one
    /// that still satisfies the constraints.
    #[must_use]
    pub fn minimize_model(
        &self,
        int_values: &HashMap<TermId, BigInt>,
        int_bounds: &HashMap<TermId, (Option<BigInt>, Option<BigInt>)>,
    ) -> HashMap<TermId, BigInt> {
        if self.style == CounterexampleStyle::Any {
            return int_values.clone();
        }

        let mut result = HashMap::new();

        for (&term_id, value) in int_values {
            let (lower, upper) = int_bounds
                .get(&term_id)
                .map(|(l, u)| (l.as_ref(), u.as_ref()))
                .unwrap_or((None, None));

            let minimized = self.minimize_int(value, lower, upper);
            result.insert(term_id, minimized);
        }

        result
    }
}

/// Round a BigInt to a "readable" value (nearest power of 10 or simple number)
fn round_to_readable(value: &BigInt) -> Option<BigInt> {
    // For small values, just return as-is
    if value.abs() < BigInt::from(100) {
        return Some(value.clone());
    }

    // Try rounding to nearest power of 10
    let abs_val = value.abs();
    let digits = abs_val.to_string().len();

    // Round to 1, 2, or 5 * 10^(digits-1)
    let base: BigInt = BigInt::from(10).pow((digits - 1) as u32);

    let candidates: Vec<BigInt> = vec![&base * 1, &base * 2, &base * 5, &base * 10];

    for candidate in candidates {
        let signed = if value.is_negative() {
            -&candidate
        } else {
            candidate.clone()
        };
        // Return if reasonably close (within 50% of original)
        let diff = (value - &signed).abs();
        if &diff * 2 <= abs_val {
            return Some(signed);
        }
    }

    None
}

/// Unset don't-care variables in a SAT model
///
/// Variables that don't affect the satisfiability of the formula
/// can be set to any value. This function identifies and marks them.
///
/// # Arguments
/// * `model` - The current SAT model (mutable)
/// * `original_clauses` - The original problem clauses
/// * `num_vars` - Number of variables in the problem
///
/// # Returns
/// A vector indicating which variables are don't-cares (true = don't care)
pub fn find_dont_cares(
    model: &[bool],
    original_clauses: &[Vec<i32>],
    num_vars: usize,
) -> Vec<bool> {
    let mut dont_care = vec![true; num_vars];

    // A variable is NOT a don't-care if it's the only satisfying literal
    // in at least one clause
    for clause in original_clauses {
        let satisfying_lits: Vec<_> = clause
            .iter()
            .filter(|&&lit| {
                let var = (lit.unsigned_abs() - 1) as usize;
                if var >= model.len() {
                    return false;
                }
                let is_positive = lit > 0;
                model[var] == is_positive
            })
            .collect();

        // If exactly one literal satisfies this clause, that variable is needed
        if satisfying_lits.len() == 1 {
            let var = (satisfying_lits[0].unsigned_abs() - 1) as usize;
            if var < dont_care.len() {
                dont_care[var] = false;
            }
        }
    }

    dont_care
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimize_int_unconstrained() {
        let minimizer = ModelMinimizer::new(CounterexampleStyle::Minimal);

        // Unconstrained: should return 0
        let result = minimizer.minimize_int(&BigInt::from(847293847), None, None);
        assert_eq!(result, BigInt::zero());
    }

    #[test]
    fn test_minimize_int_lower_bound() {
        let minimizer = ModelMinimizer::new(CounterexampleStyle::Minimal);

        // Lower bound of 5: should return 8 (smallest power of 2 >= 5)
        let result = minimizer.minimize_int(&BigInt::from(100), Some(&BigInt::from(5)), None);
        assert_eq!(result, BigInt::from(8));
    }

    #[test]
    fn test_minimize_int_upper_bound() {
        let minimizer = ModelMinimizer::new(CounterexampleStyle::Minimal);

        // Upper bound of -5: should return -8 (largest power of 2 <= -5)
        let result = minimizer.minimize_int(&BigInt::from(-100), None, Some(&BigInt::from(-5)));
        assert_eq!(result, BigInt::from(-8));
    }

    #[test]
    fn test_minimize_int_both_bounds() {
        let minimizer = ModelMinimizer::new(CounterexampleStyle::Minimal);

        // Between 3 and 7: should return 4 (power of 2 in range)
        let result = minimizer.minimize_int(
            &BigInt::from(5),
            Some(&BigInt::from(3)),
            Some(&BigInt::from(7)),
        );
        assert_eq!(result, BigInt::from(4));
    }

    #[test]
    fn test_minimize_int_tight_bounds() {
        let minimizer = ModelMinimizer::new(CounterexampleStyle::Minimal);

        // Between 42 and 42: no simpler value possible
        let result = minimizer.minimize_int(
            &BigInt::from(42),
            Some(&BigInt::from(42)),
            Some(&BigInt::from(42)),
        );
        assert_eq!(result, BigInt::from(42));
    }

    #[test]
    fn test_minimize_bitvec_unconstrained() {
        let minimizer = ModelMinimizer::new(CounterexampleStyle::Minimal);

        // 8-bit BV, unconstrained: should return 0
        let result = minimizer.minimize_bitvec(&BigInt::from(0xAB), 8, None, None);
        assert_eq!(result, BigInt::zero());
    }

    #[test]
    fn test_minimize_bitvec_nonzero() {
        let minimizer = ModelMinimizer::new(CounterexampleStyle::Minimal);

        // 8-bit BV, must be nonzero: should return 1
        let result = minimizer.minimize_bitvec(&BigInt::from(0xAB), 8, Some(&BigInt::one()), None);
        assert_eq!(result, BigInt::one());
    }

    #[test]
    fn test_minimize_bitvec_max() {
        let minimizer = ModelMinimizer::new(CounterexampleStyle::Minimal);

        // 8-bit BV, must be >= 200: should return 255 (max)
        let result =
            minimizer.minimize_bitvec(&BigInt::from(234), 8, Some(&BigInt::from(200)), None);
        assert_eq!(result, BigInt::from(255));
    }

    #[test]
    fn test_minimize_rational_unconstrained() {
        let minimizer = ModelMinimizer::new(CounterexampleStyle::Minimal);

        let weird_rational = BigRational::new(12345.into(), 6789.into());
        let result = minimizer.minimize_rational(&weird_rational, None, None);
        assert_eq!(result, BigRational::zero());
    }

    #[test]
    fn test_minimize_rational_positive() {
        let minimizer = ModelMinimizer::new(CounterexampleStyle::Minimal);

        let weird_rational = BigRational::new(12345.into(), 6789.into());
        let result = minimizer.minimize_rational(
            &weird_rational,
            Some(&BigRational::new(1.into(), 10.into())),
            None,
        );
        // Should return 1 (simplest value >= 0.1 in candidate list)
        // Note: candidates are checked in order: 0, 1, -1, 1/2, ...
        // and 1 >= 0.1, so it wins
        assert_eq!(result, BigRational::one());
    }

    #[test]
    fn test_style_any_returns_original() {
        let minimizer = ModelMinimizer::new(CounterexampleStyle::Any);

        let original = BigInt::from(847293847);
        let result = minimizer.minimize_int(&original, None, None);
        assert_eq!(result, original);
    }

    #[test]
    fn test_find_dont_cares_simple() {
        // Model: [true, true, false]
        // Clauses: [[1, 2], [2, 3]]
        // Variable 2 satisfies both clauses alone, so it's not a don't-care
        // Variable 1 is a don't-care (clause 1 is also satisfied by var 2)
        // Variable 3 is a don't-care (clause 2 is also satisfied by var 2)

        let model = vec![true, true, false];
        let clauses = vec![vec![1, 2], vec![2, 3]];

        let dont_cares = find_dont_cares(&model, &clauses, 3);

        // Variable 2 (index 1) is not a don't-care
        // This is a simplified check - full analysis is more complex
        assert!(
            dont_cares.iter().any(|&x| !x),
            "Should have some non-don't-cares"
        );
    }

    #[test]
    fn test_find_dont_cares_all_needed() {
        // Model: [true, true]
        // Clauses: [[1], [2]]
        // Both variables are needed (each is the only satisfying literal in its clause)

        let model = vec![true, true];
        let clauses = vec![vec![1], vec![2]];

        let dont_cares = find_dont_cares(&model, &clauses, 2);

        assert!(!dont_cares[0], "Variable 1 should not be a don't-care");
        assert!(!dont_cares[1], "Variable 2 should not be a don't-care");
    }
}
