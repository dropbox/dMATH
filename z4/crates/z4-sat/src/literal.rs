//! Literal and variable representation

/// A variable identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(kani, derive(kani::Arbitrary))]
pub struct Variable(pub u32);

/// A literal (variable with polarity)
///
/// Encoded as: positive literal = 2*var, negative literal = 2*var + 1
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(kani, derive(kani::Arbitrary))]
pub struct Literal(pub u32);

impl Literal {
    /// Create a positive literal
    #[inline]
    pub fn positive(var: Variable) -> Self {
        Literal(var.0 << 1)
    }

    /// Create a negative literal
    #[inline]
    pub fn negative(var: Variable) -> Self {
        Literal((var.0 << 1) | 1)
    }

    /// Get the variable
    #[inline]
    pub fn variable(self) -> Variable {
        Variable(self.0 >> 1)
    }

    /// Check if positive
    #[inline]
    pub fn is_positive(self) -> bool {
        (self.0 & 1) == 0
    }

    /// Get the negation
    #[inline]
    pub fn negated(self) -> Self {
        Literal(self.0 ^ 1)
    }

    /// Get the index for watched literal arrays
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }

    /// Create a literal from its index
    ///
    /// This is the inverse of `index()`.
    #[inline]
    pub fn from_index(idx: usize) -> Self {
        Literal(idx as u32)
    }
}

impl Variable {
    /// Get the index
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

// ============================================================================
// Kani Verification Harnesses
// ============================================================================

#[cfg(kani)]
mod verification {
    use super::*;

    /// Negation is involutive: negating twice returns the original literal
    #[kani::proof]
    fn literal_negation_involutive() {
        let lit: Literal = kani::any();
        // Bound for tractability (half of u32::MAX due to encoding)
        kani::assume(lit.0 < 1_000_000);
        assert_eq!(lit.negated().negated(), lit);
    }

    /// Variable roundtrip: creating positive/negative literals preserves variable
    #[kani::proof]
    fn literal_variable_roundtrip() {
        let var: Variable = kani::any();
        // Bound to prevent overflow in shift operation
        kani::assume(var.0 < 500_000);

        let pos = Literal::positive(var);
        let neg = Literal::negative(var);

        assert_eq!(pos.variable(), var);
        assert_eq!(neg.variable(), var);
        assert!(pos.is_positive());
        assert!(!neg.is_positive());
    }

    /// Encoding uniqueness: different variables have different literal encodings
    #[kani::proof]
    fn literal_encoding_unique() {
        let var1: Variable = kani::any();
        let var2: Variable = kani::any();
        kani::assume(var1.0 < 500_000 && var2.0 < 500_000);

        let pos1 = Literal::positive(var1);
        let pos2 = Literal::positive(var2);

        // Same encoding implies same variable
        if pos1.0 == pos2.0 {
            assert_eq!(var1, var2);
        }
    }

    /// Positive and negative literals for the same variable are different
    #[kani::proof]
    fn literal_polarity_distinct() {
        let var: Variable = kani::any();
        kani::assume(var.0 < 500_000);

        let pos = Literal::positive(var);
        let neg = Literal::negative(var);

        assert_ne!(pos, neg);
        assert_eq!(pos.negated(), neg);
        assert_eq!(neg.negated(), pos);
    }

    /// Index is consistent with encoding
    #[kani::proof]
    fn literal_index_consistent() {
        let var: Variable = kani::any();
        kani::assume(var.0 < 500_000);

        let pos = Literal::positive(var);
        let neg = Literal::negative(var);

        // Indices should be consecutive: pos = 2*var, neg = 2*var + 1
        assert_eq!(pos.index(), (var.0 as usize) * 2);
        assert_eq!(neg.index(), (var.0 as usize) * 2 + 1);
    }
}

// ============================================================================
// Property Tests (proptest)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Negation is involutive
        #[test]
        fn prop_negation_involutive(var_idx in 0u32..100_000) {
            let var = Variable(var_idx);
            let pos = Literal::positive(var);
            let neg = Literal::negative(var);

            prop_assert_eq!(pos.negated().negated(), pos);
            prop_assert_eq!(neg.negated().negated(), neg);
        }

        /// Variable extraction is correct
        #[test]
        fn prop_variable_extraction(var_idx in 0u32..100_000) {
            let var = Variable(var_idx);
            let pos = Literal::positive(var);
            let neg = Literal::negative(var);

            prop_assert_eq!(pos.variable(), var);
            prop_assert_eq!(neg.variable(), var);
        }

        /// Polarity is correct
        #[test]
        fn prop_polarity_correct(var_idx in 0u32..100_000) {
            let var = Variable(var_idx);
            let pos = Literal::positive(var);
            let neg = Literal::negative(var);

            prop_assert!(pos.is_positive());
            prop_assert!(!neg.is_positive());
        }

        /// Positive and negative are distinct
        #[test]
        fn prop_polarity_distinct(var_idx in 0u32..100_000) {
            let var = Variable(var_idx);
            let pos = Literal::positive(var);
            let neg = Literal::negative(var);

            prop_assert_ne!(pos, neg);
            prop_assert_eq!(pos.negated(), neg);
            prop_assert_eq!(neg.negated(), pos);
        }

        /// Index is consistent
        #[test]
        fn prop_index_consistent(var_idx in 0u32..100_000) {
            let var = Variable(var_idx);
            let pos = Literal::positive(var);
            let neg = Literal::negative(var);

            prop_assert_eq!(pos.index(), (var_idx as usize) * 2);
            prop_assert_eq!(neg.index(), (var_idx as usize) * 2 + 1);
        }
    }

    #[test]
    fn test_literal_basic() {
        let var = Variable(5);
        let pos = Literal::positive(var);
        let neg = Literal::negative(var);

        assert_eq!(pos.variable(), var);
        assert_eq!(neg.variable(), var);
        assert!(pos.is_positive());
        assert!(!neg.is_positive());
        assert_eq!(pos.negated(), neg);
        assert_eq!(neg.negated(), pos);
    }

    #[test]
    fn test_variable_zero() {
        let var = Variable(0);
        let pos = Literal::positive(var);
        let neg = Literal::negative(var);

        assert_eq!(pos.0, 0);
        assert_eq!(neg.0, 1);
        assert_eq!(pos.variable(), var);
        assert_eq!(neg.variable(), var);
    }
}
