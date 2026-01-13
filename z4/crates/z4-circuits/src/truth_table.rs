//! Truth table representation for Boolean functions.
//!
//! A truth table is an explicit representation of a Boolean function
//! f: {0,1}^n -> {0,1} as a vector of 2^n bits.

use std::fmt;

/// Truth table representation of a Boolean function.
///
/// For a function with n inputs, stores 2^n bits representing
/// f(0...0), f(0...01), f(0...10), ..., f(1...1)
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct TruthTable {
    /// Number of input variables
    num_inputs: usize,
    /// Truth table entries (bit i = f(binary representation of i))
    entries: Vec<bool>,
}

impl TruthTable {
    /// Create a truth table with all entries set to the given value.
    pub fn constant(num_inputs: usize, value: bool) -> Self {
        let size = 1 << num_inputs;
        TruthTable {
            num_inputs,
            entries: vec![value; size],
        }
    }

    /// Create a truth table from a function.
    ///
    /// The function takes a slice of bools representing the input assignment
    /// (index 0 = least significant bit) and returns the output.
    pub fn from_fn<F>(num_inputs: usize, f: F) -> Self
    where
        F: Fn(&[bool]) -> bool,
    {
        let size = 1 << num_inputs;
        let mut entries = Vec::with_capacity(size);

        for i in 0..size {
            let input: Vec<bool> = (0..num_inputs).map(|j| (i >> j) & 1 == 1).collect();
            entries.push(f(&input));
        }

        TruthTable {
            num_inputs,
            entries,
        }
    }

    /// Create a truth table from a vector of entries.
    ///
    /// Entry i represents f(binary representation of i).
    pub fn from_entries(num_inputs: usize, entries: Vec<bool>) -> Self {
        let expected_size = 1 << num_inputs;
        assert_eq!(
            entries.len(),
            expected_size,
            "Expected {} entries for {} inputs, got {}",
            expected_size,
            num_inputs,
            entries.len()
        );
        TruthTable {
            num_inputs,
            entries,
        }
    }

    /// Get the number of input variables.
    pub fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    /// Get the number of entries (2^num_inputs).
    pub fn num_entries(&self) -> usize {
        self.entries.len()
    }

    /// Evaluate the function on a given input.
    pub fn evaluate(&self, inputs: &[bool]) -> bool {
        assert_eq!(
            inputs.len(),
            self.num_inputs,
            "Wrong number of inputs: expected {}, got {}",
            self.num_inputs,
            inputs.len()
        );
        let index = inputs
            .iter()
            .enumerate()
            .fold(0, |acc, (i, &b)| acc | ((b as usize) << i));
        self.entries[index]
    }

    /// Get the raw entries.
    pub fn entries(&self) -> &[bool] {
        &self.entries
    }

    /// Check if this function is a constant.
    pub fn is_constant(&self) -> bool {
        if self.entries.is_empty() {
            return true;
        }
        let first = self.entries[0];
        self.entries.iter().all(|&e| e == first)
    }

    /// Check if this function is the constant true.
    pub fn is_true(&self) -> bool {
        self.entries.iter().all(|&e| e)
    }

    /// Check if this function is the constant false.
    pub fn is_false(&self) -> bool {
        self.entries.iter().all(|&e| !e)
    }

    /// Count the number of true entries (Hamming weight).
    pub fn weight(&self) -> usize {
        self.entries.iter().filter(|&&e| e).count()
    }

    /// Check if this function depends on input variable i.
    ///
    /// A function depends on x_i if there exist inputs that differ only in x_i
    /// and produce different outputs.
    pub fn depends_on(&self, var: usize) -> bool {
        assert!(var < self.num_inputs, "Variable index out of bounds");
        let bit = 1 << var;
        for i in 0..self.entries.len() {
            let j = i ^ bit;
            if i < j && self.entries[i] != self.entries[j] {
                return true;
            }
        }
        false
    }

    /// Return the set of variables this function depends on.
    pub fn essential_variables(&self) -> Vec<usize> {
        (0..self.num_inputs)
            .filter(|&i| self.depends_on(i))
            .collect()
    }

    /// Compute the negation of this function.
    pub fn negate(&self) -> TruthTable {
        TruthTable {
            num_inputs: self.num_inputs,
            entries: self.entries.iter().map(|&e| !e).collect(),
        }
    }

    /// Compute the AND of two functions.
    pub fn and(&self, other: &TruthTable) -> TruthTable {
        assert_eq!(
            self.num_inputs, other.num_inputs,
            "Functions must have same number of inputs"
        );
        TruthTable {
            num_inputs: self.num_inputs,
            entries: self
                .entries
                .iter()
                .zip(other.entries.iter())
                .map(|(&a, &b)| a && b)
                .collect(),
        }
    }

    /// Compute the OR of two functions.
    pub fn or(&self, other: &TruthTable) -> TruthTable {
        assert_eq!(
            self.num_inputs, other.num_inputs,
            "Functions must have same number of inputs"
        );
        TruthTable {
            num_inputs: self.num_inputs,
            entries: self
                .entries
                .iter()
                .zip(other.entries.iter())
                .map(|(&a, &b)| a || b)
                .collect(),
        }
    }

    /// Compute the XOR of two functions.
    pub fn xor(&self, other: &TruthTable) -> TruthTable {
        assert_eq!(
            self.num_inputs, other.num_inputs,
            "Functions must have same number of inputs"
        );
        TruthTable {
            num_inputs: self.num_inputs,
            entries: self
                .entries
                .iter()
                .zip(other.entries.iter())
                .map(|(&a, &b)| a ^ b)
                .collect(),
        }
    }

    // Common truth tables

    /// Identity function: returns input variable i.
    pub fn variable(num_inputs: usize, var: usize) -> Self {
        assert!(var < num_inputs, "Variable index out of bounds");
        TruthTable::from_fn(num_inputs, |inputs| inputs[var])
    }

    /// AND of all inputs.
    pub fn and_all(num_inputs: usize) -> Self {
        TruthTable::from_fn(num_inputs, |inputs| inputs.iter().all(|&b| b))
    }

    /// OR of all inputs.
    pub fn or_all(num_inputs: usize) -> Self {
        TruthTable::from_fn(num_inputs, |inputs| inputs.iter().any(|&b| b))
    }

    /// XOR (parity) of all inputs.
    pub fn parity(num_inputs: usize) -> Self {
        TruthTable::from_fn(num_inputs, |inputs| {
            inputs.iter().fold(false, |acc, &b| acc ^ b)
        })
    }

    /// Majority function: true iff more than half of inputs are true.
    pub fn majority(num_inputs: usize) -> Self {
        let threshold = num_inputs.div_ceil(2);
        TruthTable::from_fn(num_inputs, |inputs| {
            inputs.iter().filter(|&&b| b).count() >= threshold
        })
    }

    /// Threshold function: true iff at least k inputs are true.
    pub fn threshold(num_inputs: usize, k: usize) -> Self {
        TruthTable::from_fn(num_inputs, |inputs| {
            inputs.iter().filter(|&&b| b).count() >= k
        })
    }

    /// Mod function: true iff (number of true inputs) mod m == 0.
    pub fn mod_count(num_inputs: usize, m: usize) -> Self {
        TruthTable::from_fn(num_inputs, |inputs| {
            inputs.iter().filter(|&&b| b).count() % m == 0
        })
    }
}

impl fmt::Debug for TruthTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TruthTable({}, [", self.num_inputs)?;
        for (i, &e) in self.entries.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", if e { "1" } else { "0" })?;
        }
        write!(f, "])")
    }
}

impl fmt::Display for TruthTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Truth table ({} inputs):", self.num_inputs)?;
        for i in 0..self.entries.len() {
            let bits: String = (0..self.num_inputs)
                .map(|j| if (i >> j) & 1 == 1 { '1' } else { '0' })
                .collect();
            writeln!(
                f,
                "  {} -> {}",
                bits,
                if self.entries[i] { "1" } else { "0" }
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant() {
        let t = TruthTable::constant(2, true);
        assert!(t.evaluate(&[false, false]));
        assert!(t.evaluate(&[true, true]));
        assert!(t.is_constant());
        assert!(t.is_true());

        let f = TruthTable::constant(2, false);
        assert!(!f.evaluate(&[false, false]));
        assert!(f.is_constant());
        assert!(f.is_false());
    }

    #[test]
    fn test_variable() {
        let x0 = TruthTable::variable(2, 0);
        assert!(!x0.evaluate(&[false, false]));
        assert!(x0.evaluate(&[true, false]));
        assert!(!x0.evaluate(&[false, true]));
        assert!(x0.evaluate(&[true, true]));

        let x1 = TruthTable::variable(2, 1);
        assert!(!x1.evaluate(&[false, false]));
        assert!(!x1.evaluate(&[true, false]));
        assert!(x1.evaluate(&[false, true]));
        assert!(x1.evaluate(&[true, true]));
    }

    #[test]
    fn test_and_all() {
        let and = TruthTable::and_all(3);
        assert!(!and.evaluate(&[false, false, false]));
        assert!(!and.evaluate(&[true, true, false]));
        assert!(and.evaluate(&[true, true, true]));
    }

    #[test]
    fn test_or_all() {
        let or = TruthTable::or_all(3);
        assert!(!or.evaluate(&[false, false, false]));
        assert!(or.evaluate(&[true, false, false]));
        assert!(or.evaluate(&[true, true, true]));
    }

    #[test]
    fn test_parity() {
        let xor = TruthTable::parity(3);
        assert!(!xor.evaluate(&[false, false, false])); // 0
        assert!(xor.evaluate(&[true, false, false])); // 1
        assert!(xor.evaluate(&[false, true, false])); // 1
        assert!(!xor.evaluate(&[true, true, false])); // 0
        assert!(xor.evaluate(&[false, false, true])); // 1
        assert!(!xor.evaluate(&[true, false, true])); // 0
        assert!(!xor.evaluate(&[false, true, true])); // 0
        assert!(xor.evaluate(&[true, true, true])); // 1
    }

    #[test]
    fn test_majority() {
        let maj = TruthTable::majority(3);
        assert!(!maj.evaluate(&[false, false, false]));
        assert!(!maj.evaluate(&[true, false, false]));
        assert!(maj.evaluate(&[true, true, false]));
        assert!(maj.evaluate(&[true, true, true]));
    }

    #[test]
    fn test_depends_on() {
        let x0 = TruthTable::variable(3, 0);
        assert!(x0.depends_on(0));
        assert!(!x0.depends_on(1));
        assert!(!x0.depends_on(2));

        let and = TruthTable::and_all(3);
        assert!(and.depends_on(0));
        assert!(and.depends_on(1));
        assert!(and.depends_on(2));

        let c = TruthTable::constant(3, true);
        assert!(!c.depends_on(0));
        assert!(!c.depends_on(1));
        assert!(!c.depends_on(2));
    }

    #[test]
    fn test_essential_variables() {
        let x0 = TruthTable::variable(3, 0);
        assert_eq!(x0.essential_variables(), vec![0]);

        let and = TruthTable::and_all(3);
        assert_eq!(and.essential_variables(), vec![0, 1, 2]);
    }

    #[test]
    fn test_operations() {
        let x0 = TruthTable::variable(2, 0);
        let x1 = TruthTable::variable(2, 1);

        // x0 AND x1
        let and = x0.and(&x1);
        assert!(!and.evaluate(&[false, false]));
        assert!(!and.evaluate(&[true, false]));
        assert!(!and.evaluate(&[false, true]));
        assert!(and.evaluate(&[true, true]));

        // x0 OR x1
        let or = x0.or(&x1);
        assert!(!or.evaluate(&[false, false]));
        assert!(or.evaluate(&[true, false]));
        assert!(or.evaluate(&[false, true]));
        assert!(or.evaluate(&[true, true]));

        // x0 XOR x1
        let xor = x0.xor(&x1);
        assert!(!xor.evaluate(&[false, false]));
        assert!(xor.evaluate(&[true, false]));
        assert!(xor.evaluate(&[false, true]));
        assert!(!xor.evaluate(&[true, true]));
    }

    #[test]
    fn test_negate() {
        let x0 = TruthTable::variable(2, 0);
        let not_x0 = x0.negate();
        assert!(not_x0.evaluate(&[false, false]));
        assert!(!not_x0.evaluate(&[true, false]));
    }

    #[test]
    fn test_weight() {
        let x0 = TruthTable::variable(2, 0);
        assert_eq!(x0.weight(), 2); // True for 2 of 4 inputs

        let and = TruthTable::and_all(2);
        assert_eq!(and.weight(), 1); // True only for (1,1)

        let or = TruthTable::or_all(2);
        assert_eq!(or.weight(), 3); // True for all except (0,0)
    }
}
