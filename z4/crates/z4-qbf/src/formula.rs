//! QBF formula representation
//!
//! Represents quantified boolean formulas in prenex normal form:
//! Q₁x₁...Qₙxₙ. φ(x₁,...,xₙ)
//!
//! Where each Qᵢ is either ∃ (existential) or ∀ (universal),
//! and φ is a propositional formula in CNF.

use z4_sat::Literal;

/// Quantifier type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Quantifier {
    /// Existential quantifier (∃)
    Exists,
    /// Universal quantifier (∀)
    Forall,
}

impl Quantifier {
    /// Returns the dual quantifier (∃ ↔ ∀)
    pub fn dual(self) -> Self {
        match self {
            Quantifier::Exists => Quantifier::Forall,
            Quantifier::Forall => Quantifier::Exists,
        }
    }
}

impl std::fmt::Display for Quantifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Quantifier::Exists => write!(f, "∃"),
            Quantifier::Forall => write!(f, "∀"),
        }
    }
}

/// A quantifier block (quantifier + variables)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantifierBlock {
    /// The quantifier type
    pub quantifier: Quantifier,
    /// Variables in this block (1-indexed)
    pub variables: Vec<u32>,
}

impl QuantifierBlock {
    /// Create a new quantifier block
    pub fn new(quantifier: Quantifier, variables: Vec<u32>) -> Self {
        Self {
            quantifier,
            variables,
        }
    }

    /// Create an existential block
    pub fn exists(variables: Vec<u32>) -> Self {
        Self::new(Quantifier::Exists, variables)
    }

    /// Create a universal block
    pub fn forall(variables: Vec<u32>) -> Self {
        Self::new(Quantifier::Forall, variables)
    }
}

/// A QBF formula in prenex CNF form
///
/// The quantifier prefix is a sequence of quantifier blocks,
/// followed by a CNF matrix (list of clauses).
#[derive(Debug, Clone)]
pub struct QbfFormula {
    /// Number of variables
    pub num_vars: usize,
    /// Quantifier prefix (ordered from outermost to innermost)
    pub prefix: Vec<QuantifierBlock>,
    /// CNF matrix (clauses as lists of literals)
    pub clauses: Vec<Vec<Literal>>,
    /// Quantifier level for each variable (0-indexed by var-1)
    /// Level 0 is outermost, higher levels are more inner
    var_levels: Vec<u32>,
    /// Quantifier type for each variable (0-indexed by var-1)
    var_quantifiers: Vec<Quantifier>,
}

impl QbfFormula {
    /// Create a new QBF formula
    pub fn new(num_vars: usize, prefix: Vec<QuantifierBlock>, clauses: Vec<Vec<Literal>>) -> Self {
        // Build variable info from prefix
        let mut var_levels = vec![0u32; num_vars];
        let mut var_quantifiers = vec![Quantifier::Exists; num_vars]; // Default to existential

        for (level, block) in prefix.iter().enumerate() {
            for &var in &block.variables {
                if var > 0 && (var as usize) <= num_vars {
                    var_levels[var as usize - 1] = level as u32;
                    var_quantifiers[var as usize - 1] = block.quantifier;
                }
            }
        }

        Self {
            num_vars,
            prefix,
            clauses,
            var_levels,
            var_quantifiers,
        }
    }

    /// Get the quantifier level of a variable (1-indexed)
    pub fn var_level(&self, var: u32) -> u32 {
        if var > 0 && (var as usize) <= self.num_vars {
            self.var_levels[var as usize - 1]
        } else {
            0
        }
    }

    /// Get the quantifier type of a variable (1-indexed)
    pub fn var_quantifier(&self, var: u32) -> Quantifier {
        if var > 0 && (var as usize) <= self.num_vars {
            self.var_quantifiers[var as usize - 1]
        } else {
            Quantifier::Exists // Unquantified variables are existential
        }
    }

    /// Check if a variable is existential
    pub fn is_existential(&self, var: u32) -> bool {
        self.var_quantifier(var) == Quantifier::Exists
    }

    /// Check if a variable is universal
    pub fn is_universal(&self, var: u32) -> bool {
        self.var_quantifier(var) == Quantifier::Forall
    }

    /// Get the quantifier level of a literal
    pub fn lit_level(&self, lit: Literal) -> u32 {
        self.var_level(lit.variable().0)
    }

    /// Check if a literal is existential
    pub fn lit_is_existential(&self, lit: Literal) -> bool {
        self.is_existential(lit.variable().0)
    }

    /// Check if a literal is universal
    pub fn lit_is_universal(&self, lit: Literal) -> bool {
        self.is_universal(lit.variable().0)
    }

    /// Get the maximum quantifier level of any existential literal in a clause
    pub fn max_existential_level(&self, clause: &[Literal]) -> Option<u32> {
        clause
            .iter()
            .filter(|lit| self.lit_is_existential(**lit))
            .map(|lit| self.lit_level(*lit))
            .max()
    }

    /// Apply universal reduction to a clause
    ///
    /// Removes universal literals whose level is >= the maximum existential level.
    /// These literals cannot affect satisfiability because they can always be
    /// set to satisfy the clause after all existential decisions are made.
    pub fn universal_reduce(&self, clause: &[Literal]) -> Vec<Literal> {
        let max_exist_level = self.max_existential_level(clause);

        match max_exist_level {
            Some(max_level) => {
                clause
                    .iter()
                    .filter(|lit| {
                        // Keep existential literals and universal literals with level < max_exist
                        self.lit_is_existential(**lit) || self.lit_level(**lit) < max_level
                    })
                    .copied()
                    .collect()
            }
            None => {
                // No existential literals - this is unusual but keep universals
                clause.to_vec()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use z4_sat::Variable;

    #[test]
    fn test_quantifier_dual() {
        assert_eq!(Quantifier::Exists.dual(), Quantifier::Forall);
        assert_eq!(Quantifier::Forall.dual(), Quantifier::Exists);
    }

    #[test]
    fn test_qbf_formula_var_info() {
        // ∃x₁∀x₂∃x₃. (x₁ ∨ x₂ ∨ x₃)
        let prefix = vec![
            QuantifierBlock::exists(vec![1]),
            QuantifierBlock::forall(vec![2]),
            QuantifierBlock::exists(vec![3]),
        ];
        let clauses = vec![vec![
            Literal::positive(Variable(1)),
            Literal::positive(Variable(2)),
            Literal::positive(Variable(3)),
        ]];

        let formula = QbfFormula::new(3, prefix, clauses);

        // Check levels
        assert_eq!(formula.var_level(1), 0); // Outermost
        assert_eq!(formula.var_level(2), 1);
        assert_eq!(formula.var_level(3), 2); // Innermost

        // Check quantifiers
        assert!(formula.is_existential(1));
        assert!(formula.is_universal(2));
        assert!(formula.is_existential(3));
    }

    #[test]
    fn test_universal_reduction() {
        // ∃x₁∀x₂∃x₃. (x₁ ∨ x₂ ∨ x₃)
        // After universal reduction: (x₁ ∨ x₃) because x₂ is at level 1,
        // which is < max existential level 2, so x₂ stays
        let prefix = vec![
            QuantifierBlock::exists(vec![1]),
            QuantifierBlock::forall(vec![2]),
            QuantifierBlock::exists(vec![3]),
        ];
        let formula = QbfFormula::new(3, prefix, vec![]);

        let clause = vec![
            Literal::positive(Variable(1)),
            Literal::positive(Variable(2)),
            Literal::positive(Variable(3)),
        ];
        let reduced = formula.universal_reduce(&clause);

        // x₂ at level 1 < max_exist_level=2, so it stays
        assert_eq!(reduced.len(), 3);

        // Test case where universal is removed
        // ∃x₁∀x₂. (x₁ ∨ x₂)
        // x₂ at level 1 >= max_exist_level=0, so x₂ removed
        let prefix2 = vec![
            QuantifierBlock::exists(vec![1]),
            QuantifierBlock::forall(vec![2]),
        ];
        let formula2 = QbfFormula::new(2, prefix2, vec![]);

        let clause2 = vec![
            Literal::positive(Variable(1)),
            Literal::positive(Variable(2)),
        ];
        let reduced2 = formula2.universal_reduce(&clause2);

        // x₂ at level 1 >= max_exist_level=0, so x₂ removed? No wait...
        // max_exist_level for x₁ is 0, x₂ is at level 1
        // We remove universal literals with level >= max_exist_level
        // So x₂ (level 1) >= 0 should be removed
        assert_eq!(reduced2.len(), 1);
        assert_eq!(reduced2[0], Literal::positive(Variable(1)));
    }
}
