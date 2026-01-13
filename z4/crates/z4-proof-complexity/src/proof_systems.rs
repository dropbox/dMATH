//! Propositional proof systems for proof complexity analysis.
//!
//! This module provides representations and verification for various
//! propositional proof systems.

use z4_sat::{Literal, Variable};

/// Alias for Literal
type Lit = Literal;
/// Alias for Variable
type Var = Variable;

/// Propositional proof systems ordered by strength.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProofSystem {
    /// Tree-like Resolution (no clause reuse)
    TreeResolution,
    /// Regular Resolution (each variable resolved at most once per path)
    RegularResolution,
    /// General Resolution (CDCL produces these)
    Resolution,
    /// Extended Resolution (with auxiliary variables)
    ExtendedResolution,
    /// Frege systems (standard propositional logic)
    Frege,
    /// Extended Frege (Frege with abbreviations)
    ExtendedFrege,
    /// Cutting Planes (integer linear programming proofs)
    CuttingPlanes,
    /// Polynomial Calculus
    PolynomialCalculus,
    /// Sum of Squares (SOS)
    SumOfSquares,
}

impl ProofSystem {
    /// Returns true if this system p-simulates the other.
    ///
    /// System A p-simulates system B if any proof in B can be converted
    /// to a proof in A with at most polynomial blowup.
    pub fn p_simulates(&self, other: &ProofSystem) -> bool {
        use ProofSystem::*;
        match (self, other) {
            // Same system always simulates itself
            (a, b) if a == b => true,

            // Resolution hierarchy
            (Resolution, TreeResolution) => true,
            (Resolution, RegularResolution) => true,
            (RegularResolution, TreeResolution) => true,
            (ExtendedResolution, Resolution) => true,
            (ExtendedResolution, TreeResolution) => true,
            (ExtendedResolution, RegularResolution) => true,

            // Frege hierarchy
            (Frege, Resolution) => true,
            (Frege, ExtendedResolution) => false, // Open problem!
            (ExtendedFrege, Frege) => true,
            (ExtendedFrege, ExtendedResolution) => true,

            // Cutting Planes is incomparable with Resolution in general
            // But p-simulates tree resolution
            (CuttingPlanes, TreeResolution) => true,

            // Default: we don't know or it doesn't simulate
            _ => false,
        }
    }

    /// Known exponential lower bounds for this proof system.
    pub fn known_lower_bounds(&self) -> &'static [&'static str] {
        use ProofSystem::*;
        match self {
            TreeResolution => &[
                "Pigeonhole (PHP) - exponential",
                "Tseitin on expanders - exponential",
                "Random 3-CNF near threshold - exponential",
                "Parity - exponential",
            ],
            RegularResolution => &[
                "Pigeonhole (PHP) - exponential",
                "Tseitin on expanders - exponential",
            ],
            Resolution => &[
                "Pigeonhole (PHP) - exponential (Haken 1985)",
                "Tseitin on expanders - exponential (Urquhart 1987)",
                "Random 3-CNF - exponential (Chvatal-Szemeredi 1988)",
            ],
            ExtendedResolution => &[
                "No exponential lower bounds known!",
                "PHP has polynomial proofs (Cook 1976)",
            ],
            Frege => &["No super-polynomial lower bounds known!"],
            ExtendedFrege => &["No super-polynomial lower bounds known!"],
            CuttingPlanes => &[
                "Random CNF - exponential (Pudlak 1997)",
                "Clique-coloring - exponential",
            ],
            PolynomialCalculus => &[
                "Pigeonhole - degree lower bound (Razborov 1998)",
                "Tseitin - degree lower bound",
            ],
            SumOfSquares => &["Random 3-XOR - degree lower bound (Grigoriev-Vorobjov 2001)"],
        }
    }
}

/// A clause in a resolution proof.
pub type Clause = Vec<Lit>;

/// A step in a resolution proof.
#[derive(Debug, Clone)]
pub enum ResolutionStep {
    /// Axiom (original clause from the formula)
    Axiom(Clause),
    /// Resolution: derive C from clauses at indices i and j by resolving on variable
    Resolve {
        clause: Clause,
        parent1: usize,
        parent2: usize,
        pivot: Var,
    },
}

/// A resolution proof.
#[derive(Debug, Clone)]
pub struct ResolutionProof {
    /// Steps of the proof
    steps: Vec<ResolutionStep>,
}

impl ResolutionProof {
    /// Create a new empty proof.
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Add an axiom (original clause).
    pub fn add_axiom(&mut self, clause: Clause) -> usize {
        let idx = self.steps.len();
        self.steps.push(ResolutionStep::Axiom(clause));
        idx
    }

    /// Add a resolution step.
    pub fn add_resolution(
        &mut self,
        clause: Clause,
        parent1: usize,
        parent2: usize,
        pivot: Var,
    ) -> usize {
        let idx = self.steps.len();
        self.steps.push(ResolutionStep::Resolve {
            clause,
            parent1,
            parent2,
            pivot,
        });
        idx
    }

    /// Number of steps in the proof.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Check if proof is empty.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Get the clause at a given step.
    pub fn clause_at(&self, idx: usize) -> Option<&Clause> {
        match self.steps.get(idx)? {
            ResolutionStep::Axiom(c) => Some(c),
            ResolutionStep::Resolve { clause, .. } => Some(clause),
        }
    }

    /// Check if this is a valid refutation (derives empty clause).
    pub fn is_refutation(&self) -> bool {
        if let Some(last) = self.steps.last() {
            match last {
                ResolutionStep::Axiom(c) => c.is_empty(),
                ResolutionStep::Resolve { clause, .. } => clause.is_empty(),
            }
        } else {
            false
        }
    }

    /// Verify the proof is valid.
    ///
    /// Checks that each resolution step correctly derives its clause
    /// from its parents by resolving on the pivot variable.
    pub fn verify(&self) -> Result<(), String> {
        for (idx, step) in self.steps.iter().enumerate() {
            match step {
                ResolutionStep::Axiom(_) => {
                    // Axioms are always valid
                }
                ResolutionStep::Resolve {
                    clause,
                    parent1,
                    parent2,
                    pivot,
                } => {
                    // Check parent indices are valid
                    if *parent1 >= idx || *parent2 >= idx {
                        return Err(format!(
                            "Step {}: parent index out of bounds ({}, {} >= {})",
                            idx, parent1, parent2, idx
                        ));
                    }

                    // Get parent clauses
                    let c1 = self.clause_at(*parent1).unwrap();
                    let c2 = self.clause_at(*parent2).unwrap();

                    // Check pivot appears positive in one and negative in other
                    let pos_pivot = Lit::positive(*pivot);
                    let neg_pivot = Lit::negative(*pivot);

                    let (pos_parent, neg_parent) =
                        if c1.contains(&pos_pivot) && c2.contains(&neg_pivot) {
                            (c1, c2)
                        } else if c1.contains(&neg_pivot) && c2.contains(&pos_pivot) {
                            (c2, c1)
                        } else {
                            return Err(format!(
                                "Step {}: pivot {:?} not properly present in parents",
                                idx, pivot
                            ));
                        };

                    // Compute expected resolvent
                    let mut expected: Vec<Lit> = pos_parent
                        .iter()
                        .filter(|&&l| l != pos_pivot)
                        .chain(neg_parent.iter().filter(|&&l| l != neg_pivot))
                        .copied()
                        .collect();
                    expected.sort_by_key(|l| (l.variable().index(), !l.is_positive()));
                    expected.dedup();

                    // Check clause matches expected
                    let mut actual = clause.clone();
                    actual.sort_by_key(|l| (l.variable().index(), !l.is_positive()));

                    if actual != expected {
                        return Err(format!(
                            "Step {}: clause {:?} doesn't match expected resolvent {:?}",
                            idx, clause, expected
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Check if this is a tree resolution proof (no clause reuse).
    pub fn is_tree(&self) -> bool {
        let mut used = vec![false; self.steps.len()];

        for step in &self.steps {
            if let ResolutionStep::Resolve {
                parent1, parent2, ..
            } = step
            {
                if used[*parent1] || used[*parent2] {
                    return false;
                }
                used[*parent1] = true;
                used[*parent2] = true;
            }
        }
        true
    }

    /// Check if this is a regular resolution proof.
    ///
    /// A proof is regular if on every path from root to axiom,
    /// each variable is resolved on at most once.
    pub fn is_regular(&self) -> bool {
        // For each step, track which variables have been resolved on the path to it
        let mut resolved_on_path: Vec<std::collections::HashSet<Var>> =
            vec![std::collections::HashSet::new(); self.steps.len()];

        for (idx, step) in self.steps.iter().enumerate() {
            if let ResolutionStep::Resolve {
                parent1,
                parent2,
                pivot,
                ..
            } = step
            {
                // Union of variables resolved on paths to parents, plus this pivot
                let mut vars = resolved_on_path[*parent1].clone();
                vars.extend(&resolved_on_path[*parent2]);

                if vars.contains(pivot) {
                    return false;
                }
                vars.insert(*pivot);
                resolved_on_path[idx] = vars;
            }
        }
        true
    }

    /// Width of the proof (maximum clause size).
    pub fn width(&self) -> usize {
        self.steps
            .iter()
            .map(|step| match step {
                ResolutionStep::Axiom(c) => c.len(),
                ResolutionStep::Resolve { clause, .. } => clause.len(),
            })
            .max()
            .unwrap_or(0)
    }

    /// Space of the proof (maximum number of clauses needed at once).
    ///
    /// This is a lower bound computed by analyzing clause lifetimes.
    pub fn space(&self) -> usize {
        let n = self.steps.len();
        if n == 0 {
            return 0;
        }

        // For each step, compute when it's last used
        let mut last_use = vec![0usize; n];
        for (idx, step) in self.steps.iter().enumerate() {
            if let ResolutionStep::Resolve {
                parent1, parent2, ..
            } = step
            {
                last_use[*parent1] = idx;
                last_use[*parent2] = idx;
            }
        }

        // Simulate proof and track max clauses alive
        let mut alive = 0usize;
        let mut max_alive = 0usize;

        for idx in 0..n {
            alive += 1; // New clause derived
            max_alive = max_alive.max(alive);

            // Check if any clause dies at this step
            for (prev, &usage) in last_use.iter().enumerate().take(idx + 1) {
                if usage == idx && prev != idx {
                    alive = alive.saturating_sub(1);
                }
            }
        }

        max_alive
    }
}

impl Default for ResolutionProof {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolution_proof() {
        // Prove (A OR B) AND (NOT A OR B) AND (NOT B) is UNSAT
        // Resolution:
        // 1. A OR B (axiom)
        // 2. NOT A OR B (axiom)
        // 3. B (resolve 1,2 on A)
        // 4. NOT B (axiom)
        // 5. empty (resolve 3,4 on B)

        let a = Variable(0);
        let b = Variable(1);

        let mut proof = ResolutionProof::new();
        let s1 = proof.add_axiom(vec![Lit::positive(a), Lit::positive(b)]);
        let s2 = proof.add_axiom(vec![Lit::negative(a), Lit::positive(b)]);
        let s3 = proof.add_resolution(vec![Lit::positive(b)], s1, s2, a);
        let s4 = proof.add_axiom(vec![Lit::negative(b)]);
        let _s5 = proof.add_resolution(vec![], s3, s4, b);

        assert!(proof.is_refutation());
        assert!(proof.verify().is_ok());
        assert!(proof.is_tree());
        assert!(proof.is_regular());
        assert_eq!(proof.width(), 2);
    }

    #[test]
    fn test_non_tree_proof() {
        // Reuse a clause: (A) AND (NOT A OR B) AND (NOT A OR NOT B)
        // 1. A (axiom)
        // 2. NOT A OR B (axiom)
        // 3. B (resolve 1,2 on A)
        // 4. NOT A OR NOT B (axiom)
        // 5. NOT B (resolve 1,4 on A) -- reuses clause 1
        // 6. empty (resolve 3,5 on B)

        let a = Variable(0);
        let b = Variable(1);

        let mut proof = ResolutionProof::new();
        let s1 = proof.add_axiom(vec![Lit::positive(a)]);
        let s2 = proof.add_axiom(vec![Lit::negative(a), Lit::positive(b)]);
        let s3 = proof.add_resolution(vec![Lit::positive(b)], s1, s2, a);
        let s4 = proof.add_axiom(vec![Lit::negative(a), Lit::negative(b)]);
        let s5 = proof.add_resolution(vec![Lit::negative(b)], s1, s4, a);
        let _s6 = proof.add_resolution(vec![], s3, s5, b);

        assert!(proof.is_refutation());
        assert!(proof.verify().is_ok());
        assert!(!proof.is_tree()); // Clause 1 is used twice
    }

    #[test]
    fn test_proof_system_simulation() {
        use ProofSystem::*;

        // Resolution simulates tree resolution
        assert!(Resolution.p_simulates(&TreeResolution));
        // But tree resolution doesn't simulate resolution
        assert!(!TreeResolution.p_simulates(&Resolution));
        // Extended resolution simulates resolution
        assert!(ExtendedResolution.p_simulates(&Resolution));
    }
}
