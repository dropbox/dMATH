//! Model reconstruction for equisatisfiable transformations.
//!
//! When the solver applies equisatisfiable transformations like BVE (Bounded
//! Variable Elimination) or sweeping (equivalence merging), the resulting
//! model satisfies the transformed formula but may not satisfy the original.
//!
//! This module provides reconstruction logic to convert a model of the
//! transformed formula into a model of the original formula.
//!
//! # Reconstruction Order
//!
//! Reconstructions must be applied in reverse order of the transformations:
//! if BVE was applied before sweeping, we must undo sweeping first, then BVE.

use crate::clause::Clause;
use crate::literal::{Literal, Variable};

/// A single reconstruction step.
#[derive(Debug, Clone)]
pub enum ReconstructionStep {
    /// BVE elimination: the variable was eliminated by resolution.
    /// Contains the original clauses that contained this variable.
    BVE {
        /// The eliminated variable
        variable: Variable,
        /// Clauses containing positive occurrences of the variable
        pos_clauses: Vec<Vec<Literal>>,
        /// Clauses containing negative occurrences of the variable
        neg_clauses: Vec<Vec<Literal>>,
    },
    /// Sweeping: variables were merged due to equivalence.
    /// The lit_map maps each literal index to its canonical representative.
    Sweep {
        /// Number of variables in the original formula
        num_vars: usize,
        /// Mapping from literal index to canonical literal
        lit_map: Vec<Literal>,
    },
}

/// Stack of reconstruction steps (applied in reverse order).
#[derive(Debug, Clone, Default)]
pub struct ReconstructionStack {
    /// Steps in order they were applied (reconstruction reverses this)
    steps: Vec<ReconstructionStep>,
}

impl ReconstructionStack {
    /// Create a new empty reconstruction stack.
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Check if the stack is empty (no transformations to reconstruct).
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Push a BVE elimination step.
    pub fn push_bve(
        &mut self,
        variable: Variable,
        pos_clauses: Vec<Vec<Literal>>,
        neg_clauses: Vec<Vec<Literal>>,
    ) {
        self.steps.push(ReconstructionStep::BVE {
            variable,
            pos_clauses,
            neg_clauses,
        });
    }

    /// Push a sweep (equivalence) step.
    pub fn push_sweep(&mut self, num_vars: usize, lit_map: Vec<Literal>) {
        self.steps
            .push(ReconstructionStep::Sweep { num_vars, lit_map });
    }

    /// Reconstruct the original model from a model of the transformed formula.
    ///
    /// The model is modified in place. If the model is too short (fewer
    /// variables than the original), it will be extended.
    pub fn reconstruct(&self, model: &mut Vec<bool>) {
        // Apply steps in reverse order
        for step in self.steps.iter().rev() {
            match step {
                ReconstructionStep::BVE {
                    variable,
                    pos_clauses,
                    neg_clauses,
                } => {
                    reconstruct_bve(model, *variable, pos_clauses, neg_clauses);
                }
                ReconstructionStep::Sweep { num_vars, lit_map } => {
                    reconstruct_sweep(model, *num_vars, lit_map);
                }
            }
        }
    }

    /// Clear all reconstruction steps.
    pub fn clear(&mut self) {
        self.steps.clear();
    }

    /// Iterate over all BVE clauses for model verification.
    ///
    /// Yields references to all original clauses that were removed by BVE.
    /// These need to be verified separately since they're no longer in clause_db.
    pub fn iter_bve_clauses(&self) -> impl Iterator<Item = &[Literal]> {
        self.steps.iter().flat_map(|step| match step {
            ReconstructionStep::BVE {
                pos_clauses,
                neg_clauses,
                ..
            } => pos_clauses
                .iter()
                .chain(neg_clauses.iter())
                .map(|c| c.as_slice())
                .collect::<Vec<_>>(),
            ReconstructionStep::Sweep { .. } => vec![],
        })
    }
}

/// Reconstruct a BVE-eliminated variable.
///
/// We try both polarities and choose the one that satisfies more original
/// clauses containing the variable. If a clause would be falsified by both
/// polarities, we prefer true (positive) as a tiebreaker.
fn reconstruct_bve(
    model: &mut Vec<bool>,
    variable: Variable,
    pos_clauses: &[Vec<Literal>],
    neg_clauses: &[Vec<Literal>],
) {
    let var_idx = variable.index();

    // Ensure model is large enough
    if var_idx >= model.len() {
        model.resize(var_idx + 1, false);
    }

    // Try setting variable to true: check if any positive clause would be falsified
    let true_works = can_satisfy_with_value(model, var_idx, true, pos_clauses, neg_clauses);

    // Try setting variable to false: check if any negative clause would be falsified
    let false_works = can_satisfy_with_value(model, var_idx, false, pos_clauses, neg_clauses);

    // Choose the value that works (prefer true as tiebreaker)
    model[var_idx] = if true_works {
        true
    } else if false_works {
        false
    } else {
        // Neither fully works; use the value that satisfies more clauses
        let true_score = count_satisfied(model, var_idx, true, pos_clauses, neg_clauses);
        let false_score = count_satisfied(model, var_idx, false, pos_clauses, neg_clauses);
        true_score >= false_score
    };
}

/// Check if setting a variable to a given value can satisfy all clauses.
fn can_satisfy_with_value(
    model: &[bool],
    var_idx: usize,
    value: bool,
    pos_clauses: &[Vec<Literal>],
    neg_clauses: &[Vec<Literal>],
) -> bool {
    // Check clauses where the variable appears positively
    for clause in pos_clauses {
        if !clause_satisfied_with(model, clause, var_idx, value) {
            return false;
        }
    }
    // Check clauses where the variable appears negatively
    for clause in neg_clauses {
        if !clause_satisfied_with(model, clause, var_idx, value) {
            return false;
        }
    }
    true
}

/// Count how many clauses are satisfied with a given variable value.
fn count_satisfied(
    model: &[bool],
    var_idx: usize,
    value: bool,
    pos_clauses: &[Vec<Literal>],
    neg_clauses: &[Vec<Literal>],
) -> usize {
    let mut count = 0;
    for clause in pos_clauses {
        if clause_satisfied_with(model, clause, var_idx, value) {
            count += 1;
        }
    }
    for clause in neg_clauses {
        if clause_satisfied_with(model, clause, var_idx, value) {
            count += 1;
        }
    }
    count
}

/// Check if a clause is satisfied under the current model with a specific variable value.
fn clause_satisfied_with(model: &[bool], clause: &[Literal], var_idx: usize, value: bool) -> bool {
    for &lit in clause {
        let lit_var_idx = lit.variable().index();
        let lit_value = if lit_var_idx == var_idx {
            value
        } else if lit_var_idx < model.len() {
            model[lit_var_idx]
        } else {
            false // Unassigned variables default to false
        };

        // Literal is satisfied if (positive and true) or (negative and false)
        let lit_satisfied = if lit.is_positive() {
            lit_value
        } else {
            !lit_value
        };

        if lit_satisfied {
            return true;
        }
    }
    false
}

/// Reconstruct variables after sweeping (equivalence merging).
///
/// For each variable that was merged into another, copy the value from
/// its canonical representative.
fn reconstruct_sweep(model: &mut Vec<bool>, num_vars: usize, lit_map: &[Literal]) {
    // Ensure model is large enough
    if num_vars > model.len() {
        model.resize(num_vars, false);
    }

    // For each variable, check if it was mapped to a different representative
    for var_idx in 0..num_vars {
        let pos_lit = Literal::positive(Variable(var_idx as u32));
        let pos_idx = pos_lit.index();

        if pos_idx >= lit_map.len() {
            continue;
        }

        let mapped_lit = lit_map[pos_idx];
        let mapped_var_idx = mapped_lit.variable().index();

        // If mapped to a different variable, copy the value
        if mapped_var_idx != var_idx && mapped_var_idx < model.len() {
            let mapped_value = model[mapped_var_idx];
            // If the mapping is negated, invert the value
            model[var_idx] = if mapped_lit.is_positive() {
                mapped_value
            } else {
                !mapped_value
            };
        }
    }
}

/// Extract clause literals for reconstruction from the clause database.
///
/// Filters to only include non-empty clauses containing the specified variable.
pub fn extract_clauses_for_var(
    clauses: &[Clause],
    variable: Variable,
    clause_indices: &[usize],
) -> Vec<Vec<Literal>> {
    let mut result = Vec::new();
    for &idx in clause_indices {
        if idx < clauses.len() && !clauses[idx].is_empty() {
            // Verify the clause actually contains the variable
            let contains_var = clauses[idx]
                .literals
                .iter()
                .any(|lit| lit.variable() == variable);
            if contains_var {
                result.push(clauses[idx].literals.to_vec());
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lit(var: u32, positive: bool) -> Literal {
        if positive {
            Literal::positive(Variable(var))
        } else {
            Literal::negative(Variable(var))
        }
    }

    #[test]
    fn test_bve_reconstruct_simple() {
        // Original: (x0 ∨ x1) ∧ (¬x0 ∨ x2)
        // After eliminating x0: (x1 ∨ x2)
        // Model of transformed: [_, true, true]
        // Need to reconstruct x0

        let mut model = vec![false, true, true];

        let pos_clauses = vec![vec![lit(0, true), lit(1, true)]];
        let neg_clauses = vec![vec![lit(0, false), lit(2, true)]];

        reconstruct_bve(&mut model, Variable(0), &pos_clauses, &neg_clauses);

        // Both x0=true and x0=false satisfy the clauses with the current model
        // (x1=true satisfies first, x2=true satisfies second)
        // Either value should work
        let clause1_sat = model[0] || model[1]; // x0 ∨ x1
        let clause2_sat = !model[0] || model[2]; // ¬x0 ∨ x2

        assert!(clause1_sat, "First clause should be satisfied");
        assert!(clause2_sat, "Second clause should be satisfied");
    }

    #[test]
    fn test_bve_reconstruct_forced_true() {
        // Original: (x0 ∨ ¬x1) where x1=true in model
        // x0 must be true to satisfy the clause

        let mut model = vec![false, true];

        let pos_clauses = vec![vec![lit(0, true), lit(1, false)]];
        let neg_clauses = vec![];

        reconstruct_bve(&mut model, Variable(0), &pos_clauses, &neg_clauses);

        assert!(
            model[0],
            "x0 must be true to satisfy (x0 ∨ ¬x1) with x1=true"
        );
    }

    #[test]
    fn test_bve_reconstruct_forced_false() {
        // Original: (¬x0 ∨ ¬x1) where x1=true in model
        // x0 must be false to satisfy the clause

        let mut model = vec![true, true];

        let pos_clauses = vec![];
        let neg_clauses = vec![vec![lit(0, false), lit(1, false)]];

        reconstruct_bve(&mut model, Variable(0), &pos_clauses, &neg_clauses);

        assert!(
            !model[0],
            "x0 must be false to satisfy (¬x0 ∨ ¬x1) with x1=true"
        );
    }

    #[test]
    fn test_sweep_reconstruct_equivalence() {
        // x0 ↔ x1 means they should have the same value
        // Suppose x1 was the canonical representative and x1=true in model
        // Then x0 should be reconstructed to true

        let mut model = vec![false, true, false];

        // lit_map: x0 -> x1, x1 -> x1, x2 -> x2
        // lit_map indices: pos(0)=0, neg(0)=1, pos(1)=2, neg(1)=3, pos(2)=4, neg(2)=5
        let mut lit_map = vec![Literal(0); 6];
        lit_map[lit(0, true).index()] = lit(1, true); // pos x0 -> pos x1
        lit_map[lit(0, false).index()] = lit(1, false); // neg x0 -> neg x1
        lit_map[lit(1, true).index()] = lit(1, true); // pos x1 -> pos x1
        lit_map[lit(1, false).index()] = lit(1, false); // neg x1 -> neg x1
        lit_map[lit(2, true).index()] = lit(2, true); // pos x2 -> pos x2
        lit_map[lit(2, false).index()] = lit(2, false); // neg x2 -> neg x2

        reconstruct_sweep(&mut model, 3, &lit_map);

        assert_eq!(
            model[0], model[1],
            "x0 should equal x1 after reconstruction"
        );
        assert!(model[0], "x0 should be true (copied from x1)");
    }

    #[test]
    fn test_sweep_reconstruct_negated_equivalence() {
        // x0 ↔ ¬x1 means they should have opposite values
        // Suppose x1 was the canonical representative and x1=true in model
        // Then x0 should be reconstructed to false

        let mut model = vec![true, true];

        // lit_map: pos(x0) -> neg(x1) means x0 = ¬x1
        let mut lit_map = vec![Literal(0); 4];
        lit_map[lit(0, true).index()] = lit(1, false); // pos x0 -> neg x1
        lit_map[lit(0, false).index()] = lit(1, true); // neg x0 -> pos x1
        lit_map[lit(1, true).index()] = lit(1, true); // pos x1 -> pos x1
        lit_map[lit(1, false).index()] = lit(1, false); // neg x1 -> neg x1

        reconstruct_sweep(&mut model, 2, &lit_map);

        assert!(!model[0], "x0 should be false (negation of x1=true)");
    }

    #[test]
    fn test_reconstruction_stack_order() {
        // Test that reconstruction happens in reverse order
        let mut stack = ReconstructionStack::new();

        // First transformation: eliminate x2 via BVE
        stack.push_bve(Variable(2), vec![vec![lit(2, true), lit(1, false)]], vec![]);

        // Second transformation: sweep x0 -> x1
        let mut lit_map = vec![Literal(0); 6];
        lit_map[lit(0, true).index()] = lit(1, true);
        lit_map[lit(0, false).index()] = lit(1, false);
        lit_map[lit(1, true).index()] = lit(1, true);
        lit_map[lit(1, false).index()] = lit(1, false);
        lit_map[lit(2, true).index()] = lit(2, true);
        lit_map[lit(2, false).index()] = lit(2, false);
        stack.push_sweep(3, lit_map);

        // Model after transformations: x1=true, others irrelevant
        let mut model = vec![false, true, false];

        stack.reconstruct(&mut model);

        // After sweep reconstruction: x0 = x1 = true
        assert!(model[0], "x0 should be true after sweep reconstruction");
        assert!(model[1], "x1 should still be true");

        // After BVE reconstruction: x2 depends on (x2 ∨ ¬x1) with x1=true
        // x2 must be true to satisfy the clause
        assert!(model[2], "x2 should be true after BVE reconstruction");
    }
}
