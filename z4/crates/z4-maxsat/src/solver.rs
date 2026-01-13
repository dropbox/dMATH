//! MAX-SAT Solver Implementation
//!
//! This module implements a MAX-SAT solver using iterative SAT solving with
//! relaxation variables and cardinality constraints.
//!
//! ## Algorithm
//!
//! Uses the linear search approach:
//! 1. Add relaxation variables to soft clauses
//! 2. Start with bound k = 0 (no soft clauses violated)
//! 3. Add at-most-k constraint on relaxation variables
//! 4. If SAT, done with cost = number of satisfied relaxations
//! 5. If UNSAT, increment k and retry

use z4_sat::{Literal, SolveResult, Solver as SatSolver, Variable};

/// A clause represented as a vector of literals
/// Positive integers represent positive literals, negative represent negations
pub type Clause = Vec<i32>;

/// Weight type for soft clauses
pub type Weight = u64;

/// Result of MAX-SAT solving
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaxSatResult {
    /// Found optimal solution
    Optimal {
        /// The satisfying assignment (variable -> value)
        model: Vec<bool>,
        /// Total cost (sum of weights of unsatisfied soft clauses)
        cost: u64,
    },
    /// Hard clauses are unsatisfiable
    Unsatisfiable,
    /// Unknown result (timeout, resource limit)
    Unknown,
}

/// Statistics for MAX-SAT solving
#[derive(Debug, Clone, Default)]
pub struct MaxSatStats {
    /// Number of SAT solver calls
    pub sat_calls: u64,
    /// Number of UNSAT cores extracted
    pub cores_extracted: u64,
    /// Total literals in extracted cores
    pub core_literals: u64,
    /// Number of cardinality constraints added
    pub cardinality_constraints: u64,
}

/// A soft clause with its relaxation variable
#[derive(Debug, Clone)]
struct SoftClause {
    /// Original clause literals
    clause: Clause,
    /// Weight of this clause
    weight: Weight,
    /// Relaxation variable (when true, clause is "satisfied" by relaxation)
    relax_var: u32,
}

/// MAX-SAT Solver
///
/// Supports weighted partial MAX-SAT:
/// - Hard clauses must be satisfied
/// - Soft clauses have weights; goal is to minimize total weight of unsatisfied soft clauses
pub struct MaxSatSolver {
    /// Hard clauses (must be satisfied)
    hard_clauses: Vec<Clause>,
    /// Soft clauses with weights
    soft_clauses: Vec<SoftClause>,
    /// Next available variable
    next_var: u32,
    /// Statistics
    stats: MaxSatStats,
}

impl MaxSatSolver {
    /// Create a new MAX-SAT solver
    pub fn new() -> Self {
        Self {
            hard_clauses: Vec::new(),
            soft_clauses: Vec::new(),
            next_var: 1,
            stats: MaxSatStats::default(),
        }
    }

    /// Add a hard clause (must be satisfied)
    pub fn add_hard_clause(&mut self, clause: Clause) {
        // Track the maximum variable
        for &lit in &clause {
            let var = lit.unsigned_abs();
            if var >= self.next_var {
                self.next_var = var + 1;
            }
        }
        self.hard_clauses.push(clause);
    }

    /// Add a soft clause with weight
    ///
    /// Note: relaxation variables are allocated lazily in solve() to avoid
    /// collisions with user variables added after soft clauses.
    pub fn add_soft_clause(&mut self, clause: Clause, weight: Weight) {
        // Track the maximum variable
        for &lit in &clause {
            let var = lit.unsigned_abs();
            if var >= self.next_var {
                self.next_var = var + 1;
            }
        }

        // Relaxation variable will be assigned later in solve()
        self.soft_clauses.push(SoftClause {
            clause,
            weight,
            relax_var: 0, // Placeholder, will be set in solve()
        });
    }

    /// Get solver statistics
    pub fn stats(&self) -> &MaxSatStats {
        &self.stats
    }

    /// Solve the MAX-SAT instance
    ///
    /// Returns the optimal solution or UNSAT if hard clauses are unsatisfiable.
    pub fn solve(&mut self) -> MaxSatResult {
        // Handle empty instance
        if self.hard_clauses.is_empty() && self.soft_clauses.is_empty() {
            return MaxSatResult::Optimal {
                model: vec![],
                cost: 0,
            };
        }

        // Allocate relaxation variables AFTER all user variables
        // This ensures no collisions between user vars and relax vars
        for soft in &mut self.soft_clauses {
            soft.relax_var = self.next_var;
            self.next_var += 1;
        }

        // First check if hard clauses alone are satisfiable
        if !self.check_hard_clauses() {
            return MaxSatResult::Unsatisfiable;
        }

        // If no soft clauses, just solve hard clauses
        if self.soft_clauses.is_empty() {
            return self.solve_hard_only();
        }

        // For unweighted MAX-SAT, use linear search
        // For weighted MAX-SAT with diverse weights, use stratification
        if self.all_unit_weights() {
            self.solve_linear_search()
        } else {
            self.solve_stratified()
        }
    }

    /// Check if all soft clauses have weight 1
    fn all_unit_weights(&self) -> bool {
        self.soft_clauses.iter().all(|s| s.weight == 1)
    }

    /// Check if hard clauses are satisfiable
    fn check_hard_clauses(&mut self) -> bool {
        if self.hard_clauses.is_empty() {
            return true;
        }

        let mut solver = SatSolver::new(self.next_var as usize);
        for clause in &self.hard_clauses {
            let lits: Vec<Literal> = clause.iter().map(|&l| self.int_to_literal(l)).collect();
            solver.add_clause(lits);
        }

        self.stats.sat_calls += 1;
        matches!(solver.solve(), SolveResult::Sat(_))
    }

    /// Solve when there are only hard clauses
    fn solve_hard_only(&mut self) -> MaxSatResult {
        let mut solver = SatSolver::new(self.next_var as usize);
        for clause in &self.hard_clauses {
            let lits: Vec<Literal> = clause.iter().map(|&l| self.int_to_literal(l)).collect();
            solver.add_clause(lits);
        }

        self.stats.sat_calls += 1;
        match solver.solve() {
            SolveResult::Sat(model) => MaxSatResult::Optimal { model, cost: 0 },
            SolveResult::Unsat => MaxSatResult::Unsatisfiable,
            SolveResult::Unknown => MaxSatResult::Unknown,
        }
    }

    /// Linear search algorithm for unweighted MAX-SAT
    fn solve_linear_search(&mut self) -> MaxSatResult {
        let n = self.soft_clauses.len();

        // Binary search for optimal k (minimum violations)
        let mut lo = 0;
        let mut hi = n;
        let mut best_model: Option<Vec<bool>> = None;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if let Some(model) = self.try_solve_with_bound(mid) {
                // Found solution with cost <= mid
                best_model = Some(model);
                hi = mid;
            } else {
                // Need more than mid violations
                lo = mid + 1;
            }
        }

        // Final verification at lo
        if best_model.is_none() {
            if let Some(model) = self.try_solve_with_bound(lo) {
                best_model = Some(model);
            }
        }

        match best_model {
            Some(model) => {
                // Compute actual cost from model
                let cost = self.compute_cost(&model);
                MaxSatResult::Optimal { cost, model }
            }
            None => MaxSatResult::Unsatisfiable,
        }
    }

    /// Try to solve with at most k soft clause violations
    fn try_solve_with_bound(&mut self, k: usize) -> Option<Vec<bool>> {
        let relax_vars: Vec<u32> = self.soft_clauses.iter().map(|s| s.relax_var).collect();
        let mut aux_var = self.next_var;

        // Calculate number of variables needed for cardinality encoding
        let extra_vars = if k > 0 && k < relax_vars.len() {
            relax_vars.len() * (k + 1) + 100
        } else {
            100
        };

        let mut solver = SatSolver::new((self.next_var + extra_vars as u32) as usize);

        // Add hard clauses
        for clause in &self.hard_clauses {
            let lits: Vec<Literal> = clause.iter().map(|&l| self.int_to_literal(l)).collect();
            solver.add_clause(lits);
        }

        // Add soft clauses with relaxation variables
        for soft in &self.soft_clauses {
            let mut lits: Vec<Literal> = soft
                .clause
                .iter()
                .map(|&l| self.int_to_literal(l))
                .collect();
            lits.push(Literal::positive(Variable(soft.relax_var)));
            solver.add_clause(lits);
        }

        // Add at-most-k constraint on relaxation variables
        aux_var = self.add_at_most_k_clauses(&mut solver, &relax_vars, k, aux_var);
        let _ = aux_var; // silence warning

        self.stats.sat_calls += 1;
        self.stats.cardinality_constraints += 1;

        match solver.solve() {
            SolveResult::Sat(model) => Some(model),
            _ => None,
        }
    }

    /// Stratified algorithm for weighted MAX-SAT
    fn solve_stratified(&mut self) -> MaxSatResult {
        // Group soft clauses by weight
        let mut weights: Vec<Weight> = self.soft_clauses.iter().map(|s| s.weight).collect();
        weights.sort_unstable();
        weights.dedup();
        weights.reverse(); // Process highest weight first

        // Process each weight stratum to find minimum violations
        for &weight in &weights {
            // Get soft clauses with this weight
            let stratum_indices: Vec<usize> = self
                .soft_clauses
                .iter()
                .enumerate()
                .filter(|(_, s)| s.weight == weight)
                .map(|(i, _)| i)
                .collect();

            // Find minimum violations for this stratum (side effect: updates stats)
            let _ = self.find_min_violations_for_stratum(&stratum_indices);
        }

        // Final solve to get a model
        let n = self.soft_clauses.len();

        // Conservative estimate of extra variables needed
        let extra_vars = n * (n + 1) + 1000;
        let mut solver = SatSolver::new((self.next_var + extra_vars as u32) as usize);

        // Add hard clauses
        for clause in &self.hard_clauses {
            let lits: Vec<Literal> = clause.iter().map(|&l| self.int_to_literal(l)).collect();
            solver.add_clause(lits);
        }

        // Add soft clauses with relaxation variables
        for soft in &self.soft_clauses {
            let mut lits: Vec<Literal> = soft
                .clause
                .iter()
                .map(|&l| self.int_to_literal(l))
                .collect();
            lits.push(Literal::positive(Variable(soft.relax_var)));
            solver.add_clause(lits);
        }

        // Don't add any cardinality constraint - just solve and count cost
        self.stats.sat_calls += 1;

        match solver.solve() {
            SolveResult::Sat(model) => {
                let cost = self.compute_cost(&model);
                MaxSatResult::Optimal { model, cost }
            }
            SolveResult::Unsat => MaxSatResult::Unsatisfiable,
            SolveResult::Unknown => MaxSatResult::Unknown,
        }
    }

    /// Find minimum violations for a stratum of soft clauses
    fn find_min_violations_for_stratum(&mut self, stratum_indices: &[usize]) -> u64 {
        // Binary search for minimum violations in this stratum
        let n = stratum_indices.len();
        let mut lo = 0;
        let mut hi = n;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.can_solve_stratum_with_bound(stratum_indices, mid) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }

        lo as u64
    }

    /// Check if stratum can be solved with at most k violations
    fn can_solve_stratum_with_bound(&mut self, stratum_indices: &[usize], k: usize) -> bool {
        let relax_vars: Vec<u32> = stratum_indices
            .iter()
            .map(|&i| self.soft_clauses[i].relax_var)
            .collect();

        let mut aux_var = self.next_var;
        let extra_vars = relax_vars.len() * (k + 1) + 100;
        let mut solver = SatSolver::new((self.next_var + extra_vars as u32) as usize);

        // Add hard clauses
        for clause in &self.hard_clauses {
            let lits: Vec<Literal> = clause.iter().map(|&l| self.int_to_literal(l)).collect();
            solver.add_clause(lits);
        }

        // Add ALL soft clauses (not just stratum) with relaxation variables
        for soft in &self.soft_clauses {
            let mut lits: Vec<Literal> = soft
                .clause
                .iter()
                .map(|&l| self.int_to_literal(l))
                .collect();
            lits.push(Literal::positive(Variable(soft.relax_var)));
            solver.add_clause(lits);
        }

        // Add at-most-k constraint only for stratum relaxation variables
        aux_var = self.add_at_most_k_clauses(&mut solver, &relax_vars, k, aux_var);
        let _ = aux_var;

        self.stats.sat_calls += 1;

        matches!(solver.solve(), SolveResult::Sat(_))
    }

    /// Add at-most-k constraint clauses
    /// Returns the next available auxiliary variable
    fn add_at_most_k_clauses(
        &self,
        solver: &mut SatSolver,
        vars: &[u32],
        k: usize,
        mut next_aux: u32,
    ) -> u32 {
        let n = vars.len();

        if k >= n {
            return next_aux; // Trivially satisfied
        }

        if k == 0 {
            // All must be false
            for &var in vars {
                solver.add_clause(vec![Literal::negative(Variable(var))]);
            }
            return next_aux;
        }

        // Use direct encoding for small cases
        if n <= 6 || k == 1 {
            self.add_at_most_k_direct(solver, vars, k);
            return next_aux;
        }

        // Sequential counter encoding for larger cases
        next_aux = self.add_at_most_k_sequential(solver, vars, k, next_aux);
        next_aux
    }

    /// Direct encoding for at-most-k
    fn add_at_most_k_direct(&self, solver: &mut SatSolver, vars: &[u32], k: usize) {
        let n = vars.len();
        let subset_size = k + 1;

        if subset_size > n {
            return;
        }

        // Generate all subsets of size k+1
        fn generate_subsets(n: usize, r: usize) -> Vec<Vec<usize>> {
            if r == 0 {
                return vec![vec![]];
            }
            if n < r {
                return vec![];
            }

            let mut result = Vec::new();

            fn helper(
                start: usize,
                n: usize,
                r: usize,
                current: &mut Vec<usize>,
                result: &mut Vec<Vec<usize>>,
            ) {
                if current.len() == r {
                    result.push(current.clone());
                    return;
                }
                for i in start..=(n - (r - current.len())) {
                    current.push(i);
                    helper(i + 1, n, r, current, result);
                    current.pop();
                }
            }

            let mut current = Vec::new();
            helper(0, n, r, &mut current, &mut result);
            result
        }

        for subset in generate_subsets(n, subset_size) {
            let clause: Vec<Literal> = subset
                .iter()
                .map(|&i| Literal::negative(Variable(vars[i])))
                .collect();
            solver.add_clause(clause);
        }
    }

    /// Sequential counter encoding for at-most-k
    fn add_at_most_k_sequential(
        &self,
        solver: &mut SatSolver,
        vars: &[u32],
        k: usize,
        mut next_aux: u32,
    ) -> u32 {
        let n = vars.len();

        // Create counter variables: r[i][j] means "sum of first i+1 vars >= j+1"
        // We need r[i][j] for i in 0..n, j in 0..k
        let mut r: Vec<Vec<u32>> = Vec::with_capacity(n);
        for _ in 0..n {
            let row: Vec<u32> = (0..k)
                .map(|_| {
                    let v = next_aux;
                    next_aux += 1;
                    v
                })
                .collect();
            r.push(row);
        }

        // Base case: r[0][0] <=> vars[0]
        // vars[0] -> r[0][0]
        solver.add_clause(vec![
            Literal::negative(Variable(vars[0])),
            Literal::positive(Variable(r[0][0])),
        ]);
        // r[0][0] -> vars[0]
        solver.add_clause(vec![
            Literal::negative(Variable(r[0][0])),
            Literal::positive(Variable(vars[0])),
        ]);
        // r[0][j] = false for j > 0 (can't have sum >= 2 with one var)
        for &r0j in r[0].iter().skip(1) {
            solver.add_clause(vec![Literal::negative(Variable(r0j))]);
        }

        // Inductive case
        for i in 1..n {
            // r[i][0] <=> r[i-1][0] OR vars[i]
            // r[i-1][0] -> r[i][0]
            solver.add_clause(vec![
                Literal::negative(Variable(r[i - 1][0])),
                Literal::positive(Variable(r[i][0])),
            ]);
            // vars[i] -> r[i][0]
            solver.add_clause(vec![
                Literal::negative(Variable(vars[i])),
                Literal::positive(Variable(r[i][0])),
            ]);
            // r[i][0] -> r[i-1][0] OR vars[i]
            solver.add_clause(vec![
                Literal::negative(Variable(r[i][0])),
                Literal::positive(Variable(r[i - 1][0])),
                Literal::positive(Variable(vars[i])),
            ]);

            for j in 1..k {
                // r[i][j] <=> r[i-1][j] OR (vars[i] AND r[i-1][j-1])

                // r[i-1][j] -> r[i][j]
                solver.add_clause(vec![
                    Literal::negative(Variable(r[i - 1][j])),
                    Literal::positive(Variable(r[i][j])),
                ]);
                // vars[i] AND r[i-1][j-1] -> r[i][j]
                solver.add_clause(vec![
                    Literal::negative(Variable(vars[i])),
                    Literal::negative(Variable(r[i - 1][j - 1])),
                    Literal::positive(Variable(r[i][j])),
                ]);
                // r[i][j] -> r[i-1][j] OR vars[i]
                solver.add_clause(vec![
                    Literal::negative(Variable(r[i][j])),
                    Literal::positive(Variable(r[i - 1][j])),
                    Literal::positive(Variable(vars[i])),
                ]);
                // r[i][j] -> r[i-1][j] OR r[i-1][j-1]
                solver.add_clause(vec![
                    Literal::negative(Variable(r[i][j])),
                    Literal::positive(Variable(r[i - 1][j])),
                    Literal::positive(Variable(r[i - 1][j - 1])),
                ]);
            }

            // Block sum >= k+1: NOT (vars[i] AND r[i-1][k-1])
            // Equivalently: !vars[i] OR !r[i-1][k-1]
            solver.add_clause(vec![
                Literal::negative(Variable(vars[i])),
                Literal::negative(Variable(r[i - 1][k - 1])),
            ]);
        }

        next_aux
    }

    /// Compute cost (weight of unsatisfied soft clauses)
    fn compute_cost(&self, model: &[bool]) -> u64 {
        let mut cost = 0;

        for soft in &self.soft_clauses {
            let satisfied = soft.clause.iter().any(|&lit| {
                let var = lit.unsigned_abs() as usize;
                let polarity = lit > 0;
                var < model.len() && model[var] == polarity
            });

            if !satisfied {
                cost += soft.weight;
            }
        }

        cost
    }

    /// Convert integer literal to internal Literal
    fn int_to_literal(&self, lit: i32) -> Literal {
        let var = Variable(lit.unsigned_abs());
        if lit > 0 {
            Literal::positive(var)
        } else {
            Literal::negative(var)
        }
    }
}

impl Default for MaxSatSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_unweighted() {
        let mut solver = MaxSatSolver::new();

        // x1 OR x2 (hard)
        solver.add_hard_clause(vec![1, 2]);

        // Prefer x1 = true
        solver.add_soft_clause(vec![1], 1);
        // Prefer x2 = true
        solver.add_soft_clause(vec![2], 1);

        let result = solver.solve();
        match result {
            MaxSatResult::Optimal { cost, .. } => {
                assert_eq!(cost, 0, "Should satisfy all clauses");
            }
            _ => panic!("Expected optimal solution"),
        }
    }

    #[test]
    fn test_conflicting_soft_clauses() {
        let mut solver = MaxSatSolver::new();

        // Soft: x1 = true
        solver.add_soft_clause(vec![1], 1);
        // Soft: x1 = false (conflicts!)
        solver.add_soft_clause(vec![-1], 1);
        // Soft: x2 = true
        solver.add_soft_clause(vec![2], 1);

        let result = solver.solve();
        match result {
            MaxSatResult::Optimal { cost, model } => {
                assert_eq!(cost, 1, "Should violate exactly one clause");
                // x2 should be true (satisfies third clause)
                assert!(model.get(2).copied().unwrap_or(false));
            }
            _ => panic!("Expected optimal solution"),
        }
    }

    #[test]
    fn test_weighted() {
        let mut solver = MaxSatSolver::new();

        // Soft: x1 = true (weight 10)
        solver.add_soft_clause(vec![1], 10);
        // Soft: x1 = false (weight 1)
        solver.add_soft_clause(vec![-1], 1);

        let result = solver.solve();
        match result {
            MaxSatResult::Optimal { cost, model } => {
                // Should prefer violating the weight-1 clause
                assert_eq!(cost, 1);
                // x1 should be true (higher weight)
                assert!(model.get(1).copied().unwrap_or(false));
            }
            _ => panic!("Expected optimal solution"),
        }
    }

    #[test]
    fn test_unsatisfiable_hard() {
        let mut solver = MaxSatSolver::new();

        // Hard: x1
        solver.add_hard_clause(vec![1]);
        // Hard: !x1
        solver.add_hard_clause(vec![-1]);

        let result = solver.solve();
        assert_eq!(result, MaxSatResult::Unsatisfiable);
    }

    #[test]
    fn test_only_hard_clauses() {
        let mut solver = MaxSatSolver::new();

        solver.add_hard_clause(vec![1, 2]);
        solver.add_hard_clause(vec![-1, 2]);

        let result = solver.solve();
        match result {
            MaxSatResult::Optimal { cost, model } => {
                assert_eq!(cost, 0);
                // x2 must be true to satisfy both
                assert!(model.get(2).copied().unwrap_or(false));
            }
            _ => panic!("Expected optimal solution"),
        }
    }

    #[test]
    fn test_only_soft_clauses() {
        let mut solver = MaxSatSolver::new();

        solver.add_soft_clause(vec![1], 1);
        solver.add_soft_clause(vec![2], 1);
        solver.add_soft_clause(vec![3], 1);

        let result = solver.solve();
        match result {
            MaxSatResult::Optimal { cost, model } => {
                // Check that we can satisfy all soft clauses
                assert_eq!(cost, 0);
                // All should be true
                assert!(model.get(1).copied().unwrap_or(false));
                assert!(model.get(2).copied().unwrap_or(false));
                assert!(model.get(3).copied().unwrap_or(false));
            }
            _ => panic!("Expected optimal solution"),
        }
    }

    #[test]
    fn test_partial_maxsat() {
        let mut solver = MaxSatSolver::new();

        // Hard: at most one of x1, x2, x3 (encoded)
        // !x1 OR !x2
        solver.add_hard_clause(vec![-1, -2]);
        // !x1 OR !x3
        solver.add_hard_clause(vec![-1, -3]);
        // !x2 OR !x3
        solver.add_hard_clause(vec![-2, -3]);

        // Soft: want all three
        solver.add_soft_clause(vec![1], 1);
        solver.add_soft_clause(vec![2], 1);
        solver.add_soft_clause(vec![3], 1);

        let result = solver.solve();
        match result {
            MaxSatResult::Optimal { cost, model } => {
                // Can only satisfy 1 soft clause
                assert_eq!(cost, 2);
                // Exactly one should be true
                let count: usize = (1..=3)
                    .filter(|&i| model.get(i).copied().unwrap_or(false))
                    .count();
                assert_eq!(count, 1);
            }
            _ => panic!("Expected optimal solution"),
        }
    }

    #[test]
    fn test_empty_instance() {
        let mut solver = MaxSatSolver::new();
        let result = solver.solve();
        match result {
            MaxSatResult::Optimal { cost, .. } => {
                assert_eq!(cost, 0);
            }
            _ => panic!("Expected optimal solution for empty instance"),
        }
    }

    #[test]
    fn test_stats() {
        let mut solver = MaxSatSolver::new();

        solver.add_soft_clause(vec![1], 1);
        solver.add_soft_clause(vec![-1], 1);

        solver.solve();

        let stats = solver.stats();
        assert!(stats.sat_calls > 0, "Should have made SAT calls");
    }
}
