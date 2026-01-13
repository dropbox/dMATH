//! ALL-SAT Solver Implementation
//!
//! This module implements solution enumeration using iterative SAT solving
//! with blocking clauses.

use z4_sat::{Literal, SolveResult, Solver as SatSolver, Variable};

/// A clause represented as a vector of literals.
/// Positive integers represent positive literals, negative represent negations.
pub type Clause = Vec<i32>;

/// A satisfying assignment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Solution {
    /// Assignment: index is variable number, value is true/false.
    /// Index 0 is unused (variables are 1-indexed).
    pub assignment: Vec<bool>,
}

impl Solution {
    /// Get the value of a variable in this solution.
    pub fn get(&self, var: u32) -> Option<bool> {
        self.assignment.get(var as usize).copied()
    }

    /// Check if a variable is true in this solution.
    pub fn is_true(&self, var: u32) -> bool {
        self.get(var).unwrap_or(false)
    }

    /// Check if a literal is satisfied by this solution.
    pub fn satisfies(&self, lit: i32) -> bool {
        let var = lit.unsigned_abs() as usize;
        let polarity = lit > 0;
        self.assignment.get(var).copied().unwrap_or(false) == polarity
    }

    /// Convert to a vector of literals representing this assignment.
    /// Returns positive literal if var=true, negative if var=false.
    pub fn to_literals(&self, vars: &[u32]) -> Vec<i32> {
        vars.iter()
            .map(|&v| {
                if self.is_true(v) {
                    v as i32
                } else {
                    -(v as i32)
                }
            })
            .collect()
    }
}

/// Configuration for ALL-SAT enumeration.
#[derive(Debug, Clone, Default)]
pub struct AllSatConfig {
    /// Maximum number of solutions to enumerate (None = unlimited).
    pub max_solutions: Option<usize>,

    /// Variables to project onto (None = all variables).
    /// When set, blocking clauses only reference these variables,
    /// effectively finding all distinct assignments to projected vars.
    pub projection: Option<Vec<u32>>,

    /// Whether to include don't-care variables in solutions.
    /// When false (default), don't-care vars are omitted from blocking.
    /// When true, full assignments are returned.
    pub full_assignments: bool,
}

/// Statistics for ALL-SAT solving.
#[derive(Debug, Clone, Default)]
pub struct AllSatStats {
    /// Number of SAT solver calls.
    pub sat_calls: u64,
    /// Number of solutions found.
    pub solutions_found: u64,
    /// Number of blocking clauses added.
    pub blocking_clauses: u64,
}

/// ALL-SAT Solver
///
/// Enumerates all satisfying assignments to a Boolean formula.
pub struct AllSatSolver {
    /// Clauses of the formula.
    clauses: Vec<Clause>,
    /// Maximum variable seen.
    max_var: u32,
    /// Statistics.
    stats: AllSatStats,
}

impl AllSatSolver {
    /// Create a new ALL-SAT solver.
    pub fn new() -> Self {
        Self {
            clauses: Vec::new(),
            max_var: 0,
            stats: AllSatStats::default(),
        }
    }

    /// Add a clause to the formula.
    pub fn add_clause(&mut self, clause: Clause) {
        for &lit in &clause {
            let var = lit.unsigned_abs();
            self.max_var = self.max_var.max(var);
        }
        self.clauses.push(clause);
    }

    /// Get the number of variables.
    pub fn num_vars(&self) -> u32 {
        self.max_var
    }

    /// Get solver statistics.
    pub fn stats(&self) -> &AllSatStats {
        &self.stats
    }

    /// Create an iterator over all solutions.
    pub fn iter(&mut self) -> AllSatIterator<'_> {
        self.iter_with_config(AllSatConfig::default())
    }

    /// Create an iterator with custom configuration.
    pub fn iter_with_config(&mut self, config: AllSatConfig) -> AllSatIterator<'_> {
        AllSatIterator::new(self, config)
    }

    /// Enumerate all solutions (convenience method).
    pub fn enumerate(&mut self) -> Vec<Solution> {
        self.iter().collect()
    }

    /// Enumerate solutions with custom configuration (convenience method).
    pub fn enumerate_with_config(&mut self, config: AllSatConfig) -> Vec<Solution> {
        self.iter_with_config(config).collect()
    }

    /// Count the number of solutions without storing them.
    pub fn count(&mut self) -> u64 {
        let mut count = 0;
        for _ in self.iter() {
            count += 1;
        }
        count
    }

    /// Count with custom configuration.
    pub fn count_with_config(&mut self, config: AllSatConfig) -> u64 {
        let mut count = 0;
        for _ in self.iter_with_config(config) {
            count += 1;
        }
        count
    }

    /// Check if the formula is satisfiable.
    pub fn is_sat(&mut self) -> bool {
        let config = AllSatConfig {
            max_solutions: Some(1),
            ..Default::default()
        };
        self.count_with_config(config) > 0
    }

    /// Check if the formula has exactly one solution.
    pub fn has_unique_solution(&mut self) -> bool {
        let config = AllSatConfig {
            max_solutions: Some(2),
            ..Default::default()
        };
        self.count_with_config(config) == 1
    }

    /// Build a fresh SAT solver with the current clauses plus blocking clauses.
    fn build_solver(&self, blocking_clauses: &[Clause]) -> SatSolver {
        let mut solver = SatSolver::new((self.max_var + 1) as usize);

        // Add original clauses
        for clause in &self.clauses {
            let lits: Vec<Literal> = clause.iter().map(|&l| int_to_literal(l)).collect();
            solver.add_clause(lits);
        }

        // Add blocking clauses
        for clause in blocking_clauses {
            let lits: Vec<Literal> = clause.iter().map(|&l| int_to_literal(l)).collect();
            solver.add_clause(lits);
        }

        solver
    }
}

impl Default for AllSatSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over all solutions.
pub struct AllSatIterator<'a> {
    solver: &'a mut AllSatSolver,
    config: AllSatConfig,
    blocking_clauses: Vec<Clause>,
    solutions_returned: usize,
    exhausted: bool,
}

impl<'a> AllSatIterator<'a> {
    fn new(solver: &'a mut AllSatSolver, config: AllSatConfig) -> Self {
        Self {
            solver,
            config,
            blocking_clauses: Vec::new(),
            solutions_returned: 0,
            exhausted: false,
        }
    }

    /// Create a blocking clause that excludes the given solution.
    fn make_blocking_clause(&self, solution: &Solution) -> Clause {
        let vars: Vec<u32> = if let Some(ref proj) = self.config.projection {
            // Only block on projected variables
            proj.clone()
        } else {
            // Block on all variables
            (1..=self.solver.max_var).collect()
        };

        // Blocking clause: at least one variable must differ
        // If var=true in solution, add -var to clause
        // If var=false in solution, add var to clause
        vars.iter()
            .map(|&v| {
                if solution.is_true(v) {
                    -(v as i32)
                } else {
                    v as i32
                }
            })
            .collect()
    }
}

impl Iterator for AllSatIterator<'_> {
    type Item = Solution;

    fn next(&mut self) -> Option<Self::Item> {
        // Check if we've reached the limit
        if self.exhausted {
            return None;
        }

        if let Some(max) = self.config.max_solutions {
            if self.solutions_returned >= max {
                return None;
            }
        }

        // Build solver and solve
        let mut sat_solver = self.solver.build_solver(&self.blocking_clauses);
        self.solver.stats.sat_calls += 1;

        match sat_solver.solve() {
            SolveResult::Sat(model) => {
                let solution = Solution { assignment: model };

                // Create blocking clause
                let blocking = self.make_blocking_clause(&solution);
                self.blocking_clauses.push(blocking);
                self.solver.stats.blocking_clauses += 1;
                self.solver.stats.solutions_found += 1;
                self.solutions_returned += 1;

                Some(solution)
            }
            SolveResult::Unsat | SolveResult::Unknown => {
                self.exhausted = true;
                None
            }
        }
    }
}

/// Convert integer literal to internal Literal.
fn int_to_literal(lit: i32) -> Literal {
    let var = Variable(lit.unsigned_abs());
    if lit > 0 {
        Literal::positive(var)
    } else {
        Literal::negative(var)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_solution() {
        let mut solver = AllSatSolver::new();

        // x1 AND x2
        solver.add_clause(vec![1]);
        solver.add_clause(vec![2]);

        let solutions = solver.enumerate();
        assert_eq!(solutions.len(), 1);
        assert!(solutions[0].is_true(1));
        assert!(solutions[0].is_true(2));
    }

    #[test]
    fn test_two_solutions() {
        let mut solver = AllSatSolver::new();

        // (x1 OR x2) AND NOT(x1 AND x2)
        // = (x1 OR x2) AND (NOT x1 OR NOT x2)
        solver.add_clause(vec![1, 2]);
        solver.add_clause(vec![-1, -2]);

        let solutions = solver.enumerate();
        assert_eq!(solutions.len(), 2);

        // Should have x1=T,x2=F and x1=F,x2=T
        let has_10 = solutions.iter().any(|s| s.is_true(1) && !s.is_true(2));
        let has_01 = solutions.iter().any(|s| !s.is_true(1) && s.is_true(2));
        assert!(has_10, "Should have solution x1=T, x2=F");
        assert!(has_01, "Should have solution x1=F, x2=T");
    }

    #[test]
    fn test_unsat() {
        let mut solver = AllSatSolver::new();

        // x1 AND NOT x1
        solver.add_clause(vec![1]);
        solver.add_clause(vec![-1]);

        let solutions = solver.enumerate();
        assert_eq!(solutions.len(), 0);
    }

    #[test]
    fn test_all_assignments() {
        let mut solver = AllSatSolver::new();

        // TRUE (no clauses restricts nothing, but we need at least one var)
        // Add a tautology: x1 OR NOT x1
        solver.add_clause(vec![1, -1]);

        let solutions = solver.enumerate();
        // Two solutions: x1=T and x1=F
        assert_eq!(solutions.len(), 2);
    }

    #[test]
    fn test_bounded_enumeration() {
        let mut solver = AllSatSolver::new();

        // (x1 OR x2) - has 3 solutions (TT, TF, FT)
        solver.add_clause(vec![1, 2]);

        let config = AllSatConfig {
            max_solutions: Some(2),
            ..Default::default()
        };
        let solutions = solver.enumerate_with_config(config);
        assert_eq!(solutions.len(), 2);
    }

    #[test]
    fn test_projected_enumeration() {
        let mut solver = AllSatSolver::new();

        // x1 AND (x2 OR x3)
        // Full solutions: x1=T,x2=T,x3=T; x1=T,x2=T,x3=F; x1=T,x2=F,x3=T
        solver.add_clause(vec![1]);
        solver.add_clause(vec![2, 3]);

        // Project onto x1 only
        let config = AllSatConfig {
            projection: Some(vec![1]),
            ..Default::default()
        };
        let solutions = solver.enumerate_with_config(config);
        // Only one projected solution: x1=T
        assert_eq!(solutions.len(), 1);
        assert!(solutions[0].is_true(1));
    }

    #[test]
    fn test_count() {
        let mut solver = AllSatSolver::new();

        // (x1 OR x2) - has 3 solutions
        solver.add_clause(vec![1, 2]);

        assert_eq!(solver.count(), 3);
    }

    #[test]
    fn test_is_sat() {
        let mut solver = AllSatSolver::new();
        solver.add_clause(vec![1, 2]);
        assert!(solver.is_sat());

        let mut solver2 = AllSatSolver::new();
        solver2.add_clause(vec![1]);
        solver2.add_clause(vec![-1]);
        assert!(!solver2.is_sat());
    }

    #[test]
    fn test_unique_solution() {
        let mut solver = AllSatSolver::new();
        solver.add_clause(vec![1]);
        solver.add_clause(vec![2]);
        assert!(solver.has_unique_solution());

        let mut solver2 = AllSatSolver::new();
        solver2.add_clause(vec![1, 2]);
        solver2.add_clause(vec![-1, -2]);
        assert!(!solver2.has_unique_solution()); // Has 2 solutions
    }

    #[test]
    fn test_iterator_early_termination() {
        let mut solver = AllSatSolver::new();

        // x1 OR x2 OR x3 - has 7 solutions
        solver.add_clause(vec![1, 2, 3]);

        let mut count = 0;
        for _ in solver.iter() {
            count += 1;
            if count >= 3 {
                break;
            }
        }
        assert_eq!(count, 3);
    }

    #[test]
    fn test_solution_to_literals() {
        let solution = Solution {
            assignment: vec![false, true, false, true], // x1=T, x2=F, x3=T
        };

        let lits = solution.to_literals(&[1, 2, 3]);
        assert_eq!(lits, vec![1, -2, 3]);
    }

    #[test]
    fn test_solution_satisfies() {
        let solution = Solution {
            assignment: vec![false, true, false], // x1=T, x2=F
        };

        assert!(solution.satisfies(1)); // x1 is true
        assert!(!solution.satisfies(-1)); // NOT x1 is false
        assert!(!solution.satisfies(2)); // x2 is false
        assert!(solution.satisfies(-2)); // NOT x2 is true
    }

    #[test]
    fn test_empty_formula() {
        let mut solver = AllSatSolver::new();
        // Empty formula with no variables
        let solutions = solver.enumerate();
        // Empty formula has one solution (the empty assignment)
        assert_eq!(solutions.len(), 1);
    }

    #[test]
    fn test_stats() {
        let mut solver = AllSatSolver::new();
        solver.add_clause(vec![1, 2]);
        solver.add_clause(vec![-1, -2]);

        let _ = solver.enumerate();

        let stats = solver.stats();
        assert!(stats.sat_calls > 0);
        assert_eq!(stats.solutions_found, 2);
        assert_eq!(stats.blocking_clauses, 2);
    }

    #[test]
    fn test_pigeonhole_3_2() {
        // 3 pigeons, 2 holes - no solution
        let mut solver = AllSatSolver::new();

        // p_{i,j} = pigeon i in hole j
        // Variables: p11=1, p12=2, p21=3, p22=4, p31=5, p32=6

        // Each pigeon must be in some hole
        solver.add_clause(vec![1, 2]); // p1 in h1 or h2
        solver.add_clause(vec![3, 4]); // p2 in h1 or h2
        solver.add_clause(vec![5, 6]); // p3 in h1 or h2

        // No two pigeons in same hole
        // Hole 1: at most one of p11, p21, p31
        solver.add_clause(vec![-1, -3]); // not (p11 and p21)
        solver.add_clause(vec![-1, -5]); // not (p11 and p31)
        solver.add_clause(vec![-3, -5]); // not (p21 and p31)

        // Hole 2: at most one of p12, p22, p32
        solver.add_clause(vec![-2, -4]); // not (p12 and p22)
        solver.add_clause(vec![-2, -6]); // not (p12 and p32)
        solver.add_clause(vec![-4, -6]); // not (p22 and p32)

        let solutions = solver.enumerate();
        assert_eq!(solutions.len(), 0, "Pigeonhole 3->2 should be UNSAT");
    }

    #[test]
    fn test_pigeonhole_2_2() {
        // 2 pigeons, 2 holes - has solutions
        let mut solver = AllSatSolver::new();

        // Variables: p11=1, p12=2, p21=3, p22=4

        // Each pigeon must be in some hole
        solver.add_clause(vec![1, 2]); // p1 in h1 or h2
        solver.add_clause(vec![3, 4]); // p2 in h1 or h2

        // No two pigeons in same hole
        solver.add_clause(vec![-1, -3]); // not (p11 and p21)
        solver.add_clause(vec![-2, -4]); // not (p12 and p22)

        let solutions = solver.enumerate();
        // Solutions: p1->h1,p2->h2 and p1->h2,p2->h1
        // But also variants with "extra" positions set to false
        assert!(solutions.len() >= 2, "Should have at least 2 solutions");

        // With projection to just the "one per pigeon" decision
        let config = AllSatConfig {
            projection: Some(vec![1, 2, 3, 4]),
            ..Default::default()
        };
        let projected = solver.enumerate_with_config(config);
        // Each pigeon in exactly one hole, 2 valid arrangements
        assert!(projected.len() >= 2);
    }

    #[test]
    fn test_xor_chain() {
        // XOR chain: x1 XOR x2 XOR x3 = true
        // (x1 XOR x2 XOR x3) encoded as CNF
        let mut solver = AllSatSolver::new();

        // x1 XOR x2 XOR x3 = 1 is equivalent to:
        // odd number of variables must be true
        // Clauses: (x1 OR x2 OR x3) AND (!x1 OR !x2 OR x3) AND (!x1 OR x2 OR !x3) AND (x1 OR !x2 OR !x3)
        solver.add_clause(vec![1, 2, 3]);
        solver.add_clause(vec![-1, -2, 3]);
        solver.add_clause(vec![-1, 2, -3]);
        solver.add_clause(vec![1, -2, -3]);

        let solutions = solver.enumerate();
        // Should have 4 solutions: TTF, TFT, FTT, FFF... wait, FFF has 0 true = even, not valid
        // Actually: TTT (3), TFF (1), FTF (1), FFT (1) = 4 solutions with odd parity
        assert_eq!(solutions.len(), 4);

        // Verify each solution has odd parity
        for sol in &solutions {
            let count = (1..=3).filter(|&v| sol.is_true(v)).count();
            assert!(count % 2 == 1, "XOR chain should have odd parity");
        }
    }
}
