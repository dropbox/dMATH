//! Hard formula generators for proof complexity analysis.
//!
//! These formulas are known to require large proofs in certain proof systems.

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use z4_sat::{Literal, Variable};

use crate::graph::Graph;

/// A CNF formula represented as a list of clauses.
#[derive(Debug, Clone)]
pub struct Cnf {
    /// Number of variables
    num_vars: u32,
    /// Clauses (each clause is a list of literals)
    clauses: Vec<Vec<Literal>>,
}

impl Cnf {
    /// Create a new CNF with reserved capacity.
    pub fn new_with_capacity(num_vars: u32, clause_capacity: usize) -> Self {
        Self {
            num_vars,
            clauses: Vec::with_capacity(clause_capacity),
        }
    }

    /// Add a clause to the formula.
    pub fn add_clause(&mut self, literals: &[Literal]) {
        self.clauses.push(literals.to_vec());
    }

    /// Number of variables.
    pub fn num_vars(&self) -> usize {
        self.num_vars as usize
    }

    /// Number of clauses.
    pub fn num_clauses(&self) -> usize {
        self.clauses.len()
    }

    /// Iterate over clauses.
    pub fn clauses(&self) -> impl Iterator<Item = &Vec<Literal>> {
        self.clauses.iter()
    }

    /// Convert to a solver.
    pub fn into_solver(self) -> z4_sat::Solver {
        let mut solver = z4_sat::Solver::new(self.num_vars as usize);
        for clause in self.clauses {
            solver.add_clause(clause);
        }
        solver
    }
}

/// Alias for Variable (for cleaner code)
type Var = Variable;

/// Alias for Literal (for cleaner code)
type Lit = Literal;

/// Generate the Pigeonhole Principle formula: n+1 pigeons into n holes.
///
/// Variables: p_{i,j} means pigeon i is in hole j.
/// - For each pigeon i: at least one hole (OR_j p_{i,j})
/// - For each hole j and distinct pigeons i,k: at most one pigeon (NOT p_{i,j} OR NOT p_{k,j})
///
/// This formula is UNSATISFIABLE (n+1 pigeons can't fit in n holes).
///
/// **Proof complexity:**
/// - Resolution: exponential (2^{Omega(n)}) - Haken 1985
/// - Extended Resolution: polynomial O(n^3) - Cook 1976
///
/// # Arguments
/// * `n` - Number of holes (pigeons = n + 1)
///
/// # Example
/// ```
/// use z4_proof_complexity::hard_formulas::pigeonhole;
///
/// let php3 = pigeonhole(3);  // 4 pigeons, 3 holes
/// // 4 * 3 = 12 variables, many clauses
/// ```
pub fn pigeonhole(n: usize) -> Cnf {
    if n == 0 {
        // Edge case: 1 pigeon, 0 holes - trivially unsatisfiable
        return Cnf::new_with_capacity(1, 1);
    }

    let pigeons = n + 1;
    let holes = n;

    // Variable numbering: p_{i,j} -> i * holes + j (0-indexed)
    let var = |pigeon: usize, hole: usize| -> Var { Variable((pigeon * holes + hole) as u32) };

    // Estimate: pigeons clause count + C(pigeons, 2) * holes
    let clause_count = pigeons + (pigeons * (pigeons - 1) / 2) * holes;
    let mut cnf = Cnf::new_with_capacity((pigeons * holes) as u32, clause_count);

    // At-least-one clauses: each pigeon must be in some hole
    // For each pigeon i: (p_{i,0} OR p_{i,1} OR ... OR p_{i,n-1})
    for pigeon in 0..pigeons {
        let lits: Vec<Lit> = (0..holes)
            .map(|hole| Lit::positive(var(pigeon, hole)))
            .collect();
        cnf.add_clause(&lits);
    }

    // At-most-one clauses: each hole can have at most one pigeon
    // For each hole j and distinct pigeons i,k: (NOT p_{i,j} OR NOT p_{k,j})
    for hole in 0..holes {
        for pigeon1 in 0..pigeons {
            for pigeon2 in (pigeon1 + 1)..pigeons {
                cnf.add_clause(&[
                    Lit::negative(var(pigeon1, hole)),
                    Lit::negative(var(pigeon2, hole)),
                ]);
            }
        }
    }

    cnf
}

/// Generate Tseitin formula on a graph with given parities.
///
/// For each edge (u, v), we introduce a variable x_{u,v}.
/// For each vertex v with incident edges e_1, ..., e_k,
/// we add constraints encoding that XOR of x_{e_1}, ..., x_{e_k} = parity[v].
///
/// This is satisfiable iff the sum of parities is 0 (mod 2).
///
/// **Proof complexity:**
/// - Tree Resolution: exponential (due to XOR width)
/// - General Resolution: polynomial with extension variables
///
/// # Arguments
/// * `graph` - The underlying graph
/// * `parities` - Parity constraint for each vertex (vertex i has odd degree iff parities[i])
///
/// # Example
/// ```
/// use z4_proof_complexity::{hard_formulas::tseitin, Graph};
///
/// // Triangle graph
/// let g = Graph::complete(3);
/// let parities = vec![true, true, true];  // Odd parity at each vertex
/// let formula = tseitin(&g, &parities);
/// // UNSAT: sum of parities is odd (3 mod 2 = 1)
/// ```
pub fn tseitin(graph: &Graph, parities: &[bool]) -> Cnf {
    let n = graph.num_vertices();
    assert_eq!(parities.len(), n, "Need parity for each vertex");

    let edges: Vec<(usize, usize)> = graph.edges().collect();
    let num_edge_vars = edges.len();

    if num_edge_vars == 0 {
        // No edges: check if all parities are false
        if parities.iter().any(|&p| p) {
            // Unsatisfiable: some vertex has odd constraint but no edges
            let mut cnf = Cnf::new_with_capacity(1, 1);
            cnf.add_clause(&[]); // empty clause = UNSAT
            return cnf;
        } else {
            // Satisfiable: all parities are 0, trivially true
            return Cnf::new_with_capacity(0, 0);
        }
    }

    // Map edge to variable index
    let mut edge_to_var = std::collections::HashMap::new();
    for (idx, &(u, v)) in edges.iter().enumerate() {
        edge_to_var.insert((u.min(v), u.max(v)), Variable(idx as u32));
    }

    let get_edge_var =
        |u: usize, v: usize| -> Var { *edge_to_var.get(&(u.min(v), u.max(v))).unwrap() };

    // XOR encoding: for variables x_1, ..., x_k and parity p,
    // encode that x_1 XOR x_2 XOR ... XOR x_k = p
    // This uses 2^{k-1} clauses
    fn xor_to_cnf(vars: &[Var], parity: bool, cnf: &mut Cnf) {
        let k = vars.len();
        if k == 0 {
            if parity {
                cnf.add_clause(&[]); // UNSAT
            }
            return;
        }

        // Generate all 2^k assignments
        // Include those where XOR != parity (these are the disallowed assignments)
        for mask in 0..(1u64 << k) {
            let xor_result = (0..k).filter(|&i| (mask >> i) & 1 == 1).count() % 2 == 1;
            if xor_result != parity {
                // This assignment should be forbidden
                // Clause: at least one literal must be flipped
                let clause: Vec<Lit> = (0..k)
                    .map(|i| {
                        let is_true = (mask >> i) & 1 == 1;
                        if is_true {
                            Lit::negative(vars[i])
                        } else {
                            Lit::positive(vars[i])
                        }
                    })
                    .collect();
                cnf.add_clause(&clause);
            }
        }
    }

    // Estimate clause count (each vertex with k neighbors contributes 2^{k-1} clauses)
    let mut total_clauses = 0;
    for (v, &parity) in parities.iter().enumerate() {
        let degree = graph.degree(v);
        if degree > 0 {
            total_clauses += 1 << (degree - 1);
        } else if parity {
            total_clauses += 1;
        }
    }

    let mut cnf = Cnf::new_with_capacity(num_edge_vars as u32, total_clauses);

    // For each vertex, add XOR constraint
    for (v, &parity) in parities.iter().enumerate() {
        let neighbors: Vec<usize> = graph.neighbors(v).collect();
        if neighbors.is_empty() {
            if parity {
                cnf.add_clause(&[]); // UNSAT
            }
            continue;
        }

        let edge_vars: Vec<Var> = neighbors.iter().map(|&u| get_edge_var(v, u)).collect();

        xor_to_cnf(&edge_vars, parity, &mut cnf);
    }

    cnf
}

/// Generate a random k-CNF formula.
///
/// Each clause has exactly k literals, chosen uniformly at random.
/// The clause-to-variable ratio determines satisfiability:
/// - ratio < ~4.27 (for k=3): likely satisfiable
/// - ratio > ~4.27 (for k=3): likely unsatisfiable
///
/// **Proof complexity:**
/// - Near the threshold, formulas are hard for all known algorithms
///
/// # Arguments
/// * `k` - Clause width (typically 3)
/// * `n` - Number of variables
/// * `m` - Number of clauses
/// * `seed` - Random seed for reproducibility
///
/// # Example
/// ```
/// use z4_proof_complexity::hard_formulas::random_k_cnf;
///
/// // Random 3-SAT with 100 variables at threshold ratio
/// let formula = random_k_cnf(3, 100, 427, Some(42));
/// ```
pub fn random_k_cnf(k: usize, n: usize, m: usize, seed: Option<u64>) -> Cnf {
    if n == 0 || k == 0 {
        return Cnf::new_with_capacity(0, 0);
    }

    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    };

    let mut cnf = Cnf::new_with_capacity(n as u32, m);

    for _ in 0..m {
        // Sample k distinct variables
        let mut vars: Vec<usize> = (0..n).collect();
        vars.shuffle(&mut rng);
        let selected: Vec<usize> = vars.into_iter().take(k).collect();

        // Random polarity for each
        let clause: Vec<Lit> = selected
            .into_iter()
            .map(|v| {
                let var = Variable(v as u32);
                if rng.gen_bool(0.5) {
                    Lit::positive(var)
                } else {
                    Lit::negative(var)
                }
            })
            .collect();

        cnf.add_clause(&clause);
    }

    cnf
}

/// Generate the parity formula on n variables.
///
/// This encodes: x_1 XOR x_2 XOR ... XOR x_n = 1
///
/// **Proof complexity:**
/// - Resolution: exponential (2^{Omega(n)})
/// - Extended Resolution: polynomial (using auxiliary variables)
///
/// # Arguments
/// * `n` - Number of variables
///
/// # Example
/// ```
/// use z4_proof_complexity::hard_formulas::parity;
///
/// let formula = parity(4);  // x1 XOR x2 XOR x3 XOR x4 = 1
/// ```
pub fn parity(n: usize) -> Cnf {
    if n == 0 {
        // XOR of nothing = 0, constraint is that it equals 1
        let mut cnf = Cnf::new_with_capacity(0, 1);
        cnf.add_clause(&[]); // UNSAT
        return cnf;
    }

    let vars: Vec<Var> = (0..n).map(|i| Variable(i as u32)).collect();

    // Encode XOR = 1
    // Generate all 2^n assignments, forbid those where XOR = 0
    let clause_count = 1 << (n - 1);
    let mut cnf = Cnf::new_with_capacity(n as u32, clause_count);

    for mask in 0..(1u64 << n) {
        let xor_result = (0..n).filter(|&i| (mask >> i) & 1 == 1).count() % 2 == 1;
        if !xor_result {
            // XOR = 0, forbid this assignment
            let clause: Vec<Lit> = (0..n)
                .map(|i| {
                    let is_true = (mask >> i) & 1 == 1;
                    if is_true {
                        Lit::negative(vars[i])
                    } else {
                        Lit::positive(vars[i])
                    }
                })
                .collect();
            cnf.add_clause(&clause);
        }
    }

    cnf
}

/// Generate the ordering principle formula (OP).
///
/// The ordering principle states that every finite total order has a minimum element.
/// Variables: p_{i,j} means i < j in the ordering.
///
/// Constraints:
/// - Asymmetry: NOT(p_{i,j}) OR NOT(p_{j,i}) for all i != j
/// - Transitivity: NOT(p_{i,j}) OR NOT(p_{j,k}) OR p_{i,k} for all distinct i,j,k
/// - Totality: p_{i,j} OR p_{j,i} for all i < j
/// - No minimum: for each i, some j with p_{j,i}
///
/// This is UNSATISFIABLE.
///
/// **Proof complexity:**
/// - Resolution: polynomial O(n^4)
/// - However, tree-resolution requires 2^{Omega(n)}
///
/// # Arguments
/// * `n` - Number of elements to order
///
/// # Example
/// ```
/// use z4_proof_complexity::hard_formulas::ordering_principle;
///
/// let formula = ordering_principle(4);  // 4 elements with no minimum
/// ```
pub fn ordering_principle(n: usize) -> Cnf {
    if n <= 1 {
        // n=0 or n=1: the "no minimum" constraint can't be satisfied
        let mut cnf = Cnf::new_with_capacity(0, 1);
        cnf.add_clause(&[]); // UNSAT
        return cnf;
    }

    // Variable p_{i,j} for i != j: element i is less than element j
    // Numbering: (i, j) -> i * n + j (but skip diagonal i == j)
    let var = |i: usize, j: usize| -> Var {
        debug_assert!(i != j);
        let idx = i * n + j;
        Variable(idx as u32)
    };

    // Estimate clauses
    let pairs = n * (n - 1);
    let triples = n * (n - 1) * (n - 2);
    let clause_count = pairs / 2 + pairs + triples + n;

    let mut cnf = Cnf::new_with_capacity((n * n) as u32, clause_count);

    // Asymmetry: NOT(p_{i,j}) OR NOT(p_{j,i})
    for i in 0..n {
        for j in 0..n {
            if i < j {
                cnf.add_clause(&[Lit::negative(var(i, j)), Lit::negative(var(j, i))]);
            }
        }
    }

    // Totality: p_{i,j} OR p_{j,i}
    for i in 0..n {
        for j in (i + 1)..n {
            cnf.add_clause(&[Lit::positive(var(i, j)), Lit::positive(var(j, i))]);
        }
    }

    // Transitivity: NOT(p_{i,j}) OR NOT(p_{j,k}) OR p_{i,k}
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            for k in 0..n {
                if k == i || k == j {
                    continue;
                }
                cnf.add_clause(&[
                    Lit::negative(var(i, j)),
                    Lit::negative(var(j, k)),
                    Lit::positive(var(i, k)),
                ]);
            }
        }
    }

    // No minimum: for each i, there exists j such that j < i
    // This is: OR_{j != i} p_{j,i}
    for i in 0..n {
        let clause: Vec<Lit> = (0..n)
            .filter(|&j| j != i)
            .map(|j| Lit::positive(var(j, i)))
            .collect();
        cnf.add_clause(&clause);
    }

    cnf
}

/// Generate the graph coloring formula.
///
/// Variables: c_{v,k} means vertex v has color k.
/// - Each vertex has at least one color
/// - Adjacent vertices have different colors
///
/// # Arguments
/// * `graph` - The graph to color
/// * `k` - Number of colors
///
/// Returns SAT iff graph is k-colorable.
pub fn graph_coloring(graph: &Graph, k: usize) -> Cnf {
    let n = graph.num_vertices();
    if n == 0 || k == 0 {
        let mut cnf = Cnf::new_with_capacity(0, if k == 0 && n > 0 { 1 } else { 0 });
        if k == 0 && n > 0 {
            cnf.add_clause(&[]); // UNSAT: can't color with 0 colors
        }
        return cnf;
    }

    // Variable c_{v,c} -> v * k + c
    let var = |vertex: usize, color: usize| -> Var { Variable((vertex * k + color) as u32) };

    let edges: Vec<_> = graph.edges().collect();
    let clause_count = n + edges.len() * k;
    let mut cnf = Cnf::new_with_capacity((n * k) as u32, clause_count);

    // At-least-one: each vertex has some color
    for v in 0..n {
        let clause: Vec<Lit> = (0..k).map(|c| Lit::positive(var(v, c))).collect();
        cnf.add_clause(&clause);
    }

    // Different colors for adjacent vertices
    for (u, v) in edges {
        for c in 0..k {
            cnf.add_clause(&[Lit::negative(var(u, c)), Lit::negative(var(v, c))]);
        }
    }

    cnf
}

/// Generate the clique-coloring formula.
///
/// Encodes: G has a k-clique AND the complement of G is k-colorable.
/// This is a classic NP-hard problem used in proof complexity.
///
/// # Arguments
/// * `graph` - The graph
/// * `k` - Size of clique / number of colors
pub fn clique_coloring(graph: &Graph, k: usize) -> Cnf {
    let n = graph.num_vertices();
    if n == 0 || k == 0 {
        return Cnf::new_with_capacity(0, 0);
    }

    // Variables:
    // Clique: clique_{v,i} means vertex v is the i-th element of the clique
    // Color: color_{v,c} means vertex v has color c
    let clique_var = |v: usize, i: usize| -> Var { Variable((v * k + i) as u32) };
    let color_var = |v: usize, c: usize| -> Var { Variable((n * k + v * k + c) as u32) };

    let mut cnf = Cnf::new_with_capacity((2 * n * k) as u32, 0);

    // CLIQUE PART
    // Each position i in the clique is filled by some vertex
    for i in 0..k {
        let clause: Vec<Lit> = (0..n).map(|v| Lit::positive(clique_var(v, i))).collect();
        cnf.add_clause(&clause);
    }

    // Each vertex fills at most one position
    for v in 0..n {
        for i in 0..k {
            for j in (i + 1)..k {
                cnf.add_clause(&[
                    Lit::negative(clique_var(v, i)),
                    Lit::negative(clique_var(v, j)),
                ]);
            }
        }
    }

    // Clique constraint: if u is position i and v is position j, then u and v must be adjacent
    for i in 0..k {
        for j in (i + 1)..k {
            for u in 0..n {
                for v in 0..n {
                    if u != v && !graph.has_edge(u, v) {
                        cnf.add_clause(&[
                            Lit::negative(clique_var(u, i)),
                            Lit::negative(clique_var(v, j)),
                        ]);
                    }
                }
            }
        }
    }

    // COLORING PART (complement graph)
    // Each vertex has some color
    for v in 0..n {
        let clause: Vec<Lit> = (0..k).map(|c| Lit::positive(color_var(v, c))).collect();
        cnf.add_clause(&clause);
    }

    // Non-adjacent vertices (edges in complement) must have different colors
    for u in 0..n {
        for v in (u + 1)..n {
            if !graph.has_edge(u, v) {
                for c in 0..k {
                    cnf.add_clause(&[
                        Lit::negative(color_var(u, c)),
                        Lit::negative(color_var(v, c)),
                    ]);
                }
            }
        }
    }

    cnf
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solve(cnf: &Cnf) -> bool {
        use z4_sat::{SolveResult, Solver};
        let mut solver = Solver::new(cnf.num_vars());
        for clause in cnf.clauses() {
            solver.add_clause(clause.clone());
        }
        matches!(solver.solve(), SolveResult::Sat(_))
    }

    #[test]
    fn test_pigeonhole_small() {
        // PHP(1): 2 pigeons, 1 hole - UNSAT
        let php1 = pigeonhole(1);
        assert!(!solve(&php1));

        // PHP(2): 3 pigeons, 2 holes - UNSAT
        let php2 = pigeonhole(2);
        assert!(!solve(&php2));

        // PHP(3): 4 pigeons, 3 holes - UNSAT
        let php3 = pigeonhole(3);
        assert!(!solve(&php3));
    }

    #[test]
    fn test_pigeonhole_structure() {
        let php2 = pigeonhole(2);
        // 3 pigeons * 2 holes = 6 variables
        // 3 at-least-one clauses + C(3,2) * 2 = 3 + 6 = 9 clauses
        assert_eq!(php2.num_vars(), 6);
        assert_eq!(php2.num_clauses(), 9);
    }

    #[test]
    fn test_parity() {
        // XOR of 2 variables = 1: (x1 AND NOT x2) OR (NOT x1 AND x2)
        let p2 = parity(2);
        assert!(solve(&p2)); // SAT: x1=T, x2=F or x1=F, x2=T

        // XOR of 3 variables = 1
        let p3 = parity(3);
        assert!(solve(&p3)); // SAT

        // Structure: 2^{n-1} clauses
        assert_eq!(p2.num_clauses(), 2);
        assert_eq!(p3.num_clauses(), 4);
    }

    #[test]
    fn test_random_k_cnf_deterministic() {
        let f1 = random_k_cnf(3, 10, 20, Some(42));
        let f2 = random_k_cnf(3, 10, 20, Some(42));
        // Same seed should give same formula
        assert_eq!(f1.num_clauses(), f2.num_clauses());
    }

    #[test]
    fn test_random_k_cnf_structure() {
        let f = random_k_cnf(3, 100, 50, Some(123));
        assert_eq!(f.num_vars(), 100);
        assert_eq!(f.num_clauses(), 50);
    }

    #[test]
    fn test_ordering_principle_small() {
        // OP(2): 2 elements with no minimum - UNSAT
        let op2 = ordering_principle(2);
        assert!(!solve(&op2));

        // OP(3): 3 elements with no minimum - UNSAT
        let op3 = ordering_principle(3);
        assert!(!solve(&op3));
    }

    #[test]
    fn test_tseitin_satisfiable() {
        // Triangle with even parities everywhere
        let g = Graph::cycle(3);
        let parities = vec![false, false, false];
        let f = tseitin(&g, &parities);
        assert!(solve(&f)); // All edges = false satisfies all XOR = 0
    }

    #[test]
    fn test_tseitin_unsatisfiable() {
        // Triangle with odd parities - sum is 3 (odd), so UNSAT
        let g = Graph::cycle(3);
        let parities = vec![true, true, true];
        let f = tseitin(&g, &parities);
        assert!(!solve(&f));
    }

    #[test]
    fn test_graph_coloring_triangle() {
        let g = Graph::complete(3);

        // 2 colors: UNSAT (K3 needs 3 colors)
        let f2 = graph_coloring(&g, 2);
        assert!(!solve(&f2));

        // 3 colors: SAT
        let f3 = graph_coloring(&g, 3);
        assert!(solve(&f3));
    }

    #[test]
    fn test_graph_coloring_bipartite() {
        // Path graph is bipartite
        let g = Graph::path(4);

        // 2 colors: SAT (bipartite graphs are 2-colorable)
        let f2 = graph_coloring(&g, 2);
        assert!(solve(&f2));
    }
}
