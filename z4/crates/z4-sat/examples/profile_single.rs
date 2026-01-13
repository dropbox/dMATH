//! Profile a single benchmark file for flamegraph analysis

use std::fs;
use z4_sat::{parse_dimacs, Solver};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("benchmarks/dimacs/uf200-052.cnf");
    let iterations: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);

    let content = fs::read_to_string(path).expect("read");
    let formula = parse_dimacs(&content).expect("parse");

    eprintln!("File: {}", path);
    eprintln!(
        "Variables: {}, Clauses: {}",
        formula.num_vars,
        formula.clauses.len()
    );
    eprintln!("Running {} iterations", iterations);

    for _ in 0..iterations {
        let mut solver = Solver::new(formula.num_vars);
        for c in &formula.clauses {
            solver.add_clause(c.clone());
        }
        let _ = solver.solve();
    }
}
