//! Benchmark z4-sat on DIMACS files
//!
//! Usage: cargo run --release --example bench_dimacs -- [DIMACS_FILE...]
//!
//! Reads DIMACS CNF files, solves them with z4-sat, and reports solve timing.

use std::time::Instant;
use z4_sat::{parse_dimacs, Solver};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <dimacs_file>...", args[0]);
        eprintln!("       {} benchmarks/dimacs/*.cnf  (glob)", args[0]);
        std::process::exit(1);
    }

    let mut total_solve_time = 0.0;
    let mut total_overall_time = 0.0;
    let mut sat_count = 0;
    let mut unsat_count = 0;
    let mut error_count = 0;

    for path in &args[1..] {
        match std::fs::read_to_string(path) {
            Ok(content) => {
                let overall_start = Instant::now();
                match parse_dimacs(&content) {
                    Ok(formula) => {
                        let mut solver: Solver = formula.into_solver();
                        let solve_start = Instant::now();
                        let result = solver.solve();
                        let solve_elapsed = solve_start.elapsed().as_secs_f64();
                        let overall_elapsed = overall_start.elapsed().as_secs_f64();
                        total_solve_time += solve_elapsed;
                        total_overall_time += overall_elapsed;

                        let status = match &result {
                            z4_sat::SolveResult::Sat(_) => {
                                sat_count += 1;
                                "SAT"
                            }
                            z4_sat::SolveResult::Unsat => {
                                unsat_count += 1;
                                "UNSAT"
                            }
                            z4_sat::SolveResult::Unknown => "UNKNOWN",
                        };

                        // Print stats in verbose mode (set VERBOSE=1)
                        if std::env::var("VERBOSE").is_ok() {
                            println!(
                                "{:6} {:8.3}ms  total: {:8.3}ms  c:{:>8} d:{:>8} r:{:>6} p:{:>10}  {}",
                                status,
                                solve_elapsed * 1000.0,
                                overall_elapsed * 1000.0,
                                solver.num_conflicts(),
                                solver.num_decisions(),
                                solver.num_restarts(),
                                solver.num_propagations(),
                                path
                            );
                        } else {
                            println!(
                                "{:6} {:8.3}ms  total: {:8.3}ms  {}",
                                status,
                                solve_elapsed * 1000.0,
                                overall_elapsed * 1000.0,
                                path
                            );
                        }
                    }
                    Err(e) => {
                        error_count += 1;
                        eprintln!("PARSE ERROR: {} - {}", path, e);
                    }
                }
            }
            Err(e) => {
                error_count += 1;
                eprintln!("READ ERROR: {} - {}", path, e);
            }
        }
    }

    println!("\n--- Summary ---");
    println!(
        "Total: {} files, {} SAT, {} UNSAT, {} errors",
        args.len() - 1,
        sat_count,
        unsat_count,
        error_count
    );
    println!("Total solve time: {:.3}s", total_solve_time);
    println!("Total overall time: {:.3}s", total_overall_time);
    if sat_count + unsat_count > 0 {
        println!(
            "Average solve time: {:.3}ms",
            total_solve_time * 1000.0 / (sat_count + unsat_count) as f64
        );
    }
}
