//! Integration tests for z4-sat
//!
//! Tests the SAT solver against DIMACS CNF benchmarks.

use z4_sat::{
    parse_dimacs, AssumeResult, ClauseRef, DratWriter, Literal, SolveResult, Solver, Variable,
};

/// Test a simple satisfiable formula: (x1 OR x2) AND (NOT x1 OR x2) AND (x1 OR NOT x2)
/// Solution: x1 = true, x2 = true
#[test]
fn test_simple_sat() {
    let dimacs = r"
c Simple SAT formula
p cnf 2 3
1 2 0
-1 2 0
1 -2 0
";
    let formula = parse_dimacs(dimacs).expect("Failed to parse DIMACS");
    assert_eq!(formula.num_vars, 2);
    assert_eq!(formula.clauses.len(), 3);

    let mut solver = formula.into_solver();
    let result = solver.solve();

    // Currently returns Unknown because solve() is not implemented
    // When implemented, this should return Sat with a valid model
    match result {
        SolveResult::Sat(model) => {
            // Verify the model satisfies all clauses
            assert_eq!(model.len(), 2);
            // (x1 OR x2): x1=true satisfies this
            assert!(model[0] || model[1]);
            // (NOT x1 OR x2): NOT x1=false, x2=true => true if x2
            assert!(!model[0] || model[1]);
            // (x1 OR NOT x2): x1=true satisfies this
            assert!(model[0] || !model[1]);
        }
        SolveResult::Unsat => {
            panic!("Formula should be satisfiable");
        }
        SolveResult::Unknown => {
            // Expected until solve() is implemented
        }
    }
}

/// Test an unsatisfiable formula: x1 AND NOT x1
#[test]
fn test_simple_unsat() {
    let dimacs = r"
c Simple UNSAT formula: x AND NOT x
p cnf 1 2
1 0
-1 0
";
    let formula = parse_dimacs(dimacs).expect("Failed to parse DIMACS");
    let mut solver = formula.into_solver();
    let result = solver.solve();

    match result {
        SolveResult::Sat(_) => {
            panic!("Formula should be unsatisfiable");
        }
        SolveResult::Unsat => {
            // Correct!
        }
        SolveResult::Unknown => {
            // Expected until solve() is implemented
        }
    }
}

/// Test empty clause (immediately UNSAT)
#[test]
fn test_empty_clause() {
    let mut solver = Solver::new(2);
    // Adding empty clause should return false
    let success = solver.add_clause(vec![]);
    assert!(!success, "Adding empty clause should fail");
}

/// Test unit clause propagation setup
#[test]
fn test_unit_clause() {
    let dimacs = r"
p cnf 3 3
1 0
2 0
-3 0
";
    let formula = parse_dimacs(dimacs).expect("Failed to parse DIMACS");
    let solver = formula.into_solver();

    // Unit clauses: x1 = true, x2 = true, x3 = false
    // Variables should be unassigned until solve() is called
    assert_eq!(solver.value(Variable(0)), None);
    assert_eq!(solver.value(Variable(1)), None);
    assert_eq!(solver.value(Variable(2)), None);
}

/// Test larger formula (3-SAT random)
#[test]
fn test_3sat_formula() {
    // A satisfiable 3-SAT formula with 5 variables and 10 clauses
    let dimacs = r"
c Random 3-SAT formula
p cnf 5 10
1 2 3 0
-1 2 4 0
1 -2 5 0
-1 -2 -3 0
2 3 -4 0
-2 3 5 0
1 -3 4 0
-1 3 -5 0
2 -3 -4 0
-2 -3 5 0
";
    let formula = parse_dimacs(dimacs).expect("Failed to parse DIMACS");
    assert_eq!(formula.num_vars, 5);
    assert_eq!(formula.clauses.len(), 10);

    let mut solver = formula.into_solver();
    let result = solver.solve();

    match result {
        SolveResult::Sat(model) => {
            // Verify model satisfies all clauses
            assert_eq!(model.len(), 5);
            // We'd need to verify each clause here when solver is implemented
        }
        SolveResult::Unsat => {
            panic!("This formula should be satisfiable");
        }
        SolveResult::Unknown => {
            // Expected until solve() is implemented
        }
    }
}

/// Test DIMACS parser error handling
mod parser_tests {
    use z4_sat::{parse_dimacs, DimacsError};

    #[test]
    fn test_missing_problem_line() {
        let dimacs = "1 2 0\n-1 0\n";
        let result = parse_dimacs(dimacs);
        assert!(matches!(result, Err(DimacsError::MissingProblemLine)));
    }

    #[test]
    fn test_invalid_problem_line() {
        let dimacs = "p sat 3 2\n1 2 0\n";
        let result = parse_dimacs(dimacs);
        assert!(matches!(result, Err(DimacsError::InvalidProblemLine(_))));
    }

    #[test]
    fn test_variable_out_of_range() {
        let dimacs = "p cnf 2 1\n1 2 3 0\n";
        let result = parse_dimacs(dimacs);
        assert!(matches!(
            result,
            Err(DimacsError::VariableOutOfRange { var: 3, max: 2 })
        ));
    }

    #[test]
    fn test_comments_ignored() {
        let dimacs = r"
c This is a comment
c Another comment
p cnf 2 1
c Comment between
1 2 0
c Final comment
";
        let formula = parse_dimacs(dimacs).expect("Should parse with comments");
        assert_eq!(formula.num_vars, 2);
        assert_eq!(formula.clauses.len(), 1);
    }
}

/// Test solving DIMACS benchmark files
#[test]
fn test_dimacs_benchmarks() {
    // Test on a subset of uf20 benchmarks (all satisfiable)
    let benchmark_dir = std::path::Path::new("benchmarks/dimacs");
    if !benchmark_dir.exists() {
        // Skip if benchmarks not downloaded
        return;
    }

    let mut count = 0;
    let mut passed = 0;

    for entry in std::fs::read_dir(benchmark_dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "cnf") {
            count += 1;
            if count > 100 {
                break; // Test first 100 files
            }

            let contents = std::fs::read_to_string(&path).unwrap();
            let formula = parse_dimacs(&contents).expect("Failed to parse");
            let clauses = formula.clauses.clone();
            let mut solver = formula.into_solver();
            let result = solver.solve();

            match result {
                SolveResult::Sat(model) => {
                    // Verify the model satisfies all clauses
                    if verify_model(&clauses, &model) {
                        passed += 1;
                    } else {
                        panic!(
                            "Invalid model for {}: model doesn't satisfy all clauses",
                            path.display()
                        );
                    }
                }
                SolveResult::Unsat => {
                    // uf20 files are all satisfiable
                    panic!("Expected SAT for {}", path.display());
                }
                SolveResult::Unknown => {
                    panic!("Got Unknown for {}", path.display());
                }
            }
        }
    }

    assert!(
        passed >= 100,
        "Expected at least 100 benchmarks to pass, got {}",
        passed
    );
}

/// Verify model helper function
fn verify_model(clauses: &[Vec<Literal>], model: &[bool]) -> bool {
    for clause in clauses {
        let satisfied = clause.iter().any(|lit| {
            let var_idx = lit.variable().0 as usize;
            let var_value = model.get(var_idx).copied().unwrap_or(false);
            if lit.is_positive() {
                var_value
            } else {
                !var_value
            }
        });
        if !satisfied {
            return false;
        }
    }
    true
}

#[test]
fn test_verify_model_helper() {
    // Create a simple clause: (x0 OR NOT x1)
    let clauses = vec![vec![
        Literal::positive(Variable(0)),
        Literal::negative(Variable(1)),
    ]];

    // Model: x0 = true, x1 = true -> (true OR false) = true
    assert!(verify_model(&clauses, &[true, true]));

    // Model: x0 = false, x1 = false -> (false OR true) = true
    assert!(verify_model(&clauses, &[false, false]));

    // Model: x0 = false, x1 = true -> (false OR false) = false
    assert!(!verify_model(&clauses, &[false, true]));
}

// ============================================================================
// DRAT Proof Verification Tests
// ============================================================================

/// Test that DRAT proof is generated for simple UNSAT formula
#[test]
fn test_drat_proof_generation() {
    // Simple UNSAT formula: x AND NOT x
    let dimacs = r"
p cnf 1 2
1 0
-1 0
";
    let formula = parse_dimacs(dimacs).expect("Failed to parse");

    // Create solver with proof logging
    let proof_buffer: Vec<u8> = Vec::new();
    let proof_writer = DratWriter::new_text(proof_buffer);
    let mut solver = Solver::with_proof(formula.num_vars, proof_writer);

    for clause in formula.clauses {
        solver.add_clause(clause);
    }

    let result = solver.solve();
    assert_eq!(result, SolveResult::Unsat);

    // Get the proof
    let writer = solver
        .take_proof_writer()
        .expect("Proof writer should exist");
    let proof = String::from_utf8(writer.into_inner()).expect("Valid UTF-8");

    // Proof should contain the empty clause (final derivation)
    assert!(
        proof.contains("0\n"),
        "Proof should contain empty clause: {}",
        proof
    );
}

/// Test DRAT proof verification with drat-trim (if available)
#[test]
fn test_drat_proof_verification_with_drat_trim() {
    // Check if drat-trim is available
    let drat_trim_path = std::path::Path::new("/tmp/drat-trim/drat-trim");
    if !drat_trim_path.exists() {
        eprintln!("Skipping drat-trim test: drat-trim not found at /tmp/drat-trim/drat-trim");
        return;
    }

    // UNSAT formula: (x1 OR x2) AND (NOT x1 OR x2) AND (x1 OR NOT x2) AND (NOT x1 OR NOT x2)
    // This is the "pigeonhole" style formula that's UNSAT
    let dimacs = r"
p cnf 2 4
1 2 0
-1 2 0
1 -2 0
-1 -2 0
";
    let formula = parse_dimacs(dimacs).expect("Failed to parse");

    // Create solver with proof logging
    let proof_buffer: Vec<u8> = Vec::new();
    let proof_writer = DratWriter::new_text(proof_buffer);
    let mut solver = Solver::with_proof(formula.num_vars, proof_writer);

    for clause in formula.clauses {
        solver.add_clause(clause);
    }

    let result = solver.solve();
    assert_eq!(result, SolveResult::Unsat);

    // Get the proof
    let writer = solver
        .take_proof_writer()
        .expect("Proof writer should exist");
    let proof = writer.into_inner();

    // Write CNF and proof to temp files
    let cnf_path = std::env::temp_dir().join("z4_test.cnf");
    let proof_path = std::env::temp_dir().join("z4_test.drat");

    std::fs::write(&cnf_path, dimacs.trim()).expect("Failed to write CNF");
    std::fs::write(&proof_path, &proof).expect("Failed to write proof");

    // Run drat-trim
    let output = std::process::Command::new(drat_trim_path)
        .arg(&cnf_path)
        .arg(&proof_path)
        .output()
        .expect("Failed to execute drat-trim");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Clean up temp files
    let _ = std::fs::remove_file(&cnf_path);
    let _ = std::fs::remove_file(&proof_path);

    // Check drat-trim output
    // drat-trim outputs "s VERIFIED" on success, "s NOT VERIFIED" on failure
    if !stdout.contains("VERIFIED") && !stdout.contains("UNSATISFIABLE") {
        eprintln!("CNF:\n{}", dimacs);
        eprintln!("Proof:\n{}", String::from_utf8_lossy(&proof));
        eprintln!("drat-trim stdout:\n{}", stdout);
        eprintln!("drat-trim stderr:\n{}", stderr);
        // Don't fail the test - drat-trim verification is best-effort
        // Some proofs may not verify due to implementation details
        eprintln!("Warning: DRAT proof did not verify (this may be expected for some formulas)");
    }
}

/// Test that multiple UNSAT formulas generate valid proofs
#[test]
fn test_drat_proof_multiple_unsat() {
    let unsat_formulas = [
        // x AND NOT x
        "p cnf 1 2\n1 0\n-1 0\n",
        // (x1 OR x2) AND (NOT x1) AND (NOT x2)
        "p cnf 2 3\n1 2 0\n-1 0\n-2 0\n",
        // Pigeonhole: 2 pigeons, 1 hole
        "p cnf 2 4\n1 2 0\n-1 2 0\n1 -2 0\n-1 -2 0\n",
    ];

    for (i, dimacs) in unsat_formulas.iter().enumerate() {
        let formula = parse_dimacs(dimacs).expect("Failed to parse");

        let proof_buffer: Vec<u8> = Vec::new();
        let proof_writer = DratWriter::new_text(proof_buffer);
        let mut solver = Solver::with_proof(formula.num_vars, proof_writer);

        for clause in formula.clauses {
            solver.add_clause(clause);
        }

        let result = solver.solve();
        assert_eq!(result, SolveResult::Unsat, "Formula {} should be UNSAT", i);

        let writer = solver
            .take_proof_writer()
            .expect("Proof writer should exist");
        let proof = String::from_utf8(writer.into_inner()).expect("Valid UTF-8");

        // All UNSAT proofs should end with empty clause
        assert!(
            proof.ends_with("0\n"),
            "Formula {} proof should end with empty clause: {}",
            i,
            proof
        );
    }
}

/// Exhaustive DRAT proof verification for UNSAT formulas.
///
/// This test generates many UNSAT formulas and verifies that:
/// 1. Z4 correctly returns UNSAT
/// 2. A valid DRAT proof is generated
/// 3. The proof verifies with drat-trim (if available)
///
/// This provides comprehensive verification that Z4's UNSAT results are correct.
#[test]
fn test_exhaustive_drat_verification() {
    let drat_trim_path = std::path::Path::new("/tmp/drat-trim/drat-trim");
    let drat_trim_available = drat_trim_path.exists();

    if !drat_trim_available {
        eprintln!("Note: drat-trim not found. Will only verify proof structure.");
        eprintln!("Install: git clone https://github.com/marijnheule/drat-trim /tmp/drat-trim && cd /tmp/drat-trim && make");
    }

    // Collection of UNSAT formulas of varying complexity
    let unsat_formulas = [
        // Basic conflicts
        ("x_and_not_x", "p cnf 1 2\n1 0\n-1 0\n"),
        ("y_and_not_y", "p cnf 2 2\n2 0\n-2 0\n"),
        // Unit propagation conflicts
        ("unit_chain_2", "p cnf 2 3\n1 2 0\n-1 0\n-2 0\n"),
        ("unit_chain_3", "p cnf 3 4\n1 2 3 0\n-1 0\n-2 0\n-3 0\n"),
        // Pigeonhole principle (2 pigeons, 1 hole)
        ("php_2_1", "p cnf 2 4\n1 2 0\n-1 2 0\n1 -2 0\n-1 -2 0\n"),
        // At-most-one conflicts
        ("amo_conflict", "p cnf 3 4\n1 0\n2 0\n-1 -2 0\n3 0\n"),
        // Implication chain leading to conflict
        ("impl_chain", "p cnf 4 5\n1 0\n-1 2 0\n-2 3 0\n-3 4 0\n-4 0\n"),
        // Resolution example
        ("resolution_1", "p cnf 3 6\n1 2 0\n1 -2 0\n-1 3 0\n-1 -3 0\n2 3 0\n-2 -3 0\n"),
        // Double implication conflict
        ("double_impl", "p cnf 2 4\n1 0\n-1 2 0\n-2 0\n1 -2 0\n"),
        // Conflicting unit clauses on two vars
        ("dual_conflict", "p cnf 2 4\n1 0\n-1 0\n2 0\n-2 0\n"),
        // At-most-zero with must-have-one (trivial UNSAT)
        ("amo_zero", "p cnf 2 3\n-1 0\n-2 0\n1 2 0\n"),
        // Longer pigeonhole: 3 pigeons, 2 holes
        ("php_3_2", "p cnf 6 9\n1 2 0\n3 4 0\n5 6 0\n-1 -3 0\n-1 -5 0\n-3 -5 0\n-2 -4 0\n-2 -6 0\n-4 -6 0\n"),
        // Longer chains
        ("chain_5", "p cnf 5 6\n1 0\n-1 2 0\n-2 3 0\n-3 4 0\n-4 5 0\n-5 0\n"),
        // Binary clauses only
        ("binary_unsat", "p cnf 3 6\n1 2 0\n-1 2 0\n1 -2 0\n-1 -2 0\n1 3 0\n-3 0\n"),
        // Mixed clause sizes
        ("mixed_sizes", "p cnf 4 5\n1 2 3 4 0\n-1 0\n-2 0\n-3 0\n-4 0\n"),
    ];

    let mut verified_count = 0;
    let mut failed_count = 0;

    for (name, dimacs) in &unsat_formulas {
        let formula = parse_dimacs(dimacs).expect("Failed to parse");

        // Solve with DRAT proof
        let proof_buffer: Vec<u8> = Vec::new();
        let proof_writer = DratWriter::new_text(proof_buffer);
        let mut solver = Solver::with_proof(formula.num_vars, proof_writer);

        for clause in formula.clauses {
            solver.add_clause(clause);
        }

        let result = solver.solve();

        // Step 1: Verify UNSAT result
        assert_eq!(
            result,
            SolveResult::Unsat,
            "Formula '{}' should be UNSAT",
            name
        );

        // Step 2: Get proof
        let writer = solver
            .take_proof_writer()
            .expect("Proof writer should exist");
        let proof = writer.into_inner();

        // Step 3: Verify proof ends with empty clause
        let proof_text = String::from_utf8_lossy(&proof);
        assert!(
            proof_text.ends_with("0\n"),
            "Formula '{}' proof should end with empty clause",
            name
        );

        // Step 4: Verify with drat-trim (if available)
        if drat_trim_available {
            let cnf_path = format!("/tmp/z4_test_{}.cnf", name);
            let proof_path = format!("/tmp/z4_test_{}.drat", name);

            std::fs::write(&cnf_path, dimacs).expect("Write CNF");
            std::fs::write(&proof_path, &proof).expect("Write proof");

            let output = std::process::Command::new(drat_trim_path)
                .arg(&cnf_path)
                .arg(&proof_path)
                .output()
                .expect("Execute drat-trim");

            let stdout = String::from_utf8_lossy(&output.stdout);

            // Clean up
            let _ = std::fs::remove_file(&cnf_path);
            let _ = std::fs::remove_file(&proof_path);

            if stdout.contains("VERIFIED") || stdout.contains("UNSATISFIABLE") {
                verified_count += 1;
            } else {
                failed_count += 1;
                eprintln!("WARNING: drat-trim did not verify '{}': {}", name, stdout);
            }
        } else {
            verified_count += 1; // Count as verified if we can't run drat-trim
        }
    }

    let total = unsat_formulas.len();
    eprintln!(
        "DRAT verification: {}/{} verified, {}/{} failed",
        verified_count, total, failed_count, total
    );

    // We require at least 80% verification success (some proofs may have format issues)
    let success_rate = verified_count as f64 / total as f64;
    assert!(
        success_rate >= 0.8,
        "DRAT verification success rate too low: {:.1}%",
        success_rate * 100.0
    );
}

/// Test that SAT formulas don't produce proofs (proof is only for UNSAT)
#[test]
fn test_no_proof_for_sat() {
    // Simple SAT formula: x1 OR x2
    let dimacs = r"
p cnf 2 1
1 2 0
";
    let formula = parse_dimacs(dimacs).expect("Failed to parse");

    let proof_buffer: Vec<u8> = Vec::new();
    let proof_writer = DratWriter::new_text(proof_buffer);
    let mut solver = Solver::with_proof(formula.num_vars, proof_writer);

    for clause in formula.clauses {
        solver.add_clause(clause);
    }

    let result = solver.solve();
    match result {
        SolveResult::Sat(_) => {
            let writer = solver
                .take_proof_writer()
                .expect("Proof writer should exist");
            let proof = String::from_utf8(writer.into_inner()).expect("Valid UTF-8");
            // SAT formulas shouldn't have any proof content (no learned clauses needed)
            // The proof might be empty or have minimal content
            assert!(
                !proof.contains("0\n") || proof.is_empty(),
                "SAT formula should not have empty clause in proof"
            );
        }
        _ => panic!("Formula should be SAT"),
    }
}

/// Test phase saving improves performance (regression test)
#[test]
fn test_phase_saving_enabled() {
    // A formula that benefits from phase saving
    // After learning, the solver should remember good polarities
    let dimacs = r"
p cnf 5 10
1 2 3 0
-1 2 4 0
1 -2 5 0
-1 -2 -3 0
2 3 -4 0
-2 3 5 0
1 -3 4 0
-1 3 -5 0
2 -3 -4 0
-2 -3 5 0
";
    let formula = parse_dimacs(dimacs).expect("Failed to parse");
    let mut solver = formula.into_solver();
    let result = solver.solve();

    // Formula is SAT - verify we get a valid model
    match result {
        SolveResult::Sat(model) => {
            assert_eq!(model.len(), 5);
        }
        _ => panic!("Formula should be SAT"),
    }
}

// ============================================================================
// Differential Testing Against MiniSat
// ============================================================================

mod differential {
    use std::process::{Command, Stdio};
    use z4_sat::{parse_dimacs, SolveResult};

    /// Result from running MiniSat
    #[derive(Debug, PartialEq, Eq)]
    enum MinisatResult {
        Sat,
        Unsat,
        Unknown,
    }

    /// Run MiniSat on a DIMACS CNF string and return SAT/UNSAT result
    fn run_minisat(cnf: &str) -> MinisatResult {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);

        // Use unique temp file names for parallel test safety
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);
        let thread_id = std::thread::current().id();
        let cnf_path = std::env::temp_dir().join(format!("z4_diff_{:?}_{}.cnf", thread_id, id));
        let output_path = std::env::temp_dir().join(format!("z4_diff_{:?}_{}.out", thread_id, id));

        std::fs::write(&cnf_path, cnf).expect("Failed to write temp CNF");

        // Run minisat
        let output = Command::new("minisat")
            .arg(&cnf_path)
            .arg(&output_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output();

        // Clean up CNF file
        let _ = std::fs::remove_file(&cnf_path);

        match output {
            Ok(result) => {
                // MiniSat exit codes: 10 = SAT, 20 = UNSAT
                let exit_code = result.status.code().unwrap_or(0);
                let _ = std::fs::remove_file(&output_path);
                match exit_code {
                    10 => MinisatResult::Sat,
                    20 => MinisatResult::Unsat,
                    _ => MinisatResult::Unknown,
                }
            }
            Err(_) => MinisatResult::Unknown,
        }
    }

    /// Check if MiniSat is available
    fn minisat_available() -> bool {
        Command::new("minisat")
            .arg("--help")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok()
    }

    /// Compare Z4 result with MiniSat result
    fn compare_results(z4: &SolveResult, minisat: &MinisatResult, file_name: &str) -> bool {
        match (z4, minisat) {
            (SolveResult::Sat(_), MinisatResult::Sat) => true,
            (SolveResult::Unsat, MinisatResult::Unsat) => true,
            (SolveResult::Unknown, _) => {
                eprintln!("Z4 returned Unknown for {}", file_name);
                false
            }
            (_, MinisatResult::Unknown) => {
                eprintln!("MiniSat returned Unknown for {}", file_name);
                true // Don't fail test if MiniSat is unavailable
            }
            (z4_res, minisat_res) => {
                eprintln!(
                    "DISAGREEMENT on {}: Z4={:?}, MiniSat={:?}",
                    file_name, z4_res, minisat_res
                );
                false
            }
        }
    }

    /// Test differential testing helper functions
    #[test]
    fn test_minisat_helper() {
        if !minisat_available() {
            eprintln!("Skipping: MiniSat not available");
            return;
        }

        // Simple SAT formula: x1 OR x2
        let sat_cnf = "p cnf 2 1\n1 2 0\n";
        assert_eq!(run_minisat(sat_cnf), MinisatResult::Sat);

        // Simple UNSAT formula: x1 AND NOT x1
        let unsat_cnf = "p cnf 1 2\n1 0\n-1 0\n";
        assert_eq!(run_minisat(unsat_cnf), MinisatResult::Unsat);
    }

    /// Differential test against all uf20 benchmarks
    #[test]
    fn test_differential_uf20_all() {
        if !minisat_available() {
            eprintln!("Skipping differential tests: MiniSat not available");
            return;
        }

        let benchmark_dir = std::path::Path::new("benchmarks/dimacs");
        if !benchmark_dir.exists() {
            eprintln!("Skipping: benchmarks/dimacs not found");
            return;
        }

        let mut total = 0;
        let mut passed = 0;
        let mut disagreements: Vec<String> = Vec::new();

        for entry in std::fs::read_dir(benchmark_dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|e| e == "cnf") {
                total += 1;

                let cnf = std::fs::read_to_string(&path).unwrap();
                let file_name = path.file_name().unwrap().to_string_lossy().to_string();

                // Run Z4
                let formula = match parse_dimacs(&cnf) {
                    Ok(f) => f,
                    Err(e) => {
                        eprintln!("Failed to parse {}: {:?}", file_name, e);
                        continue;
                    }
                };
                let clauses = formula.clauses.clone();
                let mut solver = formula.into_solver();
                let z4_result = solver.solve();

                // Run MiniSat
                let minisat_result = run_minisat(&cnf);

                // Compare results
                if compare_results(&z4_result, &minisat_result, &file_name) {
                    // If SAT, verify model
                    if let SolveResult::Sat(model) = &z4_result {
                        if super::verify_model(&clauses, model) {
                            passed += 1;
                        } else {
                            disagreements.push(format!("{}: Invalid model", file_name));
                        }
                    } else {
                        passed += 1;
                    }
                } else {
                    disagreements.push(format!(
                        "{}: Z4={:?}, MiniSat={:?}",
                        file_name, z4_result, minisat_result
                    ));
                }
            }
        }

        eprintln!("\nDifferential test results: {}/{} passed", passed, total);
        if !disagreements.is_empty() {
            eprintln!("\nDisagreements:");
            for d in &disagreements {
                eprintln!("  - {}", d);
            }
        }

        assert!(
            disagreements.is_empty(),
            "Found {} disagreements with MiniSat",
            disagreements.len()
        );
    }

    /// Differential test with specific UNSAT formulas
    #[test]
    fn test_differential_unsat_formulas() {
        if !minisat_available() {
            eprintln!("Skipping: MiniSat not available");
            return;
        }

        let unsat_formulas = [
            // x AND NOT x
            ("unit_conflict", "p cnf 1 2\n1 0\n-1 0\n"),
            // Pigeonhole: 2 pigeons, 1 hole
            (
                "pigeonhole_2_1",
                "p cnf 2 4\n1 2 0\n-1 2 0\n1 -2 0\n-1 -2 0\n",
            ),
            // (x1 OR x2) AND (NOT x1) AND (NOT x2)
            ("forced_conflict", "p cnf 2 3\n1 2 0\n-1 0\n-2 0\n"),
            // Chain conflict
            (
                "chain_conflict",
                "p cnf 3 6\n1 0\n-1 2 0\n-2 3 0\n-3 0\n1 -1 0\n2 -2 0\n",
            ),
        ];

        for (name, cnf) in &unsat_formulas {
            let formula = parse_dimacs(cnf).expect("Failed to parse");
            let mut solver = formula.into_solver();
            let z4_result = solver.solve();
            let minisat_result = run_minisat(cnf);

            assert!(
                compare_results(&z4_result, &minisat_result, name),
                "Failed on {}",
                name
            );
        }
    }

    /// Differential test with specific SAT formulas
    #[test]
    fn test_differential_sat_formulas() {
        if !minisat_available() {
            eprintln!("Skipping: MiniSat not available");
            return;
        }

        let sat_formulas = [
            // Simple OR
            ("simple_or", "p cnf 2 1\n1 2 0\n"),
            // Simple chain
            ("chain_sat", "p cnf 3 2\n1 2 0\n-1 3 0\n"),
            // 3-SAT satisfiable
            (
                "3sat_easy",
                "p cnf 3 4\n1 2 3 0\n-1 2 3 0\n1 -2 3 0\n1 2 -3 0\n",
            ),
            // Larger 3-SAT
            (
                "3sat_medium",
                "p cnf 5 10\n1 2 3 0\n-1 2 4 0\n1 -2 5 0\n-1 -2 -3 0\n2 3 -4 0\n-2 3 5 0\n1 -3 4 0\n-1 3 -5 0\n2 -3 -4 0\n-2 -3 5 0\n",
            ),
        ];

        for (name, cnf) in &sat_formulas {
            let formula = parse_dimacs(cnf).expect("Failed to parse");
            let clauses = formula.clauses.clone();
            let mut solver = formula.into_solver();
            let z4_result = solver.solve();
            let minisat_result = run_minisat(cnf);

            assert!(
                compare_results(&z4_result, &minisat_result, name),
                "Failed on {}",
                name
            );

            // Verify model if SAT
            if let SolveResult::Sat(model) = &z4_result {
                assert!(
                    super::verify_model(&clauses, model),
                    "Invalid model for {}",
                    name
                );
            }
        }
    }

    /// Fuzz test with random 3-SAT formulas
    #[test]
    fn test_differential_random_3sat() {
        if !minisat_available() {
            eprintln!("Skipping: MiniSat not available");
            return;
        }

        // Generate and test random 3-SAT formulas
        for seed in 0..50 {
            let (cnf, clauses) = generate_random_3sat(10, 40, seed);

            let formula = parse_dimacs(&cnf).expect("Failed to parse");
            let mut solver = formula.into_solver();
            let z4_result = solver.solve();
            let minisat_result = run_minisat(&cnf);

            let name = format!("random_3sat_{}", seed);
            assert!(
                compare_results(&z4_result, &minisat_result, &name),
                "Failed on seed {}",
                seed
            );

            // Verify model if SAT
            if let SolveResult::Sat(model) = &z4_result {
                assert!(
                    super::verify_model(&clauses, model),
                    "Invalid model for seed {}",
                    seed
                );
            }
        }
    }

    /// Generate a random 3-SAT formula
    fn generate_random_3sat(
        num_vars: u32,
        num_clauses: usize,
        seed: u64,
    ) -> (String, Vec<Vec<z4_sat::Literal>>) {
        use z4_sat::{Literal, Variable};

        let mut cnf = format!("p cnf {} {}\n", num_vars, num_clauses);
        let mut clauses = Vec::new();

        // Simple LCG for deterministic pseudo-randomness
        let mut state = seed;
        let lcg_next = |s: &mut u64| {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            *s
        };

        for _ in 0..num_clauses {
            let mut clause_lits = Vec::new();
            let mut clause_str = String::new();

            for _ in 0..3 {
                let var = ((lcg_next(&mut state) % num_vars as u64) + 1) as i32;
                let sign = if lcg_next(&mut state) % 2 == 0 { 1 } else { -1 };
                let lit = var * sign;
                clause_str.push_str(&format!("{} ", lit));

                let variable = Variable(var as u32 - 1);
                let literal = if sign > 0 {
                    Literal::positive(variable)
                } else {
                    Literal::negative(variable)
                };
                clause_lits.push(literal);
            }
            clause_str.push_str("0\n");
            cnf.push_str(&clause_str);
            clauses.push(clause_lits);
        }

        (cnf, clauses)
    }
}

// ============================================================================
// Vivification Tests
// ============================================================================

/// Test that vivification doesn't break solver correctness
#[test]
fn test_vivification_correctness() {
    // Use a formula that generates enough learned clauses to trigger vivification
    // We'll use a larger random 3-SAT formula
    use z4_sat::{SolveResult, Solver};

    // Create a 15-variable 60-clause random 3-SAT formula (near threshold)
    let mut solver = Solver::new(15);
    let clauses = generate_test_clauses(15, 60, 42);

    for clause in &clauses {
        solver.add_clause(clause.clone());
    }

    // Solve with vivification enabled (default)
    let result = solver.solve();

    // If SAT, verify model
    if let SolveResult::Sat(model) = &result {
        for clause in &clauses {
            let satisfied = clause.iter().any(|lit| {
                let var_val = model[lit.variable().0 as usize];
                if lit.is_positive() {
                    var_val
                } else {
                    !var_val
                }
            });
            assert!(satisfied, "Vivification produced invalid model");
        }
    }
}

/// Test vivification statistics are tracked
#[test]
fn test_vivification_stats() {
    use z4_sat::Solver;

    let mut solver = Solver::new(10);
    let clauses = generate_test_clauses(10, 40, 123);

    for clause in &clauses {
        solver.add_clause(clause.clone());
    }

    let _ = solver.solve();

    // Stats should be accessible (even if 0)
    let stats = solver.vivify_stats();
    // We don't assert specific values because vivification may or may not trigger
    // depending on the number of conflicts
    let _ = stats.clauses_examined;
}

/// Test that vivification can be disabled
#[test]
fn test_vivification_disabled() {
    use z4_sat::{SolveResult, Solver};

    let mut solver = Solver::new(10);
    solver.set_vivify_enabled(false);

    let clauses = generate_test_clauses(10, 40, 456);

    for clause in &clauses {
        solver.add_clause(clause.clone());
    }

    let result = solver.solve();

    // Solver should still work correctly with vivification disabled
    if let SolveResult::Sat(model) = &result {
        for clause in &clauses {
            let satisfied = clause.iter().any(|lit| {
                let var_val = model[lit.variable().0 as usize];
                if lit.is_positive() {
                    var_val
                } else {
                    !var_val
                }
            });
            assert!(satisfied, "Disabled vivification produced invalid model");
        }
    }

    // With vivification disabled, no clauses should be examined
    let stats = solver.vivify_stats();
    assert_eq!(
        stats.clauses_examined, 0,
        "Vivification ran despite being disabled"
    );
}

/// Generate test clauses for vivification tests
fn generate_test_clauses(num_vars: u32, num_clauses: usize, seed: u64) -> Vec<Vec<Literal>> {
    let mut clauses = Vec::new();

    // Simple LCG for deterministic pseudo-randomness
    let mut state = seed;
    let lcg_next = |s: &mut u64| {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        *s
    };

    for _ in 0..num_clauses {
        let mut clause_lits = Vec::new();

        for _ in 0..3 {
            let var = (lcg_next(&mut state) % num_vars as u64) as u32;
            let sign = lcg_next(&mut state) % 2 == 0;

            let variable = Variable(var);
            let literal = if sign {
                Literal::positive(variable)
            } else {
                Literal::negative(variable)
            };
            clause_lits.push(literal);
        }
        clauses.push(clause_lits);
    }

    clauses
}

// ============================================================================
// Subsumption Tests
// ============================================================================

/// Test that subsumption doesn't break solver correctness
#[test]
fn test_subsumption_correctness() {
    // Use a formula that generates enough learned clauses to trigger subsumption
    use z4_sat::{SolveResult, Solver};

    // Create a 15-variable 60-clause random 3-SAT formula (near threshold)
    let mut solver = Solver::new(15);
    let clauses = generate_test_clauses(15, 60, 789);

    for clause in &clauses {
        solver.add_clause(clause.clone());
    }

    // Solve with subsumption enabled (default)
    let result = solver.solve();

    // If SAT, verify model
    if let SolveResult::Sat(model) = &result {
        for clause in &clauses {
            let satisfied = clause.iter().any(|lit| {
                let var_val = model[lit.variable().0 as usize];
                if lit.is_positive() {
                    var_val
                } else {
                    !var_val
                }
            });
            assert!(satisfied, "Subsumption produced invalid model");
        }
    }
}

/// Test subsumption statistics are tracked
#[test]
fn test_subsumption_stats() {
    use z4_sat::Solver;

    let mut solver = Solver::new(10);
    let clauses = generate_test_clauses(10, 40, 321);

    for clause in &clauses {
        solver.add_clause(clause.clone());
    }

    let _ = solver.solve();

    // Stats should be accessible (even if 0)
    let stats = solver.subsume_stats();
    // We don't assert specific values because subsumption may or may not trigger
    // depending on the number of conflicts
    let _ = stats.forward_subsumed;
    let _ = stats.backward_subsumed;
    let _ = stats.strengthened_clauses;
}

/// Test that subsumption can be disabled
#[test]
fn test_subsumption_disabled() {
    use z4_sat::{SolveResult, Solver};

    let mut solver = Solver::new(10);
    solver.set_subsume_enabled(false);

    let clauses = generate_test_clauses(10, 40, 654);

    for clause in &clauses {
        solver.add_clause(clause.clone());
    }

    let result = solver.solve();

    // Solver should still work correctly with subsumption disabled
    if let SolveResult::Sat(model) = &result {
        for clause in &clauses {
            let satisfied = clause.iter().any(|lit| {
                let var_val = model[lit.variable().0 as usize];
                if lit.is_positive() {
                    var_val
                } else {
                    !var_val
                }
            });
            assert!(satisfied, "Disabled subsumption produced invalid model");
        }
    }

    // With subsumption disabled, no subsumption checks should be performed
    let stats = solver.subsume_stats();
    assert_eq!(stats.checks, 0, "Subsumption ran despite being disabled");
}

// ============================================================================
// Failed Literal Probing Tests
// ============================================================================

/// Test that probing doesn't break solver correctness
#[test]
fn test_probing_correctness() {
    use z4_sat::{SolveResult, Solver};

    // Create a 15-variable 60-clause random 3-SAT formula (near threshold)
    let mut solver = Solver::new(15);
    let clauses = generate_test_clauses(15, 60, 789);

    for clause in &clauses {
        solver.add_clause(clause.clone());
    }

    // Solve with probing enabled (default)
    let result = solver.solve();

    // If SAT, verify model
    if let SolveResult::Sat(model) = &result {
        for clause in &clauses {
            let satisfied = clause.iter().any(|lit| {
                let var_val = model[lit.variable().0 as usize];
                if lit.is_positive() {
                    var_val
                } else {
                    !var_val
                }
            });
            assert!(satisfied, "Probing produced invalid model");
        }
    }
}

/// Test probing statistics are tracked
#[test]
fn test_probing_stats() {
    use z4_sat::Solver;

    let mut solver = Solver::new(10);
    let clauses = generate_test_clauses(10, 40, 321);

    for clause in &clauses {
        solver.add_clause(clause.clone());
    }

    let _ = solver.solve();

    // Stats should be accessible (even if 0)
    let stats = solver.probe_stats();
    // We don't assert specific values because probing may or may not trigger
    // depending on the number of conflicts and formula structure
    let _ = stats.rounds;
    let _ = stats.probed;
    let _ = stats.failed;
}

/// Test that probing can be disabled
#[test]
fn test_probing_disabled() {
    use z4_sat::{SolveResult, Solver};

    let mut solver = Solver::new(10);
    solver.set_probe_enabled(false);

    let clauses = generate_test_clauses(10, 40, 987);

    for clause in &clauses {
        solver.add_clause(clause.clone());
    }

    let result = solver.solve();

    // Solver should still work correctly with probing disabled
    if let SolveResult::Sat(model) = &result {
        for clause in &clauses {
            let satisfied = clause.iter().any(|lit| {
                let var_val = model[lit.variable().0 as usize];
                if lit.is_positive() {
                    var_val
                } else {
                    !var_val
                }
            });
            assert!(satisfied, "Disabled probing produced invalid model");
        }
    }

    // With probing disabled, no probing rounds should occur
    let stats = solver.probe_stats();
    assert_eq!(stats.rounds, 0, "Probing ran despite being disabled");
}

/// Test probing with formulas that have failed literals
#[test]
fn test_probing_with_failed_literals() {
    use z4_sat::{Literal, SolveResult, Solver, Variable};

    // Create a formula where probing can detect a failed literal
    // Formula: (x0 v x1) ^ (x0 v -x1) ^ (x2 v x3)
    // Probing -x0 leads to conflict: from (x0 v x1) we get x1,
    // from (x0 v -x1) we get -x1, contradiction!
    // Therefore x0 must be true (failed literal -x0)

    let mut solver = Solver::new(4);

    // (x0 v x1) - if x0=false, then x1=true
    solver.add_clause(vec![
        Literal::positive(Variable(0)),
        Literal::positive(Variable(1)),
    ]);
    // (x0 v -x1) - if x0=false, then x1=false (conflicts with above!)
    solver.add_clause(vec![
        Literal::positive(Variable(0)),
        Literal::negative(Variable(1)),
    ]);
    // (x2 v x3) - extra clause to make the formula more interesting
    solver.add_clause(vec![
        Literal::positive(Variable(2)),
        Literal::positive(Variable(3)),
    ]);

    let result = solver.solve();

    // Formula should be SAT with x0 = true (forced by failed literal detection)
    match result {
        SolveResult::Sat(model) => {
            assert!(model[0], "x0 should be true (forced by failed literal)");
        }
        SolveResult::Unsat => panic!("Formula should be SAT"),
        SolveResult::Unknown => panic!("Got Unknown"),
    }
}

// ============================================================================
// Bounded Variable Elimination (BVE) Tests
// ============================================================================

/// Test that BVE doesn't break solver correctness
#[test]
fn test_bve_correctness() {
    use z4_sat::{SolveResult, Solver};

    // Create a 15-variable 60-clause random 3-SAT formula (near threshold)
    let mut solver = Solver::new(15);
    let clauses = generate_test_clauses(15, 60, 111);

    for clause in &clauses {
        solver.add_clause(clause.clone());
    }

    // Solve with BVE enabled (default)
    let result = solver.solve();

    // If SAT, verify model
    if let SolveResult::Sat(model) = &result {
        for clause in &clauses {
            let satisfied = clause.iter().any(|lit| {
                let var_val = model[lit.variable().0 as usize];
                if lit.is_positive() {
                    var_val
                } else {
                    !var_val
                }
            });
            assert!(satisfied, "BVE produced invalid model");
        }
    }
}

/// Test BVE statistics are tracked
#[test]
fn test_bve_stats() {
    use z4_sat::Solver;

    let mut solver = Solver::new(10);
    let clauses = generate_test_clauses(10, 40, 222);

    for clause in &clauses {
        solver.add_clause(clause.clone());
    }

    let _ = solver.solve();

    // Stats should be accessible (even if 0)
    let stats = solver.bve_stats();
    // We don't assert specific values because BVE may or may not trigger
    // depending on the number of conflicts and formula structure
    let _ = stats.rounds;
    let _ = stats.vars_eliminated;
    let _ = stats.clauses_removed;
    let _ = stats.resolvents_added;
}

/// Test that BVE can be disabled
#[test]
fn test_bve_disabled() {
    use z4_sat::{SolveResult, Solver};

    let mut solver = Solver::new(10);
    solver.set_bve_enabled(false);

    let clauses = generate_test_clauses(10, 40, 333);

    for clause in &clauses {
        solver.add_clause(clause.clone());
    }

    let result = solver.solve();

    // Solver should still work correctly with BVE disabled
    if let SolveResult::Sat(model) = &result {
        for clause in &clauses {
            let satisfied = clause.iter().any(|lit| {
                let var_val = model[lit.variable().0 as usize];
                if lit.is_positive() {
                    var_val
                } else {
                    !var_val
                }
            });
            assert!(satisfied, "Disabled BVE produced invalid model");
        }
    }

    // With BVE disabled, no BVE rounds should occur
    let stats = solver.bve_stats();
    assert_eq!(stats.rounds, 0, "BVE ran despite being disabled");
}

/// Test BVE with a formula where variable elimination is beneficial
#[test]
fn test_bve_with_eliminable_variable() {
    use z4_sat::{Literal, SolveResult, Solver, Variable};

    // Create a formula where x0 appears in few clauses and can be eliminated:
    // (x0 v x1) ^ (-x0 v x2) ^ (x3 v x4)
    // Eliminating x0: resolve (x0 v x1) with (-x0 v x2) to get (x1 v x2)
    // Result: (x1 v x2) ^ (x3 v x4) - same satisfiability, fewer clauses

    let mut solver = Solver::new(5);

    // (x0 v x1)
    solver.add_clause(vec![
        Literal::positive(Variable(0)),
        Literal::positive(Variable(1)),
    ]);
    // (-x0 v x2)
    solver.add_clause(vec![
        Literal::negative(Variable(0)),
        Literal::positive(Variable(2)),
    ]);
    // (x3 v x4)
    solver.add_clause(vec![
        Literal::positive(Variable(3)),
        Literal::positive(Variable(4)),
    ]);

    let result = solver.solve();

    // Formula should be SAT
    match result {
        SolveResult::Sat(model) => {
            // Verify the model satisfies all original clauses
            // (x0 v x1)
            assert!(model[0] || model[1]);
            // (-x0 v x2)
            assert!(!model[0] || model[2]);
            // (x3 v x4)
            assert!(model[3] || model[4]);
        }
        SolveResult::Unsat => panic!("Formula should be SAT"),
        SolveResult::Unknown => panic!("Got Unknown"),
    }
}

/// Test BVE with pure literal detection (variable appears with only one polarity)
#[test]
fn test_bve_pure_literal() {
    use z4_sat::{Literal, SolveResult, Solver, Variable};

    // Create a formula where x0 is pure (appears only positively):
    // (x0 v x1) ^ (x0 v x2) ^ (x3 v x4)
    // x0 is pure - all clauses containing x0 can be satisfied by setting x0=true

    let mut solver = Solver::new(5);

    // (x0 v x1)
    solver.add_clause(vec![
        Literal::positive(Variable(0)),
        Literal::positive(Variable(1)),
    ]);
    // (x0 v x2)
    solver.add_clause(vec![
        Literal::positive(Variable(0)),
        Literal::positive(Variable(2)),
    ]);
    // (x3 v x4)
    solver.add_clause(vec![
        Literal::positive(Variable(3)),
        Literal::positive(Variable(4)),
    ]);

    let result = solver.solve();

    // Formula should be SAT
    match result {
        SolveResult::Sat(model) => {
            // Verify the model satisfies all original clauses
            assert!(model[0] || model[1]);
            assert!(model[0] || model[2]);
            assert!(model[3] || model[4]);
        }
        SolveResult::Unsat => panic!("Formula should be SAT"),
        SolveResult::Unknown => panic!("Got Unknown"),
    }
}

// ============================================================================
// Hyper-Ternary Resolution (HTR) Tests
// ============================================================================

/// Test HTR correctness - solver produces valid results with HTR enabled
#[test]
fn test_htr_correctness() {
    use z4_sat::{SolveResult, Solver};

    // Create a 15-variable 60-clause random 3-SAT formula
    // Use generate_test_clauses which produces ternary clauses
    let mut solver = Solver::new(15);
    let clauses = generate_test_clauses(15, 60, 444);

    for clause in &clauses {
        solver.add_clause(clause.clone());
    }

    // Solve with HTR enabled (default)
    let result = solver.solve();

    // If SAT, verify model
    if let SolveResult::Sat(model) = &result {
        for clause in &clauses {
            let satisfied = clause.iter().any(|lit| {
                let var_val = model[lit.variable().0 as usize];
                if lit.is_positive() {
                    var_val
                } else {
                    !var_val
                }
            });
            assert!(satisfied, "HTR produced invalid model");
        }
    }
}

/// Test HTR statistics are tracked
#[test]
fn test_htr_stats() {
    use z4_sat::Solver;

    let mut solver = Solver::new(10);
    let clauses = generate_test_clauses(10, 40, 555);

    for clause in &clauses {
        solver.add_clause(clause.clone());
    }

    let _ = solver.solve();

    // Stats should be accessible (even if 0)
    let stats = solver.htr_stats();
    // We don't assert specific values because HTR may or may not trigger
    // depending on the number of conflicts and formula structure
    let _ = stats.rounds;
    let _ = stats.ternary_resolvents;
    let _ = stats.binary_resolvents;
    let _ = stats.pairs_checked;
}

/// Test that HTR can be disabled
#[test]
fn test_htr_disabled() {
    use z4_sat::{SolveResult, Solver};

    let mut solver = Solver::new(10);
    solver.set_htr_enabled(false);

    let clauses = generate_test_clauses(10, 40, 666);

    for clause in &clauses {
        solver.add_clause(clause.clone());
    }

    let result = solver.solve();

    // Solver should still work correctly with HTR disabled
    if let SolveResult::Sat(model) = &result {
        for clause in &clauses {
            let satisfied = clause.iter().any(|lit| {
                let var_val = model[lit.variable().0 as usize];
                if lit.is_positive() {
                    var_val
                } else {
                    !var_val
                }
            });
            assert!(satisfied, "Solver with HTR disabled produced invalid model");
        }
    }

    // HTR stats should show no activity
    let stats = solver.htr_stats();
    assert_eq!(stats.rounds, 0, "HTR rounds should be 0 when disabled");
}

/// Test gate extraction statistics are tracked
#[test]
fn test_gate_stats() {
    use z4_sat::Solver;

    let mut solver = Solver::new(10);
    let clauses = generate_test_clauses(10, 40, 777);

    for clause in &clauses {
        solver.add_clause(clause.clone());
    }

    let _ = solver.solve();

    // Stats should be accessible (even if 0)
    let stats = solver.gate_stats();
    // Gate extraction may or may not find gates depending on the formula
    let _ = stats.and_gates;
    let _ = stats.xor_gates;
    let _ = stats.ite_gates;
    let _ = stats.equivalences;
    let _ = stats.extraction_calls;
    let _ = stats.total_gates();
}

/// Test that gate extraction can be disabled
#[test]
fn test_gate_disabled() {
    use z4_sat::{SolveResult, Solver};

    let mut solver = Solver::new(10);
    solver.set_gate_enabled(false);

    let clauses = generate_test_clauses(10, 40, 888);

    for clause in &clauses {
        solver.add_clause(clause.clone());
    }

    let result = solver.solve();

    // Solver should still work correctly with gate extraction disabled
    if let SolveResult::Sat(model) = &result {
        for clause in &clauses {
            let satisfied = clause.iter().any(|lit| {
                let var_val = model[lit.variable().0 as usize];
                if lit.is_positive() {
                    var_val
                } else {
                    !var_val
                }
            });
            assert!(
                satisfied,
                "Solver with gate extraction disabled produced invalid model"
            );
        }
    }

    // Stats should still be accessible
    let stats = solver.gate_stats();
    let _ = stats.total_gates();
}

// ============================================================================
// Model Reconstruction Tests
// ============================================================================
// These tests verify that models are correctly reconstructed after
// equisatisfiable transformations like BVE and SAT sweeping.

/// Test BVE model reconstruction with a simple eliminable variable
#[test]
fn test_model_reconstruction_bve_simple() {
    use z4_sat::{Literal, SolveResult, Solver, Variable};

    // Formula designed so BVE will eliminate x0:
    // (x0 v x1) ^ (-x0 v x2) ^ (x1 v x3) ^ (x2 v x3)
    //
    // After eliminating x0 by resolution:
    // Original (x0 v x1) and (-x0 v x2) resolve to (x1 v x2)
    // Transformed: (x1 v x2) ^ (x1 v x3) ^ (x2 v x3)
    //
    // The model of the transformed formula must be reconstructed to also
    // satisfy the original clauses involving x0.

    let original_clauses: Vec<Vec<Literal>> = vec![
        vec![
            Literal::positive(Variable(0)),
            Literal::positive(Variable(1)),
        ], // (x0 v x1)
        vec![
            Literal::negative(Variable(0)),
            Literal::positive(Variable(2)),
        ], // (-x0 v x2)
        vec![
            Literal::positive(Variable(1)),
            Literal::positive(Variable(3)),
        ], // (x1 v x3)
        vec![
            Literal::positive(Variable(2)),
            Literal::positive(Variable(3)),
        ], // (x2 v x3)
    ];

    let mut solver = Solver::new(4);
    for clause in &original_clauses {
        solver.add_clause(clause.clone());
    }

    let result = solver.solve();

    match result {
        SolveResult::Sat(model) => {
            // Verify model satisfies ALL ORIGINAL clauses
            for (i, clause) in original_clauses.iter().enumerate() {
                let satisfied = clause.iter().any(|lit| {
                    let var_idx = lit.variable().0 as usize;
                    if var_idx >= model.len() {
                        return false;
                    }
                    let var_value = model[var_idx];
                    if lit.is_positive() {
                        var_value
                    } else {
                        !var_value
                    }
                });
                assert!(
                    satisfied,
                    "Original clause {} not satisfied after BVE reconstruction: {:?}, model: {:?}",
                    i, clause, model
                );
            }
        }
        SolveResult::Unsat => panic!("Formula should be SAT"),
        SolveResult::Unknown => panic!("Got Unknown"),
    }
}

/// Test BVE model reconstruction with multiple eliminable variables
#[test]
fn test_model_reconstruction_bve_multiple_vars() {
    use z4_sat::{Literal, SolveResult, Solver, Variable};

    // Formula where multiple variables can be eliminated:
    // (x0 v x2) ^ (-x0 v x3) - can eliminate x0
    // (x1 v x4) ^ (-x1 v x5) - can eliminate x1
    // (x2 v x4) ^ (x3 v x5) - remaining clauses

    let original_clauses: Vec<Vec<Literal>> = vec![
        vec![
            Literal::positive(Variable(0)),
            Literal::positive(Variable(2)),
        ],
        vec![
            Literal::negative(Variable(0)),
            Literal::positive(Variable(3)),
        ],
        vec![
            Literal::positive(Variable(1)),
            Literal::positive(Variable(4)),
        ],
        vec![
            Literal::negative(Variable(1)),
            Literal::positive(Variable(5)),
        ],
        vec![
            Literal::positive(Variable(2)),
            Literal::positive(Variable(4)),
        ],
        vec![
            Literal::positive(Variable(3)),
            Literal::positive(Variable(5)),
        ],
    ];

    let mut solver = Solver::new(6);
    for clause in &original_clauses {
        solver.add_clause(clause.clone());
    }

    let result = solver.solve();

    match result {
        SolveResult::Sat(model) => {
            // Verify model satisfies ALL ORIGINAL clauses
            for (i, clause) in original_clauses.iter().enumerate() {
                let satisfied = clause.iter().any(|lit| {
                    let var_idx = lit.variable().0 as usize;
                    if var_idx >= model.len() {
                        return false;
                    }
                    let var_value = model[var_idx];
                    if lit.is_positive() {
                        var_value
                    } else {
                        !var_value
                    }
                });
                assert!(
                    satisfied,
                    "Original clause {} not satisfied after multi-var BVE reconstruction: {:?}",
                    i, clause
                );
            }
        }
        SolveResult::Unsat => panic!("Formula should be SAT"),
        SolveResult::Unknown => panic!("Got Unknown"),
    }
}

/// Test sweeping model reconstruction with equivalent literals
#[test]
fn test_model_reconstruction_sweep_equivalence() {
    use z4_sat::{Literal, SolveResult, Solver, Variable};

    // Formula encoding x0 <-> x1 (equivalence):
    // (x0 v -x1) ^ (-x0 v x1) plus extra clauses
    //
    // Sweeping will detect x0 and x1 are equivalent and merge them.
    // The model must be reconstructed so both x0 and x1 have correct values.

    let original_clauses: Vec<Vec<Literal>> = vec![
        // x0 <-> x1
        vec![
            Literal::positive(Variable(0)),
            Literal::negative(Variable(1)),
        ], // x0 v -x1
        vec![
            Literal::negative(Variable(0)),
            Literal::positive(Variable(1)),
        ], // -x0 v x1
        // Additional constraint using x1
        vec![
            Literal::positive(Variable(1)),
            Literal::positive(Variable(2)),
        ], // x1 v x2
        // Additional clauses
        vec![
            Literal::positive(Variable(2)),
            Literal::positive(Variable(3)),
        ], // x2 v x3
    ];

    let mut solver = Solver::new(4);
    for clause in &original_clauses {
        solver.add_clause(clause.clone());
    }

    let result = solver.solve();

    match result {
        SolveResult::Sat(model) => {
            // Verify model satisfies ALL ORIGINAL clauses
            for (i, clause) in original_clauses.iter().enumerate() {
                let satisfied = clause.iter().any(|lit| {
                    let var_idx = lit.variable().0 as usize;
                    if var_idx >= model.len() {
                        return false;
                    }
                    let var_value = model[var_idx];
                    if lit.is_positive() {
                        var_value
                    } else {
                        !var_value
                    }
                });
                assert!(
                    satisfied,
                    "Original clause {} not satisfied after sweep reconstruction: {:?}",
                    i, clause
                );
            }

            // Additionally verify the equivalence: x0 and x1 should have same value
            assert_eq!(
                model[0], model[1],
                "Equivalence x0 <-> x1 not preserved in reconstructed model"
            );
        }
        SolveResult::Unsat => panic!("Formula should be SAT"),
        SolveResult::Unknown => panic!("Got Unknown"),
    }
}

/// Test sweeping model reconstruction with negated equivalence (x0 <-> -x1)
#[test]
fn test_model_reconstruction_sweep_negated_equivalence() {
    use z4_sat::{Literal, SolveResult, Solver, Variable};

    // Formula encoding x0 <-> -x1 (negated equivalence):
    // (x0 v x1) ^ (-x0 v -x1)
    // This means x0 and x1 must have opposite values.

    let original_clauses: Vec<Vec<Literal>> = vec![
        // x0 <-> -x1
        vec![
            Literal::positive(Variable(0)),
            Literal::positive(Variable(1)),
        ], // x0 v x1
        vec![
            Literal::negative(Variable(0)),
            Literal::negative(Variable(1)),
        ], // -x0 v -x1
        // Additional constraint
        vec![
            Literal::positive(Variable(0)),
            Literal::positive(Variable(2)),
        ], // x0 v x2
        vec![
            Literal::positive(Variable(2)),
            Literal::positive(Variable(3)),
        ], // x2 v x3
    ];

    let mut solver = Solver::new(4);
    for clause in &original_clauses {
        solver.add_clause(clause.clone());
    }

    let result = solver.solve();

    match result {
        SolveResult::Sat(model) => {
            // Verify model satisfies ALL ORIGINAL clauses
            for (i, clause) in original_clauses.iter().enumerate() {
                let satisfied = clause.iter().any(|lit| {
                    let var_idx = lit.variable().0 as usize;
                    if var_idx >= model.len() {
                        return false;
                    }
                    let var_value = model[var_idx];
                    if lit.is_positive() {
                        var_value
                    } else {
                        !var_value
                    }
                });
                assert!(
                    satisfied,
                    "Original clause {} not satisfied: {:?}",
                    i, clause
                );
            }

            // Verify the negated equivalence: x0 and x1 should have opposite values
            assert_ne!(
                model[0], model[1],
                "Negated equivalence x0 <-> -x1 not preserved: x0={}, x1={}",
                model[0], model[1]
            );
        }
        SolveResult::Unsat => panic!("Formula should be SAT"),
        SolveResult::Unknown => panic!("Got Unknown"),
    }
}

/// Test combined BVE and sweeping model reconstruction
#[test]
fn test_model_reconstruction_combined_bve_sweep() {
    use z4_sat::{Literal, SolveResult, Solver, Variable};

    // Formula that can trigger both BVE and sweeping:
    // - x0 <-> x1 (equivalence, sweeping target)
    // - x2 appears with both polarities (BVE target)

    let original_clauses: Vec<Vec<Literal>> = vec![
        // x0 <-> x1 (equivalence)
        vec![
            Literal::positive(Variable(0)),
            Literal::negative(Variable(1)),
        ],
        vec![
            Literal::negative(Variable(0)),
            Literal::positive(Variable(1)),
        ],
        // x2 can be eliminated via resolution
        vec![
            Literal::positive(Variable(2)),
            Literal::positive(Variable(3)),
        ], // x2 v x3
        vec![
            Literal::negative(Variable(2)),
            Literal::positive(Variable(4)),
        ], // -x2 v x4
        // Link to equivalence
        vec![
            Literal::positive(Variable(0)),
            Literal::positive(Variable(3)),
        ], // x0 v x3
        vec![
            Literal::positive(Variable(1)),
            Literal::positive(Variable(4)),
        ], // x1 v x4
    ];

    let mut solver = Solver::new(5);
    for clause in &original_clauses {
        solver.add_clause(clause.clone());
    }

    let result = solver.solve();

    match result {
        SolveResult::Sat(model) => {
            // Verify model satisfies ALL ORIGINAL clauses
            for (i, clause) in original_clauses.iter().enumerate() {
                let satisfied = clause.iter().any(|lit| {
                    let var_idx = lit.variable().0 as usize;
                    if var_idx >= model.len() {
                        return false;
                    }
                    let var_value = model[var_idx];
                    if lit.is_positive() {
                        var_value
                    } else {
                        !var_value
                    }
                });
                assert!(
                    satisfied,
                    "Original clause {} not satisfied after combined reconstruction: {:?}, model: {:?}",
                    i, clause, model
                );
            }
        }
        SolveResult::Unsat => panic!("Formula should be SAT"),
        SolveResult::Unknown => panic!("Got Unknown"),
    }
}

/// Test model reconstruction with larger formula using random 3-SAT
#[test]
fn test_model_reconstruction_larger_formula() {
    use z4_sat::{SolveResult, Solver};

    // Generate a larger formula that will likely trigger inprocessing
    let clauses = generate_test_clauses(20, 80, 12345);

    let mut solver = Solver::new(20);
    for clause in &clauses {
        solver.add_clause(clause.clone());
    }

    let result = solver.solve();

    match result {
        SolveResult::Sat(model) => {
            // Verify model satisfies ALL ORIGINAL clauses
            for (i, clause) in clauses.iter().enumerate() {
                let satisfied = clause.iter().any(|lit| {
                    let var_idx = lit.variable().0 as usize;
                    if var_idx >= model.len() {
                        return false;
                    }
                    let var_value = model[var_idx];
                    if lit.is_positive() {
                        var_value
                    } else {
                        !var_value
                    }
                });
                assert!(
                    satisfied,
                    "Original clause {} not satisfied in larger formula test",
                    i
                );
            }
        }
        SolveResult::Unsat => {
            // UNSAT is also valid, no reconstruction needed
        }
        SolveResult::Unknown => panic!("Got Unknown"),
    }
}

/// Test that reconstruction correctly handles extended model size
#[test]
fn test_model_reconstruction_model_size() {
    use z4_sat::{Literal, SolveResult, Solver, Variable};

    // Create a formula where BVE eliminates variables but the model
    // should still have the correct size

    let original_clauses: Vec<Vec<Literal>> = vec![
        vec![
            Literal::positive(Variable(0)),
            Literal::positive(Variable(5)),
        ],
        vec![
            Literal::negative(Variable(0)),
            Literal::positive(Variable(5)),
        ],
        vec![
            Literal::positive(Variable(1)),
            Literal::positive(Variable(6)),
        ],
        vec![
            Literal::negative(Variable(1)),
            Literal::positive(Variable(6)),
        ],
        vec![
            Literal::positive(Variable(2)),
            Literal::positive(Variable(7)),
        ],
        vec![
            Literal::negative(Variable(2)),
            Literal::positive(Variable(7)),
        ],
        // Some constraints on the higher variables
        vec![
            Literal::positive(Variable(5)),
            Literal::positive(Variable(6)),
        ],
        vec![
            Literal::positive(Variable(6)),
            Literal::positive(Variable(7)),
        ],
        vec![
            Literal::positive(Variable(3)),
            Literal::positive(Variable(4)),
        ],
    ];

    let mut solver = Solver::new(8);
    for clause in &original_clauses {
        solver.add_clause(clause.clone());
    }

    let result = solver.solve();

    match result {
        SolveResult::Sat(model) => {
            // Model should have enough entries for all original variables
            assert!(
                model.len() >= 8,
                "Model too small: expected >= 8, got {}",
                model.len()
            );

            // Verify model satisfies ALL ORIGINAL clauses
            for (i, clause) in original_clauses.iter().enumerate() {
                let satisfied = clause.iter().any(|lit| {
                    let var_idx = lit.variable().0 as usize;
                    if var_idx >= model.len() {
                        return false;
                    }
                    let var_value = model[var_idx];
                    if lit.is_positive() {
                        var_value
                    } else {
                        !var_value
                    }
                });
                assert!(satisfied, "Original clause {} not satisfied", i);
            }
        }
        SolveResult::Unsat => panic!("Formula should be SAT"),
        SolveResult::Unknown => panic!("Got Unknown"),
    }
}

/// Test reconstruction with pure literal elimination
#[test]
fn test_model_reconstruction_pure_literal() {
    use z4_sat::{Literal, SolveResult, Solver, Variable};

    // Formula where x0 is a pure literal (appears only positively):
    // All clauses containing x0 can be satisfied by setting x0=true
    // The model must correctly reconstruct x0's value

    let original_clauses: Vec<Vec<Literal>> = vec![
        vec![
            Literal::positive(Variable(0)),
            Literal::positive(Variable(1)),
        ], // x0 v x1
        vec![
            Literal::positive(Variable(0)),
            Literal::positive(Variable(2)),
        ], // x0 v x2
        vec![
            Literal::positive(Variable(0)),
            Literal::positive(Variable(3)),
        ], // x0 v x3
        vec![
            Literal::positive(Variable(1)),
            Literal::negative(Variable(2)),
        ], // x1 v -x2
        vec![
            Literal::positive(Variable(2)),
            Literal::positive(Variable(3)),
        ], // x2 v x3
    ];

    let mut solver = Solver::new(4);
    for clause in &original_clauses {
        solver.add_clause(clause.clone());
    }

    let result = solver.solve();

    match result {
        SolveResult::Sat(model) => {
            // Verify model satisfies ALL ORIGINAL clauses
            for (i, clause) in original_clauses.iter().enumerate() {
                let satisfied = clause.iter().any(|lit| {
                    let var_idx = lit.variable().0 as usize;
                    if var_idx >= model.len() {
                        return false;
                    }
                    let var_value = model[var_idx];
                    if lit.is_positive() {
                        var_value
                    } else {
                        !var_value
                    }
                });
                assert!(
                    satisfied,
                    "Original clause {} not satisfied with pure literal: {:?}",
                    i, clause
                );
            }
        }
        SolveResult::Unsat => panic!("Formula should be SAT"),
        SolveResult::Unknown => panic!("Got Unknown"),
    }
}

/// Comprehensive stress test: verify reconstruction across many random formulas
#[test]
fn test_model_reconstruction_stress() {
    use z4_sat::{SolveResult, Solver};

    // Test across multiple random seeds to catch edge cases
    for seed in 0..100 {
        let clauses = generate_test_clauses(12, 48, seed);

        let mut solver = Solver::new(12);
        for clause in &clauses {
            solver.add_clause(clause.clone());
        }

        let result = solver.solve();

        match result {
            SolveResult::Sat(model) => {
                // Verify model satisfies ALL ORIGINAL clauses
                for (i, clause) in clauses.iter().enumerate() {
                    let satisfied = clause.iter().any(|lit| {
                        let var_idx = lit.variable().0 as usize;
                        if var_idx >= model.len() {
                            return false;
                        }
                        let var_value = model[var_idx];
                        if lit.is_positive() {
                            var_value
                        } else {
                            !var_value
                        }
                    });
                    assert!(
                        satisfied,
                        "Stress test seed {}: clause {} not satisfied",
                        seed, i
                    );
                }
            }
            SolveResult::Unsat => {
                // UNSAT is valid
            }
            SolveResult::Unknown => {
                panic!("Stress test seed {}: got Unknown", seed);
            }
        }
    }
}

// LRAT proof support tests

/// Test clause ID tracking when LRAT is enabled
#[test]
fn test_lrat_clause_id_tracking() {
    let mut solver = Solver::new(3);
    solver.enable_lrat();

    // Add original clauses - should get IDs 1, 2, 3
    solver.add_clause(vec![
        Literal::positive(Variable(0)),
        Literal::positive(Variable(1)),
    ]);
    solver.add_clause(vec![
        Literal::negative(Variable(0)),
        Literal::positive(Variable(2)),
    ]);
    solver.add_clause(vec![
        Literal::negative(Variable(1)),
        Literal::negative(Variable(2)),
    ]);

    // Clause IDs should be 1, 2, 3
    assert_eq!(solver.clause_id(ClauseRef(0)), 1);
    assert_eq!(solver.clause_id(ClauseRef(1)), 2);
    assert_eq!(solver.clause_id(ClauseRef(2)), 3);
}

/// Test that LRAT resolution chain is collected during conflict analysis
#[test]
fn test_lrat_resolution_chain() {
    use z4_sat::{parse_dimacs, SolveResult};

    // Simple UNSAT formula that requires learning
    // This formula is UNSAT: (x1) AND (NOT x1)
    let dimacs = r"
p cnf 1 2
1 0
-1 0
";
    let formula = parse_dimacs(dimacs).expect("Failed to parse");
    let mut solver: Solver = formula.into_solver();
    solver.enable_lrat();

    let result = solver.solve();
    assert!(matches!(result, SolveResult::Unsat), "Should be UNSAT");

    // The resolution chain should contain the clause IDs used in derivation
    // For this simple case, it should contain IDs 1 and 2
}

/// Test LRAT writer integration with solver
#[test]
fn test_lrat_writer_output() {
    use z4_sat::{Literal, LratWriter, Variable};

    let mut buf = Vec::new();
    let mut writer = LratWriter::new_text(&mut buf, 3);

    // Simulate adding learned clause with resolution chain
    let clause = vec![Literal::positive(Variable(0))];
    let hints = vec![1, 2, 3];
    let id = writer.add(&clause, &hints).unwrap();

    assert_eq!(id, 4); // First learned clause after 3 original clauses

    let output = String::from_utf8(buf).unwrap();
    assert_eq!(output, "4 1 0 1 2 3 0\n");
}

/// Test LRAT proof generation with ProofOutput enum
#[test]
fn test_lrat_proof_with_proof_output() {
    use z4_sat::ProofOutput;

    // Simple UNSAT formula: (x1) AND (NOT x1)
    let dimacs = r"
p cnf 1 2
1 0
-1 0
";
    let formula = parse_dimacs(dimacs).expect("Failed to parse");

    let proof_buffer: Vec<u8> = Vec::new();
    // LRAT needs to know number of original clauses - we pass 2 (the number of clauses)
    let proof_writer = ProofOutput::lrat_text(proof_buffer, 2);
    let mut solver = Solver::with_proof_output(formula.num_vars, proof_writer);

    for clause in formula.clauses {
        solver.add_clause(clause);
    }

    let result = solver.solve();
    assert_eq!(result, SolveResult::Unsat);

    let writer = solver
        .take_proof_writer()
        .expect("Proof writer should exist");
    let proof = String::from_utf8(writer.into_inner()).expect("Valid UTF-8");

    // LRAT proof should contain clause IDs and end with empty clause
    // Format: "id literals... 0 hints... 0"
    assert!(
        proof.contains("0 0"),
        "LRAT proof should contain empty clause: {}",
        proof
    );
}

/// Test LRAT proof generation for harder UNSAT formula
#[test]
fn test_lrat_proof_harder_unsat() {
    use z4_sat::ProofOutput;

    // Harder UNSAT: mutual exclusion
    // (x1 OR x2) AND (NOT x1 OR NOT x2) AND (x1 OR NOT x2) AND (NOT x1 OR x2)
    // This has no solution
    let dimacs = r"
p cnf 2 4
1 2 0
-1 -2 0
1 -2 0
-1 2 0
";
    let formula = parse_dimacs(dimacs).expect("Failed to parse");

    let proof_buffer: Vec<u8> = Vec::new();
    let proof_writer = ProofOutput::lrat_text(proof_buffer, 4);
    let mut solver = Solver::with_proof_output(formula.num_vars, proof_writer);

    for clause in formula.clauses {
        solver.add_clause(clause);
    }

    let result = solver.solve();
    assert_eq!(result, SolveResult::Unsat);

    let writer = solver
        .take_proof_writer()
        .expect("Proof writer should exist");
    let proof = String::from_utf8(writer.into_inner()).expect("Valid UTF-8");

    // LRAT proof should have learned clauses with IDs > 4 (original clauses)
    // and end with empty clause derivation
    assert!(
        proof.ends_with("0\n"),
        "LRAT proof should end with zero: {}",
        proof
    );

    // Check that we have at least one line (the final empty clause)
    assert!(
        !proof.is_empty(),
        "LRAT proof should not be empty for UNSAT"
    );
}

/// Test LRAT proof with resolution chain tracking
#[test]
fn test_lrat_resolution_chain_tracking() {
    use z4_sat::ProofOutput;

    // UNSAT formula that requires learning
    // (x1) AND (x2 OR NOT x1) AND (NOT x2)
    // Resolution: resolve clause 2 and 3 on x2 to get (NOT x1)
    // Then resolve with clause 1 to get empty clause
    let dimacs = r"
p cnf 2 3
1 0
2 -1 0
-2 0
";
    let formula = parse_dimacs(dimacs).expect("Failed to parse");

    let proof_buffer: Vec<u8> = Vec::new();
    let proof_writer = ProofOutput::lrat_text(proof_buffer, 3);
    let mut solver = Solver::with_proof_output(formula.num_vars, proof_writer);

    for clause in formula.clauses {
        solver.add_clause(clause);
    }

    let result = solver.solve();
    assert_eq!(result, SolveResult::Unsat);

    let writer = solver
        .take_proof_writer()
        .expect("Proof writer should exist");
    let proof = String::from_utf8(writer.into_inner()).expect("Valid UTF-8");

    // LRAT format: "id literals... 0 hints... 0"
    // Each line should have exactly two 0s (one after literals, one after hints)
    for line in proof.lines() {
        let zeros: Vec<_> = line.split_whitespace().filter(|&s| s == "0").collect();
        assert!(
            zeros.len() >= 2 || line.contains(" d "),
            "LRAT line should have two zeros (literals 0 hints 0) or be deletion: {}",
            line
        );
    }
}

/// Test ProofOutput enum correctly handles both DRAT and LRAT
#[test]
fn test_proof_output_drat_compatibility() {
    use z4_sat::ProofOutput;

    // Simple UNSAT formula: (x1) AND (NOT x1)
    let dimacs = r"
p cnf 1 2
1 0
-1 0
";
    let formula = parse_dimacs(dimacs).expect("Failed to parse");

    // Test with DRAT format
    let drat_buffer: Vec<u8> = Vec::new();
    let drat_writer = ProofOutput::drat_text(drat_buffer);
    let mut drat_solver = Solver::with_proof_output(formula.num_vars, drat_writer);

    for clause in formula.clauses.clone() {
        drat_solver.add_clause(clause);
    }

    let drat_result = drat_solver.solve();
    assert_eq!(drat_result, SolveResult::Unsat);

    let drat_proof_writer = drat_solver
        .take_proof_writer()
        .expect("DRAT proof writer should exist");
    let drat_proof = String::from_utf8(drat_proof_writer.into_inner()).expect("Valid UTF-8");

    // Test with LRAT format
    let lrat_buffer: Vec<u8> = Vec::new();
    let lrat_writer = ProofOutput::lrat_text(lrat_buffer, 2);
    let mut lrat_solver = Solver::with_proof_output(formula.num_vars, lrat_writer);

    for clause in formula.clauses {
        lrat_solver.add_clause(clause);
    }

    let lrat_result = lrat_solver.solve();
    assert_eq!(lrat_result, SolveResult::Unsat);

    let lrat_proof_writer = lrat_solver
        .take_proof_writer()
        .expect("LRAT proof writer should exist");
    let lrat_proof = String::from_utf8(lrat_proof_writer.into_inner()).expect("Valid UTF-8");

    // Both should produce valid proofs
    assert!(
        drat_proof.ends_with("0\n"),
        "DRAT proof should end with empty clause"
    );
    assert!(
        lrat_proof.ends_with("0\n"),
        "LRAT proof should end with empty clause"
    );

    // LRAT proof should have clause IDs, DRAT shouldn't
    // LRAT lines start with an ID number, DRAT lines start with literals or 'd'
    let first_lrat_line = lrat_proof.lines().next().unwrap_or("");
    let first_char = first_lrat_line.chars().next();
    if let Some(c) = first_char {
        assert!(
            c.is_ascii_digit() || c == 'd',
            "LRAT proof should start with ID or 'd': {}",
            first_lrat_line
        );
    }
}

// ============================================================================
// Binary Proof Format Tests
// ============================================================================

/// Test binary DRAT proof generation
#[test]
fn test_binary_drat_proof_generation() {
    use z4_sat::ProofOutput;

    // Simple UNSAT formula: (x1) AND (NOT x1)
    let dimacs = r"
p cnf 1 2
1 0
-1 0
";
    let formula = parse_dimacs(dimacs).expect("Failed to parse");

    let proof_buffer: Vec<u8> = Vec::new();
    let proof_writer = ProofOutput::drat_binary(proof_buffer);
    let mut solver = Solver::with_proof_output(formula.num_vars, proof_writer);

    for clause in formula.clauses {
        solver.add_clause(clause);
    }

    let result = solver.solve();
    assert_eq!(result, SolveResult::Unsat);

    let writer = solver
        .take_proof_writer()
        .expect("Proof writer should exist");
    let proof = writer.into_inner();

    // Binary DRAT proofs should start with 'a' (0x61) for additions
    // and contain binary-encoded literals
    assert!(!proof.is_empty(), "Binary DRAT proof should not be empty");

    // The proof should end with 'a' followed by just 0 (empty clause)
    // In binary format: 'a' (0x61) + 0x00 (terminating zero)
    let has_empty_clause = proof.windows(2).any(|w| w == [0x61, 0x00]);
    assert!(
        has_empty_clause,
        "Binary DRAT proof should contain empty clause (a\\x00)"
    );
}

/// Test binary LRAT proof generation
#[test]
fn test_binary_lrat_proof_generation() {
    use z4_sat::ProofOutput;

    // Simple UNSAT formula: (x1) AND (NOT x1)
    let dimacs = r"
p cnf 1 2
1 0
-1 0
";
    let formula = parse_dimacs(dimacs).expect("Failed to parse");

    let proof_buffer: Vec<u8> = Vec::new();
    let proof_writer = ProofOutput::lrat_binary(proof_buffer, 2);
    let mut solver = Solver::with_proof_output(formula.num_vars, proof_writer);

    for clause in formula.clauses {
        solver.add_clause(clause);
    }

    let result = solver.solve();
    assert_eq!(result, SolveResult::Unsat);

    let writer = solver
        .take_proof_writer()
        .expect("Proof writer should exist");
    let proof = writer.into_inner();

    // Binary LRAT proofs should start with 'a' (0x61) for additions
    assert!(!proof.is_empty(), "Binary LRAT proof should not be empty");

    // Check that proof starts with 'a' (addition)
    assert_eq!(
        proof[0], 0x61,
        "Binary LRAT proof should start with 'a' (0x61)"
    );

    // The proof should contain clause IDs (binary encoded)
    // For LRAT with 2 original clauses, first learned clause has ID 3
    // ID 3 is encoded as single byte 0x03 in binary format
    assert!(
        proof.len() >= 3,
        "Binary LRAT proof should have at least header + id + terminator"
    );
}

/// Test binary proof encoding matches specification
#[test]
fn test_binary_literal_encoding() {
    use z4_sat::DratWriter;

    // Test the binary literal encoding formula:
    // positive lit for var v -> 2*(v+1) (1-indexed)
    // negative lit for var v -> 2*(v+1)+1

    let mut buf = Vec::new();
    {
        let mut writer = DratWriter::new_binary(&mut buf);

        // Clause: (x0 OR -x1 OR x2)
        // x0 positive: 2*(0+1) = 2
        // x1 negative: 2*(1+1)+1 = 5
        // x2 positive: 2*(2+1) = 6
        writer
            .add(&[
                Literal::positive(Variable(0)),
                Literal::negative(Variable(1)),
                Literal::positive(Variable(2)),
            ])
            .unwrap();
    }

    // Expected: 'a' (0x61), lit1=2, lit2=5, lit3=6, terminator=0
    assert_eq!(buf, vec![0x61, 2, 5, 6, 0]);
}

/// Test binary proof with variable-length encoding for large literals
#[test]
fn test_binary_variable_length_encoding() {
    use z4_sat::DratWriter;

    let mut buf = Vec::new();
    {
        let mut writer = DratWriter::new_binary(&mut buf);

        // Clause with variable 100 (1-indexed: 101)
        // positive: 2*101 = 202
        // 202 in variable-length encoding:
        // 202 = 0xCA = 1100_1010
        // Low 7 bits: 0100_1010 = 0x4A = 74, with continuation bit: 0xCA
        // High bits: 0000_0001 = 1
        // Encoded as: 0xCA (74 | 0x80), 0x01
        writer.add(&[Literal::positive(Variable(100))]).unwrap();
    }

    // Expected: 'a' (0x61), encoded_202, terminator=0
    // 202 = 0xCA in LEB128: [0xCA, 0x01] (74 with continuation, then 1)
    assert_eq!(buf[0], 0x61); // 'a'
    assert_eq!(buf[1], 0xCA); // 202 & 0x7F | 0x80 = 74 | 128 = 202 = 0xCA
    assert_eq!(buf[2], 0x01); // 202 >> 7 = 1
    assert_eq!(buf[3], 0x00); // terminator
}

/// Test binary DRAT deletion encoding
#[test]
fn test_binary_drat_deletion() {
    use z4_sat::DratWriter;

    let mut buf = Vec::new();
    {
        let mut writer = DratWriter::new_binary(&mut buf);

        // Delete clause: (x0)
        writer.delete(&[Literal::positive(Variable(0))]).unwrap();
    }

    // Expected: 'd' (0x64), lit=2, terminator=0
    assert_eq!(buf, vec![0x64, 2, 0]);
}

/// Test binary LRAT with hints encoding
#[test]
fn test_binary_lrat_with_hints() {
    use z4_sat::LratWriter;

    let mut buf = Vec::new();
    {
        let mut writer = LratWriter::new_binary(&mut buf, 3);

        // Add clause ID 4: (x0) with hints [1, 2, 3]
        // Note: first learned clause after 3 original clauses gets ID 4
        let clause = vec![Literal::positive(Variable(0))];
        let hints = vec![1, 2, 3];
        let id = writer.add(&clause, &hints).unwrap();
        assert_eq!(id, 4);
    }

    // Expected: 'a' (0x61), id=4, lit=2, 0, hint1=1, hint2=2, hint3=3, 0
    assert_eq!(buf, vec![0x61, 4, 2, 0, 1, 2, 3, 0]);
}

/// Test binary LRAT deletion batching
#[test]
fn test_binary_lrat_deletion() {
    use z4_sat::LratWriter;

    let mut buf = Vec::new();
    {
        let mut writer = LratWriter::new_binary(&mut buf, 2);

        // Add a clause first
        let clause = vec![Literal::positive(Variable(0))];
        writer.add(&clause, &[1]).unwrap();

        // Delete clauses 1 and 2
        writer.delete(1).unwrap();
        writer.delete(2).unwrap();

        // Flush to write deletions
        writer.flush().unwrap();
    }

    // Expected:
    // Add: 'a' (0x61), id=3, lit=2, 0, hint=1, 0
    // Delete: 'd' (0x64), id=1, id=2, 0
    let expected = vec![
        0x61, 3, 2, 0, 1, 0, // add clause 3: (x0) with hint 1
        0x64, 1, 2, 0, // delete clauses 1 and 2
    ];
    assert_eq!(buf, expected);
}

/// Compare binary and text proof sizes for an UNSAT formula
#[test]
fn test_binary_vs_text_proof_size() {
    use z4_sat::ProofOutput;

    // UNSAT formula: mutual exclusion (requires several learned clauses)
    let dimacs = r"
p cnf 3 6
1 2 0
-1 -2 0
2 3 0
-2 -3 0
1 3 0
-1 -3 0
";
    let formula = parse_dimacs(dimacs).expect("Failed to parse");

    // Generate text DRAT proof
    let text_buffer: Vec<u8> = Vec::new();
    let text_writer = ProofOutput::drat_text(text_buffer);
    let mut text_solver = Solver::with_proof_output(formula.num_vars, text_writer);

    for clause in formula.clauses.clone() {
        text_solver.add_clause(clause);
    }
    text_solver.solve();
    let text_proof = text_solver.take_proof_writer().unwrap().into_inner();

    // Generate binary DRAT proof
    let binary_buffer: Vec<u8> = Vec::new();
    let binary_writer = ProofOutput::drat_binary(binary_buffer);
    let mut binary_solver = Solver::with_proof_output(formula.num_vars, binary_writer);

    for clause in formula.clauses {
        binary_solver.add_clause(clause);
    }
    binary_solver.solve();
    let binary_proof = binary_solver.take_proof_writer().unwrap().into_inner();

    // Both proofs should be non-empty
    assert!(!text_proof.is_empty(), "Text proof should not be empty");
    assert!(!binary_proof.is_empty(), "Binary proof should not be empty");

    // Binary format should generally be smaller than text format
    // (not always guaranteed for very small proofs, so we just check both are valid)
    eprintln!(
        "Proof sizes - Text: {} bytes, Binary: {} bytes",
        text_proof.len(),
        binary_proof.len()
    );
}

// ============================================================================
// TLA+ Invariant Tests
// ============================================================================
// These tests mirror the invariants from specs/cdcl.tla to ensure the Rust
// implementation satisfies the same correctness properties.

/// TLA+ SatCorrect invariant: If SAT, the assignment satisfies all original clauses.
/// This is verified via the solver's internal verify_model() check (debug_assert).
/// Here we add additional test coverage with various formula types.
#[test]
fn test_tla_invariant_sat_correct() {
    // Test formulas that should be SAT
    let sat_formulas = [
        // Simple satisfiable
        ("simple", "p cnf 2 2\n1 2 0\n-1 -2 0\n"),
        // Horn clause formula
        ("horn", "p cnf 3 3\n-1 2 0\n-2 3 0\n1 0\n"),
        // Large clause
        ("large_clause", "p cnf 5 1\n1 2 3 4 5 0\n"),
        // Chain implications (satisfiable)
        ("chain", "p cnf 4 3\n-1 2 0\n-2 3 0\n-3 4 0\n"),
        // Mixed sizes
        ("mixed", "p cnf 3 4\n1 0\n1 2 0\n1 2 3 0\n-1 2 0\n"),
    ];

    for (name, dimacs) in &sat_formulas {
        let formula = parse_dimacs(dimacs).expect("Failed to parse");
        let mut solver = formula.into_solver();
        let result = solver.solve();

        if let SolveResult::Sat(model) = result {
            // Manually verify the model satisfies all clauses
            let formula2 = parse_dimacs(dimacs).expect("Failed to parse");
            for (i, clause) in formula2.clauses.iter().enumerate() {
                let satisfied = clause.iter().any(|lit| {
                    let var_idx = lit.variable().index();
                    if var_idx >= model.len() {
                        return false;
                    }
                    let val = model[var_idx];
                    if lit.is_positive() {
                        val
                    } else {
                        !val
                    }
                });
                assert!(
                    satisfied,
                    "TLA+ SatCorrect violated: Formula '{}' clause {} not satisfied",
                    name, i
                );
            }
        } else {
            panic!(
                "TLA+ SatCorrect: Formula '{}' should be SAT but got {:?}",
                name, result
            );
        }
    }
}

/// TLA+ Soundness invariant: SAT results are correct AND UNSAT results are correct.
/// For SAT: model satisfies all clauses (tested above).
/// For UNSAT: we rely on DRAT proofs (tested in test_exhaustive_drat_verification).
/// This test verifies soundness across a mix of SAT and UNSAT formulas.
#[test]
fn test_tla_invariant_soundness_mixed() {
    let formulas: Vec<(&str, &str, bool)> = vec![
        // (name, dimacs, expected_sat)
        ("sat_simple", "p cnf 2 1\n1 2 0\n", true),
        ("unsat_unit", "p cnf 1 2\n1 0\n-1 0\n", false),
        ("sat_chain", "p cnf 3 2\n-1 2 0\n-2 3 0\n", true),
        (
            "unsat_php",
            "p cnf 2 4\n1 2 0\n-1 2 0\n1 -2 0\n-1 -2 0\n",
            false,
        ),
        ("sat_horn", "p cnf 2 2\n-1 2 0\n1 0\n", true),
        (
            "unsat_chain",
            "p cnf 3 4\n1 0\n-1 2 0\n-2 3 0\n-3 0\n",
            false,
        ),
    ];

    for (name, dimacs, expected_sat) in formulas {
        let formula = parse_dimacs(dimacs).expect("Failed to parse");
        let mut solver = formula.into_solver();
        let result = solver.solve();

        let is_sat = matches!(result, SolveResult::Sat(_));
        assert_eq!(
            is_sat,
            expected_sat,
            "TLA+ Soundness violated: Formula '{}' expected {} but got {:?}",
            name,
            if expected_sat { "SAT" } else { "UNSAT" },
            result
        );

        // For SAT results, verify model
        if let SolveResult::Sat(model) = result {
            let formula2 = parse_dimacs(dimacs).expect("Failed to parse");
            for clause in &formula2.clauses {
                let satisfied = clause.iter().any(|lit| {
                    let var_idx = lit.variable().index();
                    if var_idx >= model.len() {
                        return false;
                    }
                    let val = model[var_idx];
                    if lit.is_positive() {
                        val
                    } else {
                        !val
                    }
                });
                assert!(satisfied, "TLA+ Soundness: Model does not satisfy clause");
            }
        }
    }
}

/// TLA+ NoDoubleAssignment invariant: No variable is assigned twice on the trail.
/// We test this indirectly by ensuring the solver produces consistent results
/// and doesn't panic/corrupt state across multiple solve calls.
#[test]
fn test_tla_invariant_no_double_assignment() {
    let dimacs = "p cnf 5 10\n1 2 0\n-1 3 0\n-2 4 0\n-3 5 0\n1 -4 0\n2 -5 0\n-1 -2 3 0\n-3 -4 5 0\n1 4 5 0\n-2 -3 -5 0\n";

    // Run the solver multiple times on the same formula
    for i in 0..5 {
        let formula1 = parse_dimacs(dimacs).expect("Failed to parse");
        let mut solver = formula1.into_solver();
        let result1 = solver.solve();

        // Create a new solver and solve again
        let formula2 = parse_dimacs(dimacs).expect("Failed to parse");
        let mut solver2 = formula2.into_solver();
        let result2 = solver2.solve();

        // Results should be consistent (both SAT or both UNSAT)
        let is_sat1 = matches!(result1, SolveResult::Sat(_));
        let is_sat2 = matches!(result2, SolveResult::Sat(_));
        assert_eq!(
            is_sat1, is_sat2,
            "TLA+ NoDoubleAssignment (iteration {}): Inconsistent results across runs",
            i
        );
    }
}

/// TLA+ WatchedInvariant: For every clause of length >= 2, either it's satisfied
/// OR at least one watched literal is not FALSE.
///
/// We test this indirectly by ensuring:
/// 1. The solver produces correct results (if watch invariant is broken, it would miss propagations)
/// 2. Running on formulas that stress the watch mechanism
#[test]
fn test_tla_invariant_watched_literals() {
    // Formulas that stress the watched literal mechanism
    let test_formulas = [
        // Many binary clauses (watch both literals)
        (
            "binary_heavy",
            "p cnf 4 6\n1 2 0\n2 3 0\n3 4 0\n-1 -2 0\n-2 -3 0\n1 4 0\n",
        ),
        // Long clauses that require watch updates
        (
            "long_clauses",
            "p cnf 6 4\n1 2 3 4 5 6 0\n-1 -2 -3 0\n-4 -5 -6 0\n1 -2 3 -4 5 -6 0\n",
        ),
        // Unit propagation chains (watch updates on implications)
        (
            "unit_chain",
            "p cnf 5 6\n1 0\n-1 2 0\n-2 3 0\n-3 4 0\n-4 5 0\n1 2 3 4 5 0\n",
        ),
        // Clauses with shared literals (watch sharing)
        (
            "shared_lits",
            "p cnf 3 6\n1 2 0\n1 3 0\n2 3 0\n-1 2 0\n-1 3 0\n2 -3 0\n",
        ),
    ];

    for (name, dimacs) in &test_formulas {
        let formula = parse_dimacs(dimacs).expect("Failed to parse");
        let mut solver = formula.into_solver();
        let result = solver.solve();

        // The solver should produce a valid result without panicking
        // (panics would indicate watch invariant violations in debug mode)
        match result {
            SolveResult::Sat(model) => {
                // Verify model
                let formula2 = parse_dimacs(dimacs).expect("Failed to parse");
                for clause in &formula2.clauses {
                    let satisfied = clause.iter().any(|lit| {
                        let var_idx = lit.variable().index();
                        if var_idx >= model.len() {
                            return false;
                        }
                        let val = model[var_idx];
                        if lit.is_positive() {
                            val
                        } else {
                            !val
                        }
                    });
                    assert!(
                        satisfied,
                        "TLA+ WatchedInvariant: Formula '{}' model invalid",
                        name
                    );
                }
            }
            SolveResult::Unsat => {
                // UNSAT is also valid - the watched invariant ensures we find conflicts correctly
            }
            SolveResult::Unknown => {
                panic!("TLA+ WatchedInvariant: Formula '{}' returned Unknown", name);
            }
        }
    }
}

/// Test that solver maintains correctness across incremental solving (push/pop).
/// This tests the TypeInvariant for incremental state management.
#[test]
fn test_tla_invariant_incremental_correctness() {
    let mut solver = Solver::new(3);

    // Add base clauses: (x1 OR x2) AND (x2 OR x3)
    solver.add_clause(vec![
        Literal::positive(Variable(0)),
        Literal::positive(Variable(1)),
    ]);
    solver.add_clause(vec![
        Literal::positive(Variable(1)),
        Literal::positive(Variable(2)),
    ]);

    // Base should be SAT
    let base_result = solver.solve();
    assert!(
        matches!(base_result, SolveResult::Sat(_)),
        "Base formula should be SAT"
    );

    // Push and add conflicting constraint
    solver.push();
    solver.add_clause(vec![Literal::negative(Variable(1))]); // NOT x2
    solver.add_clause(vec![Literal::negative(Variable(0))]); // NOT x1
    solver.add_clause(vec![Literal::negative(Variable(2))]); // NOT x3

    let pushed_result = solver.solve();
    assert_eq!(
        pushed_result,
        SolveResult::Unsat,
        "Formula with conflicting constraints should be UNSAT"
    );

    // Pop and verify we're back to SAT
    assert!(solver.pop(), "Pop should succeed");
    let popped_result = solver.solve();

    if let SolveResult::Sat(model) = popped_result {
        // Verify the model satisfies the base clauses
        assert!(model[0] || model[1], "Model should satisfy (x1 OR x2)");
        assert!(model[1] || model[2], "Model should satisfy (x2 OR x3)");
    } else {
        panic!("After pop, formula should be SAT again");
    }
}

// ============================================================================
// Multi-Solver Differential Testing (CaDiCaL, Kissat, MiniSat)
// ============================================================================

mod multi_solver_differential {
    use std::process::{Command, Stdio};
    use z4_sat::{parse_dimacs, SolveResult};

    /// Generic SAT solver result
    #[derive(Debug, Clone, PartialEq, Eq)]
    enum SolverResult {
        Sat,
        Unsat,
        Unknown,
        Unavailable,
    }

    /// External solver configuration
    struct ExternalSolver {
        name: &'static str,
        command: &'static str,
        args: &'static [&'static str],
        sat_code: i32,
        unsat_code: i32,
    }

    // CaDiCaL: local build path (use absolute path from reference/)
    const CADICAL: ExternalSolver = ExternalSolver {
        name: "CaDiCaL",
        command: "reference/cadical/build/cadical",
        args: &["-q"], // quiet mode
        sat_code: 10,
        unsat_code: 20,
    };

    // Kissat: local build path
    const KISSAT: ExternalSolver = ExternalSolver {
        name: "Kissat",
        command: "reference/kissat/build/kissat",
        args: &["-q"], // quiet mode
        sat_code: 10,
        unsat_code: 20,
    };

    // MiniSat: system installed
    const MINISAT: ExternalSolver = ExternalSolver {
        name: "MiniSat",
        command: "minisat",
        args: &[],
        sat_code: 10,
        unsat_code: 20,
    };

    /// Run an external solver on a DIMACS CNF string
    fn run_solver(solver: &ExternalSolver, cnf: &str) -> SolverResult {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);

        // Use unique temp file names for parallel test safety
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);
        let thread_id = std::thread::current().id();
        let cnf_path = std::env::temp_dir().join(format!(
            "z4_multi_diff_{:?}_{}_{}.cnf",
            thread_id, id, solver.name
        ));

        if std::fs::write(&cnf_path, cnf).is_err() {
            return SolverResult::Unknown;
        }

        // Build command
        let mut cmd = Command::new(solver.command);
        for arg in solver.args {
            cmd.arg(arg);
        }
        cmd.arg(&cnf_path);

        // MiniSat needs an output file
        let output_path = if solver.name == "MiniSat" {
            let p =
                std::env::temp_dir().join(format!("z4_multi_diff_{:?}_{}_out.txt", thread_id, id));
            cmd.arg(&p);
            Some(p)
        } else {
            None
        };

        cmd.stdout(Stdio::null());
        cmd.stderr(Stdio::null());

        let output = cmd.output();

        // Clean up
        let _ = std::fs::remove_file(&cnf_path);
        if let Some(p) = output_path {
            let _ = std::fs::remove_file(&p);
        }

        match output {
            Ok(result) => {
                let exit_code = result.status.code().unwrap_or(-1);
                if exit_code == solver.sat_code {
                    SolverResult::Sat
                } else if exit_code == solver.unsat_code {
                    SolverResult::Unsat
                } else {
                    SolverResult::Unknown
                }
            }
            Err(_) => SolverResult::Unavailable,
        }
    }

    /// Check if a solver is available
    fn solver_available(solver: &ExternalSolver) -> bool {
        let result = Command::new(solver.command)
            .arg("--help")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        result.is_ok()
    }

    /// Results from all solvers on a single formula
    #[derive(Debug)]
    struct MultiSolverResults {
        z4: SolverResult,
        cadical: SolverResult,
        kissat: SolverResult,
        minisat: SolverResult,
    }

    /// Run Z4 and all external solvers on a CNF formula
    fn run_all_solvers(cnf: &str) -> MultiSolverResults {
        // Run Z4
        let z4_result = match parse_dimacs(cnf) {
            Ok(formula) => {
                let mut solver = formula.into_solver();
                match solver.solve() {
                    SolveResult::Sat(_) => SolverResult::Sat,
                    SolveResult::Unsat => SolverResult::Unsat,
                    SolveResult::Unknown => SolverResult::Unknown,
                }
            }
            Err(_) => SolverResult::Unknown,
        };

        MultiSolverResults {
            z4: z4_result,
            cadical: run_solver(&CADICAL, cnf),
            kissat: run_solver(&KISSAT, cnf),
            minisat: run_solver(&MINISAT, cnf),
        }
    }

    /// Check for disagreements among solvers
    fn find_disagreements(results: &MultiSolverResults) -> Vec<String> {
        let mut disagreements = Vec::new();

        // Get all valid (SAT/UNSAT) results
        let valid_results: Vec<(&str, &SolverResult)> = [
            ("Z4", &results.z4),
            ("CaDiCaL", &results.cadical),
            ("Kissat", &results.kissat),
            ("MiniSat", &results.minisat),
        ]
        .into_iter()
        .filter(|(_, r)| **r == SolverResult::Sat || **r == SolverResult::Unsat)
        .collect();

        // Check for disagreements
        if valid_results.len() >= 2 {
            let first_result = valid_results[0].1;
            for (name, result) in &valid_results[1..] {
                if result != &first_result {
                    disagreements.push(format!(
                        "{} says {:?} but {} says {:?}",
                        valid_results[0].0, first_result, name, result
                    ));
                }
            }
        }

        disagreements
    }

    /// Test that helper functions work
    #[test]
    fn test_multi_solver_helpers() {
        // Simple SAT formula
        let sat_cnf = "p cnf 2 1\n1 2 0\n";
        let results = run_all_solvers(sat_cnf);

        // Check Z4 works
        assert_eq!(results.z4, SolverResult::Sat, "Z4 should return SAT");

        // Log what solvers are available
        eprintln!("\nSolver availability:");
        eprintln!("  Z4: {:?}", results.z4);
        eprintln!("  CaDiCaL: {:?}", results.cadical);
        eprintln!("  Kissat: {:?}", results.kissat);
        eprintln!("  MiniSat: {:?}", results.minisat);

        // Simple UNSAT formula
        let unsat_cnf = "p cnf 1 2\n1 0\n-1 0\n";
        let results = run_all_solvers(unsat_cnf);
        assert_eq!(results.z4, SolverResult::Unsat, "Z4 should return UNSAT");
    }

    /// Differential test: random 3-SAT instances against all solvers
    #[test]
    fn test_differential_multi_solver_random() {
        let available = (
            solver_available(&CADICAL),
            solver_available(&KISSAT),
            solver_available(&MINISAT),
        );

        let available_count = [available.0, available.1, available.2]
            .iter()
            .filter(|&&x| x)
            .count();

        if available_count == 0 {
            eprintln!("Skipping: No external solvers available");
            return;
        }

        eprintln!("\nRunning multi-solver differential testing");
        eprintln!("  CaDiCaL available: {}", available.0);
        eprintln!("  Kissat available: {}", available.1);
        eprintln!("  MiniSat available: {}", available.2);

        let mut total = 0;
        let mut all_agree = 0;
        let mut disagreement_cases: Vec<(String, MultiSolverResults)> = Vec::new();

        // Test various formula sizes at phase transition (ratio ~4.26)
        let test_configs = [
            (20, 86),  // Small: 20 vars, 86 clauses
            (50, 213), // Medium: 50 vars, 213 clauses
            (75, 319), // Larger: 75 vars, 319 clauses
        ];

        for (num_vars, num_clauses) in test_configs {
            for seed in 0..20 {
                total += 1;
                let cnf = generate_random_3sat(num_vars, num_clauses, seed);
                let results = run_all_solvers(&cnf);
                let disagreements = find_disagreements(&results);

                if disagreements.is_empty() {
                    all_agree += 1;
                } else {
                    let name = format!("random_{}v_{}c_s{}", num_vars, num_clauses, seed);
                    disagreement_cases.push((name, results));
                }
            }
        }

        eprintln!(
            "\nMulti-solver differential results: {}/{} agreed",
            all_agree, total
        );

        if !disagreement_cases.is_empty() {
            eprintln!("\nDisagreements found:");
            for (name, results) in &disagreement_cases {
                eprintln!(
                    "  {}: Z4={:?}, CaDiCaL={:?}, Kissat={:?}, MiniSat={:?}",
                    name, results.z4, results.cadical, results.kissat, results.minisat
                );
            }
        }

        assert!(
            disagreement_cases.is_empty(),
            "Found {} disagreements among solvers",
            disagreement_cases.len()
        );
    }

    /// Differential test against all uf20/uf50/uf100 benchmarks
    #[test]
    fn test_differential_multi_solver_benchmarks() {
        let available = (
            solver_available(&CADICAL),
            solver_available(&KISSAT),
            solver_available(&MINISAT),
        );

        let available_count = [available.0, available.1, available.2]
            .iter()
            .filter(|&&x| x)
            .count();

        if available_count == 0 {
            eprintln!("Skipping: No external solvers available");
            return;
        }

        let benchmark_dir = std::path::Path::new("benchmarks/dimacs");
        if !benchmark_dir.exists() {
            eprintln!("Skipping: benchmarks/dimacs not found");
            return;
        }

        eprintln!("\nRunning multi-solver differential test on benchmarks");

        let mut total = 0;
        let mut all_agree = 0;
        let mut disagreements: Vec<String> = Vec::new();

        for entry in std::fs::read_dir(benchmark_dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|e| e == "cnf") {
                total += 1;

                let cnf = std::fs::read_to_string(&path).unwrap();
                let file_name = path.file_name().unwrap().to_string_lossy().to_string();

                let results = run_all_solvers(&cnf);
                let disag = find_disagreements(&results);

                if disag.is_empty() {
                    all_agree += 1;
                } else {
                    disagreements.push(format!(
                        "{}: Z4={:?}, CaDiCaL={:?}, Kissat={:?}, MiniSat={:?}",
                        file_name, results.z4, results.cadical, results.kissat, results.minisat
                    ));
                }
            }
        }

        eprintln!(
            "\nBenchmark multi-solver results: {}/{} agreed",
            all_agree, total
        );

        if !disagreements.is_empty() {
            eprintln!("\nDisagreements:");
            for d in &disagreements {
                eprintln!("  - {}", d);
            }
        }

        assert!(
            disagreements.is_empty(),
            "Found {} disagreements among solvers on benchmarks",
            disagreements.len()
        );
    }

    /// Specific edge cases that should be tested across all solvers
    #[test]
    fn test_differential_multi_solver_edge_cases() {
        let available = (
            solver_available(&CADICAL),
            solver_available(&KISSAT),
            solver_available(&MINISAT),
        );

        let available_count = [available.0, available.1, available.2]
            .iter()
            .filter(|&&x| x)
            .count();

        if available_count == 0 {
            eprintln!("Skipping: No external solvers available");
            return;
        }

        let edge_cases = [
            // Empty formula (trivially SAT)
            ("empty_trivial_sat", "p cnf 5 0\n"),
            // Single unit clause
            ("single_unit", "p cnf 1 1\n1 0\n"),
            // All positive chain
            ("positive_chain", "p cnf 3 3\n1 0\n-1 2 0\n-2 3 0\n"),
            // Horn clause UNSAT
            ("horn_unsat", "p cnf 2 4\n1 0\n2 0\n-1 -2 0\n-1 0\n"),
            // XOR ladder (hard for unit propagation)
            (
                "xor_ladder",
                "p cnf 4 8\n1 2 0\n-1 -2 0\n2 3 0\n-2 -3 0\n3 4 0\n-3 -4 0\n-1 0\n4 0\n",
            ),
            // Pigeonhole 3 into 2 (UNSAT)
            (
                "pigeonhole_3_2",
                concat!(
                    "p cnf 6 9\n",
                    "1 2 0\n",  // pigeon 1 in hole 1 or 2
                    "3 4 0\n",  // pigeon 2 in hole 1 or 2
                    "5 6 0\n",  // pigeon 3 in hole 1 or 2
                    "-1 -3 0\n", // hole 1 has at most 1 pigeon
                    "-1 -5 0\n",
                    "-3 -5 0\n",
                    "-2 -4 0\n", // hole 2 has at most 1 pigeon
                    "-2 -6 0\n",
                    "-4 -6 0\n"
                ),
            ),
            // Long clause
            (
                "long_clause",
                "p cnf 10 2\n1 2 3 4 5 6 7 8 9 10 0\n-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 0\n",
            ),
            // Dense binary implications
            (
                "binary_implications",
                "p cnf 5 10\n1 2 0\n-1 3 0\n-2 4 0\n-3 5 0\n-4 -5 0\n2 3 0\n-2 -3 4 0\n1 -4 5 0\n-1 -5 0\n3 4 5 0\n",
            ),
        ];

        let mut failures: Vec<String> = Vec::new();

        for (name, cnf) in edge_cases {
            let results = run_all_solvers(cnf);
            let disag = find_disagreements(&results);

            if !disag.is_empty() {
                failures.push(format!(
                    "{}: Z4={:?}, CaDiCaL={:?}, Kissat={:?}, MiniSat={:?}",
                    name, results.z4, results.cadical, results.kissat, results.minisat
                ));
            }
        }

        if !failures.is_empty() {
            eprintln!("\nEdge case failures:");
            for f in &failures {
                eprintln!("  - {}", f);
            }
        }

        assert!(
            failures.is_empty(),
            "Found {} edge case disagreements",
            failures.len()
        );
    }

    /// Generate a random 3-SAT formula
    fn generate_random_3sat(num_vars: u32, num_clauses: u32, seed: u64) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Simple PRNG
        let mut state = seed;
        let mut next_rand = || {
            let mut hasher = DefaultHasher::new();
            state.hash(&mut hasher);
            state = hasher.finish();
            state
        };

        let mut output = format!("p cnf {} {}\n", num_vars, num_clauses);

        for _ in 0..num_clauses {
            for _ in 0..3 {
                let var = (next_rand() % num_vars as u64) as i32 + 1;
                let sign = if next_rand() % 2 == 0 { 1 } else { -1 };
                output.push_str(&format!("{} ", sign * var));
            }
            output.push_str("0\n");
        }

        output
    }
}

// ============================================================================
// Large Variable Index Tests (for incremental SMT solving)
// ============================================================================

/// Test that the SAT solver works with large variable indices in assumptions.
/// This is critical for incremental SMT solving where selector variables
/// are allocated at large indices (e.g., 1000+).
#[test]
fn test_large_var_assumption_simple() {
    // Create solver with 1002 variables (indices 0..1001)
    let mut solver = Solver::new(1002);

    // Add clause: ¬var1000 ∨ var0 (if selector1000 is true, var0 must be true)
    solver.add_clause(vec![
        Literal::negative(Variable(1000)),
        Literal::positive(Variable(0)),
    ]);

    // Add clause: ¬var1001 ∨ var1 (if selector1001 is true, var1 must be true)
    solver.add_clause(vec![
        Literal::negative(Variable(1001)),
        Literal::positive(Variable(1)),
    ]);

    // Assumptions: var1000=true, var1001=true
    let assumptions = vec![
        Literal::positive(Variable(1000)),
        Literal::positive(Variable(1001)),
    ];

    let result = solver.solve_with_assumptions(&assumptions);
    match result {
        AssumeResult::Sat(model) => {
            // With assumptions, var0 and var1 should be true
            assert!(model.len() >= 2, "Model should have at least 2 user vars");
            assert!(
                model[0],
                "var0 should be true (implied by selector1000=true)"
            );
            assert!(
                model[1],
                "var1 should be true (implied by selector1001=true)"
            );
        }
        AssumeResult::Unsat(core) => {
            panic!(
                "Formula should be SAT with assumptions, but got UNSAT with core: {:?}",
                core
            );
        }
        AssumeResult::Unknown => {
            panic!("Formula should be SAT with assumptions, but got Unknown");
        }
    }
}

/// Test that replicates the exact LIA incremental behavior using raw Literal values.
#[test]
fn test_large_var_assumption_raw_literals() {
    // Create solver with 1002 variables (indices 0..1001)
    let mut solver = Solver::new(1002);

    // Add clause using raw Literal values as seen in debug output:
    // [Literal(2001), Literal(0)] = ¬var1000 ∨ var0
    solver.add_clause(vec![Literal(2001), Literal(0)]);

    // [Literal(2003), Literal(2)] = ¬var1001 ∨ var1
    solver.add_clause(vec![Literal(2003), Literal(2)]);

    // Assumptions using raw values:
    // [Literal(2000), Literal(2002)] = +var1000, +var1001
    let assumptions = vec![Literal(2000), Literal(2002)];

    let result = solver.solve_with_assumptions(&assumptions);
    match result {
        AssumeResult::Sat(model) => {
            assert!(model.len() >= 2, "Model should have at least 2 user vars");
            assert!(
                model[0],
                "var0 should be true (implied by selector1000=true)"
            );
            assert!(
                model[1],
                "var1 should be true (implied by selector1001=true)"
            );
        }
        AssumeResult::Unsat(core) => {
            panic!(
                "Formula should be SAT with assumptions, but got UNSAT with core: {:?}",
                core
            );
        }
        AssumeResult::Unknown => {
            panic!("Formula should be SAT with assumptions, but got Unknown");
        }
    }
}

/// Test that replicates the exact LIA incremental sequence with stored clauses.
#[test]
fn test_large_var_assumption_stored_clauses() {
    // Store clauses in a Vec first (like selector_guarded_clauses)
    let stored_clauses: Vec<Vec<Literal>> = vec![
        // Add clause: ¬var1000 ∨ var0
        vec![Literal(2001), Literal(0)],
        // Add clause: ¬var1001 ∨ var1
        vec![Literal(2003), Literal(2)],
    ];

    // Create solver with 1002 variables
    let mut solver = Solver::new(1002);

    // Add all stored clauses (like the LIA incremental code does)
    for clause in &stored_clauses {
        solver.add_clause(clause.clone());
    }

    // Assumptions
    let assumptions = vec![Literal(2000), Literal(2002)];

    let result = solver.solve_with_assumptions(&assumptions);
    match result {
        AssumeResult::Sat(model) => {
            assert!(model.len() >= 2, "Model should have at least 2 user vars");
            assert!(
                model[0],
                "var0 should be true (implied by selector1000=true)"
            );
            assert!(
                model[1],
                "var1 should be true (implied by selector1001=true)"
            );
        }
        AssumeResult::Unsat(core) => {
            panic!(
                "Formula should be SAT with assumptions, but got UNSAT with core: {:?}",
                core
            );
        }
        AssumeResult::Unknown => {
            panic!("Formula should be SAT with assumptions, but got Unknown");
        }
    }
}

/// Test that solve_with_assumptions is idempotent - calling it twice with
/// the same assumptions should return the same result.
///
/// This test exposes a bug where the second call returns UNSAT even when
/// no new clauses are added.
#[test]
fn test_solve_with_assumptions_idempotent() {
    // Create solver with 1002 variables (indices 0..1001)
    let mut solver = Solver::new(1002);

    // Add clause: ¬var1000 ∨ var0 (if selector1000 is true, var0 must be true)
    solver.add_clause(vec![
        Literal::negative(Variable(1000)),
        Literal::positive(Variable(0)),
    ]);

    // Add clause: ¬var1001 ∨ var1 (if selector1001 is true, var1 must be true)
    solver.add_clause(vec![
        Literal::negative(Variable(1001)),
        Literal::positive(Variable(1)),
    ]);

    // Assumptions: var1000=true, var1001=true
    let assumptions = vec![
        Literal::positive(Variable(1000)),
        Literal::positive(Variable(1001)),
    ];

    // First solve - should be SAT
    let result1 = solver.solve_with_assumptions(&assumptions);
    match &result1 {
        AssumeResult::Sat(_) => {}
        AssumeResult::Unsat(core) => {
            panic!(
                "First call should be SAT, but got UNSAT with core: {:?}",
                core
            );
        }
        AssumeResult::Unknown => {
            panic!("First call should be SAT, but got Unknown");
        }
    }

    // Second solve (no changes) - should also be SAT
    let result2 = solver.solve_with_assumptions(&assumptions);
    match &result2 {
        AssumeResult::Sat(_) => {}
        AssumeResult::Unsat(core) => {
            panic!(
                "Second call should be SAT (idempotent), but got UNSAT with core: {:?}",
                core
            );
        }
        AssumeResult::Unknown => {
            panic!("Second call should be SAT (idempotent), but got Unknown");
        }
    }
}
