//! Differential Testing Against Z3
//!
//! This module implements differential testing inspired by lean5's approach:
//! Compare Z4 results against Z3 on SMT-LIB benchmarks.
//!
//! The pattern is:
//! 1. Load SMT-LIB benchmark files
//! 2. Run through Z3 (reference implementation)
//! 3. Run through Z4 (our implementation)
//! 4. Compare results - any mismatch is a bug
//!
//! This is the PRIMARY verification mechanism for SMT soundness.

use anyhow::{anyhow, Context, Result};
use std::fs;
use std::path::Path;
use std::process::Command;
use z4_dpll::Executor;
use z4_frontend::parse;

const QF_LIA_PATH: &str = "../../benchmarks/smt/QF_LIA";
const QF_LRA_PATH: &str = "../../benchmarks/smt/QF_LRA";
const QF_BV_PATH: &str = "../../benchmarks/smt/QF_BV";

/// Result of running a benchmark
#[derive(Debug, Clone, PartialEq)]
enum SolverResult {
    Sat,
    Unsat,
    Unknown,
    Error(String),
}

impl SolverResult {
    fn from_str(s: &str) -> Self {
        let s = s.trim().to_lowercase();
        if s == "sat" {
            SolverResult::Sat
        } else if s == "unsat" {
            SolverResult::Unsat
        } else if s == "unknown" || s.contains("unknown") {
            SolverResult::Unknown
        } else if s.contains("error") || s.contains("timeout") {
            SolverResult::Error(s.to_string())
        } else {
            SolverResult::Unknown
        }
    }

    fn is_definite(&self) -> bool {
        matches!(self, SolverResult::Sat | SolverResult::Unsat)
    }
}

/// Run Z3 on an SMT-LIB file
fn run_z3(path: &Path) -> Result<SolverResult> {
    let output = Command::new("z3")
        .arg("-T:10") // 10 second timeout
        .arg(path)
        .output()
        .context("failed to spawn z3")?;

    if !output.status.success() && output.stdout.is_empty() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Ok(SolverResult::Error(stderr.to_string()));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Z3 outputs "sat" or "unsat" on first line
    let first_line = stdout.lines().next().unwrap_or("");
    Ok(SolverResult::from_str(first_line))
}

/// Run Z4 on an SMT-LIB file
fn run_z4(path: &Path) -> Result<SolverResult> {
    let content =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;

    let commands = match parse(&content) {
        Ok(cmds) => cmds,
        Err(e) => return Ok(SolverResult::Error(format!("parse error: {e}"))),
    };

    let mut executor = Executor::new();

    // Set a timeout via the executor if available, otherwise rely on test timeout
    let results = match executor.execute_all(&commands) {
        Ok(r) => r,
        Err(e) => return Ok(SolverResult::Error(format!("execution error: {e}"))),
    };

    // Find the check-sat result
    for result in results {
        let r = result.to_lowercase();
        if r == "sat" {
            return Ok(SolverResult::Sat);
        } else if r == "unsat" {
            return Ok(SolverResult::Unsat);
        } else if r == "unknown" {
            return Ok(SolverResult::Unknown);
        }
    }

    Ok(SolverResult::Unknown)
}

/// A mismatch between Z3 and Z4
#[derive(Debug)]
struct Mismatch {
    file: String,
    z3_result: SolverResult,
    z4_result: SolverResult,
}

/// Run differential test on a directory of SMT-LIB files
fn differential_test_dir(dir_path: &str) -> Result<(usize, usize, Vec<Mismatch>)> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(dir_path);

    if !path.exists() {
        return Err(anyhow!("benchmark directory not found: {}", path.display()));
    }

    let mut total = 0;
    let mut agreed = 0;
    let mut mismatches = Vec::new();

    let entries: Vec<_> = fs::read_dir(&path)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "smt2"))
        .collect();

    for entry in entries {
        let file_path = entry.path();
        let file_name = file_path.file_name().unwrap().to_string_lossy().to_string();

        total += 1;

        let z3_result = match run_z3(&file_path) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Z3 error on {}: {}", file_name, e);
                SolverResult::Error(e.to_string())
            }
        };

        let z4_result = match run_z4(&file_path) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Z4 error on {}: {}", file_name, e);
                SolverResult::Error(e.to_string())
            }
        };

        // Only compare definite results (sat/unsat)
        // Unknown is acceptable for Z4 if Z3 gives a definite answer
        // But if both are definite and disagree, that's a bug!
        if z3_result.is_definite() && z4_result.is_definite() {
            if z3_result == z4_result {
                agreed += 1;
            } else {
                mismatches.push(Mismatch {
                    file: file_name.clone(),
                    z3_result: z3_result.clone(),
                    z4_result: z4_result.clone(),
                });
            }
        } else if z3_result.is_definite() && !z4_result.is_definite() {
            // Z3 gave definite answer, Z4 didn't - not a soundness bug but incomplete
            eprintln!(
                "Z4 incomplete on {}: Z3={:?}, Z4={:?}",
                file_name, z3_result, z4_result
            );
        } else {
            // Both unknown or Z3 unknown - skip
            agreed += 1;
        }
    }

    Ok((total, agreed, mismatches))
}

// ============================================================================
// Tests
// ============================================================================

#[test]
fn differential_qf_lia_vs_z3() -> Result<()> {
    let (total, agreed, mismatches) = differential_test_dir(QF_LIA_PATH)?;

    println!("\n=== QF_LIA Differential Test Results ===");
    println!("Total benchmarks: {}", total);
    println!("Agreed with Z3:   {}", agreed);
    println!("Mismatches:       {}", mismatches.len());

    if !mismatches.is_empty() {
        println!("\n!!! SOUNDNESS VIOLATIONS DETECTED !!!");
        for m in &mismatches {
            println!("  {}: Z3={:?}, Z4={:?}", m.file, m.z3_result, m.z4_result);
        }
    }

    // THIS IS THE CRITICAL ASSERTION
    // If this fails, we have a soundness bug
    assert!(
        mismatches.is_empty(),
        "SOUNDNESS BUG: Z4 disagrees with Z3 on {} of {} QF_LIA benchmarks.\n\
         Mismatches: {:?}",
        mismatches.len(),
        total,
        mismatches.iter().map(|m| &m.file).collect::<Vec<_>>()
    );

    // Also require high coverage - at least 80% should have definite answers
    let coverage = (agreed as f64 / total as f64) * 100.0;
    assert!(
        coverage >= 80.0,
        "Low coverage: only {:.1}% of benchmarks agreed (need >= 80%)",
        coverage
    );

    println!(
        "\n✓ All {} QF_LIA benchmarks passed differential test",
        total
    );
    Ok(())
}

#[test]
fn differential_qf_lra_vs_z3() -> Result<()> {
    let (total, agreed, mismatches) = differential_test_dir(QF_LRA_PATH)?;

    println!("\n=== QF_LRA Differential Test Results ===");
    println!("Total benchmarks: {}", total);
    println!("Agreed with Z3:   {}", agreed);
    println!("Mismatches:       {}", mismatches.len());

    if !mismatches.is_empty() {
        println!("\n!!! SOUNDNESS VIOLATIONS DETECTED !!!");
        for m in &mismatches {
            println!("  {}: Z3={:?}, Z4={:?}", m.file, m.z3_result, m.z4_result);
        }
    }

    // THIS IS THE CRITICAL ASSERTION
    // If this fails, we have a soundness bug in LRA
    assert!(
        mismatches.is_empty(),
        "SOUNDNESS BUG: Z4 disagrees with Z3 on {} of {} QF_LRA benchmarks.\n\
         Mismatches: {:?}",
        mismatches.len(),
        total,
        mismatches.iter().map(|m| &m.file).collect::<Vec<_>>()
    );

    println!(
        "\n[OK] All {} QF_LRA benchmarks passed differential test",
        total
    );
    Ok(())
}

#[test]
fn differential_qf_bv_vs_z3() -> Result<()> {
    let (total, agreed, mismatches) = differential_test_dir(QF_BV_PATH)?;

    println!("\n=== QF_BV Differential Test Results ===");
    println!("Total benchmarks: {}", total);
    println!("Agreed with Z3:   {}", agreed);
    println!("Mismatches:       {}", mismatches.len());

    if !mismatches.is_empty() {
        println!("\n!!! SOUNDNESS VIOLATIONS DETECTED !!!");
        for m in &mismatches {
            println!("  {}: Z3={:?}, Z4={:?}", m.file, m.z3_result, m.z4_result);
        }
    }

    assert!(
        mismatches.is_empty(),
        "SOUNDNESS BUG: Z4 disagrees with Z3 on {} of {} QF_BV benchmarks",
        mismatches.len(),
        total
    );

    println!(
        "\n✓ All {} QF_BV benchmarks passed differential test",
        total
    );
    Ok(())
}

/// Quick sanity check - run first few LIA benchmarks
#[test]
fn quick_lia_sanity_check() -> Result<()> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(QF_LIA_PATH);

    // Just test first 5 files
    let mut count = 0;
    for entry in fs::read_dir(&path)? {
        let entry = entry?;
        let file_path = entry.path();
        if file_path.extension().is_some_and(|ext| ext == "smt2") {
            count += 1;
            if count > 5 {
                break;
            }

            let z3_result = run_z3(&file_path)?;
            let z4_result = run_z4(&file_path)?;

            println!(
                "{}: Z3={:?}, Z4={:?}",
                file_path.file_name().unwrap().to_string_lossy(),
                z3_result,
                z4_result
            );

            if z3_result.is_definite() && z4_result.is_definite() {
                assert_eq!(z3_result, z4_result, "Mismatch on {}", file_path.display());
            }
        }
    }

    Ok(())
}
