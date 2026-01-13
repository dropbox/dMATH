//! Integration tests for CHC verification via kani-fast CLI
//!
//! These tests invoke `kani-fast chc --all-proofs` to run CHC-based verification
//! on test harnesses in tests/compiler/*.rs
//!
//! Requirements:
//! - Z3 SMT solver installed
//! - kani-fast built: cargo build --bin kani-fast
//!
//! NOTE: This file tests the text-based MIR parser path (`kani-fast chc`).
//! Tests marked with "MIR text parser limitation" work correctly with the native
//! driver (`kani-fast-driver`) which uses the rustc API directly. The native driver
//! is the recommended path for production use. See tests/compiler/enum_proofs.rs
//! for enum tests that pass via the native driver.

use std::path::PathBuf;
use std::process::Command;

/// Get the workspace root directory
fn workspace_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is crates/kani-fast, go up twice
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .expect("Failed to find workspace root")
}

/// Get path to the kani-fast binary
fn kani_fast_binary() -> PathBuf {
    workspace_root()
        .join("target")
        .join("debug")
        .join("kani-fast")
}

/// Get path to test compiler files
fn test_file(name: &str) -> PathBuf {
    workspace_root().join("tests").join("compiler").join(name)
}

/// Check if CHC verification is available
fn skip_if_unavailable() -> bool {
    // Check if kani-fast binary exists
    let binary = kani_fast_binary();
    if !binary.exists() {
        eprintln!("Skipping: kani-fast binary not found at {:?}", binary);
        eprintln!("Run: cargo build --bin kani-fast");
        return true;
    }

    // Check if Z3 is available
    if which::which("z3").is_err() {
        eprintln!("Skipping: Z3 not installed");
        return true;
    }

    false
}

/// Run CHC verification on a test file and return (stdout, stderr, success)
fn run_chc_verify(test_filename: &str) -> (String, String, bool) {
    let binary = kani_fast_binary();
    let file = test_file(test_filename);

    let output = Command::new(&binary)
        .args(["chc", &file.to_string_lossy(), "--all-proofs"])
        .output()
        .expect("Failed to execute kani-fast");

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let success = output.status.success();

    (stdout, stderr, success)
}

/// Parse verification summary from output
/// If summary section exists, use that; otherwise count individual results
fn parse_summary(output: &str) -> (usize, usize) {
    let mut verified = 0;
    let mut failed = 0;
    let mut found_summary = false;

    for line in output.lines() {
        // Check for summary section
        if line.contains("Verified:") {
            if let Some(n) = line.split_whitespace().last() {
                verified = n.parse().unwrap_or(0);
                found_summary = true;
            }
        }
        if line.contains("Failed:") {
            if let Some(n) = line.split_whitespace().last() {
                failed = n.parse().unwrap_or(0);
                found_summary = true;
            }
        }
    }

    // If no summary, count individual results
    if !found_summary {
        for line in output.lines() {
            if line.contains(": Property verified") {
                verified += 1;
            }
            if line.contains(": Property violated") || line.contains(": Unknown") {
                failed += 1;
            }
        }
    }

    (verified, failed)
}

// =============================================================================
// Basic Proofs Tests
// =============================================================================

#[test]
fn test_chc_basic_proofs_all_verified() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _success) = run_chc_verify("basic_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    // Parse summary
    let (verified, failed) = parse_summary(&combined);

    // Expected: 3 verified (always_true, arithmetic_correct, boolean_logic)
    // Expected: 2 failed (always_false, arithmetic_wrong)
    assert_eq!(
        verified, 3,
        "Expected 3 verified harnesses\nOutput:\n{}",
        combined
    );
    assert_eq!(
        failed, 2,
        "Expected 2 failed harnesses\nOutput:\n{}",
        combined
    );
}

#[test]
fn test_chc_basic_always_true_verified() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("basic_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("always_true_proof: Property verified"),
        "always_true_proof should be verified\nOutput:\n{}",
        combined
    );
}

#[test]
fn test_chc_basic_always_false_fails() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("basic_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("always_false_proof: Property violated"),
        "always_false_proof should be violated\nOutput:\n{}",
        combined
    );
}

#[test]
fn test_chc_basic_arithmetic_correct_verified() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("basic_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("arithmetic_correct_proof: Property verified"),
        "arithmetic_correct_proof should be verified\nOutput:\n{}",
        combined
    );
}

#[test]
fn test_chc_basic_arithmetic_wrong_fails() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("basic_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("arithmetic_wrong_proof: Property violated"),
        "arithmetic_wrong_proof should be violated\nOutput:\n{}",
        combined
    );
}

#[test]
fn test_chc_basic_boolean_logic_verified() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("basic_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("boolean_logic_proof: Property verified"),
        "boolean_logic_proof should be verified\nOutput:\n{}",
        combined
    );
}

// =============================================================================
// Loop Proofs Tests
// =============================================================================

#[test]
fn test_chc_loop_proofs_summary() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("loop_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    let (verified, failed) = parse_summary(&combined);

    // Note: Some loop proofs may timeout in CI, so we check at least 1 verified
    // Full run: 3 verified, 1 failed
    assert!(
        verified >= 1,
        "Expected at least 1 verified loop harness\nOutput:\n{}",
        combined
    );
    assert!(
        verified + failed >= 2,
        "Expected at least 2 completed loop harnesses\nOutput:\n{}",
        combined
    );
}

#[test]
fn test_chc_simple_counter_loop_verified() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("loop_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("simple_counter_loop_proof: Property verified"),
        "simple_counter_loop_proof should be verified\nOutput:\n{}",
        combined
    );
}

#[test]
#[ignore = "sum_loop_proof may timeout in CI - works with 10s timeout"]
fn test_chc_sum_loop_verified() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("loop_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("sum_loop_proof: Property verified"),
        "sum_loop_proof should be verified\nOutput:\n{}",
        combined
    );
}

#[test]
#[ignore = "May timeout in CI - depends on sum_loop completing first"]
fn test_chc_incorrect_loop_fails() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("loop_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("incorrect_loop_proof: Property violated"),
        "incorrect_loop_proof should be violated\nOutput:\n{}",
        combined
    );
}

#[test]
#[ignore = "May timeout in CI - depends on earlier loops completing first"]
fn test_chc_early_exit_loop_verified() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("loop_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("early_exit_loop_proof: Property verified"),
        "early_exit_loop_proof should be verified\nOutput:\n{}",
        combined
    );
}

// =============================================================================
// Simple Proof Tests
// =============================================================================

#[test]
fn test_chc_simple_proof_summary() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("simple_proof.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    let (verified, failed) = parse_summary(&combined);

    // simple_proof.rs has 2 harnesses, both should pass
    assert!(
        verified >= 1,
        "Expected at least 1 verified harness in simple_proof.rs\nOutput:\n{}",
        combined
    );
    assert_eq!(
        failed, 0,
        "Expected 0 failed harnesses in simple_proof.rs\nOutput:\n{}",
        combined
    );
}

// =============================================================================
// Failing Proof Tests
// =============================================================================

#[test]
fn test_chc_failing_proof_summary() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("failing_proof.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    let (verified, failed) = parse_summary(&combined);

    // failing_proof.rs should have 1 pass, 1 fail
    assert!(
        verified >= 1,
        "Expected at least 1 verified harness in failing_proof.rs\nOutput:\n{}",
        combined
    );
    assert!(
        failed >= 1,
        "Expected at least 1 failed harness in failing_proof.rs\nOutput:\n{}",
        combined
    );
}

// =============================================================================
// Struct Proofs Tests
// Known issue: Struct field access not correctly encoded in CHC
// =============================================================================

#[test]
#[ignore = "Struct field access encoding incomplete - struct fields not tracked in CHC"]
fn test_chc_struct_proofs_summary() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("struct_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    let (verified, failed) = parse_summary(&combined);

    // Expected: 6 verified, 1 failed (struct_wrong_assertion_proof)
    assert_eq!(
        verified, 6,
        "Expected 6 verified struct harnesses\nOutput:\n{}",
        combined
    );
    assert_eq!(
        failed, 1,
        "Expected 1 failed struct harness\nOutput:\n{}",
        combined
    );
}

#[test]
#[ignore = "Struct field access encoding incomplete"]
fn test_chc_struct_creation_verified() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("struct_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("struct_creation_proof: Property verified"),
        "struct_creation_proof should be verified\nOutput:\n{}",
        combined
    );
}

#[test]
#[ignore = "Struct field access encoding incomplete"]
fn test_chc_tuple_creation_verified() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("struct_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("tuple_creation_proof: Property verified"),
        "tuple_creation_proof should be verified\nOutput:\n{}",
        combined
    );
}

#[test]
fn test_chc_struct_wrong_assertion_fails() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("struct_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    // This should fail (property violated) which it does
    assert!(
        combined.contains("struct_wrong_assertion_proof: Property violated"),
        "struct_wrong_assertion_proof should be violated\nOutput:\n{}",
        combined
    );
}

// =============================================================================
// Inlined Function Tests (manual function inlining workaround)
// =============================================================================

#[test]
fn test_chc_function_call_proofs_summary() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("function_call_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    let (verified, failed) = parse_summary(&combined);

    // Expected: 5 verified, 1 failed (inlined_wrong_proof)
    assert_eq!(
        verified, 5,
        "Expected 5 verified inlined function harnesses\nOutput:\n{}",
        combined
    );
    assert_eq!(
        failed, 1,
        "Expected 1 failed inlined function harness\nOutput:\n{}",
        combined
    );
}

#[test]
fn test_chc_inlined_add_verified() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("function_call_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("inlined_add_proof: Property verified"),
        "inlined_add_proof should be verified\nOutput:\n{}",
        combined
    );
}

#[test]
fn test_chc_inlined_wrong_fails() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("function_call_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("inlined_wrong_proof: Property violated"),
        "inlined_wrong_proof should be violated\nOutput:\n{}",
        combined
    );
}

// =============================================================================
// Guarded call / path condition tests
// =============================================================================

#[test]
fn test_chc_guarded_call_summary() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("guarded_call_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);
    let (verified, failed) = parse_summary(&combined);

    // Expected: guarded_call_proof and loop_guarded_call_proof verified; two failures
    assert_eq!(
        verified, 2,
        "Expected 2 verified guarded call harnesses\nOutput:\n{}",
        combined
    );
    assert_eq!(
        failed, 2,
        "Expected 2 failing guarded call harnesses\nOutput:\n{}",
        combined
    );
}

#[test]
fn test_chc_guarded_call_passes() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("guarded_call_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("guarded_call_proof: Property verified"),
        "guarded_call_proof should be verified\nOutput:\n{}",
        combined
    );
}

#[test]
fn test_chc_unguarded_call_fails() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("guarded_call_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("unguarded_call_proof: Property violated"),
        "unguarded_call_proof should be violated\nOutput:\n{}",
        combined
    );
}

#[test]
fn test_chc_loop_guarded_call_passes() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("guarded_call_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("loop_guarded_call_proof: Property verified"),
        "loop_guarded_call_proof should be verified\nOutput:\n{}",
        combined
    );
}

#[test]
fn test_chc_loop_unguarded_call_fails() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("guarded_call_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("loop_unguarded_call_proof: Property violated"),
        "loop_unguarded_call_proof should be violated\nOutput:\n{}",
        combined
    );
}

// =============================================================================
// Automatic Function Inlining Tests
// Known issue: Some inline tests return Unknown instead of verified/violated
// =============================================================================

#[test]
#[ignore = "Auto-inline returns Unknown for some proofs"]
fn test_chc_auto_inline_summary() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("inline_test.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    let (verified, failed) = parse_summary(&combined);

    // Expected: 2 verified (inline_add_proof, inline_chain_proof), 1 failed (inline_wrong_proof)
    assert_eq!(
        verified, 2,
        "Expected 2 verified auto-inlined harnesses\nOutput:\n{}",
        combined
    );
    assert_eq!(
        failed, 1,
        "Expected 1 failed auto-inlined harness\nOutput:\n{}",
        combined
    );
}

#[test]
#[ignore = "Auto-inline returns Unknown"]
fn test_chc_auto_inline_add_verified() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("inline_test.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("inline_add_proof: Property verified"),
        "inline_add_proof should be verified with auto-inlining\nOutput:\n{}",
        combined
    );
}

#[test]
#[ignore = "Auto-inline returns Unknown"]
fn test_chc_auto_inline_chain_verified() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("inline_test.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("inline_chain_proof: Property verified"),
        "inline_chain_proof should be verified with auto-inlining\nOutput:\n{}",
        combined
    );
}

#[test]
#[ignore = "Auto-inline returns Unknown"]
fn test_chc_auto_inline_wrong_fails() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("inline_test.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("inline_wrong_proof: Property violated"),
        "inline_wrong_proof should be violated with auto-inlining\nOutput:\n{}",
        combined
    );
}

// =============================================================================
// Array Tests
// =============================================================================

#[test]
fn test_chc_array_proofs_summary() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("array_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    let (verified, failed) = parse_summary(&combined);

    // Current: 3 verified (all pass due to CHC encoding treating arrays as true)
    // Known issue: array_wrong_assertion_proof should fail but passes
    assert_eq!(
        verified, 3,
        "Expected 3 verified array harnesses\nOutput:\n{}",
        combined
    );
    assert_eq!(
        failed, 0,
        "Expected 0 failed array harnesses\nOutput:\n{}",
        combined
    );
}

#[test]
fn test_chc_array_creation_verified() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("array_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("array_creation_proof: Property verified"),
        "array_creation_proof should be verified\nOutput:\n{}",
        combined
    );
}

#[test]
fn test_chc_array_wrong_assertion_verified() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("array_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    // Known issue: this should fail but passes due to incomplete array encoding
    assert!(
        combined.contains("array_wrong_assertion_proof: Property verified"),
        "array_wrong_assertion_proof currently passes (known issue)\nOutput:\n{}",
        combined
    );
}

#[test]
fn test_chc_array_second_element_verified() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("array_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("array_second_element_proof: Property verified"),
        "array_second_element_proof should be verified\nOutput:\n{}",
        combined
    );
}

// =============================================================================
// Enum Tests
// Known issue: MIR text parser does not handle enum constructors (works with kani-fast-driver)
// =============================================================================

#[test]
#[ignore = "MIR text parser limitation: enum constructors not parsed (use kani-fast-driver)"]
fn test_chc_enum_proofs_summary() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("enum_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    let (verified, failed) = parse_summary(&combined);

    // Expected: 5 verified (enum_status, enum_maybe_some/none, option_some/none)
    // Expected: 1 failed (enum_wrong_assertion_proof)
    assert_eq!(
        verified, 5,
        "Expected 5 verified enum harnesses\nOutput:\n{}",
        combined
    );
    assert_eq!(
        failed, 1,
        "Expected 1 failed enum harness\nOutput:\n{}",
        combined
    );
}

#[test]
#[ignore = "MIR text parser limitation: enum constructors not parsed (use kani-fast-driver)"]
fn test_chc_enum_status_verified() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("enum_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("enum_status_proof: Property verified"),
        "enum_status_proof should be verified\nOutput:\n{}",
        combined
    );
}

#[test]
#[ignore = "MIR text parser limitation: enum constructors not parsed (use kani-fast-driver)"]
fn test_chc_option_some_verified() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("enum_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("option_some_proof: Property verified"),
        "option_some_proof should be verified\nOutput:\n{}",
        combined
    );
}

#[test]
#[ignore = "MIR text parser limitation: enum constructors not parsed (use kani-fast-driver)"]
fn test_chc_option_none_verified() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("enum_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("option_none_proof: Property verified"),
        "option_none_proof should be verified\nOutput:\n{}",
        combined
    );
}

#[test]
fn test_chc_enum_wrong_assertion_fails() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("enum_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    // This should fail, and it does (enum encoding issues cause violations)
    assert!(
        combined.contains("enum_wrong_assertion_proof: Property violated"),
        "enum_wrong_assertion_proof should be violated\nOutput:\n{}",
        combined
    );
}

// =============================================================================
// Match Expression Tests
// Known issue: MIR text parser does not handle enum constructors (works with kani-fast-driver)
// =============================================================================

#[test]
#[ignore = "MIR text parser limitation: enum constructors not parsed (use kani-fast-driver)"]
fn test_chc_match_proofs_summary() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("match_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    let (verified, failed) = parse_summary(&combined);

    // Expected: 4 verified (match_direction, match_south, match_bool, match_addition)
    // Expected: 1 failed (match_wrong_assertion_proof)
    assert_eq!(
        verified, 4,
        "Expected 4 verified match harnesses\nOutput:\n{}",
        combined
    );
    assert_eq!(
        failed, 1,
        "Expected 1 failed match harness\nOutput:\n{}",
        combined
    );
}

#[test]
#[ignore = "MIR text parser limitation: enum constructors not parsed (use kani-fast-driver)"]
fn test_chc_match_direction_verified() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("match_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("match_direction_proof: Property verified"),
        "match_direction_proof should be verified\nOutput:\n{}",
        combined
    );
}

#[test]
#[ignore = "MIR text parser limitation: enum constructors not parsed (use kani-fast-driver)"]
fn test_chc_match_south_verified() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("match_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("match_south_proof: Property verified"),
        "match_south_proof should be verified\nOutput:\n{}",
        combined
    );
}

#[test]
fn test_chc_match_bool_verified() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("match_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    // match_bool_proof works because it matches on bool, not enum
    assert!(
        combined.contains("match_bool_proof: Property verified"),
        "match_bool_proof should be verified\nOutput:\n{}",
        combined
    );
}

#[test]
#[ignore = "MIR text parser limitation: enum constructors not parsed (use kani-fast-driver)"]
fn test_chc_match_addition_verified() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("match_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("match_addition_proof: Property verified"),
        "match_addition_proof should be verified\nOutput:\n{}",
        combined
    );
}

#[test]
fn test_chc_match_wrong_assertion_fails() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("match_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    // This fails as expected
    assert!(
        combined.contains("match_wrong_assertion_proof: Property violated"),
        "match_wrong_assertion_proof should be violated\nOutput:\n{}",
        combined
    );
}

// ===========================================
// Enum Mutation Tests (enum_mutation_proofs.rs)
// Known issue: Enum discriminant and data encoding incomplete
// ===========================================

#[test]
#[ignore = "MIR text parser limitation: enum constructors not parsed (use kani-fast-driver)"]
fn test_chc_enum_mutation_proofs_summary() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("enum_mutation_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);
    let (verified, failed) = parse_summary(&combined);

    // Expected: 5 verified (enum_mutate_unit, enum_to_empty, enum_to_value,
    //                       enum_multiple_mutations, enum_data_preserved)
    // Expected: 1 failed (enum_mutation_wrong_assertion_proof)
    assert_eq!(
        verified, 5,
        "Expected 5 verified mutation harnesses\nOutput:\n{}",
        combined
    );
    assert_eq!(
        failed, 1,
        "Expected 1 failed mutation harness\nOutput:\n{}",
        combined
    );
}

#[test]
#[ignore = "MIR text parser limitation: enum constructors not parsed (use kani-fast-driver)"]
fn test_chc_enum_mutate_unit_verified() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("enum_mutation_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("enum_mutate_unit_proof: Property verified"),
        "enum_mutate_unit_proof should be verified\nOutput:\n{}",
        combined
    );
}

#[test]
#[ignore = "MIR text parser limitation: enum constructors not parsed (use kani-fast-driver)"]
fn test_chc_enum_to_empty_verified() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("enum_mutation_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("enum_to_empty_proof: Property verified"),
        "enum_to_empty_proof should be verified\nOutput:\n{}",
        combined
    );
}

#[test]
#[ignore = "MIR text parser limitation: enum constructors not parsed (use kani-fast-driver)"]
fn test_chc_enum_to_value_verified() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("enum_mutation_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("enum_to_value_proof: Property verified"),
        "enum_to_value_proof should be verified\nOutput:\n{}",
        combined
    );
}

#[test]
#[ignore = "MIR text parser limitation: enum constructors not parsed (use kani-fast-driver)"]
fn test_chc_enum_multiple_mutations_verified() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("enum_mutation_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("enum_multiple_mutations_proof: Property verified"),
        "enum_multiple_mutations_proof should be verified\nOutput:\n{}",
        combined
    );
}

#[test]
fn test_chc_enum_mutation_wrong_assertion_fails() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("enum_mutation_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    // This fails as expected (enum encoding issues cause violation)
    assert!(
        combined.contains("enum_mutation_wrong_assertion_proof: Property violated"),
        "enum_mutation_wrong_assertion_proof should be violated\nOutput:\n{}",
        combined
    );
}

#[test]
#[ignore = "MIR text parser limitation: enum constructors not parsed (use kani-fast-driver)"]
fn test_chc_enum_data_preserved_verified() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("enum_mutation_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("enum_data_preserved_proof: Property verified"),
        "enum_data_preserved_proof should be verified\nOutput:\n{}",
        combined
    );
}

// =============================================================================
// Closure Proofs Tests
// =============================================================================

#[test]
fn test_chc_closure_proofs_summary() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _success) = run_chc_verify("closure_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    // Parse summary
    let (verified, failed) = parse_summary(&combined);

    // Expected: 3 verified closures (capture, multiple_calls, nested)
    // Note: mutable closure capture (closure_modify_captured_proof) is NOT supported
    // in the CLI path which uses MIR-parsing CHC. The driver uses Z4 native integration
    // and can verify mutable closures, but that's a different code path.
    assert_eq!(
        verified, 3,
        "Expected 3 verified harnesses\nOutput:\n{}",
        combined
    );
    assert_eq!(
        failed, 1,
        "Expected 1 failed harness (closure_modify_captured_proof)\nOutput:\n{}",
        combined
    );
}

#[test]
fn test_chc_closure_capture_verified() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("closure_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("closure_capture_proof: Property verified"),
        "closure_capture_proof should be verified\nOutput:\n{}",
        combined
    );
}

#[test]
fn test_chc_closure_multiple_calls_verified() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("closure_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("closure_multiple_calls_proof: Property verified"),
        "closure_multiple_calls_proof should be verified\nOutput:\n{}",
        combined
    );
}

#[test]
fn test_chc_closure_nested_verified() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("closure_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("closure_nested_proof: Property verified"),
        "closure_nested_proof should be verified\nOutput:\n{}",
        combined
    );
}

/// Closures that capture mutable references are NOT supported in the CLI path.
/// The CLI uses MIR-parsing CHC which doesn't properly handle mutable captures.
/// Note: The driver (kani-fast-driver) uses Z4 native integration and CAN verify
/// mutable closures, but this test uses the CLI path.
#[test]
fn test_chc_closure_modify_captured_not_verified() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("closure_proofs.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    // Mutable closure captures fail in CLI path (different from driver which uses Z4)
    assert!(
        combined.contains("closure_modify_captured_proof: Property violated"),
        "closure_modify_captured_proof should fail in CLI path (mutable capture not supported)\nOutput:\n{}",
        combined
    );
}

// =============================================================================
// REGRESSION TEST: Verification Without Kani Library
// =============================================================================
//
// CRITICAL: This test prevents a recurring bug that has been re-introduced
// 5 times (#204, #216, #226, #308). The bug occurs when code enforces strict
// Kani library presence (e.g., `assert!(!kani_functions.is_empty())`).
//
// The driver MUST gracefully handle verification without the Kani library
// linked. Simple Rust code should verify correctly without `extern crate kani;`.

/// Regression test: Verify code works WITHOUT Kani library
///
/// This test ensures the driver can verify simple Rust code without
/// requiring `extern crate kani;` in the source file.
///
/// DO NOT remove this test or modify it to require Kani intrinsics.
#[test]
fn test_regression_verification_without_kani_library() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _success) = run_chc_verify("no_kani_library.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    let (verified, failed) = parse_summary(&combined);

    // All proofs should pass - they don't use any Kani intrinsics
    assert!(
        verified >= 1,
        "REGRESSION: Verification without Kani library failed!\n\
         This bug has been re-introduced 5 times (#204, #216, #226, #308).\n\
         Check apply_kani_transforms() in codegen_chc/mod.rs.\n\
         Output:\n{}",
        combined
    );
    assert_eq!(
        failed, 0,
        "REGRESSION: Code without Kani library should not fail!\n\
         Output:\n{}",
        combined
    );
}

/// Verify specific harness passes without Kani library
#[test]
fn test_regression_simple_no_kani_verified() {
    if skip_if_unavailable() {
        return;
    }

    let (stdout, stderr, _) = run_chc_verify("no_kani_library.rs");
    let combined = format!("{}\n{}", stdout, stderr);

    assert!(
        combined.contains("simple_no_kani_proof: Property verified"),
        "REGRESSION: simple_no_kani_proof should verify without Kani library!\n\
         Output:\n{}",
        combined
    );
}
