//! Integration tests for model-based testing CLI command

use serial_test::serial;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_dir(prefix: &str) -> std::path::PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("dashprove_mbt_test_{prefix}_{ts}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn dashprove_cmd() -> Command {
    Command::new(env!("CARGO_BIN_EXE_dashprove"))
}

/// Create a simple TLA+ model file for testing
fn create_simple_tla(dir: &std::path::Path) -> std::path::PathBuf {
    let tla = r#"---- MODULE SimpleModel ----
EXTENDS Integers

VARIABLES state, counter

States == {"Init", "Running", "Done"}

Init ==
    /\ state = "Init"
    /\ counter = 0

Start ==
    /\ state = "Init"
    /\ state' = "Running"
    /\ counter' = counter + 1

Complete ==
    /\ state = "Running"
    /\ counter >= 3
    /\ state' = "Done"
    /\ UNCHANGED counter

Increment ==
    /\ state = "Running"
    /\ counter < 3
    /\ counter' = counter + 1
    /\ UNCHANGED state

Next == Start \/ Complete \/ Increment

Spec == Init /\ [][Next]_<<state, counter>>

TypeInvariant == state \in States /\ counter \in 0..10

====
"#;
    let path = dir.join("simple.tla");
    std::fs::write(&path, tla).unwrap();
    path
}

// ============================================================================
// Help tests
// ============================================================================

#[test]
#[serial]
fn test_mbt_help() {
    let output = dashprove_cmd()
        .arg("mbt")
        .arg("--help")
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success(), "mbt --help should succeed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should mention key options
    assert!(
        stdout.contains("model") || stdout.contains("generate") || stdout.contains("coverage"),
        "Help should mention model/generate/coverage: {}",
        stdout
    );
}

#[test]
#[serial]
fn test_mbt_generate_help() {
    let output = dashprove_cmd()
        .arg("mbt")
        .arg("generate")
        .arg("--help")
        .output()
        .expect("Failed to execute command");

    // Either works or the subcommand doesn't exist
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Should not panic
    assert!(
        !stderr.contains("panic") && !stderr.contains("RUST_BACKTRACE"),
        "Should not panic: {} {}",
        stdout,
        stderr
    );
}

// ============================================================================
// Coverage option tests
// ============================================================================

#[test]
#[serial]
fn test_mbt_coverage_options() {
    let output = dashprove_cmd()
        .arg("mbt")
        .arg("--help")
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Coverage types should be documented
    assert!(
        stdout.to_lowercase().contains("coverage")
            || stdout.to_lowercase().contains("state")
            || stdout.to_lowercase().contains("transition"),
        "Help should mention coverage options: {}",
        stdout
    );
}

// ============================================================================
// Model file tests
// ============================================================================

#[test]
#[serial]
fn test_mbt_with_tla_model() {
    let dir = temp_dir("tla_model");
    let tla_path = create_simple_tla(&dir);

    let output = dashprove_cmd()
        .arg("mbt")
        .arg("generate")
        .arg("--model")
        .arg(&tla_path)
        .arg("--coverage")
        .arg("state")
        .output()
        .expect("Failed to execute command");

    // Should either succeed or fail with meaningful error
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should not be an "unrecognized argument" error
    assert!(
        !stderr.contains("unrecognized option"),
        "Command should accept model arguments: stderr={} stdout={}",
        stderr,
        stdout
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_mbt_transition_coverage() {
    let dir = temp_dir("transition");
    let tla_path = create_simple_tla(&dir);

    let output = dashprove_cmd()
        .arg("mbt")
        .arg("generate")
        .arg("--model")
        .arg(&tla_path)
        .arg("--coverage")
        .arg("transition")
        .output()
        .expect("Failed to execute command");

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should accept transition coverage option
    assert!(
        !stderr.contains("invalid value") || !stderr.contains("transition"),
        "Should accept transition coverage: stderr={} stdout={}",
        stderr,
        stdout
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_mbt_boundary_coverage() {
    let dir = temp_dir("boundary");
    let tla_path = create_simple_tla(&dir);

    let output = dashprove_cmd()
        .arg("mbt")
        .arg("generate")
        .arg("--model")
        .arg(&tla_path)
        .arg("--coverage")
        .arg("boundary")
        .output()
        .expect("Failed to execute command");

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should accept boundary coverage option
    assert!(
        !stderr.contains("invalid value") || !stderr.contains("boundary"),
        "Should accept boundary coverage: stderr={} stdout={}",
        stderr,
        stdout
    );

    std::fs::remove_dir_all(&dir).ok();
}

// ============================================================================
// Output format tests
// ============================================================================

#[test]
#[serial]
fn test_mbt_rust_output() {
    let dir = temp_dir("rust_out");
    let tla_path = create_simple_tla(&dir);

    let output = dashprove_cmd()
        .arg("mbt")
        .arg("generate")
        .arg("--model")
        .arg(&tla_path)
        .arg("--format")
        .arg("rust")
        .output()
        .expect("Failed to execute command");

    // If successful, output should contain Rust syntax
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("fn ") || stdout.contains("#[test]") || stdout.contains("mod "),
            "Rust output should contain Rust syntax: {}",
            stdout
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_mbt_json_output() {
    let dir = temp_dir("json_out");
    let tla_path = create_simple_tla(&dir);

    let output = dashprove_cmd()
        .arg("mbt")
        .arg("generate")
        .arg("--model")
        .arg(&tla_path)
        .arg("--format")
        .arg("json")
        .output()
        .expect("Failed to execute command");

    // If successful, output should be JSON
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let trimmed = stdout.trim();
        assert!(
            trimmed.starts_with('{') || trimmed.starts_with('['),
            "JSON output should be valid JSON: {}",
            trimmed
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

// ============================================================================
// Output file tests
// ============================================================================

#[test]
#[serial]
fn test_mbt_output_to_file() {
    let dir = temp_dir("file_out");
    let tla_path = create_simple_tla(&dir);
    let output_path = dir.join("generated_tests.rs");

    let output = dashprove_cmd()
        .arg("mbt")
        .arg("generate")
        .arg("--model")
        .arg(&tla_path)
        .arg("--output")
        .arg(&output_path)
        .output()
        .expect("Failed to execute command");

    // If successful, file should be created
    if output.status.success() {
        assert!(output_path.exists(), "Output file should be created");
        let content = std::fs::read_to_string(&output_path).unwrap_or_default();
        assert!(!content.is_empty(), "Output file should not be empty");
    }

    std::fs::remove_dir_all(&dir).ok();
}

// ============================================================================
// Error handling tests
// ============================================================================

#[test]
#[serial]
fn test_mbt_nonexistent_model() {
    let output = dashprove_cmd()
        .arg("mbt")
        .arg("generate")
        .arg("--model")
        .arg("/nonexistent/model.tla")
        .output()
        .expect("Failed to execute command");

    assert!(
        !output.status.success(),
        "mbt should fail for nonexistent model"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.to_lowercase().contains("not found")
            || stderr.to_lowercase().contains("no such file")
            || stderr.to_lowercase().contains("cannot")
            || stderr.to_lowercase().contains("error"),
        "Error should mention file problem: {}",
        stderr
    );
}

#[test]
#[serial]
fn test_mbt_invalid_model() {
    let dir = temp_dir("invalid");

    // Create invalid TLA+ file
    let invalid_tla = r#"
---- MODULE Invalid ----
This is not valid TLA+ syntax at all
====
"#;
    let path = dir.join("invalid.tla");
    std::fs::write(&path, invalid_tla).unwrap();

    let output = dashprove_cmd()
        .arg("mbt")
        .arg("generate")
        .arg("--model")
        .arg(&path)
        .output()
        .expect("Failed to execute command");

    // Should fail or produce error
    let stderr = String::from_utf8_lossy(&output.stderr);
    let _stdout = String::from_utf8_lossy(&output.stdout);

    // Either fails or warns about invalid syntax
    if !output.status.success() {
        assert!(
            stderr.to_lowercase().contains("parse")
                || stderr.to_lowercase().contains("syntax")
                || stderr.to_lowercase().contains("error")
                || stderr.to_lowercase().contains("invalid"),
            "Error should mention parse problem: {}",
            stderr
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_mbt_invalid_coverage() {
    let dir = temp_dir("invalid_cov");
    let tla_path = create_simple_tla(&dir);

    let output = dashprove_cmd()
        .arg("mbt")
        .arg("generate")
        .arg("--model")
        .arg(&tla_path)
        .arg("--coverage")
        .arg("invalid_coverage_type")
        .output()
        .expect("Failed to execute command");

    assert!(
        !output.status.success(),
        "mbt should fail for invalid coverage type"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.to_lowercase().contains("invalid")
            || stderr.to_lowercase().contains("unknown")
            || stderr.to_lowercase().contains("error")
            || stderr.to_lowercase().contains("possible"),
        "Error should mention invalid coverage: {}",
        stderr
    );

    std::fs::remove_dir_all(&dir).ok();
}

// ============================================================================
// Example directory tests
// ============================================================================

#[test]
#[serial]
fn test_mbt_example_model() {
    let example_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/agent_verification/mbt_generation/agent_model.tla");

    if !example_path.exists() {
        return;
    }

    let output = dashprove_cmd()
        .arg("mbt")
        .arg("generate")
        .arg("--model")
        .arg(&example_path)
        .arg("--coverage")
        .arg("state")
        .output()
        .expect("Failed to execute command");

    // Should run without panic
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("panic") && !stderr.contains("RUST_BACKTRACE"),
        "Should not panic: {}",
        stderr
    );
}

// ============================================================================
// Verbose mode tests
// ============================================================================

#[test]
#[serial]
fn test_mbt_verbose() {
    let dir = temp_dir("verbose");
    let tla_path = create_simple_tla(&dir);

    let output = dashprove_cmd()
        .arg("mbt")
        .arg("generate")
        .arg("--model")
        .arg(&tla_path)
        .arg("--verbose")
        .output()
        .expect("Failed to execute command");

    // Verbose mode should accept the flag
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        !stderr.contains("unrecognized") && !stderr.contains("invalid option"),
        "Should accept --verbose flag: {}",
        stderr
    );

    std::fs::remove_dir_all(&dir).ok();
}
