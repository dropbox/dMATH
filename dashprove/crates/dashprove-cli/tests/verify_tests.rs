//! Integration tests for verify CLI command

use serial_test::serial;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_dir(prefix: &str) -> std::path::PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("dashprove_verify_test_{prefix}_{ts}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn dashprove_cmd() -> Command {
    Command::new(env!("CARGO_BIN_EXE_dashprove"))
}

/// Create a simple valid USL file for testing
fn create_simple_usl(dir: &std::path::Path) -> std::path::PathBuf {
    let usl = r#"
// Simple test specification
theorem excluded_middle {
    forall x: Bool . x or not x
}

invariant always_true {
    true
}
"#;
    let path = dir.join("simple.usl");
    std::fs::write(&path, usl).unwrap();
    path
}

/// Create a USL file with multiple properties
fn create_multi_property_usl(dir: &std::path::Path) -> std::path::PathBuf {
    let usl = r#"
// Multiple properties specification
theorem implication {
    forall p: Bool, q: Bool . (p and (p implies q)) implies q
}

theorem de_morgan {
    forall a: Bool, b: Bool . not (a and b) == (not a or not b)
}

invariant positive_squared {
    forall n: Int . n >= 0 implies n * n >= 0
}

invariant always_true {
    true
}
"#;
    let path = dir.join("multi.usl");
    std::fs::write(&path, usl).unwrap();
    path
}

/// Create a USL file with temporal properties
fn create_temporal_usl(dir: &std::path::Path) -> std::path::PathBuf {
    let usl = r#"
// Temporal properties specification
type Process = {
    id: Int,
    enabled: Bool
}

temporal no_deadlock {
    always(
        exists p: Process .
            p.enabled == true
    )
}
"#;
    let path = dir.join("temporal.usl");
    std::fs::write(&path, usl).unwrap();
    path
}

/// Create a USL file with contract properties
fn create_contract_usl(dir: &std::path::Path) -> std::path::PathBuf {
    let usl = r#"
// Contract specification
contract divide(x: Int, y: Int) -> Result<Int> {
    requires {
        y != 0
    }
    ensures {
        result * y == x
    }
    ensures_err {
        y == 0
    }
}
"#;
    let path = dir.join("contract.usl");
    std::fs::write(&path, usl).unwrap();
    path
}

/// Create an invalid USL file for error testing
fn create_invalid_usl(dir: &std::path::Path) -> std::path::PathBuf {
    let usl = r#"
// Invalid specification - syntax error
theorem broken {
    forall x: Bool . x and and y
}
"#;
    let path = dir.join("invalid.usl");
    std::fs::write(&path, usl).unwrap();
    path
}

// ============================================================================
// Basic verify command tests
// ============================================================================

#[test]
#[serial]
fn test_verify_simple_usl() {
    let dir = temp_dir("simple");
    let usl_path = create_simple_usl(&dir);

    let output = dashprove_cmd()
        .arg("verify")
        .arg(&usl_path)
        .output()
        .expect("Failed to execute command");

    // Should complete without error
    assert!(
        output.status.success(),
        "verify failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should mention properties verified
    assert!(
        stdout.contains("excluded_middle") || stdout.contains("always_true"),
        "Output should mention property names: {}",
        stdout
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_verify_multi_property_usl() {
    let dir = temp_dir("multi");
    let usl_path = create_multi_property_usl(&dir);

    let output = dashprove_cmd()
        .arg("verify")
        .arg(&usl_path)
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "verify failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should process multiple properties
    assert!(
        stdout.contains("implication")
            || stdout.contains("de_morgan")
            || stdout.contains("positive_squared"),
        "Output should mention multiple properties: {}",
        stdout
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_verify_temporal_usl() {
    let dir = temp_dir("temporal");
    let usl_path = create_temporal_usl(&dir);

    let output = dashprove_cmd()
        .arg("verify")
        .arg(&usl_path)
        .output()
        .expect("Failed to execute command");

    // Temporal properties require TLA+ backend - either succeeds or fails gracefully
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success()
            || stderr.contains("No backend available")
            || stderr.contains("Selector error"),
        "verify temporal failed unexpectedly: {}",
        stderr
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_verify_contract_usl() {
    let dir = temp_dir("contract");
    let usl_path = create_contract_usl(&dir);

    let output = dashprove_cmd()
        .arg("verify")
        .arg(&usl_path)
        .output()
        .expect("Failed to execute command");

    // Contracts require Kani backend - either succeeds or fails gracefully
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success()
            || stderr.contains("No backend available")
            || stderr.contains("Selector error"),
        "verify contract failed unexpectedly: {}",
        stderr
    );

    std::fs::remove_dir_all(&dir).ok();
}

// ============================================================================
// Error handling tests
// ============================================================================

#[test]
#[serial]
fn test_verify_invalid_usl_syntax() {
    let dir = temp_dir("invalid");
    let usl_path = create_invalid_usl(&dir);

    let output = dashprove_cmd()
        .arg("verify")
        .arg(&usl_path)
        .output()
        .expect("Failed to execute command");

    // Should fail with parse error
    assert!(
        !output.status.success(),
        "verify should fail on invalid syntax"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    // Should mention parse/syntax error
    assert!(
        stderr.to_lowercase().contains("parse")
            || stderr.to_lowercase().contains("error")
            || stderr.to_lowercase().contains("unexpected"),
        "Error should mention parse problem: {}",
        stderr
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_verify_nonexistent_file() {
    let output = dashprove_cmd()
        .arg("verify")
        .arg("/nonexistent/path/file.usl")
        .output()
        .expect("Failed to execute command");

    // Should fail with file not found
    assert!(
        !output.status.success(),
        "verify should fail for nonexistent file"
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

// ============================================================================
// Option flag tests
// ============================================================================

#[test]
#[serial]
fn test_verify_with_verbose() {
    let dir = temp_dir("verbose");
    let usl_path = create_simple_usl(&dir);

    let output = dashprove_cmd()
        .arg("verify")
        .arg(&usl_path)
        .arg("--verbose")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "verify with --verbose failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Verbose should produce more output
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should have some descriptive output
    assert!(
        stdout.len() > 10,
        "Verbose mode should produce meaningful output"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_verify_with_skip_health_check() {
    let dir = temp_dir("skip_health");
    let usl_path = create_simple_usl(&dir);

    let output = dashprove_cmd()
        .arg("verify")
        .arg(&usl_path)
        .arg("--skip-health-check")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "verify with --skip-health-check failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_verify_with_timeout() {
    let dir = temp_dir("timeout");
    let usl_path = create_simple_usl(&dir);

    let output = dashprove_cmd()
        .arg("verify")
        .arg(&usl_path)
        .arg("--timeout")
        .arg("60")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "verify with --timeout failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_verify_with_suggest() {
    let dir = temp_dir("suggest");
    let usl_path = create_simple_usl(&dir);

    let output = dashprove_cmd()
        .arg("verify")
        .arg(&usl_path)
        .arg("--suggest")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "verify with --suggest failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    std::fs::remove_dir_all(&dir).ok();
}

// ============================================================================
// Backend selection tests
// ============================================================================

#[test]
#[serial]
fn test_verify_with_lean_backend() {
    let dir = temp_dir("lean");
    let usl_path = create_simple_usl(&dir);

    let output = dashprove_cmd()
        .arg("verify")
        .arg(&usl_path)
        .arg("--backends")
        .arg("lean")
        .output()
        .expect("Failed to execute command");

    // Should complete (may or may not actually run lean depending on installation)
    assert!(
        output.status.success(),
        "verify with lean backend failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_verify_with_tlaplus_backend() {
    let dir = temp_dir("tlaplus");
    let usl_path = create_temporal_usl(&dir);

    let output = dashprove_cmd()
        .arg("verify")
        .arg(&usl_path)
        .arg("--backends")
        .arg("tla+")
        .output()
        .expect("Failed to execute command");

    // Backend may not be available - either succeeds or fails gracefully
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success()
            || stderr.contains("No backend")
            || stderr.contains("Selector error"),
        "verify with tla+ backend failed unexpectedly: {}",
        stderr
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_verify_with_kani_backend() {
    let dir = temp_dir("kani");
    let usl_path = create_contract_usl(&dir);

    let output = dashprove_cmd()
        .arg("verify")
        .arg(&usl_path)
        .arg("--backends")
        .arg("kani")
        .output()
        .expect("Failed to execute command");

    // Backend may not be available - either succeeds or fails gracefully
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success()
            || stderr.contains("No backend available")
            || stderr.contains("Selector error"),
        "verify with kani backend failed unexpectedly: {}",
        stderr
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_verify_with_multiple_backends() {
    let dir = temp_dir("multi_backend");
    let usl_path = create_simple_usl(&dir);

    let output = dashprove_cmd()
        .arg("verify")
        .arg(&usl_path)
        .arg("--backends")
        .arg("lean,alloy")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "verify with multiple backends failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    std::fs::remove_dir_all(&dir).ok();
}

// ============================================================================
// Learning integration tests
// ============================================================================

#[test]
#[serial]
fn test_verify_with_learn() {
    let dir = temp_dir("learn");
    let usl_path = create_simple_usl(&dir);
    let data_dir = dir.join("data");
    std::fs::create_dir_all(&data_dir).unwrap();

    let output = dashprove_cmd()
        .arg("verify")
        .arg(&usl_path)
        .arg("--learn")
        .arg("--data-dir")
        .arg(&data_dir)
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "verify with --learn failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Learning data directory should have been used
    // (may or may not create files depending on verification outcome)

    std::fs::remove_dir_all(&dir).ok();
}

// ============================================================================
// ML-based verification tests
// ============================================================================

#[test]
#[serial]
fn test_verify_with_ml_no_model() {
    let dir = temp_dir("ml_no_model");
    let usl_path = create_simple_usl(&dir);

    let output = dashprove_cmd()
        .arg("verify")
        .arg(&usl_path)
        .arg("--ml")
        .output()
        .expect("Failed to execute command");

    // Should complete even without a model (falls back to non-ML)
    assert!(
        output.status.success(),
        "verify with --ml (no model) failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_verify_with_ml_confidence() {
    let dir = temp_dir("ml_confidence");
    let usl_path = create_simple_usl(&dir);

    let output = dashprove_cmd()
        .arg("verify")
        .arg(&usl_path)
        .arg("--ml")
        .arg("--ml-confidence")
        .arg("0.7")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "verify with --ml-confidence failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    std::fs::remove_dir_all(&dir).ok();
}

// ============================================================================
// Using examples/usl files directly
// ============================================================================

#[test]
#[serial]
fn test_verify_examples_basic_usl() {
    // Use the actual example file from the repository
    let example_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/usl/basic.usl");

    if !example_path.exists() {
        // Skip if example file doesn't exist
        return;
    }

    let output = dashprove_cmd()
        .arg("verify")
        .arg(&example_path)
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "verify examples/usl/basic.usl failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
#[serial]
fn test_verify_examples_temporal_usl() {
    // Note: Temporal properties require TLA+ backend which may not always be available.
    // This test verifies the example can at least be parsed and processed.
    let example_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/usl/temporal.usl");

    if !example_path.exists() {
        return;
    }

    let output = dashprove_cmd()
        .arg("verify")
        .arg(&example_path)
        .output()
        .expect("Failed to execute command");

    // Either succeeds (TLA+ available) or fails with "No backend available" (expected)
    // but should NOT fail with a parse error
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success()
            || stderr.contains("No backend available")
            || stderr.contains("Selector error"),
        "verify examples/usl/temporal.usl failed unexpectedly: {}",
        stderr
    );
}

#[test]
#[serial]
fn test_verify_examples_contracts_usl() {
    let example_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/usl/contracts.usl");

    if !example_path.exists() {
        return;
    }

    let output = dashprove_cmd()
        .arg("verify")
        .arg(&example_path)
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "verify examples/usl/contracts.usl failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// ============================================================================
// Help and version tests
// ============================================================================

#[test]
#[serial]
fn test_verify_help() {
    let output = dashprove_cmd()
        .arg("verify")
        .arg("--help")
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success(), "verify --help failed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should show verify-specific help
    assert!(
        stdout.contains("--backends") || stdout.contains("specification"),
        "Help should mention verify options: {}",
        stdout
    );
}
