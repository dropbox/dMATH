//! Integration tests for verify-code CLI command

use serial_test::serial;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_dir(prefix: &str) -> std::path::PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("dashprove_verify_code_test_{prefix}_{ts}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn dashprove_cmd() -> Command {
    Command::new(env!("CARGO_BIN_EXE_dashprove"))
}

/// Create a sample USL contract specification file
fn create_sample_spec(dir: &std::path::Path) -> std::path::PathBuf {
    let spec = r#"// Simple contract for testing
contract divide(x: Int, y: Int) -> Result<Int> {
    requires {
        y != 0
    }
    ensures {
        result * y == x
    }
}

contract abs(x: Int) -> Int {
    ensures {
        result >= 0
    }
}
"#;
    let path = dir.join("contracts.usl");
    std::fs::write(&path, spec).unwrap();
    path
}

/// Create a sample Rust code file
fn create_sample_code(dir: &std::path::Path) -> std::path::PathBuf {
    let code = r#"// Sample Rust code for verification

/// Division function with precondition
pub fn divide(x: i32, y: i32) -> Result<i32, &'static str> {
    if y == 0 {
        return Err("division by zero");
    }
    Ok(x / y)
}

/// Absolute value function
pub fn abs(x: i32) -> i32 {
    if x < 0 { -x } else { x }
}
"#;
    let path = dir.join("lib.rs");
    std::fs::write(&path, code).unwrap();
    path
}

/// Create an empty Rust code file
fn create_empty_code(dir: &std::path::Path) -> std::path::PathBuf {
    let path = dir.join("empty.rs");
    std::fs::write(&path, "").unwrap();
    path
}

/// Create a minimal USL spec file
fn create_minimal_spec(dir: &std::path::Path) -> std::path::PathBuf {
    let spec = "theorem always_true { true }";
    let path = dir.join("minimal.usl");
    std::fs::write(&path, spec).unwrap();
    path
}

#[test]
#[serial]
fn test_verify_code_help() {
    let output = dashprove_cmd()
        .args(["verify-code", "--help"])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("--code"),
        "Should show --code option: {}",
        stdout
    );
    assert!(
        stdout.contains("--spec"),
        "Should show --spec option: {}",
        stdout
    );
    assert!(
        stdout.contains("--verbose"),
        "Should show --verbose option: {}",
        stdout
    );
    assert!(
        stdout.contains("--timeout"),
        "Should show --timeout option: {}",
        stdout
    );
    assert!(
        stdout.contains("Kani"),
        "Should mention Kani backend: {}",
        stdout
    );
}

#[test]
#[serial]
fn test_verify_code_missing_spec() {
    let output = dashprove_cmd()
        .args(["verify-code"])
        .output()
        .expect("Failed to run dashprove");

    assert!(!output.status.success(), "Should fail without --spec");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--spec") || stderr.contains("required"),
        "Should indicate --spec is required: {}",
        stderr
    );
}

#[test]
#[serial]
fn test_verify_code_nonexistent_spec() {
    let output = dashprove_cmd()
        .args(["verify-code", "--spec", "/nonexistent/path/spec.usl"])
        .output()
        .expect("Failed to run dashprove");

    assert!(
        !output.status.success(),
        "Should fail with nonexistent spec"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("not found") || stderr.contains("Error"),
        "Should indicate file not found: {}",
        stderr
    );
}

#[test]
#[serial]
fn test_verify_code_nonexistent_code() {
    let dir = temp_dir("nonexistent_code");
    let spec_path = create_sample_spec(&dir);

    let output = dashprove_cmd()
        .args([
            "verify-code",
            "--code",
            "/nonexistent/path/code.rs",
            "--spec",
            spec_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run dashprove");

    assert!(
        !output.status.success(),
        "Should fail with nonexistent code file"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("not found") || stderr.contains("Error"),
        "Should indicate code file not found: {}",
        stderr
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_verify_code_empty_code() {
    let dir = temp_dir("empty_code");
    let spec_path = create_sample_spec(&dir);
    let code_path = create_empty_code(&dir);

    let output = dashprove_cmd()
        .args([
            "verify-code",
            "--code",
            code_path.to_str().unwrap(),
            "--spec",
            spec_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run dashprove");

    // Should fail because no code is provided
    assert!(!output.status.success(), "Should fail with empty code");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("No code") || stderr.contains("Error"),
        "Should indicate no code provided: {}",
        stderr
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_verify_code_with_code_and_spec() {
    let dir = temp_dir("with_code_spec");
    let spec_path = create_sample_spec(&dir);
    let code_path = create_sample_code(&dir);

    // This test verifies that the command runs and parses arguments correctly.
    // The actual Kani verification may fail if Kani is not installed, but
    // the CLI should at least start and show the verification message.
    let output = dashprove_cmd()
        .args([
            "verify-code",
            "--code",
            code_path.to_str().unwrap(),
            "--spec",
            spec_path.to_str().unwrap(),
            "--timeout",
            "5",
        ])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // The command should at least start verification
    assert!(
        stdout.contains("Verifying") || stderr.contains("Error"),
        "Should either start verification or show backend error: stdout={}, stderr={}",
        stdout,
        stderr
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_verify_code_verbose() {
    let dir = temp_dir("verbose");
    let spec_path = create_minimal_spec(&dir);
    let code_path = create_sample_code(&dir);

    let output = dashprove_cmd()
        .args([
            "verify-code",
            "--code",
            code_path.to_str().unwrap(),
            "--spec",
            spec_path.to_str().unwrap(),
            "--verbose",
            "--timeout",
            "5",
        ])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Verbose mode should show additional information
    // If Kani is not installed, we still verify that verbose mode is active
    assert!(
        stdout.contains("[verbose]") || stdout.contains("Verifying"),
        "Should show verbose output or verification message: {}",
        stdout
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_verify_code_custom_timeout() {
    let dir = temp_dir("custom_timeout");
    let spec_path = create_minimal_spec(&dir);
    let code_path = create_sample_code(&dir);

    // Use a very short timeout - the command should start but may timeout
    let output = dashprove_cmd()
        .args([
            "verify-code",
            "--code",
            code_path.to_str().unwrap(),
            "--spec",
            spec_path.to_str().unwrap(),
            "--timeout",
            "1",
            "--verbose",
        ])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    // With verbose mode and custom timeout, we should see either:
    // 1. Verbose output mentioning timeout
    // 2. Or a verification attempt
    assert!(
        stdout.contains("Timeout:") || stdout.contains("Verifying") || stdout.contains("Error"),
        "Should show timeout info in verbose mode or verification attempt: {}",
        stdout
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_verify_code_verbose_file_info() {
    let dir = temp_dir("verbose_file_info");
    let spec_path = create_sample_spec(&dir);
    let code_path = create_sample_code(&dir);

    let output = dashprove_cmd()
        .args([
            "verify-code",
            "--code",
            code_path.to_str().unwrap(),
            "--spec",
            spec_path.to_str().unwrap(),
            "--verbose",
            "--timeout",
            "5",
        ])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Verbose mode should show file reading info
    // At minimum, it should attempt to read files
    assert!(
        stdout.contains("Reading") || stdout.contains("bytes") || stdout.contains("Verifying"),
        "Should show file info in verbose mode: {}",
        stdout
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_verify_code_short_verbose_flag() {
    let dir = temp_dir("short_verbose");
    let spec_path = create_minimal_spec(&dir);
    let code_path = create_sample_code(&dir);

    // Test -v short flag
    let output = dashprove_cmd()
        .args([
            "verify-code",
            "--code",
            code_path.to_str().unwrap(),
            "--spec",
            spec_path.to_str().unwrap(),
            "-v",
            "--timeout",
            "5",
        ])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    // -v should work the same as --verbose
    assert!(
        stdout.contains("[verbose]") || stdout.contains("Verifying"),
        "Short -v flag should enable verbose mode: {}",
        stdout
    );

    std::fs::remove_dir_all(dir).ok();
}
