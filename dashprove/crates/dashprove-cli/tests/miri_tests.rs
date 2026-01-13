//! Integration tests for MIRI CLI command

use serial_test::serial;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_dir(prefix: &str) -> std::path::PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("dashprove_miri_test_{prefix}_{ts}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn dashprove_cmd() -> Command {
    Command::new(env!("CARGO_BIN_EXE_dashprove"))
}

// ============================================================================
// Help tests
// ============================================================================

#[test]
#[serial]
fn test_miri_help() {
    let output = dashprove_cmd()
        .arg("miri")
        .arg("--help")
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success(), "miri --help should succeed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should mention key options
    assert!(
        stdout.contains("--project") || stdout.contains("project"),
        "Help should mention project option: {}",
        stdout
    );
}

// ============================================================================
// Path validation tests
// ============================================================================

#[test]
#[serial]
fn test_miri_nonexistent_path() {
    let output = dashprove_cmd()
        .arg("miri")
        .arg("--project")
        .arg("/nonexistent/path/to/project")
        .output()
        .expect("Failed to execute command");

    assert!(
        !output.status.success(),
        "miri should fail for nonexistent path"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.to_lowercase().contains("not")
            || stderr.to_lowercase().contains("exist")
            || stderr.to_lowercase().contains("error"),
        "Error should mention path problem: {}",
        stderr
    );
}

// ============================================================================
// Format option tests
// ============================================================================

#[test]
#[serial]
fn test_miri_json_format_option() {
    // Just test that the format option is recognized
    // Actual MIRI execution requires nightly and miri component
    let output = dashprove_cmd()
        .arg("miri")
        .arg("--help")
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Format option should be documented
    assert!(
        stdout.contains("format") || stdout.contains("json") || stdout.contains("output"),
        "Help should mention format options: {}",
        stdout
    );
}

// ============================================================================
// Flag combination tests
// ============================================================================

#[test]
#[serial]
fn test_miri_flag_combinations() {
    // Test that various flag combinations are recognized (via help)
    let output = dashprove_cmd()
        .arg("miri")
        .arg("--help")
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Various flags should be documented
    let expected_flags = [
        "stacked-borrows",
        "data-race",
        "isolation",
        "timeout",
        "verbose",
    ];

    let found = expected_flags
        .iter()
        .filter(|&flag| stdout.to_lowercase().contains(&flag.replace('-', "")))
        .count();

    // At least some flags should be present
    assert!(found >= 2, "Help should document MIRI flags: {}", stdout);
}

// ============================================================================
// Harness generation tests
// ============================================================================

#[test]
#[serial]
fn test_miri_harness_generation_help() {
    let output = dashprove_cmd()
        .arg("miri")
        .arg("--help")
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Harness generation should be documented
    assert!(
        stdout.contains("harness") || stdout.contains("generate"),
        "Help should mention harness generation: {}",
        stdout
    );
}

#[test]
#[serial]
fn test_miri_harness_with_rust_file() {
    let dir = temp_dir("harness");

    // Create a simple Rust file
    let rust_code = r#"
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

pub fn multiply(x: usize, y: usize) -> usize {
    x * y
}
"#;
    let file_path = dir.join("test_funcs.rs");
    std::fs::write(&file_path, rust_code).unwrap();

    // Note: Actual harness generation may require proper setup
    // This test verifies the command accepts the arguments
    let output = dashprove_cmd()
        .arg("miri")
        .arg("harness")
        .arg("--function")
        .arg("add")
        .arg("--file")
        .arg(&file_path)
        .output()
        .expect("Failed to execute command");

    // Either succeeds or fails with a meaningful error (not argument error)
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should not be an "unrecognized argument" error
    assert!(
        !stderr.contains("unrecognized") && !stderr.contains("invalid option"),
        "Command arguments should be recognized: stderr={} stdout={}",
        stderr,
        stdout
    );

    std::fs::remove_dir_all(&dir).ok();
}

// ============================================================================
// Setup mode tests
// ============================================================================

#[test]
#[serial]
fn test_miri_setup_option() {
    let output = dashprove_cmd()
        .arg("miri")
        .arg("--help")
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Setup option should be documented
    assert!(
        stdout.contains("setup") || stdout.contains("install"),
        "Help should mention setup option: {}",
        stdout
    );
}

// ============================================================================
// Output file tests
// ============================================================================

#[test]
#[serial]
fn test_miri_output_option() {
    let output = dashprove_cmd()
        .arg("miri")
        .arg("--help")
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Output file option should be documented
    assert!(
        stdout.contains("-o") || stdout.contains("--output") || stdout.contains("file"),
        "Help should mention output option: {}",
        stdout
    );
}

// ============================================================================
// Example files tests (if MIRI is available)
// ============================================================================

#[test]
#[serial]
#[ignore] // Ignore by default since MIRI may not be installed
fn test_miri_on_example() {
    // This test requires MIRI to be installed
    let example_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/agent_verification/miri_check");

    if !example_path.exists() {
        return;
    }

    let output = dashprove_cmd()
        .arg("miri")
        .arg("--project")
        .arg(&example_path)
        .arg("--timeout")
        .arg("60")
        .output()
        .expect("Failed to execute command");

    // Just verify it runs without panic
    // May fail if MIRI not installed, which is expected
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success()
            || stderr.contains("not available")
            || stderr.contains("not installed"),
        "miri should either succeed or report MIRI not available: {}",
        stderr
    );
}
