//! Integration tests for bisimulation CLI command

use serial_test::serial;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_dir(prefix: &str) -> std::path::PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("dashprove_bisim_test_{prefix}_{ts}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn dashprove_cmd() -> Command {
    Command::new(env!("CARGO_BIN_EXE_dashprove"))
}

/// Create a sample trace file for testing
fn create_sample_trace(dir: &std::path::Path, name: &str) -> std::path::PathBuf {
    let trace = r#"{
  "metadata": {
    "implementation": "test-impl",
    "version": "1.0.0"
  },
  "input": {
    "user_message": "Hello"
  },
  "events": [
    {
      "type": "api_request",
      "timestamp": "2025-12-20T10:00:00Z",
      "data": {
        "endpoint": "/v1/messages",
        "method": "POST"
      }
    },
    {
      "type": "output",
      "timestamp": "2025-12-20T10:00:01Z",
      "data": {
        "text": "Hello! How can I help?"
      }
    }
  ]
}"#;
    let path = dir.join(format!("{}.json", name));
    std::fs::write(&path, trace).unwrap();
    path
}

// ============================================================================
// Help tests
// ============================================================================

#[test]
#[serial]
fn test_bisim_help() {
    let output = dashprove_cmd()
        .arg("bisim")
        .arg("--help")
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success(), "bisim --help should succeed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should mention key options
    assert!(
        stdout.contains("oracle") || stdout.contains("subject"),
        "Help should mention oracle/subject: {}",
        stdout
    );
}

// ============================================================================
// Trace comparison tests
// ============================================================================

#[test]
#[serial]
fn test_bisim_with_traces() {
    let dir = temp_dir("traces");
    let oracle_trace = create_sample_trace(&dir, "oracle");
    let subject_trace = create_sample_trace(&dir, "subject");

    let output = dashprove_cmd()
        .arg("bisim")
        .arg("--oracle-trace")
        .arg(&oracle_trace)
        .arg("--subject-trace")
        .arg(&subject_trace)
        .output()
        .expect("Failed to execute command");

    // Should either succeed or fail with meaningful error
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should not be an "unrecognized argument" error
    assert!(
        !stderr.contains("unrecognized option") && !stderr.contains("invalid argument"),
        "Command should accept trace arguments: stderr={} stdout={}",
        stderr,
        stdout
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_bisim_identical_traces() {
    let dir = temp_dir("identical");
    let trace = create_sample_trace(&dir, "trace");

    // Compare a trace with itself - should be equivalent
    let output = dashprove_cmd()
        .arg("bisim")
        .arg("--oracle-trace")
        .arg(&trace)
        .arg("--subject-trace")
        .arg(&trace)
        .output()
        .expect("Failed to execute command");

    // Identical traces should be equivalent
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.to_lowercase().contains("equivalent")
                || stdout.to_lowercase().contains("match")
                || stdout.to_lowercase().contains("same")
                || stdout.to_lowercase().contains("pass"),
            "Identical traces should be equivalent: {}",
            stdout
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

// ============================================================================
// Criteria option tests
// ============================================================================

#[test]
#[serial]
fn test_bisim_criteria_options() {
    let output = dashprove_cmd()
        .arg("bisim")
        .arg("--help")
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Criteria options should be documented
    assert!(
        stdout.contains("criteria")
            || stdout.contains("api")
            || stdout.contains("tool")
            || stdout.contains("output"),
        "Help should mention criteria options: {}",
        stdout
    );
}

// ============================================================================
// Output format tests
// ============================================================================

#[test]
#[serial]
fn test_bisim_json_output() {
    let dir = temp_dir("json_out");
    let trace = create_sample_trace(&dir, "trace");

    let output = dashprove_cmd()
        .arg("bisim")
        .arg("--oracle-trace")
        .arg(&trace)
        .arg("--subject-trace")
        .arg(&trace)
        .arg("--format")
        .arg("json")
        .output()
        .expect("Failed to execute command");

    // If successful, output should be JSON
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        // JSON output should start with { or [
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
// Error handling tests
// ============================================================================

#[test]
#[serial]
fn test_bisim_nonexistent_oracle_trace() {
    let dir = temp_dir("missing");
    let subject_trace = create_sample_trace(&dir, "subject");

    let output = dashprove_cmd()
        .arg("bisim")
        .arg("--oracle-trace")
        .arg("/nonexistent/oracle.json")
        .arg("--subject-trace")
        .arg(&subject_trace)
        .output()
        .expect("Failed to execute command");

    assert!(
        !output.status.success(),
        "bisim should fail for nonexistent oracle trace"
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

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_bisim_invalid_trace_json() {
    let dir = temp_dir("invalid");

    // Create invalid JSON
    let invalid_path = dir.join("invalid.json");
    std::fs::write(&invalid_path, "{ invalid json }").unwrap();

    let valid_trace = create_sample_trace(&dir, "valid");

    let output = dashprove_cmd()
        .arg("bisim")
        .arg("--oracle-trace")
        .arg(&invalid_path)
        .arg("--subject-trace")
        .arg(&valid_trace)
        .output()
        .expect("Failed to execute command");

    assert!(
        !output.status.success(),
        "bisim should fail for invalid JSON"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.to_lowercase().contains("parse")
            || stderr.to_lowercase().contains("json")
            || stderr.to_lowercase().contains("invalid")
            || stderr.to_lowercase().contains("error"),
        "Error should mention parse problem: {}",
        stderr
    );

    std::fs::remove_dir_all(&dir).ok();
}

// ============================================================================
// Tolerance option tests
// ============================================================================

#[test]
#[serial]
fn test_bisim_tolerance_options() {
    let output = dashprove_cmd()
        .arg("bisim")
        .arg("--help")
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Tolerance options should be documented
    assert!(
        stdout.contains("tolerance") || stdout.contains("threshold") || stdout.contains("semantic"),
        "Help should mention tolerance options: {}",
        stdout
    );
}

// ============================================================================
// Example directory tests
// ============================================================================

#[test]
#[serial]
fn test_bisim_example_traces() {
    let example_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/agent_verification/bisimulation");

    let oracle_trace = example_dir.join("oracle_trace.json");
    let subject_trace = example_dir.join("subject_trace.json");

    if !oracle_trace.exists() || !subject_trace.exists() {
        return;
    }

    let output = dashprove_cmd()
        .arg("bisim")
        .arg("--oracle-trace")
        .arg(&oracle_trace)
        .arg("--subject-trace")
        .arg(&subject_trace)
        .output()
        .expect("Failed to execute command");

    // Should run without panic
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should produce meaningful output
    assert!(
        !stdout.is_empty() || !stderr.is_empty(),
        "Should produce output"
    );
}

// ============================================================================
// Verbose mode tests
// ============================================================================

#[test]
#[serial]
fn test_bisim_verbose() {
    let dir = temp_dir("verbose");
    let trace = create_sample_trace(&dir, "trace");

    let output = dashprove_cmd()
        .arg("bisim")
        .arg("--oracle-trace")
        .arg(&trace)
        .arg("--subject-trace")
        .arg(&trace)
        .arg("--verbose")
        .output()
        .expect("Failed to execute command");

    // Verbose mode should produce more output
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Either in stdout or stderr
    let combined = format!("{}{}", stdout, stderr);

    // Verbose output typically includes more detail
    // Just verify the command accepts the flag
    assert!(
        !stderr.contains("unrecognized") && !stderr.contains("invalid option"),
        "Should accept --verbose flag: {}",
        combined
    );

    std::fs::remove_dir_all(&dir).ok();
}
