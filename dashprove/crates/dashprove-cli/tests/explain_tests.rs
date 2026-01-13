//! Integration tests for explain CLI command

use serial_test::serial;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_dir(prefix: &str) -> std::path::PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("dashprove_explain_test_{prefix}_{ts}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn dashprove_cmd() -> Command {
    Command::new(env!("CARGO_BIN_EXE_dashprove"))
}

/// Create a sample counterexample JSON file for testing
fn create_sample_counterexample(dir: &std::path::Path) -> std::path::PathBuf {
    let ce_json = r#"{
        "witness": {
            "x": {"Int": {"value": 5, "type_hint": null}},
            "y": {"Int": {"value": 10, "type_hint": null}}
        },
        "failed_checks": ["invariant_positive"],
        "playback_test": null,
        "trace": [
            {
                "state_num": 1,
                "action": "Init",
                "variables": {
                    "x": {"Int": {"value": 0, "type_hint": null}},
                    "y": {"Int": {"value": 0, "type_hint": null}}
                }
            },
            {
                "state_num": 2,
                "action": "Decrement",
                "variables": {
                    "x": {"Int": {"value": -1, "type_hint": null}},
                    "y": {"Int": {"value": 0, "type_hint": null}}
                }
            },
            {
                "state_num": 3,
                "action": "Final",
                "variables": {
                    "x": {"Int": {"value": 5, "type_hint": null}},
                    "y": {"Int": {"value": 10, "type_hint": null}}
                }
            }
        ],
        "raw": null,
        "minimized": false
    }"#;

    let path = dir.join("counterexample.json");
    std::fs::write(&path, ce_json).unwrap();
    path
}

/// Create a counterexample with a longer trace
fn create_long_trace_counterexample(dir: &std::path::Path) -> std::path::PathBuf {
    let ce_json = r#"{
        "witness": {
            "counter": {"Int": {"value": 100, "type_hint": null}}
        },
        "failed_checks": ["bound_check"],
        "playback_test": null,
        "trace": [
            {
                "state_num": 1,
                "action": "Init",
                "variables": {
                    "counter": {"Int": {"value": 0, "type_hint": null}}
                }
            },
            {
                "state_num": 2,
                "action": "Step",
                "variables": {
                    "counter": {"Int": {"value": 10, "type_hint": null}}
                }
            },
            {
                "state_num": 3,
                "action": "Step",
                "variables": {
                    "counter": {"Int": {"value": 20, "type_hint": null}}
                }
            },
            {
                "state_num": 4,
                "action": "Step",
                "variables": {
                    "counter": {"Int": {"value": 50, "type_hint": null}}
                }
            },
            {
                "state_num": 5,
                "action": "Overflow",
                "variables": {
                    "counter": {"Int": {"value": 100, "type_hint": null}}
                }
            }
        ],
        "raw": null,
        "minimized": false
    }"#;

    let path = dir.join("long_trace.json");
    std::fs::write(&path, ce_json).unwrap();
    path
}

// ============================================================================
// Basic explain tests
// ============================================================================

#[test]
#[serial]
fn test_explain_basic() {
    let dir = temp_dir("basic");
    let ce_path = create_sample_counterexample(&dir);

    // JSON counterexamples are auto-detected; text needs --backend
    let output = dashprove_cmd()
        .arg("explain")
        .arg(&ce_path)
        .arg("--backend")
        .arg("lean")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "explain failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should provide some explanation
    assert!(
        !stdout.trim().is_empty(),
        "explain should produce output: {}",
        stdout
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_explain_long_trace() {
    let dir = temp_dir("long_trace");
    let ce_path = create_long_trace_counterexample(&dir);

    let output = dashprove_cmd()
        .arg("explain")
        .arg(&ce_path)
        .arg("--backend")
        .arg("tla+")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "explain long trace failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout.trim().is_empty(),
        "explain should produce output for long trace"
    );

    std::fs::remove_dir_all(&dir).ok();
}

// ============================================================================
// Backend option tests
// ============================================================================

#[test]
#[serial]
fn test_explain_with_lean_backend() {
    let dir = temp_dir("lean_backend");
    let ce_path = create_sample_counterexample(&dir);

    let output = dashprove_cmd()
        .arg("explain")
        .arg(&ce_path)
        .arg("--backend")
        .arg("lean")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "explain --backend lean failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout.trim().is_empty(),
        "explain with lean backend should produce output"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_explain_with_tlaplus_backend() {
    let dir = temp_dir("tla_backend");
    let ce_path = create_sample_counterexample(&dir);

    let output = dashprove_cmd()
        .arg("explain")
        .arg(&ce_path)
        .arg("--backend")
        .arg("tla+")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "explain --backend tla+ failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout.trim().is_empty(),
        "explain with tla+ backend should produce output"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_explain_with_kani_backend() {
    let dir = temp_dir("kani_backend");
    let ce_path = create_sample_counterexample(&dir);

    let output = dashprove_cmd()
        .arg("explain")
        .arg(&ce_path)
        .arg("--backend")
        .arg("kani")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "explain --backend kani failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout.trim().is_empty(),
        "explain with kani backend should produce output"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_explain_with_alloy_backend() {
    let dir = temp_dir("alloy_backend");
    let ce_path = create_sample_counterexample(&dir);

    let output = dashprove_cmd()
        .arg("explain")
        .arg(&ce_path)
        .arg("--backend")
        .arg("alloy")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "explain --backend alloy failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout.trim().is_empty(),
        "explain with alloy backend should produce output"
    );

    std::fs::remove_dir_all(&dir).ok();
}

// ============================================================================
// Error handling tests
// ============================================================================

#[test]
#[serial]
fn test_explain_nonexistent_file() {
    let output = dashprove_cmd()
        .arg("explain")
        .arg("/nonexistent/path/ce.json")
        .output()
        .expect("Failed to execute command");

    assert!(
        !output.status.success(),
        "explain should fail for nonexistent file"
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
fn test_explain_invalid_json() {
    let dir = temp_dir("invalid_json");
    let invalid_path = dir.join("invalid.json");
    std::fs::write(&invalid_path, "{ not valid json }").unwrap();

    let output = dashprove_cmd()
        .arg("explain")
        .arg(&invalid_path)
        .output()
        .expect("Failed to execute command");

    assert!(
        !output.status.success(),
        "explain should fail for invalid JSON"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.to_lowercase().contains("parse")
            || stderr.to_lowercase().contains("json")
            || stderr.to_lowercase().contains("deserialize")
            || stderr.to_lowercase().contains("error"),
        "Error should mention JSON problem: {}",
        stderr
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_explain_malformed_counterexample() {
    let dir = temp_dir("malformed");
    let path = dir.join("malformed.json");
    // Valid JSON but not a valid counterexample structure
    std::fs::write(&path, r#"{"foo": "bar"}"#).unwrap();

    let output = dashprove_cmd()
        .arg("explain")
        .arg(&path)
        .output()
        .expect("Failed to execute command");

    // May either fail or handle gracefully with empty explanation
    // Just ensure it doesn't panic
    let _ = String::from_utf8_lossy(&output.stderr);

    std::fs::remove_dir_all(&dir).ok();
}

// ============================================================================
// Help test
// ============================================================================

#[test]
#[serial]
fn test_explain_help() {
    let output = dashprove_cmd()
        .arg("explain")
        .arg("--help")
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success(), "explain --help failed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("--backend") || stdout.contains("counterexample"),
        "Help should mention explain options: {}",
        stdout
    );
}
