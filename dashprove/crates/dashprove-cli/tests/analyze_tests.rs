//! Integration tests for analyze CLI commands
//!
//! Uses serial_test to prevent SIGKILL from parallel process spawning

use serial_test::serial;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_dir(prefix: &str) -> std::path::PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("dashprove_analyze_test_{prefix}_{ts}"));
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
        "failed_checks": [],
        "playback_test": null,
        "trace": [
            {
                "state_num": 1,
                "action": "Init",
                "variables": {
                    "x": {"Int": {"value": 0, "type_hint": null}},
                    "y": {"Int": {"value": 0, "type_hint": null}},
                    "constant": {"Int": {"value": 42, "type_hint": null}}
                }
            },
            {
                "state_num": 2,
                "action": "Increment",
                "variables": {
                    "x": {"Int": {"value": 1, "type_hint": null}},
                    "y": {"Int": {"value": 0, "type_hint": null}},
                    "constant": {"Int": {"value": 42, "type_hint": null}}
                }
            },
            {
                "state_num": 3,
                "action": "Increment",
                "variables": {
                    "x": {"Int": {"value": 2, "type_hint": null}},
                    "y": {"Int": {"value": 0, "type_hint": null}},
                    "constant": {"Int": {"value": 42, "type_hint": null}}
                }
            },
            {
                "state_num": 4,
                "action": "Increment",
                "variables": {
                    "x": {"Int": {"value": 3, "type_hint": null}},
                    "y": {"Int": {"value": 0, "type_hint": null}},
                    "constant": {"Int": {"value": 42, "type_hint": null}}
                }
            },
            {
                "state_num": 5,
                "action": "Final",
                "variables": {
                    "x": {"Int": {"value": 5, "type_hint": null}},
                    "y": {"Int": {"value": 10, "type_hint": null}},
                    "constant": {"Int": {"value": 42, "type_hint": null}}
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

/// Create a second counterexample JSON file for diff testing
fn create_second_counterexample(dir: &std::path::Path) -> std::path::PathBuf {
    let ce_json = r#"{
        "witness": {
            "x": {"Int": {"value": 7, "type_hint": null}},
            "y": {"Int": {"value": 15, "type_hint": null}}
        },
        "failed_checks": [],
        "playback_test": null,
        "trace": [
            {
                "state_num": 1,
                "action": "Init",
                "variables": {
                    "x": {"Int": {"value": 0, "type_hint": null}},
                    "y": {"Int": {"value": 0, "type_hint": null}},
                    "constant": {"Int": {"value": 42, "type_hint": null}}
                }
            },
            {
                "state_num": 2,
                "action": "Increment",
                "variables": {
                    "x": {"Int": {"value": 2, "type_hint": null}},
                    "y": {"Int": {"value": 0, "type_hint": null}},
                    "constant": {"Int": {"value": 42, "type_hint": null}}
                }
            },
            {
                "state_num": 3,
                "action": "Final",
                "variables": {
                    "x": {"Int": {"value": 7, "type_hint": null}},
                    "y": {"Int": {"value": 15, "type_hint": null}},
                    "constant": {"Int": {"value": 42, "type_hint": null}}
                }
            }
        ],
        "raw": null,
        "minimized": false
    }"#;

    let path = dir.join("counterexample2.json");
    std::fs::write(&path, ce_json).unwrap();
    path
}

#[test]
#[serial]
fn test_analyze_suggest_text() {
    let dir = temp_dir("suggest_text");
    let ce_path = create_sample_counterexample(&dir);

    let output = dashprove_cmd()
        .args(["analyze"])
        .arg(ce_path.to_str().unwrap())
        .args(["suggest", "--format", "text"])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    // Should mention suggestions or indicate no patterns found
    assert!(
        stdout.contains("Pattern Suggestions") || stdout.contains("No patterns"),
        "Should show suggestions or indicate none found: {}",
        stdout
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_analyze_suggest_json() {
    let dir = temp_dir("suggest_json");
    let ce_path = create_sample_counterexample(&dir);

    let output = dashprove_cmd()
        .args(["analyze"])
        .arg(ce_path.to_str().unwrap())
        .args(["suggest", "--format", "json"])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    // Should be valid JSON output (either array or message)
    assert!(
        stdout.contains("[") || stdout.contains("No patterns"),
        "Should output JSON array or indicate no patterns: {}",
        stdout
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_analyze_compress_text() {
    let dir = temp_dir("compress_text");
    let ce_path = create_sample_counterexample(&dir);

    let output = dashprove_cmd()
        .args(["analyze"])
        .arg(ce_path.to_str().unwrap())
        .args(["compress", "--format", "text"])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Compressed Trace"),
        "Should show compressed trace header: {}",
        stdout
    );
    assert!(
        stdout.contains("Original states:"),
        "Should show original state count: {}",
        stdout
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_analyze_compress_mermaid() {
    let dir = temp_dir("compress_mermaid");
    let ce_path = create_sample_counterexample(&dir);

    let output = dashprove_cmd()
        .args(["analyze"])
        .arg(ce_path.to_str().unwrap())
        .args(["compress", "--format", "mermaid"])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    // Should be valid Mermaid diagram
    assert!(
        stdout.contains("flowchart") || stdout.contains("graph"),
        "Should output Mermaid diagram: {}",
        stdout
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_analyze_interleavings_text() {
    let dir = temp_dir("interleave_text");
    let ce_path = create_sample_counterexample(&dir);

    let output = dashprove_cmd()
        .args(["analyze"])
        .arg(ce_path.to_str().unwrap())
        .args(["interleavings", "--format", "text"])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Trace Interleavings"),
        "Should show interleavings header: {}",
        stdout
    );
    assert!(
        stdout.contains("Total states:"),
        "Should show total state count: {}",
        stdout
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_analyze_minimize() {
    let dir = temp_dir("minimize");
    let ce_path = create_sample_counterexample(&dir);
    let output_path = dir.join("minimized.json");

    let output = dashprove_cmd()
        .args(["analyze"])
        .arg(ce_path.to_str().unwrap())
        .args(["minimize", "-o"])
        .arg(output_path.to_str().unwrap())
        .output()
        .expect("Failed to run dashprove");

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output_path.exists(), "Output file should be created");

    // Should be valid JSON
    let content = std::fs::read_to_string(&output_path).unwrap();
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(&content);
    assert!(parsed.is_ok(), "Output should be valid JSON");

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_analyze_abstract_text() {
    let dir = temp_dir("abstract_text");
    let ce_path = create_sample_counterexample(&dir);

    let output = dashprove_cmd()
        .args(["analyze"])
        .arg(ce_path.to_str().unwrap())
        .args(["abstract", "--format", "text"])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Abstracted Trace"),
        "Should show abstracted trace header: {}",
        stdout
    );
    assert!(
        stdout.contains("Original states:"),
        "Should show original state count: {}",
        stdout
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_analyze_diff_text() {
    let dir = temp_dir("diff_text");
    let ce1_path = create_sample_counterexample(&dir);
    let ce2_path = create_second_counterexample(&dir);

    let output = dashprove_cmd()
        .args(["analyze"])
        .arg(ce1_path.to_str().unwrap())
        .args(["diff"])
        .arg(ce2_path.to_str().unwrap())
        .args(["--format", "text"])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Trace Diff"),
        "Should show trace diff header: {}",
        stdout
    );
    assert!(
        stdout.contains("Trace 1 length:") && stdout.contains("Trace 2 length:"),
        "Should show both trace lengths: {}",
        stdout
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_analyze_missing_file() {
    let output = dashprove_cmd()
        .args(["analyze", "/nonexistent/path.json", "suggest"])
        .output()
        .expect("Failed to run dashprove");

    assert!(!output.status.success(), "Should fail with missing file");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("not found") || stderr.contains("Error"),
        "Should indicate file not found: {}",
        stderr
    );
}
