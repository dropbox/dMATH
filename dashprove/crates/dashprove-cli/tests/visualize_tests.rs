//! Integration tests for visualize CLI command

use serial_test::serial;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_dir(prefix: &str) -> std::path::PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("dashprove_visualize_test_{prefix}_{ts}"));
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
                    "y": {"Int": {"value": 0, "type_hint": null}}
                }
            },
            {
                "state_num": 2,
                "action": "Increment",
                "variables": {
                    "x": {"Int": {"value": 1, "type_hint": null}},
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

// ============================================================================
// HTML visualization tests
// ============================================================================

#[test]
#[serial]
fn test_visualize_html_default() {
    let dir = temp_dir("html_default");
    let ce_path = create_sample_counterexample(&dir);

    let output = dashprove_cmd()
        .arg("visualize")
        .arg(&ce_path)
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "visualize (html default) failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // HTML output should contain HTML tags
    assert!(
        stdout.contains("<html") || stdout.contains("<!DOCTYPE") || stdout.contains("<div"),
        "HTML visualization should contain HTML tags: {}",
        stdout
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_visualize_html_explicit() {
    let dir = temp_dir("html_explicit");
    let ce_path = create_sample_counterexample(&dir);

    let output = dashprove_cmd()
        .arg("visualize")
        .arg(&ce_path)
        .arg("--format")
        .arg("html")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "visualize --format html failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("<html") || stdout.contains("<!DOCTYPE") || stdout.contains("<div"),
        "HTML visualization should contain HTML: {}",
        stdout
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_visualize_html_with_title() {
    let dir = temp_dir("html_title");
    let ce_path = create_sample_counterexample(&dir);

    let output = dashprove_cmd()
        .arg("visualize")
        .arg(&ce_path)
        .arg("--format")
        .arg("html")
        .arg("--title")
        .arg("Test Counterexample")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "visualize --title failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Title should appear in output
    assert!(
        stdout.contains("Test Counterexample"),
        "HTML should contain the provided title: {}",
        stdout
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_visualize_html_with_output_file() {
    let dir = temp_dir("html_output");
    let ce_path = create_sample_counterexample(&dir);
    let output_path = dir.join("trace.html");

    let output = dashprove_cmd()
        .arg("visualize")
        .arg(&ce_path)
        .arg("--format")
        .arg("html")
        .arg("-o")
        .arg(&output_path)
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "visualize -o failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(output_path.exists(), "Output file should be created");
    let content = std::fs::read_to_string(&output_path).unwrap();
    assert!(!content.is_empty(), "Output file should not be empty");
    assert!(
        content.contains("<html") || content.contains("<!DOCTYPE"),
        "Output file should contain HTML"
    );

    std::fs::remove_dir_all(&dir).ok();
}

// ============================================================================
// Mermaid visualization tests
// ============================================================================

#[test]
#[serial]
fn test_visualize_mermaid() {
    let dir = temp_dir("mermaid");
    let ce_path = create_sample_counterexample(&dir);

    let output = dashprove_cmd()
        .arg("visualize")
        .arg(&ce_path)
        .arg("--format")
        .arg("mermaid")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "visualize --format mermaid failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Mermaid should contain diagram syntax
    assert!(
        stdout.contains("graph")
            || stdout.contains("flowchart")
            || stdout.contains("stateDiagram")
            || stdout.contains("-->"),
        "Mermaid visualization should contain diagram syntax: {}",
        stdout
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_visualize_mermaid_with_output_file() {
    let dir = temp_dir("mermaid_output");
    let ce_path = create_sample_counterexample(&dir);
    let output_path = dir.join("trace.mmd");

    let output = dashprove_cmd()
        .arg("visualize")
        .arg(&ce_path)
        .arg("--format")
        .arg("mermaid")
        .arg("-o")
        .arg(&output_path)
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "visualize mermaid -o failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(output_path.exists(), "Output file should be created");
    let content = std::fs::read_to_string(&output_path).unwrap();
    assert!(!content.is_empty(), "Output file should not be empty");

    std::fs::remove_dir_all(&dir).ok();
}

// ============================================================================
// DOT visualization tests
// ============================================================================

#[test]
#[serial]
fn test_visualize_dot() {
    let dir = temp_dir("dot");
    let ce_path = create_sample_counterexample(&dir);

    let output = dashprove_cmd()
        .arg("visualize")
        .arg(&ce_path)
        .arg("--format")
        .arg("dot")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "visualize --format dot failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // DOT should contain graph definition
    assert!(
        stdout.contains("digraph") || stdout.contains("graph ") || stdout.contains("->"),
        "DOT visualization should contain graph syntax: {}",
        stdout
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_visualize_dot_with_output_file() {
    let dir = temp_dir("dot_output");
    let ce_path = create_sample_counterexample(&dir);
    let output_path = dir.join("trace.dot");

    let output = dashprove_cmd()
        .arg("visualize")
        .arg(&ce_path)
        .arg("--format")
        .arg("dot")
        .arg("-o")
        .arg(&output_path)
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "visualize dot -o failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(output_path.exists(), "Output file should be created");
    let content = std::fs::read_to_string(&output_path).unwrap();
    assert!(
        content.contains("digraph") || content.contains("->"),
        "Output file should contain DOT graph"
    );

    std::fs::remove_dir_all(&dir).ok();
}

// ============================================================================
// Error handling tests
// ============================================================================

#[test]
#[serial]
fn test_visualize_nonexistent_file() {
    let output = dashprove_cmd()
        .arg("visualize")
        .arg("/nonexistent/path/ce.json")
        .output()
        .expect("Failed to execute command");

    assert!(
        !output.status.success(),
        "visualize should fail for nonexistent file"
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
fn test_visualize_invalid_json() {
    let dir = temp_dir("invalid_json");
    let invalid_path = dir.join("invalid.json");
    std::fs::write(&invalid_path, "{ this is not valid json }").unwrap();

    let output = dashprove_cmd()
        .arg("visualize")
        .arg(&invalid_path)
        .output()
        .expect("Failed to execute command");

    assert!(
        !output.status.success(),
        "visualize should fail for invalid JSON"
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
fn test_visualize_invalid_format() {
    let dir = temp_dir("invalid_format");
    let ce_path = create_sample_counterexample(&dir);

    let output = dashprove_cmd()
        .arg("visualize")
        .arg(&ce_path)
        .arg("--format")
        .arg("invalid_format")
        .output()
        .expect("Failed to execute command");

    assert!(
        !output.status.success(),
        "visualize should fail for invalid format"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.to_lowercase().contains("unknown")
            || stderr.to_lowercase().contains("invalid")
            || stderr.to_lowercase().contains("unsupported")
            || stderr.to_lowercase().contains("error"),
        "Error should mention invalid format: {}",
        stderr
    );

    std::fs::remove_dir_all(&dir).ok();
}

// ============================================================================
// Help test
// ============================================================================

#[test]
#[serial]
fn test_visualize_help() {
    let output = dashprove_cmd()
        .arg("visualize")
        .arg("--help")
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success(), "visualize --help failed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("--format") && stdout.contains("--output"),
        "Help should mention visualize options: {}",
        stdout
    );
}
