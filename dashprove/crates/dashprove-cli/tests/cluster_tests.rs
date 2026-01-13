//! Integration tests for cluster CLI command

use serial_test::serial;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_dir(prefix: &str) -> std::path::PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("dashprove_cluster_test_{prefix}_{ts}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn dashprove_cmd() -> Command {
    Command::new(env!("CARGO_BIN_EXE_dashprove"))
}

/// Create a sample counterexample JSON file for testing
fn create_counterexample_a(dir: &std::path::Path) -> std::path::PathBuf {
    let ce_json = r#"{
        "witness": {
            "x": {"Int": {"value": 5, "type_hint": null}},
            "y": {"Int": {"value": 10, "type_hint": null}}
        },
        "failed_checks": [
            {"check_id": "safety.1", "property": "safety", "description": "x < y violated"}
        ],
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
                "action": "Step",
                "variables": {
                    "x": {"Int": {"value": 5, "type_hint": null}},
                    "y": {"Int": {"value": 10, "type_hint": null}}
                }
            }
        ],
        "raw": null,
        "minimized": false
    }"#;

    let path = dir.join("counterexample_a.json");
    std::fs::write(&path, ce_json).unwrap();
    path
}

/// Create a second counterexample similar to the first (for same cluster)
fn create_counterexample_b(dir: &std::path::Path) -> std::path::PathBuf {
    let ce_json = r#"{
        "witness": {
            "x": {"Int": {"value": 6, "type_hint": null}},
            "y": {"Int": {"value": 12, "type_hint": null}}
        },
        "failed_checks": [
            {"check_id": "safety.1", "property": "safety", "description": "x < y violated"}
        ],
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
                "action": "Step",
                "variables": {
                    "x": {"Int": {"value": 6, "type_hint": null}},
                    "y": {"Int": {"value": 12, "type_hint": null}}
                }
            }
        ],
        "raw": null,
        "minimized": false
    }"#;

    let path = dir.join("counterexample_b.json");
    std::fs::write(&path, ce_json).unwrap();
    path
}

/// Create a third counterexample different from the first two (different cluster)
fn create_counterexample_c(dir: &std::path::Path) -> std::path::PathBuf {
    let ce_json = r#"{
        "witness": {
            "a": {"Int": {"value": 100, "type_hint": null}}
        },
        "failed_checks": [
            {"check_id": "liveness.1", "property": "liveness", "description": "deadlock detected"}
        ],
        "playback_test": null,
        "trace": [
            {
                "state_num": 1,
                "action": "Start",
                "variables": {
                    "a": {"Int": {"value": 0, "type_hint": null}},
                    "locked": {"Bool": false}
                }
            },
            {
                "state_num": 2,
                "action": "Lock",
                "variables": {
                    "a": {"Int": {"value": 50, "type_hint": null}},
                    "locked": {"Bool": true}
                }
            },
            {
                "state_num": 3,
                "action": "Deadlock",
                "variables": {
                    "a": {"Int": {"value": 100, "type_hint": null}},
                    "locked": {"Bool": true}
                }
            }
        ],
        "raw": null,
        "minimized": false
    }"#;

    let path = dir.join("counterexample_c.json");
    std::fs::write(&path, ce_json).unwrap();
    path
}

#[test]
#[serial]
fn test_cluster_text_output() {
    let dir = temp_dir("cluster_text");
    let ce_a = create_counterexample_a(&dir);
    let ce_b = create_counterexample_b(&dir);

    let output = dashprove_cmd()
        .arg("cluster")
        .arg(ce_a.to_str().unwrap())
        .arg(ce_b.to_str().unwrap())
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
        stdout.contains("Clustered") && stdout.contains("clusters"),
        "Should show clustering summary: {}",
        stdout
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_cluster_json_output() {
    let dir = temp_dir("cluster_json");
    let ce_a = create_counterexample_a(&dir);
    let ce_b = create_counterexample_b(&dir);

    let output = dashprove_cmd()
        .arg("cluster")
        .arg(ce_a.to_str().unwrap())
        .arg(ce_b.to_str().unwrap())
        .args(["--format", "json"])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    // Should be valid JSON with expected fields
    assert!(
        stdout.contains("total_counterexamples")
            && stdout.contains("num_clusters")
            && stdout.contains("clusters"),
        "Should contain JSON fields: {}",
        stdout
    );

    // Verify JSON parses correctly
    let json: serde_json::Value = serde_json::from_str(&stdout).expect("Invalid JSON output");
    assert_eq!(json["total_counterexamples"], 2);
    assert!(json["num_clusters"].as_i64().unwrap() >= 1);

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_cluster_mermaid_output() {
    let dir = temp_dir("cluster_mermaid");
    let ce_a = create_counterexample_a(&dir);
    let ce_b = create_counterexample_b(&dir);

    let output = dashprove_cmd()
        .arg("cluster")
        .arg(ce_a.to_str().unwrap())
        .arg(ce_b.to_str().unwrap())
        .args(["--format", "mermaid"])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    // Mermaid output should contain pie chart syntax
    assert!(
        stdout.contains("pie") || stdout.contains("flowchart"),
        "Should contain Mermaid syntax: {}",
        stdout
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_cluster_flowchart_output() {
    let dir = temp_dir("cluster_flowchart");
    let ce_a = create_counterexample_a(&dir);
    let ce_b = create_counterexample_b(&dir);

    let output = dashprove_cmd()
        .arg("cluster")
        .arg(ce_a.to_str().unwrap())
        .arg(ce_b.to_str().unwrap())
        .args(["--format", "flowchart"])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("flowchart"),
        "Should contain Mermaid flowchart: {}",
        stdout
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_cluster_html_output() {
    let dir = temp_dir("cluster_html");
    let ce_a = create_counterexample_a(&dir);
    let ce_b = create_counterexample_b(&dir);

    let output = dashprove_cmd()
        .arg("cluster")
        .arg(ce_a.to_str().unwrap())
        .arg(ce_b.to_str().unwrap())
        .args(["--format", "html", "--title", "Test Clusters"])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("<!DOCTYPE html") && stdout.contains("Test Clusters"),
        "Should contain HTML with title: {}",
        stdout
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_cluster_file_output() {
    let dir = temp_dir("cluster_file");
    let ce_a = create_counterexample_a(&dir);
    let ce_b = create_counterexample_b(&dir);
    let output_file = dir.join("clusters.html");

    let output = dashprove_cmd()
        .arg("cluster")
        .arg(ce_a.to_str().unwrap())
        .arg(ce_b.to_str().unwrap())
        .args(["--format", "html", "-o"])
        .arg(output_file.to_str().unwrap())
        .output()
        .expect("Failed to run dashprove");

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Check file was created
    assert!(output_file.exists(), "Output file should exist");
    let content = std::fs::read_to_string(&output_file).unwrap();
    assert!(
        content.contains("<!DOCTYPE html"),
        "File should contain HTML"
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_cluster_threshold() {
    let dir = temp_dir("cluster_threshold");
    let ce_a = create_counterexample_a(&dir);
    let ce_b = create_counterexample_b(&dir);
    let ce_c = create_counterexample_c(&dir);

    // High threshold - should create more clusters
    let output = dashprove_cmd()
        .arg("cluster")
        .arg(ce_a.to_str().unwrap())
        .arg(ce_b.to_str().unwrap())
        .arg(ce_c.to_str().unwrap())
        .args(["--threshold", "0.9", "--format", "json"])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let json: serde_json::Value = serde_json::from_str(&stdout).expect("Invalid JSON");
    assert_eq!(json["total_counterexamples"], 3);
    assert_eq!(json["similarity_threshold"], 0.9);

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_cluster_invalid_threshold() {
    let dir = temp_dir("cluster_invalid_threshold");
    let ce_a = create_counterexample_a(&dir);
    let ce_b = create_counterexample_b(&dir);

    // Invalid threshold > 1.0
    let output = dashprove_cmd()
        .arg("cluster")
        .arg(ce_a.to_str().unwrap())
        .arg(ce_b.to_str().unwrap())
        .args(["--threshold", "1.5"])
        .output()
        .expect("Failed to run dashprove");

    assert!(
        !output.status.success(),
        "Should fail with invalid threshold"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Threshold") || stderr.contains("threshold"),
        "Should mention threshold error: {}",
        stderr
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_cluster_missing_file() {
    let dir = temp_dir("cluster_missing");
    let ce_a = create_counterexample_a(&dir);
    let missing = dir.join("does_not_exist.json");

    let output = dashprove_cmd()
        .arg("cluster")
        .arg(ce_a.to_str().unwrap())
        .arg(missing.to_str().unwrap())
        .output()
        .expect("Failed to run dashprove");

    assert!(!output.status.success(), "Should fail with missing file");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("not found") || stderr.contains("does_not_exist"),
        "Should mention missing file: {}",
        stderr
    );

    std::fs::remove_dir_all(dir).ok();
}
