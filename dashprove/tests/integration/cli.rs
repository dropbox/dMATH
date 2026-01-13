//! CLI integration tests
//!
//! Tests the dashprove CLI binary functionality.

use dashprove_backends::traits::{
    BackendId, CounterexampleValue, FailedCheck, StructuredCounterexample, TraceState,
    VerificationStatus,
};
use dashprove_learning::{LearnableResult, ProofLearningSystem};
use dashprove_usl::ast::{Expr, Invariant, Property};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Generate a unique temp file path to avoid race conditions in parallel tests.
fn unique_temp_file(base_name: &str, extension: &str) -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let thread_id = std::thread::current().id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "dashprove_test_{base_name}_{id}_{thread_id:?}_{nanos}.{extension}"
    ))
}

fn dashprove_cmd() -> Command {
    // First try the environment variable (set when running via cargo test)
    // The binary is named "dashprove" not "dashprove-cli"
    if let Ok(path) = std::env::var("CARGO_BIN_EXE_dashprove") {
        Command::new(path)
    } else {
        // Try to find the binary in common locations
        let candidates = [
            "./target/debug/dashprove",
            "./target/release/dashprove",
            "dashprove",
        ];

        for candidate in candidates {
            if std::path::Path::new(candidate).exists() {
                return Command::new(candidate);
            }
        }

        // Default: assume it's built and in target/debug
        Command::new("./target/debug/dashprove")
    }
}

fn create_learning_dir_with_data() -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let thread_id = std::thread::current().id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let base_dir =
        std::env::temp_dir().join(format!("dashprove_cli_history_{id}_{thread_id:?}_{nanos}"));
    std::fs::create_dir_all(&base_dir).expect("create temp dir");

    let mut system = ProofLearningSystem::new();

    // Proof entry
    let proof = LearnableResult {
        property: Property::Invariant(Invariant {
            name: "hist_proof".to_string(),
            body: Expr::Bool(true),
        }),
        backend: BackendId::Lean4,
        status: VerificationStatus::Proven,
        tactics: vec!["simp".to_string()],
        time_taken: Duration::from_millis(5),
        proof_output: None,
    };
    system.record(&proof);

    // Counterexample entry
    let mut cx = StructuredCounterexample::new();
    cx.witness.insert(
        "n".to_string(),
        CounterexampleValue::Int {
            value: 1,
            type_hint: None,
        },
    );
    cx.failed_checks.push(FailedCheck {
        check_id: "c1".to_string(),
        description: "hist failure".to_string(),
        location: None,
        function: None,
    });
    let mut state = TraceState::new(1);
    state.action = Some("step".to_string());
    cx.trace.push(state);
    system.record_counterexample("hist_prop", BackendId::Kani, cx, None);

    system.save_to_dir(&base_dir).expect("save learning data");
    base_dir
}

#[test]
fn test_cli_help() {
    let output = dashprove_cmd()
        .arg("--help")
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("dashprove"));
    assert!(stdout.contains("verify"));
    assert!(stdout.contains("export"));
}

// Note: CLI doesn't have --version flag, test removed

#[test]
fn test_cli_verify_help() {
    let output = dashprove_cmd()
        .args(["verify", "--help"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--backends"));
    assert!(stdout.contains("--timeout"));
    assert!(stdout.contains("--learn"));
}

#[test]
fn test_cli_export_help() {
    let output = dashprove_cmd()
        .args(["export", "--help"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--target"));
    assert!(stdout.contains("--output"));
}

#[test]
fn test_cli_backends() {
    let output = dashprove_cmd()
        .arg("backends")
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should list backend names
    assert!(stdout.contains("LEAN") || stdout.contains("lean") || stdout.contains("Lean"));
}

#[test]
fn test_cli_explain_help() {
    let output = dashprove_cmd()
        .args(["explain", "--help"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--backend"));
}

#[test]
fn test_cli_corpus_help() {
    let output = dashprove_cmd()
        .args(["corpus", "--help"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("stats") || stdout.contains("search"));
}

#[test]
fn test_cli_search_help() {
    let output = dashprove_cmd()
        .args(["search", "--help"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("query") || stdout.contains("QUERY"));
}

#[test]
fn test_cli_prove_help() {
    let output = dashprove_cmd()
        .args(["prove", "--help"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--hints"));
}

#[test]
fn test_cli_verify_missing_file() {
    let output = dashprove_cmd()
        .args(["verify", "nonexistent.usl"])
        .output()
        .expect("Failed to execute command");

    // Should fail with non-zero exit code
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("not found") || stderr.contains("No such file") || stderr.contains("error")
    );
}

#[test]
fn test_cli_export_missing_file() {
    let output = dashprove_cmd()
        .args(["export", "nonexistent.usl", "--target", "lean"])
        .output()
        .expect("Failed to execute command");

    // Should fail with non-zero exit code
    assert!(!output.status.success());
}

#[test]
fn test_cli_verify_with_temp_file() {
    use std::io::Write;

    // Create temp file with valid USL
    let temp_file = unique_temp_file("spec", "usl");

    {
        let mut file = std::fs::File::create(&temp_file).expect("create temp file");
        writeln!(file, "theorem test {{ true }}").expect("write to temp file");
    }

    let output = dashprove_cmd()
        .args(["verify", temp_file.to_str().unwrap(), "--skip-health-check"])
        .output()
        .expect("Failed to execute command");

    // Clean up
    std::fs::remove_file(&temp_file).ok();

    // Verification should complete (may fail due to missing backends, but should parse)
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Either succeeds or reports backend unavailable (not a parse error)
    let combined = format!("{}{}", stdout, stderr);
    assert!(
        output.status.success()
            || combined.contains("unavailable")
            || combined.contains("not found")
            || combined.contains("not installed"),
        "Expected success or backend unavailable, got: {}",
        combined
    );
}

#[test]
fn test_cli_export_to_lean() {
    use std::io::Write;

    // Create temp file with valid USL
    let temp_file = unique_temp_file("export_lean", "usl");

    {
        let mut file = std::fs::File::create(&temp_file).expect("create temp file");
        writeln!(
            file,
            "theorem excluded_middle {{ forall x: Bool . x or not x }}"
        )
        .expect("write to temp file");
    }

    let output = dashprove_cmd()
        .args(["export", temp_file.to_str().unwrap(), "--target", "lean"])
        .output()
        .expect("Failed to execute command");

    // Clean up
    std::fs::remove_file(&temp_file).ok();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should output LEAN code
    assert!(stdout.contains("theorem") || stdout.contains("namespace"));
}

#[test]
fn test_cli_export_to_tlaplus() {
    use std::io::Write;

    let temp_file = unique_temp_file("export_tla", "usl");

    {
        let mut file = std::fs::File::create(&temp_file).expect("create temp file");
        writeln!(file, "temporal safety {{ always(eventually(done)) }}")
            .expect("write to temp file");
    }

    let output = dashprove_cmd()
        .args(["export", temp_file.to_str().unwrap(), "--target", "tla+"])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_file(&temp_file).ok();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("MODULE") || stdout.contains("SPECIFICATION"));
}

#[test]
fn test_cli_export_to_kani() {
    use std::io::Write;

    let temp_file = unique_temp_file("export_kani", "usl");

    {
        let mut file = std::fs::File::create(&temp_file).expect("create temp file");
        writeln!(file, "contract add(x: Int, y: Int) -> Int {{ requires {{ x >= 0 }} ensures {{ result >= x }} }}").expect("write to temp file");
    }

    let output = dashprove_cmd()
        .args(["export", temp_file.to_str().unwrap(), "--target", "kani"])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_file(&temp_file).ok();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("kani"));
}

#[test]
fn test_cli_export_to_alloy() {
    use std::io::Write;

    let temp_file = unique_temp_file("export_alloy", "usl");

    {
        let mut file = std::fs::File::create(&temp_file).expect("create temp file");
        writeln!(file, "invariant safety {{ true }}").expect("write to temp file");
    }

    let output = dashprove_cmd()
        .args(["export", temp_file.to_str().unwrap(), "--target", "alloy"])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_file(&temp_file).ok();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Alloy output should have module or sig
    assert!(
        stdout.contains("module")
            || stdout.contains("sig")
            || stdout.contains("fact")
            || stdout.contains("pred")
    );
}

#[test]
fn test_cli_corpus_stats_empty() {
    use std::time::{SystemTime, UNIX_EPOCH};

    // Create empty temp directory for data
    let temp_dir = std::env::temp_dir().join(format!(
        "dashprove_cli_test_{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&temp_dir).expect("create temp dir");

    let output = dashprove_cmd()
        .args(["corpus", "stats", "--data-dir", temp_dir.to_str().unwrap()])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_dir_all(&temp_dir).ok();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should report 0 proofs
    assert!(stdout.contains("0") || stdout.contains("empty") || stdout.contains("Total"));
}

#[test]
fn test_cli_corpus_history_counterexamples() {
    let temp_dir = create_learning_dir_with_data();

    let output = dashprove_cmd()
        .args([
            "corpus",
            "history",
            "--period",
            "day",
            "--format",
            "json",
            "--data-dir",
            temp_dir.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_dir_all(&temp_dir).ok();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("total_count") || stdout.contains("Total counterexamples"),
        "Unexpected output: {}",
        stdout
    );
}

#[test]
fn test_cli_corpus_history_proofs() {
    let temp_dir = create_learning_dir_with_data();

    let output = dashprove_cmd()
        .args([
            "corpus",
            "history",
            "--corpus",
            "proofs",
            "--period",
            "day",
            "--format",
            "text",
            "--data-dir",
            temp_dir.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_dir_all(&temp_dir).ok();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Total proofs"),
        "Expected proof summary, got: {}",
        stdout
    );
}

#[test]
fn test_cli_monitor_help() {
    let output = dashprove_cmd()
        .args(["monitor", "--help"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--target"));
    assert!(stdout.contains("--assertions"));
    assert!(stdout.contains("--logging"));
    assert!(stdout.contains("--metrics"));
}

#[test]
fn test_cli_monitor_missing_file() {
    let output = dashprove_cmd()
        .args(["monitor", "nonexistent.usl"])
        .output()
        .expect("Failed to execute command");

    // Should fail with non-zero exit code
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("not found") || stderr.contains("No such file") || stderr.contains("error")
    );
}

#[test]
fn test_cli_monitor_rust_target() {
    use std::io::Write;

    let temp_file = unique_temp_file("monitor_rust", "usl");

    {
        let mut file = std::fs::File::create(&temp_file).expect("create temp file");
        writeln!(file, "theorem safety {{ forall x: Bool . x or not x }}")
            .expect("write to temp file");
    }

    let output = dashprove_cmd()
        .args(["monitor", temp_file.to_str().unwrap(), "--target", "rust"])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_file(&temp_file).ok();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should output Rust monitor code
    assert!(stdout.contains("pub struct"));
    assert!(stdout.contains("SafetyMonitor"));
    assert!(stdout.contains("check_safety"));
    assert!(stdout.contains("check_all"));
}

#[test]
fn test_cli_monitor_typescript_target() {
    use std::io::Write;

    let temp_file = unique_temp_file("monitor_ts", "usl");

    {
        let mut file = std::fs::File::create(&temp_file).expect("create temp file");
        writeln!(file, "theorem test_prop {{ true }}").expect("write to temp file");
    }

    let output = dashprove_cmd()
        .args([
            "monitor",
            temp_file.to_str().unwrap(),
            "--target",
            "typescript",
        ])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_file(&temp_file).ok();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should output TypeScript monitor code
    assert!(stdout.contains("export class"));
    assert!(stdout.contains("TestPropMonitor"));
    assert!(stdout.contains("checkTestProp"));
    assert!(stdout.contains("checkAll"));
}

#[test]
fn test_cli_monitor_python_target() {
    use std::io::Write;

    let temp_file = unique_temp_file("monitor_py", "usl");

    {
        let mut file = std::fs::File::create(&temp_file).expect("create temp file");
        writeln!(file, "invariant my_invariant {{ true }}").expect("write to temp file");
    }

    let output = dashprove_cmd()
        .args(["monitor", temp_file.to_str().unwrap(), "--target", "python"])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_file(&temp_file).ok();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should output Python monitor code
    assert!(stdout.contains("class MyInvariantMonitor"));
    assert!(stdout.contains("def check_my_invariant"));
    assert!(stdout.contains("def check_all"));
}

#[test]
fn test_cli_monitor_with_assertions() {
    use std::io::Write;

    let temp_file = unique_temp_file("monitor_assert", "usl");

    {
        let mut file = std::fs::File::create(&temp_file).expect("create temp file");
        writeln!(file, "theorem test {{ true }}").expect("write to temp file");
    }

    let output = dashprove_cmd()
        .args(["monitor", temp_file.to_str().unwrap(), "--assertions"])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_file(&temp_file).ok();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should mention assertions
    assert!(stdout.contains("Assertions: enabled"));
}

#[test]
fn test_cli_monitor_with_logging() {
    use std::io::Write;

    let temp_file = unique_temp_file("monitor_log", "usl");

    {
        let mut file = std::fs::File::create(&temp_file).expect("create temp file");
        writeln!(file, "theorem test {{ true }}").expect("write to temp file");
    }

    let output = dashprove_cmd()
        .args(["monitor", temp_file.to_str().unwrap(), "--logging"])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_file(&temp_file).ok();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should mention logging and include tracing
    assert!(stdout.contains("Logging: enabled"));
    assert!(stdout.contains("tracing"));
}

#[test]
fn test_cli_monitor_multiple_properties() {
    use std::io::Write;

    let temp_file = unique_temp_file("monitor_multi", "usl");

    {
        let mut file = std::fs::File::create(&temp_file).expect("create temp file");
        writeln!(
            file,
            r#"
theorem prop1 {{ true }}
theorem prop2 {{ true }}
invariant inv1 {{ true }}
"#
        )
        .expect("write to temp file");
    }

    let output = dashprove_cmd()
        .args(["monitor", temp_file.to_str().unwrap()])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_file(&temp_file).ok();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should report 3 properties and generate checks for all
    assert!(stdout.contains("3 properties"));
    assert!(stdout.contains("check_prop1"));
    assert!(stdout.contains("check_prop2"));
    assert!(stdout.contains("check_inv1"));
}

#[test]
fn test_cli_monitor_contract_support() {
    use std::io::Write;

    let temp_file = unique_temp_file("monitor_contract", "usl");

    {
        let mut file = std::fs::File::create(&temp_file).expect("create temp file");
        writeln!(
            file,
            r#"
contract divide(x: Int, y: Int) -> Result<Int> {{
    requires {{ y != 0 }}
    ensures {{ result * y == x }}
    ensures_err {{ y == 0 }}
}}
"#
        )
        .expect("write to temp file");
    }

    let output = dashprove_cmd()
        .args(["monitor", temp_file.to_str().unwrap(), "--target", "rust"])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_file(&temp_file).ok();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("check_divide_requires"));
    assert!(stdout.contains("check_divide_ensures"));
    assert!(stdout.contains("check_divide_ensures_err"));
}

#[test]
fn test_cli_monitor_output_to_file() {
    use std::io::Write;

    let temp_file = unique_temp_file("monitor_file", "usl");
    let output_file = unique_temp_file("monitor_output", "rs");

    {
        let mut file = std::fs::File::create(&temp_file).expect("create temp file");
        writeln!(file, "theorem test {{ true }}").expect("write to temp file");
    }

    let output = dashprove_cmd()
        .args([
            "monitor",
            temp_file.to_str().unwrap(),
            "--output",
            output_file.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());

    // Check output file was created
    assert!(output_file.exists(), "Output file should be created");

    let content = std::fs::read_to_string(&output_file).expect("read output file");
    assert!(content.contains("TestMonitor"));

    // Clean up
    std::fs::remove_file(&temp_file).ok();
    std::fs::remove_file(&output_file).ok();
}

#[test]
fn test_cli_monitor_invalid_target() {
    use std::io::Write;

    let temp_file = unique_temp_file("monitor_invalid", "usl");

    {
        let mut file = std::fs::File::create(&temp_file).expect("create temp file");
        writeln!(file, "theorem test {{ true }}").expect("write to temp file");
    }

    let output = dashprove_cmd()
        .args([
            "monitor",
            temp_file.to_str().unwrap(),
            "--target",
            "invalid",
        ])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_file(&temp_file).ok();

    // Should fail with non-zero exit code
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Unknown target"));
}

#[test]
fn test_cli_visualize_help() {
    let output = dashprove_cmd()
        .args(["visualize", "--help"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--format"));
    assert!(stdout.contains("--output"));
    assert!(stdout.contains("--title"));
}

#[test]
fn test_cli_visualize_missing_file() {
    let output = dashprove_cmd()
        .args(["visualize", "nonexistent.json"])
        .output()
        .expect("Failed to execute command");

    // Should fail with non-zero exit code
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("not found") || stderr.contains("No such file") || stderr.contains("error")
    );
}

#[test]
fn test_cli_visualize_html_format() {
    use std::io::Write;

    let temp_file = unique_temp_file("counterexample_html", "json");

    // Create a valid counterexample JSON
    {
        let mut file = std::fs::File::create(&temp_file).expect("create temp file");
        writeln!(
            file,
            r#"{{
    "witness": {{"x": {{"Int": {{"value": 42, "type_hint": null}}}}}},
    "failed_checks": [{{"check_id": "check.1", "description": "test check", "location": null, "function": null}}],
    "playback_test": null,
    "trace": [],
    "raw": null,
    "minimized": false
}}"#
        )
        .expect("write to temp file");
    }

    let output = dashprove_cmd()
        .args(["visualize", temp_file.to_str().unwrap(), "--format", "html"])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_file(&temp_file).ok();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should output HTML
    assert!(stdout.contains("<!DOCTYPE html>"));
    assert!(stdout.contains("mermaid"));
}

#[test]
fn test_cli_visualize_mermaid_format() {
    use std::io::Write;

    let temp_file = unique_temp_file("counterexample_mmd", "json");

    // Create a valid counterexample JSON
    {
        let mut file = std::fs::File::create(&temp_file).expect("create temp file");
        writeln!(
            file,
            r#"{{
    "witness": {{"y": {{"Bool": true}}}},
    "failed_checks": [],
    "playback_test": null,
    "trace": [],
    "raw": null,
    "minimized": false
}}"#
        )
        .expect("write to temp file");
    }

    let output = dashprove_cmd()
        .args([
            "visualize",
            temp_file.to_str().unwrap(),
            "--format",
            "mermaid",
        ])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_file(&temp_file).ok();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should output Mermaid flowchart
    assert!(stdout.contains("flowchart") || stdout.contains("graph"));
}

#[test]
fn test_cli_visualize_dot_format() {
    use std::io::Write;

    let temp_file = unique_temp_file("counterexample_dot", "json");

    // Create a valid counterexample JSON
    {
        let mut file = std::fs::File::create(&temp_file).expect("create temp file");
        writeln!(
            file,
            r#"{{
    "witness": {{"z": {{"String": "test"}}}},
    "failed_checks": [{{"check_id": "assert.1", "description": "assertion failed", "location": null, "function": null}}],
    "playback_test": null,
    "trace": [],
    "raw": null,
    "minimized": false
}}"#
        )
        .expect("write to temp file");
    }

    let output = dashprove_cmd()
        .args(["visualize", temp_file.to_str().unwrap(), "--format", "dot"])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_file(&temp_file).ok();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should output DOT graph
    assert!(stdout.contains("digraph"));
}

#[test]
fn test_cli_visualize_with_title() {
    use std::io::Write;

    let temp_file = unique_temp_file("counterexample_title", "json");

    // Create a valid counterexample JSON
    {
        let mut file = std::fs::File::create(&temp_file).expect("create temp file");
        writeln!(
            file,
            r#"{{
    "witness": {{}},
    "failed_checks": [],
    "playback_test": null,
    "trace": [],
    "raw": null,
    "minimized": false
}}"#
        )
        .expect("write to temp file");
    }

    let output = dashprove_cmd()
        .args([
            "visualize",
            temp_file.to_str().unwrap(),
            "--format",
            "html",
            "--title",
            "My Custom Title",
        ])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_file(&temp_file).ok();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should include the custom title
    assert!(stdout.contains("My Custom Title"));
}

#[test]
fn test_cli_visualize_output_to_file() {
    use std::io::Write;

    let temp_file = unique_temp_file("counterexample_out", "json");
    let output_file = unique_temp_file("visualization", "html");

    // Create a valid counterexample JSON
    {
        let mut file = std::fs::File::create(&temp_file).expect("create temp file");
        writeln!(
            file,
            r#"{{
    "witness": {{"x": {{"Int": {{"value": 1, "type_hint": null}}}}}},
    "failed_checks": [],
    "playback_test": null,
    "trace": [],
    "raw": null,
    "minimized": false
}}"#
        )
        .expect("write to temp file");
    }

    let output = dashprove_cmd()
        .args([
            "visualize",
            temp_file.to_str().unwrap(),
            "--output",
            output_file.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());

    // Check output file was created
    assert!(output_file.exists(), "Output file should be created");

    let content = std::fs::read_to_string(&output_file).expect("read output file");
    assert!(content.contains("<!DOCTYPE html>"));

    // Clean up
    std::fs::remove_file(&temp_file).ok();
    std::fs::remove_file(&output_file).ok();
}

#[test]
fn test_cli_visualize_invalid_format() {
    use std::io::Write;

    let temp_file = unique_temp_file("counterexample_invalid_fmt", "json");

    // Create a valid counterexample JSON
    {
        let mut file = std::fs::File::create(&temp_file).expect("create temp file");
        writeln!(
            file,
            r#"{{
    "witness": {{}},
    "failed_checks": [],
    "playback_test": null,
    "trace": [],
    "raw": null,
    "minimized": false
}}"#
        )
        .expect("write to temp file");
    }

    let output = dashprove_cmd()
        .args([
            "visualize",
            temp_file.to_str().unwrap(),
            "--format",
            "invalid",
        ])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_file(&temp_file).ok();

    // Should fail with non-zero exit code
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Unknown format"));
}

#[test]
fn test_cli_visualize_invalid_json() {
    use std::io::Write;

    let temp_file = unique_temp_file("counterexample_badjson", "json");

    // Create an invalid JSON file
    {
        let mut file = std::fs::File::create(&temp_file).expect("create temp file");
        writeln!(file, "not valid json").expect("write to temp file");
    }

    let output = dashprove_cmd()
        .args(["visualize", temp_file.to_str().unwrap()])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_file(&temp_file).ok();

    // Should fail with non-zero exit code
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Failed to parse") || stderr.contains("JSON"));
}

#[test]
fn test_cli_visualize_html_has_download_buttons() {
    use std::io::Write;

    let temp_file = unique_temp_file("counterexample_dl", "json");

    // Create a valid counterexample JSON
    {
        let mut file = std::fs::File::create(&temp_file).expect("create temp file");
        writeln!(
            file,
            r#"{{
    "witness": {{}},
    "failed_checks": [],
    "playback_test": null,
    "trace": [],
    "raw": null,
    "minimized": false
}}"#
        )
        .expect("write to temp file");
    }

    let output = dashprove_cmd()
        .args(["visualize", temp_file.to_str().unwrap(), "--format", "html"])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_file(&temp_file).ok();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should have download buttons
    assert!(stdout.contains("Download Mermaid"));
    assert!(stdout.contains("Download DOT"));
    assert!(stdout.contains("download-buttons"));
}

// =============================================================================
// Topics command tests
// =============================================================================

#[test]
fn test_cli_topics_help() {
    let output = dashprove_cmd()
        .args(["topics", "--help"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("topics") || stdout.contains("help"));
}

#[test]
fn test_cli_topics_overview() {
    let output = dashprove_cmd()
        .args(["topics"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should show available topics
    assert!(stdout.contains("usl") || stdout.contains("USL"));
    assert!(stdout.contains("backends"));
}

#[test]
fn test_cli_topics_usl() {
    let output = dashprove_cmd()
        .args(["topics", "usl"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should contain USL documentation
    assert!(
        stdout.contains("Unified Specification Language")
            || stdout.contains("USL")
            || stdout.contains("specification")
    );
}

#[test]
fn test_cli_topics_backends() {
    let output = dashprove_cmd()
        .args(["topics", "backends"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should contain backend information
    assert!(
        stdout.contains("LEAN")
            || stdout.contains("TLA")
            || stdout.contains("verification backend")
    );
}

#[test]
fn test_cli_topics_counterexamples() {
    let output = dashprove_cmd()
        .args(["topics", "counterexamples"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should contain counterexample documentation
    assert!(stdout.contains("counterexample") || stdout.contains("Counterexample"));
}

#[test]
fn test_cli_topics_learning() {
    let output = dashprove_cmd()
        .args(["topics", "learning"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should contain learning system documentation
    assert!(
        stdout.contains("learning") || stdout.contains("corpus") || stdout.contains("Learning")
    );
}

#[test]
fn test_cli_topics_properties() {
    let output = dashprove_cmd()
        .args(["topics", "properties"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should contain property type documentation
    assert!(
        stdout.contains("theorem")
            || stdout.contains("invariant")
            || stdout.contains("contract")
            || stdout.contains("Properties")
    );
}

#[test]
fn test_cli_topics_unknown() {
    let output = dashprove_cmd()
        .args(["topics", "nonexistent_topic"])
        .output()
        .expect("Failed to execute command");

    // Unknown topic should fail
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Unknown topic") || stderr.contains("unknown"));
}

// =============================================================================
// Analyze command tests
// =============================================================================

#[test]
fn test_cli_analyze_help() {
    let output = dashprove_cmd()
        .args(["analyze", "--help"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("analyze") || stdout.contains("Analyze"));
}

#[test]
fn test_cli_analyze_missing_file() {
    let output = dashprove_cmd()
        .args(["analyze", "nonexistent.json", "suggest"])
        .output()
        .expect("Failed to execute command");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("not found") || stderr.contains("No such file"));
}

#[test]
fn test_cli_analyze_suggest() {
    let temp_file = unique_temp_file("cli_analyze_suggest", "json");
    {
        let content = r#"{
    "property": "test_property",
    "witness": {"x": {"Int": {"value": 5}}},
    "failed_checks": [{"check_id": "c1", "description": "assertion failed"}],
    "trace": [
        {"state_num": 0, "variables": {"x": {"Int": {"value": 0}}}},
        {"state_num": 1, "variables": {"x": {"Int": {"value": 1}}}},
        {"state_num": 2, "variables": {"x": {"Int": {"value": 2}}}}
    ],
    "raw": null,
    "minimized": false
}"#;
        std::fs::write(&temp_file, content).expect("write to temp file");
    }

    let output = dashprove_cmd()
        .args(["analyze", temp_file.to_str().unwrap(), "suggest"])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_file(&temp_file).ok();

    assert!(output.status.success());
    // Output should either show suggestions or indicate no patterns found
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("suggestion")
            || stdout.contains("Suggestion")
            || stdout.contains("pattern")
            || stdout.contains("No patterns")
    );
}

#[test]
fn test_cli_analyze_suggest_json_format() {
    let temp_file = unique_temp_file("cli_analyze_suggest_json", "json");
    {
        let content = r#"{
    "property": "test_property",
    "witness": {"x": {"Int": {"value": 5}}},
    "failed_checks": [{"check_id": "c1", "description": "assertion failed"}],
    "trace": [
        {"state_num": 0, "variables": {"x": {"Int": {"value": 0}}}},
        {"state_num": 1, "variables": {"x": {"Int": {"value": 1}}}}
    ],
    "raw": null,
    "minimized": false
}"#;
        std::fs::write(&temp_file, content).expect("write to temp file");
    }

    let output = dashprove_cmd()
        .args([
            "analyze",
            temp_file.to_str().unwrap(),
            "suggest",
            "--format",
            "json",
        ])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_file(&temp_file).ok();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // JSON output should be valid JSON (either [] or array of suggestions)
    assert!(stdout.starts_with('[') || stdout.contains("No patterns"));
}

#[test]
fn test_cli_analyze_compress() {
    let temp_file = unique_temp_file("cli_analyze_compress", "json");
    {
        let content = r#"{
    "property": "test_property",
    "witness": {"x": {"Int": {"value": 5}}},
    "failed_checks": [],
    "trace": [
        {"state_num": 0, "variables": {"x": {"Int": {"value": 0}}}},
        {"state_num": 1, "variables": {"x": {"Int": {"value": 0}}}},
        {"state_num": 2, "variables": {"x": {"Int": {"value": 1}}}},
        {"state_num": 3, "variables": {"x": {"Int": {"value": 1}}}},
        {"state_num": 4, "variables": {"x": {"Int": {"value": 2}}}}
    ],
    "raw": null,
    "minimized": false
}"#;
        std::fs::write(&temp_file, content).expect("write to temp file");
    }

    let output = dashprove_cmd()
        .args(["analyze", temp_file.to_str().unwrap(), "compress"])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_file(&temp_file).ok();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should output compression info
    assert!(
        stdout.contains("compress")
            || stdout.contains("segment")
            || stdout.contains("Segment")
            || stdout.contains("states")
    );
}

#[test]
fn test_cli_analyze_compress_json_output() {
    let temp_file = unique_temp_file("cli_analyze_compress_json", "json");
    let output_file = unique_temp_file("cli_analyze_compress_out", "json");
    {
        let content = r#"{
    "property": "test_property",
    "witness": {"x": {"Int": {"value": 5}}},
    "failed_checks": [],
    "trace": [
        {"state_num": 0, "variables": {"x": {"Int": {"value": 0}}}},
        {"state_num": 1, "variables": {"x": {"Int": {"value": 1}}}}
    ],
    "raw": null,
    "minimized": false
}"#;
        std::fs::write(&temp_file, content).expect("write to temp file");
    }

    let output = dashprove_cmd()
        .args([
            "analyze",
            temp_file.to_str().unwrap(),
            "compress",
            "--format",
            "json",
            "--output",
            output_file.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute command");

    let output_exists = output_file.exists();
    std::fs::remove_file(&temp_file).ok();
    std::fs::remove_file(&output_file).ok();

    assert!(output.status.success());
    assert!(output_exists, "Output file should be created");
}

#[test]
fn test_cli_analyze_minimize() {
    let temp_file = unique_temp_file("cli_analyze_minimize", "json");
    {
        let content = r#"{
    "property": "test_property",
    "witness": {"x": {"Int": {"value": 5}}},
    "failed_checks": [{"check_id": "c1", "description": "fail at end"}],
    "trace": [
        {"state_num": 0, "variables": {"x": {"Int": {"value": 0}}}},
        {"state_num": 1, "variables": {"x": {"Int": {"value": 1}}}},
        {"state_num": 2, "variables": {"x": {"Int": {"value": 2}}}},
        {"state_num": 3, "variables": {"x": {"Int": {"value": 3}}}},
        {"state_num": 4, "variables": {"x": {"Int": {"value": 4}}}}
    ],
    "raw": null,
    "minimized": false
}"#;
        std::fs::write(&temp_file, content).expect("write to temp file");
    }

    let output = dashprove_cmd()
        .args([
            "analyze",
            temp_file.to_str().unwrap(),
            "minimize",
            "--max-states",
            "3",
        ])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_file(&temp_file).ok();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should output minimization info
    assert!(
        stdout.contains("minim")
            || stdout.contains("state")
            || stdout.contains("reduced")
            || stdout.contains("{")
    );
}

#[test]
fn test_cli_analyze_abstract() {
    let temp_file = unique_temp_file("cli_analyze_abstract", "json");
    {
        // Create a trace with repeating patterns for abstraction
        let content = r#"{
    "property": "test_property",
    "witness": {"counter": {"Int": {"value": 10}}},
    "failed_checks": [],
    "trace": [
        {"state_num": 0, "action": "init", "variables": {"counter": {"Int": {"value": 0}}}},
        {"state_num": 1, "action": "increment", "variables": {"counter": {"Int": {"value": 1}}}},
        {"state_num": 2, "action": "increment", "variables": {"counter": {"Int": {"value": 2}}}},
        {"state_num": 3, "action": "increment", "variables": {"counter": {"Int": {"value": 3}}}},
        {"state_num": 4, "action": "done", "variables": {"counter": {"Int": {"value": 3}}}}
    ],
    "raw": null,
    "minimized": false
}"#;
        std::fs::write(&temp_file, content).expect("write to temp file");
    }

    let output = dashprove_cmd()
        .args([
            "analyze",
            temp_file.to_str().unwrap(),
            "abstract",
            "--min-group-size",
            "2",
        ])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_file(&temp_file).ok();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should output abstraction info
    assert!(
        stdout.contains("abstract")
            || stdout.contains("segment")
            || stdout.contains("group")
            || stdout.contains("Segment")
    );
}

#[test]
fn test_cli_analyze_diff() {
    let temp_file1 = unique_temp_file("cli_analyze_diff1", "json");
    let temp_file2 = unique_temp_file("cli_analyze_diff2", "json");
    {
        let content1 = r#"{
    "property": "test_property",
    "witness": {"x": {"Int": {"value": 5}}},
    "failed_checks": [],
    "trace": [
        {"state_num": 0, "variables": {"x": {"Int": {"value": 0}}}},
        {"state_num": 1, "variables": {"x": {"Int": {"value": 1}}}},
        {"state_num": 2, "variables": {"x": {"Int": {"value": 2}}}}
    ],
    "raw": null,
    "minimized": false
}"#;
        let content2 = r#"{
    "property": "test_property",
    "witness": {"x": {"Int": {"value": 10}}},
    "failed_checks": [],
    "trace": [
        {"state_num": 0, "variables": {"x": {"Int": {"value": 0}}}},
        {"state_num": 1, "variables": {"x": {"Int": {"value": 2}}}},
        {"state_num": 2, "variables": {"x": {"Int": {"value": 4}}}}
    ],
    "raw": null,
    "minimized": false
}"#;
        std::fs::write(&temp_file1, content1).expect("write to temp file");
        std::fs::write(&temp_file2, content2).expect("write to temp file");
    }

    // Note: diff takes a positional argument, not --other
    let output = dashprove_cmd()
        .args([
            "analyze",
            temp_file1.to_str().unwrap(),
            "diff",
            temp_file2.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_file(&temp_file1).ok();
    std::fs::remove_file(&temp_file2).ok();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should output diff info
    assert!(
        stdout.contains("diff")
            || stdout.contains("Diff")
            || stdout.contains("differ")
            || stdout.contains("state")
    );
}

#[test]
fn test_cli_analyze_interleavings() {
    let temp_file = unique_temp_file("cli_analyze_interleavings", "json");
    {
        let content = r#"{
    "property": "test_property",
    "witness": {"x": {"Int": {"value": 5}}},
    "failed_checks": [],
    "trace": [
        {"state_num": 0, "action": "a", "variables": {"x": {"Int": {"value": 0}}}},
        {"state_num": 1, "action": "b", "variables": {"x": {"Int": {"value": 1}}}},
        {"state_num": 2, "action": "a", "variables": {"x": {"Int": {"value": 2}}}}
    ],
    "raw": null,
    "minimized": false
}"#;
        std::fs::write(&temp_file, content).expect("write to temp file");
    }

    let output = dashprove_cmd()
        .args(["analyze", temp_file.to_str().unwrap(), "interleavings"])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_file(&temp_file).ok();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should output interleaving info
    assert!(
        stdout.contains("interleav")
            || stdout.contains("Interleav")
            || stdout.contains("sequence")
            || stdout.contains("[")
    );
}

#[test]
fn test_cli_analyze_invalid_json() {
    let temp_file = unique_temp_file("cli_analyze_invalid", "json");
    {
        std::fs::write(&temp_file, "{ invalid json }").expect("write to temp file");
    }

    let output = dashprove_cmd()
        .args(["analyze", temp_file.to_str().unwrap(), "suggest"])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_file(&temp_file).ok();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("parse")
            || stderr.contains("Parse")
            || stderr.contains("JSON")
            || stderr.contains("invalid")
    );
}

// =============================================================================
// Train command tests
// =============================================================================

#[test]
fn test_cli_train_help() {
    let output = dashprove_cmd()
        .args(["train", "--help"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("train") || stdout.contains("Train"));
}

#[test]
fn test_cli_train_empty_corpus() {
    // Training on empty corpus should fail gracefully
    let temp_dir = unique_temp_file("cli_train_empty", "dir");
    std::fs::create_dir_all(&temp_dir).ok();

    let output = dashprove_cmd()
        .args(["train", "--data-dir", temp_dir.to_str().unwrap()])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_dir_all(&temp_dir).ok();

    // Should either fail gracefully or succeed with empty training
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !output.status.success()
            || stdout.contains("No training")
            || stdout.contains("empty")
            || stderr.contains("No training")
            || stderr.contains("empty")
    );
}

#[test]
fn test_cli_train_with_learning_rate() {
    let temp_dir = unique_temp_file("cli_train_lr", "dir");
    std::fs::create_dir_all(&temp_dir).ok();

    let output = dashprove_cmd()
        .args([
            "train",
            "--data-dir",
            temp_dir.to_str().unwrap(),
            "--learning-rate",
            "0.01",
            "--epochs",
            "1",
        ])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_dir_all(&temp_dir).ok();

    // Command should at least parse arguments correctly
    // May fail due to empty corpus, but shouldn't error on args
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("Invalid")
            || stderr.contains("training")
            || stderr.contains("corpus")
            || stderr.contains("No")
    );
}

#[test]
fn test_cli_train_with_early_stopping() {
    let temp_dir = unique_temp_file("cli_train_es", "dir");
    std::fs::create_dir_all(&temp_dir).ok();

    let output = dashprove_cmd()
        .args([
            "train",
            "--data-dir",
            temp_dir.to_str().unwrap(),
            "--early-stopping",
            "--patience",
            "5",
            "--epochs",
            "1",
        ])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_dir_all(&temp_dir).ok();

    // Arguments should be valid
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("Invalid argument") && !stderr.contains("unknown option"));
}

#[test]
fn test_cli_train_with_scheduler() {
    let temp_dir = unique_temp_file("cli_train_sched", "dir");
    std::fs::create_dir_all(&temp_dir).ok();

    let output = dashprove_cmd()
        .args([
            "train",
            "--data-dir",
            temp_dir.to_str().unwrap(),
            "--lr-scheduler",
            "cosine",
            "--epochs",
            "1",
        ])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_dir_all(&temp_dir).ok();

    // Scheduler argument should be valid
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("Invalid scheduler") && !stderr.contains("unknown scheduler"));
}

// =============================================================================
// Tune command tests
// =============================================================================

#[test]
fn test_cli_tune_help() {
    let output = dashprove_cmd()
        .args(["tune", "--help"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("tune")
            || stdout.contains("Tune")
            || stdout.contains("hyperparameter")
            || stdout.contains("search")
    );
}

#[test]
fn test_cli_tune_grid_search() {
    let temp_dir = unique_temp_file("cli_tune_grid", "dir");
    std::fs::create_dir_all(&temp_dir).ok();

    let output = dashprove_cmd()
        .args([
            "tune",
            "--data-dir",
            temp_dir.to_str().unwrap(),
            "--method",
            "grid",
            "--max-trials",
            "1",
        ])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_dir_all(&temp_dir).ok();

    // Arguments should be valid
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("Invalid method") && !stderr.contains("unknown method"));
}

#[test]
fn test_cli_tune_random_search() {
    let temp_dir = unique_temp_file("cli_tune_random", "dir");
    std::fs::create_dir_all(&temp_dir).ok();

    let output = dashprove_cmd()
        .args([
            "tune",
            "--data-dir",
            temp_dir.to_str().unwrap(),
            "--method",
            "random",
            "--max-trials",
            "2",
        ])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_dir_all(&temp_dir).ok();

    // Arguments should be valid
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("Invalid method") && !stderr.contains("unknown method"));
}

#[test]
fn test_cli_tune_bayesian_search() {
    let temp_dir = unique_temp_file("cli_tune_bayesian", "dir");
    std::fs::create_dir_all(&temp_dir).ok();

    let output = dashprove_cmd()
        .args([
            "tune",
            "--data-dir",
            temp_dir.to_str().unwrap(),
            "--method",
            "bayesian",
            "--max-trials",
            "1",
        ])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_dir_all(&temp_dir).ok();

    // Arguments should be valid
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("Invalid method") && !stderr.contains("unknown method"));
}

// =============================================================================
// Ensemble command tests
// =============================================================================

#[test]
fn test_cli_ensemble_help() {
    let output = dashprove_cmd()
        .args(["ensemble", "--help"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("ensemble")
            || stdout.contains("Ensemble")
            || stdout.contains("model")
            || stdout.contains("combine")
    );
}

#[test]
fn test_cli_ensemble_voting_aggregation() {
    let temp_dir = unique_temp_file("cli_ensemble_vote", "dir");
    std::fs::create_dir_all(&temp_dir).ok();

    let output = dashprove_cmd()
        .args([
            "ensemble",
            "--data-dir",
            temp_dir.to_str().unwrap(),
            "--aggregation",
            "voting",
            "--num-models",
            "2",
        ])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_dir_all(&temp_dir).ok();

    // Arguments should be valid
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("Invalid aggregation") && !stderr.contains("unknown aggregation method")
    );
}

#[test]
fn test_cli_ensemble_averaging_aggregation() {
    let temp_dir = unique_temp_file("cli_ensemble_avg", "dir");
    std::fs::create_dir_all(&temp_dir).ok();

    let output = dashprove_cmd()
        .args([
            "ensemble",
            "--data-dir",
            temp_dir.to_str().unwrap(),
            "--aggregation",
            "averaging",
            "--num-models",
            "2",
        ])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_dir_all(&temp_dir).ok();

    // Arguments should be valid
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("Invalid aggregation") && !stderr.contains("unknown aggregation method")
    );
}

#[test]
fn test_cli_ensemble_stacking_aggregation() {
    let temp_dir = unique_temp_file("cli_ensemble_stack", "dir");
    std::fs::create_dir_all(&temp_dir).ok();

    let output = dashprove_cmd()
        .args([
            "ensemble",
            "--data-dir",
            temp_dir.to_str().unwrap(),
            "--aggregation",
            "stacking",
            "--num-models",
            "2",
        ])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_dir_all(&temp_dir).ok();

    // Arguments should be valid
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("Invalid aggregation") && !stderr.contains("unknown aggregation method")
    );
}

// =============================================================================
// Cluster command tests
// =============================================================================

#[test]
fn test_cli_cluster_help() {
    let output = dashprove_cmd()
        .args(["cluster", "--help"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("cluster")
            || stdout.contains("Cluster")
            || stdout.contains("counterexample")
    );
}

#[test]
fn test_cli_cluster_missing_input() {
    let output = dashprove_cmd()
        .args(["cluster", "nonexistent_dir"])
        .output()
        .expect("Failed to execute command");

    // Should fail with missing input
    assert!(!output.status.success());
}

#[test]
fn test_cli_cluster_requires_two_files() {
    // Cluster requires at least 2 counterexample files
    let temp_file = unique_temp_file("cli_cluster_single", "json");
    {
        let content = r#"{
    "property": "test_property",
    "witness": {"x": {"Int": {"value": 5}}},
    "failed_checks": [],
    "trace": [{"state_num": 0, "variables": {"x": {"Int": {"value": 0}}}}],
    "raw": null,
    "minimized": false
}"#;
        std::fs::write(&temp_file, content).expect("write to temp file");
    }

    let output = dashprove_cmd()
        .args(["cluster", temp_file.to_str().unwrap()])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_file(&temp_file).ok();

    // Should fail because cluster requires at least 2 files
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    // Error should mention missing argument or require 2 files
    assert!(
        stderr.contains("required")
            || stderr.contains("arguments")
            || stderr.contains("2")
            || stderr.contains("PATHS")
    );
}

#[test]
fn test_cli_cluster_with_counterexamples() {
    let ce1 = unique_temp_file("cli_cluster_ce1", "json");
    let ce2 = unique_temp_file("cli_cluster_ce2", "json");

    let content1 = r#"{
    "property": "prop1",
    "witness": {"x": {"Int": {"value": 1}}},
    "failed_checks": [{"check_id": "c1", "description": "fail1"}],
    "trace": [{"state_num": 0, "variables": {"x": {"Int": {"value": 1}}}}],
    "raw": null,
    "minimized": false
}"#;
    let content2 = r#"{
    "property": "prop2",
    "witness": {"x": {"Int": {"value": 2}}},
    "failed_checks": [{"check_id": "c1", "description": "fail2"}],
    "trace": [{"state_num": 0, "variables": {"x": {"Int": {"value": 2}}}}],
    "raw": null,
    "minimized": false
}"#;

    std::fs::write(&ce1, content1).expect("write ce1");
    std::fs::write(&ce2, content2).expect("write ce2");

    let output = dashprove_cmd()
        .args(["cluster", ce1.to_str().unwrap(), ce2.to_str().unwrap()])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_file(&ce1).ok();
    std::fs::remove_file(&ce2).ok();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should produce some output about clusters
    assert!(
        stdout.contains("cluster")
            || stdout.contains("Cluster")
            || stdout.contains("similarity")
            || stdout.contains("counterexample")
    );
}

#[test]
fn test_cli_cluster_output_format() {
    let ce1 = unique_temp_file("cli_cluster_fmt1", "json");
    let ce2 = unique_temp_file("cli_cluster_fmt2", "json");

    let content1 = r#"{
    "property": "prop1",
    "witness": {"x": {"Int": {"value": 1}}},
    "failed_checks": [],
    "trace": [{"state_num": 0, "variables": {"x": {"Int": {"value": 1}}}}],
    "raw": null,
    "minimized": false
}"#;
    let content2 = r#"{
    "property": "prop2",
    "witness": {"x": {"Int": {"value": 2}}},
    "failed_checks": [],
    "trace": [{"state_num": 0, "variables": {"x": {"Int": {"value": 2}}}}],
    "raw": null,
    "minimized": false
}"#;
    std::fs::write(&ce1, content1).expect("write ce1");
    std::fs::write(&ce2, content2).expect("write ce2");

    let output = dashprove_cmd()
        .args([
            "cluster",
            ce1.to_str().unwrap(),
            ce2.to_str().unwrap(),
            "--format",
            "json",
        ])
        .output()
        .expect("Failed to execute command");

    std::fs::remove_file(&ce1).ok();
    std::fs::remove_file(&ce2).ok();

    // Format argument should be accepted and command should succeed
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // JSON format should produce JSON-like output
    assert!(stdout.contains("{") || stdout.contains("["));
}
