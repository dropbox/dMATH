//! Integration tests for monitor CLI command

use serial_test::serial;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_dir(prefix: &str) -> std::path::PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("dashprove_monitor_test_{prefix}_{ts}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn dashprove_cmd() -> Command {
    Command::new(env!("CARGO_BIN_EXE_dashprove"))
}

/// Create a simple valid USL file for testing
fn create_simple_usl(dir: &std::path::Path) -> std::path::PathBuf {
    let usl = r#"
// Simple test specification for monitors
theorem excluded_middle {
    forall x: Bool . x or not x
}

invariant always_true {
    true
}

invariant positive_squared {
    forall n: Int . n >= 0 implies n * n >= 0
}
"#;
    let path = dir.join("simple.usl");
    std::fs::write(&path, usl).unwrap();
    path
}

// ============================================================================
// Rust monitor tests
// ============================================================================

#[test]
#[serial]
fn test_monitor_rust_default() {
    let dir = temp_dir("rust_monitor");
    let usl_path = create_simple_usl(&dir);

    let output = dashprove_cmd()
        .arg("monitor")
        .arg(&usl_path)
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "monitor (rust default) failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Rust monitor should contain fn definitions
    assert!(
        stdout.contains("fn ") || stdout.contains("pub fn"),
        "Rust monitor should contain function definitions: {}",
        stdout
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_monitor_rust_explicit() {
    let dir = temp_dir("rust_explicit");
    let usl_path = create_simple_usl(&dir);

    let output = dashprove_cmd()
        .arg("monitor")
        .arg(&usl_path)
        .arg("--target")
        .arg("rust")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "monitor --target rust failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("fn ") || stdout.contains("impl"),
        "Rust monitor should contain Rust syntax: {}",
        stdout
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_monitor_rust_with_assertions() {
    let dir = temp_dir("rust_assertions");
    let usl_path = create_simple_usl(&dir);

    let output = dashprove_cmd()
        .arg("monitor")
        .arg(&usl_path)
        .arg("--target")
        .arg("rust")
        .arg("--assertions")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "monitor --assertions failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // The assertions flag should be documented in header and/or code
    assert!(
        stdout.contains("Assertions: enabled") || stdout.contains("assertions"),
        "Rust monitor with assertions should indicate assertion mode: {}",
        stdout
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_monitor_rust_with_logging() {
    let dir = temp_dir("rust_logging");
    let usl_path = create_simple_usl(&dir);

    let output = dashprove_cmd()
        .arg("monitor")
        .arg(&usl_path)
        .arg("--target")
        .arg("rust")
        .arg("--logging")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "monitor --logging failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // With logging, should contain log or tracing macros
    assert!(
        stdout.contains("log")
            || stdout.contains("tracing")
            || stdout.contains("info!")
            || stdout.contains("warn!"),
        "Rust monitor with logging should contain logging: {}",
        stdout
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_monitor_rust_with_metrics() {
    let dir = temp_dir("rust_metrics");
    let usl_path = create_simple_usl(&dir);

    let output = dashprove_cmd()
        .arg("monitor")
        .arg(&usl_path)
        .arg("--target")
        .arg("rust")
        .arg("--metrics")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "monitor --metrics failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // The metrics flag should be documented in header and/or code
    assert!(
        stdout.contains("Metrics: enabled") || stdout.contains("metrics"),
        "Rust monitor with metrics should indicate metrics mode: {}",
        stdout
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_monitor_rust_with_output_file() {
    let dir = temp_dir("rust_output");
    let usl_path = create_simple_usl(&dir);
    let output_path = dir.join("monitor.rs");

    let output = dashprove_cmd()
        .arg("monitor")
        .arg(&usl_path)
        .arg("--target")
        .arg("rust")
        .arg("-o")
        .arg(&output_path)
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "monitor -o failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(output_path.exists(), "Output file should be created");
    let content = std::fs::read_to_string(&output_path).unwrap();
    assert!(!content.is_empty(), "Output file should not be empty");
    assert!(
        content.contains("fn "),
        "Output file should contain Rust code"
    );

    std::fs::remove_dir_all(&dir).ok();
}

// ============================================================================
// TypeScript monitor tests
// ============================================================================

#[test]
#[serial]
fn test_monitor_typescript() {
    let dir = temp_dir("ts_monitor");
    let usl_path = create_simple_usl(&dir);

    let output = dashprove_cmd()
        .arg("monitor")
        .arg(&usl_path)
        .arg("--target")
        .arg("typescript")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "monitor --target typescript failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // TypeScript should contain function or class definitions
    assert!(
        stdout.contains("function ")
            || stdout.contains("class ")
            || stdout.contains("const ")
            || stdout.contains("export "),
        "TypeScript monitor should contain TS syntax: {}",
        stdout
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_monitor_typescript_with_assertions() {
    let dir = temp_dir("ts_assertions");
    let usl_path = create_simple_usl(&dir);

    let output = dashprove_cmd()
        .arg("monitor")
        .arg(&usl_path)
        .arg("--target")
        .arg("typescript")
        .arg("--assertions")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "monitor typescript --assertions failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // The assertions flag should be documented in header and/or code
    assert!(
        stdout.contains("Assertions: enabled") || stdout.contains("assertions"),
        "TypeScript monitor with assertions should indicate assertion mode: {}",
        stdout
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_monitor_typescript_with_logging() {
    let dir = temp_dir("ts_logging");
    let usl_path = create_simple_usl(&dir);

    let output = dashprove_cmd()
        .arg("monitor")
        .arg(&usl_path)
        .arg("--target")
        .arg("typescript")
        .arg("--logging")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "monitor typescript --logging failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // The logging flag should be documented in header and/or code
    assert!(
        stdout.contains("Logging: enabled") || stdout.contains("logging"),
        "TypeScript monitor with logging should indicate logging mode: {}",
        stdout
    );

    std::fs::remove_dir_all(&dir).ok();
}

// ============================================================================
// Python monitor tests
// ============================================================================

#[test]
#[serial]
fn test_monitor_python() {
    let dir = temp_dir("py_monitor");
    let usl_path = create_simple_usl(&dir);

    let output = dashprove_cmd()
        .arg("monitor")
        .arg(&usl_path)
        .arg("--target")
        .arg("python")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "monitor --target python failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Python should contain def or class
    assert!(
        stdout.contains("def ") || stdout.contains("class "),
        "Python monitor should contain Python syntax: {}",
        stdout
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_monitor_python_with_assertions() {
    let dir = temp_dir("py_assertions");
    let usl_path = create_simple_usl(&dir);

    let output = dashprove_cmd()
        .arg("monitor")
        .arg(&usl_path)
        .arg("--target")
        .arg("python")
        .arg("--assertions")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "monitor python --assertions failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // The assertions flag should be documented in header and/or code
    assert!(
        stdout.contains("Assertions: enabled") || stdout.contains("assertions"),
        "Python monitor with assertions should indicate assertion mode: {}",
        stdout
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_monitor_python_with_logging() {
    let dir = temp_dir("py_logging");
    let usl_path = create_simple_usl(&dir);

    let output = dashprove_cmd()
        .arg("monitor")
        .arg(&usl_path)
        .arg("--target")
        .arg("python")
        .arg("--logging")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "monitor python --logging failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("logging") || stdout.contains("logger") || stdout.contains("print("),
        "Python monitor with logging should contain logging: {}",
        stdout
    );

    std::fs::remove_dir_all(&dir).ok();
}

// ============================================================================
// Combined options tests
// ============================================================================

#[test]
#[serial]
fn test_monitor_all_options() {
    let dir = temp_dir("all_options");
    let usl_path = create_simple_usl(&dir);

    let output = dashprove_cmd()
        .arg("monitor")
        .arg(&usl_path)
        .arg("--target")
        .arg("rust")
        .arg("--assertions")
        .arg("--logging")
        .arg("--metrics")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "monitor with all options failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should have content
    assert!(!stdout.is_empty(), "Output should not be empty");

    std::fs::remove_dir_all(&dir).ok();
}

// ============================================================================
// Error handling tests
// ============================================================================

#[test]
#[serial]
fn test_monitor_nonexistent_file() {
    let output = dashprove_cmd()
        .arg("monitor")
        .arg("/nonexistent/path/file.usl")
        .output()
        .expect("Failed to execute command");

    assert!(
        !output.status.success(),
        "monitor should fail for nonexistent file"
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
fn test_monitor_invalid_usl() {
    let dir = temp_dir("invalid_usl");
    let invalid_usl = r#"
// Invalid specification
theorem broken {
    forall x: Bool . x and and y
}
"#;
    let path = dir.join("invalid.usl");
    std::fs::write(&path, invalid_usl).unwrap();

    let output = dashprove_cmd()
        .arg("monitor")
        .arg(&path)
        .output()
        .expect("Failed to execute command");

    assert!(
        !output.status.success(),
        "monitor should fail on invalid USL"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
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
fn test_monitor_invalid_target() {
    let dir = temp_dir("invalid_target");
    let usl_path = create_simple_usl(&dir);

    let output = dashprove_cmd()
        .arg("monitor")
        .arg(&usl_path)
        .arg("--target")
        .arg("invalid_language")
        .output()
        .expect("Failed to execute command");

    assert!(
        !output.status.success(),
        "monitor should fail for invalid target"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.to_lowercase().contains("unknown")
            || stderr.to_lowercase().contains("invalid")
            || stderr.to_lowercase().contains("unsupported")
            || stderr.to_lowercase().contains("error"),
        "Error should mention invalid target: {}",
        stderr
    );

    std::fs::remove_dir_all(&dir).ok();
}

// ============================================================================
// Using examples/usl files directly
// ============================================================================

#[test]
#[serial]
fn test_monitor_examples_basic() {
    let example_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/usl/basic.usl");

    if !example_path.exists() {
        return;
    }

    let output = dashprove_cmd()
        .arg("monitor")
        .arg(&example_path)
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "monitor examples/usl/basic.usl failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// ============================================================================
// Help test
// ============================================================================

#[test]
#[serial]
fn test_monitor_help() {
    let output = dashprove_cmd()
        .arg("monitor")
        .arg("--help")
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success(), "monitor --help failed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("--target")
            && stdout.contains("--assertions")
            && stdout.contains("--logging"),
        "Help should mention monitor options: {}",
        stdout
    );
}
