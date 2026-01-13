//! Integration tests for prove CLI command (interactive mode)

use serial_test::serial;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_dir(prefix: &str) -> std::path::PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("dashprove_prove_test_{prefix}_{ts}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn dashprove_cmd() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_dashprove"));
    // Set a timeout for interactive commands to prevent hanging
    cmd.env("DASHPROVE_INTERACTIVE_TIMEOUT", "5");
    cmd
}

/// Create a sample USL specification with multiple properties
fn create_sample_spec(dir: &std::path::Path) -> std::path::PathBuf {
    let spec = r#"// Sample specification for prove command testing
theorem excluded_middle {
    forall x: Bool . x or not x
}

theorem de_morgan {
    forall a: Bool, b: Bool . not (a and b) == (not a or not b)
}

invariant always_positive {
    forall n: Int . n * n >= 0
}
"#;
    let path = dir.join("sample.usl");
    std::fs::write(&path, spec).unwrap();
    path
}

/// Create an empty USL specification (no properties)
fn create_empty_spec(dir: &std::path::Path) -> std::path::PathBuf {
    let spec = "// Empty specification with no properties\n";
    let path = dir.join("empty.usl");
    std::fs::write(&path, spec).unwrap();
    path
}

#[test]
#[serial]
fn test_prove_help() {
    let output = dashprove_cmd()
        .args(["prove", "--help"])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("--hints"),
        "Should show --hints option: {}",
        stdout
    );
    assert!(
        stdout.contains("Interactive"),
        "Should mention interactive mode: {}",
        stdout
    );
}

#[test]
#[serial]
fn test_prove_missing_file() {
    let output = dashprove_cmd()
        .args(["prove", "/nonexistent/path/spec.usl"])
        .stdin(Stdio::null())
        .output()
        .expect("Failed to run dashprove");

    assert!(
        !output.status.success(),
        "Should fail with nonexistent file"
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
fn test_prove_empty_spec() {
    let dir = temp_dir("empty_spec");
    let spec_path = create_empty_spec(&dir);

    let output = dashprove_cmd()
        .args(["prove", spec_path.to_str().unwrap()])
        .stdin(Stdio::null())
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should indicate no properties found
    assert!(
        stdout.contains("No properties"),
        "Should indicate no properties found: {}",
        stdout
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_prove_list_command() {
    let dir = temp_dir("list_cmd");
    let spec_path = create_sample_spec(&dir);

    // Create a child process with piped stdin
    let mut child = dashprove_cmd()
        .args(["prove", spec_path.to_str().unwrap()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn dashprove");

    // Send commands
    {
        let stdin = child.stdin.as_mut().unwrap();
        writeln!(stdin, "list").unwrap();
        writeln!(stdin, "quit").unwrap();
    }

    let output = child.wait_with_output().expect("Failed to wait for child");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Should show the interactive header and property list
    assert!(
        stdout.contains("Interactive"),
        "Should show interactive mode header: {}",
        stdout
    );
    assert!(
        stdout.contains("excluded_middle"),
        "Should list excluded_middle property: {}",
        stdout
    );
    assert!(
        stdout.contains("de_morgan"),
        "Should list de_morgan property: {}",
        stdout
    );
    assert!(
        stdout.contains("always_positive"),
        "Should list always_positive property: {}",
        stdout
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_prove_quit_command() {
    let dir = temp_dir("quit_cmd");
    let spec_path = create_sample_spec(&dir);

    let mut child = dashprove_cmd()
        .args(["prove", spec_path.to_str().unwrap()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn dashprove");

    {
        let stdin = child.stdin.as_mut().unwrap();
        writeln!(stdin, "quit").unwrap();
    }

    let output = child.wait_with_output().expect("Failed to wait for child");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Exiting"),
        "Should show exiting message: {}",
        stdout
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_prove_select_command() {
    let dir = temp_dir("select_cmd");
    let spec_path = create_sample_spec(&dir);

    let mut child = dashprove_cmd()
        .args(["prove", spec_path.to_str().unwrap()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn dashprove");

    {
        let stdin = child.stdin.as_mut().unwrap();
        writeln!(stdin, "select 1").unwrap();
        writeln!(stdin, "quit").unwrap();
    }

    let output = child.wait_with_output().expect("Failed to wait for child");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Selected:") && stdout.contains("excluded_middle"),
        "Should show selected property: {}",
        stdout
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_prove_select_invalid_number() {
    let dir = temp_dir("select_invalid");
    let spec_path = create_sample_spec(&dir);

    let mut child = dashprove_cmd()
        .args(["prove", spec_path.to_str().unwrap()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn dashprove");

    {
        let stdin = child.stdin.as_mut().unwrap();
        writeln!(stdin, "select 99").unwrap();
        writeln!(stdin, "quit").unwrap();
    }

    let output = child.wait_with_output().expect("Failed to wait for child");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Invalid property number"),
        "Should show invalid property error: {}",
        stdout
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_prove_select_without_number() {
    let dir = temp_dir("select_no_num");
    let spec_path = create_sample_spec(&dir);

    let mut child = dashprove_cmd()
        .args(["prove", spec_path.to_str().unwrap()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn dashprove");

    {
        let stdin = child.stdin.as_mut().unwrap();
        writeln!(stdin, "select").unwrap();
        writeln!(stdin, "quit").unwrap();
    }

    let output = child.wait_with_output().expect("Failed to wait for child");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Usage:") && stdout.contains("select"),
        "Should show usage message: {}",
        stdout
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_prove_export_without_selection() {
    let dir = temp_dir("export_no_sel");
    let spec_path = create_sample_spec(&dir);

    let mut child = dashprove_cmd()
        .args(["prove", spec_path.to_str().unwrap()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn dashprove");

    {
        let stdin = child.stdin.as_mut().unwrap();
        writeln!(stdin, "export lean").unwrap();
        writeln!(stdin, "quit").unwrap();
    }

    let output = child.wait_with_output().expect("Failed to wait for child");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("No property selected"),
        "Should indicate no property selected: {}",
        stdout
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_prove_export_lean() {
    let dir = temp_dir("export_lean");
    let spec_path = create_sample_spec(&dir);

    let mut child = dashprove_cmd()
        .args(["prove", spec_path.to_str().unwrap()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn dashprove");

    {
        let stdin = child.stdin.as_mut().unwrap();
        writeln!(stdin, "select 1").unwrap();
        writeln!(stdin, "export lean").unwrap();
        writeln!(stdin, "quit").unwrap();
    }

    let output = child.wait_with_output().expect("Failed to wait for child");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("lean output"),
        "Should show lean output header: {}",
        stdout
    );
    // LEAN 4 output should contain theorem syntax
    assert!(
        stdout.contains("theorem") || stdout.contains("Theorem"),
        "Should contain Lean theorem syntax: {}",
        stdout
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_prove_export_tlaplus() {
    let dir = temp_dir("export_tla");
    let spec_path = create_sample_spec(&dir);

    let mut child = dashprove_cmd()
        .args(["prove", spec_path.to_str().unwrap()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn dashprove");

    {
        let stdin = child.stdin.as_mut().unwrap();
        writeln!(stdin, "select 1").unwrap();
        writeln!(stdin, "export tla+").unwrap();
        writeln!(stdin, "quit").unwrap();
    }

    let output = child.wait_with_output().expect("Failed to wait for child");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("tla+ output"),
        "Should show TLA+ output header: {}",
        stdout
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_prove_export_unknown_backend() {
    let dir = temp_dir("export_unknown");
    let spec_path = create_sample_spec(&dir);

    let mut child = dashprove_cmd()
        .args(["prove", spec_path.to_str().unwrap()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn dashprove");

    {
        let stdin = child.stdin.as_mut().unwrap();
        writeln!(stdin, "select 1").unwrap();
        writeln!(stdin, "export unknown").unwrap();
        writeln!(stdin, "quit").unwrap();
    }

    let output = child.wait_with_output().expect("Failed to wait for child");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Unknown target"),
        "Should show unknown target error: {}",
        stdout
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_prove_tactics_without_selection() {
    let dir = temp_dir("tactics_no_sel");
    let spec_path = create_sample_spec(&dir);

    let mut child = dashprove_cmd()
        .args(["prove", spec_path.to_str().unwrap()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn dashprove");

    {
        let stdin = child.stdin.as_mut().unwrap();
        writeln!(stdin, "tactics").unwrap();
        writeln!(stdin, "quit").unwrap();
    }

    let output = child.wait_with_output().expect("Failed to wait for child");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("No property selected"),
        "Should indicate no property selected: {}",
        stdout
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_prove_tactics_with_selection() {
    let dir = temp_dir("tactics_sel");
    let spec_path = create_sample_spec(&dir);

    let mut child = dashprove_cmd()
        .args(["prove", spec_path.to_str().unwrap()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn dashprove");

    {
        let stdin = child.stdin.as_mut().unwrap();
        writeln!(stdin, "select 1").unwrap();
        writeln!(stdin, "tactics").unwrap();
        writeln!(stdin, "quit").unwrap();
    }

    let output = child.wait_with_output().expect("Failed to wait for child");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Compiler suggestions"),
        "Should show compiler suggestions: {}",
        stdout
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_prove_show_command() {
    let dir = temp_dir("show_cmd");
    let spec_path = create_sample_spec(&dir);

    let mut child = dashprove_cmd()
        .args(["prove", spec_path.to_str().unwrap()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn dashprove");

    {
        let stdin = child.stdin.as_mut().unwrap();
        writeln!(stdin, "select 1").unwrap();
        writeln!(stdin, "show").unwrap();
        writeln!(stdin, "quit").unwrap();
    }

    let output = child.wait_with_output().expect("Failed to wait for child");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Theorem #1"),
        "Show should print property header: {}",
        stdout
    );
    assert!(
        stdout.contains("Body:"),
        "Show should include property details: {}",
        stdout
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_prove_similar_without_selection() {
    let dir = temp_dir("similar_no_sel");
    let spec_path = create_sample_spec(&dir);

    let mut child = dashprove_cmd()
        .args(["prove", spec_path.to_str().unwrap()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn dashprove");

    {
        let stdin = child.stdin.as_mut().unwrap();
        writeln!(stdin, "similar").unwrap();
        writeln!(stdin, "quit").unwrap();
    }

    let output = child.wait_with_output().expect("Failed to wait for child");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("No property selected"),
        "Should indicate no property selected: {}",
        stdout
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_prove_similar_without_hints() {
    let dir = temp_dir("similar_no_hints");
    let spec_path = create_sample_spec(&dir);

    let mut child = dashprove_cmd()
        .args(["prove", spec_path.to_str().unwrap()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn dashprove");

    {
        let stdin = child.stdin.as_mut().unwrap();
        writeln!(stdin, "select 1").unwrap();
        writeln!(stdin, "similar").unwrap();
        writeln!(stdin, "quit").unwrap();
    }

    let output = child.wait_with_output().expect("Failed to wait for child");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    // Without --hints, learning data is not loaded
    assert!(
        stdout.contains("Learning data not loaded") || stdout.contains("No similar proofs"),
        "Should indicate learning data not loaded or no similar proofs: {}",
        stdout
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_prove_unknown_command() {
    let dir = temp_dir("unknown_cmd");
    let spec_path = create_sample_spec(&dir);

    let mut child = dashprove_cmd()
        .args(["prove", spec_path.to_str().unwrap()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn dashprove");

    {
        let stdin = child.stdin.as_mut().unwrap();
        writeln!(stdin, "foobar").unwrap();
        writeln!(stdin, "quit").unwrap();
    }

    let output = child.wait_with_output().expect("Failed to wait for child");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Unknown command") && stdout.contains("foobar"),
        "Should show unknown command error: {}",
        stdout
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_prove_search_command() {
    let dir = temp_dir("search_cmd");
    let spec_path = create_sample_spec(&dir);

    let mut child = dashprove_cmd()
        .args(["prove", spec_path.to_str().unwrap()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn dashprove");

    {
        let stdin = child.stdin.as_mut().unwrap();
        writeln!(stdin, "search de").unwrap();
        writeln!(stdin, "quit").unwrap();
    }

    let output = child.wait_with_output().expect("Failed to wait for child");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.to_lowercase().contains("search results"),
        "Search should print a results header: {}",
        stdout
    );
    assert!(
        stdout.contains("de_morgan"),
        "Search should include matching property name: {}",
        stdout
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_prove_help_command() {
    let dir = temp_dir("help_cmd");
    let spec_path = create_sample_spec(&dir);

    let mut child = dashprove_cmd()
        .args(["prove", spec_path.to_str().unwrap()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn dashprove");

    {
        let stdin = child.stdin.as_mut().unwrap();
        writeln!(stdin, "help").unwrap();
        writeln!(stdin, "quit").unwrap();
    }

    let output = child.wait_with_output().expect("Failed to wait for child");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Commands:"),
        "Should show commands header: {}",
        stdout
    );
    assert!(
        stdout.contains("select") && stdout.contains("list") && stdout.contains("tactics"),
        "Should list available commands: {}",
        stdout
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_prove_eof_handling() {
    let dir = temp_dir("eof");
    let spec_path = create_sample_spec(&dir);

    let mut child = dashprove_cmd()
        .args(["prove", spec_path.to_str().unwrap()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn dashprove");

    // Send some commands then close stdin (EOF)
    {
        let stdin = child.stdin.as_mut().unwrap();
        writeln!(stdin, "list").unwrap();
        // Drop stdin to send EOF
    }
    drop(child.stdin.take());

    let output = child.wait_with_output().expect("Failed to wait for child");

    // Should exit gracefully on EOF
    assert!(
        output.status.success(),
        "Should exit gracefully on EOF: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_prove_shorthand_commands() {
    let dir = temp_dir("shorthand");
    let spec_path = create_sample_spec(&dir);

    let mut child = dashprove_cmd()
        .args(["prove", spec_path.to_str().unwrap()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn dashprove");

    {
        let stdin = child.stdin.as_mut().unwrap();
        // Test shorthand commands
        writeln!(stdin, "ls").unwrap(); // shorthand for list
        writeln!(stdin, "sel 1").unwrap(); // shorthand for select
        writeln!(stdin, "tac").unwrap(); // shorthand for tactics
        writeln!(stdin, "sim").unwrap(); // shorthand for similar
        writeln!(stdin, "?").unwrap(); // shorthand for help
        writeln!(stdin, "q").unwrap(); // shorthand for quit
    }

    let output = child.wait_with_output().expect("Failed to wait for child");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    // Verify shorthand commands worked
    assert!(
        stdout.contains("Properties:"),
        "ls should list properties: {}",
        stdout
    );
    assert!(
        stdout.contains("Selected:"),
        "sel should select property: {}",
        stdout
    );
    assert!(stdout.contains("Exiting"), "q should exit: {}", stdout);

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_prove_prompt_changes_after_selection() {
    let dir = temp_dir("prompt_change");
    let spec_path = create_sample_spec(&dir);

    let mut child = dashprove_cmd()
        .args(["prove", spec_path.to_str().unwrap()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn dashprove");

    {
        let stdin = child.stdin.as_mut().unwrap();
        writeln!(stdin, "select 2").unwrap();
        writeln!(stdin, "quit").unwrap();
    }

    let output = child.wait_with_output().expect("Failed to wait for child");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    // After selecting property 2 (de_morgan), the prompt should include the property name
    assert!(
        stdout.contains("[de_morgan]>"),
        "Prompt should include selected property name: {}",
        stdout
    );

    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[serial]
fn test_prove_empty_line_handling() {
    let dir = temp_dir("empty_line");
    let spec_path = create_sample_spec(&dir);

    let mut child = dashprove_cmd()
        .args(["prove", spec_path.to_str().unwrap()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn dashprove");

    {
        let stdin = child.stdin.as_mut().unwrap();
        writeln!(stdin).unwrap(); // empty line
        writeln!(stdin, "   ").unwrap(); // whitespace only
        writeln!(stdin, "list").unwrap();
        writeln!(stdin, "quit").unwrap();
    }

    let output = child.wait_with_output().expect("Failed to wait for child");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    // Empty lines should be silently ignored, list should still work
    assert!(
        stdout.contains("Properties:"),
        "Empty lines should not break the REPL: {}",
        stdout
    );

    std::fs::remove_dir_all(dir).ok();
}
