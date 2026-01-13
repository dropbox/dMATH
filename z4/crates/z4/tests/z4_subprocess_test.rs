// Test for Z4 subprocess spawning
// Reproduces issue: https://docs.KANI_FAST_CRITICAL_Z4_SIGKILL_BUG_2026-01-02.md

use std::process::Command;

#[test]
fn test_z4_subprocess_spawn() {
    let z4_path = env!("CARGO_BIN_EXE_z4");

    let test_input = r#"(set-logic QF_LIA)
(declare-const x Int)
(assert (= x 42))
(check-sat)
"#;

    // Write to temp file
    let temp_path = "/tmp/z4_subprocess_test.smt2";
    std::fs::write(temp_path, test_input).unwrap();

    // Spawn Z4
    let output = Command::new(z4_path)
        .arg(temp_path)
        .output()
        .expect("Failed to spawn z4");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    eprintln!("Z4 stdout: {:?}", stdout);
    eprintln!("Z4 stderr: {:?}", stderr);
    eprintln!("Z4 exit: {:?}", output.status);

    assert!(
        output.status.success(),
        "Z4 exited with {:?}",
        output.status
    );
    assert!(stdout.contains("sat"), "Expected 'sat', got: {}", stdout);
}

#[test]
fn test_z4_stdin_spawn() {
    let z4_path = env!("CARGO_BIN_EXE_z4");

    let test_input = r#"(set-logic QF_LIA)
(declare-const x Int)
(assert (= x 42))
(check-sat)
"#;

    // Spawn Z4 with stdin
    use std::io::Write;
    use std::process::Stdio;

    let mut child = Command::new(z4_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn z4");

    {
        let stdin = child.stdin.as_mut().unwrap();
        stdin.write_all(test_input.as_bytes()).unwrap();
    }

    let output = child.wait_with_output().expect("Failed to wait on z4");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    eprintln!("Z4 stdin stdout: {:?}", stdout);
    eprintln!("Z4 stdin stderr: {:?}", stderr);
    eprintln!("Z4 stdin exit: {:?}", output.status);

    assert!(
        output.status.success(),
        "Z4 exited with {:?}",
        output.status
    );
    assert!(stdout.contains("sat"), "Expected 'sat', got: {}", stdout);
}
