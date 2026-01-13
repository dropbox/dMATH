//! Integration tests for export CLI command

use serial_test::serial;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_dir(prefix: &str) -> std::path::PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("dashprove_export_test_{prefix}_{ts}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn dashprove_cmd() -> Command {
    Command::new(env!("CARGO_BIN_EXE_dashprove"))
}

/// Create a simple valid USL file for testing
fn create_simple_usl(dir: &std::path::Path) -> std::path::PathBuf {
    let usl = r#"
// Simple test specification
theorem excluded_middle {
    forall x: Bool . x or not x
}

invariant always_true {
    true
}
"#;
    let path = dir.join("simple.usl");
    std::fs::write(&path, usl).unwrap();
    path
}

// ============================================================================
// Export to LEAN tests
// ============================================================================

#[test]
#[serial]
fn test_export_to_lean() {
    let dir = temp_dir("lean_export");
    let usl_path = create_simple_usl(&dir);

    let output = dashprove_cmd()
        .arg("export")
        .arg(&usl_path)
        .arg("--target")
        .arg("lean")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "export to lean failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // LEAN export should contain theorem declarations
    assert!(
        stdout.contains("theorem") || stdout.contains("def") || stdout.contains("lemma"),
        "LEAN export should contain theorem declarations: {}",
        stdout
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_export_to_lean_with_output_file() {
    let dir = temp_dir("lean_export_file");
    let usl_path = create_simple_usl(&dir);
    let output_path = dir.join("output.lean");

    let output = dashprove_cmd()
        .arg("export")
        .arg(&usl_path)
        .arg("--target")
        .arg("lean")
        .arg("-o")
        .arg(&output_path)
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "export to lean file failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Output file should exist and have content
    assert!(output_path.exists(), "Output file should be created");
    let content = std::fs::read_to_string(&output_path).unwrap();
    assert!(!content.is_empty(), "Output file should not be empty");

    std::fs::remove_dir_all(&dir).ok();
}

// ============================================================================
// Export to TLA+ tests
// ============================================================================

#[test]
#[serial]
fn test_export_to_tlaplus() {
    let dir = temp_dir("tla_export");
    let usl_path = create_simple_usl(&dir);

    let output = dashprove_cmd()
        .arg("export")
        .arg(&usl_path)
        .arg("--target")
        .arg("tla+")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "export to tla+ failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // TLA+ export should contain module structure
    assert!(
        stdout.contains("MODULE")
            || stdout.contains("EXTENDS")
            || stdout.contains("VARIABLE")
            || stdout.contains("---"),
        "TLA+ export should contain TLA+ syntax: {}",
        stdout
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_export_to_tlaplus_with_output() {
    let dir = temp_dir("tla_export_file");
    let usl_path = create_simple_usl(&dir);
    let output_path = dir.join("output.tla");

    let output = dashprove_cmd()
        .arg("export")
        .arg(&usl_path)
        .arg("--target")
        .arg("tla+")
        .arg("--output")
        .arg(&output_path)
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "export to tla+ file failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(output_path.exists(), "TLA+ output file should be created");
    let content = std::fs::read_to_string(&output_path).unwrap();
    assert!(!content.is_empty(), "TLA+ output should not be empty");

    std::fs::remove_dir_all(&dir).ok();
}

// ============================================================================
// Export to Alloy tests
// ============================================================================

#[test]
#[serial]
fn test_export_to_alloy() {
    let dir = temp_dir("alloy_export");
    let usl_path = create_simple_usl(&dir);

    let output = dashprove_cmd()
        .arg("export")
        .arg(&usl_path)
        .arg("--target")
        .arg("alloy")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "export to alloy failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Alloy export should contain signature or fact declarations
    assert!(
        stdout.contains("sig")
            || stdout.contains("fact")
            || stdout.contains("pred")
            || stdout.contains("assert"),
        "Alloy export should contain Alloy syntax: {}",
        stdout
    );

    std::fs::remove_dir_all(&dir).ok();
}

// ============================================================================
// Export to Kani tests
// ============================================================================

#[test]
#[serial]
fn test_export_to_kani() {
    let dir = temp_dir("kani_export");
    let usl_path = create_simple_usl(&dir);

    let output = dashprove_cmd()
        .arg("export")
        .arg(&usl_path)
        .arg("--target")
        .arg("kani")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "export to kani failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Kani export should contain Rust proof harness or kani attributes
    assert!(
        stdout.contains("kani")
            || stdout.contains("fn ")
            || stdout.contains("#[")
            || stdout.contains("proof"),
        "Kani export should contain Rust/Kani syntax: {}",
        stdout
    );

    std::fs::remove_dir_all(&dir).ok();
}

// ============================================================================
// Error handling tests
// ============================================================================

#[test]
#[serial]
fn test_export_invalid_target() {
    let dir = temp_dir("invalid_target");
    let usl_path = create_simple_usl(&dir);

    let output = dashprove_cmd()
        .arg("export")
        .arg(&usl_path)
        .arg("--target")
        .arg("nonexistent_backend")
        .output()
        .expect("Failed to execute command");

    // Should fail with unknown target error
    assert!(
        !output.status.success(),
        "export should fail for invalid target"
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

#[test]
#[serial]
fn test_export_nonexistent_file() {
    let output = dashprove_cmd()
        .arg("export")
        .arg("/nonexistent/path/file.usl")
        .arg("--target")
        .arg("lean")
        .output()
        .expect("Failed to execute command");

    // Should fail with file not found
    assert!(
        !output.status.success(),
        "export should fail for nonexistent file"
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
fn test_export_invalid_usl_syntax() {
    let dir = temp_dir("invalid_syntax");
    let invalid_usl = r#"
// Invalid specification - syntax error
theorem broken {
    forall x: Bool . x and and y
}
"#;
    let path = dir.join("invalid.usl");
    std::fs::write(&path, invalid_usl).unwrap();

    let output = dashprove_cmd()
        .arg("export")
        .arg(&path)
        .arg("--target")
        .arg("lean")
        .output()
        .expect("Failed to execute command");

    // Should fail with parse error
    assert!(
        !output.status.success(),
        "export should fail on invalid syntax"
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

// ============================================================================
// Using examples/usl files directly
// ============================================================================

#[test]
#[serial]
fn test_export_examples_basic_to_lean() {
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
        .arg("export")
        .arg(&example_path)
        .arg("--target")
        .arg("lean")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "export examples/usl/basic.usl to lean failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
#[serial]
fn test_export_examples_basic_to_alloy() {
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
        .arg("export")
        .arg(&example_path)
        .arg("--target")
        .arg("alloy")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "export examples/usl/basic.usl to alloy failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// ============================================================================
// Export to Coq tests
// ============================================================================

#[test]
#[serial]
fn test_export_to_coq() {
    let dir = temp_dir("coq_export");
    let usl_path = create_simple_usl(&dir);

    let output = dashprove_cmd()
        .arg("export")
        .arg(&usl_path)
        .arg("--target")
        .arg("coq")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "export to coq failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Coq export should contain theorem, Lemma, Definition, or proof keywords
    assert!(
        stdout.contains("Theorem")
            || stdout.contains("Lemma")
            || stdout.contains("Definition")
            || stdout.contains("Proof")
            || stdout.contains("Require"),
        "Coq export should contain Coq syntax: {}",
        stdout
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_export_to_coq_with_output_file() {
    let dir = temp_dir("coq_export_file");
    let usl_path = create_simple_usl(&dir);
    let output_path = dir.join("output.v");

    let output = dashprove_cmd()
        .arg("export")
        .arg(&usl_path)
        .arg("--target")
        .arg("coq")
        .arg("-o")
        .arg(&output_path)
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "export to coq file failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(output_path.exists(), "Coq output file should be created");
    let content = std::fs::read_to_string(&output_path).unwrap();
    assert!(!content.is_empty(), "Coq output should not be empty");

    std::fs::remove_dir_all(&dir).ok();
}

// ============================================================================
// Export to Isabelle tests
// ============================================================================

#[test]
#[serial]
fn test_export_to_isabelle() {
    let dir = temp_dir("isabelle_export");
    let usl_path = create_simple_usl(&dir);

    let output = dashprove_cmd()
        .arg("export")
        .arg(&usl_path)
        .arg("--target")
        .arg("isabelle")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "export to isabelle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Isabelle export should contain theory structure
    assert!(
        stdout.contains("theory")
            || stdout.contains("lemma")
            || stdout.contains("theorem")
            || stdout.contains("imports")
            || stdout.contains("begin")
            || stdout.contains("definition"),
        "Isabelle export should contain Isabelle/HOL syntax: {}",
        stdout
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_export_to_isabelle_with_output_file() {
    let dir = temp_dir("isabelle_export_file");
    let usl_path = create_simple_usl(&dir);
    let output_path = dir.join("output.thy");

    let output = dashprove_cmd()
        .arg("export")
        .arg(&usl_path)
        .arg("--target")
        .arg("isabelle")
        .arg("-o")
        .arg(&output_path)
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "export to isabelle file failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(
        output_path.exists(),
        "Isabelle output file should be created"
    );
    let content = std::fs::read_to_string(&output_path).unwrap();
    assert!(!content.is_empty(), "Isabelle output should not be empty");

    std::fs::remove_dir_all(&dir).ok();
}

// ============================================================================
// Export to Dafny tests
// ============================================================================

#[test]
#[serial]
fn test_export_to_dafny() {
    let dir = temp_dir("dafny_export");
    let usl_path = create_simple_usl(&dir);

    let output = dashprove_cmd()
        .arg("export")
        .arg(&usl_path)
        .arg("--target")
        .arg("dafny")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "export to dafny failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Dafny export should contain lemma, method, function, predicate keywords
    assert!(
        stdout.contains("lemma")
            || stdout.contains("method")
            || stdout.contains("function")
            || stdout.contains("predicate")
            || stdout.contains("forall")
            || stdout.contains("requires")
            || stdout.contains("ensures"),
        "Dafny export should contain Dafny syntax: {}",
        stdout
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_export_to_dafny_with_output_file() {
    let dir = temp_dir("dafny_export_file");
    let usl_path = create_simple_usl(&dir);
    let output_path = dir.join("output.dfy");

    let output = dashprove_cmd()
        .arg("export")
        .arg(&usl_path)
        .arg("--target")
        .arg("dafny")
        .arg("-o")
        .arg(&output_path)
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "export to dafny file failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(output_path.exists(), "Dafny output file should be created");
    let content = std::fs::read_to_string(&output_path).unwrap();
    assert!(!content.is_empty(), "Dafny output should not be empty");

    std::fs::remove_dir_all(&dir).ok();
}

// ============================================================================
// Export to SMT-LIB2 tests
// ============================================================================

#[test]
#[serial]
fn test_export_to_smtlib() {
    let dir = temp_dir("smtlib_export");
    let usl_path = create_simple_usl(&dir);

    let output = dashprove_cmd()
        .arg("export")
        .arg(&usl_path)
        .arg("--target")
        .arg("smtlib")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "export to smtlib failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // SMT-LIB2 export should contain s-expressions
    assert!(
        stdout.contains("set-logic")
            || stdout.contains("declare-fun")
            || stdout.contains("assert")
            || stdout.contains("check-sat")
            || stdout.contains("("),
        "SMT-LIB2 export should contain SMT-LIB2 syntax: {}",
        stdout
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_export_to_smtlib_with_output_file() {
    let dir = temp_dir("smtlib_export_file");
    let usl_path = create_simple_usl(&dir);
    let output_path = dir.join("output.smt2");

    let output = dashprove_cmd()
        .arg("export")
        .arg(&usl_path)
        .arg("--target")
        .arg("smtlib")
        .arg("-o")
        .arg(&output_path)
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "export to smtlib file failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(
        output_path.exists(),
        "SMT-LIB2 output file should be created"
    );
    let content = std::fs::read_to_string(&output_path).unwrap();
    assert!(!content.is_empty(), "SMT-LIB2 output should not be empty");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_export_to_smtlib_with_logic() {
    let dir = temp_dir("smtlib_logic_export");
    let usl_path = create_simple_usl(&dir);

    let output = dashprove_cmd()
        .arg("export")
        .arg(&usl_path)
        .arg("--target")
        .arg("smtlib:QF_LIA")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "export to smtlib:QF_LIA failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should contain the specific logic
    assert!(
        stdout.contains("QF_LIA") || stdout.contains("set-logic"),
        "SMT-LIB2 export with logic should set the logic: {}",
        stdout
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_export_smtlib_alias_smt() {
    let dir = temp_dir("smt_alias_export");
    let usl_path = create_simple_usl(&dir);

    // Test "smt" alias
    let output = dashprove_cmd()
        .arg("export")
        .arg(&usl_path)
        .arg("--target")
        .arg("smt")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "export to smt alias failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_export_smtlib2_alias() {
    let dir = temp_dir("smtlib2_alias_export");
    let usl_path = create_simple_usl(&dir);

    // Test "smtlib2" alias
    let output = dashprove_cmd()
        .arg("export")
        .arg(&usl_path)
        .arg("--target")
        .arg("smtlib2")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "export to smtlib2 alias failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    std::fs::remove_dir_all(&dir).ok();
}

// ============================================================================
// Export examples to new backends
// ============================================================================

#[test]
#[serial]
fn test_export_examples_basic_to_coq() {
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
        .arg("export")
        .arg(&example_path)
        .arg("--target")
        .arg("coq")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "export examples/usl/basic.usl to coq failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
#[serial]
fn test_export_examples_basic_to_isabelle() {
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
        .arg("export")
        .arg(&example_path)
        .arg("--target")
        .arg("isabelle")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "export examples/usl/basic.usl to isabelle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
#[serial]
fn test_export_examples_basic_to_dafny() {
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
        .arg("export")
        .arg(&example_path)
        .arg("--target")
        .arg("dafny")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "export examples/usl/basic.usl to dafny failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
#[serial]
fn test_export_examples_basic_to_smtlib() {
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
        .arg("export")
        .arg(&example_path)
        .arg("--target")
        .arg("smtlib")
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "export examples/usl/basic.usl to smtlib failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// ============================================================================
// Help test
// ============================================================================

#[test]
#[serial]
fn test_export_help() {
    let output = dashprove_cmd()
        .arg("export")
        .arg("--help")
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success(), "export --help failed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should show export-specific help
    assert!(
        stdout.contains("--target") && stdout.contains("--output"),
        "Help should mention export options: {}",
        stdout
    );
}

#[test]
#[serial]
fn test_export_help_shows_all_backends() {
    let output = dashprove_cmd()
        .arg("export")
        .arg("--help")
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success(), "export --help failed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should mention all supported backends
    assert!(
        stdout.contains("coq") && stdout.contains("isabelle") && stdout.contains("dafny"),
        "Help should mention new backends: {}",
        stdout
    );
    assert!(
        stdout.contains("smtlib"),
        "Help should mention smtlib: {}",
        stdout
    );
}
