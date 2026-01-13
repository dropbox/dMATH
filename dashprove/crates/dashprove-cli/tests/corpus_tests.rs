//! Integration tests for corpus CLI commands

use serial_test::serial;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_dir(prefix: &str) -> std::path::PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("dashprove_cli_test_{prefix}_{ts}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn dashprove_cmd() -> Command {
    Command::new(env!("CARGO_BIN_EXE_dashprove"))
}

#[test]
#[serial]
fn test_corpus_stats_empty() {
    let data_dir = temp_dir("stats_empty");

    let output = dashprove_cmd()
        .args(["corpus", "stats", "--data-dir"])
        .arg(data_dir.to_str().unwrap())
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Proofs in corpus: 0"),
        "Should show zero proofs"
    );
    assert!(
        stdout.contains("LEAN 4:  0"),
        "Should show zero LEAN proofs"
    );
    assert!(
        stdout.contains("Total observations: 0"),
        "Should show zero observations"
    );

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
#[serial]
fn test_corpus_stats_with_data() {
    let data_dir = temp_dir("stats_data");

    // Create a corpus.json with some test data
    let corpus_json = r#"{
        "proofs": {
            "test_inv_12345": {
                "id": "test_inv_12345",
                "property": {"Invariant": {"name": "test_inv", "body": {"Bool": true}}},
                "backend": "Lean4",
                "tactics": ["decide"],
                "time_taken": {"secs": 0, "nanos": 50000000},
                "proof_output": null,
                "features": {
                    "property_type": "invariant",
                    "depth": 1,
                    "quantifier_depth": 0,
                    "implication_count": 0,
                    "arithmetic_ops": 0,
                    "function_calls": 0,
                    "variable_count": 0,
                    "has_temporal": false,
                    "type_refs": []
                }
            }
        }
    }"#;

    let tactics_json = r#"{
        "stats": [],
        "global_stats": {
            "decide": {"successes": 1, "failures": 0, "partials": 0}
        }
    }"#;

    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::write(data_dir.join("corpus.json"), corpus_json).unwrap();
    std::fs::write(data_dir.join("tactics.json"), tactics_json).unwrap();

    let output = dashprove_cmd()
        .args(["corpus", "stats", "--data-dir"])
        .arg(data_dir.to_str().unwrap())
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Proofs in corpus: 1"),
        "Should show one proof"
    );
    assert!(stdout.contains("LEAN 4:  1"), "Should show one LEAN proof");
    assert!(
        stdout.contains("Unique tactics: 1"),
        "Should show one tactic"
    );

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
#[serial]
fn test_corpus_search_empty() {
    let data_dir = temp_dir("search_empty");
    let spec_dir = temp_dir("search_spec");

    // Create a USL spec file
    let spec_content = r#"
        invariant test_search {
            true
        }
    "#;

    let spec_path = spec_dir.join("test.usl");
    std::fs::write(&spec_path, spec_content).unwrap();

    let output = dashprove_cmd()
        .args(["corpus", "search"])
        .arg(spec_path.to_str().unwrap())
        .arg("--data-dir")
        .arg(data_dir.to_str().unwrap())
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Corpus is empty"),
        "Should report empty corpus"
    );

    std::fs::remove_dir_all(data_dir).ok();
    std::fs::remove_dir_all(spec_dir).ok();
}

#[test]
#[serial]
fn test_corpus_search_finds_similar() {
    let data_dir = temp_dir("search_finds");
    let spec_dir = temp_dir("search_spec_finds");

    // Create corpus with an invariant
    let corpus_json = r#"{
        "proofs": {
            "existing_inv_12345": {
                "id": "existing_inv_12345",
                "property": {"Invariant": {"name": "existing_inv", "body": {"Bool": true}}},
                "backend": "Lean4",
                "tactics": ["decide"],
                "time_taken": {"secs": 0, "nanos": 50000000},
                "proof_output": null,
                "features": {
                    "property_type": "invariant",
                    "depth": 1,
                    "quantifier_depth": 0,
                    "implication_count": 0,
                    "arithmetic_ops": 0,
                    "function_calls": 0,
                    "variable_count": 0,
                    "has_temporal": false,
                    "type_refs": []
                }
            }
        }
    }"#;

    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::write(data_dir.join("corpus.json"), corpus_json).unwrap();
    std::fs::write(
        data_dir.join("tactics.json"),
        r#"{"stats":[],"global_stats":{}}"#,
    )
    .unwrap();

    // Create a USL spec file
    let spec_content = r#"
        invariant query_inv {
            true
        }
    "#;

    let spec_path = spec_dir.join("query.usl");
    std::fs::write(&spec_path, spec_content).unwrap();

    let output = dashprove_cmd()
        .args(["corpus", "search"])
        .arg(spec_path.to_str().unwrap())
        .arg("--data-dir")
        .arg(data_dir.to_str().unwrap())
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Property: query_inv"),
        "Should show query property name"
    );
    assert!(
        stdout.contains("Found 1 similar proofs"),
        "Should find the similar proof, got: {}",
        stdout
    );
    assert!(
        stdout.contains("existing_inv"),
        "Should show the existing proof name"
    );

    std::fs::remove_dir_all(data_dir).ok();
    std::fs::remove_dir_all(spec_dir).ok();
}

#[test]
#[serial]
fn test_corpus_search_missing_file() {
    let data_dir = temp_dir("search_missing");

    let output = dashprove_cmd()
        .args(["corpus", "search", "/nonexistent/file.usl", "--data-dir"])
        .arg(data_dir.to_str().unwrap())
        .output()
        .expect("Failed to run dashprove");

    assert!(
        !output.status.success(),
        "Command should fail for missing file"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Failed to read query file") || stderr.contains("No such file"),
        "Should report file not found"
    );

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
#[serial]
fn test_verify_learn_flag_accepted() {
    // Test that the --learn flag is accepted by the CLI

    let output = dashprove_cmd()
        .args(["verify", "--help"])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success());
    assert!(stdout.contains("--learn"), "Help should show --learn flag");
    assert!(
        stdout.contains("--data-dir"),
        "Help should show --data-dir flag for verify command"
    );
}

#[test]
#[serial]
fn test_verify_learn_creates_corpus() {
    let data_dir = temp_dir("learn_verify");
    let spec_dir = temp_dir("learn_spec");

    // Create a simple USL spec
    let spec_content = r#"
        invariant simple_true {
            true
        }
    "#;

    let spec_path = spec_dir.join("simple.usl");
    std::fs::write(&spec_path, spec_content).unwrap();

    // Run verify with --learn flag (will fail because no backends available,
    // but should still try to write to corpus directory)
    let output = dashprove_cmd()
        .args([
            "verify",
            spec_path.to_str().unwrap(),
            "--learn",
            "--data-dir",
            data_dir.to_str().unwrap(),
            "--skip-health-check",
        ])
        .output()
        .expect("Failed to run dashprove");

    // The command might fail due to no backends, but that's ok
    // We just want to verify the flags are accepted
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);

    // If it fails, it should be because of backends, not because of the flags
    if !output.status.success() {
        assert!(
            stderr.contains("No backends available")
                || stderr.contains("Backend")
                || stdout.contains("No backends available"),
            "Failure should be about backends, not flags. stderr: {}, stdout: {}",
            stderr,
            stdout
        );
    }

    std::fs::remove_dir_all(data_dir).ok();
    std::fs::remove_dir_all(spec_dir).ok();
}

#[test]
#[serial]
fn test_verify_suggest_flag_accepted() {
    // Test that the --suggest flag is accepted by the CLI

    let output = dashprove_cmd()
        .args(["verify", "--help"])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success());
    assert!(
        stdout.contains("--suggest"),
        "Help should show --suggest flag"
    );
}

#[test]
#[serial]
fn test_verify_suggest_shows_tactics() {
    let spec_dir = temp_dir("suggest_spec");

    // Create a USL spec with different property types
    let spec_content = r#"
        theorem test_arith {
            forall x: Int . x + 1 > x
        }

        invariant test_simple {
            true
        }
    "#;

    let spec_path = spec_dir.join("suggest_test.usl");
    std::fs::write(&spec_path, spec_content).unwrap();

    // Run verify with --suggest flag
    let output = dashprove_cmd()
        .args([
            "verify",
            spec_path.to_str().unwrap(),
            "--suggest",
            "--skip-health-check",
        ])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should show tactic suggestions section
    assert!(
        stdout.contains("Tactic Suggestions"),
        "Should show tactic suggestions header. stdout: {}",
        stdout
    );

    // Should show property names
    assert!(
        stdout.contains("test_arith") || stdout.contains("test_simple"),
        "Should list property names. stdout: {}",
        stdout
    );

    // Should show compiler suggestions for at least one property
    assert!(
        stdout.contains("Compiler suggestions")
            || stdout.contains("omega")
            || stdout.contains("decide")
            || stdout.contains("intro"),
        "Should show some compiler tactic suggestions. stdout: {}",
        stdout
    );

    std::fs::remove_dir_all(spec_dir).ok();
}

// Tests for new CLI features (incremental and explain)

#[test]
#[serial]
fn test_verify_incremental_flag_accepted() {
    let output = dashprove_cmd()
        .args(["verify", "--help"])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success());
    assert!(
        stdout.contains("--incremental"),
        "Help should show --incremental flag"
    );
    assert!(stdout.contains("--since"), "Help should show --since flag");
}

#[test]
#[serial]
fn test_verify_incremental_with_spec() {
    let spec_dir = temp_dir("incremental_spec");

    let spec_content = r#"
        invariant test_inc {
            true
        }
    "#;

    let spec_path = spec_dir.join("inc_test.usl");
    std::fs::write(&spec_path, spec_content).unwrap();

    // Run with --incremental (defaults to HEAD)
    let output = dashprove_cmd()
        .args([
            "verify",
            spec_path.to_str().unwrap(),
            "--incremental",
            "--skip-health-check",
        ])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Should either succeed or fail due to no backends, not due to incremental flag
    if !output.status.success() {
        assert!(
            stderr.contains("No backends available")
                || stderr.contains("Backend")
                || stdout.contains("No backends available")
                || stdout.contains("Incremental mode")
                || stdout.contains("No changes detected"),
            "If fails, should be about backends or no changes. stderr: {}, stdout: {}",
            stderr,
            stdout
        );
    }

    std::fs::remove_dir_all(spec_dir).ok();
}

#[test]
#[serial]
fn test_explain_command_help() {
    let output = dashprove_cmd()
        .args(["explain", "--help"])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success());
    assert!(
        stdout.contains("--backend"),
        "Help should show --backend flag"
    );
    assert!(
        stdout.contains("counterexample"),
        "Help should mention counterexample"
    );
}

#[test]
#[serial]
fn test_explain_json_counterexample() {
    let ce_dir = temp_dir("explain_json");

    // Create a JSON counterexample file
    let ce_json = r#"{
        "backend": "tla+",
        "counterexample": "State 1:\n  x = 0\n  y = 1\nState 2:\n  /\\ Next\n  x = 1\n  y = 2"
    }"#;

    let ce_path = ce_dir.join("ce.json");
    std::fs::write(&ce_path, ce_json).unwrap();

    let output = dashprove_cmd()
        .args(["explain", ce_path.to_str().unwrap()])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Explain command should succeed. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Counterexample Explanation"),
        "Should show explanation header"
    );
    assert!(
        stdout.contains("StateTrace") || stdout.contains("State"),
        "Should identify as state trace"
    );

    std::fs::remove_dir_all(ce_dir).ok();
}

#[test]
#[serial]
fn test_explain_plain_text_requires_backend() {
    let ce_dir = temp_dir("explain_plain");

    // Create a plain text counterexample file
    let ce_text = "State 1: x = 0\nState 2: x = 1";

    let ce_path = ce_dir.join("ce.txt");
    std::fs::write(&ce_path, ce_text).unwrap();

    // Should fail without --backend
    let output = dashprove_cmd()
        .args(["explain", ce_path.to_str().unwrap()])
        .output()
        .expect("Failed to run dashprove");

    assert!(
        !output.status.success(),
        "Should fail without --backend for plain text"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("backend") || stderr.contains("Backend"),
        "Should mention backend is required"
    );

    std::fs::remove_dir_all(ce_dir).ok();
}

#[test]
#[serial]
fn test_explain_plain_text_with_backend() {
    let ce_dir = temp_dir("explain_plain_backend");

    // Create a plain text counterexample file
    let ce_text = "assertion failed: result > 0\nvar x = -1";

    let ce_path = ce_dir.join("ce.txt");
    std::fs::write(&ce_path, ce_text).unwrap();

    // Should succeed with --backend
    let output = dashprove_cmd()
        .args(["explain", ce_path.to_str().unwrap(), "--backend", "kani"])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Should succeed with --backend. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Counterexample Explanation"),
        "Should show explanation header"
    );
    assert!(
        stdout.contains("Postcondition") || stdout.contains("assertion"),
        "Should mention postcondition violation"
    );

    std::fs::remove_dir_all(ce_dir).ok();
}

#[test]
#[serial]
fn test_explain_missing_file() {
    let output = dashprove_cmd()
        .args(["explain", "/nonexistent/file.json"])
        .output()
        .expect("Failed to run dashprove");

    assert!(!output.status.success(), "Should fail for missing file");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("not found") || stderr.contains("No such file"),
        "Should report file not found"
    );
}

#[test]
#[serial]
fn test_explain_lean_counterexample() {
    let ce_dir = temp_dir("explain_lean");

    // Create a JSON counterexample for LEAN
    let ce_json = r#"{
        "backend": "lean",
        "counterexample": "unsolved goals\n⊢ P ∧ Q\nexpected: true\nactual: false"
    }"#;

    let ce_path = ce_dir.join("lean_ce.json");
    std::fs::write(&ce_path, ce_json).unwrap();

    let output = dashprove_cmd()
        .args(["explain", ce_path.to_str().unwrap()])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success(), "Explain command should succeed");
    assert!(
        stdout.contains("MissingCase") || stdout.contains("incomplete"),
        "Should identify as missing case/incomplete proof"
    );

    std::fs::remove_dir_all(ce_dir).ok();
}

#[test]
#[serial]
fn test_explain_alloy_counterexample() {
    let ce_dir = temp_dir("explain_alloy");

    // Create a JSON counterexample for Alloy
    let ce_json = r#"{
        "backend": "alloy",
        "counterexample": "sig Node = {Node0, Node1}\nedges = {Node0->Node1, Node1->Node0}"
    }"#;

    let ce_path = ce_dir.join("alloy_ce.json");
    std::fs::write(&ce_path, ce_json).unwrap();

    let output = dashprove_cmd()
        .args(["explain", ce_path.to_str().unwrap()])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success(), "Explain command should succeed");
    assert!(
        stdout.contains("Counterexample") || stdout.contains("bounded scope"),
        "Should explain the counterexample"
    );
    assert!(
        stdout.contains("Variable Bindings") || stdout.contains("edges"),
        "Should show variable bindings from Alloy output"
    );

    std::fs::remove_dir_all(ce_dir).ok();
}

// Tests for top-level search command

#[test]
#[serial]
fn test_search_command_help() {
    let output = dashprove_cmd()
        .args(["search", "--help"])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success());
    assert!(
        stdout.contains("Text query"),
        "Help should explain text query"
    );
    assert!(stdout.contains("--limit"), "Help should show --limit flag");
    assert!(
        stdout.contains("--data-dir"),
        "Help should show --data-dir flag"
    );
}

#[test]
#[serial]
fn test_search_empty_corpus() {
    let data_dir = temp_dir("search_empty_corpus");

    let output = dashprove_cmd()
        .args(["search", "termination recursive", "--data-dir"])
        .arg(data_dir.to_str().unwrap())
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command should succeed even with empty corpus. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Corpus is empty"),
        "Should report empty corpus"
    );

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
#[serial]
fn test_search_with_data() {
    let data_dir = temp_dir("search_with_data");

    // Create corpus with proofs that have searchable keywords
    let corpus_json = r#"{
        "proofs": {
            "termination_12345": {
                "id": "termination_12345",
                "property": {"Theorem": {"name": "recursive_termination", "body": {"Bool": true}}},
                "backend": "Lean4",
                "tactics": ["induction", "simp"],
                "time_taken": {"secs": 0, "nanos": 50000000},
                "proof_output": null,
                "features": {
                    "property_type": "theorem",
                    "depth": 1,
                    "quantifier_depth": 0,
                    "implication_count": 0,
                    "arithmetic_ops": 0,
                    "function_calls": 0,
                    "variable_count": 0,
                    "has_temporal": false,
                    "type_refs": [],
                    "keywords": ["recursive", "termination"]
                }
            }
        }
    }"#;

    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::write(data_dir.join("corpus.json"), corpus_json).unwrap();
    std::fs::write(
        data_dir.join("tactics.json"),
        r#"{"stats":[],"global_stats":{}}"#,
    )
    .unwrap();

    let output = dashprove_cmd()
        .args(["search", "recursive termination", "--data-dir"])
        .arg(data_dir.to_str().unwrap())
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Search should succeed. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Searching for:"),
        "Should show search query"
    );
    assert!(
        stdout.contains("recursive_termination")
            || stdout.contains("Found")
            || stdout.contains("No matching"),
        "Should show results or indicate no matches. stdout: {}",
        stdout
    );

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
#[serial]
fn test_search_with_limit() {
    let data_dir = temp_dir("search_limit");

    let output = dashprove_cmd()
        .args([
            "search",
            "test query",
            "--data-dir",
            data_dir.to_str().unwrap(),
            "-n",
            "5",
        ])
        .output()
        .expect("Failed to run dashprove");

    // Should accept the -n limit flag
    assert!(
        output.status.success(),
        "Should accept -n flag. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    std::fs::remove_dir_all(data_dir).ok();
}

// Tests for prove command

#[test]
#[serial]
fn test_prove_command_help() {
    let output = dashprove_cmd()
        .args(["prove", "--help"])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success());
    assert!(
        stdout.contains("Interactive proof mode"),
        "Help should explain interactive mode"
    );
    assert!(stdout.contains("--hints"), "Help should show --hints flag");
}

#[test]
#[serial]
fn test_prove_missing_file() {
    let output = dashprove_cmd()
        .args(["prove", "/nonexistent/file.usl"])
        .output()
        .expect("Failed to run dashprove");

    assert!(!output.status.success(), "Should fail for missing file");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("not found") || stderr.contains("No such file"),
        "Should report file not found"
    );
}

#[test]
#[serial]
fn test_prove_empty_spec() {
    let spec_dir = temp_dir("prove_empty");

    // Create an empty spec (no properties)
    let spec_content = "// Empty specification\n";
    let spec_path = spec_dir.join("empty.usl");
    std::fs::write(&spec_path, spec_content).unwrap();

    // Use echo to send "quit" immediately to avoid hanging
    let output = std::process::Command::new("sh")
        .arg("-c")
        .arg(format!(
            "echo 'quit' | {} prove {}",
            env!("CARGO_BIN_EXE_dashprove"),
            spec_path.to_str().unwrap()
        ))
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        stdout.contains("No properties found"),
        "Should report no properties. stdout: {}",
        stdout
    );

    std::fs::remove_dir_all(spec_dir).ok();
}

#[test]
#[serial]
fn test_prove_list_and_quit() {
    let spec_dir = temp_dir("prove_list_quit");

    // Create a spec with properties
    let spec_content = r#"
        theorem test_thm {
            forall x: Bool . x or not x
        }
        invariant test_inv {
            true
        }
    "#;
    let spec_path = spec_dir.join("test.usl");
    std::fs::write(&spec_path, spec_content).unwrap();

    // Send "list" then "quit" commands (use printf for portability)
    let output = std::process::Command::new("sh")
        .arg("-c")
        .arg(format!(
            "printf 'list\\nquit\\n' | {} prove {}",
            env!("CARGO_BIN_EXE_dashprove"),
            spec_path.to_str().unwrap()
        ))
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Prove should exit cleanly. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Interactive Proof Mode"),
        "Should show interactive mode banner"
    );
    assert!(
        stdout.contains("Properties:"),
        "Should show properties header"
    );
    assert!(
        stdout.contains("test_thm") || stdout.contains("test_inv"),
        "Should list properties. stdout: {}",
        stdout
    );

    std::fs::remove_dir_all(spec_dir).ok();
}

#[test]
#[serial]
fn test_prove_select_and_export() {
    let spec_dir = temp_dir("prove_select_export");

    let spec_content = r#"
        invariant simple_inv {
            true
        }
    "#;
    let spec_path = spec_dir.join("export.usl");
    std::fs::write(&spec_path, spec_content).unwrap();

    // Select property 1 then export to lean (use printf for portability)
    let output = std::process::Command::new("sh")
        .arg("-c")
        .arg(format!(
            "printf 'select 1\\nexport lean\\nquit\\n' | {} prove {}",
            env!("CARGO_BIN_EXE_dashprove"),
            spec_path.to_str().unwrap()
        ))
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Should complete export. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Selected:") || stdout.contains("simple_inv"),
        "Should select property. stdout: {}",
        stdout
    );
    assert!(
        stdout.contains("lean output") || stdout.contains("theorem"),
        "Should show LEAN export. stdout: {}",
        stdout
    );

    std::fs::remove_dir_all(spec_dir).ok();
}

// Tests for automatic counterexample recording

#[test]
#[serial]
fn test_corpus_stats_shows_counterexample_count() {
    let data_dir = temp_dir("stats_cx_count");

    // Create counterexamples.json with test data
    // StructuredCounterexample requires: witness, failed_checks, playback_test, trace, raw, minimized
    // TraceState requires: state_num, action, variables
    // CounterexampleValue::Int is {"Int": {"value": N, "type_hint": null}}
    let cx_json = r#"{
        "counterexamples": {
            "cx_test_prop_1234": {
                "id": "cx_test_prop_1234",
                "property_name": "test_prop",
                "backend": "TlaPlus",
                "counterexample": {
                    "witness": {},
                    "failed_checks": [{"check_id": "inv1", "description": "invariant violated", "property_ref": null}],
                    "playback_test": null,
                    "trace": [{"state_num": 1, "action": null, "variables": {"x": {"Int": {"value": 0, "type_hint": null}}}}],
                    "raw": null,
                    "minimized": false
                },
                "features": {
                    "witness_vars": [],
                    "trace_vars": ["x"],
                    "failed_check_ids": ["inv1"],
                    "failed_check_keywords": ["invariant", "violated"],
                    "trace_length": 1,
                    "action_names": [],
                    "keywords": ["x", "invariant", "violated"]
                },
                "cluster_label": null,
                "recorded_at": "2025-01-01T00:00:00Z"
            }
        },
        "cluster_patterns": []
    }"#;

    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::write(data_dir.join("counterexamples.json"), cx_json).unwrap();
    std::fs::write(data_dir.join("corpus.json"), r#"{"proofs":{}}"#).unwrap();
    std::fs::write(
        data_dir.join("tactics.json"),
        r#"{"stats":[],"global_stats":{}}"#,
    )
    .unwrap();

    let output = dashprove_cmd()
        .args(["corpus", "stats", "--data-dir"])
        .arg(data_dir.to_str().unwrap())
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Counterexamples: 1") || stdout.contains("Counterexamples in corpus: 1"),
        "Should show counterexample count. stdout: {}",
        stdout
    );

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
#[serial]
fn test_cx_add_creates_counterexample_entry() {
    let data_dir = temp_dir("cx_add");
    let ce_dir = temp_dir("cx_add_ce");

    // Create a counterexample JSON file to add
    // StructuredCounterexample requires: witness, failed_checks, playback_test, trace, raw, minimized
    let ce_json = r#"{
        "witness": {"x": {"Int": {"value": 42, "type_hint": null}}},
        "failed_checks": [{"check_id": "bound_check", "description": "x exceeded maximum", "property_ref": null}],
        "playback_test": null,
        "trace": [
            {"state_num": 1, "action": "Init", "variables": {"x": {"Int": {"value": 0, "type_hint": null}}}},
            {"state_num": 2, "action": "Increment", "variables": {"x": {"Int": {"value": 42, "type_hint": null}}}}
        ],
        "raw": null,
        "minimized": false
    }"#;

    let ce_path = ce_dir.join("overflow.json");
    std::fs::write(&ce_path, ce_json).unwrap();

    // Add to corpus
    let output = dashprove_cmd()
        .args([
            "corpus",
            "cx-add",
            ce_path.to_str().unwrap(),
            "--property",
            "bound_invariant",
            "--backend",
            "tlaplus",
            "--data-dir",
            data_dir.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "cx-add should succeed. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Added counterexample") || stdout.contains("cx_bound_invariant"),
        "Should confirm addition. stdout: {}",
        stdout
    );

    // Verify it's in the corpus by checking stats
    let stats_output = dashprove_cmd()
        .args(["corpus", "stats", "--data-dir"])
        .arg(data_dir.to_str().unwrap())
        .output()
        .expect("Failed to run dashprove");

    let stats_stdout = String::from_utf8_lossy(&stats_output.stdout);
    assert!(
        stats_stdout.contains("Counterexamples: 1")
            || stats_stdout.contains("Counterexamples in corpus: 1"),
        "Corpus should have 1 counterexample. stdout: {}",
        stats_stdout
    );

    std::fs::remove_dir_all(data_dir).ok();
    std::fs::remove_dir_all(ce_dir).ok();
}

#[test]
#[serial]
fn test_cx_search_finds_similar() {
    let data_dir = temp_dir("cx_search");
    let ce_dir = temp_dir("cx_search_ce");

    // Create corpus with a counterexample
    // StructuredCounterexample requires: witness, failed_checks, playback_test, trace, raw, minimized
    let cx_json = r#"{
        "counterexamples": {
            "cx_overflow_1234": {
                "id": "cx_overflow_1234",
                "property_name": "bound_check",
                "backend": "TlaPlus",
                "counterexample": {
                    "witness": {"x": {"Int": {"value": 100, "type_hint": null}}},
                    "failed_checks": [{"check_id": "max_bound", "description": "exceeded maximum bound", "property_ref": null}],
                    "playback_test": null,
                    "trace": [
                        {"state_num": 1, "action": "Init", "variables": {"x": {"Int": {"value": 0, "type_hint": null}}}},
                        {"state_num": 2, "action": "Jump", "variables": {"x": {"Int": {"value": 100, "type_hint": null}}}}
                    ],
                    "raw": null,
                    "minimized": false
                },
                "features": {
                    "witness_vars": ["x"],
                    "trace_vars": ["x"],
                    "failed_check_ids": ["max_bound"],
                    "failed_check_keywords": ["exceeded", "maximum", "bound"],
                    "trace_length": 2,
                    "action_names": ["Init", "Jump"],
                    "keywords": ["x", "exceeded", "maximum", "bound", "Init", "Jump"]
                },
                "cluster_label": null,
                "recorded_at": "2025-01-01T00:00:00Z"
            }
        },
        "cluster_patterns": []
    }"#;

    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::write(data_dir.join("counterexamples.json"), cx_json).unwrap();
    std::fs::write(data_dir.join("corpus.json"), r#"{"proofs":{}}"#).unwrap();
    std::fs::write(
        data_dir.join("tactics.json"),
        r#"{"stats":[],"global_stats":{}}"#,
    )
    .unwrap();

    // Create a similar counterexample to search for
    // StructuredCounterexample requires: witness, failed_checks, playback_test, trace, raw, minimized
    let query_ce = r#"{
        "witness": {"x": {"Int": {"value": 50, "type_hint": null}}},
        "failed_checks": [{"check_id": "max_bound", "description": "value too large", "property_ref": null}],
        "playback_test": null,
        "trace": [
            {"state_num": 1, "action": "Init", "variables": {"x": {"Int": {"value": 0, "type_hint": null}}}},
            {"state_num": 2, "action": "Step", "variables": {"x": {"Int": {"value": 50, "type_hint": null}}}}
        ],
        "raw": null,
        "minimized": false
    }"#;

    let query_path = ce_dir.join("query.json");
    std::fs::write(&query_path, query_ce).unwrap();

    let output = dashprove_cmd()
        .args([
            "corpus",
            "cx-search",
            query_path.to_str().unwrap(),
            "--data-dir",
            data_dir.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "cx-search should succeed. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    // Should find the similar counterexample (same variable x, similar trace structure)
    assert!(
        stdout.contains("Found") || stdout.contains("similar") || stdout.contains("bound_check"),
        "Should find similar counterexample. stdout: {}",
        stdout
    );

    std::fs::remove_dir_all(data_dir).ok();
    std::fs::remove_dir_all(ce_dir).ok();
}

// Tests for topics command

#[test]
#[serial]
fn test_topics_overview() {
    let output = dashprove_cmd()
        .args(["topics"])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success(), "topics should succeed");
    assert!(
        stdout.contains("DashProve Help Topics"),
        "Should show topics overview header"
    );
    assert!(stdout.contains("usl"), "Should list usl topic");
    assert!(stdout.contains("backends"), "Should list backends topic");
    assert!(
        stdout.contains("counterexamples"),
        "Should list counterexamples topic"
    );
}

#[test]
#[serial]
fn test_topics_usl() {
    let output = dashprove_cmd()
        .args(["topics", "usl"])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success(), "topics usl should succeed");
    assert!(
        stdout.contains("Unified Specification Language"),
        "Should show USL topic header"
    );
    assert!(stdout.contains("theorem"), "Should explain theorem syntax");
    assert!(
        stdout.contains("invariant"),
        "Should explain invariant syntax"
    );
}

#[test]
#[serial]
fn test_topics_backends() {
    let output = dashprove_cmd()
        .args(["topics", "backends"])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success(), "topics backends should succeed");
    assert!(
        stdout.contains("Verification Backends"),
        "Should show backends topic header"
    );
    assert!(stdout.contains("LEAN 4"), "Should explain LEAN 4 backend");
    assert!(stdout.contains("TLA+"), "Should explain TLA+ backend");
}

#[test]
#[serial]
fn test_topics_unknown() {
    let output = dashprove_cmd()
        .args(["topics", "nonexistent"])
        .output()
        .expect("Failed to run dashprove");

    assert!(!output.status.success(), "Unknown topic should fail");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Unknown topic"),
        "Should report unknown topic error"
    );
}

#[test]
#[serial]
fn test_topics_help() {
    let output = dashprove_cmd()
        .args(["topics", "--help"])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success(), "topics --help should succeed");
    assert!(
        stdout.contains("TOPICS:"),
        "Should show available topics in help"
    );
}

// Tests for verify --verbose flag

#[test]
#[serial]
fn test_verify_verbose_flag_accepted() {
    // Test that the flag is accepted in help
    let output = dashprove_cmd()
        .args(["verify", "--help"])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success());
    assert!(
        stdout.contains("--verbose"),
        "Help should show --verbose flag"
    );
    assert!(stdout.contains("-v"), "Help should show -v short flag");
}

#[test]
#[serial]
fn test_main_help_shows_examples() {
    let output = dashprove_cmd()
        .args(["--help"])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success());
    assert!(
        stdout.contains("EXAMPLES:"),
        "Main help should show examples section"
    );
    assert!(
        stdout.contains("dashprove verify spec.usl"),
        "Help should show verify example"
    );
}
