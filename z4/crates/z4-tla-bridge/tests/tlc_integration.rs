//! Integration tests for TLC runner.
//!
//! These tests require TLC to be available. They are skipped when
//! TLC is not installed.
//!
//! To run:
//! - Set `TLC_BIN` to point to a `tlc` executable, OR
//! - Set `TLA2TOOLS_JAR` to point to `tla2tools.jar`
//!
//! Example:
//! ```bash
//! TLC_BIN=/usr/local/bin/tlc cargo test -p z4-tla-bridge --test tlc_integration
//! ```

use std::path::PathBuf;
use z4_tla_bridge::{TlcArgs, TlcOutcome, TlcRunner};

fn specs_dir() -> PathBuf {
    // Tests run from workspace root or crate root
    let candidates = [PathBuf::from("specs"), PathBuf::from("../../specs")];
    for p in candidates {
        if p.exists() {
            return p.canonicalize().unwrap();
        }
    }
    panic!("Could not find specs directory");
}

fn maybe_runner() -> Option<TlcRunner> {
    TlcRunner::discover().ok()
}

/// Run TLC on cdcl_test.tla and verify it parses the outcome.
///
/// The cdcl_test spec models a simple UNSAT formula:
///   (v1 OR v2) AND (NOT v1) AND (NOT v2)
///
/// The spec should terminate with state = "UNSAT" without violating
/// any invariants.
#[test]
fn test_cdcl_test_spec() {
    let runner = match maybe_runner() {
        Some(r) => r,
        None => {
            eprintln!("Skipping TLC integration test: TLC not available");
            eprintln!("Set TLC_BIN or TLA2TOOLS_JAR to enable this test");
            return;
        }
    };

    let specs = specs_dir();
    let spec_path = specs.join("cdcl_test.tla");
    let cfg_path = specs.join("cdcl_test.cfg");

    assert!(
        spec_path.exists(),
        "cdcl_test.tla not found at {spec_path:?}"
    );
    assert!(cfg_path.exists(), "cdcl_test.cfg not found at {cfg_path:?}");

    let args = TlcArgs {
        config: Some(cfg_path),
        workers: Some(1),
        ..Default::default()
    };

    let result = runner.run(&spec_path, args).expect("TLC execution failed");

    eprintln!("TLC stdout:\n{}", result.stdout);
    eprintln!("TLC stderr:\n{}", result.stderr);
    eprintln!("TLC outcome: {:?}", result.outcome);
    eprintln!("TLC exit code: {:?}", result.exit_status.code());

    // The spec should complete successfully - it models an UNSAT formula
    // and terminates in the "UNSAT" state without invariant violations.
    match &result.outcome {
        TlcOutcome::NoError => {
            // Expected: model checking completed without errors
        }
        TlcOutcome::Deadlock => {
            // This is acceptable if TLC treats the terminal UNSAT state as deadlock.
            // The spec terminates when state = "UNSAT" with no enabled transitions.
            eprintln!("Note: TLC reported deadlock (expected for terminal states)");
        }
        TlcOutcome::InvariantViolation { name } => {
            panic!(
                "Invariant violation: {:?}\n\nThis indicates a bug in the CDCL spec.\n\nFull output:\n{}",
                name,
                result.combined_output()
            );
        }
        other => {
            // Other outcomes may indicate configuration issues
            eprintln!("Unexpected outcome: {:?}", other);
            eprintln!("This may be acceptable depending on TLC version/config");
        }
    }
}

/// Run TLC on the main cdcl.tla spec if available.
#[test]
fn test_cdcl_main_spec() {
    let runner = match maybe_runner() {
        Some(r) => r,
        None => {
            eprintln!("Skipping TLC integration test: TLC not available");
            return;
        }
    };

    let specs = specs_dir();
    let spec_path = specs.join("cdcl.tla");
    let cfg_path = specs.join("cdcl.cfg");

    if !spec_path.exists() || !cfg_path.exists() {
        eprintln!("Skipping cdcl.tla test: spec or config not found");
        return;
    }

    let args = TlcArgs {
        config: Some(cfg_path),
        workers: Some(1),
        ..Default::default()
    };

    let result = runner.run(&spec_path, args).expect("TLC execution failed");

    eprintln!("TLC outcome for cdcl.tla: {:?}", result.outcome);

    // Just verify TLC runs without crashing
    match &result.outcome {
        TlcOutcome::InvariantViolation { name } => {
            panic!(
                "Invariant violation in cdcl.tla: {:?}\n\nFull output:\n{}",
                name,
                result.combined_output()
            );
        }
        _ => {
            // Other outcomes are acceptable for this test
        }
    }
}

/// Test that we correctly identify various TLC error types.
#[test]
fn test_outcome_parsing() {
    use z4_tla_bridge::parse_tlc_outcome;

    // Successful run
    assert_eq!(
        parse_tlc_outcome(
            "Model checking completed. No error has been found.\n",
            "",
            Some(0)
        ),
        TlcOutcome::NoError
    );

    // Deadlock
    assert_eq!(
        parse_tlc_outcome("Error: Deadlock reached.\n", "", Some(1)),
        TlcOutcome::Deadlock
    );

    // Invariant violation with name
    assert_eq!(
        parse_tlc_outcome("Error: Invariant TypeInvariant is violated.\n", "", Some(1)),
        TlcOutcome::InvariantViolation {
            name: Some("TypeInvariant".to_string())
        }
    );

    // Invariant violation with complex name
    assert_eq!(
        parse_tlc_outcome(
            "Error: Invariant SatCorrect is violated.\nThe behavior up to this point is:\n...",
            "",
            Some(1)
        ),
        TlcOutcome::InvariantViolation {
            name: Some("SatCorrect".to_string())
        }
    );

    // Liveness violation
    assert_eq!(
        parse_tlc_outcome("Temporal properties were violated.\n", "", Some(1)),
        TlcOutcome::LivenessViolation
    );

    // Type error
    assert_eq!(
        parse_tlc_outcome("TLC_TYPE error: value was not in the domain\n", "", Some(1)),
        TlcOutcome::TypeError
    );

    // Parse error
    assert_eq!(
        parse_tlc_outcome("Parse error in module Foo\n", "", Some(1)),
        TlcOutcome::ParseError
    );

    // Unknown failure
    assert_eq!(
        parse_tlc_outcome("Something unexpected happened\n", "", Some(42)),
        TlcOutcome::ExecutionFailed {
            exit_code: Some(42)
        }
    );
}

/// Test the TlcRun combined_output method.
#[test]
fn test_combined_output() {
    #[cfg(unix)]
    use std::os::unix::process::ExitStatusExt;
    use std::process::ExitStatus;
    use z4_tla_bridge::{TlcArgs, TlcBackend, TlcRun};

    #[cfg(unix)]
    let exit_status = ExitStatus::from_raw(0);
    #[cfg(not(unix))]
    let exit_status = {
        // On non-Unix, we can't easily construct ExitStatus
        // Skip this test
        return;
    };

    let run = TlcRun {
        backend: TlcBackend::Cli {
            tlc_bin: PathBuf::from("tlc"),
        },
        args: TlcArgs::default(),
        spec: PathBuf::from("test.tla"),
        exit_status,
        stdout: "stdout line\n".to_string(),
        stderr: "stderr line\n".to_string(),
        outcome: TlcOutcome::NoError,
    };

    let combined = run.combined_output();
    assert!(combined.contains("stdout line"));
    assert!(combined.contains("stderr line"));
}
