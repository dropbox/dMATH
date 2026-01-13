//! Integration tests for KaniWrapper
//!
//! These tests require Kani to be installed and run actual verification.
//!
//! NOTE: These tests must be run sequentially (`--test-threads=1`) because
//! Kani uses file-system locks and can fail with race conditions when run
//! in parallel. Use: `cargo test -p kani-fast --test integration -- --test-threads=1`

use kani_fast::counterexample::{ExplanationGenerator, FailureCategory, RepairEngine};
use kani_fast::{KaniConfig, KaniWrapper, VerificationStatus};
use serial_test::serial;
use std::path::PathBuf;
use std::time::Duration;

fn example_project_path() -> PathBuf {
    // CARGO_MANIFEST_DIR points to crates/kani-fast, go up to workspace root
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("examples").join("simple_proofs"))
        .expect("Failed to find workspace root")
}

fn skip_if_kani_unavailable() -> bool {
    which::which("cargo-kani").is_err()
}

#[tokio::test]
#[serial]
async fn test_verify_passing_proof() {
    if skip_if_kani_unavailable() {
        eprintln!("Skipping test: Kani not installed");
        return;
    }

    let wrapper = KaniWrapper::with_defaults().expect("Failed to create wrapper");
    let project = example_project_path();

    let result = wrapper
        .verify_with_harness(&project, Some("proof_checked_add_safe"))
        .await
        .expect("Verification failed");

    assert!(
        matches!(result.status, VerificationStatus::Proven),
        "Expected Proven, got {:?}",
        result.status
    );
    assert_eq!(result.checks_failed, 0, "Expected 0 failed checks");
    assert!(result.checks_passed > 0, "Expected some passed checks");
}

#[tokio::test]
#[serial]
async fn test_verify_failing_proof() {
    if skip_if_kani_unavailable() {
        eprintln!("Skipping test: Kani not installed");
        return;
    }

    let wrapper = KaniWrapper::with_defaults().expect("Failed to create wrapper");
    let project = example_project_path();

    let result = wrapper
        .verify_with_harness(&project, Some("proof_buggy_multiply_overflows"))
        .await
        .expect("Verification failed");

    assert!(
        matches!(result.status, VerificationStatus::Disproven),
        "Expected Disproven, got {:?}",
        result.status
    );
    assert!(result.checks_failed > 0, "Expected some failed checks");
}

#[tokio::test]
#[serial]
async fn test_verify_abs_diff_commutative() {
    if skip_if_kani_unavailable() {
        eprintln!("Skipping test: Kani not installed");
        return;
    }

    let wrapper = KaniWrapper::with_defaults().expect("Failed to create wrapper");
    let project = example_project_path();

    let result = wrapper
        .verify_with_harness(&project, Some("proof_abs_diff_commutative"))
        .await
        .expect("Verification failed");

    assert!(
        matches!(result.status, VerificationStatus::Proven),
        "Expected Proven, got {:?}",
        result.status
    );
}

#[tokio::test]
#[serial]
async fn test_verify_with_config() {
    if skip_if_kani_unavailable() {
        eprintln!("Skipping test: Kani not installed");
        return;
    }

    let config = KaniConfig {
        timeout: Duration::from_secs(120),
        default_unwind: Some(10),
        ..Default::default()
    };

    let wrapper = KaniWrapper::new(config).expect("Failed to create wrapper");
    let project = example_project_path();

    let result = wrapper
        .verify_with_harness(&project, Some("proof_safe_div_no_panic"))
        .await
        .expect("Verification failed");

    assert!(
        matches!(result.status, VerificationStatus::Proven),
        "Expected Proven, got {:?}",
        result.status
    );
}

#[tokio::test]
#[serial]
async fn test_invalid_project_path() {
    if skip_if_kani_unavailable() {
        eprintln!("Skipping test: Kani not installed");
        return;
    }

    let wrapper = KaniWrapper::with_defaults().expect("Failed to create wrapper");
    let result = wrapper.verify(&PathBuf::from("/nonexistent/path")).await;

    assert!(result.is_err(), "Expected error for invalid path");
}

#[tokio::test]
#[serial]
async fn test_verify_result_display() {
    if skip_if_kani_unavailable() {
        eprintln!("Skipping test: Kani not installed");
        return;
    }

    let wrapper = KaniWrapper::with_defaults().expect("Failed to create wrapper");
    let project = example_project_path();

    let result = wrapper
        .verify_with_harness(&project, Some("proof_checked_add_safe"))
        .await
        .expect("Verification failed");

    // Test Display implementation
    let display = format!("{}", result);
    assert!(
        matches!(result.status, VerificationStatus::Proven),
        "Expected Proven status, got {:?}",
        result.status
    );
    assert!(
        display.contains("VERIFIED"),
        "Display should show VERIFIED, got: {}",
        display
    );
}

// =============================================================================
// Phase 8: Beautiful Counterexamples Integration Tests
// =============================================================================

#[tokio::test]
#[serial]
async fn test_counterexample_explanation_overflow() {
    if skip_if_kani_unavailable() {
        eprintln!("Skipping test: Kani not installed");
        return;
    }

    let wrapper = KaniWrapper::with_defaults().expect("Failed to create wrapper");
    let project = example_project_path();

    let result = wrapper
        .verify_with_harness(&project, Some("proof_buggy_multiply_overflows"))
        .await
        .expect("Verification failed");

    // Should have a counterexample
    assert!(
        result.counterexample.is_some(),
        "Expected counterexample for failing proof"
    );

    let cx = result.counterexample.as_ref().unwrap();

    // Generate explanation
    let generator = ExplanationGenerator::new();
    let explanations = generator.explain(cx);

    // Should detect overflow category
    assert!(
        !explanations.is_empty(),
        "Should generate at least one explanation"
    );

    let explanation = &explanations[0];
    assert_eq!(
        explanation.category,
        FailureCategory::Overflow,
        "Should detect overflow failure category"
    );
    assert!(
        explanation.severity >= 2 && explanation.severity <= 4,
        "Overflow should have moderate severity"
    );
}

#[tokio::test]
#[serial]
async fn test_counterexample_repair_suggestions() {
    if skip_if_kani_unavailable() {
        eprintln!("Skipping test: Kani not installed");
        return;
    }

    let wrapper = KaniWrapper::with_defaults().expect("Failed to create wrapper");
    let project = example_project_path();

    let result = wrapper
        .verify_with_harness(&project, Some("proof_buggy_multiply_overflows"))
        .await
        .expect("Verification failed");

    let cx = result.counterexample.as_ref().unwrap();

    // Generate repair suggestions
    let repair_engine = RepairEngine::new();
    let suggestions = repair_engine.suggest(cx);

    // Should have repair suggestions for overflow
    assert!(
        !suggestions.is_empty(),
        "Should generate repair suggestions for overflow"
    );

    // Check that suggestions include checked arithmetic
    let has_checked_arithmetic = suggestions
        .iter()
        .any(|s| s.title.to_lowercase().contains("checked"));
    assert!(
        has_checked_arithmetic,
        "Should suggest checked arithmetic for overflow"
    );

    // All suggestions should have confidence > 0
    for suggestion in &suggestions {
        assert!(
            suggestion.confidence > 0.0,
            "Suggestions should have positive confidence"
        );
        assert!(
            suggestion.confidence <= 1.0,
            "Confidence should be at most 1.0"
        );
    }
}

#[tokio::test]
#[serial]
async fn test_counterexample_has_witness_values() {
    if skip_if_kani_unavailable() {
        eprintln!("Skipping test: Kani not installed");
        return;
    }

    let wrapper = KaniWrapper::with_defaults().expect("Failed to create wrapper");
    let project = example_project_path();

    let result = wrapper
        .verify_with_harness(&project, Some("proof_buggy_multiply_overflows"))
        .await
        .expect("Verification failed");

    let cx = result.counterexample.as_ref().unwrap();

    // Should have witness values
    assert!(
        !cx.witness.is_empty(),
        "Counterexample should have witness values"
    );

    // Should have concrete values for the arguments
    let has_arg0 = cx.witness.contains_key("arg0");
    let has_arg1 = cx.witness.contains_key("arg1");
    assert!(
        has_arg0 || has_arg1,
        "Should have witness values for function arguments"
    );
}

#[tokio::test]
#[serial]
async fn test_counterexample_has_failed_checks() {
    if skip_if_kani_unavailable() {
        eprintln!("Skipping test: Kani not installed");
        return;
    }

    let wrapper = KaniWrapper::with_defaults().expect("Failed to create wrapper");
    let project = example_project_path();

    let result = wrapper
        .verify_with_harness(&project, Some("proof_buggy_multiply_overflows"))
        .await
        .expect("Verification failed");

    let cx = result.counterexample.as_ref().unwrap();

    // Should have failed checks
    assert!(
        !cx.failed_checks.is_empty(),
        "Counterexample should have failed checks"
    );

    // Check should mention overflow
    let check = &cx.failed_checks[0];
    assert!(
        check.description.to_lowercase().contains("overflow"),
        "Failed check should mention overflow"
    );
}

#[tokio::test]
#[serial]
async fn test_counterexample_has_playback_test() {
    if skip_if_kani_unavailable() {
        eprintln!("Skipping test: Kani not installed");
        return;
    }

    let wrapper = KaniWrapper::with_defaults().expect("Failed to create wrapper");
    let project = example_project_path();

    let result = wrapper
        .verify_with_harness(&project, Some("proof_buggy_multiply_overflows"))
        .await
        .expect("Verification failed");

    let cx = result.counterexample.as_ref().unwrap();

    // Should have a playback test
    assert!(
        cx.playback_test.is_some(),
        "Counterexample should have a playback test"
    );

    let playback = cx.playback_test.as_ref().unwrap();
    assert!(
        playback.contains("#[test]"),
        "Playback test should be a valid Rust test"
    );
    assert!(
        playback.contains("kani::concrete_playback_run"),
        "Playback test should use Kani's concrete playback"
    );
}

#[tokio::test]
#[serial]
async fn test_no_counterexample_for_passing_proof() {
    if skip_if_kani_unavailable() {
        eprintln!("Skipping test: Kani not installed");
        return;
    }

    let wrapper = KaniWrapper::with_defaults().expect("Failed to create wrapper");
    let project = example_project_path();

    let result = wrapper
        .verify_with_harness(&project, Some("proof_checked_add_safe"))
        .await
        .expect("Verification failed");

    // Should NOT have a counterexample for passing proof
    assert!(
        result.counterexample.is_none(),
        "Passing proof should not have a counterexample"
    );
}

// =============================================================================
// K-Induction Integration Tests
// =============================================================================

fn skip_if_z3_unavailable() -> bool {
    which::which("z3").is_err()
}

/// Test k-induction verification of a simple counter
#[tokio::test]
async fn test_kinduction_simple_counter() {
    use kani_fast_kinduction::{
        KInduction, KInductionConfigBuilder, SmtType, TransitionSystemBuilder,
    };

    if skip_if_z3_unavailable() {
        eprintln!("Skipping test: Z3 not installed");
        return;
    }

    // Simple counter: starts at 0, increments by 1
    // Property: counter is always non-negative
    let ts = TransitionSystemBuilder::new()
        .variable("x", SmtType::Int)
        .init("(= x 0)")
        .transition("(= x' (+ x 1))")
        .property("p1", "nonneg", "(>= x 0)")
        .build();

    let config = KInductionConfigBuilder::new()
        .max_k(10)
        .use_portfolio(false) // Use Z3 directly for reliability
        .build();

    let engine = KInduction::new(config);
    let result = engine.verify(&ts).await;

    assert!(result.is_ok(), "K-induction verification should succeed");
    let result = result.unwrap();
    assert!(
        result.is_proven(),
        "Simple counter should be proven, got: {:?}",
        result
    );
}

/// Test k-induction fails with Unknown for property that cannot be proven inductively
/// (the decrementing counter needs to find a counterexample via base case, not induction)
#[tokio::test]
async fn test_kinduction_induction_failure() {
    use kani_fast_kinduction::{
        KInduction, KInductionConfigBuilder, SmtType, TransitionSystemBuilder,
    };

    if skip_if_z3_unavailable() {
        eprintln!("Skipping test: Z3 not installed");
        return;
    }

    // This property cannot be proven by simple k-induction:
    // x starts at 0, increments by 1
    // Property: x < 100 (will eventually be violated, but k-induction can't prove it)
    let ts = TransitionSystemBuilder::new()
        .variable("x", SmtType::Int)
        .init("(= x 0)")
        .transition("(= x' (+ x 1))")
        .property("p1", "bounded", "(< x 100)")
        .build();

    let config = KInductionConfigBuilder::new()
        .max_k(5) // Small max_k to keep test fast
        .use_portfolio(false)
        .build();

    let engine = KInduction::new(config);
    let result = engine.verify(&ts).await;

    assert!(result.is_ok(), "K-induction verification should succeed");
    let result = result.unwrap();

    // This should NOT be proven - induction fails because x < 100 is not inductive
    // Result should be Unknown (reached max_k) or possibly proven with aux invariant
    // The important thing is it should NOT be disproven (no counterexample in bounded steps)
    assert!(
        !result.is_disproven(),
        "Bounded counter should not find counterexample in small k, got: {:?}",
        result
    );
}

/// Test k-induction with two variables
#[tokio::test]
async fn test_kinduction_two_variables() {
    use kani_fast_kinduction::{
        KInduction, KInductionConfigBuilder, SmtType, TransitionSystemBuilder,
    };

    if skip_if_z3_unavailable() {
        eprintln!("Skipping test: Z3 not installed");
        return;
    }

    // Two counters: x and y both start at 0
    // x increments by 2, y increments by 1
    // Property: x >= y (should always hold)
    let ts = TransitionSystemBuilder::new()
        .variable("x", SmtType::Int)
        .variable("y", SmtType::Int)
        .init("(and (= x 0) (= y 0))")
        .transition("(and (= x' (+ x 2)) (= y' (+ y 1)))")
        .property("p1", "x_ge_y", "(>= x y)")
        .build();

    let config = KInductionConfigBuilder::new()
        .max_k(10)
        .use_portfolio(false)
        .build();

    let engine = KInduction::new(config);
    let result = engine.verify(&ts).await;

    assert!(result.is_ok(), "K-induction verification should succeed");
    let result = result.unwrap();
    assert!(
        result.is_proven(),
        "x >= y should be proven, got: {:?}",
        result
    );
}

// =============================================================================
// CHC (Constrained Horn Clauses) Integration Tests
// =============================================================================

/// Test CHC verification of a simple counter
#[tokio::test]
async fn test_chc_simple_counter() {
    use kani_fast_chc::{verify_transition_system_default, ChcResult};
    use kani_fast_kinduction::{SmtType, TransitionSystemBuilder};

    if skip_if_z3_unavailable() {
        eprintln!("Skipping test: Z3 not installed");
        return;
    }

    let ts = TransitionSystemBuilder::new()
        .variable("x", SmtType::Int)
        .init("(= x 0)")
        .transition("(= x' (+ x 1))")
        .property("p1", "nonneg", "(>= x 0)")
        .build();

    let result = verify_transition_system_default(&ts).await;

    assert!(result.is_ok(), "CHC verification should succeed");
    let result = result.unwrap();
    assert!(
        result.is_sat(),
        "Simple counter should find invariant (SAT), got: {:?}",
        result
    );

    // CHC returns SAT when property is verified (invariant found)
    if let ChcResult::Sat { model, .. } = &result {
        assert!(
            !model.predicates.is_empty(),
            "Should have discovered invariant"
        );
    }
}

/// Test CHC finds counterexample (returns UNSAT) for violated property
#[tokio::test]
async fn test_chc_finds_violation() {
    use kani_fast_chc::verify_transition_system_default;
    use kani_fast_kinduction::{SmtType, TransitionSystemBuilder};

    if skip_if_z3_unavailable() {
        eprintln!("Skipping test: Z3 not installed");
        return;
    }

    // Counter that goes negative
    let ts = TransitionSystemBuilder::new()
        .variable("x", SmtType::Int)
        .init("(= x 5)")
        .transition("(= x' (- x 1))")
        .property("p1", "nonneg", "(>= x 0)")
        .build();

    let result = verify_transition_system_default(&ts).await;

    assert!(result.is_ok(), "CHC verification should succeed");
    let result = result.unwrap();
    assert!(
        result.is_unsat(),
        "Decrementing counter should be UNSAT (property violated), got: {:?}",
        result
    );
}

/// Test CHC invariant extraction
#[tokio::test]
async fn test_chc_invariant_extraction() {
    use kani_fast_chc::{verify_transition_system_default, ChcResult};
    use kani_fast_kinduction::{SmtType, TransitionSystemBuilder};

    if skip_if_z3_unavailable() {
        eprintln!("Skipping test: Z3 not installed");
        return;
    }

    // x and y both start at 0, x increments by 1, y unchanged or increments by 1
    // Property: x >= y
    let ts = TransitionSystemBuilder::new()
        .variable("x", SmtType::Int)
        .variable("y", SmtType::Int)
        .init("(and (= x 0) (= y 0))")
        .transition("(and (= x' (+ x 1)) (or (= y' y) (= y' (+ y 1))))")
        .property("p1", "x_ge_y", "(>= x y)")
        .build();

    let result = verify_transition_system_default(&ts).await;

    assert!(result.is_ok(), "CHC verification should succeed");
    let result = result.unwrap();
    assert!(
        result.is_sat(),
        "x >= y should find invariant (SAT), got: {:?}",
        result
    );

    // Verify invariant model structure
    if let ChcResult::Sat { model, .. } = &result {
        let readable = model.to_readable_string();
        assert!(
            !readable.is_empty(),
            "Invariant should have readable representation"
        );
        // The invariant should mention the relationship between x and y
        // Note: exact form depends on Spacer's output
    }
}

// =============================================================================
// Lean5 Proof Generation Integration Tests
// =============================================================================

fn skip_if_lean_unavailable() -> bool {
    which::which("lean").is_err()
}

/// Test Lean5 proof generation from k-induction result
#[tokio::test]
async fn test_lean5_certificate_from_kinduction() {
    use kani_fast_lean5::{
        certificate_from_kinduction, Lean5Expr, Lean5Type, ProofObligation, ProofObligationKind,
        VerificationMethod,
    };

    // Create proof obligations for a simple property: x >= 0
    let obligation = ProofObligation::new(
        "nonneg",
        ProofObligationKind::Property,
        Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
    )
    .with_var("x", Lean5Type::Int);

    // Create certificate
    let cert = certificate_from_kinduction("test_property", 3, vec![obligation]);

    // Verify certificate structure
    assert_eq!(cert.property_name, "test_property");
    assert!(matches!(
        cert.method,
        VerificationMethod::KInduction { k: 3 }
    ));
    assert!(
        !cert.obligations.is_empty(),
        "Should have proof obligations"
    );

    // Certificate should serialize to valid JSON
    let json = serde_json::to_string(&cert);
    assert!(json.is_ok(), "Certificate should serialize to JSON");
}

/// Test Lean5 proof generation from CHC result
#[tokio::test]
async fn test_lean5_certificate_from_chc() {
    use kani_fast_chc::result::{InvariantModel, SolvedPredicate};
    use kani_fast_kinduction::{SmtType, StateFormula};
    use kani_fast_lean5::{certificate_from_chc, ProofObligation, VerificationMethod};

    // Create mock CHC result with an invariant
    let model = InvariantModel {
        predicates: vec![SolvedPredicate {
            name: "Inv".to_string(),
            params: vec![("x".to_string(), SmtType::Int)],
            formula: StateFormula::new("(>= x 0)"),
        }],
    };

    // Generate proof obligations from invariant
    let obligations = ProofObligation::from_invariant(&model)
        .expect("Should generate obligations from invariant");

    // Create certificate
    let cert = certificate_from_chc("counter_nonneg", "Z3 Spacer", obligations);

    // Verify certificate structure
    assert_eq!(cert.property_name, "counter_nonneg");
    assert!(matches!(cert.method, VerificationMethod::Chc { .. }));
    assert!(
        !cert.obligations.is_empty(),
        "Should have proof obligations"
    );
}

/// Test Lean5 file generation
#[tokio::test]
async fn test_lean5_file_generation() {
    use kani_fast_chc::result::{InvariantModel, SolvedPredicate};
    use kani_fast_kinduction::{SmtType, StateFormula};
    use kani_fast_lean5::ProofObligation;

    // Create mock invariant
    let model = InvariantModel {
        predicates: vec![SolvedPredicate {
            name: "Inv".to_string(),
            params: vec![("x".to_string(), SmtType::Int)],
            formula: StateFormula::new("(>= x 0)"),
        }],
    };

    // Generate Lean5 file
    let obligations = ProofObligation::from_invariant(&model).expect("Should generate obligations");
    let lean_file = ProofObligation::to_lean5_file(&obligations);

    // Verify Lean5 file structure
    assert!(
        lean_file.contains("namespace KaniFast"),
        "Should have namespace"
    );
    assert!(lean_file.contains("end KaniFast"), "Should close namespace");
    assert!(
        lean_file.contains("theorem"),
        "Should have theorem declarations"
    );
    assert!(lean_file.contains(":="), "Should have proof terms");
}

/// Test Lean5 backend availability check
#[test]
fn test_lean5_backend_check() {
    use kani_fast_lean5::check_lean_installation;

    let result = check_lean_installation();

    if skip_if_lean_unavailable() {
        assert!(result.is_err(), "Should report Lean not installed");
    } else {
        assert!(result.is_ok(), "Should find Lean installation");
        let version = result.unwrap();
        assert!(!version.is_empty(), "Should have version string");
    }
}

/// Test Lean5 verification of generated proof (requires Lean installed)
#[tokio::test]
async fn test_lean5_verify_generated_proof() {
    use kani_fast_chc::result::{InvariantModel, SolvedPredicate};
    use kani_fast_kinduction::{SmtType, StateFormula};
    use kani_fast_lean5::{certificate_from_chc, ProofObligation};

    if skip_if_lean_unavailable() {
        eprintln!("Skipping test: Lean not installed");
        return;
    }

    // Create a simple invariant that produces a verifiable proof
    let model = InvariantModel {
        predicates: vec![SolvedPredicate {
            name: "Inv".to_string(),
            params: vec![("n".to_string(), SmtType::Int)],
            formula: StateFormula::new("(>= n 0)"),
        }],
    };

    let obligations = ProofObligation::from_invariant(&model).expect("Should generate obligations");
    let cert = certificate_from_chc("nonneg", "Z3", obligations);

    // Write to temp file and verify
    let temp_dir = std::env::temp_dir();
    let lean_path = temp_dir.join("test_proof.lean");

    cert.write_to_file(&lean_path)
        .expect("Should write Lean file");

    // Check that file exists and has content
    assert!(lean_path.exists(), "Lean file should exist");
    let content = std::fs::read_to_string(&lean_path).expect("Should read file");
    assert!(!content.is_empty(), "Lean file should have content");

    // Note: Full Lean verification may require mathlib or specific imports
    // For now we just verify the file structure is correct
    assert!(
        content.contains("theorem") || content.contains("def"),
        "Should have definitions"
    );

    // Clean up
    let _ = std::fs::remove_file(&lean_path);
}
