//! Tests for MNBaB backend

use super::*;
use crate::traits::{BackendError, HealthStatus, VerificationBackend, VerificationStatus};

#[test]
fn test_mnbab_backend_id() {
    let backend = MNBaBBackend::new();
    assert_eq!(backend.id(), BackendId::MNBaB);
}

#[test]
fn test_mnbab_supports_neural_properties() {
    let backend = MNBaBBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::NeuralRobustness));
}

#[test]
fn test_mnbab_config_default() {
    let config = MNBaBConfig::default();
    assert_eq!(config.epsilon, 0.01);
    assert_eq!(config.branching_strategy, BranchingStrategy::Score);
    assert_eq!(config.multi_neuron_count, 3);
    assert_eq!(config.max_branches, 10000);
}

#[test]
fn test_branching_strategy_as_str() {
    assert_eq!(BranchingStrategy::Score.as_str(), "score");
    assert_eq!(BranchingStrategy::FSB.as_str(), "fsb");
    assert_eq!(BranchingStrategy::Babsr.as_str(), "babsr");
    assert_eq!(BranchingStrategy::Input.as_str(), "input");
}

#[test]
fn test_generate_mnbab_script() {
    let config = MNBaBConfig::default();
    let spec = create_test_spec();
    let script = script::generate_mnbab_script(&spec, &config).unwrap();

    assert!(script.contains("import mnbab"));
    assert!(script.contains("MNBAB_RESULT_START"));
    assert!(script.contains("MNBAB_RESULT_END"));
}

#[test]
fn test_parse_mnbab_output_verified() {
    let stdout = r#"
MNBAB_RESULT_START
{
  "verified_count": 10,
  "total_count": 10,
  "verification_rate": 1.0,
  "epsilon": 0.01,
  "branching_strategy": "score",
  "multi_neuron_count": 3,
  "max_branches": 10000,
  "total_branches_explored": 500,
  "input_dim": 2,
  "output_dim": 2,
  "num_counterexamples": 0
}
MNBAB_RESULT_END
MNBAB_STATUS: VERIFIED
"#;

    let (status, cex) = script::parse_mnbab_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(cex.is_none());
}

#[test]
fn test_parse_mnbab_output_not_verified() {
    let stdout = r#"
MNBAB_RESULT_START
{
  "verified_count": 3,
  "total_count": 10,
  "verification_rate": 0.3,
  "epsilon": 0.1,
  "branching_strategy": "fsb",
  "multi_neuron_count": 3,
  "max_branches": 10000,
  "total_branches_explored": 2500,
  "input_dim": 2,
  "output_dim": 2,
  "num_counterexamples": 7,
  "counterexample": {
    "sample_index": 0,
    "original_input": [0.5, 0.5],
    "true_label": 1,
    "counterexample": [0.45, 0.55],
    "branches_explored": 100
  }
}
MNBAB_RESULT_END
MNBAB_STATUS: NOT_VERIFIED
"#;

    let (status, cex) = script::parse_mnbab_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    let cex = cex.expect("expected counterexample");
    assert!(!cex.failed_checks.is_empty());
    assert!(cex.failed_checks[0].description.contains("sample 0"));
    assert!(cex.witness.contains_key("epsilon"));
    assert!(cex.witness.contains_key("adversarial_input"));
}

#[test]
fn test_parse_mnbab_output_error() {
    let stdout = "MNBAB_ERROR: Missing dependency";
    let (status, _) = script::parse_mnbab_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[tokio::test]
async fn test_mnbab_health_check() {
    let backend = MNBaBBackend::new();
    let status = backend.health_check().await;
    assert!(matches!(
        status,
        HealthStatus::Healthy | HealthStatus::Unavailable { .. }
    ));
}

#[tokio::test]
async fn test_mnbab_verify_returns_result_or_unavailable() {
    let backend = MNBaBBackend::new();
    let spec = create_test_spec();
    let result = backend.verify(&spec).await;

    match result {
        Ok(r) => assert_eq!(r.backend, BackendId::MNBaB),
        Err(BackendError::Unavailable(_)) => {}
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}

fn create_test_spec() -> dashprove_usl::typecheck::TypedSpec {
    use dashprove_usl::ast::Spec;
    use dashprove_usl::typecheck::TypedSpec;
    use std::collections::HashMap;

    TypedSpec {
        spec: Spec::default(),
        type_info: HashMap::new(),
    }
}
