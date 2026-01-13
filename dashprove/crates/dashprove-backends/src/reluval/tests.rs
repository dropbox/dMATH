//! Tests for ReluVal backend

use super::*;
use crate::traits::{BackendError, HealthStatus, VerificationBackend, VerificationStatus};

#[test]
fn test_reluval_backend_id() {
    let backend = ReluValBackend::new();
    assert_eq!(backend.id(), BackendId::ReluVal);
}

#[test]
fn test_reluval_supports_neural_properties() {
    let backend = ReluValBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::NeuralRobustness));
}

#[test]
fn test_reluval_config_default() {
    let config = ReluValConfig::default();
    assert_eq!(config.epsilon, 0.01);
    assert_eq!(config.refinement_mode, RefinementMode::Bisection);
    assert_eq!(config.max_iterations, 1000);
    assert_eq!(config.precision, 1e-6);
}

#[test]
fn test_refinement_mode_as_str() {
    assert_eq!(RefinementMode::Bisection.as_str(), "bisection");
    assert_eq!(RefinementMode::Gradient.as_str(), "gradient");
    assert_eq!(RefinementMode::LayerWise.as_str(), "layerwise");
    assert_eq!(RefinementMode::Smear.as_str(), "smear");
}

#[test]
fn test_generate_reluval_script() {
    let config = ReluValConfig::default();
    let spec = create_test_spec();
    let script = script::generate_reluval_script(&spec, &config).unwrap();

    assert!(script.contains("import reluval"));
    assert!(script.contains("RELUVAL_RESULT_START"));
    assert!(script.contains("RELUVAL_RESULT_END"));
}

#[test]
fn test_parse_reluval_output_verified() {
    let stdout = r#"
RELUVAL_RESULT_START
{
  "verified_count": 10,
  "total_count": 10,
  "verification_rate": 1.0,
  "epsilon": 0.01,
  "refinement_mode": "bisection",
  "max_iterations": 1000,
  "precision": 1e-6,
  "total_iterations_used": 150,
  "input_dim": 2,
  "output_dim": 2,
  "num_counterexamples": 0
}
RELUVAL_RESULT_END
RELUVAL_STATUS: VERIFIED
"#;

    let (status, cex) = script::parse_reluval_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(cex.is_none());
}

#[test]
fn test_parse_reluval_output_not_verified() {
    let stdout = r#"
RELUVAL_RESULT_START
{
  "verified_count": 3,
  "total_count": 10,
  "verification_rate": 0.3,
  "epsilon": 0.1,
  "refinement_mode": "gradient",
  "max_iterations": 1000,
  "precision": 1e-6,
  "total_iterations_used": 800,
  "input_dim": 2,
  "output_dim": 2,
  "num_counterexamples": 7,
  "counterexample": {
    "sample_index": 0,
    "original_input": [0.5, 0.5],
    "true_label": 1,
    "counterexample": [0.45, 0.55],
    "iterations_used": 30
  }
}
RELUVAL_RESULT_END
RELUVAL_STATUS: NOT_VERIFIED
"#;

    let (status, cex) = script::parse_reluval_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    let cex = cex.expect("expected counterexample");
    assert!(!cex.failed_checks.is_empty());
    assert!(cex.failed_checks[0].description.contains("sample 0"));
    assert!(cex.witness.contains_key("epsilon"));
    assert!(cex.witness.contains_key("adversarial_input"));
}

#[test]
fn test_parse_reluval_output_error() {
    let stdout = "RELUVAL_ERROR: Missing dependency";
    let (status, _) = script::parse_reluval_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[tokio::test]
async fn test_reluval_health_check() {
    let backend = ReluValBackend::new();
    let status = backend.health_check().await;
    assert!(matches!(
        status,
        HealthStatus::Healthy | HealthStatus::Unavailable { .. }
    ));
}

#[tokio::test]
async fn test_reluval_verify_returns_result_or_unavailable() {
    let backend = ReluValBackend::new();
    let spec = create_test_spec();
    let result = backend.verify(&spec).await;

    match result {
        Ok(r) => assert_eq!(r.backend, BackendId::ReluVal),
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
