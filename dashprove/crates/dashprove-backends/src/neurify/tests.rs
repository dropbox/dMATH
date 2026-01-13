//! Tests for Neurify backend

use super::*;
use crate::traits::{BackendError, HealthStatus, VerificationBackend, VerificationStatus};

#[test]
fn test_neurify_backend_id() {
    let backend = NeurifyBackend::new();
    assert_eq!(backend.id(), BackendId::Neurify);
}

#[test]
fn test_neurify_supports_neural_properties() {
    let backend = NeurifyBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::NeuralRobustness));
}

#[test]
fn test_neurify_config_default() {
    let config = NeurifyConfig::default();
    assert_eq!(config.epsilon, 0.01);
    assert_eq!(config.split_method, SplitMethod::Gradient);
    assert_eq!(config.max_splits, 5000);
    assert!(config.use_symbolic);
}

#[test]
fn test_split_method_as_str() {
    assert_eq!(SplitMethod::Gradient.as_str(), "gradient");
    assert_eq!(SplitMethod::Input.as_str(), "input");
    assert_eq!(SplitMethod::ReLU.as_str(), "relu");
    assert_eq!(SplitMethod::Adaptive.as_str(), "adaptive");
}

#[test]
fn test_generate_neurify_script() {
    let config = NeurifyConfig::default();
    let spec = create_test_spec();
    let script = script::generate_neurify_script(&spec, &config).unwrap();

    assert!(script.contains("import neurify"));
    assert!(script.contains("NEURIFY_RESULT_START"));
    assert!(script.contains("NEURIFY_RESULT_END"));
}

#[test]
fn test_parse_neurify_output_verified() {
    let stdout = r#"
NEURIFY_RESULT_START
{
  "verified_count": 10,
  "total_count": 10,
  "verification_rate": 1.0,
  "epsilon": 0.01,
  "split_method": "gradient",
  "max_splits": 5000,
  "use_symbolic": "True",
  "total_splits_used": 200,
  "input_dim": 2,
  "output_dim": 2,
  "num_counterexamples": 0
}
NEURIFY_RESULT_END
NEURIFY_STATUS: VERIFIED
"#;

    let (status, cex) = script::parse_neurify_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(cex.is_none());
}

#[test]
fn test_parse_neurify_output_not_verified() {
    let stdout = r#"
NEURIFY_RESULT_START
{
  "verified_count": 3,
  "total_count": 10,
  "verification_rate": 0.3,
  "epsilon": 0.1,
  "split_method": "input",
  "max_splits": 5000,
  "use_symbolic": "True",
  "total_splits_used": 1500,
  "input_dim": 2,
  "output_dim": 2,
  "num_counterexamples": 7,
  "counterexample": {
    "sample_index": 0,
    "original_input": [0.5, 0.5],
    "true_label": 1,
    "counterexample": [0.45, 0.55],
    "splits_used": 50
  }
}
NEURIFY_RESULT_END
NEURIFY_STATUS: NOT_VERIFIED
"#;

    let (status, cex) = script::parse_neurify_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    let cex = cex.expect("expected counterexample");
    assert!(!cex.failed_checks.is_empty());
    assert!(cex.failed_checks[0].description.contains("sample 0"));
    assert!(cex.witness.contains_key("epsilon"));
    assert!(cex.witness.contains_key("adversarial_input"));
}

#[test]
fn test_parse_neurify_output_error() {
    let stdout = "NEURIFY_ERROR: Missing dependency";
    let (status, _) = script::parse_neurify_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[tokio::test]
async fn test_neurify_health_check() {
    let backend = NeurifyBackend::new();
    let status = backend.health_check().await;
    assert!(matches!(
        status,
        HealthStatus::Healthy | HealthStatus::Unavailable { .. }
    ));
}

#[tokio::test]
async fn test_neurify_verify_returns_result_or_unavailable() {
    let backend = NeurifyBackend::new();
    let spec = create_test_spec();
    let result = backend.verify(&spec).await;

    match result {
        Ok(r) => assert_eq!(r.backend, BackendId::Neurify),
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
