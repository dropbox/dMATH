//! Tests for VeriNet backend

use super::*;
use crate::traits::{BackendError, HealthStatus, VerificationBackend, VerificationStatus};

#[test]
fn test_verinet_backend_id() {
    let backend = VeriNetBackend::new();
    assert_eq!(backend.id(), BackendId::VeriNet);
}

#[test]
fn test_verinet_supports_neural_properties() {
    let backend = VeriNetBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::NeuralRobustness));
}

#[test]
fn test_verinet_config_default() {
    let config = VeriNetConfig::default();
    assert_eq!(config.epsilon, 0.01);
    assert_eq!(config.strategy, SplittingStrategy::Input);
    assert_eq!(config.max_depth, 15);
}

#[test]
fn test_splitting_strategy_as_str() {
    assert_eq!(SplittingStrategy::Input.as_str(), "input");
    assert_eq!(SplittingStrategy::ReLU.as_str(), "relu");
    assert_eq!(SplittingStrategy::Adaptive.as_str(), "adaptive");
}

#[test]
fn test_generate_verinet_script() {
    let config = VeriNetConfig::default();
    let spec = create_test_spec();
    let script = script::generate_verinet_script(&spec, &config).unwrap();

    assert!(script.contains("import verinet"));
    assert!(script.contains("VERINET_RESULT_START"));
    assert!(script.contains("VERINET_RESULT_END"));
}

#[test]
fn test_parse_verinet_output_verified() {
    let stdout = r#"
VERINET_RESULT_START
{
  "verified_count": 10,
  "total_count": 10,
  "verification_rate": 1.0,
  "epsilon": 0.01,
  "strategy": "input",
  "max_depth": 15,
  "input_dim": 2,
  "output_dim": 2,
  "num_counterexamples": 0
}
VERINET_RESULT_END
VERINET_STATUS: VERIFIED
"#;

    let (status, cex) = script::parse_verinet_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(cex.is_none());
}

#[test]
fn test_parse_verinet_output_not_verified() {
    let stdout = r#"
VERINET_RESULT_START
{
  "verified_count": 3,
  "total_count": 10,
  "verification_rate": 0.3,
  "epsilon": 0.1,
  "strategy": "relu",
  "max_depth": 15,
  "input_dim": 2,
  "output_dim": 2,
  "num_counterexamples": 7,
  "counterexample": {
    "sample_index": 0,
    "original_input": [0.5, 0.5],
    "true_label": 1,
    "counterexample": [0.45, 0.55]
  }
}
VERINET_RESULT_END
VERINET_STATUS: NOT_VERIFIED
"#;

    let (status, cex) = script::parse_verinet_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    let cex = cex.expect("expected counterexample");
    assert!(!cex.failed_checks.is_empty());
    assert!(cex.failed_checks[0].description.contains("sample 0"));
    assert!(cex.witness.contains_key("epsilon"));
    assert!(cex.witness.contains_key("adversarial_input"));
}

#[test]
fn test_parse_verinet_output_error() {
    let stdout = "VERINET_ERROR: Missing dependency";
    let (status, _) = script::parse_verinet_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[tokio::test]
async fn test_verinet_health_check() {
    let backend = VeriNetBackend::new();
    let status = backend.health_check().await;
    assert!(matches!(
        status,
        HealthStatus::Healthy | HealthStatus::Unavailable { .. }
    ));
}

#[tokio::test]
async fn test_verinet_verify_returns_result_or_unavailable() {
    let backend = VeriNetBackend::new();
    let spec = create_test_spec();
    let result = backend.verify(&spec).await;

    match result {
        Ok(r) => assert_eq!(r.backend, BackendId::VeriNet),
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
