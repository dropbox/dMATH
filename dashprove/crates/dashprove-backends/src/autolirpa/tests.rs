//! Tests for Auto-LiRPA backend

use super::*;
use crate::traits::{BackendError, HealthStatus, VerificationBackend, VerificationStatus};

#[test]
fn test_autolirpa_backend_id() {
    let backend = AutoLirpaBackend::new();
    assert_eq!(backend.id(), BackendId::AutoLiRPA);
}

#[test]
fn test_autolirpa_supports_neural_properties() {
    let backend = AutoLirpaBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::NeuralRobustness));
}

#[test]
fn test_autolirpa_config_default() {
    let config = AutoLirpaConfig::default();
    assert_eq!(config.epsilon, 0.01);
    assert_eq!(config.bound_method, BoundMethod::IBP);
    assert!(!config.use_gpu);
}

#[test]
fn test_autolirpa_config_crown() {
    let config = AutoLirpaConfig::crown();
    assert_eq!(config.bound_method, BoundMethod::CROWN);
}

#[test]
fn test_autolirpa_config_alpha_crown() {
    let config = AutoLirpaConfig::alpha_crown();
    assert_eq!(config.bound_method, BoundMethod::AlphaCrown);
    assert_eq!(config.opt_iterations, 100);
}

#[test]
fn test_bound_method_as_str() {
    assert_eq!(BoundMethod::IBP.as_str(), "IBP");
    assert_eq!(BoundMethod::CROWN.as_str(), "CROWN");
    assert_eq!(BoundMethod::IBPCrown.as_str(), "IBP+CROWN");
    assert_eq!(BoundMethod::ForwardCrown.as_str(), "Forward+CROWN");
    assert_eq!(BoundMethod::AlphaCrown.as_str(), "alpha-CROWN");
}

#[test]
fn test_generate_autolirpa_script() {
    let config = AutoLirpaConfig::default();
    let spec = create_test_spec();
    let script = script::generate_autolirpa_script(&spec, &config).unwrap();

    assert!(script.contains("from auto_LiRPA"));
    assert!(script.contains("AUTOLIRPA_RESULT_START"));
    assert!(script.contains("AUTOLIRPA_RESULT_END"));
}

#[test]
fn test_parse_autolirpa_output_verified() {
    let stdout = r#"
AUTOLIRPA_INFO: Using synthetic test network
AUTOLIRPA_RESULT_START
{
  "verified_count": 10,
  "total_count": 10,
  "verification_rate": 1.0,
  "epsilon": 0.01,
  "bound_method": "IBP",
  "input_dim": 2,
  "output_dim": 2,
  "num_counterexamples": 0
}
AUTOLIRPA_RESULT_END
AUTOLIRPA_SUMMARY: Verified 10/10 (100.00%)
AUTOLIRPA_STATUS: VERIFIED
"#;

    let (status, cex) = script::parse_autolirpa_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(cex.is_none());
}

#[test]
fn test_parse_autolirpa_output_not_verified() {
    let stdout = r#"
AUTOLIRPA_RESULT_START
{
  "verified_count": 3,
  "total_count": 10,
  "verification_rate": 0.3,
  "epsilon": 0.1,
  "bound_method": "CROWN",
  "input_dim": 2,
  "output_dim": 2,
  "num_counterexamples": 7,
  "counterexample": {
    "sample_index": 0,
    "original_input": [0.5, 0.5],
    "true_label": 1,
    "lower_bounds": [-0.1, 0.2],
    "upper_bounds": [0.3, 0.4]
  }
}
AUTOLIRPA_RESULT_END
AUTOLIRPA_STATUS: NOT_VERIFIED
"#;

    let (status, cex) = script::parse_autolirpa_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    assert!(cex.is_some());

    let cex = cex.unwrap();
    assert!(cex.witness.contains_key("original_input"));
    assert!(cex.witness.contains_key("lower_bounds"));
    assert!(cex.witness.contains_key("upper_bounds"));
}

#[test]
fn test_parse_autolirpa_output_error() {
    let stdout = "AUTOLIRPA_ERROR: Missing dependency";
    let (status, _) = script::parse_autolirpa_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[tokio::test]
async fn test_autolirpa_health_check() {
    let backend = AutoLirpaBackend::new();
    let status = backend.health_check().await;
    assert!(matches!(
        status,
        HealthStatus::Healthy | HealthStatus::Unavailable { .. }
    ));
}

#[tokio::test]
async fn test_autolirpa_verify_returns_result_or_unavailable() {
    let backend = AutoLirpaBackend::new();
    let spec = create_test_spec();
    let result = backend.verify(&spec).await;

    match result {
        Ok(r) => assert_eq!(r.backend, BackendId::AutoLiRPA),
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
