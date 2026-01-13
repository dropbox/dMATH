//! Tests for DNNV backend

use super::*;
use crate::traits::{BackendError, HealthStatus, VerificationBackend, VerificationStatus};

#[test]
fn test_dnnv_backend_id() {
    let backend = DnnvBackend::new();
    assert_eq!(backend.id(), BackendId::DNNV);
}

#[test]
fn test_dnnv_supports_neural_properties() {
    let backend = DnnvBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::NeuralRobustness));
}

#[test]
fn test_dnnv_config_default() {
    let config = DnnvConfig::default();
    assert_eq!(config.epsilon, 0.01);
    assert_eq!(config.verifier, VerifierBackend::Planet);
}

#[test]
fn test_verifier_backend_as_str() {
    assert_eq!(VerifierBackend::Planet.as_str(), "planet");
    assert_eq!(VerifierBackend::Marabou.as_str(), "marabou");
    assert_eq!(VerifierBackend::Eran.as_str(), "eran");
    assert_eq!(VerifierBackend::MIPVerify.as_str(), "mipverify");
    assert_eq!(VerifierBackend::Nnenum.as_str(), "nnenum");
    assert_eq!(VerifierBackend::Neurify.as_str(), "neurify");
    assert_eq!(VerifierBackend::Reluplex.as_str(), "reluplex");
}

#[test]
fn test_generate_dnnv_script() {
    let config = DnnvConfig::default();
    let spec = create_test_spec();
    let script = script::generate_dnnv_script(&spec, &config).unwrap();

    assert!(script.contains("import dnnv"));
    assert!(script.contains("DNNV_RESULT_START"));
    assert!(script.contains("DNNV_RESULT_END"));
}

#[test]
fn test_parse_dnnv_output_verified() {
    let stdout = r#"
DNNV_RESULT_START
{
  "verified_count": 10,
  "total_count": 10,
  "verification_rate": 1.0,
  "epsilon": 0.01,
  "verifier": "planet",
  "input_dim": 2,
  "output_dim": 2,
  "num_counterexamples": 0
}
DNNV_RESULT_END
DNNV_STATUS: VERIFIED
"#;

    let (status, cex) = script::parse_dnnv_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(cex.is_none());
}

#[test]
fn test_parse_dnnv_output_not_verified() {
    let stdout = r#"
DNNV_RESULT_START
{
  "verified_count": 3,
  "total_count": 10,
  "verification_rate": 0.3,
  "epsilon": 0.1,
  "verifier": "marabou",
  "input_dim": 2,
  "output_dim": 2,
  "num_counterexamples": 7,
  "counterexample": {
    "sample_index": 0,
    "original_input": [0.5, 0.5]
  }
}
DNNV_RESULT_END
DNNV_STATUS: NOT_VERIFIED
"#;

    let (status, cex) = script::parse_dnnv_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    let cex = cex.expect("expected counterexample");
    assert!(!cex.failed_checks.is_empty());
    assert!(cex.failed_checks[0].description.contains("sample 0"));
    assert!(cex.witness.contains_key("epsilon"));
    assert!(cex.witness.contains_key("original_input"));
}

#[test]
fn test_parse_dnnv_output_error() {
    let stdout = "DNNV_ERROR: Missing dependency";
    let (status, _) = script::parse_dnnv_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[tokio::test]
async fn test_dnnv_health_check() {
    let backend = DnnvBackend::new();
    let status = backend.health_check().await;
    assert!(matches!(
        status,
        HealthStatus::Healthy | HealthStatus::Unavailable { .. }
    ));
}

#[tokio::test]
async fn test_dnnv_verify_returns_result_or_unavailable() {
    let backend = DnnvBackend::new();
    let spec = create_test_spec();
    let result = backend.verify(&spec).await;

    match result {
        Ok(r) => assert_eq!(r.backend, BackendId::DNNV),
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
