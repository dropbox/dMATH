//! Tests for NNV backend

use super::*;
use crate::traits::{BackendError, HealthStatus, VerificationBackend, VerificationStatus};

#[test]
fn test_nnv_backend_id() {
    let backend = NnvBackend::new();
    assert_eq!(backend.id(), BackendId::NNV);
}

#[test]
fn test_nnv_supports_neural_properties() {
    let backend = NnvBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::NeuralRobustness));
    assert!(supported.contains(&PropertyType::NeuralReachability));
}

#[test]
fn test_nnv_config_default() {
    let config = NnvConfig::default();
    assert_eq!(config.epsilon, 0.01);
    assert_eq!(config.method, VerificationMethod::Star);
    assert!(!config.use_gpu);
    assert_eq!(config.num_workers, 1);
}

#[test]
fn test_nnv_config_fast() {
    let config = NnvConfig::fast();
    assert_eq!(config.method, VerificationMethod::Zonotope);
}

#[test]
fn test_nnv_config_hybrid() {
    let config = NnvConfig::hybrid();
    assert_eq!(config.method, VerificationMethod::Hybrid);
}

#[test]
fn test_verification_method_as_str() {
    assert_eq!(VerificationMethod::Star.as_str(), "star");
    assert_eq!(VerificationMethod::Zonotope.as_str(), "zonotope");
    assert_eq!(VerificationMethod::Polytope.as_str(), "polytope");
    assert_eq!(VerificationMethod::Hybrid.as_str(), "hybrid");
}

#[test]
fn test_generate_nnv_script() {
    let config = NnvConfig::default();
    let spec = create_test_spec();
    let script = script::generate_nnv_script(&spec, &config).unwrap();

    assert!(script.contains("import nnv"));
    assert!(script.contains("NNV_RESULT_START"));
    assert!(script.contains("NNV_RESULT_END"));
    assert!(script.contains("verification_rate"));
}

#[test]
fn test_parse_nnv_output_verified() {
    let stdout = r#"
NNV_INFO: Using synthetic test network
NNV_RESULT_START
{
  "verified_count": 10,
  "total_count": 10,
  "verification_rate": 1.0,
  "epsilon": 0.01,
  "method": "star",
  "input_dim": 2,
  "output_dim": 2,
  "num_counterexamples": 0
}
NNV_RESULT_END
NNV_SUMMARY: Verified 10/10 samples (100.00%)
NNV_STATUS: VERIFIED
"#;

    let (status, cex) = script::parse_nnv_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(cex.is_none());
}

#[test]
fn test_parse_nnv_output_not_verified() {
    let stdout = r#"
NNV_RESULT_START
{
  "verified_count": 3,
  "total_count": 10,
  "verification_rate": 0.3,
  "epsilon": 0.1,
  "method": "zonotope",
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
NNV_RESULT_END
NNV_STATUS: NOT_VERIFIED
"#;

    let (status, cex) = script::parse_nnv_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    assert!(cex.is_some());

    let cex = cex.unwrap();
    assert!(cex.witness.contains_key("original_input"));
    assert!(cex.witness.contains_key("true_label"));
}

#[test]
fn test_parse_nnv_output_partial() {
    let stdout = r#"
NNV_RESULT_START
{
  "verified_count": 9,
  "total_count": 10,
  "verification_rate": 0.9,
  "epsilon": 0.05,
  "method": "star",
  "input_dim": 2,
  "output_dim": 2,
  "num_counterexamples": 1
}
NNV_RESULT_END
NNV_STATUS: PARTIALLY_VERIFIED
"#;

    let (status, _) = script::parse_nnv_output(stdout, "");
    assert!(
        matches!(status, VerificationStatus::Partial { verified_percentage } if (verified_percentage - 90.0).abs() < 0.1)
    );
}

#[test]
fn test_parse_nnv_output_error() {
    let stdout = "NNV_ERROR: Missing dependency: nnv";
    let (status, _) = script::parse_nnv_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[test]
fn test_parse_nnv_output_fallback_status() {
    let stdout = "Some output\nNNV_STATUS: VERIFIED\n";
    let (status, _) = script::parse_nnv_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
}

#[tokio::test]
async fn test_nnv_health_check() {
    let backend = NnvBackend::new();
    let status = backend.health_check().await;
    // NNV is likely not installed in test environment
    assert!(matches!(
        status,
        HealthStatus::Healthy | HealthStatus::Unavailable { .. }
    ));
}

#[tokio::test]
async fn test_nnv_verify_returns_result_or_unavailable() {
    let backend = NnvBackend::new();
    let spec = create_test_spec();
    let result = backend.verify(&spec).await;

    // Either succeeds or reports unavailable
    match result {
        Ok(r) => {
            assert_eq!(r.backend, BackendId::NNV);
        }
        Err(BackendError::Unavailable(_)) => {
            // Expected if NNV not installed
        }
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}

/// Create a minimal test spec for testing
fn create_test_spec() -> dashprove_usl::typecheck::TypedSpec {
    use dashprove_usl::ast::Spec;
    use dashprove_usl::typecheck::TypedSpec;
    use std::collections::HashMap;

    TypedSpec {
        spec: Spec::default(),
        type_info: HashMap::new(),
    }
}
