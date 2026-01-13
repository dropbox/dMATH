//! Tests for nnenum backend

use super::*;
use crate::traits::{BackendError, HealthStatus, VerificationBackend, VerificationStatus};

#[test]
fn test_nnenum_backend_id() {
    let backend = NnenumBackend::new();
    assert_eq!(backend.id(), BackendId::Nnenum);
}

#[test]
fn test_nnenum_supports_neural_properties() {
    let backend = NnenumBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::NeuralRobustness));
    assert!(supported.contains(&PropertyType::NeuralReachability));
}

#[test]
fn test_nnenum_config_default() {
    let config = NnenumConfig::default();
    assert_eq!(config.epsilon, 0.01);
    assert_eq!(config.strategy, EnumerationStrategy::DepthFirst);
    assert!(!config.exact_arithmetic);
    assert_eq!(config.num_processes, 1);
}

#[test]
fn test_nnenum_config_fast() {
    let config = NnenumConfig::fast();
    assert_eq!(config.strategy, EnumerationStrategy::BestFirst);
    assert!(!config.exact_arithmetic);
    assert!(config.max_lp_calls.is_some());
}

#[test]
fn test_nnenum_config_complete() {
    let config = NnenumConfig::complete();
    assert_eq!(config.strategy, EnumerationStrategy::DepthFirst);
    assert!(config.exact_arithmetic);
    assert!(config.max_lp_calls.is_none());
}

#[test]
fn test_enumeration_strategy_as_str() {
    assert_eq!(EnumerationStrategy::DepthFirst.as_str(), "depth_first");
    assert_eq!(EnumerationStrategy::BreadthFirst.as_str(), "breadth_first");
    assert_eq!(EnumerationStrategy::BestFirst.as_str(), "best_first");
    assert_eq!(EnumerationStrategy::Mixed.as_str(), "mixed");
}

#[test]
fn test_generate_nnenum_script() {
    let config = NnenumConfig::default();
    let spec = create_test_spec();
    let script = script::generate_nnenum_script(&spec, &config).unwrap();

    assert!(script.contains("import nnenum"));
    assert!(script.contains("NNENUM_RESULT_START"));
    assert!(script.contains("NNENUM_RESULT_END"));
    assert!(script.contains("verification_rate"));
}

#[test]
fn test_parse_nnenum_output_verified() {
    let stdout = r#"
NNENUM_INFO: Using synthetic test network
NNENUM_RESULT_START
{
  "verified_count": 10,
  "total_count": 10,
  "verification_rate": 1.0,
  "epsilon": 0.01,
  "strategy": "depth_first",
  "exact_arithmetic": false,
  "input_dim": 2,
  "output_dim": 2,
  "num_counterexamples": 0
}
NNENUM_RESULT_END
NNENUM_SUMMARY: Verified 10/10 samples (100.00%)
NNENUM_STATUS: VERIFIED
"#;

    let (status, cex) = script::parse_nnenum_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(cex.is_none());
}

#[test]
fn test_parse_nnenum_output_not_verified() {
    let stdout = r#"
NNENUM_RESULT_START
{
  "verified_count": 2,
  "total_count": 10,
  "verification_rate": 0.2,
  "epsilon": 0.1,
  "strategy": "best_first",
  "exact_arithmetic": true,
  "input_dim": 2,
  "output_dim": 2,
  "num_counterexamples": 8,
  "counterexample": {
    "sample_index": 1,
    "original_input": [0.3, 0.7],
    "true_label": 0,
    "counterexample": [0.25, 0.75]
  }
}
NNENUM_RESULT_END
NNENUM_STATUS: NOT_VERIFIED
"#;

    let (status, cex) = script::parse_nnenum_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    assert!(cex.is_some());

    let cex = cex.unwrap();
    assert!(cex.witness.contains_key("original_input"));
    assert!(cex.witness.contains_key("true_label"));
}

#[test]
fn test_parse_nnenum_output_partial() {
    let stdout = r#"
NNENUM_RESULT_START
{
  "verified_count": 9,
  "total_count": 10,
  "verification_rate": 0.9,
  "epsilon": 0.05,
  "strategy": "depth_first",
  "exact_arithmetic": false,
  "input_dim": 2,
  "output_dim": 2,
  "num_counterexamples": 1
}
NNENUM_RESULT_END
NNENUM_STATUS: PARTIALLY_VERIFIED
"#;

    let (status, _) = script::parse_nnenum_output(stdout, "");
    assert!(
        matches!(status, VerificationStatus::Partial { verified_percentage } if (verified_percentage - 90.0).abs() < 0.1)
    );
}

#[test]
fn test_parse_nnenum_output_error() {
    let stdout = "NNENUM_ERROR: Missing dependency: nnenum";
    let (status, _) = script::parse_nnenum_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[test]
fn test_parse_nnenum_output_fallback_status() {
    let stdout = "Some output\nNNENUM_STATUS: VERIFIED\n";
    let (status, _) = script::parse_nnenum_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
}

#[tokio::test]
async fn test_nnenum_health_check() {
    let backend = NnenumBackend::new();
    let status = backend.health_check().await;
    assert!(matches!(
        status,
        HealthStatus::Healthy | HealthStatus::Unavailable { .. }
    ));
}

#[tokio::test]
async fn test_nnenum_verify_returns_result_or_unavailable() {
    let backend = NnenumBackend::new();
    let spec = create_test_spec();
    let result = backend.verify(&spec).await;

    match result {
        Ok(r) => {
            assert_eq!(r.backend, BackendId::Nnenum);
        }
        Err(BackendError::Unavailable(_)) => {
            // Expected if nnenum not installed
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
