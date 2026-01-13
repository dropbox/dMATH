//! Tests for Venus backend

use super::*;
use crate::traits::{BackendError, HealthStatus, VerificationBackend, VerificationStatus};

#[test]
fn test_venus_backend_id() {
    let backend = VenusBackend::new();
    assert_eq!(backend.id(), BackendId::Venus);
}

#[test]
fn test_venus_supports_neural_properties() {
    let backend = VenusBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::NeuralRobustness));
}

#[test]
fn test_venus_config_default() {
    let config = VenusConfig::default();
    assert_eq!(config.epsilon, 0.01);
    assert_eq!(config.solver, SolverBackend::Gurobi);
    assert!(config.use_bnb);
}

#[test]
fn test_solver_backend_as_str() {
    assert_eq!(SolverBackend::Gurobi.as_str(), "gurobi");
    assert_eq!(SolverBackend::GLPK.as_str(), "glpk");
    assert_eq!(SolverBackend::CBC.as_str(), "cbc");
}

#[test]
fn test_generate_venus_script() {
    let config = VenusConfig::default();
    let spec = create_test_spec();
    let script = script::generate_venus_script(&spec, &config).unwrap();

    assert!(script.contains("import venus"));
    assert!(script.contains("VENUS_RESULT_START"));
    assert!(script.contains("VENUS_RESULT_END"));
}

#[test]
fn test_parse_venus_output_verified() {
    let stdout = r#"
VENUS_RESULT_START
{
  "verified_count": 10,
  "total_count": 10,
  "verification_rate": 1.0,
  "epsilon": 0.01,
  "solver": "gurobi",
  "use_bnb": true,
  "input_dim": 2,
  "output_dim": 2,
  "num_counterexamples": 0
}
VENUS_RESULT_END
VENUS_STATUS: VERIFIED
"#;

    let (status, cex) = script::parse_venus_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(cex.is_none());
}

#[test]
fn test_parse_venus_output_not_verified() {
    let stdout = r#"
VENUS_RESULT_START
{
  "verified_count": 3,
  "total_count": 10,
  "verification_rate": 0.3,
  "epsilon": 0.1,
  "solver": "glpk",
  "use_bnb": true,
  "input_dim": 2,
  "output_dim": 2,
  "num_counterexamples": 7,
  "counterexample": {
    "sample_index": 0,
    "original_input": [0.5, 0.5]
  }
}
VENUS_RESULT_END
VENUS_STATUS: NOT_VERIFIED
"#;

    let (status, cex) = script::parse_venus_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    let cex = cex.expect("expected counterexample");
    assert!(!cex.failed_checks.is_empty());
    assert!(cex.failed_checks[0].description.contains("sample 0"));
    assert!(cex.witness.contains_key("epsilon"));
    assert!(cex.witness.contains_key("original_input"));
}

#[test]
fn test_parse_venus_output_error() {
    let stdout = "VENUS_ERROR: Missing dependency";
    let (status, _) = script::parse_venus_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[tokio::test]
async fn test_venus_health_check() {
    let backend = VenusBackend::new();
    let status = backend.health_check().await;
    assert!(matches!(
        status,
        HealthStatus::Healthy | HealthStatus::Unavailable { .. }
    ));
}

#[tokio::test]
async fn test_venus_verify_returns_result_or_unavailable() {
    let backend = VenusBackend::new();
    let spec = create_test_spec();
    let result = backend.verify(&spec).await;

    match result {
        Ok(r) => assert_eq!(r.backend, BackendId::Venus),
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
