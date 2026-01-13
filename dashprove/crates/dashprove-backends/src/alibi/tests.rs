//! Tests for Alibi backend

use super::*;
use crate::traits::{BackendError, HealthStatus, VerificationBackend, VerificationStatus};

fn create_test_spec() -> dashprove_usl::typecheck::TypedSpec {
    use dashprove_usl::ast::Spec;
    use dashprove_usl::typecheck::TypedSpec;
    use std::collections::HashMap;

    TypedSpec {
        spec: Spec::default(),
        type_info: HashMap::new(),
    }
}

#[test]
fn test_explainer_strings() {
    assert_eq!(AlibiExplainer::AnchorTabular.as_str(), "anchor_tabular");
    assert_eq!(AlibiExplainer::AnchorText.as_str(), "anchor_text");
    assert_eq!(AlibiExplainer::Counterfactual.as_str(), "counterfactual");
}

#[test]
fn test_default_config_values() {
    let cfg = AlibiConfig::default();
    assert_eq!(cfg.explainer, AlibiExplainer::AnchorTabular);
    assert_eq!(cfg.sample_size, 800);
    assert!(cfg.precision_threshold > 0.0);
    assert!(cfg.coverage_threshold > 0.0);
}

#[test]
fn test_counterfactual_builder() {
    let cfg = AlibiConfig::counterfactual();
    assert_eq!(cfg.explainer, AlibiExplainer::Counterfactual);
    assert!(cfg.precision_threshold <= 0.8);
}

#[test]
fn test_backend_id_and_supports() {
    let backend = AlibiBackend::new();
    assert_eq!(backend.id(), BackendId::Alibi);
    assert!(backend.supports().contains(&PropertyType::Interpretability));
}

#[test]
fn test_script_generation() {
    let cfg = AlibiConfig::default();
    let spec = create_test_spec();
    let script = script::generate_alibi_script(&spec, &cfg).unwrap();

    assert!(script.contains("ALIBI_RESULT_START"));
    assert!(script.contains("ALIBI_RESULT_END"));
    assert!(script.contains("ALIBI_STATUS:"));
}

#[test]
fn test_parse_verified_output() {
    let stdout = r#"
ALIBI_RESULT_START
{
  "status": "success",
  "precision": 0.9,
  "coverage": 0.75,
  "precision_threshold": 0.8,
  "coverage_threshold": 0.6,
  "rule": "f1 > 0.5"
}
ALIBI_RESULT_END
ALIBI_STATUS: VERIFIED
"#;
    let (status, ce) = script::parse_alibi_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(ce.is_none());
}

#[test]
fn test_parse_partial_output() {
    let stdout = r#"
ALIBI_RESULT_START
{
  "status": "success",
  "precision": 0.65,
  "coverage": 0.5,
  "precision_threshold": 0.8,
  "coverage_threshold": 0.6,
  "rule": "f2 <= 1.0"
}
ALIBI_RESULT_END
ALIBI_STATUS: PARTIALLY_VERIFIED
"#;
    let (status, ce) = script::parse_alibi_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Partial { .. }));
    assert!(ce.is_some());
}

#[test]
fn test_parse_error_output() {
    let stdout = "ALIBI_ERROR: missing alibi";
    let (status, _) = script::parse_alibi_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[tokio::test]
async fn test_alibi_health_check() {
    let backend = AlibiBackend::new();
    let status = backend.health_check().await;
    match status {
        HealthStatus::Healthy => println!("Alibi available"),
        HealthStatus::Unavailable { reason } => println!("Alibi unavailable: {}", reason),
        HealthStatus::Degraded { reason } => println!("Alibi degraded: {}", reason),
    }
}

#[tokio::test]
async fn test_alibi_verify_returns_result_or_unavailable() {
    let backend = AlibiBackend::new();
    let spec = create_test_spec();

    match backend.verify(&spec).await {
        Ok(result) => {
            assert_eq!(result.backend, BackendId::Alibi);
            println!("Alibi verification status: {:?}", result.status);
        }
        Err(BackendError::Unavailable(reason)) => {
            println!("Alibi unavailable: {}", reason);
        }
        Err(e) => {
            println!("Alibi error (expected if deps missing): {}", e);
        }
    }
}
