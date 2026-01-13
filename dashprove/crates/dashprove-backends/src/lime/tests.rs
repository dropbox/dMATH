//! Tests for LIME backend

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
fn test_task_type_strings() {
    assert_eq!(LimeTaskType::Classification.as_str(), "classification");
    assert_eq!(LimeTaskType::Regression.as_str(), "regression");
}

#[test]
fn test_kernel_width_to_python() {
    assert_eq!(KernelWidth::Auto.to_python(), "None");
    assert_eq!(KernelWidth::Fixed(0.5).to_python(), "0.5");
}

#[test]
fn test_default_config() {
    let cfg = LimeConfig::default();
    assert_eq!(cfg.task_type, LimeTaskType::Classification);
    assert_eq!(cfg.num_features, 6);
    assert_eq!(cfg.num_samples, 1500);
    assert!(cfg.discretize_continuous);
    assert!(cfg.fidelity_threshold > 0.0);
}

#[test]
fn test_regression_builder() {
    let cfg = LimeConfig::regression();
    assert_eq!(cfg.task_type, LimeTaskType::Regression);
    assert!(cfg.fidelity_threshold <= 0.6);
}

#[test]
fn test_backend_id_and_supports() {
    let backend = LimeBackend::new();
    assert_eq!(backend.id(), BackendId::LIME);
    assert!(backend.supports().contains(&PropertyType::Interpretability));
}

#[test]
fn test_script_generation() {
    let cfg = LimeConfig::default();
    let spec = create_test_spec();
    let script = script::generate_lime_script(&spec, &cfg).unwrap();

    assert!(script.contains("LIME_RESULT_START"));
    assert!(script.contains("LIME_RESULT_END"));
    assert!(script.contains("LIME_STATUS:"));
}

#[test]
fn test_parse_verified_output() {
    let stdout = r#"
LIME_RESULT_START
{
  "status": "success",
  "fidelity": 0.8,
  "coverage": 0.75,
  "fidelity_threshold": 0.6,
  "max_weight": 0.2,
  "top_feature": "f1"
}
LIME_RESULT_END
LIME_STATUS: VERIFIED
"#;
    let (status, ce) = script::parse_lime_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(ce.is_none());
}

#[test]
fn test_parse_partial_output() {
    let stdout = r#"
LIME_RESULT_START
{
  "status": "success",
  "fidelity": 0.5,
  "coverage": 0.65,
  "fidelity_threshold": 0.7,
  "max_weight": 0.08,
  "top_feature": "f3"
}
LIME_RESULT_END
LIME_STATUS: PARTIALLY_VERIFIED
"#;
    let (status, ce) = script::parse_lime_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Partial { .. }));
    assert!(ce.is_some());
}

#[test]
fn test_parse_error_output() {
    let stdout = "LIME_ERROR: dependency missing";
    let (status, _) = script::parse_lime_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[tokio::test]
async fn test_lime_health_check() {
    let backend = LimeBackend::new();
    let status = backend.health_check().await;
    match status {
        HealthStatus::Healthy => println!("LIME available"),
        HealthStatus::Unavailable { reason } => println!("LIME unavailable: {}", reason),
        HealthStatus::Degraded { reason } => println!("LIME degraded: {}", reason),
    }
}

#[tokio::test]
async fn test_lime_verify_returns_result_or_unavailable() {
    let backend = LimeBackend::new();
    let spec = create_test_spec();

    match backend.verify(&spec).await {
        Ok(result) => {
            assert_eq!(result.backend, BackendId::LIME);
            println!("LIME verification status: {:?}", result.status);
        }
        Err(BackendError::Unavailable(reason)) => {
            println!("LIME unavailable: {}", reason);
        }
        Err(e) => {
            println!("LIME error (expected if deps missing): {}", e);
        }
    }
}
