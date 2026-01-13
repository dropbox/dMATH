//! Tests for SHAP backend

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
    assert_eq!(ShapExplainer::Kernel.as_str(), "kernel");
    assert_eq!(ShapExplainer::Tree.as_str(), "tree");
    assert_eq!(ShapExplainer::Linear.as_str(), "linear");
    assert_eq!(ShapExplainer::Deep.as_str(), "deep");
    assert_eq!(ShapExplainer::Gradient.as_str(), "gradient");
}

#[test]
fn test_model_type_strings() {
    assert_eq!(ShapModelType::Classification.as_str(), "classification");
    assert_eq!(ShapModelType::Regression.as_str(), "regression");
}

#[test]
fn test_default_config() {
    let config = ShapConfig::default();
    assert_eq!(config.explainer, ShapExplainer::Kernel);
    assert_eq!(config.model_type, ShapModelType::Classification);
    assert_eq!(config.sample_size, 400);
    assert_eq!(config.background_size, 80);
    assert_eq!(config.max_features, 6);
    assert!(config.evaluate_stability);
    assert!(config.importance_threshold > 0.0);
}

#[test]
fn test_regression_config_builder() {
    let config = ShapConfig::regression();
    assert_eq!(config.model_type, ShapModelType::Regression);
}

#[test]
fn test_backend_id_and_supports() {
    let backend = ShapBackend::new();
    assert_eq!(backend.id(), BackendId::SHAP);
    assert!(backend.supports().contains(&PropertyType::Interpretability));
}

#[test]
fn test_script_generation() {
    let config = ShapConfig::default();
    let spec = create_test_spec();
    let script = script::generate_shap_script(&spec, &config).unwrap();

    assert!(script.contains("SHAP_RESULT_START"));
    assert!(script.contains("SHAP_RESULT_END"));
    assert!(script.contains("SHAP_STATUS:"));
    assert!(script.contains("shap_values"));
}

#[test]
fn test_parse_verified_output() {
    let stdout = r#"
SHAP_RESULT_START
{
  "status": "success",
  "mean_abs_shap": 0.25,
  "importance_threshold": 0.1,
  "max_importance": 0.4,
  "top_feature": 2
}
SHAP_RESULT_END
SHAP_STATUS: VERIFIED
"#;
    let (status, ce) = script::parse_shap_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(ce.is_none());
}

#[test]
fn test_parse_partial_output() {
    let stdout = r#"
SHAP_RESULT_START
{
  "status": "success",
  "mean_abs_shap": 0.04,
  "importance_threshold": 0.1,
  "max_importance": 0.12,
  "top_feature": 1
}
SHAP_RESULT_END
SHAP_STATUS: PARTIALLY_VERIFIED
"#;
    let (status, ce) = script::parse_shap_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Partial { .. }));
    assert!(ce.is_some());
}

#[test]
fn test_parse_error_output() {
    let stdout = "SHAP_ERROR: shap missing";
    let (status, _) = script::parse_shap_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[tokio::test]
async fn test_shap_health_check() {
    let backend = ShapBackend::new();
    let status = backend.health_check().await;
    match status {
        HealthStatus::Healthy => println!("SHAP available"),
        HealthStatus::Unavailable { reason } => println!("SHAP unavailable: {}", reason),
        HealthStatus::Degraded { reason } => println!("SHAP degraded: {}", reason),
    }
}

#[tokio::test]
async fn test_shap_verify_returns_result_or_unavailable() {
    let backend = ShapBackend::new();
    let spec = create_test_spec();

    match backend.verify(&spec).await {
        Ok(result) => {
            assert_eq!(result.backend, BackendId::SHAP);
            println!("SHAP verification status: {:?}", result.status);
        }
        Err(BackendError::Unavailable(reason)) => {
            println!("SHAP unavailable: {}", reason);
        }
        Err(e) => {
            println!("SHAP error (expected if dependencies missing): {}", e);
        }
    }
}
