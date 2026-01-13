//! Tests for InterpretML backend

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
    assert_eq!(InterpretExplainer::ExplainableBoosting.as_str(), "ebm");
    assert_eq!(InterpretExplainer::Linear.as_str(), "linear");
    assert_eq!(InterpretExplainer::DecisionTree.as_str(), "tree");
}

#[test]
fn test_task_strings() {
    assert_eq!(InterpretTask::Classification.as_str(), "classification");
    assert_eq!(InterpretTask::Regression.as_str(), "regression");
}

#[test]
fn test_default_config_values() {
    let cfg = InterpretMlConfig::default();
    assert_eq!(cfg.explainer, InterpretExplainer::ExplainableBoosting);
    assert_eq!(cfg.task, InterpretTask::Classification);
    assert_eq!(cfg.max_bins, 16);
    assert_eq!(cfg.max_interactions, 2);
    assert!(cfg.importance_threshold > 0.0);
}

#[test]
fn test_linear_builder() {
    let cfg = InterpretMlConfig::linear();
    assert_eq!(cfg.explainer, InterpretExplainer::Linear);
}

#[test]
fn test_backend_id_and_supports() {
    let backend = InterpretMlBackend::new();
    assert_eq!(backend.id(), BackendId::InterpretML);
    assert!(backend.supports().contains(&PropertyType::Interpretability));
}

#[test]
fn test_script_generation() {
    let cfg = InterpretMlConfig::default();
    let spec = create_test_spec();
    let script = script::generate_interpretml_script(&spec, &cfg).unwrap();

    assert!(script.contains("INTERPRET_RESULT_START"));
    assert!(script.contains("INTERPRET_RESULT_END"));
    assert!(script.contains("INTERPRET_STATUS:"));
}

#[test]
fn test_parse_verified_output() {
    let stdout = r#"
INTERPRET_RESULT_START
{
  "status": "success",
  "mean_importance": 0.08,
  "importance_threshold": 0.05,
  "max_importance": 0.2,
  "top_feature": "f1"
}
INTERPRET_RESULT_END
INTERPRET_STATUS: VERIFIED
"#;
    let (status, ce) = script::parse_interpretml_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(ce.is_none());
}

#[test]
fn test_parse_partial_output() {
    let stdout = r#"
INTERPRET_RESULT_START
{
  "status": "success",
  "mean_importance": 0.025,
  "importance_threshold": 0.05,
  "max_importance": 0.07,
  "top_feature": "f3"
}
INTERPRET_RESULT_END
INTERPRET_STATUS: PARTIALLY_VERIFIED
"#;
    let (status, ce) = script::parse_interpretml_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Partial { .. }));
    assert!(ce.is_some());
}

#[test]
fn test_parse_error_output() {
    let stdout = "INTERPRET_ERROR: missing package";
    let (status, _) = script::parse_interpretml_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[tokio::test]
async fn test_interpretml_health_check() {
    let backend = InterpretMlBackend::new();
    let status = backend.health_check().await;
    match status {
        HealthStatus::Healthy => println!("InterpretML available"),
        HealthStatus::Unavailable { reason } => println!("InterpretML unavailable: {}", reason),
        HealthStatus::Degraded { reason } => println!("InterpretML degraded: {}", reason),
    }
}

#[tokio::test]
async fn test_interpretml_verify_returns_result_or_unavailable() {
    let backend = InterpretMlBackend::new();
    let spec = create_test_spec();

    match backend.verify(&spec).await {
        Ok(result) => {
            assert_eq!(result.backend, BackendId::InterpretML);
            println!("InterpretML status: {:?}", result.status);
        }
        Err(BackendError::Unavailable(reason)) => {
            println!("InterpretML unavailable: {}", reason);
        }
        Err(e) => {
            println!("InterpretML error (expected if deps missing): {}", e);
        }
    }
}
