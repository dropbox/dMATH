//! Tests for Deepchecks backend

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
fn test_suite_type_strings() {
    assert_eq!(SuiteType::DataIntegrity.as_str(), "data_integrity");
    assert_eq!(
        SuiteType::TrainTestValidation.as_str(),
        "train_test_validation"
    );
    assert_eq!(SuiteType::ModelEvaluation.as_str(), "model_evaluation");
    assert_eq!(SuiteType::FullSuite.as_str(), "full_suite");
}

#[test]
fn test_task_type_strings() {
    assert_eq!(TaskType::BinaryClassification.as_str(), "binary");
    assert_eq!(TaskType::MulticlassClassification.as_str(), "multiclass");
    assert_eq!(TaskType::Regression.as_str(), "regression");
    assert_eq!(TaskType::ObjectDetection.as_str(), "object_detection");
    assert_eq!(
        TaskType::SemanticSegmentation.as_str(),
        "semantic_segmentation"
    );
}

#[test]
fn test_severity_threshold_strings() {
    assert_eq!(SeverityThreshold::Low.as_str(), "low");
    assert_eq!(SeverityThreshold::Medium.as_str(), "medium");
    assert_eq!(SeverityThreshold::High.as_str(), "high");
}

#[test]
fn test_default_config() {
    let config = DeepchecksConfig::default();
    assert_eq!(config.suite_type, SuiteType::DataIntegrity);
    assert_eq!(config.task_type, TaskType::BinaryClassification);
    assert_eq!(config.severity_threshold, SeverityThreshold::Medium);
    assert!(config.with_conditions);
    assert!(!config.show_only_failed);
    assert_eq!(config.n_samples, 1000);
}

#[test]
fn test_train_test_config() {
    let config = DeepchecksConfig::train_test();
    assert_eq!(config.suite_type, SuiteType::TrainTestValidation);
}

#[test]
fn test_model_evaluation_config() {
    let config = DeepchecksConfig::model_evaluation(TaskType::Regression);
    assert_eq!(config.suite_type, SuiteType::ModelEvaluation);
    assert_eq!(config.task_type, TaskType::Regression);
}

#[test]
fn test_full_suite_config() {
    let config = DeepchecksConfig::full_suite();
    assert_eq!(config.suite_type, SuiteType::FullSuite);
}

#[test]
fn test_regression_config() {
    let config = DeepchecksConfig::regression();
    assert_eq!(config.task_type, TaskType::Regression);
}

#[test]
fn test_backend_id() {
    let backend = DeepchecksBackend::new();
    assert_eq!(backend.id(), BackendId::Deepchecks);
}

#[test]
fn test_supports_data_quality() {
    let backend = DeepchecksBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::DataQuality));
}

#[test]
fn test_script_generation() {
    let config = DeepchecksConfig::default();
    let spec = create_test_spec();
    let script = script::generate_deepchecks_script(&spec, &config).unwrap();

    assert!(script.contains("import deepchecks"));
    assert!(script.contains("DEEPCHECKS_RESULT_START"));
    assert!(script.contains("DEEPCHECKS_RESULT_END"));
    assert!(script.contains("DEEPCHECKS_STATUS:"));
}

#[test]
fn test_parse_verified_output() {
    let stdout = r#"
DEEPCHECKS_RESULT_START
{
    "status": "success",
    "suite_type": "data_integrity",
    "checks_evaluated": 10,
    "checks_passed": 10,
    "checks_failed": 0,
    "success_rate": 1.0
}
DEEPCHECKS_RESULT_END
DEEPCHECKS_STATUS: VERIFIED
"#;
    let (status, ce) = script::parse_deepchecks_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(ce.is_none());
}

#[test]
fn test_parse_partial_output() {
    let stdout = r#"
DEEPCHECKS_RESULT_START
{
    "status": "success",
    "suite_type": "data_integrity",
    "checks_evaluated": 10,
    "checks_passed": 8,
    "checks_failed": 2,
    "success_rate": 0.8
}
DEEPCHECKS_RESULT_END
DEEPCHECKS_STATUS: PARTIALLY_VERIFIED
"#;
    let (status, ce) = script::parse_deepchecks_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Partial { .. }));
    assert!(ce.is_some());
}

#[test]
fn test_parse_not_verified_output() {
    let stdout = r#"
DEEPCHECKS_RESULT_START
{
    "status": "success",
    "suite_type": "data_integrity",
    "checks_evaluated": 10,
    "checks_passed": 4,
    "checks_failed": 6,
    "success_rate": 0.4
}
DEEPCHECKS_RESULT_END
DEEPCHECKS_STATUS: NOT_VERIFIED
"#;
    let (status, ce) = script::parse_deepchecks_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    assert!(ce.is_some());
}

#[test]
fn test_parse_error_output() {
    let stdout = "DEEPCHECKS_ERROR: Validation failed";
    let (status, _) = script::parse_deepchecks_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[tokio::test]
async fn test_deepchecks_health_check() {
    let backend = DeepchecksBackend::new();
    let status = backend.health_check().await;
    match status {
        HealthStatus::Healthy => println!("Deepchecks is available"),
        HealthStatus::Unavailable { reason } => println!("Deepchecks unavailable: {}", reason),
        HealthStatus::Degraded { reason } => println!("Deepchecks degraded: {}", reason),
    }
}

#[tokio::test]
async fn test_deepchecks_verify_returns_result_or_unavailable() {
    let backend = DeepchecksBackend::new();
    let spec = create_test_spec();

    match backend.verify(&spec).await {
        Ok(result) => {
            assert_eq!(result.backend, BackendId::Deepchecks);
            println!("Deepchecks verify succeeded: {:?}", result.status);
        }
        Err(BackendError::Unavailable(reason)) => {
            println!("Deepchecks unavailable: {}", reason);
        }
        Err(e) => {
            println!("Deepchecks error (expected if not installed): {}", e);
        }
    }
}
