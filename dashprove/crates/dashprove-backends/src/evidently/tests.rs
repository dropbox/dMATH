//! Tests for Evidently backend

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
fn test_report_type_strings() {
    assert_eq!(ReportType::DataDrift.as_str(), "data_drift");
    assert_eq!(ReportType::DataQuality.as_str(), "data_quality");
    assert_eq!(
        ReportType::RegressionPerformance.as_str(),
        "regression_performance"
    );
    assert_eq!(
        ReportType::ClassificationPerformance.as_str(),
        "classification_performance"
    );
    assert_eq!(ReportType::TargetDrift.as_str(), "target_drift");
}

#[test]
fn test_stat_test_method_strings() {
    assert_eq!(StatTestMethod::KolmogorovSmirnov.as_str(), "ks");
    assert_eq!(StatTestMethod::ChiSquared.as_str(), "chi2");
    assert_eq!(StatTestMethod::PSI.as_str(), "psi");
    assert_eq!(StatTestMethod::JensenShannon.as_str(), "jensenshannon");
    assert_eq!(StatTestMethod::Wasserstein.as_str(), "wasserstein");
}

#[test]
fn test_output_format_strings() {
    assert_eq!(OutputFormat::JSON.as_str(), "json");
    assert_eq!(OutputFormat::HTML.as_str(), "html");
    assert_eq!(OutputFormat::Dict.as_str(), "dict");
}

#[test]
fn test_default_config() {
    let config = EvidentlyConfig::default();
    assert_eq!(config.report_type, ReportType::DataDrift);
    assert_eq!(config.stat_test_method, StatTestMethod::KolmogorovSmirnov);
    assert_eq!(config.output_format, OutputFormat::JSON);
    assert!((config.drift_threshold - 0.1).abs() < 1e-6);
    assert_eq!(config.n_samples, 1000);
    assert!((config.stattest_threshold - 0.05).abs() < 1e-6);
}

#[test]
fn test_data_quality_config() {
    let config = EvidentlyConfig::data_quality();
    assert_eq!(config.report_type, ReportType::DataQuality);
}

#[test]
fn test_classification_performance_config() {
    let config = EvidentlyConfig::classification_performance();
    assert_eq!(config.report_type, ReportType::ClassificationPerformance);
}

#[test]
fn test_regression_performance_config() {
    let config = EvidentlyConfig::regression_performance();
    assert_eq!(config.report_type, ReportType::RegressionPerformance);
}

#[test]
fn test_target_drift_config() {
    let config = EvidentlyConfig::target_drift();
    assert_eq!(config.report_type, ReportType::TargetDrift);
}

#[test]
fn test_psi_config() {
    let config = EvidentlyConfig::with_psi();
    assert_eq!(config.stat_test_method, StatTestMethod::PSI);
}

#[test]
fn test_backend_id() {
    let backend = EvidentlyBackend::new();
    assert_eq!(backend.id(), BackendId::Evidently);
}

#[test]
fn test_supports_data_quality() {
    let backend = EvidentlyBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::DataQuality));
}

#[test]
fn test_script_generation() {
    let config = EvidentlyConfig::default();
    let spec = create_test_spec();
    let script = script::generate_evidently_script(&spec, &config).unwrap();

    assert!(script.contains("import evidently"));
    assert!(script.contains("EVIDENTLY_RESULT_START"));
    assert!(script.contains("EVIDENTLY_RESULT_END"));
    assert!(script.contains("EVIDENTLY_STATUS:"));
}

#[test]
fn test_parse_verified_output() {
    let stdout = r#"
EVIDENTLY_RESULT_START
{
    "status": "success",
    "report_type": "data_drift",
    "drift_detected": false,
    "drift_score": 0.0,
    "drift_threshold": 0.1,
    "total_features": 4,
    "drifted_features": []
}
EVIDENTLY_RESULT_END
EVIDENTLY_STATUS: VERIFIED
"#;
    let (status, ce) = script::parse_evidently_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(ce.is_none());
}

#[test]
fn test_parse_partial_output() {
    let stdout = r#"
EVIDENTLY_RESULT_START
{
    "status": "success",
    "report_type": "data_drift",
    "drift_detected": true,
    "drift_score": 0.15,
    "drift_threshold": 0.1,
    "total_features": 4,
    "drifted_features": ["feature_1"]
}
EVIDENTLY_RESULT_END
EVIDENTLY_STATUS: PARTIALLY_VERIFIED
"#;
    let (status, ce) = script::parse_evidently_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Partial { .. }));
    assert!(ce.is_some());
}

#[test]
fn test_parse_not_verified_output() {
    let stdout = r#"
EVIDENTLY_RESULT_START
{
    "status": "success",
    "report_type": "data_drift",
    "drift_detected": true,
    "drift_score": 0.5,
    "drift_threshold": 0.1,
    "total_features": 4,
    "drifted_features": ["feature_1", "feature_2"]
}
EVIDENTLY_RESULT_END
EVIDENTLY_STATUS: NOT_VERIFIED
"#;
    let (status, ce) = script::parse_evidently_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    assert!(ce.is_some());
}

#[test]
fn test_parse_error_output() {
    let stdout = "EVIDENTLY_ERROR: Report generation failed";
    let (status, _) = script::parse_evidently_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[tokio::test]
async fn test_evidently_health_check() {
    let backend = EvidentlyBackend::new();
    let status = backend.health_check().await;
    match status {
        HealthStatus::Healthy => println!("Evidently is available"),
        HealthStatus::Unavailable { reason } => println!("Evidently unavailable: {}", reason),
        HealthStatus::Degraded { reason } => println!("Evidently degraded: {}", reason),
    }
}

#[tokio::test]
async fn test_evidently_verify_returns_result_or_unavailable() {
    let backend = EvidentlyBackend::new();
    let spec = create_test_spec();

    match backend.verify(&spec).await {
        Ok(result) => {
            assert_eq!(result.backend, BackendId::Evidently);
            println!("Evidently verify succeeded: {:?}", result.status);
        }
        Err(BackendError::Unavailable(reason)) => {
            println!("Evidently unavailable: {}", reason);
        }
        Err(e) => {
            println!("Evidently error (expected if not installed): {}", e);
        }
    }
}
