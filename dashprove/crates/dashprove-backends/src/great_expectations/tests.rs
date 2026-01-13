//! Tests for Great Expectations backend

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
fn test_validation_level_strings() {
    assert_eq!(ValidationLevel::Basic.as_str(), "basic");
    assert_eq!(ValidationLevel::Standard.as_str(), "standard");
    assert_eq!(ValidationLevel::Comprehensive.as_str(), "comprehensive");
}

#[test]
fn test_data_source_type_strings() {
    assert_eq!(DataSourceType::Pandas.as_str(), "pandas");
    assert_eq!(DataSourceType::Spark.as_str(), "spark");
    assert_eq!(DataSourceType::SQL.as_str(), "sql");
    assert_eq!(DataSourceType::File.as_str(), "file");
}

#[test]
fn test_result_format_strings() {
    assert_eq!(ResultFormat::JSON.as_str(), "json");
    assert_eq!(ResultFormat::HTML.as_str(), "html");
    assert_eq!(ResultFormat::Markdown.as_str(), "markdown");
}

#[test]
fn test_default_config() {
    let config = GreatExpectationsConfig::default();
    assert_eq!(config.validation_level, ValidationLevel::Standard);
    assert_eq!(config.data_source_type, DataSourceType::Pandas);
    assert_eq!(config.result_format, ResultFormat::JSON);
    assert!(config.evaluation_parameters);
    assert!(config.catch_exceptions);
    assert!(!config.include_unexpected_rows);
    assert_eq!(config.max_unexpected_values, 20);
}

#[test]
fn test_basic_config() {
    let config = GreatExpectationsConfig::basic();
    assert_eq!(config.validation_level, ValidationLevel::Basic);
}

#[test]
fn test_comprehensive_config() {
    let config = GreatExpectationsConfig::comprehensive();
    assert_eq!(config.validation_level, ValidationLevel::Comprehensive);
    assert!(config.include_unexpected_rows);
}

#[test]
fn test_spark_config() {
    let config = GreatExpectationsConfig::spark();
    assert_eq!(config.data_source_type, DataSourceType::Spark);
}

#[test]
fn test_sql_config() {
    let config = GreatExpectationsConfig::sql();
    assert_eq!(config.data_source_type, DataSourceType::SQL);
}

#[test]
fn test_backend_id() {
    let backend = GreatExpectationsBackend::new();
    assert_eq!(backend.id(), BackendId::GreatExpectations);
}

#[test]
fn test_supports_data_quality() {
    let backend = GreatExpectationsBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::DataQuality));
}

#[test]
fn test_script_generation() {
    let config = GreatExpectationsConfig::default();
    let spec = create_test_spec();
    let script = script::generate_great_expectations_script(&spec, &config).unwrap();

    assert!(script.contains("import great_expectations"));
    assert!(script.contains("GX_RESULT_START"));
    assert!(script.contains("GX_RESULT_END"));
    assert!(script.contains("GX_STATUS:"));
}

#[test]
fn test_parse_verified_output() {
    let stdout = r#"
GX_RESULT_START
{
    "status": "success",
    "validation_level": "standard",
    "expectations_evaluated": 10,
    "expectations_passed": 10,
    "expectations_failed": 0,
    "success_rate": 1.0
}
GX_RESULT_END
GX_STATUS: VERIFIED
"#;
    let (status, ce) = script::parse_great_expectations_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(ce.is_none());
}

#[test]
fn test_parse_partial_output() {
    let stdout = r#"
GX_RESULT_START
{
    "status": "success",
    "validation_level": "standard",
    "expectations_evaluated": 10,
    "expectations_passed": 9,
    "expectations_failed": 1,
    "success_rate": 0.9,
    "validation_results": []
}
GX_RESULT_END
GX_STATUS: PARTIALLY_VERIFIED
"#;
    let (status, ce) = script::parse_great_expectations_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Partial { .. }));
    assert!(ce.is_some());
}

#[test]
fn test_parse_not_verified_output() {
    let stdout = r#"
GX_RESULT_START
{
    "status": "success",
    "validation_level": "standard",
    "expectations_evaluated": 10,
    "expectations_passed": 5,
    "expectations_failed": 5,
    "success_rate": 0.5,
    "validation_results": []
}
GX_RESULT_END
GX_STATUS: NOT_VERIFIED
"#;
    let (status, ce) = script::parse_great_expectations_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    assert!(ce.is_some());
}

#[test]
fn test_parse_error_output() {
    let stdout = "GX_ERROR: Data validation failed";
    let (status, _) = script::parse_great_expectations_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[tokio::test]
async fn test_gx_health_check() {
    let backend = GreatExpectationsBackend::new();
    let status = backend.health_check().await;
    match status {
        HealthStatus::Healthy => println!("Great Expectations is available"),
        HealthStatus::Unavailable { reason } => {
            println!("Great Expectations unavailable: {}", reason)
        }
        HealthStatus::Degraded { reason } => println!("Great Expectations degraded: {}", reason),
    }
}

#[tokio::test]
async fn test_gx_verify_returns_result_or_unavailable() {
    let backend = GreatExpectationsBackend::new();
    let spec = create_test_spec();

    match backend.verify(&spec).await {
        Ok(result) => {
            assert_eq!(result.backend, BackendId::GreatExpectations);
            println!("Great Expectations verify succeeded: {:?}", result.status);
        }
        Err(BackendError::Unavailable(reason)) => {
            println!("Great Expectations unavailable: {}", reason);
        }
        Err(e) => {
            println!(
                "Great Expectations error (expected if not installed): {}",
                e
            );
        }
    }
}
