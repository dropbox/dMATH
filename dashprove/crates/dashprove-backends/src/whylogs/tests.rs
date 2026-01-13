//! Tests for WhyLogs backend

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
fn test_profile_type_strings() {
    assert_eq!(ProfileType::Standard.as_str(), "standard");
    assert_eq!(ProfileType::Lightweight.as_str(), "lightweight");
    assert_eq!(ProfileType::Detailed.as_str(), "detailed");
}

#[test]
fn test_constraint_type_strings() {
    assert_eq!(ConstraintType::Schema.as_str(), "schema");
    assert_eq!(ConstraintType::Value.as_str(), "value");
    assert_eq!(ConstraintType::Distribution.as_str(), "distribution");
    assert_eq!(ConstraintType::All.as_str(), "all");
}

#[test]
fn test_output_format_strings() {
    assert_eq!(WhyLogsOutputFormat::JSON.as_str(), "json");
    assert_eq!(WhyLogsOutputFormat::Protobuf.as_str(), "protobuf");
    assert_eq!(WhyLogsOutputFormat::Flat.as_str(), "flat");
}

#[test]
fn test_default_config() {
    let config = WhyLogsConfig::default();
    assert_eq!(config.profile_type, ProfileType::Standard);
    assert_eq!(config.constraint_type, ConstraintType::All);
    assert_eq!(config.output_format, WhyLogsOutputFormat::JSON);
    assert_eq!(config.n_samples, 1000);
    assert!(config.track_histograms);
    assert!(config.track_frequent_items);
    assert!(config.track_cardinality);
}

#[test]
fn test_lightweight_config() {
    let config = WhyLogsConfig::lightweight();
    assert_eq!(config.profile_type, ProfileType::Lightweight);
    assert!(!config.track_histograms);
    assert!(!config.track_frequent_items);
}

#[test]
fn test_detailed_config() {
    let config = WhyLogsConfig::detailed();
    assert_eq!(config.profile_type, ProfileType::Detailed);
    assert!(config.track_histograms);
    assert!(config.track_frequent_items);
    assert!(config.track_cardinality);
}

#[test]
fn test_schema_only_config() {
    let config = WhyLogsConfig::schema_only();
    assert_eq!(config.constraint_type, ConstraintType::Schema);
}

#[test]
fn test_value_constraints_config() {
    let config = WhyLogsConfig::value_constraints();
    assert_eq!(config.constraint_type, ConstraintType::Value);
}

#[test]
fn test_backend_id() {
    let backend = WhyLogsBackend::new();
    assert_eq!(backend.id(), BackendId::WhyLogs);
}

#[test]
fn test_supports_data_quality() {
    let backend = WhyLogsBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::DataQuality));
}

#[test]
fn test_script_generation() {
    let config = WhyLogsConfig::default();
    let spec = create_test_spec();
    let script = script::generate_whylogs_script(&spec, &config).unwrap();

    assert!(script.contains("import whylogs"));
    assert!(script.contains("WHYLOGS_RESULT_START"));
    assert!(script.contains("WHYLOGS_RESULT_END"));
    assert!(script.contains("WHYLOGS_STATUS:"));
}

#[test]
fn test_parse_verified_output() {
    let stdout = r#"
WHYLOGS_RESULT_START
{
    "status": "success",
    "profile_type": "standard",
    "constraints_evaluated": 5,
    "constraints_passed": 5,
    "constraints_failed": 0,
    "success_rate": 1.0
}
WHYLOGS_RESULT_END
WHYLOGS_STATUS: VERIFIED
"#;
    let (status, ce) = script::parse_whylogs_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(ce.is_none());
}

#[test]
fn test_parse_partial_output() {
    let stdout = r#"
WHYLOGS_RESULT_START
{
    "status": "success",
    "profile_type": "standard",
    "constraints_evaluated": 5,
    "constraints_passed": 4,
    "constraints_failed": 1,
    "success_rate": 0.8,
    "constraint_results": []
}
WHYLOGS_RESULT_END
WHYLOGS_STATUS: PARTIALLY_VERIFIED
"#;
    let (status, ce) = script::parse_whylogs_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Partial { .. }));
    assert!(ce.is_some());
}

#[test]
fn test_parse_not_verified_output() {
    let stdout = r#"
WHYLOGS_RESULT_START
{
    "status": "success",
    "profile_type": "standard",
    "constraints_evaluated": 5,
    "constraints_passed": 2,
    "constraints_failed": 3,
    "success_rate": 0.4,
    "constraint_results": []
}
WHYLOGS_RESULT_END
WHYLOGS_STATUS: NOT_VERIFIED
"#;
    let (status, ce) = script::parse_whylogs_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    assert!(ce.is_some());
}

#[test]
fn test_parse_error_output() {
    let stdout = "WHYLOGS_ERROR: Profiling failed";
    let (status, _) = script::parse_whylogs_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[tokio::test]
async fn test_whylogs_health_check() {
    let backend = WhyLogsBackend::new();
    let status = backend.health_check().await;
    match status {
        HealthStatus::Healthy => println!("WhyLogs is available"),
        HealthStatus::Unavailable { reason } => println!("WhyLogs unavailable: {}", reason),
        HealthStatus::Degraded { reason } => println!("WhyLogs degraded: {}", reason),
    }
}

#[tokio::test]
async fn test_whylogs_verify_returns_result_or_unavailable() {
    let backend = WhyLogsBackend::new();
    let spec = create_test_spec();

    match backend.verify(&spec).await {
        Ok(result) => {
            assert_eq!(result.backend, BackendId::WhyLogs);
            println!("WhyLogs verify succeeded: {:?}", result.status);
        }
        Err(BackendError::Unavailable(reason)) => {
            println!("WhyLogs unavailable: {}", reason);
        }
        Err(e) => {
            println!("WhyLogs error (expected if not installed): {}", e);
        }
    }
}
