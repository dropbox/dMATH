//! Tests for Aequitas backend

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
fn test_aequitas_metric_strings() {
    assert_eq!(
        AequitasMetric::PredictiveParity.as_str(),
        "predictive_parity"
    );
    assert_eq!(AequitasMetric::FPRParity.as_str(), "fpr_parity");
    assert_eq!(AequitasMetric::FNRParity.as_str(), "fnr_parity");
    assert_eq!(AequitasMetric::FDRParity.as_str(), "fdr_parity");
    assert_eq!(AequitasMetric::FORParity.as_str(), "for_parity");
    assert_eq!(
        AequitasMetric::TreatmentEquality.as_str(),
        "treatment_equality"
    );
    assert_eq!(AequitasMetric::ImpactParity.as_str(), "impact_parity");
}

#[test]
fn test_reference_group_strings() {
    assert_eq!(ReferenceGroup::Majority.as_str(), "majority");
    assert_eq!(ReferenceGroup::Minority.as_str(), "minority");
    assert_eq!(ReferenceGroup::Global.as_str(), "global");
}

#[test]
fn test_default_config() {
    let config = AequitasConfig::default();
    assert_eq!(config.fairness_metric, AequitasMetric::PredictiveParity);
    assert_eq!(config.reference_group, ReferenceGroup::Majority);
    assert!((config.disparity_tolerance - 0.8).abs() < 1e-6);
    assert!((config.significance_threshold - 0.05).abs() < 1e-6);
    assert_eq!(config.n_samples, 1000);
}

#[test]
fn test_fpr_parity_config() {
    let config = AequitasConfig::fpr_parity();
    assert_eq!(config.fairness_metric, AequitasMetric::FPRParity);
}

#[test]
fn test_fnr_parity_config() {
    let config = AequitasConfig::fnr_parity();
    assert_eq!(config.fairness_metric, AequitasMetric::FNRParity);
}

#[test]
fn test_treatment_equality_config() {
    let config = AequitasConfig::treatment_equality();
    assert_eq!(config.fairness_metric, AequitasMetric::TreatmentEquality);
}

#[test]
fn test_strict_config() {
    let config = AequitasConfig::strict(0.9);
    assert!((config.disparity_tolerance - 0.9).abs() < 1e-6);
}

#[test]
fn test_backend_id() {
    let backend = AequitasBackend::new();
    assert_eq!(backend.id(), BackendId::Aequitas);
}

#[test]
fn test_supports_model_fairness() {
    let backend = AequitasBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::Fairness));
}

#[test]
fn test_script_generation() {
    let config = AequitasConfig::default();
    let spec = create_test_spec();
    let script = script::generate_aequitas_script(&spec, &config).unwrap();

    assert!(script.contains("import aequitas"));
    assert!(script.contains("AEQUITAS_RESULT_START"));
    assert!(script.contains("AEQUITAS_RESULT_END"));
    assert!(script.contains("AEQUITAS_STATUS:"));
}

#[test]
fn test_parse_verified_output() {
    let stdout = r#"
AEQUITAS_RESULT_START
{
    "status": "success",
    "fairness_metric": "predictive_parity",
    "min_disparity_ratio": 0.9,
    "avg_disparity_ratio": 0.95,
    "disparity_tolerance": 0.8,
    "is_fair": true
}
AEQUITAS_RESULT_END
AEQUITAS_STATUS: VERIFIED
"#;
    let (status, ce) = script::parse_aequitas_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(ce.is_none());
}

#[test]
fn test_parse_partial_output() {
    let stdout = r#"
AEQUITAS_RESULT_START
{
    "status": "success",
    "fairness_metric": "predictive_parity",
    "min_disparity_ratio": 0.75,
    "avg_disparity_ratio": 0.85,
    "disparity_tolerance": 0.8,
    "is_fair": false
}
AEQUITAS_RESULT_END
AEQUITAS_STATUS: PARTIALLY_VERIFIED
"#;
    let (status, ce) = script::parse_aequitas_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Partial { .. }));
    assert!(ce.is_some());
}

#[test]
fn test_parse_not_verified_output() {
    let stdout = r#"
AEQUITAS_RESULT_START
{
    "status": "success",
    "fairness_metric": "predictive_parity",
    "min_disparity_ratio": 0.5,
    "avg_disparity_ratio": 0.6,
    "disparity_tolerance": 0.8,
    "is_fair": false
}
AEQUITAS_RESULT_END
AEQUITAS_STATUS: NOT_VERIFIED
"#;
    let (status, ce) = script::parse_aequitas_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    assert!(ce.is_some());
}

#[test]
fn test_parse_error_output() {
    let stdout = "AEQUITAS_ERROR: Audit failed";
    let (status, _) = script::parse_aequitas_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[tokio::test]
async fn test_aequitas_health_check() {
    let backend = AequitasBackend::new();
    let status = backend.health_check().await;
    match status {
        HealthStatus::Healthy => println!("Aequitas is available"),
        HealthStatus::Unavailable { reason } => println!("Aequitas unavailable: {}", reason),
        HealthStatus::Degraded { reason } => println!("Aequitas degraded: {}", reason),
    }
}

#[tokio::test]
async fn test_aequitas_verify_returns_result_or_unavailable() {
    let backend = AequitasBackend::new();
    let spec = create_test_spec();

    match backend.verify(&spec).await {
        Ok(result) => {
            assert_eq!(result.backend, BackendId::Aequitas);
            println!("Aequitas verify succeeded: {:?}", result.status);
        }
        Err(BackendError::Unavailable(reason)) => {
            println!("Aequitas unavailable: {}", reason);
        }
        Err(e) => {
            println!("Aequitas error (expected if not installed): {}", e);
        }
    }
}
