//! Tests for Fairlearn backend

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
fn test_fairness_metric_strings() {
    assert_eq!(
        FairnessMetric::DemographicParity.as_str(),
        "demographic_parity"
    );
    assert_eq!(FairnessMetric::EqualizedOdds.as_str(), "equalized_odds");
    assert_eq!(
        FairnessMetric::EqualOpportunity.as_str(),
        "equal_opportunity"
    );
    assert_eq!(
        FairnessMetric::TruePositiveRateParity.as_str(),
        "true_positive_rate_parity"
    );
    assert_eq!(
        FairnessMetric::FalsePositiveRateParity.as_str(),
        "false_positive_rate_parity"
    );
}

#[test]
fn test_mitigation_method_strings() {
    assert_eq!(MitigationMethod::None.as_str(), "none");
    assert_eq!(
        MitigationMethod::ExponentiatedGradient.as_str(),
        "exponentiated_gradient"
    );
    assert_eq!(MitigationMethod::GridSearch.as_str(), "grid_search");
    assert_eq!(
        MitigationMethod::ThresholdOptimizer.as_str(),
        "threshold_optimizer"
    );
}

#[test]
fn test_fairness_constraint_strings() {
    assert_eq!(
        FairnessConstraint::DemographicParity.as_str(),
        "demographic_parity"
    );
    assert_eq!(FairnessConstraint::EqualizedOdds.as_str(), "equalized_odds");
    assert_eq!(
        FairnessConstraint::TruePositiveRateParity.as_str(),
        "true_positive_rate_parity"
    );
    assert_eq!(
        FairnessConstraint::BoundedGroupLoss.as_str(),
        "bounded_group_loss"
    );
}

#[test]
fn test_default_config() {
    let config = FairlearnConfig::default();
    assert_eq!(config.fairness_metric, FairnessMetric::DemographicParity);
    assert_eq!(config.mitigation_method, MitigationMethod::None);
    assert_eq!(
        config.fairness_constraint,
        FairnessConstraint::DemographicParity
    );
    assert!((config.fairness_threshold - 0.1).abs() < 1e-6);
    assert_eq!(config.n_samples, 1000);
}

#[test]
fn test_equalized_odds_config() {
    let config = FairlearnConfig::equalized_odds();
    assert_eq!(config.fairness_metric, FairnessMetric::EqualizedOdds);
    assert_eq!(
        config.fairness_constraint,
        FairnessConstraint::EqualizedOdds
    );
}

#[test]
fn test_with_mitigation_config() {
    let config = FairlearnConfig::with_mitigation();
    assert_eq!(
        config.mitigation_method,
        MitigationMethod::ExponentiatedGradient
    );
}

#[test]
fn test_threshold_optimizer_config() {
    let config = FairlearnConfig::threshold_optimizer();
    assert_eq!(
        config.mitigation_method,
        MitigationMethod::ThresholdOptimizer
    );
}

#[test]
fn test_strict_config() {
    let config = FairlearnConfig::strict(0.05);
    assert!((config.fairness_threshold - 0.05).abs() < 1e-6);
}

#[test]
fn test_backend_id() {
    let backend = FairlearnBackend::new();
    assert_eq!(backend.id(), BackendId::Fairlearn);
}

#[test]
fn test_supports_model_fairness() {
    let backend = FairlearnBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::Fairness));
}

#[test]
fn test_script_generation() {
    let config = FairlearnConfig::default();
    let spec = create_test_spec();
    let script = script::generate_fairlearn_script(&spec, &config).unwrap();

    assert!(script.contains("import fairlearn"));
    assert!(script.contains("FAIRLEARN_RESULT_START"));
    assert!(script.contains("FAIRLEARN_RESULT_END"));
    assert!(script.contains("FAIRLEARN_STATUS:"));
}

#[test]
fn test_parse_verified_output() {
    let stdout = r#"
FAIRLEARN_RESULT_START
{
    "status": "success",
    "fairness_metric": "demographic_parity",
    "primary_metric_value": 0.05,
    "fairness_threshold": 0.1,
    "is_fair": true
}
FAIRLEARN_RESULT_END
FAIRLEARN_STATUS: VERIFIED
"#;
    let (status, ce) = script::parse_fairlearn_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(ce.is_none());
}

#[test]
fn test_parse_partial_output() {
    let stdout = r#"
FAIRLEARN_RESULT_START
{
    "status": "success",
    "fairness_metric": "demographic_parity",
    "primary_metric_value": 0.15,
    "fairness_threshold": 0.1,
    "is_fair": false
}
FAIRLEARN_RESULT_END
FAIRLEARN_STATUS: PARTIALLY_VERIFIED
"#;
    let (status, ce) = script::parse_fairlearn_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Partial { .. }));
    assert!(ce.is_some());
}

#[test]
fn test_parse_not_verified_output() {
    let stdout = r#"
FAIRLEARN_RESULT_START
{
    "status": "success",
    "fairness_metric": "demographic_parity",
    "primary_metric_value": 0.5,
    "fairness_threshold": 0.1,
    "is_fair": false
}
FAIRLEARN_RESULT_END
FAIRLEARN_STATUS: NOT_VERIFIED
"#;
    let (status, ce) = script::parse_fairlearn_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    assert!(ce.is_some());
}

#[test]
fn test_parse_error_output() {
    let stdout = "FAIRLEARN_ERROR: Assessment failed";
    let (status, _) = script::parse_fairlearn_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[tokio::test]
async fn test_fairlearn_health_check() {
    let backend = FairlearnBackend::new();
    let status = backend.health_check().await;
    match status {
        HealthStatus::Healthy => println!("Fairlearn is available"),
        HealthStatus::Unavailable { reason } => println!("Fairlearn unavailable: {}", reason),
        HealthStatus::Degraded { reason } => println!("Fairlearn degraded: {}", reason),
    }
}

#[tokio::test]
async fn test_fairlearn_verify_returns_result_or_unavailable() {
    let backend = FairlearnBackend::new();
    let spec = create_test_spec();

    match backend.verify(&spec).await {
        Ok(result) => {
            assert_eq!(result.backend, BackendId::Fairlearn);
            println!("Fairlearn verify succeeded: {:?}", result.status);
        }
        Err(BackendError::Unavailable(reason)) => {
            println!("Fairlearn unavailable: {}", reason);
        }
        Err(e) => {
            println!("Fairlearn error (expected if not installed): {}", e);
        }
    }
}
