//! Tests for AIF360 backend

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
fn test_bias_metric_strings() {
    assert_eq!(
        BiasMetric::StatisticalParityDifference.as_str(),
        "statistical_parity_difference"
    );
    assert_eq!(BiasMetric::DisparateImpact.as_str(), "disparate_impact");
    assert_eq!(
        BiasMetric::AverageOddsDifference.as_str(),
        "average_odds_difference"
    );
    assert_eq!(
        BiasMetric::EqualOpportunityDifference.as_str(),
        "equal_opportunity_difference"
    );
    assert_eq!(BiasMetric::TheilIndex.as_str(), "theil_index");
}

#[test]
fn test_mitigation_algorithm_strings() {
    assert_eq!(AIF360MitigationAlgorithm::None.as_str(), "none");
    assert_eq!(AIF360MitigationAlgorithm::Reweighing.as_str(), "reweighing");
    assert_eq!(
        AIF360MitigationAlgorithm::DisparateImpactRemover.as_str(),
        "disparate_impact_remover"
    );
    assert_eq!(AIF360MitigationAlgorithm::LFR.as_str(), "lfr");
    assert_eq!(
        AIF360MitigationAlgorithm::PrejudiceRemover.as_str(),
        "prejudice_remover"
    );
    assert_eq!(
        AIF360MitigationAlgorithm::AdversarialDebiasing.as_str(),
        "adversarial_debiasing"
    );
    assert_eq!(
        AIF360MitigationAlgorithm::CalibratedEqOdds.as_str(),
        "calibrated_eq_odds"
    );
}

#[test]
fn test_default_config() {
    let config = AIF360Config::default();
    assert_eq!(config.bias_metric, BiasMetric::StatisticalParityDifference);
    assert_eq!(config.mitigation_algorithm, AIF360MitigationAlgorithm::None);
    assert!((config.fairness_threshold - 0.1).abs() < 1e-6);
    assert!((config.disparate_impact_threshold - 0.8).abs() < 1e-6);
    assert_eq!(config.n_samples, 1000);
}

#[test]
fn test_disparate_impact_config() {
    let config = AIF360Config::disparate_impact();
    assert_eq!(config.bias_metric, BiasMetric::DisparateImpact);
}

#[test]
fn test_with_reweighing_config() {
    let config = AIF360Config::with_reweighing();
    assert_eq!(
        config.mitigation_algorithm,
        AIF360MitigationAlgorithm::Reweighing
    );
}

#[test]
fn test_equalized_odds_config() {
    let config = AIF360Config::equalized_odds();
    assert_eq!(config.bias_metric, BiasMetric::AverageOddsDifference);
}

#[test]
fn test_calibrated_eq_odds_config() {
    let config = AIF360Config::calibrated_eq_odds();
    assert_eq!(
        config.mitigation_algorithm,
        AIF360MitigationAlgorithm::CalibratedEqOdds
    );
}

#[test]
fn test_backend_id() {
    let backend = AIF360Backend::new();
    assert_eq!(backend.id(), BackendId::AIF360);
}

#[test]
fn test_supports_model_fairness() {
    let backend = AIF360Backend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::Fairness));
}

#[test]
fn test_script_generation() {
    let config = AIF360Config::default();
    let spec = create_test_spec();
    let script = script::generate_aif360_script(&spec, &config).unwrap();

    assert!(script.contains("import aif360"));
    assert!(script.contains("AIF360_RESULT_START"));
    assert!(script.contains("AIF360_RESULT_END"));
    assert!(script.contains("AIF360_STATUS:"));
}

#[test]
fn test_parse_verified_output() {
    let stdout = r#"
AIF360_RESULT_START
{
    "status": "success",
    "bias_metric": "statistical_parity_difference",
    "primary_metric_value": 0.05,
    "fairness_threshold": 0.1,
    "is_fair": true
}
AIF360_RESULT_END
AIF360_STATUS: VERIFIED
"#;
    let (status, ce) = script::parse_aif360_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(ce.is_none());
}

#[test]
fn test_parse_partial_output() {
    let stdout = r#"
AIF360_RESULT_START
{
    "status": "success",
    "bias_metric": "statistical_parity_difference",
    "primary_metric_value": 0.15,
    "fairness_threshold": 0.1,
    "is_fair": false
}
AIF360_RESULT_END
AIF360_STATUS: PARTIALLY_VERIFIED
"#;
    let (status, ce) = script::parse_aif360_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Partial { .. }));
    assert!(ce.is_some());
}

#[test]
fn test_parse_not_verified_output() {
    let stdout = r#"
AIF360_RESULT_START
{
    "status": "success",
    "bias_metric": "statistical_parity_difference",
    "primary_metric_value": 0.5,
    "fairness_threshold": 0.1,
    "is_fair": false
}
AIF360_RESULT_END
AIF360_STATUS: NOT_VERIFIED
"#;
    let (status, ce) = script::parse_aif360_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    assert!(ce.is_some());
}

#[test]
fn test_parse_error_output() {
    let stdout = "AIF360_ERROR: Assessment failed";
    let (status, _) = script::parse_aif360_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[tokio::test]
async fn test_aif360_health_check() {
    let backend = AIF360Backend::new();
    let status = backend.health_check().await;
    match status {
        HealthStatus::Healthy => println!("AIF360 is available"),
        HealthStatus::Unavailable { reason } => println!("AIF360 unavailable: {}", reason),
        HealthStatus::Degraded { reason } => println!("AIF360 degraded: {}", reason),
    }
}

#[tokio::test]
async fn test_aif360_verify_returns_result_or_unavailable() {
    let backend = AIF360Backend::new();
    let spec = create_test_spec();

    match backend.verify(&spec).await {
        Ok(result) => {
            assert_eq!(result.backend, BackendId::AIF360);
            println!("AIF360 verify succeeded: {:?}", result.status);
        }
        Err(BackendError::Unavailable(reason)) => {
            println!("AIF360 unavailable: {}", reason);
        }
        Err(e) => {
            println!("AIF360 error (expected if not installed): {}", e);
        }
    }
}
