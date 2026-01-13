//! Tests for Captum backend

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
fn test_method_strings() {
    assert_eq!(
        CaptumMethod::IntegratedGradients.as_str(),
        "integrated_gradients"
    );
    assert_eq!(CaptumMethod::Saliency.as_str(), "saliency");
    assert_eq!(CaptumMethod::DeepLift.as_str(), "deeplift");
    assert_eq!(CaptumMethod::GradientShap.as_str(), "gradient_shap");
}

#[test]
fn test_default_config_values() {
    let cfg = CaptumConfig::default();
    assert_eq!(cfg.method, CaptumMethod::IntegratedGradients);
    assert_eq!(cfg.steps, 50);
    assert!(!cfg.use_noise_tunnel);
    assert_eq!(cfg.top_k, 5);
    assert!(cfg.attribution_threshold > 0.0);
}

#[test]
fn test_noise_tunnel_builder() {
    let cfg = CaptumConfig::with_noise_tunnel();
    assert!(cfg.use_noise_tunnel);
}

#[test]
fn test_gradient_shap_builder() {
    let cfg = CaptumConfig::gradient_shap();
    assert_eq!(cfg.method, CaptumMethod::GradientShap);
    assert!(cfg.use_noise_tunnel);
}

#[test]
fn test_backend_id_and_supports() {
    let backend = CaptumBackend::new();
    assert_eq!(backend.id(), BackendId::Captum);
    assert!(backend.supports().contains(&PropertyType::Interpretability));
}

#[test]
fn test_script_generation() {
    let cfg = CaptumConfig::default();
    let spec = create_test_spec();
    let script = script::generate_captum_script(&spec, &cfg).unwrap();

    assert!(script.contains("CAPTUM_RESULT_START"));
    assert!(script.contains("CAPTUM_RESULT_END"));
    assert!(script.contains("CAPTUM_STATUS:"));
}

#[test]
fn test_parse_verified_output() {
    let stdout = r#"
CAPTUM_RESULT_START
{
  "status": "success",
  "attribution_mean": 0.12,
  "attribution_threshold": 0.05,
  "attribution_max": 0.3,
  "stability_gap": 0.01,
  "top_feature": 2
}
CAPTUM_RESULT_END
CAPTUM_STATUS: VERIFIED
"#;
    let (status, ce) = script::parse_captum_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(ce.is_none());
}

#[test]
fn test_parse_partial_output() {
    let stdout = r#"
CAPTUM_RESULT_START
{
  "status": "success",
  "attribution_mean": 0.02,
  "attribution_threshold": 0.05,
  "attribution_max": 0.08,
  "stability_gap": 0.02,
  "top_feature": 1
}
CAPTUM_RESULT_END
CAPTUM_STATUS: PARTIALLY_VERIFIED
"#;
    let (status, ce) = script::parse_captum_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Partial { .. }));
    assert!(ce.is_some());
}

#[test]
fn test_parse_error_output() {
    let stdout = "CAPTUM_ERROR: torch missing";
    let (status, _) = script::parse_captum_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

#[tokio::test]
async fn test_captum_health_check() {
    let backend = CaptumBackend::new();
    let status = backend.health_check().await;
    match status {
        HealthStatus::Healthy => println!("Captum available"),
        HealthStatus::Unavailable { reason } => println!("Captum unavailable: {}", reason),
        HealthStatus::Degraded { reason } => println!("Captum degraded: {}", reason),
    }
}

#[tokio::test]
async fn test_captum_verify_returns_result_or_unavailable() {
    let backend = CaptumBackend::new();
    let spec = create_test_spec();

    match backend.verify(&spec).await {
        Ok(result) => {
            assert_eq!(result.backend, BackendId::Captum);
            println!("Captum verification status: {:?}", result.status);
        }
        Err(BackendError::Unavailable(reason)) => {
            println!("Captum unavailable: {}", reason);
        }
        Err(e) => {
            println!("Captum error (expected if deps missing): {}", e);
        }
    }
}
