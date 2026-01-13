//! Tests for Foolbox backend

use super::config::{FoolboxAttack, FoolboxConfig};
use super::script::{generate_foolbox_script, parse_foolbox_output};
use crate::traits::{BackendId, PropertyType, VerificationBackend, VerificationStatus};
use dashprove_usl::ast::Spec;
use dashprove_usl::typecheck::TypedSpec;
use std::collections::HashMap;

fn create_test_spec() -> TypedSpec {
    TypedSpec {
        spec: Spec::default(),
        type_info: HashMap::new(),
    }
}

#[test]
fn test_attack_types() {
    assert_eq!(
        FoolboxAttack::FGSM.foolbox_class(),
        "LinfFastGradientAttack"
    );
    assert_eq!(FoolboxAttack::LinfPGD.foolbox_class(), "LinfPGD");
    assert_eq!(FoolboxAttack::L2PGD.foolbox_class(), "L2PGD");
    assert_eq!(
        FoolboxAttack::CarliniWagner.foolbox_class(),
        "L2CarliniWagnerAttack"
    );
    assert_eq!(
        FoolboxAttack::DeepFool.foolbox_class(),
        "LinfDeepFoolAttack"
    );
}

#[test]
fn test_default_config() {
    let config = FoolboxConfig::default();
    assert_eq!(config.epsilon, 0.3);
    assert_eq!(config.attack_type, FoolboxAttack::FGSM);
    assert_eq!(config.num_samples, 100);
}

#[test]
fn test_pgd_config() {
    let config = FoolboxConfig::pgd();
    assert_eq!(config.attack_type, FoolboxAttack::LinfPGD);
    assert_eq!(config.epsilon, 0.031);
}

#[test]
fn test_deepfool_config() {
    let config = FoolboxConfig::deepfool();
    assert_eq!(config.attack_type, FoolboxAttack::DeepFool);
    assert_eq!(config.steps, 50);
}

#[test]
fn test_backend_id() {
    let backend = super::FoolboxBackend::new();
    assert_eq!(backend.id(), BackendId::Foolbox);
}

#[test]
fn test_backend_supports() {
    let backend = super::FoolboxBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::AdversarialRobustness));
}

#[test]
fn test_generate_script() {
    let spec = create_test_spec();
    let config = FoolboxConfig::default();

    let script = generate_foolbox_script(&spec, &config).unwrap();
    assert!(script.contains("import foolbox"));
    assert!(script.contains("LinfFastGradientAttack"));
    assert!(script.contains("FOOLBOX_RESULT_START"));
    assert!(script.contains("FOOLBOX_RESULT_END"));
}

#[test]
fn test_parse_output_robust() {
    let output = r#"
FOOLBOX_INFO: Using synthetic test model
FOOLBOX_RESULT_START
{
  "clean_accuracy": 0.95,
  "adversarial_accuracy": 0.94,
  "robust_accuracy": 0.94,
  "attack_success_rate": 0.005,
  "num_adversarial_examples": 5,
  "max_perturbation": 0.1,
  "mean_perturbation": 0.05,
  "epsilon": 0.3,
  "attack_type": "LinfFastGradientAttack",
  "num_samples": 100
}
FOOLBOX_RESULT_END
FOOLBOX_STATUS: ROBUST
"#;

    let (status, counterexample) = parse_foolbox_output(output, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(counterexample.is_none());
}

#[test]
fn test_parse_output_not_robust() {
    let output = r#"
FOOLBOX_RESULT_START
{
  "clean_accuracy": 0.95,
  "adversarial_accuracy": 0.20,
  "robust_accuracy": 0.20,
  "attack_success_rate": 0.75,
  "num_adversarial_examples": 75,
  "max_perturbation": 0.3,
  "mean_perturbation": 0.15,
  "epsilon": 0.3,
  "attack_type": "LinfFastGradientAttack",
  "num_samples": 100,
  "adversarial_example": {
    "original_input": [0.1, 0.2, 0.3],
    "adversarial_input": [0.4, 0.5, 0.6],
    "original_prediction": 7,
    "adversarial_prediction": 3,
    "true_label": 7,
    "perturbation_norm": 0.25
  }
}
FOOLBOX_RESULT_END
FOOLBOX_STATUS: NOT_ROBUST
"#;

    let (status, counterexample) = parse_foolbox_output(output, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    assert!(counterexample.is_some());

    let ce = counterexample.unwrap();
    assert!(ce.witness.contains_key("original_input"));
    assert!(ce.witness.contains_key("adversarial_input"));
}

#[test]
fn test_parse_output_partial() {
    let output = r#"
FOOLBOX_RESULT_START
{
  "clean_accuracy": 0.95,
  "adversarial_accuracy": 0.90,
  "robust_accuracy": 0.90,
  "attack_success_rate": 0.05,
  "num_adversarial_examples": 5,
  "max_perturbation": 0.1,
  "mean_perturbation": 0.05,
  "epsilon": 0.3,
  "attack_type": "LinfFastGradientAttack",
  "num_samples": 100
}
FOOLBOX_RESULT_END
FOOLBOX_STATUS: PARTIALLY_ROBUST
"#;

    let (status, _) = parse_foolbox_output(output, "");
    match status {
        VerificationStatus::Partial {
            verified_percentage,
        } => {
            assert!(verified_percentage > 85.0);
        }
        _ => panic!("Expected Partial status"),
    }
}

#[test]
fn test_parse_output_error() {
    let output = "FOOLBOX_ERROR: Missing dependency: torch";
    let (status, _) = parse_foolbox_output(output, "");

    match status {
        VerificationStatus::Unknown { reason } => {
            assert!(reason.contains("Missing dependency"));
        }
        _ => panic!("Expected Unknown status with error"),
    }
}

#[test]
fn test_parse_output_fallback_status() {
    let output = "Some other output\nFOOLBOX_STATUS: ROBUST\n";
    let (status, _) = parse_foolbox_output(output, "");
    assert!(matches!(status, VerificationStatus::Proven));
}

#[test]
fn test_pgd_attack_in_script() {
    let spec = create_test_spec();
    let config = FoolboxConfig::pgd();

    let script = generate_foolbox_script(&spec, &config).unwrap();
    assert!(script.contains("LinfPGD"));
}

#[test]
fn test_deepfool_attack_in_script() {
    let spec = create_test_spec();
    let config = FoolboxConfig::deepfool();

    let script = generate_foolbox_script(&spec, &config).unwrap();
    assert!(script.contains("LinfDeepFoolAttack"));
}
