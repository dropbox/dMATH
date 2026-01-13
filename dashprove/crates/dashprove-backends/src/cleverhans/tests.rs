//! Tests for CleverHans backend

use super::config::{CleverHansAttack, CleverHansConfig};
use super::script::{generate_cleverhans_script, parse_cleverhans_output};
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
        CleverHansAttack::FGSM.function_name(),
        "fast_gradient_method"
    );
    assert_eq!(
        CleverHansAttack::BIM.function_name(),
        "projected_gradient_descent"
    );
    assert_eq!(
        CleverHansAttack::MIM.function_name(),
        "momentum_iterative_method"
    );
    assert_eq!(CleverHansAttack::SPSA.function_name(), "spsa");
    assert_eq!(
        CleverHansAttack::CarliniWagnerL2.function_name(),
        "carlini_wagner_l2"
    );
    assert_eq!(
        CleverHansAttack::ElasticNet.function_name(),
        "elastic_net_method"
    );
}

#[test]
fn test_attack_modules() {
    assert_eq!(CleverHansAttack::FGSM.module(), "cleverhans.torch.attacks");
    assert_eq!(CleverHansAttack::BIM.module(), "cleverhans.torch.attacks");
}

#[test]
fn test_default_config() {
    let config = CleverHansConfig::default();
    assert_eq!(config.epsilon, 0.3);
    assert_eq!(config.attack_type, CleverHansAttack::FGSM);
    assert_eq!(config.num_samples, 100);
    assert_eq!(config.clip_min, 0.0);
    assert_eq!(config.clip_max, 1.0);
}

#[test]
fn test_bim_config() {
    let config = CleverHansConfig::bim();
    assert_eq!(config.attack_type, CleverHansAttack::BIM);
    assert_eq!(config.epsilon, 0.031);
    assert_eq!(config.nb_iter, 40);
}

#[test]
fn test_mim_config() {
    let config = CleverHansConfig::mim();
    assert_eq!(config.attack_type, CleverHansAttack::MIM);
    assert_eq!(config.epsilon, 0.031);
}

#[test]
fn test_carlini_wagner_config() {
    let config = CleverHansConfig::carlini_wagner();
    assert_eq!(config.attack_type, CleverHansAttack::CarliniWagnerL2);
    assert_eq!(config.epsilon, 0.5);
    assert_eq!(config.nb_iter, 1000);
}

#[test]
fn test_backend_id() {
    let backend = super::CleverHansBackend::new();
    assert_eq!(backend.id(), BackendId::CleverHans);
}

#[test]
fn test_backend_supports() {
    let backend = super::CleverHansBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::AdversarialRobustness));
}

#[test]
fn test_generate_script() {
    let spec = create_test_spec();
    let config = CleverHansConfig::default();

    let script = generate_cleverhans_script(&spec, &config).unwrap();
    assert!(script.contains("import torch"));
    assert!(script.contains("cleverhans"));
    assert!(script.contains("fast_gradient_method"));
    assert!(script.contains("CLEVERHANS_RESULT_START"));
    assert!(script.contains("CLEVERHANS_RESULT_END"));
}

#[test]
fn test_parse_output_robust() {
    let output = r#"
CLEVERHANS_INFO: Using synthetic test model
CLEVERHANS_RESULT_START
{
  "clean_accuracy": 0.95,
  "adversarial_accuracy": 0.94,
  "robust_accuracy": 0.94,
  "attack_success_rate": 0.005,
  "num_adversarial_examples": 5,
  "max_perturbation": 0.1,
  "mean_perturbation": 0.05,
  "epsilon": 0.3,
  "attack_type": "fast_gradient_method",
  "num_samples": 100
}
CLEVERHANS_RESULT_END

CLEVERHANS_SUMMARY: Clean accuracy: 95.00%, Robust accuracy: 94.00%
CLEVERHANS_STATUS: ROBUST
"#;

    let (status, counterexample) = parse_cleverhans_output(output, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(counterexample.is_none());
}

#[test]
fn test_parse_output_not_robust() {
    let output = r#"
CLEVERHANS_RESULT_START
{
  "clean_accuracy": 0.95,
  "adversarial_accuracy": 0.20,
  "robust_accuracy": 0.20,
  "attack_success_rate": 0.79,
  "num_adversarial_examples": 75,
  "max_perturbation": 0.3,
  "mean_perturbation": 0.15,
  "epsilon": 0.3,
  "attack_type": "fast_gradient_method",
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
CLEVERHANS_RESULT_END
CLEVERHANS_STATUS: NOT_ROBUST
"#;

    let (status, counterexample) = parse_cleverhans_output(output, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    assert!(counterexample.is_some());

    let ce = counterexample.unwrap();
    assert!(ce.witness.contains_key("original_input"));
    assert!(ce.witness.contains_key("adversarial_input"));
}

#[test]
fn test_parse_output_partial() {
    let output = r#"
CLEVERHANS_RESULT_START
{
  "clean_accuracy": 0.95,
  "adversarial_accuracy": 0.90,
  "robust_accuracy": 0.90,
  "attack_success_rate": 0.05,
  "num_adversarial_examples": 5,
  "max_perturbation": 0.1,
  "mean_perturbation": 0.05,
  "epsilon": 0.3,
  "attack_type": "fast_gradient_method",
  "num_samples": 100
}
CLEVERHANS_RESULT_END
CLEVERHANS_STATUS: PARTIALLY_ROBUST
"#;

    let (status, _) = parse_cleverhans_output(output, "");
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
    let output = "CLEVERHANS_ERROR: Missing dependency: torch";
    let (status, _) = parse_cleverhans_output(output, "");

    match status {
        VerificationStatus::Unknown { reason } => {
            assert!(reason.contains("Missing dependency"));
        }
        _ => panic!("Expected Unknown status with error"),
    }
}

#[test]
fn test_parse_output_fallback_status() {
    let output = "Some other output\nCLEVERHANS_STATUS: ROBUST\n";
    let (status, _) = parse_cleverhans_output(output, "");
    assert!(matches!(status, VerificationStatus::Proven));
}

#[test]
fn test_different_attack_types_in_script() {
    let spec = create_test_spec();

    // Test BIM config
    let bim_config = CleverHansConfig::bim();
    let bim_script = generate_cleverhans_script(&spec, &bim_config).unwrap();
    assert!(bim_script.contains("projected_gradient_descent"));

    // Test MIM config
    let mim_config = CleverHansConfig::mim();
    let mim_script = generate_cleverhans_script(&spec, &mim_config).unwrap();
    assert!(mim_script.contains("momentum_iterative_method"));
}

#[test]
fn test_cw_attack_in_script() {
    let spec = create_test_spec();
    let config = CleverHansConfig::carlini_wagner();

    let script = generate_cleverhans_script(&spec, &config).unwrap();
    assert!(script.contains("carlini_wagner_l2"));
}
