//! Tests for ART backend

use super::config::{ArtConfig, AttackType};
use super::script::{generate_art_script, parse_art_output};
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
    assert_eq!(AttackType::FGSM.art_class(), "FastGradientMethod");
    assert_eq!(AttackType::PGD.art_class(), "ProjectedGradientDescent");
    assert_eq!(AttackType::CW.art_class(), "CarliniL2Method");
    assert_eq!(AttackType::DeepFool.art_class(), "DeepFool");
    assert_eq!(
        AttackType::AutoPGD.art_class(),
        "AutoProjectedGradientDescent"
    );
}

#[test]
fn test_attack_modules() {
    assert_eq!(AttackType::FGSM.art_module(), "art.attacks.evasion");
    assert_eq!(AttackType::PGD.art_module(), "art.attacks.evasion");
}

#[test]
fn test_default_config() {
    let config = ArtConfig::default();
    assert_eq!(config.epsilon, 0.3);
    assert_eq!(config.attack_type, AttackType::FGSM);
    assert_eq!(config.num_samples, 100);
}

#[test]
fn test_pgd_config() {
    let config = ArtConfig::pgd();
    assert_eq!(config.attack_type, AttackType::PGD);
    assert_eq!(config.epsilon, 0.031);
    assert_eq!(config.max_iter, 40);
}

#[test]
fn test_auto_pgd_config() {
    let config = ArtConfig::auto_pgd();
    assert_eq!(config.attack_type, AttackType::AutoPGD);
    assert_eq!(config.max_iter, 100);
}

#[test]
fn test_backend_id() {
    let backend = super::ArtBackend::new();
    assert_eq!(backend.id(), BackendId::ART);
}

#[test]
fn test_backend_supports() {
    let backend = super::ArtBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::AdversarialRobustness));
}

#[test]
fn test_generate_script() {
    let spec = create_test_spec();
    let config = ArtConfig::default();

    let script = generate_art_script(&spec, &config).unwrap();
    assert!(script.contains("import art"));
    assert!(script.contains("FastGradientMethod"));
    assert!(script.contains("ART_RESULT_START"));
    assert!(script.contains("ART_RESULT_END"));
}

#[test]
fn test_parse_output_robust() {
    let output = r#"
ART_INFO: Using synthetic test model
ART_RESULT_START
{
  "clean_accuracy": 0.95,
  "adversarial_accuracy": 0.94,
  "robust_accuracy": 0.94,
  "attack_success_rate": 0.005,
  "num_adversarial_examples": 5,
  "max_perturbation": 0.1,
  "mean_perturbation": 0.05,
  "epsilon": 0.3,
  "attack_type": "FastGradientMethod",
  "num_samples": 100
}
ART_RESULT_END

ART_SUMMARY: Clean accuracy: 95.00%, Robust accuracy: 94.00%
ART_STATUS: ROBUST
"#;

    let (status, counterexample) = parse_art_output(output, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(counterexample.is_none());
}

#[test]
fn test_parse_output_not_robust() {
    let output = r#"
ART_RESULT_START
{
  "clean_accuracy": 0.95,
  "adversarial_accuracy": 0.20,
  "robust_accuracy": 0.20,
  "attack_success_rate": 0.79,
  "num_adversarial_examples": 75,
  "max_perturbation": 0.3,
  "mean_perturbation": 0.15,
  "epsilon": 0.3,
  "attack_type": "FastGradientMethod",
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
ART_RESULT_END
ART_STATUS: NOT_ROBUST
"#;

    let (status, counterexample) = parse_art_output(output, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    assert!(counterexample.is_some());

    let ce = counterexample.unwrap();
    // Check that witness contains the expected keys
    assert!(ce.witness.contains_key("original_input"));
    assert!(ce.witness.contains_key("adversarial_input"));
}

#[test]
fn test_parse_output_partial() {
    let output = r#"
ART_RESULT_START
{
  "clean_accuracy": 0.95,
  "adversarial_accuracy": 0.90,
  "robust_accuracy": 0.90,
  "attack_success_rate": 0.05,
  "num_adversarial_examples": 5,
  "max_perturbation": 0.1,
  "mean_perturbation": 0.05,
  "epsilon": 0.3,
  "attack_type": "FastGradientMethod",
  "num_samples": 100
}
ART_RESULT_END
ART_STATUS: PARTIALLY_ROBUST
"#;

    let (status, _) = parse_art_output(output, "");
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
    let output = "ART_ERROR: Missing dependency: torch";
    let (status, _) = parse_art_output(output, "");

    match status {
        VerificationStatus::Unknown { reason } => {
            assert!(reason.contains("Missing dependency"));
        }
        _ => panic!("Expected Unknown status with error"),
    }
}

#[test]
fn test_parse_output_fallback_status() {
    // Test fallback parsing when JSON is not available
    let output = "Some other output\nART_STATUS: ROBUST\n";
    let (status, _) = parse_art_output(output, "");
    assert!(matches!(status, VerificationStatus::Proven));
}

#[test]
fn test_different_attack_types_in_script() {
    let spec = create_test_spec();

    // Test PGD config
    let pgd_config = ArtConfig::pgd();
    let pgd_script = generate_art_script(&spec, &pgd_config).unwrap();
    assert!(pgd_script.contains("ProjectedGradientDescent"));

    // Test AutoPGD config
    let auto_config = ArtConfig::auto_pgd();
    let auto_script = generate_art_script(&spec, &auto_config).unwrap();
    assert!(auto_script.contains("AutoProjectedGradientDescent"));
}

#[test]
fn test_cw_attack_in_script() {
    let spec = create_test_spec();
    let config = ArtConfig {
        attack_type: AttackType::CW,
        ..Default::default()
    };

    let script = generate_art_script(&spec, &config).unwrap();
    assert!(script.contains("CarliniL2Method"));
}

#[test]
fn test_deepfool_attack_in_script() {
    let spec = create_test_spec();
    let config = ArtConfig {
        attack_type: AttackType::DeepFool,
        ..Default::default()
    };

    let script = generate_art_script(&spec, &config).unwrap();
    assert!(script.contains("DeepFool"));
}
