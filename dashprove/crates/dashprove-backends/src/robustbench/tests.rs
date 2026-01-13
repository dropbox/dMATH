//! Tests for RobustBench backend

use super::config::{RobustBenchConfig, RobustBenchDataset, ThreatModel};
use super::script::{generate_robustbench_script, parse_robustbench_output};
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
fn test_datasets() {
    assert_eq!(RobustBenchDataset::Cifar10.dataset_name(), "cifar10");
    assert_eq!(RobustBenchDataset::Cifar100.dataset_name(), "cifar100");
    assert_eq!(RobustBenchDataset::ImageNet.dataset_name(), "imagenet");
}

#[test]
fn test_threat_models() {
    assert_eq!(ThreatModel::Linf.model_name(), "Linf");
    assert_eq!(ThreatModel::L2.model_name(), "L2");
    assert_eq!(ThreatModel::Corruptions.model_name(), "corruptions");
}

#[test]
fn test_default_epsilon() {
    let eps = 8.0 / 255.0;
    assert!((ThreatModel::Linf.default_epsilon() - eps).abs() < 1e-6);
    assert_eq!(ThreatModel::L2.default_epsilon(), 0.5);
    assert_eq!(ThreatModel::Corruptions.default_epsilon(), 0.0);
}

#[test]
fn test_default_config() {
    let config = RobustBenchConfig::default();
    assert_eq!(config.dataset, RobustBenchDataset::Cifar10);
    assert_eq!(config.threat_model, ThreatModel::Linf);
    assert!((config.epsilon - 8.0 / 255.0).abs() < 1e-6);
    assert_eq!(config.num_samples, 1000);
    assert!(config.use_autoattack);
}

#[test]
fn test_cifar10_linf_config() {
    let config = RobustBenchConfig::cifar10_linf();
    assert_eq!(config.dataset, RobustBenchDataset::Cifar10);
    assert_eq!(config.threat_model, ThreatModel::Linf);
}

#[test]
fn test_cifar10_l2_config() {
    let config = RobustBenchConfig::cifar10_l2();
    assert_eq!(config.dataset, RobustBenchDataset::Cifar10);
    assert_eq!(config.threat_model, ThreatModel::L2);
    assert_eq!(config.epsilon, 0.5);
}

#[test]
fn test_cifar100_config() {
    let config = RobustBenchConfig::cifar100_linf();
    assert_eq!(config.dataset, RobustBenchDataset::Cifar100);
}

#[test]
fn test_imagenet_config() {
    let config = RobustBenchConfig::imagenet_linf();
    assert_eq!(config.dataset, RobustBenchDataset::ImageNet);
    assert_eq!(config.batch_size, 32);
}

#[test]
fn test_corruptions_config() {
    let config = RobustBenchConfig::corruptions();
    assert_eq!(config.threat_model, ThreatModel::Corruptions);
    assert!(!config.use_autoattack);
}

#[test]
fn test_backend_id() {
    let backend = super::RobustBenchBackend::new();
    assert_eq!(backend.id(), BackendId::RobustBench);
}

#[test]
fn test_backend_supports() {
    let backend = super::RobustBenchBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::AdversarialRobustness));
}

#[test]
fn test_generate_script() {
    let spec = create_test_spec();
    let config = RobustBenchConfig::default();

    let script = generate_robustbench_script(&spec, &config).unwrap();
    assert!(script.contains("import torch"));
    assert!(script.contains("robustbench"));
    assert!(script.contains("cifar10"));
    assert!(script.contains("ROBUSTBENCH_RESULT_START"));
    assert!(script.contains("ROBUSTBENCH_RESULT_END"));
}

#[test]
fn test_parse_output_robust() {
    let output = r#"
ROBUSTBENCH_INFO: Loading model
ROBUSTBENCH_RESULT_START
{
  "clean_accuracy": 0.95,
  "robust_accuracy": 0.88,
  "attack_success_rate": 0.074,
  "num_adversarial_examples": 70,
  "max_perturbation": 0.031,
  "mean_perturbation": 0.015,
  "epsilon": 0.031,
  "dataset": "cifar10",
  "threat_model": "Linf",
  "model_name": "Carmon2019Unlabeled",
  "num_samples": 1000,
  "used_autoattack": true
}
ROBUSTBENCH_RESULT_END

ROBUSTBENCH_SUMMARY: Clean accuracy: 95.00%
ROBUSTBENCH_STATUS: ROBUST
"#;

    let (status, counterexample) = parse_robustbench_output(output, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(counterexample.is_none());
}

#[test]
fn test_parse_output_not_robust() {
    let output = r#"
ROBUSTBENCH_RESULT_START
{
  "clean_accuracy": 0.95,
  "robust_accuracy": 0.25,
  "attack_success_rate": 0.74,
  "num_adversarial_examples": 700,
  "max_perturbation": 0.031,
  "mean_perturbation": 0.020,
  "epsilon": 0.031,
  "dataset": "cifar10",
  "threat_model": "Linf",
  "model_name": "Standard",
  "num_samples": 1000,
  "used_autoattack": true,
  "adversarial_example": {
    "original_input": [0.1, 0.2, 0.3],
    "adversarial_input": [0.13, 0.17, 0.33],
    "original_prediction": 7,
    "adversarial_prediction": 3,
    "true_label": 7,
    "perturbation_norm": 0.031
  }
}
ROBUSTBENCH_RESULT_END
ROBUSTBENCH_STATUS: NOT_ROBUST
"#;

    let (status, counterexample) = parse_robustbench_output(output, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    assert!(counterexample.is_some());

    let ce = counterexample.unwrap();
    assert!(ce.witness.contains_key("original_input"));
    assert!(ce.witness.contains_key("adversarial_input"));
}

#[test]
fn test_parse_output_partial() {
    let output = r#"
ROBUSTBENCH_RESULT_START
{
  "clean_accuracy": 0.95,
  "robust_accuracy": 0.60,
  "attack_success_rate": 0.37,
  "num_adversarial_examples": 350,
  "max_perturbation": 0.031,
  "mean_perturbation": 0.018,
  "epsilon": 0.031,
  "dataset": "cifar10",
  "threat_model": "Linf",
  "model_name": "Rice2020Overfitting",
  "num_samples": 1000,
  "used_autoattack": true
}
ROBUSTBENCH_RESULT_END
ROBUSTBENCH_STATUS: PARTIALLY_ROBUST
"#;

    let (status, _) = parse_robustbench_output(output, "");
    match status {
        VerificationStatus::Partial {
            verified_percentage,
        } => {
            assert!(verified_percentage > 55.0);
        }
        _ => panic!("Expected Partial status"),
    }
}

#[test]
fn test_parse_output_error() {
    let output = "ROBUSTBENCH_ERROR: Failed to load model: Model not found";
    let (status, _) = parse_robustbench_output(output, "");

    match status {
        VerificationStatus::Unknown { reason } => {
            assert!(reason.contains("Failed to load model"));
        }
        _ => panic!("Expected Unknown status with error"),
    }
}

#[test]
fn test_parse_output_fallback_status() {
    let output = "Some other output\nROBUSTBENCH_STATUS: ROBUST\n";
    let (status, _) = parse_robustbench_output(output, "");
    assert!(matches!(status, VerificationStatus::Proven));
}

#[test]
fn test_different_datasets_in_script() {
    let spec = create_test_spec();

    // Test CIFAR-100 config
    let cifar100_config = RobustBenchConfig::cifar100_linf();
    let cifar100_script = generate_robustbench_script(&spec, &cifar100_config).unwrap();
    assert!(cifar100_script.contains("cifar100"));

    // Test ImageNet config
    let imagenet_config = RobustBenchConfig::imagenet_linf();
    let imagenet_script = generate_robustbench_script(&spec, &imagenet_config).unwrap();
    assert!(imagenet_script.contains("imagenet"));
}

#[test]
fn test_l2_threat_model_in_script() {
    let spec = create_test_spec();
    let config = RobustBenchConfig::cifar10_l2();

    let script = generate_robustbench_script(&spec, &config).unwrap();
    assert!(script.contains("L2"));
}
