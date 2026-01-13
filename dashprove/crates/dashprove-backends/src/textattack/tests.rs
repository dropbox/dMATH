//! Tests for TextAttack backend

use super::config::{TextAttackConfig, TextAttackRecipe};
use super::script::{generate_textattack_script, parse_textattack_output};
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
fn test_attack_recipes() {
    assert_eq!(TextAttackRecipe::TextFooler.recipe_name(), "textfooler");
    assert_eq!(TextAttackRecipe::BertAttack.recipe_name(), "bert-attack");
    assert_eq!(TextAttackRecipe::BAE.recipe_name(), "bae");
    assert_eq!(TextAttackRecipe::DeepWordBug.recipe_name(), "deepwordbug");
    assert_eq!(TextAttackRecipe::TextBugger.recipe_name(), "textbugger");
    assert_eq!(TextAttackRecipe::PWWS.recipe_name(), "pwws");
    assert_eq!(TextAttackRecipe::CheckList.recipe_name(), "checklist");
    assert_eq!(TextAttackRecipe::A2T.recipe_name(), "a2t");
    assert_eq!(TextAttackRecipe::Clare.recipe_name(), "clare");
}

#[test]
fn test_attack_descriptions() {
    assert!(TextAttackRecipe::TextFooler
        .description()
        .to_lowercase()
        .contains("word substitution"));
    assert!(TextAttackRecipe::DeepWordBug
        .description()
        .to_lowercase()
        .contains("character"));
}

#[test]
fn test_default_config() {
    let config = TextAttackConfig::default();
    assert_eq!(config.attack_recipe, TextAttackRecipe::TextFooler);
    assert_eq!(config.num_examples, 100);
    assert_eq!(config.max_percent_words, 0.2);
    assert_eq!(config.min_similarity, 0.8);
}

#[test]
fn test_bert_attack_config() {
    let config = TextAttackConfig::bert_attack();
    assert_eq!(config.attack_recipe, TextAttackRecipe::BertAttack);
    assert_eq!(config.min_similarity, 0.85);
}

#[test]
fn test_deep_word_bug_config() {
    let config = TextAttackConfig::deep_word_bug();
    assert_eq!(config.attack_recipe, TextAttackRecipe::DeepWordBug);
    assert_eq!(config.max_percent_words, 0.3);
}

#[test]
fn test_pwws_config() {
    let config = TextAttackConfig::pwws();
    assert_eq!(config.attack_recipe, TextAttackRecipe::PWWS);
}

#[test]
fn test_sentiment_config() {
    let config = TextAttackConfig::for_sentiment();
    assert!(config.model_name.unwrap().contains("SST-2"));
    assert_eq!(config.dataset_name, Some("sst2".to_string()));
}

#[test]
fn test_backend_id() {
    let backend = super::TextAttackBackend::new();
    assert_eq!(backend.id(), BackendId::TextAttack);
}

#[test]
fn test_backend_supports() {
    let backend = super::TextAttackBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::AdversarialRobustness));
}

#[test]
fn test_generate_script() {
    let spec = create_test_spec();
    let config = TextAttackConfig::default();

    let script = generate_textattack_script(&spec, &config).unwrap();
    assert!(script.contains("import textattack"));
    assert!(script.contains("textfooler"));
    assert!(script.contains("TEXTATTACK_RESULT_START"));
    assert!(script.contains("TEXTATTACK_RESULT_END"));
}

#[test]
fn test_parse_output_robust() {
    let output = r#"
TEXTATTACK_INFO: Loading model bert-base
TEXTATTACK_RESULT_START
{
  "total_examples": 100,
  "successful_attacks": 3,
  "failed_attacks": 90,
  "skipped_attacks": 7,
  "attack_success_rate": 0.032,
  "robust_accuracy": 0.968,
  "average_queries": 150.5,
  "attack_recipe": "textfooler",
  "model_name": "bert-base",
  "dataset_name": "sst2"
}
TEXTATTACK_RESULT_END

TEXTATTACK_SUMMARY: Attack success rate: 3.20%
TEXTATTACK_STATUS: ROBUST
"#;

    let (status, counterexample) = parse_textattack_output(output, "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert!(counterexample.is_none());
}

#[test]
fn test_parse_output_not_robust() {
    let output = r#"
TEXTATTACK_RESULT_START
{
  "total_examples": 100,
  "successful_attacks": 65,
  "failed_attacks": 30,
  "skipped_attacks": 5,
  "attack_success_rate": 0.684,
  "robust_accuracy": 0.316,
  "average_queries": 200.0,
  "attack_recipe": "textfooler",
  "model_name": "bert-base",
  "dataset_name": "sst2",
  "adversarial_examples": [
    {
      "original_text": "This movie is great!",
      "perturbed_text": "This movie is wonderful!",
      "original_output": 1,
      "perturbed_output": 0,
      "num_queries": 50
    }
  ]
}
TEXTATTACK_RESULT_END
TEXTATTACK_STATUS: NOT_ROBUST
"#;

    let (status, counterexample) = parse_textattack_output(output, "");
    assert!(matches!(status, VerificationStatus::Disproven));
    assert!(counterexample.is_some());

    let ce = counterexample.unwrap();
    assert!(ce.witness.contains_key("original_text"));
    assert!(ce.witness.contains_key("perturbed_text"));
}

#[test]
fn test_parse_output_partial() {
    let output = r#"
TEXTATTACK_RESULT_START
{
  "total_examples": 100,
  "successful_attacks": 15,
  "failed_attacks": 80,
  "skipped_attacks": 5,
  "attack_success_rate": 0.158,
  "robust_accuracy": 0.842,
  "average_queries": 175.0,
  "attack_recipe": "textfooler",
  "model_name": "bert-base",
  "dataset_name": "sst2"
}
TEXTATTACK_RESULT_END
TEXTATTACK_STATUS: PARTIALLY_ROBUST
"#;

    let (status, _) = parse_textattack_output(output, "");
    match status {
        VerificationStatus::Partial {
            verified_percentage,
        } => {
            assert!(verified_percentage > 80.0);
        }
        _ => panic!("Expected Partial status"),
    }
}

#[test]
fn test_parse_output_error() {
    let output = "TEXTATTACK_ERROR: Missing dependency: transformers";
    let (status, _) = parse_textattack_output(output, "");

    match status {
        VerificationStatus::Unknown { reason } => {
            assert!(reason.contains("Missing dependency"));
        }
        _ => panic!("Expected Unknown status with error"),
    }
}

#[test]
fn test_parse_output_fallback_status() {
    let output = "Some other output\nTEXTATTACK_STATUS: ROBUST\n";
    let (status, _) = parse_textattack_output(output, "");
    assert!(matches!(status, VerificationStatus::Proven));
}

#[test]
fn test_different_recipes_in_script() {
    let spec = create_test_spec();

    // Test BERT-Attack config
    let bert_config = TextAttackConfig::bert_attack();
    let bert_script = generate_textattack_script(&spec, &bert_config).unwrap();
    assert!(bert_script.contains("bert-attack"));

    // Test DeepWordBug config
    let dwb_config = TextAttackConfig::deep_word_bug();
    let dwb_script = generate_textattack_script(&spec, &dwb_config).unwrap();
    assert!(dwb_script.contains("deepwordbug"));
}
