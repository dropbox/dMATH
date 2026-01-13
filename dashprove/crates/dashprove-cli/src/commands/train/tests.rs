//! Tests for train command
//!
//! Unit tests for training functionality.

#![cfg(test)]

use dashprove::backends::traits::{BackendId, VerificationStatus};
use dashprove::learning::{LearnableResult, ProofCorpus};
use dashprove::usl::ast::{Expr, Property, Theorem};
use std::time::Duration;
use tempfile::TempDir;

use crate::commands::train::{
    config::{SchedulerType, TrainConfig},
    core::run_train,
    ensemble::{run_ensemble, EnsembleConfig},
};
use dashprove::ai::strategy::{StrategyModel, StrategyPredictor};

fn make_proof(name: &str, backend: BackendId) -> LearnableResult {
    LearnableResult {
        property: Property::Theorem(Theorem {
            name: name.to_string(),
            body: Expr::Bool(true),
        }),
        backend,
        status: VerificationStatus::Proven,
        tactics: vec!["simp".to_string()],
        time_taken: Duration::from_millis(100),
        proof_output: None,
    }
}

fn default_config(data_path: &str) -> TrainConfig<'_> {
    TrainConfig {
        data_dir: Some(data_path),
        output: None,
        learning_rate: 0.01,
        epochs: 5,
        verbose: false,
        early_stopping: false,
        patience: 5,
        min_delta: 0.001,
        validation_split: 0.2,
        lr_scheduler: SchedulerType::Constant,
        lr_step_size: 10,
        lr_gamma: 0.5,
        lr_min: 0.0001,
        lr_warmup_epochs: 5,
        checkpoint: false,
        checkpoint_dir: None,
        checkpoint_interval: 0,
        keep_best: 3,
        resume: None,
    }
}

#[test]
fn test_train_empty_corpus() {
    let temp_dir = TempDir::new().unwrap();
    let data_path = temp_dir.path().to_str().unwrap();

    let config = default_config(data_path);

    // Should succeed without error even with empty corpus
    let result = run_train(config);
    assert!(result.is_ok());
}

#[test]
fn test_train_with_corpus() {
    let temp_dir = TempDir::new().unwrap();
    let data_path = temp_dir.path().to_str().unwrap();
    let corpus_path = temp_dir.path().join("proof_corpus.json");

    // Create a corpus with some proofs
    let mut corpus = ProofCorpus::new();
    corpus.insert(&make_proof("thm1", BackendId::Lean4));
    corpus.insert(&make_proof("thm2", BackendId::Lean4));
    corpus.insert(&make_proof("thm3", BackendId::TlaPlus));
    corpus.save_to_file(&corpus_path).unwrap();

    let mut config = default_config(data_path);
    config.verbose = true;

    let result = run_train(config);
    assert!(result.is_ok());

    // Check model was saved
    let model_path = temp_dir.path().join("strategy_model.json");
    assert!(model_path.exists());

    // Verify model can be loaded
    let loaded = StrategyPredictor::load(&model_path);
    assert!(loaded.is_ok());
}

#[test]
fn test_train_custom_output_path() {
    let temp_dir = TempDir::new().unwrap();
    let data_path = temp_dir.path().to_str().unwrap();
    let corpus_path = temp_dir.path().join("proof_corpus.json");
    let custom_output = temp_dir.path().join("custom_dir").join("my_model.json");

    // Create corpus
    let mut corpus = ProofCorpus::new();
    corpus.insert(&make_proof("thm1", BackendId::Lean4));
    corpus.save_to_file(&corpus_path).unwrap();

    let mut config = default_config(data_path);
    config.output = Some(custom_output.to_str().unwrap());

    let result = run_train(config);
    assert!(result.is_ok());

    // Check model was saved to custom path
    assert!(custom_output.exists());
}

#[test]
fn test_train_different_hyperparameters() {
    let temp_dir = TempDir::new().unwrap();
    let data_path = temp_dir.path().to_str().unwrap();
    let corpus_path = temp_dir.path().join("proof_corpus.json");

    // Create corpus
    let mut corpus = ProofCorpus::new();
    for i in 0..10 {
        corpus.insert(&make_proof(&format!("thm{}", i), BackendId::Lean4));
    }
    corpus.save_to_file(&corpus_path).unwrap();

    // Test with higher learning rate and more epochs
    let mut config = default_config(data_path);
    config.learning_rate = 0.1;
    config.epochs = 20;

    let result = run_train(config);
    assert!(result.is_ok());
}

#[test]
fn test_train_corpus_with_multiple_backends() {
    let temp_dir = TempDir::new().unwrap();
    let data_path = temp_dir.path().to_str().unwrap();
    let corpus_path = temp_dir.path().join("proof_corpus.json");

    // Create corpus with diverse backends
    let mut corpus = ProofCorpus::new();
    corpus.insert(&make_proof("lean_thm", BackendId::Lean4));
    corpus.insert(&make_proof("tla_thm", BackendId::TlaPlus));
    corpus.insert(&make_proof("kani_thm", BackendId::Kani));
    corpus.insert(&make_proof("alloy_thm", BackendId::Alloy));
    corpus.insert(&make_proof("coq_thm", BackendId::Coq));
    corpus.save_to_file(&corpus_path).unwrap();

    let mut config = default_config(data_path);
    config.epochs = 10;
    config.verbose = true;

    let result = run_train(config);
    assert!(result.is_ok());
}

#[test]
fn test_train_with_early_stopping() {
    let temp_dir = TempDir::new().unwrap();
    let data_path = temp_dir.path().to_str().unwrap();
    let corpus_path = temp_dir.path().join("proof_corpus.json");

    // Create corpus with enough examples for validation split
    let mut corpus = ProofCorpus::new();
    for i in 0..20 {
        let backend = if i % 2 == 0 {
            BackendId::Lean4
        } else {
            BackendId::TlaPlus
        };
        corpus.insert(&make_proof(&format!("es_thm_{}", i), backend));
    }
    corpus.save_to_file(&corpus_path).unwrap();

    let mut config = default_config(data_path);
    config.early_stopping = true;
    config.patience = 3;
    config.min_delta = 0.001;
    config.validation_split = 0.2;
    config.epochs = 100; // High epochs to test early stopping kicks in
    config.verbose = true;

    let result = run_train(config);
    assert!(result.is_ok());

    // Check model was saved
    let model_path = temp_dir.path().join("strategy_model.json");
    assert!(model_path.exists());
}

#[test]
fn test_train_early_stopping_small_corpus() {
    let temp_dir = TempDir::new().unwrap();
    let data_path = temp_dir.path().to_str().unwrap();
    let corpus_path = temp_dir.path().join("proof_corpus.json");

    // Small corpus - should handle edge case gracefully
    let mut corpus = ProofCorpus::new();
    for i in 0..5 {
        corpus.insert(&make_proof(&format!("small_thm_{}", i), BackendId::Lean4));
    }
    corpus.save_to_file(&corpus_path).unwrap();

    let mut config = default_config(data_path);
    config.early_stopping = true;
    config.patience = 2;
    config.epochs = 50;

    let result = run_train(config);
    assert!(result.is_ok());
}

#[test]
fn test_train_with_step_scheduler() {
    let temp_dir = TempDir::new().unwrap();
    let data_path = temp_dir.path().to_str().unwrap();
    let corpus_path = temp_dir.path().join("proof_corpus.json");

    // Create corpus
    let mut corpus = ProofCorpus::new();
    for i in 0..15 {
        corpus.insert(&make_proof(&format!("step_thm_{}", i), BackendId::Lean4));
    }
    corpus.save_to_file(&corpus_path).unwrap();

    let mut config = default_config(data_path);
    config.lr_scheduler = SchedulerType::Step;
    config.lr_step_size = 5;
    config.lr_gamma = 0.5;
    config.epochs = 15;
    config.verbose = true;

    let result = run_train(config);
    assert!(result.is_ok());

    let model_path = temp_dir.path().join("strategy_model.json");
    assert!(model_path.exists());
}

#[test]
fn test_train_with_exponential_scheduler() {
    let temp_dir = TempDir::new().unwrap();
    let data_path = temp_dir.path().to_str().unwrap();
    let corpus_path = temp_dir.path().join("proof_corpus.json");

    // Create corpus
    let mut corpus = ProofCorpus::new();
    for i in 0..10 {
        corpus.insert(&make_proof(&format!("exp_thm_{}", i), BackendId::TlaPlus));
    }
    corpus.save_to_file(&corpus_path).unwrap();

    let mut config = default_config(data_path);
    config.lr_scheduler = SchedulerType::Exponential;
    config.lr_gamma = 0.95;
    config.epochs = 10;

    let result = run_train(config);
    assert!(result.is_ok());
}

#[test]
fn test_train_with_cosine_scheduler() {
    let temp_dir = TempDir::new().unwrap();
    let data_path = temp_dir.path().to_str().unwrap();
    let corpus_path = temp_dir.path().join("proof_corpus.json");

    // Create corpus
    let mut corpus = ProofCorpus::new();
    for i in 0..12 {
        corpus.insert(&make_proof(&format!("cos_thm_{}", i), BackendId::Kani));
    }
    corpus.save_to_file(&corpus_path).unwrap();

    let mut config = default_config(data_path);
    config.lr_scheduler = SchedulerType::Cosine;
    config.lr_min = 0.001;
    config.epochs = 10;

    let result = run_train(config);
    assert!(result.is_ok());
}

#[test]
fn test_train_with_scheduler_and_early_stopping() {
    let temp_dir = TempDir::new().unwrap();
    let data_path = temp_dir.path().to_str().unwrap();
    let corpus_path = temp_dir.path().join("proof_corpus.json");

    // Create corpus
    let mut corpus = ProofCorpus::new();
    for i in 0..25 {
        let backend = if i % 3 == 0 {
            BackendId::Lean4
        } else if i % 3 == 1 {
            BackendId::TlaPlus
        } else {
            BackendId::Coq
        };
        corpus.insert(&make_proof(&format!("combo_thm_{}", i), backend));
    }
    corpus.save_to_file(&corpus_path).unwrap();

    let mut config = default_config(data_path);
    config.lr_scheduler = SchedulerType::Step;
    config.lr_step_size = 10;
    config.lr_gamma = 0.5;
    config.early_stopping = true;
    config.patience = 3;
    config.epochs = 100;
    config.verbose = true;

    let result = run_train(config);
    assert!(result.is_ok());

    let model_path = temp_dir.path().join("strategy_model.json");
    assert!(model_path.exists());
}

#[test]
fn test_run_ensemble_writes_file() {
    let dir = TempDir::new().unwrap();
    let model_a = dir.path().join("model_a.json");
    let model_b = dir.path().join("model_b.json");

    StrategyPredictor::new().save(&model_a).unwrap();
    StrategyPredictor::new().save(&model_b).unwrap();

    let output = dir.path().join("ensemble.json");

    run_ensemble(EnsembleConfig {
        models: vec![model_a.to_str().unwrap(), model_b.to_str().unwrap()],
        weights: Some("0.6,0.4"),
        method: "soft",
        output: Some(output.to_str().unwrap()),
        data_dir: None,
        verbose: false,
    })
    .unwrap();

    let model = StrategyModel::load(&output).unwrap();
    match model {
        StrategyModel::Ensemble { ensemble } => {
            assert_eq!(ensemble.members.len(), 2);
        }
        _ => panic!("expected ensemble model"),
    }
}

#[test]
fn test_parse_weights_validates_length() {
    use crate::commands::train::ensemble::parse_weights;
    let err = parse_weights(Some("0.5"), 2).unwrap_err();
    let msg = format!("{}", err);
    assert!(msg.contains("weights for 2 models"));
}

#[test]
fn test_scheduler_type_parsing() {
    assert_eq!(
        "constant".parse::<SchedulerType>().unwrap(),
        SchedulerType::Constant
    );
    assert_eq!(
        "step".parse::<SchedulerType>().unwrap(),
        SchedulerType::Step
    );
    assert_eq!(
        "exponential".parse::<SchedulerType>().unwrap(),
        SchedulerType::Exponential
    );
    assert_eq!(
        "exp".parse::<SchedulerType>().unwrap(),
        SchedulerType::Exponential
    );
    assert_eq!(
        "cosine".parse::<SchedulerType>().unwrap(),
        SchedulerType::Cosine
    );
    assert_eq!(
        "cos".parse::<SchedulerType>().unwrap(),
        SchedulerType::Cosine
    );
    assert_eq!(
        "plateau".parse::<SchedulerType>().unwrap(),
        SchedulerType::ReduceOnPlateau
    );
    assert_eq!(
        "warmup".parse::<SchedulerType>().unwrap(),
        SchedulerType::WarmupDecay
    );
    assert!("invalid".parse::<SchedulerType>().is_err());
}
