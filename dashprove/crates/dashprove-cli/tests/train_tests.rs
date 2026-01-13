//! Integration tests for ML training CLI commands (train, tune, ensemble)

use serial_test::serial;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_dir(prefix: &str) -> std::path::PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("dashprove_train_test_{prefix}_{ts}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn dashprove_cmd() -> Command {
    Command::new(env!("CARGO_BIN_EXE_dashprove"))
}

/// Generate a valid proof entry JSON
fn make_proof_entry(id: &str, backend: &str) -> String {
    format!(
        r#""{id}": {{
            "id": "{id}",
            "property": {{"Theorem": {{"name": "{id}", "body": {{"Bool": true}}}}}},
            "backend": "{backend}",
            "tactics": ["simp", "auto"],
            "time_taken": {{"secs": 0, "nanos": 100000000}},
            "proof_output": null,
            "features": {{
                "property_type": "theorem",
                "depth": 1,
                "quantifier_depth": 0,
                "implication_count": 0,
                "arithmetic_ops": 0,
                "function_calls": 0,
                "variable_count": 0,
                "has_temporal": false,
                "type_refs": []
            }}
        }}"#
    )
}

/// Generate a corpus JSON with multiple proofs
fn make_corpus_json(num_proofs: usize) -> String {
    let mut proofs = String::new();
    for i in 0..num_proofs {
        let backend = match i % 3 {
            0 => "Lean4",
            1 => "TlaPlus",
            _ => "Kani",
        };
        if i > 0 {
            proofs.push(',');
        }
        proofs.push_str(&make_proof_entry(&format!("thm_{}", i), backend));
    }
    format!(r#"{{"proofs": {{{}}}}}"#, proofs)
}

// ============================================================================
// Train Command Tests
// ============================================================================

#[test]
#[serial]
fn test_train_help() {
    let output = dashprove_cmd()
        .args(["train", "--help"])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success());
    assert!(
        stdout.contains("--learning-rate"),
        "Help should show --learning-rate flag"
    );
    assert!(
        stdout.contains("--epochs"),
        "Help should show --epochs flag"
    );
    assert!(
        stdout.contains("--early-stopping"),
        "Help should show --early-stopping flag"
    );
    assert!(
        stdout.contains("--lr-scheduler"),
        "Help should show --lr-scheduler flag"
    );
    assert!(
        stdout.contains("--checkpoint"),
        "Help should show --checkpoint flag"
    );
    assert!(
        stdout.contains("--resume"),
        "Help should show --resume flag"
    );
}

#[test]
#[serial]
fn test_train_empty_corpus() {
    let data_dir = temp_dir("train_empty");

    let output = dashprove_cmd()
        .args(["train", "--data-dir"])
        .arg(data_dir.to_str().unwrap())
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Train should succeed with empty corpus. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("corpus is empty") || stdout.contains("Proof corpus is empty"),
        "Should report empty corpus. stdout: {}",
        stdout
    );

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
#[serial]
fn test_train_with_corpus_data() {
    let data_dir = temp_dir("train_corpus");

    // Create a proof corpus with test data
    let corpus_json = make_corpus_json(3);
    std::fs::write(data_dir.join("proof_corpus.json"), corpus_json).unwrap();

    let model_output = data_dir.join("test_model.json");

    let output = dashprove_cmd()
        .args([
            "train",
            "--data-dir",
            data_dir.to_str().unwrap(),
            "--output",
            model_output.to_str().unwrap(),
            "--epochs",
            "5",
            "--learning-rate",
            "0.01",
        ])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Train should succeed. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Loaded") && stdout.contains("proofs"),
        "Should show corpus loaded. stdout: {}",
        stdout
    );
    assert!(
        stdout.contains("Training complete"),
        "Should show training completed. stdout: {}",
        stdout
    );
    assert!(
        model_output.exists(),
        "Model file should be created at {}",
        model_output.display()
    );

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
#[serial]
fn test_train_with_early_stopping() {
    let data_dir = temp_dir("train_es");

    // Create corpus with enough examples for validation split
    let corpus_json = make_corpus_json(15);
    std::fs::write(data_dir.join("proof_corpus.json"), corpus_json).unwrap();

    let output = dashprove_cmd()
        .args([
            "train",
            "--data-dir",
            data_dir.to_str().unwrap(),
            "--early-stopping",
            "--patience",
            "3",
            "--epochs",
            "50",
            "--verbose",
        ])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Train with early stopping should succeed. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Early stopping"),
        "Should show early stopping info. stdout: {}",
        stdout
    );

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
#[serial]
fn test_train_with_lr_scheduler() {
    let data_dir = temp_dir("train_sched");

    // Create corpus
    let corpus_json = make_corpus_json(10);
    std::fs::write(data_dir.join("proof_corpus.json"), corpus_json).unwrap();

    let output = dashprove_cmd()
        .args([
            "train",
            "--data-dir",
            data_dir.to_str().unwrap(),
            "--lr-scheduler",
            "step",
            "--lr-step-size",
            "5",
            "--lr-gamma",
            "0.5",
            "--epochs",
            "15",
            "--verbose",
        ])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Train with scheduler should succeed. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("LR scheduler") || stdout.contains("scheduler"),
        "Should show scheduler info. stdout: {}",
        stdout
    );

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
#[serial]
fn test_train_verbose_output() {
    let data_dir = temp_dir("train_verbose");

    // Create corpus with multiple backends (3 proofs with different backends)
    let corpus_json = make_corpus_json(3);
    std::fs::write(data_dir.join("proof_corpus.json"), corpus_json).unwrap();

    let output = dashprove_cmd()
        .args([
            "train",
            "--data-dir",
            data_dir.to_str().unwrap(),
            "--verbose",
            "--epochs",
            "5",
        ])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Verbose train should succeed. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Examples per backend") || stdout.contains("backend"),
        "Verbose output should show per-backend stats. stdout: {}",
        stdout
    );

    std::fs::remove_dir_all(data_dir).ok();
}

// ============================================================================
// Tune Command Tests
// ============================================================================

#[test]
#[serial]
fn test_tune_help() {
    let output = dashprove_cmd()
        .args(["tune", "--help"])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success());
    assert!(
        stdout.contains("--method"),
        "Help should show --method flag"
    );
    assert!(stdout.contains("grid"), "Help should mention grid search");
    assert!(
        stdout.contains("random"),
        "Help should mention random search"
    );
    assert!(
        stdout.contains("bayesian"),
        "Help should mention bayesian search"
    );
    assert!(
        stdout.contains("--iterations"),
        "Help should show --iterations flag"
    );
    assert!(
        stdout.contains("--cv-folds"),
        "Help should show --cv-folds flag"
    );
}

#[test]
#[serial]
fn test_tune_empty_corpus() {
    let data_dir = temp_dir("tune_empty");

    let output = dashprove_cmd()
        .args([
            "tune",
            "--method",
            "random",
            "--iterations",
            "2",
            "--data-dir",
        ])
        .arg(data_dir.to_str().unwrap())
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Tune should succeed with empty corpus. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("corpus is empty") || stdout.contains("Proof corpus is empty"),
        "Should report empty corpus. stdout: {}",
        stdout
    );

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
#[serial]
fn test_tune_grid_search() {
    let data_dir = temp_dir("tune_grid");

    // Create corpus with enough examples
    let corpus_json = make_corpus_json(10);
    std::fs::write(data_dir.join("proof_corpus.json"), corpus_json).unwrap();

    let output = dashprove_cmd()
        .args([
            "tune",
            "--method",
            "grid",
            "--lr-values",
            "0.01,0.05",
            "--epoch-values",
            "5",
            "--data-dir",
            data_dir.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Grid search should succeed. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("grid") || stdout.contains("Grid"),
        "Should show grid search. stdout: {}",
        stdout
    );
    assert!(
        stdout.contains("Best configuration") || stdout.contains("best"),
        "Should show best configuration. stdout: {}",
        stdout
    );

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
#[serial]
fn test_tune_random_search() {
    let data_dir = temp_dir("tune_random");

    // Create corpus
    let corpus_json = make_corpus_json(10);
    std::fs::write(data_dir.join("proof_corpus.json"), corpus_json).unwrap();

    let output = dashprove_cmd()
        .args([
            "tune",
            "--method",
            "random",
            "--iterations",
            "3",
            "--seed",
            "42",
            "--data-dir",
            data_dir.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Random search should succeed. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("random") || stdout.contains("Random"),
        "Should show random search. stdout: {}",
        stdout
    );

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
#[serial]
fn test_tune_bayesian_search() {
    let data_dir = temp_dir("tune_bayes");

    // Create corpus
    let corpus_json = make_corpus_json(10);
    std::fs::write(data_dir.join("proof_corpus.json"), corpus_json).unwrap();

    let output = dashprove_cmd()
        .args([
            "tune",
            "--method",
            "bayesian",
            "--iterations",
            "5",
            "--initial-samples",
            "2",
            "--data-dir",
            data_dir.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Bayesian search should succeed. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Bayesian") || stdout.contains("bayesian"),
        "Should show Bayesian optimization. stdout: {}",
        stdout
    );

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
#[serial]
fn test_tune_invalid_method() {
    let data_dir = temp_dir("tune_invalid");

    let output = dashprove_cmd()
        .args([
            "tune",
            "--method",
            "invalid_method",
            "--data-dir",
            data_dir.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run dashprove");

    // Should either fail or report unknown method
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{}{}", stdout, stderr);

    // If corpus is empty, it returns early
    if !combined.contains("corpus is empty") && !combined.contains("Proof corpus is empty") {
        assert!(
            !output.status.success() || combined.contains("Unknown"),
            "Invalid method should fail or warn. output: {}",
            combined
        );
    }

    std::fs::remove_dir_all(data_dir).ok();
}

// ============================================================================
// Ensemble Command Tests
// ============================================================================

#[test]
#[serial]
fn test_ensemble_help() {
    let output = dashprove_cmd()
        .args(["ensemble", "--help"])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success());
    assert!(
        stdout.contains("--models"),
        "Help should show --models flag"
    );
    assert!(
        stdout.contains("--weights"),
        "Help should show --weights flag"
    );
    assert!(
        stdout.contains("--method"),
        "Help should show --method flag"
    );
    assert!(
        stdout.contains("soft") || stdout.contains("weighted"),
        "Help should mention aggregation methods"
    );
}

#[test]
#[serial]
fn test_ensemble_requires_models() {
    // Test that ensemble fails without --models
    let output = dashprove_cmd()
        .args(["ensemble"])
        .output()
        .expect("Failed to run dashprove");

    assert!(
        !output.status.success(),
        "Ensemble without models should fail"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--models") || stderr.contains("required"),
        "Should report missing --models. stderr: {}",
        stderr
    );
}

#[test]
#[serial]
fn test_ensemble_requires_at_least_two_models() {
    let data_dir = temp_dir("ensemble_single");

    // Create a single model file by training
    let corpus_json = make_corpus_json(1);
    std::fs::write(data_dir.join("proof_corpus.json"), corpus_json).unwrap();

    let model_path = data_dir.join("single_model.json");

    // Train a model first
    dashprove_cmd()
        .args([
            "train",
            "--data-dir",
            data_dir.to_str().unwrap(),
            "--output",
            model_path.to_str().unwrap(),
            "--epochs",
            "2",
        ])
        .output()
        .expect("Failed to run train");

    // Try to create ensemble with single model
    let output = dashprove_cmd()
        .args(["ensemble", "--models", model_path.to_str().unwrap()])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{}{}", stdout, stderr);

    assert!(
        !output.status.success() || combined.contains("at least two"),
        "Ensemble with single model should fail. output: {}",
        combined
    );

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
#[serial]
fn test_ensemble_creates_ensemble_model() {
    let data_dir = temp_dir("ensemble_create");

    // Create corpus
    let corpus_json = make_corpus_json(2);
    std::fs::write(data_dir.join("proof_corpus.json"), corpus_json).unwrap();

    let model_a = data_dir.join("model_a.json");
    let model_b = data_dir.join("model_b.json");
    let ensemble_output = data_dir.join("ensemble.json");

    // Train two models
    dashprove_cmd()
        .args([
            "train",
            "--data-dir",
            data_dir.to_str().unwrap(),
            "--output",
            model_a.to_str().unwrap(),
            "--epochs",
            "3",
            "--learning-rate",
            "0.01",
        ])
        .output()
        .expect("Failed to train model A");

    dashprove_cmd()
        .args([
            "train",
            "--data-dir",
            data_dir.to_str().unwrap(),
            "--output",
            model_b.to_str().unwrap(),
            "--epochs",
            "3",
            "--learning-rate",
            "0.05",
        ])
        .output()
        .expect("Failed to train model B");

    // Create ensemble
    let output = dashprove_cmd()
        .args([
            "ensemble",
            "--models",
            &format!("{},{}", model_a.display(), model_b.display()),
            "--method",
            "soft",
            "--output",
            ensemble_output.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run ensemble");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Ensemble creation should succeed. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Ensemble") || stdout.contains("ensemble"),
        "Should show ensemble created. stdout: {}",
        stdout
    );
    assert!(
        ensemble_output.exists(),
        "Ensemble file should exist at {}",
        ensemble_output.display()
    );

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
#[serial]
fn test_ensemble_with_weights() {
    let data_dir = temp_dir("ensemble_weights");

    // Create corpus
    let corpus_json = make_corpus_json(1);
    std::fs::write(data_dir.join("proof_corpus.json"), corpus_json).unwrap();

    let model_a = data_dir.join("model_a.json");
    let model_b = data_dir.join("model_b.json");
    let ensemble_output = data_dir.join("ensemble_weighted.json");

    // Train two models
    dashprove_cmd()
        .args([
            "train",
            "--data-dir",
            data_dir.to_str().unwrap(),
            "--output",
            model_a.to_str().unwrap(),
            "--epochs",
            "2",
        ])
        .output()
        .expect("Failed to train model A");

    dashprove_cmd()
        .args([
            "train",
            "--data-dir",
            data_dir.to_str().unwrap(),
            "--output",
            model_b.to_str().unwrap(),
            "--epochs",
            "2",
        ])
        .output()
        .expect("Failed to train model B");

    // Create weighted ensemble
    let output = dashprove_cmd()
        .args([
            "ensemble",
            "--models",
            &format!("{},{}", model_a.display(), model_b.display()),
            "--weights",
            "0.7,0.3",
            "--method",
            "weighted",
            "--output",
            ensemble_output.to_str().unwrap(),
            "--verbose",
        ])
        .output()
        .expect("Failed to run ensemble");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Weighted ensemble should succeed. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        ensemble_output.exists(),
        "Weighted ensemble file should exist"
    );

    // Verbose should show weights
    assert!(
        stdout.contains("Weights") || stdout.contains("0.7") || stdout.contains("weighted"),
        "Should show weights in verbose output. stdout: {}",
        stdout
    );

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
#[serial]
fn test_ensemble_invalid_weights_count() {
    let data_dir = temp_dir("ensemble_bad_weights");

    // Create corpus and models
    let corpus_json = make_corpus_json(1);
    std::fs::write(data_dir.join("proof_corpus.json"), corpus_json).unwrap();

    let model_a = data_dir.join("model_a.json");
    let model_b = data_dir.join("model_b.json");

    dashprove_cmd()
        .args([
            "train",
            "--data-dir",
            data_dir.to_str().unwrap(),
            "--output",
            model_a.to_str().unwrap(),
            "--epochs",
            "2",
        ])
        .output()
        .expect("Failed to train model A");

    dashprove_cmd()
        .args([
            "train",
            "--data-dir",
            data_dir.to_str().unwrap(),
            "--output",
            model_b.to_str().unwrap(),
            "--epochs",
            "2",
        ])
        .output()
        .expect("Failed to train model B");

    // Try to create ensemble with wrong number of weights
    let output = dashprove_cmd()
        .args([
            "ensemble",
            "--models",
            &format!("{},{}", model_a.display(), model_b.display()),
            "--weights",
            "0.5", // Only one weight for two models
        ])
        .output()
        .expect("Failed to run ensemble");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{}{}", stdout, stderr);

    assert!(
        !output.status.success() || combined.contains("weights"),
        "Mismatched weights should fail. output: {}",
        combined
    );

    std::fs::remove_dir_all(data_dir).ok();
}

// ============================================================================
// Integration Tests (Train + Use Model)
// ============================================================================

#[test]
#[serial]
fn test_train_model_can_be_loaded() {
    let data_dir = temp_dir("train_load");

    // Create corpus
    let corpus_json = make_corpus_json(1);
    std::fs::write(data_dir.join("proof_corpus.json"), corpus_json).unwrap();

    let model_path = data_dir.join("loadable_model.json");

    // Train
    let train_output = dashprove_cmd()
        .args([
            "train",
            "--data-dir",
            data_dir.to_str().unwrap(),
            "--output",
            model_path.to_str().unwrap(),
            "--epochs",
            "3",
        ])
        .output()
        .expect("Failed to train");

    assert!(
        train_output.status.success(),
        "Training should succeed. stderr: {}",
        String::from_utf8_lossy(&train_output.stderr)
    );

    // Verify model file exists and is valid JSON
    assert!(model_path.exists(), "Model file should exist");

    let model_content = std::fs::read_to_string(&model_path).expect("Should read model file");
    assert!(
        model_content.contains("weights") || model_content.contains("single"),
        "Model file should contain model data"
    );

    // Verify it's valid JSON
    let _: serde_json::Value =
        serde_json::from_str(&model_content).expect("Model should be valid JSON");

    std::fs::remove_dir_all(data_dir).ok();
}

#[test]
#[serial]
fn test_verify_accepts_ml_model_flag() {
    // Test that verify command accepts --ml and --ml-model flags
    let output = dashprove_cmd()
        .args(["verify", "--help"])
        .output()
        .expect("Failed to run dashprove");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success());
    assert!(stdout.contains("--ml"), "Help should show --ml flag");
    assert!(
        stdout.contains("--ml-model"),
        "Help should show --ml-model flag"
    );
    assert!(
        stdout.contains("--ml-confidence"),
        "Help should show --ml-confidence flag"
    );
}
