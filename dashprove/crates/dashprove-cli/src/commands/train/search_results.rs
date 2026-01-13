//! Search result handling and printing
//!
//! Helper functions for processing and displaying hyperparameter search results.

use dashprove::ai::strategy::{
    BayesianOptimizationResult, CVHyperparameterSearchResult, HyperparameterSearchResult,
    StrategyPredictor, TrainingExample, TrainingStats,
};
use std::path::Path;

/// Train final model with best hyperparameters and save it
pub fn train_and_save_model(
    search_result: &HyperparameterSearchResult,
    training_examples: &[TrainingExample],
    stats: &TrainingStats,
    output_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let best_hp = search_result.best_hyperparameters();
    println!("\nTraining final model with best hyperparameters...");
    println!("  Learning rate: {:.6}", best_hp.learning_rate);
    println!("  Epochs: {}", best_hp.epochs);
    println!(
        "  Validation split: {:.1}%",
        best_hp.validation_split * 100.0
    );
    if best_hp.early_stopping.is_some() {
        println!("  Early stopping: enabled");
    }

    let mut predictor = StrategyPredictor::new();
    let _final_result = predictor.train_with_best_hyperparameters(training_examples, search_result);

    save_predictor_and_print_summary(
        &predictor,
        stats,
        output_path,
        search_result.total_evaluated,
        search_result.best_val_loss(),
        search_result.best_val_accuracy(),
    )
}

/// Train and save model from CV search results
pub fn train_and_save_from_cv(
    cv_result: &CVHyperparameterSearchResult,
    training_examples: &[TrainingExample],
    stats: &TrainingStats,
    output_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let best = cv_result.best_result();
    let best_hp = &best.hyperparameters;
    println!("\nTraining final model with best hyperparameters (from CV)...");
    println!("  Learning rate: {:.6}", best_hp.learning_rate);
    println!("  Epochs: {}", best_hp.epochs);
    println!(
        "  Validation split: {:.1}%",
        best_hp.validation_split * 100.0
    );
    println!(
        "  Mean val loss: {:.4} (+/- {:.4})",
        best.mean_val_loss, best.std_val_loss
    );

    let mut predictor = StrategyPredictor::new();

    // Train with best hyperparameters
    if let Some(es_config) = &best_hp.early_stopping {
        predictor.train_with_early_stopping(
            training_examples,
            best_hp.learning_rate,
            best_hp.epochs,
            best_hp.validation_split,
            es_config.clone(),
        );
    } else {
        predictor.train(training_examples, best_hp.learning_rate, best_hp.epochs);
    }

    save_predictor_and_print_summary(
        &predictor,
        stats,
        output_path,
        cv_result.results.len(),
        best.mean_val_loss,
        best.mean_val_accuracy,
    )
}

/// Train and save model from Bayesian optimization results
pub fn train_and_save_from_bayesian(
    bayesian_result: &BayesianOptimizationResult,
    training_examples: &[TrainingExample],
    stats: &TrainingStats,
    output_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let best_hp = bayesian_result.best_hyperparameters();
    println!("\nTraining final model with best hyperparameters (from Bayesian optimization)...");
    println!("  Learning rate: {:.6}", best_hp.learning_rate);
    println!("  Epochs: {}", best_hp.epochs);
    println!(
        "  Validation split: {:.1}%",
        best_hp.validation_split * 100.0
    );
    if best_hp.early_stopping.is_some() {
        println!("  Early stopping: enabled");
    }

    let mut predictor = StrategyPredictor::new();

    // Train with best hyperparameters
    if let Some(es_config) = &best_hp.early_stopping {
        predictor.train_with_scheduler_and_early_stopping(
            training_examples,
            best_hp.learning_rate,
            best_hp.epochs,
            best_hp.validation_split,
            best_hp.lr_scheduler.clone(),
            es_config.clone(),
        );
    } else {
        predictor.train_with_scheduler(
            training_examples,
            best_hp.learning_rate,
            best_hp.epochs,
            best_hp.validation_split,
            best_hp.lr_scheduler.clone(),
        );
    }

    let best_result = bayesian_result.best_result();
    save_predictor_and_print_summary(
        &predictor,
        stats,
        output_path,
        bayesian_result.total_iterations,
        best_result.val_loss,
        best_result.val_accuracy,
    )
}

/// Save predictor and print final summary
pub fn save_predictor_and_print_summary(
    predictor: &StrategyPredictor,
    stats: &TrainingStats,
    output_path: &Path,
    configs_evaluated: usize,
    best_val_loss: f64,
    best_val_accuracy: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    // Ensure parent directory exists
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create directory {}: {}", parent.display(), e))?;
    }

    // Save the model
    predictor
        .save(output_path)
        .map_err(|e| format!("Failed to save model to {}: {}", output_path.display(), e))?;

    // Print summary
    println!("\nHyperparameter tuning complete!");
    println!("  Training examples: {}", stats.total_examples);
    println!("  Configurations evaluated: {}", configs_evaluated);
    println!("  Best validation loss: {:.4}", best_val_loss);
    println!(
        "  Best validation accuracy: {:.1}%",
        best_val_accuracy * 100.0
    );

    println!("\nModel saved to: {}", output_path.display());
    println!("\nTo use this model for verification:");
    println!(
        "  dashprove verify spec.usl --ml --ml-model {}",
        output_path.display()
    );

    Ok(())
}

/// Print hyperparameter search results
pub fn print_search_results(result: &HyperparameterSearchResult, verbose: bool) {
    println!("\nSearch Results ({} method):", result.search_method);
    println!("  Evaluated {} configurations", result.total_evaluated);

    let best = result.best_result();
    println!("\nBest configuration:");
    println!("  Learning rate: {:.6}", best.hyperparameters.learning_rate);
    println!("  Epochs: {}", best.hyperparameters.epochs);
    println!("  Validation loss: {:.4}", best.val_loss);
    println!("  Validation accuracy: {:.1}%", best.val_accuracy * 100.0);
    println!("  Train loss: {:.4}", best.train_loss);
    println!("  Train accuracy: {:.1}%", best.train_accuracy * 100.0);
    if best.stopped_early {
        println!("  Early stopped at epoch: {}", best.epochs_trained);
    }

    if verbose && result.total_evaluated > 1 {
        println!("\nAll configurations (sorted by validation loss):");
        println!(
            "{:<12} {:>8} {:>12} {:>12} {:>12}",
            "LR", "Epochs", "Val Loss", "Val Acc%", "Stopped"
        );
        println!("{}", "-".repeat(60));

        for r in result.sorted_by_loss().iter().take(10) {
            let stopped = if r.stopped_early { "yes" } else { "no" };
            println!(
                "{:<12.6} {:>8} {:>12.4} {:>11.1}% {:>12}",
                r.hyperparameters.learning_rate,
                r.hyperparameters.epochs,
                r.val_loss,
                r.val_accuracy * 100.0,
                stopped
            );
        }

        if result.total_evaluated > 10 {
            println!(
                "  ... and {} more configurations",
                result.total_evaluated - 10
            );
        }
    }
}

/// Print CV search results
pub fn print_cv_search_results(result: &CVHyperparameterSearchResult, verbose: bool) {
    println!("\nCV Search Results ({} method):", result.search_method);
    println!("  Evaluated {} configurations", result.results.len());
    println!("  Cross-validation folds: {}", result.k_folds);

    let best = result.best_result();
    println!("\nBest configuration:");
    println!("  Learning rate: {:.6}", best.hyperparameters.learning_rate);
    println!("  Epochs: {}", best.hyperparameters.epochs);
    println!(
        "  Mean val loss: {:.4} (+/- {:.4})",
        best.mean_val_loss, best.std_val_loss
    );
    println!(
        "  Mean val accuracy: {:.1}% (+/- {:.1}%)",
        best.mean_val_accuracy * 100.0,
        best.std_val_accuracy * 100.0
    );

    if verbose && result.results.len() > 1 {
        println!("\nAll configurations (sorted by mean validation loss):");
        println!(
            "{:<12} {:>8} {:>14} {:>14}",
            "LR", "Epochs", "Mean Val Loss", "Mean Val Acc%"
        );
        println!("{}", "-".repeat(60));

        for r in result.sorted_by_loss().iter().take(10) {
            println!(
                "{:<12.6} {:>8} {:>8.4}+/-{:.4} {:>7.1}%+/-{:.1}%",
                r.hyperparameters.learning_rate,
                r.hyperparameters.epochs,
                r.mean_val_loss,
                r.std_val_loss,
                r.mean_val_accuracy * 100.0,
                r.std_val_accuracy * 100.0
            );
        }

        if result.results.len() > 10 {
            println!(
                "  ... and {} more configurations",
                result.results.len() - 10
            );
        }
    }
}

/// Print Bayesian optimization results
pub fn print_bayesian_results(result: &BayesianOptimizationResult, verbose: bool) {
    println!("\nBayesian Optimization Results:");
    println!("  Total iterations: {}", result.total_iterations);
    println!("  Initial random samples: {}", result.n_initial_samples);

    let best = result.best_result();
    println!("\nBest configuration:");
    println!("  Learning rate: {:.6}", best.hyperparameters.learning_rate);
    println!("  Epochs: {}", best.hyperparameters.epochs);
    println!("  Validation loss: {:.4}", best.val_loss);
    println!("  Validation accuracy: {:.1}%", best.val_accuracy * 100.0);
    println!("  Train loss: {:.4}", best.train_loss);
    println!("  Train accuracy: {:.1}%", best.train_accuracy * 100.0);
    if best.stopped_early {
        println!("  Early stopped at epoch: {}", best.epochs_trained);
    }

    if verbose && result.evaluations.len() > 1 {
        println!("\nAll configurations (sorted by validation loss):");
        println!(
            "{:<12} {:>8} {:>12} {:>12} {:>12}",
            "LR", "Epochs", "Val Loss", "Val Acc%", "Stopped"
        );
        println!("{}", "-".repeat(60));

        for r in result.sorted_by_loss().iter().take(10) {
            let stopped = if r.stopped_early { "yes" } else { "no" };
            println!(
                "{:<12.6} {:>8} {:>12.4} {:>11.1}% {:>12}",
                r.hyperparameters.learning_rate,
                r.hyperparameters.epochs,
                r.val_loss,
                r.val_accuracy * 100.0,
                stopped
            );
        }

        if result.evaluations.len() > 10 {
            println!(
                "  ... and {} more configurations",
                result.evaluations.len() - 10
            );
        }

        // Show acquisition function progress
        if !result.acquisition_history.is_empty() {
            println!("\nAcquisition function history (after initial samples):");
            let gp_evals = result
                .acquisition_history
                .iter()
                .skip(result.n_initial_samples)
                .collect::<Vec<_>>();
            for (i, acq) in gp_evals.iter().enumerate().take(5) {
                println!(
                    "  Iteration {}: {:.4}",
                    i + result.n_initial_samples + 1,
                    acq
                );
            }
            if gp_evals.len() > 5 {
                println!("  ...");
            }
        }
    }
}
