//! Checkpoint-based training
//!
//! Training with model checkpointing and resume from checkpoint support.

use dashprove::ai::strategy::{
    CheckpointConfig, CheckpointedTrainingResult, StrategyPredictor, TrainingExample, TrainingStats,
};
use std::path::{Path, PathBuf};

use super::config::TrainConfig;

/// Run training with checkpointing enabled
pub fn run_train_with_checkpointing(
    config: &TrainConfig<'_>,
    training_examples: &[TrainingExample],
    stats: &TrainingStats,
    output_path: &Path,
    data_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint_dir = config
        .checkpoint_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| data_dir.join("checkpoints"));

    println!("Training with checkpointing enabled");
    println!("  Checkpoint directory: {}", checkpoint_dir.display());

    let checkpoint_config = CheckpointConfig::new(&checkpoint_dir)
        .with_save_interval(config.checkpoint_interval)
        .with_keep_best(config.keep_best)
        .with_save_on_improvement(true)
        .with_history(true);

    let mut predictor = StrategyPredictor::new();
    let result = predictor
        .train_with_checkpointing(
            training_examples,
            config.learning_rate,
            config.epochs,
            config.validation_split,
            &checkpoint_config,
        )
        .map_err(|e| format!("Checkpointed training failed: {}", e))?;

    // Save the final model
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create directory {}: {}", parent.display(), e))?;
    }

    predictor
        .save(output_path)
        .map_err(|e| format!("Failed to save model: {}", e))?;

    // Print checkpointing summary
    print_checkpointing_summary(stats, output_path, &result);

    Ok(())
}

/// Resume training from a checkpoint
pub fn run_train_resume(
    config: &TrainConfig<'_>,
    training_examples: &[TrainingExample],
    stats: &TrainingStats,
    output_path: &Path,
    resume_path: &str,
    data_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Resuming training from checkpoint: {}", resume_path);

    let checkpoint_config = if config.checkpoint {
        let checkpoint_dir = config
            .checkpoint_dir
            .map(PathBuf::from)
            .unwrap_or_else(|| data_dir.join("checkpoints"));

        Some(
            CheckpointConfig::new(&checkpoint_dir)
                .with_save_interval(config.checkpoint_interval)
                .with_keep_best(config.keep_best)
                .with_save_on_improvement(true)
                .with_history(true),
        )
    } else {
        None
    };

    let (predictor, result) = StrategyPredictor::resume_from_checkpoint(
        resume_path,
        config.epochs,
        config.learning_rate,
        training_examples,
        config.validation_split,
        checkpoint_config.as_ref(),
    )
    .map_err(|e| format!("Failed to resume from checkpoint: {}", e))?;

    // Save the final model
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create directory {}: {}", parent.display(), e))?;
    }

    predictor
        .save(output_path)
        .map_err(|e| format!("Failed to save model: {}", e))?;

    // Print checkpointing summary
    print_checkpointing_summary(stats, output_path, &result);

    Ok(())
}

/// Print checkpointing training summary
fn print_checkpointing_summary(
    stats: &TrainingStats,
    output_path: &Path,
    result: &CheckpointedTrainingResult,
) {
    println!("\nCheckpointed training complete!");
    println!("  Total examples: {}", stats.total_examples);
    println!("  Total epochs: {}", result.total_epochs);
    println!("  Best epoch: {}", result.best_epoch);
    println!("  Best validation loss: {:.4}", result.best_val_loss);
    println!("  Checkpoints saved: {}", result.checkpoints_saved.len());

    if !result.checkpoints_saved.is_empty() {
        println!("\n  Saved checkpoints:");
        for path in &result.checkpoints_saved {
            println!("    - {}", path);
        }
    }

    if let Some(best_path) = &result.best_checkpoint_path {
        println!("\n  Best checkpoint: {}", best_path);
    }

    println!("\nModel saved to: {}", output_path.display());
    println!("\nTo use this model for verification:");
    println!(
        "  dashprove verify spec.usl --ml --ml-model {}",
        output_path.display()
    );
}
