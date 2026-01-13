//! Training summary output
//!
//! Functions for printing training results and statistics.

use dashprove::ai::strategy::{EarlyStoppingResult, ScheduledTrainingResult, TrainingStats};
use std::path::Path;

/// Print training statistics summary
pub fn print_training_summary(
    stats: &TrainingStats,
    output_path: &Path,
    verbose: bool,
    early_stopping: Option<&EarlyStoppingResult>,
    scheduled: Option<&ScheduledTrainingResult>,
) {
    println!("\nTraining complete!");
    println!("  Total examples: {}", stats.total_examples);
    println!("  Successful: {}", stats.successful_examples);
    println!("  Failed: {}", stats.failed_examples);

    if verbose && !stats.examples_per_backend.is_empty() {
        println!("\n  Examples per backend:");
        let mut backend_list: Vec<_> = stats.examples_per_backend.iter().collect();
        backend_list.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        for (backend, count) in backend_list {
            println!("    {:?}: {}", backend, count);
        }
    }

    // Print early stopping information if available
    if let Some(es_result) = early_stopping {
        println!("\n  Early stopping results:");
        if es_result.stopped_early {
            println!(
                "    Stopped early at epoch {} (best: epoch {})",
                es_result.final_epoch, es_result.best_epoch
            );
        } else {
            println!(
                "    Completed all epochs (best: epoch {})",
                es_result.best_epoch
            );
        }
        println!("    Best validation loss: {:.4}", es_result.best_val_loss);

        if verbose {
            println!(
                "    Total epochs trained: {}",
                es_result.history.epochs.len()
            );
            if let Some(final_acc) = es_result.history.final_train_accuracy() {
                println!("    Final train accuracy: {:.1}%", final_acc * 100.0);
            }
            if let Some(final_val_acc) = es_result.history.final_val_accuracy() {
                println!(
                    "    Final validation accuracy: {:.1}%",
                    final_val_acc * 100.0
                );
            }
            if let Some(ratio) = es_result.improvement_ratio() {
                println!("    Improvement ratio: {:.2}x", 1.0 / ratio);
            }
        }
    }

    // Print scheduler information if available
    if let Some(sched_result) = scheduled {
        println!("\n  Learning rate scheduler results:");
        println!("    Scheduler: {}", sched_result.scheduler_name);
        if verbose && !sched_result.lr_history.is_empty() {
            println!(
                "    Initial LR: {:.6}, Final LR: {:.6}",
                sched_result.lr_history.first().copied().unwrap_or(0.0),
                sched_result.final_lr
            );
            let reduction = sched_result.lr_reduction_factor();
            if reduction < 1.0 {
                println!("    LR reduction factor: {:.2}x", 1.0 / reduction);
            }
            if let Some(acc) = sched_result.final_accuracy() {
                println!("    Final train accuracy: {:.1}%", acc * 100.0);
            }
            if let Some(val_acc) = sched_result.final_val_accuracy() {
                println!("    Final validation accuracy: {:.1}%", val_acc * 100.0);
            }
        }
    }

    println!("\nModel saved to: {}", output_path.display());
    println!("\nTo use this model for verification:");
    println!(
        "  dashprove verify spec.usl --ml --ml-model {}",
        output_path.display()
    );
}
