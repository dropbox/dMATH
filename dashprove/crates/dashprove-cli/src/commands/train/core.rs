//! Core training functionality
//!
//! Contains the main run_train function and scheduler builder.

use dashprove::ai::strategy::{
    EarlyStoppingConfig, EarlyStoppingResult, LearningRateScheduler, ScheduledTrainingResult,
    StrategyPredictor, TrainingDataGenerator, TrainingExample,
};
use dashprove::learning::ProofCorpus;
use std::path::PathBuf;

use super::checkpoint::{run_train_resume, run_train_with_checkpointing};
use super::config::{default_data_dir, SchedulerType, TrainConfig};
use super::summary::print_training_summary;

/// Run the train command
pub fn run_train(config: TrainConfig<'_>) -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = config
        .data_dir
        .map(PathBuf::from)
        .unwrap_or_else(default_data_dir);

    let corpus_path = data_dir.join("proof_corpus.json");

    if config.verbose {
        println!("Loading proof corpus from: {}", corpus_path.display());
    }

    // Load the proof corpus
    let corpus = ProofCorpus::load_or_default(&corpus_path).map_err(|e| {
        format!(
            "Failed to load corpus from {}: {}",
            corpus_path.display(),
            e
        )
    })?;

    let corpus_size = corpus.len();
    if corpus_size == 0 {
        println!(
            "Proof corpus is empty. Run verifications with --learn to build the corpus first."
        );
        println!("Example: dashprove verify spec.usl --learn");
        return Ok(());
    }

    println!("Loaded {} proofs from corpus", corpus_size);

    // Convert corpus entries to training data format
    let entries: Vec<_> = corpus
        .entries()
        .map(|entry| {
            let time_secs = entry.time_taken.as_secs_f64();
            (
                &entry.property,
                entry.backend,
                entry.tactics.as_slice(),
                time_secs,
            )
        })
        .collect();

    // Generate training examples
    let generator = TrainingDataGenerator::from_corpus_entries(entries);
    let stats = generator.stats();
    let training_examples: Vec<TrainingExample> = generator.get_training_data();

    if training_examples.is_empty() {
        println!("No successful proofs in corpus to train on.");
        return Ok(());
    }

    // Build the learning rate scheduler
    let scheduler = build_scheduler(&config);
    let using_scheduler = config.lr_scheduler != SchedulerType::Constant;

    if config.verbose {
        println!(
            "Training with learning_rate={}, epochs={}",
            config.learning_rate, config.epochs
        );
        if using_scheduler {
            println!("LR scheduler: {:?}", config.lr_scheduler);
        }
        if config.early_stopping {
            println!(
                "Early stopping enabled: patience={}, min_delta={}, validation_split={}",
                config.patience, config.min_delta, config.validation_split
            );
        }
        if config.checkpoint {
            let checkpoint_dir = config
                .checkpoint_dir
                .map(PathBuf::from)
                .unwrap_or_else(|| data_dir.join("checkpoints"));
            println!("Checkpointing enabled: dir={}", checkpoint_dir.display());
            if config.checkpoint_interval > 0 {
                println!(
                    "  Checkpoint interval: {} epochs",
                    config.checkpoint_interval
                );
            }
            println!("  Keep best: {}", config.keep_best);
        }
    }

    // Determine output path
    let output_path = config
        .output
        .map(PathBuf::from)
        .unwrap_or_else(|| data_dir.join("strategy_model.json"));

    // Handle resume from checkpoint
    if let Some(resume_path) = config.resume {
        return run_train_resume(
            &config,
            &training_examples,
            &stats,
            &output_path,
            resume_path,
            &data_dir,
        );
    }

    // Handle checkpointed training
    if config.checkpoint {
        return run_train_with_checkpointing(
            &config,
            &training_examples,
            &stats,
            &output_path,
            &data_dir,
        );
    }

    // Standard training (no checkpointing)
    let mut predictor = StrategyPredictor::new();

    // Store training results
    let (early_stopping_result, scheduled_result): (
        Option<EarlyStoppingResult>,
        Option<ScheduledTrainingResult>,
    ) = if using_scheduler {
        // Use scheduler-based training
        if config.early_stopping {
            let es_config = EarlyStoppingConfig::new(config.patience, config.min_delta);
            let result = predictor.train_with_scheduler_and_early_stopping(
                &training_examples,
                config.learning_rate,
                config.epochs,
                config.validation_split,
                scheduler,
                es_config,
            );
            (result.early_stopping_info.clone(), Some(result))
        } else {
            let result = predictor.train_with_scheduler(
                &training_examples,
                config.learning_rate,
                config.epochs,
                config.validation_split,
                scheduler,
            );
            (None, Some(result))
        }
    } else if config.early_stopping {
        let es_config = EarlyStoppingConfig::new(config.patience, config.min_delta);
        let result = predictor.train_with_early_stopping(
            &training_examples,
            config.learning_rate,
            config.epochs,
            config.validation_split,
            es_config,
        );
        (Some(result), None)
    } else {
        predictor.train(&training_examples, config.learning_rate, config.epochs);
        (None, None)
    };

    // Ensure parent directory exists
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create directory {}: {}", parent.display(), e))?;
    }

    // Save the model
    predictor
        .save(&output_path)
        .map_err(|e| format!("Failed to save model to {}: {}", output_path.display(), e))?;

    // Print training summary
    print_training_summary(
        &stats,
        &output_path,
        config.verbose,
        early_stopping_result.as_ref(),
        scheduled_result.as_ref(),
    );

    Ok(())
}

/// Build a learning rate scheduler from the config
pub fn build_scheduler(config: &TrainConfig<'_>) -> LearningRateScheduler {
    match config.lr_scheduler {
        SchedulerType::Constant => LearningRateScheduler::Constant,
        SchedulerType::Step => LearningRateScheduler::step(config.lr_step_size, config.lr_gamma),
        SchedulerType::Exponential => LearningRateScheduler::exponential(config.lr_gamma),
        SchedulerType::Cosine => LearningRateScheduler::cosine(config.lr_min, config.epochs),
        SchedulerType::ReduceOnPlateau => {
            LearningRateScheduler::reduce_on_plateau(config.lr_gamma, config.patience)
        }
        SchedulerType::WarmupDecay => LearningRateScheduler::warmup_decay(
            config.lr_warmup_epochs,
            config.learning_rate,
            config.lr_gamma,
        ),
    }
}
