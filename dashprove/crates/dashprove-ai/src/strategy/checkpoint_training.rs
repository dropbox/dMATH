//! Model checkpointing functionality for StrategyPredictor
//!
//! This module provides checkpoint-based training with model saving,
//! best model tracking, and training resumption.

use super::{
    backend_to_idx, Checkpoint, CheckpointConfig, CheckpointedTrainingResult, EpochMetrics,
    StrategyPredictor, TrainingExample, TrainingHistory,
};

// ============================================================================
// Model Checkpointing
// ============================================================================

/// Checkpointing functionality for the StrategyPredictor
impl StrategyPredictor {
    /// Train with checkpointing enabled
    ///
    /// Saves model checkpoints during training according to the configuration.
    /// This allows recovery from interrupted training and keeping the best model.
    ///
    /// # Arguments
    /// * `data` - Training data
    /// * `learning_rate` - Initial learning rate
    /// * `epochs` - Number of epochs to train
    /// * `validation_split` - Fraction of data to use for validation
    /// * `checkpoint_config` - Checkpointing configuration
    ///
    /// # Returns
    /// `CheckpointedTrainingResult` with training history and checkpoint info
    pub fn train_with_checkpointing(
        &mut self,
        data: &[TrainingExample],
        learning_rate: f64,
        epochs: usize,
        validation_split: f64,
        checkpoint_config: &CheckpointConfig,
    ) -> std::io::Result<CheckpointedTrainingResult> {
        // Create checkpoint directory if it doesn't exist
        std::fs::create_dir_all(&checkpoint_config.checkpoint_dir)?;

        // Split data into train and validation sets
        let val_size = (data.len() as f64 * validation_split).max(1.0) as usize;
        let train_size = data.len().saturating_sub(val_size);
        let (train_data, val_data) = data.split_at(train_size);

        let mut history = TrainingHistory::new();
        let mut checkpoints_saved: Vec<String> = Vec::new();
        let mut best_val_loss = f64::INFINITY;
        let mut best_epoch = 0;
        let mut best_checkpoint_path: Option<String> = None;
        // Track best checkpoints by (val_loss, path)
        let mut best_checkpoints: Vec<(f64, String)> = Vec::new();

        for epoch in 0..epochs {
            // Training
            let mut train_loss = 0.0;
            let mut train_correct = 0;
            for example in train_data {
                self.train_step(example, learning_rate);
                let features = &example.features;
                let prediction = self.predict_backend(features);
                if prediction.backend == example.backend {
                    train_correct += 1;
                }
                train_loss += self.compute_loss(example);
            }
            let train_accuracy = if train_data.is_empty() {
                0.0
            } else {
                train_correct as f64 / train_data.len() as f64
            };
            train_loss /= train_data.len().max(1) as f64;

            // Validation
            let mut val_loss = 0.0;
            let mut val_correct = 0;
            for example in val_data {
                let prediction = self.predict_backend(&example.features);
                if prediction.backend == example.backend {
                    val_correct += 1;
                }
                val_loss += self.compute_loss(example);
            }
            let val_accuracy = if val_data.is_empty() {
                0.0
            } else {
                val_correct as f64 / val_data.len() as f64
            };
            val_loss /= val_data.len().max(1) as f64;

            // Record metrics
            history.epochs.push(EpochMetrics {
                epoch,
                train_loss,
                train_accuracy,
                val_loss: Some(val_loss),
                val_accuracy: Some(val_accuracy),
            });

            // Check if this is the best validation loss
            let is_improvement = val_loss < best_val_loss;
            if is_improvement {
                best_val_loss = val_loss;
                best_epoch = epoch;
            }

            // Determine if we should save a checkpoint
            let should_save_periodic = checkpoint_config.save_every_n_epochs > 0
                && (epoch + 1) % checkpoint_config.save_every_n_epochs == 0;
            let should_save_improvement = checkpoint_config.save_on_improvement && is_improvement;

            if should_save_periodic || should_save_improvement {
                let checkpoint = Checkpoint {
                    epoch,
                    val_loss,
                    val_accuracy,
                    train_loss,
                    train_accuracy,
                    model: self.clone(),
                    history: if checkpoint_config.include_history {
                        Some(history.clone())
                    } else {
                        None
                    },
                    timestamp: format!(
                        "{}",
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_secs())
                            .unwrap_or(0)
                    ),
                };

                let filename = checkpoint.filename();
                let path = checkpoint_config.checkpoint_dir.join(&filename);
                let json = serde_json::to_string_pretty(&checkpoint)?;
                std::fs::write(&path, json)?;

                let path_str = path.to_string_lossy().to_string();
                checkpoints_saved.push(path_str.clone());

                if is_improvement {
                    best_checkpoint_path = Some(path_str.clone());
                }

                // Track for keep_best_n
                best_checkpoints.push((val_loss, path_str));

                // Prune old checkpoints if keep_best_n > 0
                if checkpoint_config.keep_best_n > 0
                    && best_checkpoints.len() > checkpoint_config.keep_best_n
                {
                    // Sort by validation loss (ascending)
                    best_checkpoints.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                    // Remove checkpoints beyond keep_best_n
                    while best_checkpoints.len() > checkpoint_config.keep_best_n {
                        if let Some((_, path)) = best_checkpoints.pop() {
                            // Don't delete if it's the current best
                            if Some(&path) != best_checkpoint_path.as_ref() {
                                let _ = std::fs::remove_file(&path);
                            }
                        }
                    }
                }
            }
        }

        Ok(CheckpointedTrainingResult {
            history,
            checkpoints_saved,
            best_checkpoint_path,
            best_val_loss,
            best_epoch,
            total_epochs: epochs,
        })
    }

    /// Load the best model from a checkpointed training result
    pub fn load_best_checkpoint(
        result: &CheckpointedTrainingResult,
    ) -> std::io::Result<Option<StrategyPredictor>> {
        if let Some(ref path) = result.best_checkpoint_path {
            let json = std::fs::read_to_string(path)?;
            let checkpoint: Checkpoint = serde_json::from_str(&json)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            Ok(Some(checkpoint.model))
        } else {
            Ok(None)
        }
    }

    /// Resume training from a checkpoint
    ///
    /// Loads a checkpoint and continues training from where it left off.
    pub fn resume_from_checkpoint<P: AsRef<std::path::Path>>(
        checkpoint_path: P,
        additional_epochs: usize,
        learning_rate: f64,
        data: &[TrainingExample],
        validation_split: f64,
        checkpoint_config: Option<&CheckpointConfig>,
    ) -> std::io::Result<(StrategyPredictor, CheckpointedTrainingResult)> {
        // Load checkpoint
        let json = std::fs::read_to_string(checkpoint_path)?;
        let checkpoint: Checkpoint = serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let mut predictor = checkpoint.model;
        let starting_epoch = checkpoint.epoch + 1;

        // If no checkpoint config provided, just train without checkpointing
        if let Some(config) = checkpoint_config {
            // Continue with checkpointing
            let mut result = predictor.train_with_checkpointing(
                data,
                learning_rate,
                additional_epochs,
                validation_split,
                config,
            )?;

            // Adjust epoch numbers in the new history
            for metrics in &mut result.history.epochs {
                metrics.epoch += starting_epoch;
            }
            result.best_epoch += starting_epoch;

            Ok((predictor, result))
        } else {
            // Train without checkpointing
            let history = predictor.train_with_metrics(
                data,
                learning_rate,
                additional_epochs,
                validation_split,
            );

            Ok((
                predictor,
                CheckpointedTrainingResult {
                    history: history.clone(),
                    checkpoints_saved: Vec::new(),
                    best_checkpoint_path: None,
                    best_val_loss: history.final_val_loss().unwrap_or(f64::INFINITY),
                    best_epoch: history.best_val_accuracy().map(|(e, _)| e).unwrap_or(0),
                    total_epochs: additional_epochs,
                },
            ))
        }
    }

    /// Helper to compute cross-entropy loss for a single example
    pub(super) fn compute_loss(&self, example: &TrainingExample) -> f64 {
        let h1 = self.hidden1.forward_relu(&example.features.features);
        let h2 = self.hidden2.forward_relu(&h1);
        let probs = self.backend_output.forward_softmax(&h2);

        let target_idx = backend_to_idx(example.backend);
        if target_idx < probs.len() {
            -probs[target_idx].max(1e-10).ln()
        } else {
            10.0 // Large loss for invalid target
        }
    }
}
