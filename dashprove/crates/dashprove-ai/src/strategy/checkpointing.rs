//! Model checkpointing for training
//!
//! This module provides checkpointing functionality for saving model snapshots
//! during training, which is useful for:
//! - Recovering from interrupted training
//! - Keeping the best model based on validation metrics
//! - Analyzing training progress over time

use serde::{Deserialize, Serialize};

use super::{StrategyPredictor, TrainingHistory};

/// Configuration for model checkpointing during training
///
/// Checkpointing allows saving model snapshots during training, which is useful for:
/// - Recovering from interrupted training
/// - Keeping the best model based on validation metrics
/// - Analyzing training progress over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    /// Directory to save checkpoints
    pub checkpoint_dir: std::path::PathBuf,
    /// Save checkpoint every N epochs (0 to disable periodic saving)
    pub save_every_n_epochs: usize,
    /// Keep only the N best checkpoints by validation loss (0 to keep all)
    pub keep_best_n: usize,
    /// Always save when validation loss improves
    pub save_on_improvement: bool,
    /// Include training history in checkpoint
    pub include_history: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            checkpoint_dir: std::path::PathBuf::from("."),
            save_every_n_epochs: 0,
            keep_best_n: 3,
            save_on_improvement: true,
            include_history: true,
        }
    }
}

impl CheckpointConfig {
    /// Create a new checkpoint configuration with the specified directory
    pub fn new<P: Into<std::path::PathBuf>>(checkpoint_dir: P) -> Self {
        Self {
            checkpoint_dir: checkpoint_dir.into(),
            ..Default::default()
        }
    }

    /// Set how often to save periodic checkpoints
    pub fn with_save_interval(mut self, epochs: usize) -> Self {
        self.save_every_n_epochs = epochs;
        self
    }

    /// Set how many best checkpoints to keep
    pub fn with_keep_best(mut self, n: usize) -> Self {
        self.keep_best_n = n;
        self
    }

    /// Enable or disable saving on validation improvement
    pub fn with_save_on_improvement(mut self, enabled: bool) -> Self {
        self.save_on_improvement = enabled;
        self
    }

    /// Include or exclude training history in checkpoints
    pub fn with_history(mut self, include: bool) -> Self {
        self.include_history = include;
        self
    }
}

/// A saved model checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Epoch at which checkpoint was saved
    pub epoch: usize,
    /// Validation loss at checkpoint time
    pub val_loss: f64,
    /// Validation accuracy at checkpoint time
    pub val_accuracy: f64,
    /// Training loss at checkpoint time
    pub train_loss: f64,
    /// Training accuracy at checkpoint time
    pub train_accuracy: f64,
    /// Model weights (serialized)
    pub model: StrategyPredictor,
    /// Optional training history up to this point
    pub history: Option<TrainingHistory>,
    /// Timestamp when checkpoint was created
    pub timestamp: String,
}

impl Checkpoint {
    /// Get the checkpoint filename based on epoch and validation loss
    pub fn filename(&self) -> String {
        format!(
            "checkpoint_epoch_{:04}_val_loss_{:.6}.json",
            self.epoch, self.val_loss
        )
    }
}

/// Result of training with checkpointing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointedTrainingResult {
    /// Training history
    pub history: TrainingHistory,
    /// List of checkpoints saved during training
    pub checkpoints_saved: Vec<String>,
    /// Path to the best checkpoint
    pub best_checkpoint_path: Option<String>,
    /// Best validation loss achieved
    pub best_val_loss: f64,
    /// Epoch of best validation loss
    pub best_epoch: usize,
    /// Total epochs trained
    pub total_epochs: usize,
}
