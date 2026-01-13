//! Training metrics and result types for the strategy module.
//!
//! This module contains types for tracking training progress, evaluation results,
//! early stopping configuration, and cross-validation.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use dashprove_backends::BackendId;

/// Metrics for a single training epoch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochMetrics {
    /// Epoch number (0-indexed)
    pub epoch: usize,
    /// Average training loss for this epoch
    pub train_loss: f64,
    /// Training accuracy for this epoch
    pub train_accuracy: f64,
    /// Validation loss (if validation split was used)
    pub val_loss: Option<f64>,
    /// Validation accuracy (if validation split was used)
    pub val_accuracy: Option<f64>,
}

/// Training history containing metrics for all epochs
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingHistory {
    /// Per-epoch metrics
    pub epochs: Vec<EpochMetrics>,
}

impl TrainingHistory {
    /// Create a new empty training history
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the final training accuracy
    pub fn final_train_accuracy(&self) -> Option<f64> {
        self.epochs.last().map(|e| e.train_accuracy)
    }

    /// Get the final validation accuracy
    pub fn final_val_accuracy(&self) -> Option<f64> {
        self.epochs.last().and_then(|e| e.val_accuracy)
    }

    /// Get the final training loss
    pub fn final_train_loss(&self) -> Option<f64> {
        self.epochs.last().map(|e| e.train_loss)
    }

    /// Get the final validation loss
    pub fn final_val_loss(&self) -> Option<f64> {
        self.epochs.last().and_then(|e| e.val_loss)
    }

    /// Get the best validation accuracy and its epoch
    pub fn best_val_accuracy(&self) -> Option<(usize, f64)> {
        self.epochs
            .iter()
            .filter_map(|e| e.val_accuracy.map(|acc| (e.epoch, acc)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    }

    /// Check if training is likely overfitting (validation loss increasing)
    pub fn is_overfitting(&self, lookback: usize) -> bool {
        if self.epochs.len() < lookback + 1 {
            return false;
        }
        let recent: Vec<_> = self
            .epochs
            .iter()
            .rev()
            .take(lookback)
            .filter_map(|e| e.val_loss)
            .collect();
        if recent.len() < lookback {
            return false;
        }
        // Check if validation loss is monotonically increasing
        recent.windows(2).all(|w| w[0] >= w[1])
    }
}

/// Result of model evaluation on test data
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvaluationResult {
    /// Overall accuracy (correct / total)
    pub accuracy: f64,
    /// Average cross-entropy loss
    pub loss: f64,
    /// Total number of test examples
    pub total_examples: usize,
    /// Number of correct predictions
    pub correct_predictions: usize,
    /// Accuracy per backend
    pub per_backend_accuracy: HashMap<BackendId, f64>,
    /// Average confidence for correct predictions
    pub avg_confidence_correct: f64,
    /// Average confidence for incorrect predictions
    pub avg_confidence_incorrect: f64,
}

impl EvaluationResult {
    /// Get accuracy for a specific backend
    pub fn backend_accuracy(&self, backend: BackendId) -> Option<f64> {
        self.per_backend_accuracy.get(&backend).copied()
    }

    /// Check if model is well-calibrated (high confidence on correct, low on incorrect)
    pub fn is_well_calibrated(&self) -> bool {
        self.avg_confidence_correct > self.avg_confidence_incorrect + 0.1
    }
}

/// Configuration for early stopping during training
///
/// Early stopping monitors validation loss and stops training when
/// improvement stalls, preventing overfitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    /// Number of epochs to wait for improvement before stopping
    pub patience: usize,
    /// Minimum change in validation loss to qualify as improvement
    pub min_delta: f64,
    /// Whether to restore the best model weights when stopping
    pub restore_best_weights: bool,
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            patience: 5,
            min_delta: 0.001,
            restore_best_weights: true,
        }
    }
}

impl EarlyStoppingConfig {
    /// Create a new early stopping configuration
    pub fn new(patience: usize, min_delta: f64) -> Self {
        Self {
            patience,
            min_delta,
            restore_best_weights: true,
        }
    }

    /// Set whether to restore best weights when stopping
    pub fn with_restore_best(mut self, restore: bool) -> Self {
        self.restore_best_weights = restore;
        self
    }
}

/// Result of training with early stopping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingResult {
    /// Training history with all epoch metrics
    pub history: TrainingHistory,
    /// Whether training stopped early
    pub stopped_early: bool,
    /// Epoch at which training stopped (or completed)
    pub final_epoch: usize,
    /// Best epoch (lowest validation loss)
    pub best_epoch: usize,
    /// Best validation loss achieved
    pub best_val_loss: f64,
    /// Number of epochs without improvement when stopped
    pub epochs_without_improvement: usize,
}

impl EarlyStoppingResult {
    /// Check if training converged (stopped early with good validation)
    pub fn converged(&self) -> bool {
        self.stopped_early && self.best_val_loss < 1.0
    }

    /// Get the improvement ratio (best loss / first loss)
    pub fn improvement_ratio(&self) -> Option<f64> {
        self.history
            .epochs
            .first()
            .and_then(|first| first.val_loss)
            .map(|first_loss| self.best_val_loss / first_loss)
    }
}

/// Result of k-fold cross-validation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CrossValidationResult {
    /// Number of folds
    pub k: usize,
    /// Mean accuracy across folds
    pub mean_accuracy: f64,
    /// Standard deviation of accuracy
    pub std_accuracy: f64,
    /// Mean loss across folds
    pub mean_loss: f64,
    /// Standard deviation of loss
    pub std_loss: f64,
    /// Individual fold results
    pub fold_results: Vec<EvaluationResult>,
}

impl CrossValidationResult {
    /// Get the 95% confidence interval for accuracy
    pub fn accuracy_confidence_interval(&self) -> (f64, f64) {
        // Using 1.96 * std for 95% CI (assumes normal distribution)
        let margin = 1.96 * self.std_accuracy / (self.k as f64).sqrt();
        (self.mean_accuracy - margin, self.mean_accuracy + margin)
    }

    /// Check if model has high variance (potential overfitting)
    pub fn has_high_variance(&self) -> bool {
        self.std_accuracy > 0.1 // More than 10% standard deviation
    }
}
