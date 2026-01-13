//! Learning rate scheduler types for the strategy module.
//!
//! This module contains types for adjusting learning rates during training,
//! including various scheduling strategies and their results.

use serde::{Deserialize, Serialize};

use super::training::{EarlyStoppingResult, TrainingHistory};

/// Learning rate scheduler type
///
/// Defines different strategies for adjusting the learning rate during training.
/// Learning rate scheduling can help training converge faster and to better optima.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum LearningRateScheduler {
    /// Constant learning rate (no scheduling)
    #[default]
    Constant,
    /// Step decay: multiply by factor every step_size epochs
    Step {
        /// Number of epochs between learning rate reductions
        step_size: usize,
        /// Factor to multiply learning rate by (typically 0.1-0.5)
        gamma: f64,
    },
    /// Exponential decay: lr = initial_lr * gamma^epoch
    Exponential {
        /// Decay factor per epoch (typically 0.95-0.99)
        gamma: f64,
    },
    /// Cosine annealing: lr oscillates between max and min following cosine curve
    Cosine {
        /// Minimum learning rate
        min_lr: f64,
        /// Period of the cosine cycle in epochs
        t_max: usize,
    },
    /// Reduce on plateau: reduce lr when validation loss stalls
    ReduceOnPlateau {
        /// Factor to multiply learning rate by (typically 0.1-0.5)
        factor: f64,
        /// Number of epochs to wait before reducing
        patience: usize,
        /// Minimum change to qualify as improvement
        threshold: f64,
        /// Minimum learning rate (won't reduce below this)
        min_lr: f64,
    },
    /// Linear warmup followed by decay
    WarmupDecay {
        /// Number of warmup epochs
        warmup_epochs: usize,
        /// Final learning rate after warmup (peak)
        peak_lr: f64,
        /// Decay rate after warmup (exponential)
        decay_rate: f64,
    },
}

impl LearningRateScheduler {
    /// Create a step decay scheduler
    pub fn step(step_size: usize, gamma: f64) -> Self {
        LearningRateScheduler::Step { step_size, gamma }
    }

    /// Create an exponential decay scheduler
    pub fn exponential(gamma: f64) -> Self {
        LearningRateScheduler::Exponential { gamma }
    }

    /// Create a cosine annealing scheduler
    pub fn cosine(min_lr: f64, t_max: usize) -> Self {
        LearningRateScheduler::Cosine { min_lr, t_max }
    }

    /// Create a reduce-on-plateau scheduler
    pub fn reduce_on_plateau(factor: f64, patience: usize) -> Self {
        LearningRateScheduler::ReduceOnPlateau {
            factor,
            patience,
            threshold: 0.001,
            min_lr: 1e-6,
        }
    }

    /// Create a warmup decay scheduler
    pub fn warmup_decay(warmup_epochs: usize, peak_lr: f64, decay_rate: f64) -> Self {
        LearningRateScheduler::WarmupDecay {
            warmup_epochs,
            peak_lr,
            decay_rate,
        }
    }

    /// Compute the learning rate for a given epoch
    ///
    /// # Arguments
    /// * `initial_lr` - The initial learning rate
    /// * `epoch` - Current epoch (0-indexed)
    /// * `val_loss_history` - History of validation losses (for plateau detection)
    pub fn get_lr(&self, initial_lr: f64, epoch: usize, val_loss_history: &[f64]) -> f64 {
        match self {
            LearningRateScheduler::Constant => initial_lr,

            LearningRateScheduler::Step { step_size, gamma } => {
                let num_reductions = epoch / step_size;
                initial_lr * gamma.powi(num_reductions as i32)
            }

            LearningRateScheduler::Exponential { gamma } => initial_lr * gamma.powi(epoch as i32),

            LearningRateScheduler::Cosine { min_lr, t_max } => {
                let t_max = (*t_max).max(1);
                let cos_value =
                    (std::f64::consts::PI * (epoch % t_max) as f64 / t_max as f64).cos();
                min_lr + (initial_lr - min_lr) * (1.0 + cos_value) / 2.0
            }

            LearningRateScheduler::ReduceOnPlateau {
                factor,
                patience,
                threshold,
                min_lr,
            } => {
                if val_loss_history.len() < *patience + 1 {
                    return initial_lr;
                }

                // Check if we should reduce
                let mut num_reductions = 0;
                let mut best_loss = f64::INFINITY;
                let mut epochs_without_improvement = 0;

                for &loss in val_loss_history {
                    if loss < best_loss - threshold {
                        best_loss = loss;
                        epochs_without_improvement = 0;
                    } else {
                        epochs_without_improvement += 1;
                        if epochs_without_improvement >= *patience {
                            num_reductions += 1;
                            epochs_without_improvement = 0;
                        }
                    }
                }

                (initial_lr * factor.powi(num_reductions)).max(*min_lr)
            }

            LearningRateScheduler::WarmupDecay {
                warmup_epochs,
                peak_lr,
                decay_rate,
            } => {
                if epoch < *warmup_epochs {
                    // Linear warmup
                    initial_lr + (peak_lr - initial_lr) * (epoch + 1) as f64 / *warmup_epochs as f64
                } else {
                    // Exponential decay after warmup
                    let decay_epoch = epoch - warmup_epochs;
                    peak_lr * decay_rate.powi(decay_epoch as i32)
                }
            }
        }
    }

    /// Get a descriptive name for the scheduler
    pub fn name(&self) -> &'static str {
        match self {
            LearningRateScheduler::Constant => "constant",
            LearningRateScheduler::Step { .. } => "step",
            LearningRateScheduler::Exponential { .. } => "exponential",
            LearningRateScheduler::Cosine { .. } => "cosine",
            LearningRateScheduler::ReduceOnPlateau { .. } => "reduce_on_plateau",
            LearningRateScheduler::WarmupDecay { .. } => "warmup_decay",
        }
    }
}

/// Result of training with a learning rate scheduler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledTrainingResult {
    /// Full training history
    pub history: TrainingHistory,
    /// Learning rate history (one per epoch)
    pub lr_history: Vec<f64>,
    /// Final learning rate used
    pub final_lr: f64,
    /// Scheduler that was used
    pub scheduler_name: String,
    /// Whether training used early stopping
    pub used_early_stopping: bool,
    /// Early stopping result (if used)
    pub early_stopping_info: Option<EarlyStoppingResult>,
}

impl ScheduledTrainingResult {
    /// Get the final training accuracy
    pub fn final_accuracy(&self) -> Option<f64> {
        self.history.final_train_accuracy()
    }

    /// Get the final validation accuracy
    pub fn final_val_accuracy(&self) -> Option<f64> {
        self.history.final_val_accuracy()
    }

    /// Get the learning rate at a specific epoch
    pub fn lr_at_epoch(&self, epoch: usize) -> Option<f64> {
        self.lr_history.get(epoch).copied()
    }

    /// Get the total learning rate reduction factor
    pub fn lr_reduction_factor(&self) -> f64 {
        if let (Some(&first), Some(&last)) = (self.lr_history.first(), self.lr_history.last()) {
            if first > 0.0 {
                last / first
            } else {
                1.0
            }
        } else {
            1.0
        }
    }
}
