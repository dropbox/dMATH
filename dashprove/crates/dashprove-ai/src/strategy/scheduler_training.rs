//! Learning rate scheduler-based training methods for StrategyPredictor
//!
//! This module contains training methods that use learning rate schedulers
//! to dynamically adjust the learning rate during training:
//! - Training with various scheduler types (Step, Exponential, Cosine, etc.)
//! - Combined scheduler + early stopping training

use super::{
    DenseLayer, EarlyStoppingConfig, EarlyStoppingResult, EpochMetrics, LearningRateScheduler,
    ScheduledTrainingResult, StrategyPredictor, TrainingExample, TrainingHistory,
};

impl StrategyPredictor {
    /// Train with a learning rate scheduler
    ///
    /// This method provides fine-grained control over learning rate during training.
    /// Different schedulers can help with different training scenarios:
    /// - Step: Simple periodic reduction, good baseline
    /// - Exponential: Smooth decay, good for general training
    /// - Cosine: Cyclic learning, can escape local minima
    /// - ReduceOnPlateau: Adaptive, reduces when stuck
    /// - WarmupDecay: Good for training from scratch
    ///
    /// # Arguments
    /// * `data` - Training examples
    /// * `initial_lr` - Starting learning rate
    /// * `epochs` - Number of training epochs
    /// * `validation_split` - Fraction for validation (0.0-0.5)
    /// * `scheduler` - Learning rate scheduling strategy
    ///
    /// # Returns
    /// `ScheduledTrainingResult` with full history and lr tracking
    pub fn train_with_scheduler(
        &mut self,
        data: &[TrainingExample],
        initial_lr: f64,
        epochs: usize,
        validation_split: f64,
        scheduler: LearningRateScheduler,
    ) -> ScheduledTrainingResult {
        let mut history = TrainingHistory::new();
        let mut lr_history = Vec::new();
        let mut val_loss_history = Vec::new();

        if data.is_empty() {
            return ScheduledTrainingResult {
                history,
                lr_history,
                final_lr: initial_lr,
                scheduler_name: scheduler.name().to_string(),
                used_early_stopping: false,
                early_stopping_info: None,
            };
        }

        // Split data into training and validation sets
        let val_size = ((data.len() as f64) * validation_split.clamp(0.0, 0.5)) as usize;
        let (train_data, val_data) = if val_size > 0 {
            let split_idx = data.len() - val_size;
            (&data[..split_idx], Some(&data[split_idx..]))
        } else {
            (data, None)
        };

        for epoch in 0..epochs {
            // Get learning rate for this epoch
            let lr = scheduler.get_lr(initial_lr, epoch, &val_loss_history);
            lr_history.push(lr);

            // Training pass
            let mut train_loss = 0.0;
            let mut train_correct = 0;

            for example in train_data {
                let loss = self.train_step_with_loss(example, lr);
                train_loss += loss;

                let prediction = self.predict_backend(&example.features);
                if prediction.backend == example.backend {
                    train_correct += 1;
                }
            }

            let train_accuracy = train_correct as f64 / train_data.len() as f64;
            train_loss /= train_data.len() as f64;

            // Validation pass
            let (val_loss, val_accuracy) = if let Some(val) = val_data {
                let eval = self.evaluate(val);
                val_loss_history.push(eval.loss);
                (Some(eval.loss), Some(eval.accuracy))
            } else {
                (None, None)
            };

            history.epochs.push(EpochMetrics {
                epoch,
                train_loss,
                train_accuracy,
                val_loss,
                val_accuracy,
            });
        }

        let final_lr = lr_history.last().copied().unwrap_or(initial_lr);

        ScheduledTrainingResult {
            history,
            lr_history,
            final_lr,
            scheduler_name: scheduler.name().to_string(),
            used_early_stopping: false,
            early_stopping_info: None,
        }
    }

    /// Train with both a learning rate scheduler and early stopping
    ///
    /// Combines the benefits of learning rate scheduling with automatic stopping
    /// when training stalls. This is the most sophisticated training method.
    ///
    /// # Arguments
    /// * `data` - Training examples
    /// * `initial_lr` - Starting learning rate
    /// * `max_epochs` - Maximum epochs to train
    /// * `validation_split` - Fraction for validation (0.1-0.3 recommended)
    /// * `scheduler` - Learning rate scheduling strategy
    /// * `early_stopping` - Early stopping configuration
    ///
    /// # Returns
    /// `ScheduledTrainingResult` with lr tracking and early stopping info
    pub fn train_with_scheduler_and_early_stopping(
        &mut self,
        data: &[TrainingExample],
        initial_lr: f64,
        max_epochs: usize,
        validation_split: f64,
        scheduler: LearningRateScheduler,
        early_stopping: EarlyStoppingConfig,
    ) -> ScheduledTrainingResult {
        let mut history = TrainingHistory::new();
        let mut lr_history = Vec::new();
        let mut val_loss_history = Vec::new();

        if data.is_empty() {
            return ScheduledTrainingResult {
                history,
                lr_history,
                final_lr: initial_lr,
                scheduler_name: scheduler.name().to_string(),
                used_early_stopping: true,
                early_stopping_info: Some(EarlyStoppingResult {
                    history: TrainingHistory::new(),
                    stopped_early: false,
                    final_epoch: 0,
                    best_epoch: 0,
                    best_val_loss: f64::INFINITY,
                    epochs_without_improvement: 0,
                }),
            };
        }

        // Split data into training and validation sets
        let val_size = ((data.len() as f64) * validation_split.clamp(0.1, 0.5)).max(1.0) as usize;
        let split_idx = data.len().saturating_sub(val_size);
        let (train_data, val_data) = (&data[..split_idx], &data[split_idx..]);

        if train_data.is_empty() || val_data.is_empty() {
            return ScheduledTrainingResult {
                history,
                lr_history,
                final_lr: initial_lr,
                scheduler_name: scheduler.name().to_string(),
                used_early_stopping: true,
                early_stopping_info: Some(EarlyStoppingResult {
                    history: TrainingHistory::new(),
                    stopped_early: false,
                    final_epoch: 0,
                    best_epoch: 0,
                    best_val_loss: f64::INFINITY,
                    epochs_without_improvement: 0,
                }),
            };
        }

        // Track best model state
        let mut best_val_loss = f64::INFINITY;
        let mut best_epoch = 0;
        let mut epochs_without_improvement = 0;
        let mut best_weights: Option<(DenseLayer, DenseLayer, DenseLayer)> = None;
        let mut stopped_early = false;
        let mut final_epoch = max_epochs.saturating_sub(1);

        for epoch in 0..max_epochs {
            // Get learning rate for this epoch
            let lr = scheduler.get_lr(initial_lr, epoch, &val_loss_history);
            lr_history.push(lr);

            // Training pass
            let mut train_loss = 0.0;
            let mut train_correct = 0;

            for example in train_data {
                let loss = self.train_step_with_loss(example, lr);
                train_loss += loss;

                let prediction = self.predict_backend(&example.features);
                if prediction.backend == example.backend {
                    train_correct += 1;
                }
            }

            let train_accuracy = train_correct as f64 / train_data.len() as f64;
            train_loss /= train_data.len() as f64;

            // Validation pass
            let eval = self.evaluate(val_data);
            let val_loss = eval.loss;
            let val_accuracy = eval.accuracy;
            val_loss_history.push(val_loss);

            history.epochs.push(EpochMetrics {
                epoch,
                train_loss,
                train_accuracy,
                val_loss: Some(val_loss),
                val_accuracy: Some(val_accuracy),
            });

            // Check for improvement
            if val_loss < best_val_loss - early_stopping.min_delta {
                best_val_loss = val_loss;
                best_epoch = epoch;
                epochs_without_improvement = 0;

                if early_stopping.restore_best_weights {
                    best_weights = Some((
                        self.hidden1.clone(),
                        self.hidden2.clone(),
                        self.backend_output.clone(),
                    ));
                }
            } else {
                epochs_without_improvement += 1;
            }

            // Check early stopping condition
            if epochs_without_improvement >= early_stopping.patience {
                stopped_early = true;
                final_epoch = epoch;

                if early_stopping.restore_best_weights {
                    if let Some((h1, h2, bo)) = best_weights.take() {
                        self.hidden1 = h1;
                        self.hidden2 = h2;
                        self.backend_output = bo;
                    }
                }
                break;
            }
        }

        // If completed without early stopping, optionally restore best weights
        if !stopped_early && early_stopping.restore_best_weights {
            if let Some((h1, h2, bo)) = best_weights.take() {
                self.hidden1 = h1;
                self.hidden2 = h2;
                self.backend_output = bo;
            }
        }

        let final_lr = lr_history.last().copied().unwrap_or(initial_lr);

        ScheduledTrainingResult {
            history: history.clone(),
            lr_history,
            final_lr,
            scheduler_name: scheduler.name().to_string(),
            used_early_stopping: true,
            early_stopping_info: Some(EarlyStoppingResult {
                history,
                stopped_early,
                final_epoch,
                best_epoch,
                best_val_loss,
                epochs_without_improvement,
            }),
        }
    }
}
