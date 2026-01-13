//! Core training methods for StrategyPredictor
//!
//! This module contains fundamental training algorithms including:
//! - Basic training with gradient descent
//! - Training with metrics tracking
//! - Training with L2 regularization
//! - Training with early stopping
//! - K-fold cross-validation
//! - Model evaluation

use super::{
    backend_to_idx, relu_derivative, CrossValidationResult, DenseLayer, EarlyStoppingConfig,
    EarlyStoppingResult, EpochMetrics, EvaluationResult, StrategyPredictor, TrainingExample,
    TrainingHistory,
};
use dashprove_backends::BackendId;
use std::collections::HashMap;

impl StrategyPredictor {
    /// Update weights from training data (simple gradient descent)
    pub fn train(&mut self, data: &[TrainingExample], learning_rate: f64, epochs: usize) {
        for _epoch in 0..epochs {
            for example in data {
                self.train_step(example, learning_rate);
            }
        }
    }

    /// Train with metrics tracking, returning full training history
    ///
    /// This method tracks loss and accuracy per epoch, enabling analysis of
    /// training progress and detection of overfitting.
    ///
    /// # Arguments
    /// * `data` - Training examples
    /// * `learning_rate` - Learning rate for gradient descent
    /// * `epochs` - Number of training epochs
    /// * `validation_split` - Fraction of data to use for validation (0.0-1.0)
    ///
    /// # Returns
    /// `TrainingHistory` containing per-epoch metrics
    pub fn train_with_metrics(
        &mut self,
        data: &[TrainingExample],
        learning_rate: f64,
        epochs: usize,
        validation_split: f64,
    ) -> TrainingHistory {
        let mut history = TrainingHistory::new();

        if data.is_empty() {
            return history;
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
            // Training pass
            let mut train_loss = 0.0;
            let mut train_correct = 0;

            for example in train_data {
                let loss = self.train_step_with_loss(example, learning_rate);
                train_loss += loss;

                // Check if prediction is correct
                let prediction = self.predict_backend(&example.features);
                if prediction.backend == example.backend {
                    train_correct += 1;
                }
            }

            let train_accuracy = train_correct as f64 / train_data.len() as f64;
            train_loss /= train_data.len() as f64;

            // Validation pass (if we have validation data)
            let (val_loss, val_accuracy) = if let Some(val) = val_data {
                let eval = self.evaluate(val);
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

        history
    }

    /// Train with metrics and L2 regularization (weight decay)
    ///
    /// This method extends `train_with_metrics` with L2 regularization support,
    /// which helps prevent overfitting by penalizing large weights.
    ///
    /// # Arguments
    /// * `data` - Training examples
    /// * `learning_rate` - Learning rate for gradient descent
    /// * `epochs` - Number of training epochs
    /// * `validation_split` - Fraction of data to use for validation (0.0-1.0)
    /// * `weight_decay` - L2 regularization coefficient (recommended: 1e-4 to 1e-2)
    ///
    /// # Returns
    /// `TrainingHistory` containing per-epoch metrics
    pub fn train_with_regularization(
        &mut self,
        data: &[TrainingExample],
        learning_rate: f64,
        epochs: usize,
        validation_split: f64,
        weight_decay: f64,
    ) -> TrainingHistory {
        let mut history = TrainingHistory::new();

        if data.is_empty() {
            return history;
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
            // Training pass with weight decay
            let mut train_loss = 0.0;
            let mut train_correct = 0;

            for example in train_data {
                let loss = self.train_step_with_loss_full(example, learning_rate, weight_decay);
                train_loss += loss;

                // Check if prediction is correct
                let prediction = self.predict_backend(&example.features);
                if prediction.backend == example.backend {
                    train_correct += 1;
                }
            }

            let train_accuracy = train_correct as f64 / train_data.len() as f64;
            train_loss /= train_data.len() as f64;

            // Validation pass (if we have validation data)
            let (val_loss, val_accuracy) = if let Some(val) = val_data {
                let eval = self.evaluate(val);
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

        history
    }

    /// Single training step that returns loss for metrics tracking
    pub(crate) fn train_step_with_loss(
        &mut self,
        example: &TrainingExample,
        learning_rate: f64,
    ) -> f64 {
        self.train_step_with_loss_full(example, learning_rate, 0.0)
    }

    /// Single training step with full backpropagation that returns loss
    ///
    /// # Arguments
    /// * `example` - Training example
    /// * `learning_rate` - Learning rate for gradient descent
    /// * `weight_decay` - L2 regularization coefficient (0.0 for no regularization)
    pub(crate) fn train_step_with_loss_full(
        &mut self,
        example: &TrainingExample,
        learning_rate: f64,
        weight_decay: f64,
    ) -> f64 {
        let features = &example.features;

        // Forward pass - keep pre-activation values for ReLU derivative
        let h1_pre = self.hidden1.forward(&features.features);
        let h1 = h1_pre.iter().map(|&x| x.max(0.0)).collect::<Vec<_>>();
        let h2_pre = self.hidden2.forward(&h1);
        let h2 = h2_pre.iter().map(|&x| x.max(0.0)).collect::<Vec<_>>();
        let backend_probs = self.backend_output.forward_softmax(&h2);

        // Compute cross-entropy loss
        let target_idx = backend_to_idx(example.backend);
        let loss = if target_idx < backend_probs.len() {
            -backend_probs[target_idx].max(1e-10).ln()
        } else {
            0.0
        };

        // Compute backend loss gradient (cross-entropy derivative: softmax - one_hot)
        let mut backend_grad = backend_probs.clone();
        if target_idx < backend_grad.len() {
            backend_grad[target_idx] -= 1.0;
        }

        // Backpropagate through backend output layer -> get gradient w.r.t. h2
        let h2_grad = self
            .backend_output
            .backward(&h2, &backend_grad, learning_rate, weight_decay);

        // Apply ReLU derivative to h2_grad
        let h2_grad_relu = relu_derivative(&h2_pre, &h2_grad);

        // Backpropagate through hidden2 -> get gradient w.r.t. h1
        let h1_grad = self
            .hidden2
            .backward(&h1, &h2_grad_relu, learning_rate, weight_decay);

        // Apply ReLU derivative to h1_grad
        let h1_grad_relu = relu_derivative(&h1_pre, &h1_grad);

        // Backpropagate through hidden1 (updates weights, input gradient not needed)
        let _ = self.hidden1.backward(
            &features.features,
            &h1_grad_relu,
            learning_rate,
            weight_decay,
        );

        loss
    }

    /// Evaluate model accuracy on a test set
    ///
    /// Computes accuracy, average loss, and per-backend metrics on unseen data.
    pub fn evaluate(&self, test_data: &[TrainingExample]) -> EvaluationResult {
        if test_data.is_empty() {
            return EvaluationResult::default();
        }

        let mut total_loss = 0.0;
        let mut correct = 0;
        let mut per_backend_correct: HashMap<BackendId, usize> = HashMap::new();
        let mut per_backend_total: HashMap<BackendId, usize> = HashMap::new();
        let mut predictions: Vec<(BackendId, BackendId, f64)> = Vec::new(); // (actual, predicted, confidence)

        for example in test_data {
            let prediction = self.predict_backend(&example.features);

            // Track per-backend stats
            *per_backend_total.entry(example.backend).or_default() += 1;

            if prediction.backend == example.backend {
                correct += 1;
                *per_backend_correct.entry(example.backend).or_default() += 1;
            }

            // Compute loss
            let h1 = self.hidden1.forward_relu(&example.features.features);
            let h2 = self.hidden2.forward_relu(&h1);
            let backend_probs = self.backend_output.forward_softmax(&h2);
            let target_idx = backend_to_idx(example.backend);
            if target_idx < backend_probs.len() {
                total_loss += -backend_probs[target_idx].max(1e-10).ln();
            }

            predictions.push((example.backend, prediction.backend, prediction.confidence));
        }

        // Compute per-backend accuracy
        let per_backend_accuracy: HashMap<BackendId, f64> = per_backend_total
            .iter()
            .map(|(&backend, &total)| {
                let correct_count = *per_backend_correct.get(&backend).unwrap_or(&0);
                (backend, correct_count as f64 / total as f64)
            })
            .collect();

        // Compute average confidence for correct/incorrect predictions
        let (correct_conf_sum, correct_count): (f64, usize) = predictions
            .iter()
            .filter(|(actual, pred, _)| actual == pred)
            .map(|(_, _, conf)| (*conf, 1))
            .fold((0.0, 0), |(s, c), (conf, n)| (s + conf, c + n));

        let (incorrect_conf_sum, incorrect_count): (f64, usize) = predictions
            .iter()
            .filter(|(actual, pred, _)| actual != pred)
            .map(|(_, _, conf)| (*conf, 1))
            .fold((0.0, 0), |(s, c), (conf, n)| (s + conf, c + n));

        let avg_confidence_correct = if correct_count > 0 {
            correct_conf_sum / correct_count as f64
        } else {
            0.0
        };

        let avg_confidence_incorrect = if incorrect_count > 0 {
            incorrect_conf_sum / incorrect_count as f64
        } else {
            0.0
        };

        EvaluationResult {
            accuracy: correct as f64 / test_data.len() as f64,
            loss: total_loss / test_data.len() as f64,
            total_examples: test_data.len(),
            correct_predictions: correct,
            per_backend_accuracy,
            avg_confidence_correct,
            avg_confidence_incorrect,
        }
    }

    /// Train with early stopping to prevent overfitting
    ///
    /// Monitors validation loss and stops training when improvement stalls.
    /// This is the recommended training method for production models.
    ///
    /// # Arguments
    /// * `data` - Training examples
    /// * `learning_rate` - Learning rate for gradient descent
    /// * `max_epochs` - Maximum number of epochs to train
    /// * `validation_split` - Fraction of data for validation (0.1-0.3 recommended)
    /// * `early_stopping` - Early stopping configuration
    ///
    /// # Returns
    /// `EarlyStoppingResult` with training history and stopping information
    pub fn train_with_early_stopping(
        &mut self,
        data: &[TrainingExample],
        learning_rate: f64,
        max_epochs: usize,
        validation_split: f64,
        early_stopping: EarlyStoppingConfig,
    ) -> EarlyStoppingResult {
        let mut history = TrainingHistory::new();

        if data.is_empty() {
            return EarlyStoppingResult {
                history,
                stopped_early: false,
                final_epoch: 0,
                best_epoch: 0,
                best_val_loss: f64::INFINITY,
                epochs_without_improvement: 0,
            };
        }

        // Split data into training and validation sets
        let val_size = ((data.len() as f64) * validation_split.clamp(0.1, 0.5)).max(1.0) as usize;
        let split_idx = data.len().saturating_sub(val_size);
        let (train_data, val_data) = (&data[..split_idx], &data[split_idx..]);

        if train_data.is_empty() || val_data.is_empty() {
            return EarlyStoppingResult {
                history,
                stopped_early: false,
                final_epoch: 0,
                best_epoch: 0,
                best_val_loss: f64::INFINITY,
                epochs_without_improvement: 0,
            };
        }

        // Track best model state
        let mut best_val_loss = f64::INFINITY;
        let mut best_epoch = 0;
        let mut epochs_without_improvement = 0;
        let mut best_weights: Option<(DenseLayer, DenseLayer, DenseLayer)> = None;

        for epoch in 0..max_epochs {
            // Training pass
            let mut train_loss = 0.0;
            let mut train_correct = 0;

            for example in train_data {
                let loss = self.train_step_with_loss(example, learning_rate);
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

                // Save best weights if configured
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
                // Restore best weights if configured and available
                if early_stopping.restore_best_weights {
                    if let Some((h1, h2, bo)) = best_weights {
                        self.hidden1 = h1;
                        self.hidden2 = h2;
                        self.backend_output = bo;
                    }
                }

                return EarlyStoppingResult {
                    history,
                    stopped_early: true,
                    final_epoch: epoch,
                    best_epoch,
                    best_val_loss,
                    epochs_without_improvement,
                };
            }
        }

        // Training completed without early stopping
        // Optionally restore best weights
        if early_stopping.restore_best_weights {
            if let Some((h1, h2, bo)) = best_weights {
                self.hidden1 = h1;
                self.hidden2 = h2;
                self.backend_output = bo;
            }
        }

        EarlyStoppingResult {
            history,
            stopped_early: false,
            final_epoch: max_epochs.saturating_sub(1),
            best_epoch,
            best_val_loss,
            epochs_without_improvement,
        }
    }

    /// Perform k-fold cross-validation
    ///
    /// Splits data into k folds, trains on k-1 folds, tests on remaining fold,
    /// and averages results across all folds.
    ///
    /// # Arguments
    /// * `data` - All training examples
    /// * `k` - Number of folds (typically 5 or 10)
    /// * `learning_rate` - Learning rate for training
    /// * `epochs` - Epochs per fold
    ///
    /// # Returns
    /// Cross-validation results with mean and std dev of accuracy/loss
    pub fn cross_validate(
        data: &[TrainingExample],
        k: usize,
        learning_rate: f64,
        epochs: usize,
    ) -> CrossValidationResult {
        if data.is_empty() || k < 2 {
            return CrossValidationResult::default();
        }

        let k = k.min(data.len()); // Can't have more folds than samples
        let fold_size = data.len() / k;
        let mut fold_results: Vec<EvaluationResult> = Vec::new();

        for fold_idx in 0..k {
            // Create a fresh predictor for this fold
            let mut predictor = StrategyPredictor::new();

            // Split into train/test for this fold
            let test_start = fold_idx * fold_size;
            let test_end = if fold_idx == k - 1 {
                data.len()
            } else {
                test_start + fold_size
            };

            let test_fold: Vec<TrainingExample> = data[test_start..test_end].to_vec();
            let train_fold: Vec<TrainingExample> = data
                .iter()
                .enumerate()
                .filter(|(i, _)| *i < test_start || *i >= test_end)
                .map(|(_, ex)| ex.clone())
                .collect();

            // Train on training fold
            predictor.train(&train_fold, learning_rate, epochs);

            // Evaluate on test fold
            let eval = predictor.evaluate(&test_fold);
            fold_results.push(eval);
        }

        // Compute mean and std dev of metrics
        let accuracies: Vec<f64> = fold_results.iter().map(|r| r.accuracy).collect();
        let losses: Vec<f64> = fold_results.iter().map(|r| r.loss).collect();

        let mean_accuracy = accuracies.iter().sum::<f64>() / k as f64;
        let mean_loss = losses.iter().sum::<f64>() / k as f64;

        let std_accuracy = (accuracies
            .iter()
            .map(|&a| (a - mean_accuracy).powi(2))
            .sum::<f64>()
            / k as f64)
            .sqrt();

        let std_loss =
            (losses.iter().map(|&l| (l - mean_loss).powi(2)).sum::<f64>() / k as f64).sqrt();

        CrossValidationResult {
            k,
            mean_accuracy,
            std_accuracy,
            mean_loss,
            std_loss,
            fold_results,
        }
    }

    /// Single training step for one example with full backpropagation
    ///
    /// Performs forward pass, computes loss gradient, and backpropagates
    /// through all layers (hidden1, hidden2, backend_output).
    pub(crate) fn train_step(&mut self, example: &TrainingExample, learning_rate: f64) {
        self.train_step_full(example, learning_rate, 0.0);
    }

    /// Single training step with full backpropagation and optional weight decay
    ///
    /// # Arguments
    /// * `example` - Training example
    /// * `learning_rate` - Learning rate for gradient descent
    /// * `weight_decay` - L2 regularization coefficient (0.0 for no regularization)
    pub(crate) fn train_step_full(
        &mut self,
        example: &TrainingExample,
        learning_rate: f64,
        weight_decay: f64,
    ) {
        let features = &example.features;

        // Forward pass - keep pre-activation values for ReLU derivative
        let h1_pre = self.hidden1.forward(&features.features);
        let h1 = h1_pre.iter().map(|&x| x.max(0.0)).collect::<Vec<_>>();
        let h2_pre = self.hidden2.forward(&h1);
        let h2 = h2_pre.iter().map(|&x| x.max(0.0)).collect::<Vec<_>>();
        let backend_probs = self.backend_output.forward_softmax(&h2);

        // Compute backend loss gradient (cross-entropy derivative: softmax - one_hot)
        let target_idx = backend_to_idx(example.backend);
        let mut backend_grad = backend_probs.clone();
        if target_idx < backend_grad.len() {
            backend_grad[target_idx] -= 1.0;
        }

        // Backpropagate through backend output layer -> get gradient w.r.t. h2
        let h2_grad = self
            .backend_output
            .backward(&h2, &backend_grad, learning_rate, weight_decay);

        // Apply ReLU derivative to h2_grad
        let h2_grad_relu = relu_derivative(&h2_pre, &h2_grad);

        // Backpropagate through hidden2 -> get gradient w.r.t. h1
        let h1_grad = self
            .hidden2
            .backward(&h1, &h2_grad_relu, learning_rate, weight_decay);

        // Apply ReLU derivative to h1_grad
        let h1_grad_relu = relu_derivative(&h1_pre, &h1_grad);

        // Backpropagate through hidden1 (updates weights, input gradient not needed)
        let _ = self.hidden1.backward(
            &features.features,
            &h1_grad_relu,
            learning_rate,
            weight_decay,
        );
    }
}
