//! Hyperparameter configuration types for ML model training
//!
//! This module provides data structures for hyperparameter search, including
//! grid search and random search configurations.

use super::{EarlyStoppingConfig, LearningRateScheduler};
use serde::{Deserialize, Serialize};

/// Hyperparameter configuration for ML model training
///
/// Defines the full set of hyperparameters that can be tuned during training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hyperparameters {
    /// Learning rate for gradient descent
    pub learning_rate: f64,
    /// Number of training epochs
    pub epochs: usize,
    /// Fraction of data used for validation (0.0-0.5)
    pub validation_split: f64,
    /// Learning rate scheduler configuration
    pub lr_scheduler: LearningRateScheduler,
    /// Early stopping configuration (None to disable)
    pub early_stopping: Option<EarlyStoppingConfig>,
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            epochs: 100,
            validation_split: 0.2,
            lr_scheduler: LearningRateScheduler::Constant,
            early_stopping: Some(EarlyStoppingConfig::default()),
        }
    }
}

impl Hyperparameters {
    /// Create a new hyperparameter configuration
    pub fn new(learning_rate: f64, epochs: usize) -> Self {
        Self {
            learning_rate,
            epochs,
            ..Default::default()
        }
    }

    /// Set the validation split ratio
    pub fn with_validation_split(mut self, split: f64) -> Self {
        self.validation_split = split.clamp(0.0, 0.5);
        self
    }

    /// Set the learning rate scheduler
    pub fn with_scheduler(mut self, scheduler: LearningRateScheduler) -> Self {
        self.lr_scheduler = scheduler;
        self
    }

    /// Set early stopping configuration
    pub fn with_early_stopping(mut self, config: EarlyStoppingConfig) -> Self {
        self.early_stopping = Some(config);
        self
    }

    /// Disable early stopping
    pub fn without_early_stopping(mut self) -> Self {
        self.early_stopping = None;
        self
    }
}

/// Result of evaluating a single hyperparameter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterResult {
    /// The hyperparameters used
    pub hyperparameters: Hyperparameters,
    /// Final validation loss (lower is better)
    pub val_loss: f64,
    /// Final validation accuracy (higher is better)
    pub val_accuracy: f64,
    /// Final training loss
    pub train_loss: f64,
    /// Final training accuracy
    pub train_accuracy: f64,
    /// Number of epochs actually trained (may be less if early stopping triggered)
    pub epochs_trained: usize,
    /// Whether early stopping was triggered
    pub stopped_early: bool,
}

impl HyperparameterResult {
    /// Check if this result is better than another (lower val_loss or higher accuracy on tie)
    pub fn is_better_than(&self, other: &HyperparameterResult) -> bool {
        if (self.val_loss - other.val_loss).abs() < 1e-9 {
            self.val_accuracy > other.val_accuracy
        } else {
            self.val_loss < other.val_loss
        }
    }
}

/// Result of a hyperparameter search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterSearchResult {
    /// All evaluated configurations and their results
    pub results: Vec<HyperparameterResult>,
    /// Index of the best result in `results`
    pub best_idx: usize,
    /// Total number of configurations evaluated
    pub total_evaluated: usize,
    /// Search method used
    pub search_method: String,
}

impl HyperparameterSearchResult {
    /// Get the best hyperparameter configuration
    pub fn best_hyperparameters(&self) -> &Hyperparameters {
        &self.results[self.best_idx].hyperparameters
    }

    /// Get the best result
    pub fn best_result(&self) -> &HyperparameterResult {
        &self.results[self.best_idx]
    }

    /// Get the best validation loss achieved
    pub fn best_val_loss(&self) -> f64 {
        self.results[self.best_idx].val_loss
    }

    /// Get the best validation accuracy achieved
    pub fn best_val_accuracy(&self) -> f64 {
        self.results[self.best_idx].val_accuracy
    }

    /// Get results sorted by validation loss (ascending)
    pub fn sorted_by_loss(&self) -> Vec<&HyperparameterResult> {
        let mut sorted: Vec<_> = self.results.iter().collect();
        sorted.sort_by(|a, b| a.val_loss.partial_cmp(&b.val_loss).unwrap());
        sorted
    }
}

/// Hyperparameter search space for grid search
#[derive(Debug, Clone)]
pub struct GridSearchSpace {
    /// Learning rates to try
    pub learning_rates: Vec<f64>,
    /// Epoch counts to try
    pub epochs: Vec<usize>,
    /// Validation splits to try
    pub validation_splits: Vec<f64>,
    /// Learning rate schedulers to try
    pub schedulers: Vec<LearningRateScheduler>,
    /// Early stopping patience values to try (None = no early stopping)
    pub patience_values: Vec<Option<usize>>,
}

impl Default for GridSearchSpace {
    fn default() -> Self {
        Self {
            learning_rates: vec![0.001, 0.01, 0.05, 0.1],
            epochs: vec![50, 100],
            validation_splits: vec![0.2],
            schedulers: vec![LearningRateScheduler::Constant],
            patience_values: vec![Some(5)],
        }
    }
}

impl GridSearchSpace {
    /// Create a new search space with custom learning rates
    pub fn new(learning_rates: Vec<f64>) -> Self {
        Self {
            learning_rates,
            ..Default::default()
        }
    }

    /// Set epoch values to search
    pub fn with_epochs(mut self, epochs: Vec<usize>) -> Self {
        self.epochs = epochs;
        self
    }

    /// Set validation splits to search
    pub fn with_validation_splits(mut self, splits: Vec<f64>) -> Self {
        self.validation_splits = splits;
        self
    }

    /// Set schedulers to search
    pub fn with_schedulers(mut self, schedulers: Vec<LearningRateScheduler>) -> Self {
        self.schedulers = schedulers;
        self
    }

    /// Set patience values to search (use None for no early stopping)
    pub fn with_patience_values(mut self, values: Vec<Option<usize>>) -> Self {
        self.patience_values = values;
        self
    }

    /// Convenience method to disable early stopping entirely
    pub fn without_early_stopping(mut self) -> Self {
        self.patience_values = vec![None];
        self
    }

    /// Calculate total number of configurations to evaluate
    pub fn total_configurations(&self) -> usize {
        self.learning_rates.len()
            * self.epochs.len()
            * self.validation_splits.len()
            * self.schedulers.len()
            * self.patience_values.len()
    }

    /// Generate all hyperparameter configurations
    pub fn generate_configs(&self) -> Vec<Hyperparameters> {
        let mut configs = Vec::with_capacity(self.total_configurations());

        for &lr in &self.learning_rates {
            for &epochs in &self.epochs {
                for &val_split in &self.validation_splits {
                    for scheduler in &self.schedulers {
                        for &patience in &self.patience_values {
                            let early_stopping =
                                patience.map(|p| EarlyStoppingConfig::new(p, 0.001));
                            configs.push(Hyperparameters {
                                learning_rate: lr,
                                epochs,
                                validation_split: val_split,
                                lr_scheduler: scheduler.clone(),
                                early_stopping,
                            });
                        }
                    }
                }
            }
        }

        configs
    }
}

/// Random search configuration
#[derive(Debug, Clone)]
pub struct RandomSearchConfig {
    /// Number of random configurations to try
    pub n_iterations: usize,
    /// Learning rate range (min, max) - sampled log-uniformly
    pub lr_range: (f64, f64),
    /// Epoch range (min, max)
    pub epoch_range: (usize, usize),
    /// Validation split range (min, max)
    pub val_split_range: (f64, f64),
    /// Schedulers to sample from
    pub schedulers: Vec<LearningRateScheduler>,
    /// Patience range for early stopping (min, max), None means sometimes disable
    pub patience_range: (usize, usize),
    /// Probability of using early stopping (0.0-1.0)
    pub early_stopping_prob: f64,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for RandomSearchConfig {
    fn default() -> Self {
        Self {
            n_iterations: 20,
            lr_range: (0.0001, 0.5),
            epoch_range: (20, 200),
            val_split_range: (0.1, 0.3),
            schedulers: vec![
                LearningRateScheduler::Constant,
                LearningRateScheduler::step(10, 0.5),
                LearningRateScheduler::exponential(0.95),
                LearningRateScheduler::cosine(0.0001, 100),
            ],
            patience_range: (3, 10),
            early_stopping_prob: 0.8,
            seed: 42,
        }
    }
}

impl RandomSearchConfig {
    /// Create a new random search configuration
    pub fn new(n_iterations: usize) -> Self {
        Self {
            n_iterations,
            ..Default::default()
        }
    }

    /// Set the learning rate range (sampled log-uniformly)
    pub fn with_lr_range(mut self, min: f64, max: f64) -> Self {
        self.lr_range = (min, max);
        self
    }

    /// Set the epoch range
    pub fn with_epoch_range(mut self, min: usize, max: usize) -> Self {
        self.epoch_range = (min, max);
        self
    }

    /// Set the validation split range
    pub fn with_val_split_range(mut self, min: f64, max: f64) -> Self {
        self.val_split_range = (min, max);
        self
    }

    /// Set the schedulers to sample from
    pub fn with_schedulers(mut self, schedulers: Vec<LearningRateScheduler>) -> Self {
        self.schedulers = schedulers;
        self
    }

    /// Set the random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Generate random hyperparameter configurations using a simple PRNG
    pub fn generate_configs(&self) -> Vec<Hyperparameters> {
        let mut configs = Vec::with_capacity(self.n_iterations);
        let mut rng = SimpleRng::new(self.seed);

        for _ in 0..self.n_iterations {
            // Log-uniform sampling for learning rate
            let log_min = self.lr_range.0.ln();
            let log_max = self.lr_range.1.ln();
            let lr = (log_min + rng.next_f64() * (log_max - log_min)).exp();

            // Uniform sampling for epochs
            let epochs = self.epoch_range.0
                + (rng.next_f64() * (self.epoch_range.1 - self.epoch_range.0) as f64) as usize;

            // Uniform sampling for validation split
            let val_split = self.val_split_range.0
                + rng.next_f64() * (self.val_split_range.1 - self.val_split_range.0);

            // Uniform sampling for scheduler
            let scheduler_idx =
                (rng.next_f64() * self.schedulers.len() as f64) as usize % self.schedulers.len();
            let scheduler = self.schedulers[scheduler_idx].clone();

            // Early stopping with probability
            let early_stopping = if rng.next_f64() < self.early_stopping_prob {
                let patience = self.patience_range.0
                    + (rng.next_f64() * (self.patience_range.1 - self.patience_range.0) as f64)
                        as usize;
                Some(EarlyStoppingConfig::new(patience, 0.001))
            } else {
                None
            };

            configs.push(Hyperparameters {
                learning_rate: lr,
                epochs,
                validation_split: val_split,
                lr_scheduler: scheduler,
                early_stopping,
            });
        }

        configs
    }
}

/// Simple pseudo-random number generator (xorshift64)
pub(super) struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub(super) fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    pub(super) fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    pub(super) fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }
}
