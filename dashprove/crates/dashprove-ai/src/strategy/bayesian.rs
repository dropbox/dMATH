//! Bayesian optimization for hyperparameter search
//!
//! This module provides Bayesian optimization using a Gaussian Process surrogate
//! model to efficiently search the hyperparameter space. More sample-efficient
//! than random search for continuous hyperparameters.

use serde::{Deserialize, Serialize};

use super::hyperparameters::{HyperparameterResult, Hyperparameters, SimpleRng};
use super::{EarlyStoppingConfig, LearningRateScheduler};

/// Bayesian optimization for hyperparameter search
///
/// Uses a Gaussian Process surrogate model to efficiently search the
/// hyperparameter space. More sample-efficient than random search for
/// continuous hyperparameters.
#[derive(Debug, Clone)]
pub struct BayesianOptimizer {
    /// Learning rate bounds (min, max)
    pub lr_bounds: (f64, f64),
    /// Epoch bounds (min, max)
    pub epoch_bounds: (usize, usize),
    /// Validation split bounds (min, max)
    pub val_split_bounds: (f64, f64),
    /// Number of initial random samples before using GP
    pub n_initial_samples: usize,
    /// Total iterations (including initial samples)
    pub n_iterations: usize,
    /// Exploration-exploitation trade-off (higher = more exploration)
    pub kappa: f64,
    /// Random seed
    pub seed: u64,
    /// Optional schedulers to sample from
    pub schedulers: Vec<LearningRateScheduler>,
    /// Patience values to try (for early stopping)
    pub patience_values: Vec<Option<usize>>,
}

impl Default for BayesianOptimizer {
    fn default() -> Self {
        Self {
            lr_bounds: (0.0001, 0.5),
            epoch_bounds: (10, 200),
            val_split_bounds: (0.1, 0.3),
            n_initial_samples: 5,
            n_iterations: 25,
            kappa: 2.576, // 99% confidence bound
            seed: 42,
            schedulers: vec![LearningRateScheduler::Constant],
            patience_values: vec![None, Some(5)],
        }
    }
}

impl BayesianOptimizer {
    /// Create a new Bayesian optimizer
    pub fn new(n_iterations: usize) -> Self {
        Self {
            n_iterations,
            ..Default::default()
        }
    }

    /// Set learning rate bounds
    pub fn with_lr_bounds(mut self, min: f64, max: f64) -> Self {
        self.lr_bounds = (min, max);
        self
    }

    /// Set epoch bounds
    pub fn with_epoch_bounds(mut self, min: usize, max: usize) -> Self {
        self.epoch_bounds = (min, max);
        self
    }

    /// Set number of initial random samples
    pub fn with_initial_samples(mut self, n: usize) -> Self {
        self.n_initial_samples = n;
        self
    }

    /// Set exploration-exploitation trade-off parameter
    pub fn with_kappa(mut self, kappa: f64) -> Self {
        self.kappa = kappa;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set validation split bounds
    pub fn with_val_split_bounds(mut self, min: f64, max: f64) -> Self {
        self.val_split_bounds = (min, max);
        self
    }

    /// Set schedulers to try
    pub fn with_schedulers(mut self, schedulers: Vec<LearningRateScheduler>) -> Self {
        self.schedulers = schedulers;
        self
    }

    /// Set patience values for early stopping
    pub fn with_patience_values(mut self, patience: Vec<Option<usize>>) -> Self {
        self.patience_values = patience;
        self
    }

    /// Convert hyperparameters to a normalized feature vector [0, 1]
    pub(super) fn hp_to_features(&self, hp: &Hyperparameters) -> Vec<f64> {
        let lr_norm = (hp.learning_rate.ln() - self.lr_bounds.0.ln())
            / (self.lr_bounds.1.ln() - self.lr_bounds.0.ln());
        let epochs_norm = (hp.epochs - self.epoch_bounds.0) as f64
            / (self.epoch_bounds.1 - self.epoch_bounds.0) as f64;
        let val_split_norm = (hp.validation_split - self.val_split_bounds.0)
            / (self.val_split_bounds.1 - self.val_split_bounds.0);

        vec![
            lr_norm.clamp(0.0, 1.0),
            epochs_norm.clamp(0.0, 1.0),
            val_split_norm.clamp(0.0, 1.0),
        ]
    }

    /// Convert normalized features to hyperparameters
    pub(super) fn features_to_hp(
        &self,
        features: &[f64],
        scheduler: LearningRateScheduler,
        patience: Option<usize>,
    ) -> Hyperparameters {
        let lr = ((features[0].clamp(0.0, 1.0) * (self.lr_bounds.1.ln() - self.lr_bounds.0.ln()))
            + self.lr_bounds.0.ln())
        .exp();
        let epochs = (features[1].clamp(0.0, 1.0)
            * (self.epoch_bounds.1 - self.epoch_bounds.0) as f64) as usize
            + self.epoch_bounds.0;
        let val_split = features[2].clamp(0.0, 1.0)
            * (self.val_split_bounds.1 - self.val_split_bounds.0)
            + self.val_split_bounds.0;

        let early_stopping = patience.map(|p| EarlyStoppingConfig::new(p, 0.001));

        Hyperparameters {
            learning_rate: lr,
            epochs,
            validation_split: val_split,
            lr_scheduler: scheduler,
            early_stopping,
        }
    }

    /// Generate a random hyperparameter configuration
    pub(super) fn random_hp(&self, rng: &mut SimpleRng) -> Hyperparameters {
        // Log-uniform for learning rate
        let log_min = self.lr_bounds.0.ln();
        let log_max = self.lr_bounds.1.ln();
        let lr = (log_min + rng.next_f64() * (log_max - log_min)).exp();

        // Uniform for epochs
        let epochs = self.epoch_bounds.0
            + (rng.next_f64() * (self.epoch_bounds.1 - self.epoch_bounds.0) as f64) as usize;

        // Uniform for validation split
        let val_split = self.val_split_bounds.0
            + rng.next_f64() * (self.val_split_bounds.1 - self.val_split_bounds.0);

        // Random scheduler
        let scheduler_idx =
            (rng.next_f64() * self.schedulers.len() as f64) as usize % self.schedulers.len();
        let scheduler = self.schedulers[scheduler_idx].clone();

        // Random patience
        let patience_idx = (rng.next_f64() * self.patience_values.len() as f64) as usize
            % self.patience_values.len();
        let patience = self.patience_values[patience_idx];
        let early_stopping = patience.map(|p| EarlyStoppingConfig::new(p, 0.001));

        Hyperparameters {
            learning_rate: lr,
            epochs,
            validation_split: val_split,
            lr_scheduler: scheduler,
            early_stopping,
        }
    }

    /// Compute squared exponential (RBF) kernel between two points
    pub(super) fn rbf_kernel(x1: &[f64], x2: &[f64], length_scale: f64) -> f64 {
        let sq_dist: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        (-sq_dist / (2.0 * length_scale * length_scale)).exp()
    }

    /// Compute the kernel matrix
    pub(super) fn compute_kernel_matrix(
        points: &[Vec<f64>],
        length_scale: f64,
        noise: f64,
    ) -> Vec<Vec<f64>> {
        let n = points.len();
        let mut k = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                k[i][j] = Self::rbf_kernel(&points[i], &points[j], length_scale);
                if i == j {
                    k[i][j] += noise; // Add noise to diagonal for numerical stability
                }
            }
        }
        k
    }

    /// Simple Cholesky decomposition for positive definite matrix
    pub(super) fn cholesky(matrix: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
        let n = matrix.len();
        let mut l = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..=i {
                let sum: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();
                if i == j {
                    let val = matrix[i][i] - sum;
                    if val <= 0.0 {
                        return None; // Not positive definite
                    }
                    l[i][j] = val.sqrt();
                } else {
                    l[i][j] = (matrix[i][j] - sum) / l[j][j];
                }
            }
        }
        Some(l)
    }

    /// Solve L * x = b for x using forward substitution
    pub(super) fn solve_lower(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
        let n = b.len();
        let mut x = vec![0.0; n];
        for i in 0..n {
            let mut sum = b[i];
            for j in 0..i {
                sum -= l[i][j] * x[j];
            }
            x[i] = sum / l[i][i];
        }
        x
    }

    /// Solve L^T * x = b for x using backward substitution
    pub(super) fn solve_upper(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
        let n = b.len();
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = b[i];
            for j in (i + 1)..n {
                sum -= l[j][i] * x[j];
            }
            x[i] = sum / l[i][i];
        }
        x
    }

    /// Predict mean and variance at a new point using GP
    pub(super) fn gp_predict(
        x_new: &[f64],
        x_train: &[Vec<f64>],
        y_train: &[f64],
        l: &[Vec<f64>],
        length_scale: f64,
    ) -> (f64, f64) {
        if x_train.is_empty() {
            return (0.0, 1.0);
        }

        // Compute k_star (kernel between new point and training points)
        let k_star: Vec<f64> = x_train
            .iter()
            .map(|x| Self::rbf_kernel(x_new, x, length_scale))
            .collect();

        // alpha = K^{-1} y = L^{-T} L^{-1} y
        let temp = Self::solve_lower(l, y_train);
        let alpha = Self::solve_upper(l, &temp);

        // Mean: k_star^T * alpha
        let mean: f64 = k_star.iter().zip(alpha.iter()).map(|(k, a)| k * a).sum();

        // Variance: k(x*, x*) - k_star^T * K^{-1} * k_star
        let v = Self::solve_lower(l, &k_star);
        let variance = 1.0 - v.iter().map(|x| x * x).sum::<f64>();

        (mean, variance.max(1e-6))
    }

    /// Upper Confidence Bound acquisition function
    pub(super) fn ucb(mean: f64, variance: f64, kappa: f64) -> f64 {
        // We're minimizing loss, so we want lower confidence bound
        mean - kappa * variance.sqrt()
    }
}

/// Result of Bayesian optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BayesianOptimizationResult {
    /// All evaluated configurations and their results
    pub evaluations: Vec<HyperparameterResult>,
    /// Index of best configuration
    pub best_idx: usize,
    /// History of acquisition function values (for debugging)
    pub acquisition_history: Vec<f64>,
    /// Number of initial random samples used
    pub n_initial_samples: usize,
    /// Total iterations
    pub total_iterations: usize,
}

impl BayesianOptimizationResult {
    /// Get the best hyperparameters found
    pub fn best_hyperparameters(&self) -> &Hyperparameters {
        &self.evaluations[self.best_idx].hyperparameters
    }

    /// Get the best result
    pub fn best_result(&self) -> &HyperparameterResult {
        &self.evaluations[self.best_idx]
    }

    /// Get results sorted by validation loss (ascending)
    pub fn sorted_by_loss(&self) -> Vec<&HyperparameterResult> {
        let mut sorted: Vec<_> = self.evaluations.iter().collect();
        sorted.sort_by(|a, b| a.val_loss.partial_cmp(&b.val_loss).unwrap());
        sorted
    }
}
