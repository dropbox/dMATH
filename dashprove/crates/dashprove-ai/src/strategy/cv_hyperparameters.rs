//! Cross-validation in hyperparameter search
//!
//! This module provides cross-validated hyperparameter search result types.
//! Cross-validation provides more robust estimates of model performance by
//! evaluating on multiple train/test splits.

use serde::{Deserialize, Serialize};

use super::hyperparameters::{HyperparameterResult, Hyperparameters};

/// Result of hyperparameter search with cross-validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVHyperparameterResult {
    /// The hyperparameters evaluated
    pub hyperparameters: Hyperparameters,
    /// Mean validation loss across folds
    pub mean_val_loss: f64,
    /// Standard deviation of validation loss across folds
    pub std_val_loss: f64,
    /// Mean validation accuracy across folds
    pub mean_val_accuracy: f64,
    /// Standard deviation of validation accuracy across folds
    pub std_val_accuracy: f64,
    /// Mean training loss across folds
    pub mean_train_loss: f64,
    /// Mean training accuracy across folds
    pub mean_train_accuracy: f64,
    /// Number of folds used
    pub k_folds: usize,
    /// Per-fold results
    pub fold_results: Vec<HyperparameterResult>,
}

/// Result of cross-validated hyperparameter search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVHyperparameterSearchResult {
    /// Total configurations evaluated
    pub total_evaluated: usize,
    /// All results sorted by mean validation loss
    pub results: Vec<CVHyperparameterResult>,
    /// Index of best configuration
    pub best_idx: usize,
    /// Search method used (grid, random, bayesian)
    pub search_method: String,
    /// Number of CV folds used
    pub k_folds: usize,
}

impl CVHyperparameterSearchResult {
    /// Get the best hyperparameter configuration
    pub fn best_hyperparameters(&self) -> &Hyperparameters {
        &self.results[self.best_idx].hyperparameters
    }

    /// Get the best result
    pub fn best_result(&self) -> &CVHyperparameterResult {
        &self.results[self.best_idx]
    }

    /// Get results sorted by mean validation loss (ascending)
    pub fn sorted_by_loss(&self) -> Vec<&CVHyperparameterResult> {
        let mut sorted: Vec<_> = self.results.iter().collect();
        sorted.sort_by(|a, b| a.mean_val_loss.partial_cmp(&b.mean_val_loss).unwrap());
        sorted
    }

    /// Get the top N results by mean validation loss
    pub fn top_n(&self, n: usize) -> Vec<&CVHyperparameterResult> {
        self.sorted_by_loss().into_iter().take(n).collect()
    }
}
