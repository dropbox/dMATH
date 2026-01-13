//! Hyperparameter search functionality for StrategyPredictor
//!
//! This module provides grid search, random search, cross-validated search,
//! and Bayesian optimization for finding optimal hyperparameters.

use super::{
    BayesianOptimizationResult, BayesianOptimizer, CVHyperparameterResult,
    CVHyperparameterSearchResult, GridSearchSpace, HyperparameterResult,
    HyperparameterSearchResult, Hyperparameters, LearningRateScheduler, RandomSearchConfig,
    SimpleRng, StrategyPredictor, TrainingExample,
};

// ============================================================================
// Hyperparameter Search
// ============================================================================

/// Hyperparameter search functionality for the StrategyPredictor
impl StrategyPredictor {
    /// Perform grid search over hyperparameter space
    ///
    /// Evaluates all combinations of hyperparameters in the search space
    /// and returns the best configuration along with all results.
    ///
    /// # Arguments
    /// * `data` - Training data
    /// * `search_space` - Grid of hyperparameter values to search
    ///
    /// # Returns
    /// `HyperparameterSearchResult` with all evaluated configurations
    pub fn grid_search(
        data: &[TrainingExample],
        search_space: &GridSearchSpace,
    ) -> HyperparameterSearchResult {
        let configs = search_space.generate_configs();
        Self::search_configs(data, configs, "grid")
    }

    /// Perform random search over hyperparameter space
    ///
    /// Samples random hyperparameter configurations and evaluates them.
    /// Often more efficient than grid search for high-dimensional spaces.
    ///
    /// # Arguments
    /// * `data` - Training data
    /// * `config` - Random search configuration
    ///
    /// # Returns
    /// `HyperparameterSearchResult` with all evaluated configurations
    pub fn random_search(
        data: &[TrainingExample],
        config: &RandomSearchConfig,
    ) -> HyperparameterSearchResult {
        let configs = config.generate_configs();
        Self::search_configs(data, configs, "random")
    }

    /// Internal: evaluate a list of hyperparameter configurations
    fn search_configs(
        data: &[TrainingExample],
        configs: Vec<Hyperparameters>,
        method: &str,
    ) -> HyperparameterSearchResult {
        let mut results = Vec::with_capacity(configs.len());
        let mut best_idx = 0;
        let mut best_val_loss = f64::INFINITY;

        for (i, hp) in configs.into_iter().enumerate() {
            let result = Self::evaluate_hyperparameters(data, hp);

            if result.val_loss < best_val_loss {
                best_val_loss = result.val_loss;
                best_idx = i;
            }

            results.push(result);
        }

        HyperparameterSearchResult {
            total_evaluated: results.len(),
            results,
            best_idx,
            search_method: method.to_string(),
        }
    }

    /// Evaluate a single hyperparameter configuration
    ///
    /// Trains a fresh model with the given hyperparameters and returns
    /// the validation metrics.
    pub fn evaluate_hyperparameters(
        data: &[TrainingExample],
        hp: Hyperparameters,
    ) -> HyperparameterResult {
        let mut predictor = StrategyPredictor::new();

        // Determine which training method to use based on configuration
        let uses_scheduler = !matches!(hp.lr_scheduler, LearningRateScheduler::Constant);

        let (train_loss, train_acc, val_loss, val_acc, epochs_trained, stopped_early) =
            if uses_scheduler {
                if let Some(ref es_config) = hp.early_stopping {
                    // Scheduler + early stopping
                    let result = predictor.train_with_scheduler_and_early_stopping(
                        data,
                        hp.learning_rate,
                        hp.epochs,
                        hp.validation_split,
                        hp.lr_scheduler.clone(),
                        es_config.clone(),
                    );
                    let es_info = result.early_stopping_info.as_ref();
                    let epochs = result.history.epochs.len();
                    let (tl, ta) = result
                        .history
                        .epochs
                        .last()
                        .map(|e| (e.train_loss, e.train_accuracy))
                        .unwrap_or((f64::INFINITY, 0.0));
                    let (vl, va) = result
                        .history
                        .epochs
                        .last()
                        .and_then(|e| e.val_loss.zip(e.val_accuracy))
                        .unwrap_or((f64::INFINITY, 0.0));
                    (
                        tl,
                        ta,
                        vl,
                        va,
                        epochs,
                        es_info.map(|e| e.stopped_early).unwrap_or(false),
                    )
                } else {
                    // Scheduler only
                    let result = predictor.train_with_scheduler(
                        data,
                        hp.learning_rate,
                        hp.epochs,
                        hp.validation_split,
                        hp.lr_scheduler.clone(),
                    );
                    let epochs = result.history.epochs.len();
                    let (tl, ta) = result
                        .history
                        .epochs
                        .last()
                        .map(|e| (e.train_loss, e.train_accuracy))
                        .unwrap_or((f64::INFINITY, 0.0));
                    let (vl, va) = result
                        .history
                        .epochs
                        .last()
                        .and_then(|e| e.val_loss.zip(e.val_accuracy))
                        .unwrap_or((f64::INFINITY, 0.0));
                    (tl, ta, vl, va, epochs, false)
                }
            } else if let Some(ref es_config) = hp.early_stopping {
                // Early stopping only
                let result = predictor.train_with_early_stopping(
                    data,
                    hp.learning_rate,
                    hp.epochs,
                    hp.validation_split,
                    es_config.clone(),
                );
                let epochs = result.history.epochs.len();
                let (tl, ta) = result
                    .history
                    .epochs
                    .last()
                    .map(|e| (e.train_loss, e.train_accuracy))
                    .unwrap_or((f64::INFINITY, 0.0));
                let (vl, va) = result
                    .history
                    .epochs
                    .last()
                    .and_then(|e| e.val_loss.zip(e.val_accuracy))
                    .unwrap_or((f64::INFINITY, 0.0));
                (tl, ta, vl, va, epochs, result.stopped_early)
            } else {
                // Basic training with metrics
                let history = predictor.train_with_metrics(
                    data,
                    hp.learning_rate,
                    hp.epochs,
                    hp.validation_split,
                );
                let epochs = history.epochs.len();
                let (tl, ta) = history
                    .epochs
                    .last()
                    .map(|e| (e.train_loss, e.train_accuracy))
                    .unwrap_or((f64::INFINITY, 0.0));
                let (vl, va) = history
                    .epochs
                    .last()
                    .and_then(|e| e.val_loss.zip(e.val_accuracy))
                    .unwrap_or((f64::INFINITY, 0.0));
                (tl, ta, vl, va, epochs, false)
            };

        HyperparameterResult {
            hyperparameters: hp,
            val_loss,
            val_accuracy: val_acc,
            train_loss,
            train_accuracy: train_acc,
            epochs_trained,
            stopped_early,
        }
    }

    /// Train with the best hyperparameters from a search result
    ///
    /// Convenience method to train a model using the best configuration
    /// found during hyperparameter search.
    pub fn train_with_best_hyperparameters(
        &mut self,
        data: &[TrainingExample],
        search_result: &HyperparameterSearchResult,
    ) -> HyperparameterResult {
        let best_hp = search_result.best_hyperparameters().clone();
        Self::evaluate_hyperparameters(data, best_hp)
    }
}

// ============================================================================
// Cross-validation in Hyperparameter Search
// ============================================================================

/// Cross-validated hyperparameter search for StrategyPredictor
impl StrategyPredictor {
    /// Perform grid search with cross-validation
    ///
    /// Evaluates all combinations using k-fold cross-validation for more
    /// robust hyperparameter selection. This is slower but provides
    /// better estimates of model performance.
    ///
    /// # Arguments
    /// * `data` - Training data
    /// * `search_space` - Grid of hyperparameter values to search
    /// * `k` - Number of CV folds
    ///
    /// # Returns
    /// `CVHyperparameterSearchResult` with cross-validated results
    pub fn grid_search_with_cv(
        data: &[TrainingExample],
        search_space: &GridSearchSpace,
        k: usize,
    ) -> CVHyperparameterSearchResult {
        let configs = search_space.generate_configs();
        Self::search_configs_with_cv(data, configs, k, "grid")
    }

    /// Perform random search with cross-validation
    ///
    /// Samples random hyperparameter configurations and evaluates them using
    /// k-fold cross-validation. Combines efficiency of random search with
    /// robustness of cross-validation.
    ///
    /// # Arguments
    /// * `data` - Training data
    /// * `config` - Random search configuration
    /// * `k` - Number of CV folds
    ///
    /// # Returns
    /// `CVHyperparameterSearchResult` with cross-validated results
    pub fn random_search_with_cv(
        data: &[TrainingExample],
        config: &RandomSearchConfig,
        k: usize,
    ) -> CVHyperparameterSearchResult {
        let configs = config.generate_configs();
        Self::search_configs_with_cv(data, configs, k, "random")
    }

    /// Internal: evaluate configurations with cross-validation
    fn search_configs_with_cv(
        data: &[TrainingExample],
        configs: Vec<Hyperparameters>,
        k: usize,
        method: &str,
    ) -> CVHyperparameterSearchResult {
        let k = k.max(2).min(data.len()); // At least 2 folds, at most data.len() folds
        let mut results = Vec::with_capacity(configs.len());
        let mut best_idx = 0;
        let mut best_mean_val_loss = f64::INFINITY;

        for (i, hp) in configs.into_iter().enumerate() {
            let cv_result = Self::evaluate_hyperparameters_cv(data, hp, k);

            if cv_result.mean_val_loss < best_mean_val_loss {
                best_mean_val_loss = cv_result.mean_val_loss;
                best_idx = i;
            }

            results.push(cv_result);
        }

        CVHyperparameterSearchResult {
            total_evaluated: results.len(),
            results,
            best_idx,
            search_method: method.to_string(),
            k_folds: k,
        }
    }

    /// Evaluate a single hyperparameter configuration with cross-validation
    pub(super) fn evaluate_hyperparameters_cv(
        data: &[TrainingExample],
        hp: Hyperparameters,
        k: usize,
    ) -> CVHyperparameterResult {
        let fold_size = data.len() / k;
        let mut fold_results: Vec<HyperparameterResult> = Vec::with_capacity(k);

        for fold_idx in 0..k {
            // Split into train/test for this fold
            let test_start = fold_idx * fold_size;
            let test_end = if fold_idx == k - 1 {
                data.len()
            } else {
                test_start + fold_size
            };

            let train_fold: Vec<TrainingExample> = data
                .iter()
                .enumerate()
                .filter(|(i, _)| *i < test_start || *i >= test_end)
                .map(|(_, ex)| ex.clone())
                .collect();

            // Evaluate on this fold (internally uses validation split from train_fold)
            let result = Self::evaluate_hyperparameters(&train_fold, hp.clone());
            fold_results.push(result);
        }

        // Compute mean and std of metrics across folds
        let val_losses: Vec<f64> = fold_results.iter().map(|r| r.val_loss).collect();
        let val_accs: Vec<f64> = fold_results.iter().map(|r| r.val_accuracy).collect();
        let train_losses: Vec<f64> = fold_results.iter().map(|r| r.train_loss).collect();
        let train_accs: Vec<f64> = fold_results.iter().map(|r| r.train_accuracy).collect();

        let mean_val_loss = val_losses.iter().sum::<f64>() / k as f64;
        let mean_val_accuracy = val_accs.iter().sum::<f64>() / k as f64;
        let mean_train_loss = train_losses.iter().sum::<f64>() / k as f64;
        let mean_train_accuracy = train_accs.iter().sum::<f64>() / k as f64;

        let std_val_loss = (val_losses
            .iter()
            .map(|&l| (l - mean_val_loss).powi(2))
            .sum::<f64>()
            / k as f64)
            .sqrt();

        let std_val_accuracy = (val_accs
            .iter()
            .map(|&a| (a - mean_val_accuracy).powi(2))
            .sum::<f64>()
            / k as f64)
            .sqrt();

        CVHyperparameterResult {
            hyperparameters: hp,
            mean_val_loss,
            std_val_loss,
            mean_val_accuracy,
            std_val_accuracy,
            mean_train_loss,
            mean_train_accuracy,
            k_folds: k,
            fold_results,
        }
    }
}

// ============================================================================
// Bayesian Optimization
// ============================================================================

/// Bayesian optimization for StrategyPredictor
impl StrategyPredictor {
    /// Perform Bayesian optimization for hyperparameter search
    ///
    /// Uses a Gaussian Process surrogate model to efficiently search the
    /// hyperparameter space. After initial random samples, uses the
    /// Upper Confidence Bound (UCB) acquisition function to select the
    /// next point to evaluate.
    ///
    /// # Arguments
    /// * `data` - Training data
    /// * `optimizer` - Bayesian optimization configuration
    ///
    /// # Returns
    /// `BayesianOptimizationResult` with all evaluations and best configuration
    pub fn bayesian_optimize(
        data: &[TrainingExample],
        optimizer: &BayesianOptimizer,
    ) -> BayesianOptimizationResult {
        let mut rng = SimpleRng::new(optimizer.seed);
        let mut evaluations: Vec<HyperparameterResult> = Vec::new();
        let mut x_train: Vec<Vec<f64>> = Vec::new();
        let mut y_train: Vec<f64> = Vec::new();
        let mut acquisition_history: Vec<f64> = Vec::new();

        let length_scale = 0.3; // GP length scale
        let noise = 1e-6; // Observation noise

        // Initial random sampling
        for _ in 0..optimizer.n_initial_samples.min(optimizer.n_iterations) {
            let hp = optimizer.random_hp(&mut rng);
            let features = optimizer.hp_to_features(&hp);
            let result = Self::evaluate_hyperparameters(data, hp);

            x_train.push(features);
            y_train.push(result.val_loss);
            evaluations.push(result);
        }

        // Bayesian optimization iterations
        for _iter in optimizer.n_initial_samples..optimizer.n_iterations {
            // Compute kernel matrix and its Cholesky decomposition
            let k_matrix = BayesianOptimizer::compute_kernel_matrix(&x_train, length_scale, noise);
            let l = match BayesianOptimizer::cholesky(&k_matrix) {
                Some(l) => l,
                None => {
                    // If Cholesky fails, fall back to random sampling
                    let hp = optimizer.random_hp(&mut rng);
                    let features = optimizer.hp_to_features(&hp);
                    let result = Self::evaluate_hyperparameters(data, hp);
                    x_train.push(features);
                    y_train.push(result.val_loss);
                    evaluations.push(result);
                    acquisition_history.push(f64::NAN);
                    continue;
                }
            };

            // Find the point that maximizes acquisition function
            // Use a simple grid search over the feature space
            let mut best_acq = f64::INFINITY;
            let mut best_features = vec![0.5, 0.5, 0.5];

            // Grid search in feature space
            for lr_idx in 0..10 {
                for epoch_idx in 0..10 {
                    for val_idx in 0..5 {
                        let features = vec![
                            (lr_idx as f64 + 0.5) / 10.0,
                            (epoch_idx as f64 + 0.5) / 10.0,
                            (val_idx as f64 + 0.5) / 5.0,
                        ];

                        let (mean, variance) = BayesianOptimizer::gp_predict(
                            &features,
                            &x_train,
                            &y_train,
                            &l,
                            length_scale,
                        );

                        let acq = BayesianOptimizer::ucb(mean, variance, optimizer.kappa);

                        if acq < best_acq {
                            best_acq = acq;
                            best_features = features;
                        }
                    }
                }
            }

            acquisition_history.push(best_acq);

            // Select random scheduler and patience for this iteration
            let scheduler_idx = (rng.next_f64() * optimizer.schedulers.len() as f64) as usize
                % optimizer.schedulers.len();
            let scheduler = optimizer.schedulers[scheduler_idx].clone();

            let patience_idx = (rng.next_f64() * optimizer.patience_values.len() as f64) as usize
                % optimizer.patience_values.len();
            let patience = optimizer.patience_values[patience_idx];

            let hp = optimizer.features_to_hp(&best_features, scheduler, patience);
            let result = Self::evaluate_hyperparameters(data, hp);

            x_train.push(best_features);
            y_train.push(result.val_loss);
            evaluations.push(result);
        }

        // Find best result
        let best_idx = evaluations
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.val_loss.partial_cmp(&b.val_loss).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        BayesianOptimizationResult {
            evaluations,
            best_idx,
            acquisition_history,
            n_initial_samples: optimizer.n_initial_samples,
            total_iterations: optimizer.n_iterations,
        }
    }

    /// Perform Bayesian optimization with cross-validation
    ///
    /// Combines Bayesian optimization with k-fold cross-validation for
    /// robust hyperparameter selection. More expensive but provides
    /// better generalization estimates.
    pub fn bayesian_optimize_with_cv(
        data: &[TrainingExample],
        optimizer: &BayesianOptimizer,
        k: usize,
    ) -> CVHyperparameterSearchResult {
        let mut rng = SimpleRng::new(optimizer.seed);
        let mut results: Vec<CVHyperparameterResult> = Vec::new();
        let mut x_train: Vec<Vec<f64>> = Vec::new();
        let mut y_train: Vec<f64> = Vec::new();

        let length_scale = 0.3;
        let noise = 1e-6;
        let k = k.max(2).min(data.len());

        // Initial random sampling
        for _ in 0..optimizer.n_initial_samples.min(optimizer.n_iterations) {
            let hp = optimizer.random_hp(&mut rng);
            let features = optimizer.hp_to_features(&hp);
            let cv_result = Self::evaluate_hyperparameters_cv(data, hp, k);

            x_train.push(features);
            y_train.push(cv_result.mean_val_loss);
            results.push(cv_result);
        }

        // Bayesian optimization iterations
        for _iter in optimizer.n_initial_samples..optimizer.n_iterations {
            let k_matrix = BayesianOptimizer::compute_kernel_matrix(&x_train, length_scale, noise);
            let l = match BayesianOptimizer::cholesky(&k_matrix) {
                Some(l) => l,
                None => {
                    let hp = optimizer.random_hp(&mut rng);
                    let features = optimizer.hp_to_features(&hp);
                    let cv_result = Self::evaluate_hyperparameters_cv(data, hp, k);
                    x_train.push(features);
                    y_train.push(cv_result.mean_val_loss);
                    results.push(cv_result);
                    continue;
                }
            };

            // Find best acquisition point
            let mut best_acq = f64::INFINITY;
            let mut best_features = vec![0.5, 0.5, 0.5];

            for lr_idx in 0..10 {
                for epoch_idx in 0..10 {
                    for val_idx in 0..5 {
                        let features = vec![
                            (lr_idx as f64 + 0.5) / 10.0,
                            (epoch_idx as f64 + 0.5) / 10.0,
                            (val_idx as f64 + 0.5) / 5.0,
                        ];

                        let (mean, variance) = BayesianOptimizer::gp_predict(
                            &features,
                            &x_train,
                            &y_train,
                            &l,
                            length_scale,
                        );

                        let acq = BayesianOptimizer::ucb(mean, variance, optimizer.kappa);

                        if acq < best_acq {
                            best_acq = acq;
                            best_features = features;
                        }
                    }
                }
            }

            let scheduler_idx = (rng.next_f64() * optimizer.schedulers.len() as f64) as usize
                % optimizer.schedulers.len();
            let scheduler = optimizer.schedulers[scheduler_idx].clone();

            let patience_idx = (rng.next_f64() * optimizer.patience_values.len() as f64) as usize
                % optimizer.patience_values.len();
            let patience = optimizer.patience_values[patience_idx];

            let hp = optimizer.features_to_hp(&best_features, scheduler, patience);
            let cv_result = Self::evaluate_hyperparameters_cv(data, hp, k);

            x_train.push(best_features);
            y_train.push(cv_result.mean_val_loss);
            results.push(cv_result);
        }

        let best_idx = results
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.mean_val_loss.partial_cmp(&b.mean_val_loss).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        CVHyperparameterSearchResult {
            total_evaluated: results.len(),
            results,
            best_idx,
            search_method: "bayesian".to_string(),
            k_folds: k,
        }
    }
}
