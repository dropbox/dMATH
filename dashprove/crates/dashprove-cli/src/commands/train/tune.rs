//! Hyperparameter tuning functionality
//!
//! Grid search, random search, and Bayesian optimization for training.

use dashprove::ai::strategy::{
    BayesianOptimizer, GridSearchSpace, RandomSearchConfig, StrategyPredictor,
    TrainingDataGenerator, TrainingExample, TrainingStats,
};
use dashprove::learning::ProofCorpus;
use std::path::PathBuf;

use super::config::default_data_dir;
use super::search_results::{
    print_bayesian_results, print_cv_search_results, print_search_results,
    train_and_save_from_bayesian, train_and_save_from_cv, train_and_save_model,
};

/// Configuration for the tune command
pub struct TuneConfig<'a> {
    /// Search method (grid, random, bayesian)
    pub method: &'a str,
    /// Directory containing learning data (default: ~/.dashprove)
    pub data_dir: Option<&'a str>,
    /// Output path for the tuned model
    pub output: Option<&'a str>,
    /// Number of configurations to try (for random/bayesian search)
    pub iterations: usize,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Learning rates to try (comma-separated, for grid search)
    pub lr_values: &'a str,
    /// Epoch counts to try (comma-separated, for grid search)
    pub epoch_values: &'a str,
    /// Minimum learning rate (for random/bayesian search)
    pub lr_min: f64,
    /// Maximum learning rate (for random/bayesian search)
    pub lr_max: f64,
    /// Minimum epochs (for random/bayesian search)
    pub epochs_min: usize,
    /// Maximum epochs (for random/bayesian search)
    pub epochs_max: usize,
    /// Number of initial random samples before using GP (for bayesian search)
    pub initial_samples: usize,
    /// Exploration-exploitation trade-off parameter (for bayesian search)
    pub kappa: f64,
    /// Number of cross-validation folds (0 = no CV)
    pub cv_folds: usize,
    /// Show verbose output
    pub verbose: bool,
}

/// Run the tune command (hyperparameter search)
pub fn run_tune(config: TuneConfig<'_>) -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = config
        .data_dir
        .map(PathBuf::from)
        .unwrap_or_else(default_data_dir);

    let corpus_path = data_dir.join("proof_corpus.json");

    if config.verbose {
        println!("Loading proof corpus from: {}", corpus_path.display());
    }

    // Load the proof corpus
    let corpus = ProofCorpus::load_or_default(&corpus_path).map_err(|e| {
        format!(
            "Failed to load corpus from {}: {}",
            corpus_path.display(),
            e
        )
    })?;

    let corpus_size = corpus.len();
    if corpus_size == 0 {
        println!(
            "Proof corpus is empty. Run verifications with --learn to build the corpus first."
        );
        println!("Example: dashprove verify spec.usl --learn");
        return Ok(());
    }

    println!("Loaded {} proofs from corpus", corpus_size);

    // Convert corpus entries to training data format
    let entries: Vec<_> = corpus
        .entries()
        .map(|entry| {
            let time_secs = entry.time_taken.as_secs_f64();
            (
                &entry.property,
                entry.backend,
                entry.tactics.as_slice(),
                time_secs,
            )
        })
        .collect();

    // Generate training examples
    let generator = TrainingDataGenerator::from_corpus_entries(entries);
    let stats = generator.stats();
    let training_examples: Vec<TrainingExample> = generator.get_training_data();

    if training_examples.is_empty() {
        println!("No successful proofs in corpus to train on.");
        return Ok(());
    }

    println!("Generated {} training examples", training_examples.len());

    // Determine output path
    let output_path = config
        .output
        .map(PathBuf::from)
        .unwrap_or_else(|| data_dir.join("strategy_model.json"));

    // Run hyperparameter search based on method
    match config.method.to_lowercase().as_str() {
        "grid" => run_grid_search(&config, &training_examples, &stats, &output_path),
        "random" => run_random_search(&config, &training_examples, &stats, &output_path),
        "bayesian" | "bayes" | "bo" => {
            run_bayesian_search(&config, &training_examples, &stats, &output_path)
        }
        _ => Err(format!(
            "Unknown search method '{}'. Use 'grid', 'random', or 'bayesian'.",
            config.method
        )
        .into()),
    }
}

fn run_grid_search(
    config: &TuneConfig<'_>,
    training_examples: &[TrainingExample],
    stats: &TrainingStats,
    output_path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\nStarting grid search...");

    // Parse learning rates and epochs from comma-separated strings
    let lr_values: Vec<f64> = config
        .lr_values
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    let epoch_values: Vec<usize> = config
        .epoch_values
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    if lr_values.is_empty() {
        return Err("No valid learning rates provided".into());
    }
    if epoch_values.is_empty() {
        return Err("No valid epoch counts provided".into());
    }

    let space = GridSearchSpace::new(lr_values)
        .with_epochs(epoch_values)
        .with_patience_values(vec![Some(5)]); // Use early stopping by default

    println!(
        "Evaluating {} configurations...",
        space.total_configurations()
    );

    if config.cv_folds > 0 {
        println!("Using {}-fold cross-validation", config.cv_folds);
        let cv_result =
            StrategyPredictor::grid_search_with_cv(training_examples, &space, config.cv_folds);
        print_cv_search_results(&cv_result, config.verbose);
        train_and_save_from_cv(&cv_result, training_examples, stats, output_path)?;
        return Ok(());
    }

    let search_result = StrategyPredictor::grid_search(training_examples, &space);
    print_search_results(&search_result, config.verbose);
    train_and_save_model(&search_result, training_examples, stats, output_path)
}

fn run_random_search(
    config: &TuneConfig<'_>,
    training_examples: &[TrainingExample],
    stats: &TrainingStats,
    output_path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "\nStarting random search with {} iterations...",
        config.iterations
    );

    let random_config = RandomSearchConfig::new(config.iterations)
        .with_lr_range(config.lr_min, config.lr_max)
        .with_epoch_range(config.epochs_min, config.epochs_max)
        .with_seed(config.seed);

    if config.cv_folds > 0 {
        println!("Using {}-fold cross-validation", config.cv_folds);
        let cv_result = StrategyPredictor::random_search_with_cv(
            training_examples,
            &random_config,
            config.cv_folds,
        );
        print_cv_search_results(&cv_result, config.verbose);
        train_and_save_from_cv(&cv_result, training_examples, stats, output_path)?;
        return Ok(());
    }

    let search_result = StrategyPredictor::random_search(training_examples, &random_config);
    print_search_results(&search_result, config.verbose);
    train_and_save_model(&search_result, training_examples, stats, output_path)
}

fn run_bayesian_search(
    config: &TuneConfig<'_>,
    training_examples: &[TrainingExample],
    stats: &TrainingStats,
    output_path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "\nStarting Bayesian optimization with {} iterations...",
        config.iterations
    );
    println!("  Initial random samples: {}", config.initial_samples);
    println!("  Exploration parameter (kappa): {:.3}", config.kappa);

    let optimizer = BayesianOptimizer::new(config.iterations)
        .with_lr_bounds(config.lr_min, config.lr_max)
        .with_epoch_bounds(config.epochs_min, config.epochs_max)
        .with_initial_samples(config.initial_samples)
        .with_kappa(config.kappa)
        .with_seed(config.seed);

    // CV and non-CV versions return different types
    if config.cv_folds > 0 {
        println!("Using {}-fold cross-validation", config.cv_folds);
        let cv_result = StrategyPredictor::bayesian_optimize_with_cv(
            training_examples,
            &optimizer,
            config.cv_folds,
        );
        print_cv_search_results(&cv_result, config.verbose);
        train_and_save_from_cv(&cv_result, training_examples, stats, output_path)
    } else {
        let bayesian_result = StrategyPredictor::bayesian_optimize(training_examples, &optimizer);
        print_bayesian_results(&bayesian_result, config.verbose);
        train_and_save_from_bayesian(&bayesian_result, training_examples, stats, output_path)
    }
}
