//! Ensemble model building
//!
//! Combine multiple trained models into an ensemble.

use dashprove::ai::strategy::{
    EnsembleAggregation, EnsembleMember, EnsembleStrategyPredictor, StrategyModel,
};
use std::path::{Path, PathBuf};

use super::config::default_data_dir;

/// Configuration for the ensemble command
pub struct EnsembleConfig<'a> {
    /// Paths to model files to combine
    pub models: Vec<&'a str>,
    /// Optional comma-separated weights aligned with models
    pub weights: Option<&'a str>,
    /// Aggregation method (soft|weighted)
    pub method: &'a str,
    /// Output path for the ensemble model
    pub output: Option<&'a str>,
    /// Directory containing learning data (default: ~/.dashprove)
    pub data_dir: Option<&'a str>,
    /// Show verbose output
    pub verbose: bool,
}

/// Combine multiple trained models into an ensemble and save it
pub fn run_ensemble(config: EnsembleConfig<'_>) -> Result<(), Box<dyn std::error::Error>> {
    if config.models.len() < 2 {
        return Err("Ensemble requires at least two model paths".into());
    }

    let data_dir = config
        .data_dir
        .map(PathBuf::from)
        .unwrap_or_else(default_data_dir);

    let output_path = config
        .output
        .map(PathBuf::from)
        .unwrap_or_else(|| data_dir.join("strategy_ensemble.json"));

    let aggregation = match config.method.to_lowercase().as_str() {
        "soft" | "soft_voting" | "average" => EnsembleAggregation::SoftVoting,
        "weighted" | "weighted_majority" | "hard" | "vote" => EnsembleAggregation::WeightedMajority,
        other => {
            return Err(
                format!("Unknown ensemble method '{}'. Use soft or weighted.", other).into(),
            )
        }
    };

    let weights = parse_weights(config.weights, config.models.len())?;

    if config.verbose {
        println!("Building ensemble with {} models", config.models.len());
        println!("Aggregation: {:?}", aggregation);
        if !weights.is_empty() {
            println!("Weights: {:?}", weights);
        }
    }

    let mut members = Vec::with_capacity(config.models.len());

    for (idx, model_path) in config.models.iter().enumerate() {
        let resolved = resolve_model_path(model_path, &data_dir);
        let model = StrategyModel::load(&resolved)
            .map_err(|e| format!("Failed to load model at {}: {}", resolved.display(), e))?;

        let predictor = match model {
            StrategyModel::Single { model } => *model,
            StrategyModel::Ensemble { .. } => {
                return Err(format!(
                    "Model {} is already an ensemble; provide base models instead",
                    resolved.display()
                )
                .into())
            }
        };

        let mut member = EnsembleMember::new(predictor);
        if let Some(weight) = weights.get(idx) {
            member = member.with_weight(*weight);
        }
        let name = resolved
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(model_path);
        member.name = Some(name.to_string());
        members.push(member);
    }

    let ensemble = EnsembleStrategyPredictor::new(members).with_aggregation(aggregation);
    let ensemble_model = StrategyModel::from(ensemble);

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create directory {}: {}", parent.display(), e))?;
    }

    ensemble_model
        .save(&output_path)
        .map_err(|e| format!("Failed to save ensemble: {}", e))?;

    println!(
        "\nEnsemble built with {} models using {:?} aggregation",
        config.models.len(),
        aggregation
    );
    println!("Ensemble saved to: {}", output_path.display());
    println!(
        "\nUse this ensemble for verification:\n  dashprove verify spec.usl --ml --ml-model {}",
        output_path.display()
    );

    Ok(())
}

pub fn parse_weights(
    raw_weights: Option<&str>,
    expected: usize,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let Some(raw) = raw_weights else {
        return Ok(Vec::new());
    };

    let weights: Vec<f64> = raw
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|w| {
            w.trim()
                .parse::<f64>()
                .map_err(|e| format!("Invalid weight '{}': {}", w.trim(), e))
        })
        .collect::<Result<_, _>>()?;

    if !weights.is_empty() && weights.len() != expected {
        return Err(format!("Provided {} weights for {} models", weights.len(), expected).into());
    }

    Ok(weights)
}

fn resolve_model_path(model_path: &str, data_dir: &Path) -> PathBuf {
    let candidate = PathBuf::from(model_path);
    if candidate.exists() {
        return candidate;
    }

    let in_data_dir = data_dir.join(model_path);
    if in_data_dir.exists() {
        in_data_dir
    } else {
        candidate
    }
}
