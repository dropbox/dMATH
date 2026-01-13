//! Ensemble model support for strategy prediction
//!
//! This module provides ensembling capabilities that combine multiple
//! `StrategyPredictor` models for improved prediction accuracy and robustness.

use super::{
    backend_to_idx, idx_to_backend, BackendPrediction, EvaluationResult, PropertyFeatureVector,
    StrategyPrediction, StrategyPredictor, TacticPrediction, TimePrediction, TrainingExample,
    MAX_TACTICS, NUM_BACKENDS,
};
use dashprove_backends::BackendId;
use dashprove_usl::ast::Property;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Aggregation strategy for combining model outputs in an ensemble
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum EnsembleAggregation {
    /// Average probability distributions across models (soft voting)
    #[default]
    SoftVoting,
    /// Use weighted majority voting on top-1 predictions
    WeightedMajority,
}

/// A single model in an ensemble
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleMember {
    /// Optional label for the model (e.g., checkpoint name)
    pub name: Option<String>,
    /// Weight applied during aggregation (normalized at runtime)
    pub weight: f64,
    /// The underlying predictor
    pub model: StrategyPredictor,
}

impl EnsembleMember {
    /// Create a new member with default weight 1.0
    pub fn new(model: StrategyPredictor) -> Self {
        Self {
            name: None,
            weight: 1.0,
            model,
        }
    }

    /// Set a human-readable name for the member
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Override the member weight
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    /// Sanitize weight for aggregation
    fn effective_weight(&self) -> f64 {
        if self.weight.is_finite() && self.weight > 0.0 {
            self.weight
        } else {
            0.0
        }
    }
}

/// Ensemble of strategy predictors with configurable aggregation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleStrategyPredictor {
    /// Models that participate in the ensemble
    pub members: Vec<EnsembleMember>,
    /// Aggregation method for combining predictions
    #[serde(default)]
    pub aggregation: EnsembleAggregation,
}

impl EnsembleStrategyPredictor {
    /// Create an ensemble from members using soft voting by default
    pub fn new(members: Vec<EnsembleMember>) -> Self {
        Self {
            members,
            aggregation: EnsembleAggregation::SoftVoting,
        }
    }

    /// Convenience helper to build an ensemble from plain predictors
    pub fn from_models(models: Vec<StrategyPredictor>) -> Self {
        let members = models.into_iter().map(EnsembleMember::new).collect();
        Self::new(members)
    }

    /// Configure aggregation method
    pub fn with_aggregation(mut self, aggregation: EnsembleAggregation) -> Self {
        self.aggregation = aggregation;
        self
    }

    /// Add a new member to the ensemble
    pub fn add_member(&mut self, member: EnsembleMember) {
        self.members.push(member);
    }

    /// Compute backend probability distribution after aggregation
    fn backend_distribution(&self, features: &PropertyFeatureVector) -> Vec<(BackendId, f64)> {
        if self.members.is_empty() {
            return StrategyPredictor::new().backend_probabilities(features);
        }

        let mut scores = vec![0.0f64; NUM_BACKENDS];
        let mut weight_sum = 0.0;

        for member in &self.members {
            let weight = member.effective_weight();
            if weight == 0.0 {
                continue;
            }
            weight_sum += weight;

            match self.aggregation {
                EnsembleAggregation::SoftVoting => {
                    for (backend, prob) in member.model.backend_probabilities(features) {
                        let idx = backend_to_idx(backend);
                        if idx < scores.len() {
                            scores[idx] += weight * prob;
                        }
                    }
                }
                EnsembleAggregation::WeightedMajority => {
                    let prediction = member.model.predict_backend(features);
                    let idx = backend_to_idx(prediction.backend);
                    if idx < scores.len() {
                        scores[idx] += weight * prediction.confidence.max(1e-6);
                    }
                    // Give smaller credit to alternatives to avoid overcommitment
                    for (alt_backend, alt_conf) in prediction.alternatives.iter().take(2) {
                        let alt_idx = backend_to_idx(*alt_backend);
                        if alt_idx < scores.len() {
                            scores[alt_idx] += weight * alt_conf * 0.25;
                        }
                    }
                }
            }
        }

        if weight_sum == 0.0 {
            return StrategyPredictor::new().backend_probabilities(features);
        }

        // Normalize
        let total: f64 = scores.iter().sum::<f64>().max(f64::EPSILON);
        scores
            .into_iter()
            .enumerate()
            .map(|(idx, score)| (idx_to_backend(idx), score / total))
            .collect()
    }

    /// Predict backend using aggregated distribution
    pub fn predict_backend(&self, features: &PropertyFeatureVector) -> BackendPrediction {
        let mut distribution = self.backend_distribution(features);
        distribution.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        if let Some((backend, prob)) = distribution.first().copied() {
            let alternatives = distribution
                .iter()
                .skip(1)
                .take(3)
                .map(|(b, p)| (*b, *p))
                .collect();

            BackendPrediction {
                backend,
                confidence: prob,
                alternatives,
            }
        } else {
            StrategyPredictor::new().predict_backend(features)
        }
    }

    /// Predict tactics by pooling member recommendations
    pub fn predict_tactics(
        &self,
        features: &PropertyFeatureVector,
        n: usize,
    ) -> Vec<TacticPrediction> {
        if self.members.is_empty() {
            return StrategyPredictor::new().predict_tactics(features, n);
        }

        let mut scores: HashMap<(usize, String), f64> = HashMap::new();
        let total_weight: f64 = self
            .members
            .iter()
            .map(|m| m.effective_weight())
            .filter(|w| *w > 0.0)
            .sum::<f64>()
            .max(f64::EPSILON);

        for member in &self.members {
            let weight = member.effective_weight();
            if weight == 0.0 {
                continue;
            }
            let predictions = member.model.predict_tactics(features, n);
            for tactic in predictions {
                let key = (tactic.position, tactic.tactic.clone());
                let score = scores.entry(key).or_insert(0.0);
                *score += weight * tactic.confidence;
            }
        }

        let mut ranked: Vec<_> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        ranked
            .into_iter()
            .take(n.min(MAX_TACTICS))
            .map(|((position, tactic), score)| TacticPrediction {
                tactic,
                position,
                confidence: (score / total_weight).clamp(0.0, 1.0),
            })
            .collect()
    }

    /// Predict expected time using weighted average of members
    pub fn predict_time(&self, features: &PropertyFeatureVector) -> TimePrediction {
        if self.members.is_empty() {
            return StrategyPredictor::new().predict_time(features);
        }

        let mut weighted_time = 0.0;
        let mut weighted_var = 0.0;
        let mut weighted_conf = 0.0;
        let mut weight_sum = 0.0;

        for member in &self.members {
            let weight = member.effective_weight();
            if weight == 0.0 {
                continue;
            }
            let time_pred = member.model.predict_time(features);
            weight_sum += weight;
            weighted_time += weight * time_pred.expected_seconds;
            weighted_conf += weight * time_pred.confidence;
            weighted_var += weight * (time_pred.range.1 - time_pred.range.0);
        }

        if weight_sum == 0.0 {
            return StrategyPredictor::new().predict_time(features);
        }

        let avg_time = weighted_time / weight_sum;
        let avg_conf = (weighted_conf / weight_sum).clamp(0.0, 1.0);
        let avg_range = (weighted_var / weight_sum).abs();

        TimePrediction {
            expected_seconds: avg_time,
            confidence: avg_conf,
            range: (avg_time - avg_range * 0.5, avg_time + avg_range * 0.5),
        }
    }

    /// Full strategy prediction using ensemble aggregation
    pub fn predict_strategy(&self, property: &Property) -> StrategyPrediction {
        let features = PropertyFeatureVector::from_property(property);
        let backend = self.predict_backend(&features);
        let tactics = self.predict_tactics(&features, MAX_TACTICS);
        let time = self.predict_time(&features);

        StrategyPrediction {
            backend,
            tactics,
            time,
            features,
        }
    }

    /// Evaluate ensemble accuracy on a test set
    pub fn evaluate(&self, test_data: &[TrainingExample]) -> EvaluationResult {
        if test_data.is_empty() {
            return EvaluationResult::default();
        }

        let mut total_loss = 0.0;
        let mut correct = 0;
        let mut per_backend_correct: HashMap<BackendId, usize> = HashMap::new();
        let mut per_backend_total: HashMap<BackendId, usize> = HashMap::new();
        let mut predictions: Vec<(BackendId, BackendId, f64)> = Vec::new();

        for example in test_data {
            let distribution = self.backend_distribution(&example.features);
            let prediction = self.predict_backend(&example.features);

            *per_backend_total.entry(example.backend).or_default() += 1;
            if prediction.backend == example.backend {
                correct += 1;
                *per_backend_correct.entry(example.backend).or_default() += 1;
            }

            // Cross-entropy using aggregated distribution
            let target_prob = distribution
                .iter()
                .find(|(backend, _)| *backend == example.backend)
                .map(|(_, prob)| *prob)
                .unwrap_or(1e-9);
            total_loss += -target_prob.max(1e-9).ln();

            predictions.push((example.backend, prediction.backend, prediction.confidence));
        }

        let total_examples = test_data.len();
        let accuracy = correct as f64 / total_examples as f64;
        let loss = total_loss / total_examples as f64;

        let mut per_backend_accuracy: HashMap<BackendId, f64> = HashMap::new();
        for (backend, total) in per_backend_total {
            let correct = per_backend_correct.get(&backend).copied().unwrap_or(0);
            per_backend_accuracy.insert(backend, correct as f64 / total as f64);
        }

        // Confidence calibration stats
        let mut confidence_correct = Vec::new();
        let mut confidence_incorrect = Vec::new();
        for (actual, predicted, confidence) in predictions {
            if actual == predicted {
                confidence_correct.push(confidence);
            } else {
                confidence_incorrect.push(confidence);
            }
        }

        let avg_confidence_correct = if confidence_correct.is_empty() {
            0.0
        } else {
            confidence_correct.iter().sum::<f64>() / confidence_correct.len() as f64
        };

        let avg_confidence_incorrect = if confidence_incorrect.is_empty() {
            0.0
        } else {
            confidence_incorrect.iter().sum::<f64>() / confidence_incorrect.len() as f64
        };

        EvaluationResult {
            accuracy,
            loss,
            total_examples,
            correct_predictions: correct,
            per_backend_accuracy,
            avg_confidence_correct,
            avg_confidence_incorrect,
        }
    }
}

/// Wrapper over single or ensemble predictors for easier loading and saving
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "model_type", rename_all = "snake_case")]
pub enum StrategyModel {
    /// Single trained predictor
    Single { model: Box<StrategyPredictor> },
    /// Ensemble of predictors
    Ensemble {
        ensemble: Box<EnsembleStrategyPredictor>,
    },
}

impl StrategyModel {
    /// Predict backend for given features
    pub fn predict_backend(&self, features: &PropertyFeatureVector) -> BackendPrediction {
        match self {
            StrategyModel::Single { model } => model.predict_backend(features),
            StrategyModel::Ensemble { ensemble } => ensemble.predict_backend(features),
        }
    }

    /// Predict tactics for a property
    pub fn predict_tactics(
        &self,
        features: &PropertyFeatureVector,
        n: usize,
    ) -> Vec<TacticPrediction> {
        match self {
            StrategyModel::Single { model } => model.predict_tactics(features, n),
            StrategyModel::Ensemble { ensemble } => ensemble.predict_tactics(features, n),
        }
    }

    /// Predict expected verification time
    pub fn predict_time(&self, features: &PropertyFeatureVector) -> TimePrediction {
        match self {
            StrategyModel::Single { model } => model.predict_time(features),
            StrategyModel::Ensemble { ensemble } => ensemble.predict_time(features),
        }
    }

    /// Full strategy prediction
    pub fn predict_strategy(&self, property: &Property) -> StrategyPrediction {
        match self {
            StrategyModel::Single { model } => model.predict_strategy(property),
            StrategyModel::Ensemble { ensemble } => ensemble.predict_strategy(property),
        }
    }

    /// Evaluate the model on a dataset
    pub fn evaluate(&self, data: &[TrainingExample]) -> EvaluationResult {
        match self {
            StrategyModel::Single { model } => model.evaluate(data),
            StrategyModel::Ensemble { ensemble } => ensemble.evaluate(data),
        }
    }

    /// Save model (single or ensemble) to disk
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)
    }

    /// Load model from disk, supporting legacy single-model files
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Self> {
        let path_ref = path.as_ref();
        let json = std::fs::read_to_string(path_ref)?;

        if let Ok(model) = serde_json::from_str::<StrategyModel>(&json) {
            return Ok(model);
        }

        // Fallback to legacy single-model encoding
        serde_json::from_str::<StrategyPredictor>(&json)
            .map(StrategyModel::from)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Extract inner single predictor if available
    pub fn as_single(&self) -> Option<&StrategyPredictor> {
        match self {
            StrategyModel::Single { model } => Some(model.as_ref()),
            StrategyModel::Ensemble { .. } => None,
        }
    }

    /// Consume and return inner predictor if single variant
    pub fn into_single(self) -> Option<StrategyPredictor> {
        match self {
            StrategyModel::Single { model } => Some(*model),
            StrategyModel::Ensemble { .. } => None,
        }
    }
}

impl From<StrategyPredictor> for StrategyModel {
    fn from(model: StrategyPredictor) -> Self {
        StrategyModel::Single {
            model: Box::new(model),
        }
    }
}

impl From<EnsembleStrategyPredictor> for StrategyModel {
    fn from(ensemble: EnsembleStrategyPredictor) -> Self {
        StrategyModel::Ensemble {
            ensemble: Box::new(ensemble),
        }
    }
}
