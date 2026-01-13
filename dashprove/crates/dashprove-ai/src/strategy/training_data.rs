//! Training data types and generators for the strategy module.
//!
//! This module contains types for managing training examples and generating
//! training data from proof corpus entries.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use dashprove_backends::BackendId;
use dashprove_usl::ast::Property;

use super::features::PropertyFeatureVector;
use super::predictions::{BackendPrediction, TacticPrediction, TimePrediction};

/// Full strategy prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyPrediction {
    /// Backend prediction
    pub backend: BackendPrediction,
    /// Tactic predictions in order
    pub tactics: Vec<TacticPrediction>,
    /// Time prediction
    pub time: TimePrediction,
    /// Input features used
    pub features: PropertyFeatureVector,
}

/// Training example for the predictor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    /// Input features
    pub features: PropertyFeatureVector,
    /// Correct backend
    pub backend: BackendId,
    /// Successful tactics used
    pub tactics: Vec<String>,
    /// Actual verification time in seconds
    pub time_seconds: f64,
    /// Whether verification succeeded
    pub success: bool,
}

impl TrainingExample {
    /// Create a training example from a property and verification result
    pub fn from_verification(
        property: &Property,
        backend: BackendId,
        tactics: Vec<String>,
        time_seconds: f64,
        success: bool,
    ) -> Self {
        TrainingExample {
            features: PropertyFeatureVector::from_property(property),
            backend,
            tactics,
            time_seconds,
            success,
        }
    }
}

/// Training data generator from proof corpus
#[derive(Debug, Default)]
pub struct TrainingDataGenerator {
    examples: Vec<TrainingExample>,
}

impl TrainingDataGenerator {
    /// Create a new training data generator
    pub fn new() -> Self {
        TrainingDataGenerator {
            examples: Vec::new(),
        }
    }

    /// Add an example from a successful verification
    pub fn add_success(
        &mut self,
        property: &Property,
        backend: BackendId,
        tactics: Vec<String>,
        time_seconds: f64,
    ) {
        self.examples.push(TrainingExample::from_verification(
            property,
            backend,
            tactics,
            time_seconds,
            true,
        ));
    }

    /// Add an example from a failed verification (negative example)
    pub fn add_failure(
        &mut self,
        property: &Property,
        backend: BackendId,
        tactics: Vec<String>,
        time_seconds: f64,
    ) {
        self.examples.push(TrainingExample::from_verification(
            property,
            backend,
            tactics,
            time_seconds,
            false,
        ));
    }

    /// Get training examples (successes only for supervised learning)
    pub fn get_training_data(&self) -> Vec<TrainingExample> {
        self.examples
            .iter()
            .filter(|e| e.success)
            .cloned()
            .collect()
    }

    /// Get all examples including failures
    pub fn get_all_examples(&self) -> &[TrainingExample] {
        &self.examples
    }

    /// Statistics about the training data
    pub fn stats(&self) -> TrainingStats {
        let total = self.examples.len();
        let successes = self.examples.iter().filter(|e| e.success).count();

        let mut backend_counts: HashMap<BackendId, usize> = HashMap::new();
        for example in &self.examples {
            if example.success {
                *backend_counts.entry(example.backend).or_default() += 1;
            }
        }

        TrainingStats {
            total_examples: total,
            successful_examples: successes,
            failed_examples: total - successes,
            examples_per_backend: backend_counts,
        }
    }

    /// Generate training data from proof corpus entries
    ///
    /// This converts stored proofs into training examples for the ML model.
    /// Each successful proof becomes a training example with:
    /// - Features extracted from the property
    /// - The backend that was used
    /// - The tactics that worked
    /// - The time taken
    pub fn from_corpus_entries<'a, I>(entries: I) -> Self
    where
        I: IntoIterator<Item = (&'a Property, BackendId, &'a [String], f64)>,
    {
        let mut generator = Self::new();
        for (property, backend, tactics, time_seconds) in entries {
            generator.add_success(property, backend, tactics.to_vec(), time_seconds);
        }
        generator
    }
}

/// Statistics about training data
#[derive(Debug, Clone)]
pub struct TrainingStats {
    /// Total number of examples
    pub total_examples: usize,
    /// Number of successful verifications
    pub successful_examples: usize,
    /// Number of failed verifications
    pub failed_examples: usize,
    /// Examples per backend
    pub examples_per_backend: HashMap<BackendId, usize>,
}
