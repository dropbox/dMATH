use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ============ Corpus Stats Types ============

/// Response for corpus statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct CorpusStatsResponse {
    /// Proof corpus statistics
    pub proofs: ProofStatsResponse,
    /// Counterexample corpus statistics
    pub counterexamples: CounterexampleStatsResponse,
    /// Tactic statistics
    pub tactics: TacticStatsResponse,
}

/// Proof corpus statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct ProofStatsResponse {
    /// Total number of proofs
    pub total: usize,
    /// Count by backend
    pub by_backend: HashMap<String, usize>,
}

/// Counterexample corpus statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct CounterexampleStatsResponse {
    /// Total number of counterexamples
    pub total: usize,
    /// Number of cluster patterns
    pub cluster_patterns: usize,
    /// Count by backend
    pub by_backend: HashMap<String, usize>,
}

/// Tactic statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct TacticStatsResponse {
    /// Total tactic observations
    pub total_observations: u32,
    /// Number of unique tactics
    pub unique_tactics: usize,
    /// Top tactics by Wilson score (tactic name -> score)
    pub top_tactics: Vec<TacticScoreResponse>,
}

/// A single tactic with its score
#[derive(Debug, Serialize, Deserialize)]
pub struct TacticScoreResponse {
    /// Tactic name
    pub name: String,
    /// Wilson score
    pub score: f64,
}
