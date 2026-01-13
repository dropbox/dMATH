//! Backend reputation tracking system
//!
//! This module provides reputation tracking for verification backends based on
//! their historical success rates. Reputation scores can be used with the
//! `WeightedConsensus` merge strategy in the dispatcher.
//!
//! # Overview
//!
//! The reputation system tracks:
//! - **Success rate**: How often a backend produces correct proofs
//! - **Response time**: Average verification latency
//! - **Consistency**: Whether results match consensus
//! - **Coverage**: What types of properties the backend handles well
//!
//! Reputation scores decay over time to give more weight to recent performance.
//!
//! # Example
//!
//! ```rust,no_run
//! use dashprove_learning::{ReputationConfig, ReputationTracker};
//! use dashprove_backends::BackendId;
//! use std::time::Duration;
//!
//! // Create tracker with default config
//! let mut tracker = ReputationTracker::new(ReputationConfig::default());
//!
//! // Record successful verification
//! tracker.record_success(BackendId::Lean4, Duration::from_millis(500));
//!
//! // Record failure (disagreed with consensus or timed out)
//! tracker.record_failure(BackendId::Alloy);
//!
//! // Get current reputation scores for WeightedConsensus
//! let weights = tracker.compute_weights();
//! assert!(weights[&BackendId::Lean4] > weights.get(&BackendId::Alloy).copied().unwrap_or(0.5));
//! ```

use chrono::{DateTime, Utc};
use dashprove_backends::{BackendId, PropertyType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Key for domain-specific reputation tracking
///
/// Combines a backend ID with a property type to track backend performance
/// in specific domains (e.g., "Lean4 for theorems" vs "Lean4 for contracts").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DomainKey {
    /// The backend being tracked
    pub backend: BackendId,
    /// The property type this stat applies to
    pub property_type: PropertyType,
}

impl DomainKey {
    /// Create a new domain key
    pub fn new(backend: BackendId, property_type: PropertyType) -> Self {
        Self {
            backend,
            property_type,
        }
    }
}

impl std::fmt::Display for DomainKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}/{:?}", self.backend, self.property_type)
    }
}

/// Summary of domain-specific (backend + property type) performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainSummary {
    /// The backend
    pub backend: BackendId,
    /// The property type
    pub property_type: PropertyType,
    /// Number of successful verifications
    pub successes: usize,
    /// Number of failed verifications
    pub failures: usize,
    /// Simple success rate (successes / total)
    pub success_rate: f64,
    /// Computed reputation score
    pub reputation: f64,
    /// Average response time in milliseconds (if available)
    pub avg_response_time_ms: Option<f64>,
}

impl DomainSummary {
    /// Total observations for this domain
    pub fn total_observations(&self) -> usize {
        self.successes + self.failures
    }
}

/// Configuration for reputation tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationConfig {
    /// Decay factor for historical observations (0.0-1.0)
    /// Higher values = slower decay = more influence from old data
    /// Default: 0.95 (observations lose ~50% influence after ~14 observations)
    pub decay_factor: f64,

    /// Minimum observations before reputation is considered reliable
    /// Default: 5
    pub min_observations: usize,

    /// Default reputation for backends with insufficient data
    /// Default: 0.5
    pub default_reputation: f64,

    /// Bonus weight for faster-than-average response times
    /// Default: 0.05
    pub speed_bonus: f64,

    /// Maximum reputation score (clamped)
    /// Default: 0.99
    pub max_reputation: f64,

    /// Minimum reputation score (clamped)
    /// Default: 0.01
    pub min_reputation: f64,

    /// Weight for success rate in overall reputation (remainder goes to speed)
    /// Default: 0.85
    pub success_weight: f64,
}

impl Default for ReputationConfig {
    fn default() -> Self {
        Self {
            decay_factor: 0.95,
            min_observations: 5,
            default_reputation: 0.5,
            speed_bonus: 0.05,
            max_reputation: 0.99,
            min_reputation: 0.01,
            success_weight: 0.85,
        }
    }
}

impl ReputationConfig {
    /// Create a configuration that prioritizes recent performance
    pub fn recent_focused() -> Self {
        Self {
            decay_factor: 0.85, // Faster decay
            min_observations: 3,
            ..Default::default()
        }
    }

    /// Create a configuration that emphasizes long-term track record
    pub fn stable_focused() -> Self {
        Self {
            decay_factor: 0.98, // Very slow decay
            min_observations: 10,
            ..Default::default()
        }
    }
}

/// Statistics for a single backend
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BackendStats {
    /// Total number of successful verifications
    pub successes: usize,
    /// Total number of failed verifications
    pub failures: usize,
    /// Exponentially weighted moving average of success rate
    pub ewma_success_rate: f64,
    /// Sum of all response times (in milliseconds)
    pub total_response_time_ms: u64,
    /// Number of response time observations
    pub response_time_count: usize,
    /// Exponentially weighted moving average of response time
    pub ewma_response_time_ms: f64,
    /// When the stats were last updated
    pub last_updated: Option<DateTime<Utc>>,
    /// When this backend was first observed
    pub first_observed: Option<DateTime<Utc>>,
}

impl BackendStats {
    /// Create new backend stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Total number of observations
    pub fn total_observations(&self) -> usize {
        self.successes + self.failures
    }

    /// Simple success rate (successes / total)
    pub fn simple_success_rate(&self) -> f64 {
        let total = self.total_observations();
        if total == 0 {
            return 0.5; // Default
        }
        self.successes as f64 / total as f64
    }

    /// Average response time in milliseconds
    pub fn avg_response_time_ms(&self) -> Option<f64> {
        if self.response_time_count == 0 {
            return None;
        }
        Some(self.total_response_time_ms as f64 / self.response_time_count as f64)
    }

    /// Record a successful verification
    pub fn record_success(&mut self, response_time: Duration, decay: f64) {
        let now = Utc::now();
        self.successes += 1;

        // Update EWMA success rate
        let new_value = 1.0;
        if self.total_observations() == 1 {
            self.ewma_success_rate = new_value;
        } else {
            self.ewma_success_rate = decay * self.ewma_success_rate + (1.0 - decay) * new_value;
        }

        // Update response time stats
        let time_ms = response_time.as_millis() as u64;
        self.total_response_time_ms += time_ms;
        self.response_time_count += 1;
        let time_f64 = time_ms as f64;
        if self.response_time_count == 1 {
            self.ewma_response_time_ms = time_f64;
        } else {
            self.ewma_response_time_ms =
                decay * self.ewma_response_time_ms + (1.0 - decay) * time_f64;
        }

        self.last_updated = Some(now);
        if self.first_observed.is_none() {
            self.first_observed = Some(now);
        }
    }

    /// Record a failed verification
    pub fn record_failure(&mut self, decay: f64) {
        let now = Utc::now();
        self.failures += 1;

        // Update EWMA success rate
        let new_value = 0.0;
        if self.total_observations() == 1 {
            self.ewma_success_rate = new_value;
        } else {
            self.ewma_success_rate = decay * self.ewma_success_rate + (1.0 - decay) * new_value;
        }

        self.last_updated = Some(now);
        if self.first_observed.is_none() {
            self.first_observed = Some(now);
        }
    }
}

/// Single reputation observation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationObservation {
    /// Which backend
    pub backend: BackendId,
    /// Whether verification succeeded
    pub success: bool,
    /// Response time (if available)
    pub response_time: Option<Duration>,
    /// When this observation was recorded
    pub timestamp: DateTime<Utc>,
    /// Property category (if known) for domain-specific tracking
    pub property_category: Option<String>,
}

/// Main reputation tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationTracker {
    /// Configuration
    config: ReputationConfig,
    /// Per-backend statistics (aggregate across all property types)
    stats: HashMap<BackendId, BackendStats>,
    /// Domain-specific statistics (backend + property type)
    #[serde(default)]
    domain_stats: HashMap<DomainKey, BackendStats>,
    /// Overall average response time for speed comparison
    global_avg_response_time_ms: f64,
    /// Count of all observations for global average
    global_observation_count: usize,
}

impl ReputationTracker {
    /// Create a new reputation tracker with the given configuration
    pub fn new(config: ReputationConfig) -> Self {
        Self {
            config,
            stats: HashMap::new(),
            domain_stats: HashMap::new(),
            global_avg_response_time_ms: 1000.0, // Default 1 second
            global_observation_count: 0,
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &ReputationConfig {
        &self.config
    }

    /// Record a successful verification
    pub fn record_success(&mut self, backend: BackendId, response_time: Duration) {
        let decay = self.config.decay_factor;
        let stats = self.stats.entry(backend).or_default();
        stats.record_success(response_time, decay);

        // Update global average
        let time_ms = response_time.as_millis() as f64;
        self.global_observation_count += 1;
        if self.global_observation_count == 1 {
            self.global_avg_response_time_ms = time_ms;
        } else {
            // Simple moving average for global
            self.global_avg_response_time_ms = (self.global_avg_response_time_ms
                * (self.global_observation_count - 1) as f64
                + time_ms)
                / self.global_observation_count as f64;
        }
    }

    /// Record a failed verification (timeout, error, or disagreement with consensus)
    pub fn record_failure(&mut self, backend: BackendId) {
        let decay = self.config.decay_factor;
        let stats = self.stats.entry(backend).or_default();
        stats.record_failure(decay);
    }

    /// Record a full observation
    pub fn record_observation(&mut self, obs: ReputationObservation) {
        if obs.success {
            let time = obs.response_time.unwrap_or(Duration::from_secs(1));
            self.record_success(obs.backend, time);
        } else {
            self.record_failure(obs.backend);
        }
    }

    // ==========================================================================
    // Domain-specific tracking (by property type)
    // ==========================================================================

    /// Record a successful verification for a specific domain (backend + property type)
    ///
    /// This updates both the aggregate backend stats and the domain-specific stats.
    pub fn record_domain_success(
        &mut self,
        backend: BackendId,
        property_type: PropertyType,
        response_time: Duration,
    ) {
        // Update aggregate stats
        self.record_success(backend, response_time);

        // Update domain-specific stats
        let decay = self.config.decay_factor;
        let key = DomainKey::new(backend, property_type);
        let stats = self.domain_stats.entry(key).or_default();
        stats.record_success(response_time, decay);
    }

    /// Record a failed verification for a specific domain (backend + property type)
    ///
    /// This updates both the aggregate backend stats and the domain-specific stats.
    pub fn record_domain_failure(&mut self, backend: BackendId, property_type: PropertyType) {
        // Update aggregate stats
        self.record_failure(backend);

        // Update domain-specific stats
        let decay = self.config.decay_factor;
        let key = DomainKey::new(backend, property_type);
        let stats = self.domain_stats.entry(key).or_default();
        stats.record_failure(decay);
    }

    /// Get domain-specific statistics for a backend + property type combination
    pub fn get_domain_stats(&self, key: &DomainKey) -> Option<&BackendStats> {
        self.domain_stats.get(key)
    }

    /// Get all tracked domain keys
    pub fn domain_keys(&self) -> impl Iterator<Item = &DomainKey> {
        self.domain_stats.keys()
    }

    /// Compute the domain-specific reputation score for a backend + property type
    ///
    /// Falls back to aggregate reputation if domain-specific data is insufficient.
    pub fn compute_domain_reputation(&self, key: &DomainKey) -> f64 {
        let Some(stats) = self.domain_stats.get(key) else {
            // Fall back to aggregate reputation
            return self.compute_reputation(&key.backend);
        };

        // Not enough domain-specific data - fall back to aggregate
        if stats.total_observations() < self.config.min_observations {
            return self.compute_reputation(&key.backend);
        }

        // Compute domain-specific reputation using same formula as aggregate
        let success_component = stats.ewma_success_rate * self.config.success_weight;

        let speed_component = if stats.response_time_count > 0 {
            let ratio = self.global_avg_response_time_ms / stats.ewma_response_time_ms.max(1.0);
            let speed_score = (ratio - 1.0).clamp(-1.0, 1.0) * self.config.speed_bonus;
            ((1.0 - self.config.success_weight) * 0.5) + speed_score
        } else {
            (1.0 - self.config.success_weight) * 0.5
        };

        let reputation = success_component + speed_component;
        reputation.clamp(self.config.min_reputation, self.config.max_reputation)
    }

    /// Compute domain-specific weights for all tracked backend + property type combinations
    ///
    /// Returns a HashMap suitable for domain-aware weighted consensus.
    pub fn compute_domain_weights(&self) -> HashMap<DomainKey, f64> {
        self.domain_stats
            .keys()
            .map(|key| (*key, self.compute_domain_reputation(key)))
            .collect()
    }

    /// Get all property types that a specific backend has been used for
    pub fn property_types_for_backend(&self, backend: BackendId) -> Vec<PropertyType> {
        self.domain_stats
            .keys()
            .filter(|k| k.backend == backend)
            .map(|k| k.property_type)
            .collect()
    }

    /// Get all backends that have been used for a specific property type
    pub fn backends_for_property_type(&self, property_type: PropertyType) -> Vec<BackendId> {
        self.domain_stats
            .keys()
            .filter(|k| k.property_type == property_type)
            .map(|k| k.backend)
            .collect()
    }

    /// Get the number of tracked domain-specific entries
    pub fn domain_count(&self) -> usize {
        self.domain_stats.len()
    }

    /// Get total domain-specific observations
    pub fn total_domain_observations(&self) -> usize {
        self.domain_stats
            .values()
            .map(|s| s.total_observations())
            .sum()
    }

    /// Get the best backend for a specific property type based on domain reputation
    ///
    /// Returns None if no backends have been tracked for this property type.
    pub fn best_backend_for_property_type(&self, property_type: PropertyType) -> Option<BackendId> {
        self.domain_stats
            .keys()
            .filter(|k| k.property_type == property_type)
            .max_by(|a, b| {
                let rep_a = self.compute_domain_reputation(a);
                let rep_b = self.compute_domain_reputation(b);
                rep_a
                    .partial_cmp(&rep_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|k| k.backend)
    }

    /// Get a summary of domain-specific performance
    pub fn domain_summary(&self) -> Vec<DomainSummary> {
        self.domain_stats
            .iter()
            .map(|(key, stats)| DomainSummary {
                backend: key.backend,
                property_type: key.property_type,
                successes: stats.successes,
                failures: stats.failures,
                success_rate: stats.simple_success_rate(),
                reputation: self.compute_domain_reputation(key),
                avg_response_time_ms: stats.avg_response_time_ms(),
            })
            .collect()
    }

    /// Get statistics for a specific backend
    pub fn get_stats(&self, backend: &BackendId) -> Option<&BackendStats> {
        self.stats.get(backend)
    }

    /// Get all tracked backends
    pub fn backends(&self) -> impl Iterator<Item = &BackendId> {
        self.stats.keys()
    }

    /// Compute the reputation score for a single backend
    pub fn compute_reputation(&self, backend: &BackendId) -> f64 {
        let Some(stats) = self.stats.get(backend) else {
            return self.config.default_reputation;
        };

        // Not enough data
        if stats.total_observations() < self.config.min_observations {
            return self.config.default_reputation;
        }

        // Base reputation from success rate (use EWMA)
        let success_component = stats.ewma_success_rate * self.config.success_weight;

        // Speed bonus (faster than average = bonus)
        let speed_component = if stats.response_time_count > 0 {
            let ratio = self.global_avg_response_time_ms / stats.ewma_response_time_ms.max(1.0);
            // ratio > 1.0 means faster than average
            let speed_score = (ratio - 1.0).clamp(-1.0, 1.0) * self.config.speed_bonus;
            ((1.0 - self.config.success_weight) * 0.5) + speed_score
        } else {
            (1.0 - self.config.success_weight) * 0.5
        };

        let reputation = success_component + speed_component;
        reputation.clamp(self.config.min_reputation, self.config.max_reputation)
    }

    /// Compute reputation weights for all tracked backends
    ///
    /// Returns a HashMap suitable for use with `MergeStrategy::WeightedConsensus`
    pub fn compute_weights(&self) -> HashMap<BackendId, f64> {
        self.stats
            .keys()
            .map(|backend| (*backend, self.compute_reputation(backend)))
            .collect()
    }

    /// Get the number of tracked backends
    pub fn backend_count(&self) -> usize {
        self.stats.len()
    }

    /// Get the total number of observations across all backends
    pub fn total_observations(&self) -> usize {
        self.stats.values().map(|s| s.total_observations()).sum()
    }

    /// Persist the tracker to a JSON file
    pub fn save_to_file<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), crate::LearningError> {
        crate::io::write_json_atomic(path, self)
    }

    /// Load a tracker from disk
    pub fn load_from_file<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, crate::LearningError> {
        crate::io::read_json(path)
    }

    /// Load a tracker if it exists, otherwise return default
    pub fn load_or_default<P: AsRef<std::path::Path>>(
        path: P,
        config: ReputationConfig,
    ) -> Result<Self, crate::LearningError> {
        match std::fs::File::open(path.as_ref()) {
            Ok(_) => Self::load_from_file(path),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::new(config)),
            Err(e) => Err(e.into()),
        }
    }

    /// Merge stats from another tracker (for combining distributed observations)
    pub fn merge(&mut self, other: &ReputationTracker) {
        for (backend, other_stats) in &other.stats {
            let stats = self.stats.entry(*backend).or_default();

            // Merge counts
            stats.successes += other_stats.successes;
            stats.failures += other_stats.failures;
            stats.total_response_time_ms += other_stats.total_response_time_ms;
            stats.response_time_count += other_stats.response_time_count;

            // Take newer timestamps
            if let Some(other_last) = other_stats.last_updated {
                stats.last_updated = Some(match stats.last_updated {
                    Some(current) if current > other_last => current,
                    _ => other_last,
                });
            }
            if let Some(other_first) = other_stats.first_observed {
                stats.first_observed = Some(match stats.first_observed {
                    Some(current) if current < other_first => current,
                    _ => other_first,
                });
            }

            // Recompute EWMA from merged totals
            let total = stats.total_observations();
            if total > 0 {
                stats.ewma_success_rate = stats.successes as f64 / total as f64;
            }
            if stats.response_time_count > 0 {
                stats.ewma_response_time_ms =
                    stats.total_response_time_ms as f64 / stats.response_time_count as f64;
            }
        }

        // Update global stats
        self.global_observation_count += other.global_observation_count;
        if self.global_observation_count > 0 {
            self.global_avg_response_time_ms = (self.global_avg_response_time_ms
                * (self.global_observation_count - other.global_observation_count) as f64
                + other.global_avg_response_time_ms * other.global_observation_count as f64)
                / self.global_observation_count as f64;
        }

        // Merge domain-specific stats
        for (key, other_stats) in &other.domain_stats {
            let stats = self.domain_stats.entry(*key).or_default();

            // Merge counts
            stats.successes += other_stats.successes;
            stats.failures += other_stats.failures;
            stats.total_response_time_ms += other_stats.total_response_time_ms;
            stats.response_time_count += other_stats.response_time_count;

            // Take newer timestamps
            if let Some(other_last) = other_stats.last_updated {
                stats.last_updated = Some(match stats.last_updated {
                    Some(current) if current > other_last => current,
                    _ => other_last,
                });
            }
            if let Some(other_first) = other_stats.first_observed {
                stats.first_observed = Some(match stats.first_observed {
                    Some(current) if current < other_first => current,
                    _ => other_first,
                });
            }

            // Recompute EWMA from merged totals
            let total = stats.total_observations();
            if total > 0 {
                stats.ewma_success_rate = stats.successes as f64 / total as f64;
            }
            if stats.response_time_count > 0 {
                stats.ewma_response_time_ms =
                    stats.total_response_time_ms as f64 / stats.response_time_count as f64;
            }
        }
    }
}

impl Default for ReputationTracker {
    fn default() -> Self {
        Self::new(ReputationConfig::default())
    }
}

/// Convert a property type string (from PropertyFeatures) to a PropertyType enum.
///
/// The string property types come from the similarity module's `extract_features()` function.
/// This function maps those strings to the backend PropertyType enum for domain-specific
/// reputation tracking.
///
/// Delegates to [`PropertyType::parse`] for the canonical string-to-enum mapping.
///
/// # Returns
/// - `Some(PropertyType)` if the string matches a known property type
/// - `None` if the string is unknown or empty
#[inline]
pub fn property_type_from_string(s: &str) -> Option<PropertyType> {
    PropertyType::parse(s)
}

/// Builder for constructing reputation from proof corpus
#[derive(Debug)]
pub struct ReputationFromCorpus<'a> {
    corpus: &'a crate::ProofCorpus,
    config: ReputationConfig,
    include_domain_stats: bool,
}

impl<'a> ReputationFromCorpus<'a> {
    /// Create a new builder
    pub fn new(corpus: &'a crate::ProofCorpus) -> Self {
        Self {
            corpus,
            config: ReputationConfig::default(),
            include_domain_stats: true, // Include domain stats by default
        }
    }

    /// Use custom configuration
    pub fn with_config(mut self, config: ReputationConfig) -> Self {
        self.config = config;
        self
    }

    /// Enable or disable domain-specific reputation tracking
    ///
    /// When enabled (default), the builder will extract property types from
    /// the proof corpus entries and build domain-specific reputation stats.
    pub fn with_domain_stats(mut self, include: bool) -> Self {
        self.include_domain_stats = include;
        self
    }

    /// Build the reputation tracker from corpus data
    ///
    /// Only uses successful proofs from the corpus, so the "failure" rate
    /// represents backends that weren't chosen or failed before being recorded.
    ///
    /// If domain stats are enabled (default), also populates domain-specific
    /// reputation by extracting property types from each proof entry.
    pub fn build(self) -> ReputationTracker {
        let mut tracker = ReputationTracker::new(self.config);

        for entry in self.corpus.entries() {
            // Always record aggregate success
            if self.include_domain_stats {
                // Try to extract property type from features and record domain-specific
                if let Some(property_type) =
                    property_type_from_string(&entry.features.property_type)
                {
                    tracker.record_domain_success(entry.backend, property_type, entry.time_taken);
                } else {
                    // Unknown property type - only record aggregate
                    tracker.record_success(entry.backend, entry.time_taken);
                }
            } else {
                tracker.record_success(entry.backend, entry.time_taken);
            }
        }

        tracker
    }

    /// Build and return summary statistics about the bootstrapping
    ///
    /// Returns the tracker along with counts of how many entries were processed
    /// for aggregate vs domain-specific tracking.
    pub fn build_with_stats(self) -> (ReputationTracker, BootstrapStats) {
        let include_domain = self.include_domain_stats;
        let mut tracker = ReputationTracker::new(self.config);
        let mut stats = BootstrapStats::default();

        for entry in self.corpus.entries() {
            stats.total_entries += 1;

            if include_domain {
                if let Some(property_type) =
                    property_type_from_string(&entry.features.property_type)
                {
                    tracker.record_domain_success(entry.backend, property_type, entry.time_taken);
                    stats.domain_entries += 1;
                } else {
                    tracker.record_success(entry.backend, entry.time_taken);
                    stats.aggregate_only_entries += 1;
                }
            } else {
                tracker.record_success(entry.backend, entry.time_taken);
                stats.aggregate_only_entries += 1;
            }
        }

        (tracker, stats)
    }
}

/// Statistics from bootstrapping reputation from a proof corpus
#[derive(Debug, Clone, Default)]
pub struct BootstrapStats {
    /// Total proof entries processed
    pub total_entries: usize,
    /// Entries with recognized property types (recorded with domain stats)
    pub domain_entries: usize,
    /// Entries without recognized property types (aggregate only)
    pub aggregate_only_entries: usize,
}

impl BootstrapStats {
    /// Percentage of entries that had recognized property types
    pub fn domain_coverage(&self) -> f64 {
        if self.total_entries == 0 {
            return 0.0;
        }
        self.domain_entries as f64 / self.total_entries as f64 * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = ReputationConfig::default();
        assert_eq!(config.decay_factor, 0.95);
        assert_eq!(config.min_observations, 5);
        assert_eq!(config.default_reputation, 0.5);
    }

    #[test]
    fn test_backend_stats_record_success() {
        let mut stats = BackendStats::new();
        let decay = 0.95;

        stats.record_success(Duration::from_millis(100), decay);

        assert_eq!(stats.successes, 1);
        assert_eq!(stats.failures, 0);
        assert_eq!(stats.total_observations(), 1);
        assert_eq!(stats.ewma_success_rate, 1.0);
        assert!(stats.last_updated.is_some());
    }

    #[test]
    fn test_backend_stats_record_failure() {
        let mut stats = BackendStats::new();
        let decay = 0.95;

        stats.record_failure(decay);

        assert_eq!(stats.successes, 0);
        assert_eq!(stats.failures, 1);
        assert_eq!(stats.total_observations(), 1);
        assert_eq!(stats.ewma_success_rate, 0.0);
    }

    #[test]
    fn test_backend_stats_mixed() {
        let mut stats = BackendStats::new();
        let decay = 0.95;

        // 3 successes, 2 failures
        stats.record_success(Duration::from_millis(100), decay);
        stats.record_success(Duration::from_millis(200), decay);
        stats.record_failure(decay);
        stats.record_success(Duration::from_millis(150), decay);
        stats.record_failure(decay);

        assert_eq!(stats.successes, 3);
        assert_eq!(stats.failures, 2);
        assert_eq!(stats.simple_success_rate(), 0.6);
        // EWMA will be different due to weighting
        assert!(stats.ewma_success_rate > 0.0 && stats.ewma_success_rate < 1.0);
    }

    #[test]
    fn test_tracker_record_success() {
        let mut tracker = ReputationTracker::new(ReputationConfig::default());

        tracker.record_success(BackendId::Lean4, Duration::from_millis(500));

        assert_eq!(tracker.backend_count(), 1);
        assert_eq!(tracker.total_observations(), 1);
        let stats = tracker.get_stats(&BackendId::Lean4).unwrap();
        assert_eq!(stats.successes, 1);
    }

    #[test]
    fn test_tracker_record_failure() {
        let mut tracker = ReputationTracker::new(ReputationConfig::default());

        tracker.record_failure(BackendId::Alloy);

        assert_eq!(tracker.backend_count(), 1);
        let stats = tracker.get_stats(&BackendId::Alloy).unwrap();
        assert_eq!(stats.failures, 1);
    }

    #[test]
    fn test_compute_reputation_insufficient_data() {
        let tracker = ReputationTracker::new(ReputationConfig::default());

        // No data at all
        let rep = tracker.compute_reputation(&BackendId::Lean4);
        assert_eq!(rep, 0.5); // Default
    }

    #[test]
    fn test_compute_reputation_good_backend() {
        let mut tracker = ReputationTracker::new(ReputationConfig::default());

        // Record 10 successes with fast response times
        for _ in 0..10 {
            tracker.record_success(BackendId::Lean4, Duration::from_millis(100));
        }

        let rep = tracker.compute_reputation(&BackendId::Lean4);
        assert!(
            rep > 0.8,
            "Good backend should have high reputation: {}",
            rep
        );
    }

    #[test]
    fn test_compute_reputation_poor_backend() {
        let mut tracker = ReputationTracker::new(ReputationConfig::default());

        // Record 10 failures
        for _ in 0..10 {
            tracker.record_failure(BackendId::Alloy);
        }

        let rep = tracker.compute_reputation(&BackendId::Alloy);
        assert!(
            rep < 0.2,
            "Poor backend should have low reputation: {}",
            rep
        );
    }

    #[test]
    fn test_compute_weights() {
        let mut tracker = ReputationTracker::new(ReputationConfig::default());

        // Good backend
        for _ in 0..10 {
            tracker.record_success(BackendId::Lean4, Duration::from_millis(100));
        }

        // Mediocre backend
        for _ in 0..5 {
            tracker.record_success(BackendId::Coq, Duration::from_millis(500));
            tracker.record_failure(BackendId::Coq);
        }

        let weights = tracker.compute_weights();

        assert!(weights.contains_key(&BackendId::Lean4));
        assert!(weights.contains_key(&BackendId::Coq));
        assert!(weights[&BackendId::Lean4] > weights[&BackendId::Coq]);
    }

    #[test]
    fn test_reputation_clamping() {
        let mut tracker = ReputationTracker::new(ReputationConfig::default());

        // Many successes
        for _ in 0..100 {
            tracker.record_success(BackendId::Lean4, Duration::from_millis(10));
        }

        let rep = tracker.compute_reputation(&BackendId::Lean4);
        assert!(rep <= 0.99, "Reputation should be clamped to max: {}", rep);

        // Many failures
        for _ in 0..100 {
            tracker.record_failure(BackendId::Alloy);
        }

        let rep = tracker.compute_reputation(&BackendId::Alloy);
        assert!(rep >= 0.01, "Reputation should be clamped to min: {}", rep);
    }

    #[test]
    fn test_tracker_merge() {
        let mut tracker1 = ReputationTracker::new(ReputationConfig::default());
        let mut tracker2 = ReputationTracker::new(ReputationConfig::default());

        // Different observations on each tracker
        for _ in 0..5 {
            tracker1.record_success(BackendId::Lean4, Duration::from_millis(100));
        }
        for _ in 0..5 {
            tracker2.record_success(BackendId::Lean4, Duration::from_millis(200));
        }
        tracker2.record_success(BackendId::Coq, Duration::from_millis(300));

        tracker1.merge(&tracker2);

        assert_eq!(tracker1.backend_count(), 2);
        let lean_stats = tracker1.get_stats(&BackendId::Lean4).unwrap();
        assert_eq!(lean_stats.successes, 10);
    }

    #[test]
    fn test_persistence_roundtrip() {
        let mut tracker = ReputationTracker::new(ReputationConfig::default());

        tracker.record_success(BackendId::Lean4, Duration::from_millis(100));
        tracker.record_failure(BackendId::Alloy);

        let temp_dir = std::env::temp_dir().join("dashprove_reputation_test");
        std::fs::create_dir_all(&temp_dir).unwrap();
        let path = temp_dir.join("reputation.json");

        tracker.save_to_file(&path).unwrap();
        let loaded = ReputationTracker::load_from_file(&path).unwrap();

        assert_eq!(loaded.backend_count(), 2);
        assert_eq!(loaded.get_stats(&BackendId::Lean4).unwrap().successes, 1);
        assert_eq!(loaded.get_stats(&BackendId::Alloy).unwrap().failures, 1);

        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn test_speed_bonus() {
        let mut tracker = ReputationTracker::new(ReputationConfig::default());

        // Fast backend
        for _ in 0..10 {
            tracker.record_success(BackendId::Lean4, Duration::from_millis(50));
        }

        // Slow backend (same success rate, but slower)
        for _ in 0..10 {
            tracker.record_success(BackendId::Isabelle, Duration::from_millis(2000));
        }

        let fast_rep = tracker.compute_reputation(&BackendId::Lean4);
        let slow_rep = tracker.compute_reputation(&BackendId::Isabelle);

        // Fast backend should have slightly higher reputation due to speed bonus
        assert!(
            fast_rep > slow_rep,
            "Fast: {}, Slow: {}",
            fast_rep,
            slow_rep
        );
    }

    #[test]
    fn test_config_recent_focused() {
        let config = ReputationConfig::recent_focused();
        assert!(config.decay_factor < ReputationConfig::default().decay_factor);
        assert!(config.min_observations < ReputationConfig::default().min_observations);
    }

    #[test]
    fn test_config_stable_focused() {
        let config = ReputationConfig::stable_focused();
        assert!(config.decay_factor > ReputationConfig::default().decay_factor);
        assert!(config.min_observations > ReputationConfig::default().min_observations);
    }

    // ==========================================================================
    // Domain-specific tracking tests
    // ==========================================================================

    #[test]
    fn test_domain_key_creation() {
        let key = DomainKey::new(BackendId::Lean4, PropertyType::Theorem);
        assert_eq!(key.backend, BackendId::Lean4);
        assert_eq!(key.property_type, PropertyType::Theorem);
    }

    #[test]
    fn test_domain_key_display() {
        let key = DomainKey::new(BackendId::Lean4, PropertyType::Theorem);
        let display = format!("{}", key);
        assert!(display.contains("Lean4"));
        assert!(display.contains("Theorem"));
    }

    #[test]
    fn test_domain_key_equality() {
        let key1 = DomainKey::new(BackendId::Lean4, PropertyType::Theorem);
        let key2 = DomainKey::new(BackendId::Lean4, PropertyType::Theorem);
        let key3 = DomainKey::new(BackendId::Lean4, PropertyType::Contract);
        let key4 = DomainKey::new(BackendId::Coq, PropertyType::Theorem);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3); // Same backend, different property type
        assert_ne!(key1, key4); // Different backend, same property type
    }

    #[test]
    fn test_domain_success_recording() {
        let mut tracker = ReputationTracker::new(ReputationConfig::default());

        tracker.record_domain_success(
            BackendId::Lean4,
            PropertyType::Theorem,
            Duration::from_millis(100),
        );

        // Check aggregate stats
        assert_eq!(tracker.backend_count(), 1);
        assert_eq!(tracker.total_observations(), 1);

        // Check domain stats
        assert_eq!(tracker.domain_count(), 1);
        let key = DomainKey::new(BackendId::Lean4, PropertyType::Theorem);
        let domain_stats = tracker.get_domain_stats(&key).unwrap();
        assert_eq!(domain_stats.successes, 1);
    }

    #[test]
    fn test_domain_failure_recording() {
        let mut tracker = ReputationTracker::new(ReputationConfig::default());

        tracker.record_domain_failure(BackendId::Alloy, PropertyType::Temporal);

        // Check aggregate stats
        let stats = tracker.get_stats(&BackendId::Alloy).unwrap();
        assert_eq!(stats.failures, 1);

        // Check domain stats
        let key = DomainKey::new(BackendId::Alloy, PropertyType::Temporal);
        let domain_stats = tracker.get_domain_stats(&key).unwrap();
        assert_eq!(domain_stats.failures, 1);
    }

    #[test]
    fn test_domain_reputation_fallback_to_aggregate() {
        let mut tracker = ReputationTracker::new(ReputationConfig::default());

        // Record many aggregate successes for Lean4
        for _ in 0..10 {
            tracker.record_success(BackendId::Lean4, Duration::from_millis(100));
        }

        // Query domain reputation for a property type with no domain data
        let key = DomainKey::new(BackendId::Lean4, PropertyType::Contract);
        let rep = tracker.compute_domain_reputation(&key);

        // Should fall back to aggregate reputation
        let aggregate_rep = tracker.compute_reputation(&BackendId::Lean4);
        assert_eq!(rep, aggregate_rep);
    }

    #[test]
    fn test_domain_reputation_uses_domain_data() {
        let mut tracker = ReputationTracker::new(ReputationConfig::default());

        // Record many domain-specific successes
        for _ in 0..10 {
            tracker.record_domain_success(
                BackendId::Lean4,
                PropertyType::Theorem,
                Duration::from_millis(100),
            );
        }

        // Record failures for a different domain of same backend
        for _ in 0..10 {
            tracker.record_domain_failure(BackendId::Lean4, PropertyType::Contract);
        }

        let theorem_key = DomainKey::new(BackendId::Lean4, PropertyType::Theorem);
        let contract_key = DomainKey::new(BackendId::Lean4, PropertyType::Contract);

        let theorem_rep = tracker.compute_domain_reputation(&theorem_key);
        let contract_rep = tracker.compute_domain_reputation(&contract_key);

        // Theorem should have much better domain reputation
        assert!(
            theorem_rep > contract_rep,
            "Theorem: {}, Contract: {}",
            theorem_rep,
            contract_rep
        );
    }

    #[test]
    fn test_best_backend_for_property_type() {
        let mut tracker = ReputationTracker::new(ReputationConfig::default());

        // Make Lean4 good at Theorems
        for _ in 0..10 {
            tracker.record_domain_success(
                BackendId::Lean4,
                PropertyType::Theorem,
                Duration::from_millis(100),
            );
        }

        // Make Coq mediocre at Theorems
        for _ in 0..5 {
            tracker.record_domain_success(
                BackendId::Coq,
                PropertyType::Theorem,
                Duration::from_millis(200),
            );
            tracker.record_domain_failure(BackendId::Coq, PropertyType::Theorem);
        }

        let best = tracker.best_backend_for_property_type(PropertyType::Theorem);
        assert_eq!(best, Some(BackendId::Lean4));
    }

    #[test]
    fn test_backends_for_property_type() {
        let mut tracker = ReputationTracker::new(ReputationConfig::default());

        tracker.record_domain_success(
            BackendId::Lean4,
            PropertyType::Theorem,
            Duration::from_millis(100),
        );
        tracker.record_domain_success(
            BackendId::Coq,
            PropertyType::Theorem,
            Duration::from_millis(100),
        );
        tracker.record_domain_success(
            BackendId::TlaPlus,
            PropertyType::Temporal,
            Duration::from_millis(100),
        );

        let backends = tracker.backends_for_property_type(PropertyType::Theorem);
        assert_eq!(backends.len(), 2);
        assert!(backends.contains(&BackendId::Lean4));
        assert!(backends.contains(&BackendId::Coq));
    }

    #[test]
    fn test_property_types_for_backend() {
        let mut tracker = ReputationTracker::new(ReputationConfig::default());

        tracker.record_domain_success(
            BackendId::Lean4,
            PropertyType::Theorem,
            Duration::from_millis(100),
        );
        tracker.record_domain_success(
            BackendId::Lean4,
            PropertyType::Contract,
            Duration::from_millis(100),
        );

        let prop_types = tracker.property_types_for_backend(BackendId::Lean4);
        assert_eq!(prop_types.len(), 2);
        assert!(prop_types.contains(&PropertyType::Theorem));
        assert!(prop_types.contains(&PropertyType::Contract));
    }

    #[test]
    fn test_domain_summary() {
        let mut tracker = ReputationTracker::new(ReputationConfig::default());

        for _ in 0..10 {
            tracker.record_domain_success(
                BackendId::Lean4,
                PropertyType::Theorem,
                Duration::from_millis(100),
            );
        }
        for _ in 0..5 {
            tracker.record_domain_failure(BackendId::Lean4, PropertyType::Theorem);
        }

        let summaries = tracker.domain_summary();
        assert_eq!(summaries.len(), 1);

        let summary = &summaries[0];
        assert_eq!(summary.backend, BackendId::Lean4);
        assert_eq!(summary.property_type, PropertyType::Theorem);
        assert_eq!(summary.successes, 10);
        assert_eq!(summary.failures, 5);
        assert!((summary.success_rate - 0.667).abs() < 0.01);
    }

    #[test]
    fn test_domain_merge() {
        let mut tracker1 = ReputationTracker::new(ReputationConfig::default());
        let mut tracker2 = ReputationTracker::new(ReputationConfig::default());

        // Add observations to tracker1
        for _ in 0..5 {
            tracker1.record_domain_success(
                BackendId::Lean4,
                PropertyType::Theorem,
                Duration::from_millis(100),
            );
        }

        // Add observations to tracker2
        for _ in 0..5 {
            tracker2.record_domain_success(
                BackendId::Lean4,
                PropertyType::Theorem,
                Duration::from_millis(200),
            );
        }
        tracker2.record_domain_success(
            BackendId::Coq,
            PropertyType::Contract,
            Duration::from_millis(300),
        );

        tracker1.merge(&tracker2);

        assert_eq!(tracker1.domain_count(), 2);

        let lean_key = DomainKey::new(BackendId::Lean4, PropertyType::Theorem);
        let lean_stats = tracker1.get_domain_stats(&lean_key).unwrap();
        assert_eq!(lean_stats.successes, 10);
    }

    #[test]
    fn test_domain_count_and_observations() {
        let mut tracker = ReputationTracker::new(ReputationConfig::default());

        tracker.record_domain_success(
            BackendId::Lean4,
            PropertyType::Theorem,
            Duration::from_millis(100),
        );
        tracker.record_domain_success(
            BackendId::Coq,
            PropertyType::Contract,
            Duration::from_millis(100),
        );
        tracker.record_domain_failure(BackendId::Lean4, PropertyType::Theorem);

        assert_eq!(tracker.domain_count(), 2);
        assert_eq!(tracker.total_domain_observations(), 3);
    }

    #[test]
    fn test_domain_weights() {
        let mut tracker = ReputationTracker::new(ReputationConfig::default());

        for _ in 0..10 {
            tracker.record_domain_success(
                BackendId::Lean4,
                PropertyType::Theorem,
                Duration::from_millis(100),
            );
        }
        for _ in 0..10 {
            tracker.record_domain_failure(BackendId::Alloy, PropertyType::Temporal);
        }

        let weights = tracker.compute_domain_weights();

        let lean_key = DomainKey::new(BackendId::Lean4, PropertyType::Theorem);
        let alloy_key = DomainKey::new(BackendId::Alloy, PropertyType::Temporal);

        assert!(weights[&lean_key] > weights[&alloy_key]);
    }

    // ==========================================================================
    // Property type string conversion tests
    // ==========================================================================

    #[test]
    fn test_property_type_from_string_theorem() {
        assert_eq!(
            property_type_from_string("theorem"),
            Some(PropertyType::Theorem)
        );
        assert_eq!(
            property_type_from_string("Theorem"),
            Some(PropertyType::Theorem)
        );
        assert_eq!(
            property_type_from_string("THEOREM"),
            Some(PropertyType::Theorem)
        );
    }

    #[test]
    fn test_property_type_from_string_contract() {
        assert_eq!(
            property_type_from_string("contract"),
            Some(PropertyType::Contract)
        );
    }

    #[test]
    fn test_property_type_from_string_invariant() {
        assert_eq!(
            property_type_from_string("invariant"),
            Some(PropertyType::Invariant)
        );
    }

    #[test]
    fn test_property_type_from_string_temporal() {
        assert_eq!(
            property_type_from_string("temporal"),
            Some(PropertyType::Temporal)
        );
    }

    #[test]
    fn test_property_type_from_string_unknown() {
        assert_eq!(property_type_from_string("unknown_type"), None);
        assert_eq!(property_type_from_string(""), None);
        assert_eq!(property_type_from_string("foo"), None);
    }

    #[test]
    fn test_property_type_from_string_all_variants() {
        // Test all supported string variants
        let cases = vec![
            ("theorem", PropertyType::Theorem),
            ("contract", PropertyType::Contract),
            ("invariant", PropertyType::Invariant),
            ("refinement", PropertyType::Refinement),
            ("temporal", PropertyType::Temporal),
            ("probabilistic", PropertyType::Probabilistic),
            ("security", PropertyType::SecurityProtocol),
            ("semantic", PropertyType::Theorem),
            ("platform_api", PropertyType::PlatformApi),
            ("memory_safety", PropertyType::MemorySafety),
            ("data_race", PropertyType::DataRace),
            ("fuzzing", PropertyType::Fuzzing),
            ("property_based", PropertyType::PropertyBased),
            ("lint", PropertyType::Lint),
            ("neural_robustness", PropertyType::NeuralRobustness),
        ];

        for (input, expected) in cases {
            assert_eq!(
                property_type_from_string(input),
                Some(expected),
                "Failed for input: {}",
                input
            );
        }
    }

    // ==========================================================================
    // ReputationFromCorpus domain bootstrapping tests
    // ==========================================================================

    #[test]
    fn test_reputation_from_corpus_with_domain_stats() {
        use crate::corpus::ProofCorpus;
        use crate::LearnableResult;
        use dashprove_usl::ast::{Invariant, Property};

        let mut corpus = ProofCorpus::new();

        // Add a proof with known property type
        let result = LearnableResult {
            property: Property::Invariant(Invariant {
                name: "test_inv".to_string(),
                body: dashprove_usl::ast::Expr::Bool(true),
            }),
            backend: BackendId::Lean4,
            status: dashprove_backends::VerificationStatus::Proven,
            tactics: vec!["decide".to_string()],
            time_taken: Duration::from_millis(100),
            proof_output: None,
        };
        corpus.insert(&result);

        // Build reputation with domain stats
        let tracker = ReputationFromCorpus::new(&corpus).build();

        // Should have both aggregate and domain-specific stats
        assert_eq!(tracker.backend_count(), 1);
        assert_eq!(tracker.total_observations(), 1);

        // Domain stats should be populated since "invariant" is a known type
        assert_eq!(tracker.domain_count(), 1);
        let key = DomainKey::new(BackendId::Lean4, PropertyType::Invariant);
        assert!(tracker.get_domain_stats(&key).is_some());
    }

    #[test]
    fn test_reputation_from_corpus_without_domain_stats() {
        use crate::corpus::ProofCorpus;
        use crate::LearnableResult;
        use dashprove_usl::ast::{Invariant, Property};

        let mut corpus = ProofCorpus::new();

        let result = LearnableResult {
            property: Property::Invariant(Invariant {
                name: "test_inv".to_string(),
                body: dashprove_usl::ast::Expr::Bool(true),
            }),
            backend: BackendId::Lean4,
            status: dashprove_backends::VerificationStatus::Proven,
            tactics: vec![],
            time_taken: Duration::from_millis(100),
            proof_output: None,
        };
        corpus.insert(&result);

        // Build reputation WITHOUT domain stats
        let tracker = ReputationFromCorpus::new(&corpus)
            .with_domain_stats(false)
            .build();

        // Should have only aggregate stats
        assert_eq!(tracker.backend_count(), 1);
        assert_eq!(tracker.total_observations(), 1);
        assert_eq!(tracker.domain_count(), 0); // No domain stats
    }

    #[test]
    fn test_reputation_from_corpus_build_with_stats() {
        use crate::corpus::ProofCorpus;
        use crate::LearnableResult;
        use dashprove_usl::ast::{Invariant, Property, Theorem};

        let mut corpus = ProofCorpus::new();

        // Add proofs with known property types
        for i in 0..5 {
            let result = LearnableResult {
                property: Property::Theorem(Theorem {
                    name: format!("thm_{}", i),
                    body: dashprove_usl::ast::Expr::Bool(true),
                }),
                backend: BackendId::Coq,
                status: dashprove_backends::VerificationStatus::Proven,
                tactics: vec![],
                time_taken: Duration::from_millis(200),
                proof_output: None,
            };
            corpus.insert(&result);
        }

        for i in 0..3 {
            let result = LearnableResult {
                property: Property::Invariant(Invariant {
                    name: format!("inv_{}", i),
                    body: dashprove_usl::ast::Expr::Bool(true),
                }),
                backend: BackendId::TlaPlus,
                status: dashprove_backends::VerificationStatus::Proven,
                tactics: vec![],
                time_taken: Duration::from_millis(300),
                proof_output: None,
            };
            corpus.insert(&result);
        }

        let (tracker, stats) = ReputationFromCorpus::new(&corpus).build_with_stats();

        assert_eq!(stats.total_entries, 8);
        assert_eq!(stats.domain_entries, 8); // All should have known types
        assert_eq!(stats.aggregate_only_entries, 0);
        assert!((stats.domain_coverage() - 100.0).abs() < 0.01);

        // Verify tracker has correct domain stats
        assert_eq!(tracker.domain_count(), 2); // theorem + invariant
        assert_eq!(tracker.backend_count(), 2); // Coq + TlaPlus
    }

    #[test]
    fn test_bootstrap_stats_domain_coverage() {
        let stats = BootstrapStats {
            total_entries: 100,
            domain_entries: 75,
            aggregate_only_entries: 25,
        };

        assert!((stats.domain_coverage() - 75.0).abs() < 0.01);
    }

    #[test]
    fn test_bootstrap_stats_domain_coverage_empty() {
        let stats = BootstrapStats::default();
        assert_eq!(stats.domain_coverage(), 0.0);
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    #[kani::proof]
    #[kani::unwind(3)]
    fn kani_ewma_bounded() {
        let decay: f64 = kani::any();
        kani::assume(decay >= 0.0 && decay <= 1.0);

        let mut stats = BackendStats::new();

        // Record some observations
        stats.record_success(Duration::from_millis(100), decay);
        stats.record_failure(decay);

        // EWMA should always be in [0, 1]
        assert!(stats.ewma_success_rate >= 0.0);
        assert!(stats.ewma_success_rate <= 1.0);
    }

    #[kani::proof]
    #[kani::unwind(3)]
    fn kani_reputation_bounded() {
        let config = ReputationConfig::default();
        let mut tracker = ReputationTracker::new(config);

        // Record some observations
        tracker.record_success(BackendId::Lean4, Duration::from_millis(100));
        tracker.record_failure(BackendId::Lean4);
        tracker.record_success(BackendId::Lean4, Duration::from_millis(100));

        let rep = tracker.compute_reputation(&BackendId::Lean4);

        // Reputation should always be bounded
        assert!(rep >= 0.01);
        assert!(rep <= 0.99);
    }

    #[kani::proof]
    fn kani_default_for_unknown_backend() {
        let tracker = ReputationTracker::new(ReputationConfig::default());

        // Unknown backend should get default reputation
        let rep = tracker.compute_reputation(&BackendId::Lean4);
        assert_eq!(rep, 0.5);
    }

    #[kani::proof]
    #[kani::unwind(3)]
    fn kani_total_observations_consistent() {
        let mut tracker = ReputationTracker::new(ReputationConfig::default());

        tracker.record_success(BackendId::Lean4, Duration::from_millis(100));
        tracker.record_failure(BackendId::Coq);

        let lean_obs = tracker
            .get_stats(&BackendId::Lean4)
            .map_or(0, |s| s.total_observations());
        let coq_obs = tracker
            .get_stats(&BackendId::Coq)
            .map_or(0, |s| s.total_observations());

        // Total should match sum of individual observations
        assert_eq!(lean_obs + coq_obs, 2);
    }

    #[kani::proof]
    fn kani_simple_success_rate_bounded() {
        let mut stats = BackendStats::new();

        let successes: usize = kani::any();
        let failures: usize = kani::any();

        kani::assume(successes <= 100);
        kani::assume(failures <= 100);
        kani::assume(successes + failures > 0);

        stats.successes = successes;
        stats.failures = failures;

        let rate = stats.simple_success_rate();

        assert!(rate >= 0.0);
        assert!(rate <= 1.0);
    }
}
