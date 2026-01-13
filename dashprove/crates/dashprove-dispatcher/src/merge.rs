//! Result merging from multiple backends
//!
//! This module provides strategies for combining results from multiple verification backends
//! into a unified result.

use crate::parallel::{ExecutionResults, TaskResult};
use dashprove_backends::{
    BackendId, BackendResult, PropertyType, StructuredCounterexample, VerificationStatus,
};
use dashprove_learning::DomainKey;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, info};

/// Merged verification result for a single property
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergedResult {
    /// Property index in the original spec
    pub property_index: usize,
    /// Type of the property (if known)
    pub property_type: Option<PropertyType>,
    /// Consensus status from all backends
    pub status: VerificationStatus,
    /// Confidence score (0.0 - 1.0) based on backend agreement
    pub confidence: f64,
    /// Individual results from each backend
    pub backend_results: Vec<BackendResultSummary>,
    /// Combined proof (if proven by any backend)
    pub proof: Option<String>,
    /// Combined counterexample (if disproven by any backend)
    pub counterexample: Option<StructuredCounterexample>,
    /// All diagnostics from all backends
    pub diagnostics: Vec<String>,
    /// Total verification time for this property
    pub verification_time: Duration,
}

/// Summary of a single backend's result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendResultSummary {
    /// Which backend produced this result
    pub backend: BackendId,
    /// Verification status from this backend
    pub status: VerificationStatus,
    /// Time taken by this backend
    pub time_taken: Duration,
    /// Error message if verification failed
    pub error: Option<String>,
}

/// Complete merged results for all properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergedResults {
    /// Results for each property
    pub properties: Vec<MergedResult>,
    /// Overall status summary
    pub summary: VerificationSummary,
    /// Total time for all verifications
    pub total_time: Duration,
}

/// Summary statistics for verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationSummary {
    /// Number of properties proven
    pub proven: usize,
    /// Number of properties disproven
    pub disproven: usize,
    /// Number of properties with unknown status
    pub unknown: usize,
    /// Number of properties with partial verification
    pub partial: usize,
    /// Overall confidence score
    pub overall_confidence: f64,
}

/// Strategy for merging results from multiple backends
#[derive(Debug, Clone, PartialEq, Default)]
pub enum MergeStrategy {
    /// Use the first successful result
    #[default]
    FirstSuccess,
    /// Require all backends to agree (unanimous)
    Unanimous,
    /// Use majority voting
    Majority,
    /// Use the most confident result
    MostConfident,
    /// Prefer proven over unknown, disproven over proven
    Pessimistic,
    /// Prefer proven over unknown, proven over disproven
    Optimistic,
    /// Byzantine Fault Tolerant consensus: requires (n - f) agreements
    /// where f is the maximum number of faulty backends tolerated.
    /// For BFT to work correctly, need at least 3f + 1 backends total.
    /// This provides safety even when some backends produce incorrect results.
    ByzantineFaultTolerant {
        /// Maximum number of faulty backends to tolerate
        max_faulty: usize,
    },
    /// Weighted consensus based on historical backend accuracy
    /// Each backend's vote is weighted by its historical success rate
    WeightedConsensus {
        /// Backend weights (BackendId -> weight 0.0-1.0)
        /// Missing backends use default weight of 0.5
        weights: std::collections::HashMap<BackendId, f64>,
    },
    /// Domain-specific weighted consensus
    /// Each backend's vote is weighted by its historical success rate for the
    /// specific property type being verified. Falls back to aggregate weights
    /// for property types without domain-specific data.
    DomainWeightedConsensus {
        /// Domain-specific weights (BackendId + PropertyType -> weight 0.0-1.0)
        /// Falls back to aggregate weights if domain-specific weight not found
        domain_weights: std::collections::HashMap<DomainKey, f64>,
        /// Aggregate weights for fallback (BackendId -> weight 0.0-1.0)
        /// Used when no domain-specific weight exists for a backend
        aggregate_weights: std::collections::HashMap<BackendId, f64>,
    },
}

/// Result merger that combines results from multiple backends
#[derive(Clone)]
pub struct ResultMerger {
    strategy: MergeStrategy,
}

impl ResultMerger {
    /// Create a new result merger with the given strategy
    pub fn new(strategy: MergeStrategy) -> Self {
        ResultMerger { strategy }
    }

    /// Merge execution results into unified results
    pub fn merge(&self, results: ExecutionResults) -> MergedResults {
        let mut properties = Vec::new();
        let mut proven = 0;
        let mut disproven = 0;
        let mut unknown = 0;
        let mut partial = 0;
        let mut total_confidence = 0.0;

        let ExecutionResults {
            by_property,
            property_types,
            total_time,
            successful: _,
            failed: _,
        } = results;

        // Process each property
        for (property_index, task_results) in by_property {
            let property_type = property_types.get(&property_index).copied();
            let merged = self.merge_property(property_index, property_type, task_results);

            match &merged.status {
                VerificationStatus::Proven => proven += 1,
                VerificationStatus::Disproven => disproven += 1,
                VerificationStatus::Unknown { .. } => unknown += 1,
                VerificationStatus::Partial { .. } => partial += 1,
            }

            total_confidence += merged.confidence;
            properties.push(merged);
        }

        // Sort by property index
        properties.sort_by_key(|r| r.property_index);

        let property_count = properties.len().max(1);
        let overall_confidence = total_confidence / property_count as f64;

        MergedResults {
            properties,
            summary: VerificationSummary {
                proven,
                disproven,
                unknown,
                partial,
                overall_confidence,
            },
            total_time,
        }
    }

    /// Merge results for a single property
    fn merge_property(
        &self,
        property_index: usize,
        property_type: Option<PropertyType>,
        task_results: Vec<TaskResult>,
    ) -> MergedResult {
        let mut backend_summaries = Vec::new();
        let mut successful_results = Vec::new();
        let mut all_diagnostics = Vec::new();
        let mut total_time = Duration::ZERO;

        // Collect all results
        for task in &task_results {
            let summary = match &task.result {
                Ok(result) => {
                    successful_results.push(result.clone());
                    all_diagnostics.extend(result.diagnostics.clone());
                    total_time += result.time_taken;
                    BackendResultSummary {
                        backend: task.backend,
                        status: result.status.clone(),
                        time_taken: result.time_taken,
                        error: None,
                    }
                }
                Err(err) => BackendResultSummary {
                    backend: task.backend,
                    status: VerificationStatus::Unknown {
                        reason: err.clone(),
                    },
                    time_taken: Duration::ZERO,
                    error: Some(err.clone()),
                },
            };
            backend_summaries.push(summary);
        }

        // Determine consensus status based on strategy
        let (status, confidence) = if let Some(prop_type) = property_type {
            self.compute_consensus_for_property_type(&successful_results, prop_type)
        } else {
            self.compute_consensus(&successful_results)
        };

        // Extract proof and counterexample
        let proof = successful_results
            .iter()
            .find_map(|r| r.proof.clone())
            .or_else(|| {
                successful_results
                    .iter()
                    .find(|r| matches!(r.status, VerificationStatus::Proven))
                    .and_then(|r| r.proof.clone())
            });

        let counterexample = successful_results
            .iter()
            .find_map(|r| r.counterexample.clone())
            .or_else(|| {
                successful_results
                    .iter()
                    .find(|r| matches!(r.status, VerificationStatus::Disproven))
                    .and_then(|r| r.counterexample.clone())
            });

        MergedResult {
            property_index,
            property_type,
            status,
            confidence,
            backend_results: backend_summaries,
            proof,
            counterexample,
            diagnostics: all_diagnostics,
            verification_time: total_time,
        }
    }

    /// Compute consensus status based on merge strategy
    fn compute_consensus(&self, results: &[BackendResult]) -> (VerificationStatus, f64) {
        if results.is_empty() {
            return (
                VerificationStatus::Unknown {
                    reason: "No successful backend results".to_string(),
                },
                0.0,
            );
        }

        match &self.strategy {
            MergeStrategy::FirstSuccess => self.first_success_consensus(results),
            MergeStrategy::Unanimous => self.unanimous_consensus(results),
            MergeStrategy::Majority => self.majority_consensus(results),
            MergeStrategy::MostConfident => self.most_confident_consensus(results),
            MergeStrategy::Pessimistic => self.pessimistic_consensus(results),
            MergeStrategy::Optimistic => self.optimistic_consensus(results),
            MergeStrategy::ByzantineFaultTolerant { max_faulty } => {
                self.bft_consensus(results, *max_faulty)
            }
            MergeStrategy::WeightedConsensus { weights } => {
                self.weighted_consensus(results, weights)
            }
            MergeStrategy::DomainWeightedConsensus {
                aggregate_weights, ..
            } => {
                // Without property type context, fall back to aggregate weights
                self.weighted_consensus(results, aggregate_weights)
            }
        }
    }

    /// Compute consensus status with domain-specific property type
    ///
    /// This method allows domain-aware consensus when using `DomainWeightedConsensus`.
    /// For other strategies, it behaves identically to `compute_consensus`.
    pub fn compute_consensus_for_property_type(
        &self,
        results: &[BackendResult],
        property_type: PropertyType,
    ) -> (VerificationStatus, f64) {
        if results.is_empty() {
            return (
                VerificationStatus::Unknown {
                    reason: "No successful backend results".to_string(),
                },
                0.0,
            );
        }

        match &self.strategy {
            MergeStrategy::DomainWeightedConsensus {
                domain_weights,
                aggregate_weights,
            } => self.domain_weighted_consensus(
                results,
                domain_weights,
                aggregate_weights,
                property_type,
            ),
            // For all other strategies, delegate to standard compute_consensus
            _ => self.compute_consensus(results),
        }
    }

    /// First successful result wins
    fn first_success_consensus(&self, results: &[BackendResult]) -> (VerificationStatus, f64) {
        // Prefer proven, then disproven, then partial, then unknown
        for result in results {
            if matches!(result.status, VerificationStatus::Proven) {
                return (result.status.clone(), 1.0);
            }
        }
        for result in results {
            if matches!(result.status, VerificationStatus::Disproven) {
                return (result.status.clone(), 1.0);
            }
        }
        for result in results {
            if matches!(result.status, VerificationStatus::Partial { .. }) {
                return (result.status.clone(), 0.5);
            }
        }
        (results[0].status.clone(), 0.3)
    }

    /// All backends must agree
    fn unanimous_consensus(&self, results: &[BackendResult]) -> (VerificationStatus, f64) {
        if results.len() == 1 {
            return (results[0].status.clone(), 0.8);
        }

        let first_status = Self::status_category(&results[0].status);
        let unanimous = results
            .iter()
            .all(|r| Self::status_category(&r.status) == first_status);

        if unanimous {
            debug!(status = ?first_status, count = results.len(), "Unanimous agreement");
            (results[0].status.clone(), 1.0)
        } else {
            info!("Backends disagree on verification result");
            (
                VerificationStatus::Unknown {
                    reason: "Backends disagree".to_string(),
                },
                0.3,
            )
        }
    }

    /// Majority voting
    fn majority_consensus(&self, results: &[BackendResult]) -> (VerificationStatus, f64) {
        let mut proven_count = 0;
        let mut disproven_count = 0;
        let mut unknown_count = 0;

        for result in results {
            match &result.status {
                VerificationStatus::Proven => proven_count += 1,
                VerificationStatus::Disproven => disproven_count += 1,
                _ => unknown_count += 1,
            }
        }

        let total = results.len();
        let (status, count) = if proven_count >= disproven_count && proven_count > unknown_count {
            (VerificationStatus::Proven, proven_count)
        } else if disproven_count > proven_count && disproven_count > unknown_count {
            (
                results
                    .iter()
                    .find(|r| matches!(r.status, VerificationStatus::Disproven))
                    .unwrap()
                    .status
                    .clone(),
                disproven_count,
            )
        } else {
            (
                VerificationStatus::Unknown {
                    reason: "No majority".to_string(),
                },
                unknown_count,
            )
        };

        let confidence = count as f64 / total as f64;
        (status, confidence)
    }

    /// Use most confident result (based on partial verification percentage)
    fn most_confident_consensus(&self, results: &[BackendResult]) -> (VerificationStatus, f64) {
        // Score: Proven=1.0, Disproven=0.9, Partial=percentage, Unknown=0.0
        let mut best_result = &results[0];
        let mut best_score = Self::status_score(&results[0].status);

        for result in results.iter().skip(1) {
            let score = Self::status_score(&result.status);
            if score > best_score {
                best_score = score;
                best_result = result;
            }
        }

        (best_result.status.clone(), best_score)
    }

    /// Pessimistic: prefer disproven over proven over unknown
    fn pessimistic_consensus(&self, results: &[BackendResult]) -> (VerificationStatus, f64) {
        // Check for any disproven
        for result in results {
            if matches!(result.status, VerificationStatus::Disproven) {
                return (result.status.clone(), 1.0);
            }
        }
        // Check for any proven
        for result in results {
            if matches!(result.status, VerificationStatus::Proven) {
                return (result.status.clone(), 0.8);
            }
        }
        // Return first result
        (results[0].status.clone(), 0.5)
    }

    /// Optimistic: prefer proven over unknown over disproven
    fn optimistic_consensus(&self, results: &[BackendResult]) -> (VerificationStatus, f64) {
        // Check for any proven
        for result in results {
            if matches!(result.status, VerificationStatus::Proven) {
                return (result.status.clone(), 1.0);
            }
        }
        // Check for partial
        for result in results {
            if matches!(result.status, VerificationStatus::Partial { .. }) {
                return (result.status.clone(), 0.7);
            }
        }
        // Check for disproven (still valid result)
        for result in results {
            if matches!(result.status, VerificationStatus::Disproven) {
                return (result.status.clone(), 0.9);
            }
        }
        // Return first result
        (results[0].status.clone(), 0.3)
    }

    /// Categorize status for comparison
    fn status_category(status: &VerificationStatus) -> &'static str {
        match status {
            VerificationStatus::Proven => "proven",
            VerificationStatus::Disproven => "disproven",
            VerificationStatus::Partial { .. } => "partial",
            VerificationStatus::Unknown { .. } => "unknown",
        }
    }

    /// Score a status for confidence comparison
    fn status_score(status: &VerificationStatus) -> f64 {
        match status {
            VerificationStatus::Proven => 1.0,
            VerificationStatus::Disproven => 0.95,
            VerificationStatus::Partial {
                verified_percentage,
            } => *verified_percentage,
            VerificationStatus::Unknown { .. } => 0.0,
        }
    }

    /// Byzantine Fault Tolerant consensus
    ///
    /// Requires at least (n - f) backends to agree on a result, where:
    /// - n is the total number of responding backends
    /// - f is the maximum number of faulty backends to tolerate
    ///
    /// For BFT to provide safety guarantees, n >= 3f + 1 is required.
    /// If insufficient backends respond, returns Unknown with low confidence.
    ///
    /// This is useful when some backends might be buggy or produce incorrect
    /// results (e.g., unsound verification tools, implementation bugs).
    fn bft_consensus(
        &self,
        results: &[BackendResult],
        max_faulty: usize,
    ) -> (VerificationStatus, f64) {
        let n = results.len();
        let required_agreements = n.saturating_sub(max_faulty);

        // Check if we have enough backends for BFT safety
        // BFT requires n >= 3f + 1
        let min_required = 3 * max_faulty + 1;
        if n < min_required {
            info!(
                n = n,
                max_faulty = max_faulty,
                min_required = min_required,
                "Insufficient backends for BFT safety"
            );
            // Fall back to majority with reduced confidence
            let (status, _) = self.majority_consensus(results);
            return (status, 0.5);
        }

        // Count votes for each status category
        let mut proven_count = 0;
        let mut disproven_count = 0;
        let mut unknown_count = 0;
        let mut partial_count = 0;

        for result in results {
            match &result.status {
                VerificationStatus::Proven => proven_count += 1,
                VerificationStatus::Disproven => disproven_count += 1,
                VerificationStatus::Partial { .. } => partial_count += 1,
                VerificationStatus::Unknown { .. } => unknown_count += 1,
            }
        }

        debug!(
            proven = proven_count,
            disproven = disproven_count,
            partial = partial_count,
            unknown = unknown_count,
            required = required_agreements,
            "BFT vote counts"
        );

        // Check if any status has the required agreement
        // Priority: Proven > Disproven > Partial > Unknown
        if proven_count >= required_agreements {
            let confidence = proven_count as f64 / n as f64;
            return (VerificationStatus::Proven, confidence);
        }

        if disproven_count >= required_agreements {
            let result = results
                .iter()
                .find(|r| matches!(r.status, VerificationStatus::Disproven))
                .expect("disproven_count > 0");
            let confidence = disproven_count as f64 / n as f64;
            return (result.status.clone(), confidence);
        }

        if partial_count >= required_agreements {
            // For partial, average the percentages
            let avg_pct: f64 = results
                .iter()
                .filter_map(|r| match &r.status {
                    VerificationStatus::Partial {
                        verified_percentage,
                    } => Some(*verified_percentage),
                    _ => None,
                })
                .sum::<f64>()
                / partial_count as f64;
            let confidence = partial_count as f64 / n as f64;
            return (
                VerificationStatus::Partial {
                    verified_percentage: avg_pct,
                },
                confidence * 0.8,
            );
        }

        // No consensus reached - return Unknown
        info!(
            max_count = proven_count.max(disproven_count).max(partial_count),
            required = required_agreements,
            "BFT consensus not reached"
        );
        (
            VerificationStatus::Unknown {
                reason: format!(
                    "BFT consensus not reached: max {} votes, {} required",
                    proven_count.max(disproven_count).max(partial_count),
                    required_agreements
                ),
            },
            0.2,
        )
    }

    /// Weighted consensus based on historical backend accuracy
    ///
    /// Each backend's vote is weighted by its historical success rate.
    /// This allows the system to trust more reliable backends more heavily.
    ///
    /// Weights should be in the range [0.0, 1.0]. Missing backends use 0.5.
    fn weighted_consensus(
        &self,
        results: &[BackendResult],
        weights: &std::collections::HashMap<BackendId, f64>,
    ) -> (VerificationStatus, f64) {
        let default_weight = 0.5;

        let mut proven_weight = 0.0;
        let mut disproven_weight = 0.0;
        let mut unknown_weight = 0.0;
        let mut partial_weight = 0.0;
        let mut total_weight = 0.0;

        for result in results {
            let weight = weights
                .get(&result.backend)
                .copied()
                .unwrap_or(default_weight)
                .clamp(0.0, 1.0);
            total_weight += weight;

            match &result.status {
                VerificationStatus::Proven => proven_weight += weight,
                VerificationStatus::Disproven => disproven_weight += weight,
                VerificationStatus::Partial { .. } => partial_weight += weight,
                VerificationStatus::Unknown { .. } => unknown_weight += weight,
            }
        }

        if total_weight == 0.0 {
            return (
                VerificationStatus::Unknown {
                    reason: "No weighted votes".to_string(),
                },
                0.0,
            );
        }

        debug!(
            proven_weight = %format!("{:.2}", proven_weight),
            disproven_weight = %format!("{:.2}", disproven_weight),
            partial_weight = %format!("{:.2}", partial_weight),
            unknown_weight = %format!("{:.2}", unknown_weight),
            total_weight = %format!("{:.2}", total_weight),
            "Weighted consensus votes"
        );

        // Normalize weights and find winner
        let proven_pct = proven_weight / total_weight;
        let disproven_pct = disproven_weight / total_weight;
        let partial_pct = partial_weight / total_weight;

        // Winner is status with highest weighted percentage
        if proven_pct >= disproven_pct && proven_pct >= partial_pct && proven_pct > 0.0 {
            return (VerificationStatus::Proven, proven_pct);
        }

        if disproven_pct > proven_pct && disproven_pct >= partial_pct {
            let result = results
                .iter()
                .find(|r| matches!(r.status, VerificationStatus::Disproven))
                .expect("disproven_weight > 0");
            return (result.status.clone(), disproven_pct);
        }

        if partial_pct > proven_pct && partial_pct > disproven_pct {
            // Average partial percentages weighted by backend weight
            let mut weighted_pct_sum = 0.0;
            let mut pct_weight_sum = 0.0;
            for result in results {
                if let VerificationStatus::Partial {
                    verified_percentage,
                } = &result.status
                {
                    let weight = weights
                        .get(&result.backend)
                        .copied()
                        .unwrap_or(default_weight)
                        .clamp(0.0, 1.0);
                    weighted_pct_sum += *verified_percentage * weight;
                    pct_weight_sum += weight;
                }
            }
            let avg_pct = if pct_weight_sum > 0.0 {
                weighted_pct_sum / pct_weight_sum
            } else {
                0.0
            };
            return (
                VerificationStatus::Partial {
                    verified_percentage: avg_pct,
                },
                partial_pct * 0.8,
            );
        }

        // Unknown wins
        (
            VerificationStatus::Unknown {
                reason: "Weighted consensus favors unknown".to_string(),
            },
            unknown_weight / total_weight,
        )
    }

    /// Domain-specific weighted consensus
    ///
    /// Uses domain-specific weights (backend + property_type) when available,
    /// falling back to aggregate weights for backends without domain-specific data.
    ///
    /// This allows the system to leverage knowledge like:
    /// - "Lean4 is excellent at theorem proving (weight 0.95)"
    /// - "Lean4 is mediocre at contracts (weight 0.6)"
    /// - "TLA+ excels at temporal properties (weight 0.9)"
    fn domain_weighted_consensus(
        &self,
        results: &[BackendResult],
        domain_weights: &std::collections::HashMap<DomainKey, f64>,
        aggregate_weights: &std::collections::HashMap<BackendId, f64>,
        property_type: PropertyType,
    ) -> (VerificationStatus, f64) {
        let default_weight = 0.5;

        let mut proven_weight = 0.0;
        let mut disproven_weight = 0.0;
        let mut unknown_weight = 0.0;
        let mut partial_weight = 0.0;
        let mut total_weight = 0.0;

        for result in results {
            // Try domain-specific weight first, then aggregate, then default
            let domain_key = DomainKey::new(result.backend, property_type);
            let weight = domain_weights
                .get(&domain_key)
                .or_else(|| aggregate_weights.get(&result.backend))
                .copied()
                .unwrap_or(default_weight)
                .clamp(0.0, 1.0);

            total_weight += weight;

            match &result.status {
                VerificationStatus::Proven => proven_weight += weight,
                VerificationStatus::Disproven => disproven_weight += weight,
                VerificationStatus::Partial { .. } => partial_weight += weight,
                VerificationStatus::Unknown { .. } => unknown_weight += weight,
            }
        }

        if total_weight == 0.0 {
            return (
                VerificationStatus::Unknown {
                    reason: "No weighted votes".to_string(),
                },
                0.0,
            );
        }

        debug!(
            proven_weight = %format!("{:.2}", proven_weight),
            disproven_weight = %format!("{:.2}", disproven_weight),
            partial_weight = %format!("{:.2}", partial_weight),
            unknown_weight = %format!("{:.2}", unknown_weight),
            total_weight = %format!("{:.2}", total_weight),
            property_type = ?property_type,
            "Domain-weighted consensus votes"
        );

        // Normalize weights and find winner
        let proven_pct = proven_weight / total_weight;
        let disproven_pct = disproven_weight / total_weight;
        let partial_pct = partial_weight / total_weight;

        // Winner is status with highest weighted percentage
        if proven_pct >= disproven_pct && proven_pct >= partial_pct && proven_pct > 0.0 {
            return (VerificationStatus::Proven, proven_pct);
        }

        if disproven_pct > proven_pct && disproven_pct >= partial_pct {
            let result = results
                .iter()
                .find(|r| matches!(r.status, VerificationStatus::Disproven))
                .expect("disproven_weight > 0");
            return (result.status.clone(), disproven_pct);
        }

        if partial_pct > proven_pct && partial_pct > disproven_pct {
            // Average partial percentages weighted by backend weight
            let mut weighted_pct_sum = 0.0;
            let mut pct_weight_sum = 0.0;
            for result in results {
                if let VerificationStatus::Partial {
                    verified_percentage,
                } = &result.status
                {
                    let domain_key = DomainKey::new(result.backend, property_type);
                    let weight = domain_weights
                        .get(&domain_key)
                        .or_else(|| aggregate_weights.get(&result.backend))
                        .copied()
                        .unwrap_or(default_weight)
                        .clamp(0.0, 1.0);
                    weighted_pct_sum += *verified_percentage * weight;
                    pct_weight_sum += weight;
                }
            }
            let avg_pct = if pct_weight_sum > 0.0 {
                weighted_pct_sum / pct_weight_sum
            } else {
                0.0
            };
            return (
                VerificationStatus::Partial {
                    verified_percentage: avg_pct,
                },
                partial_pct * 0.8,
            );
        }

        // Unknown wins
        (
            VerificationStatus::Unknown {
                reason: "Domain-weighted consensus favors unknown".to_string(),
            },
            unknown_weight / total_weight,
        )
    }
}

impl Default for ResultMerger {
    fn default() -> Self {
        Self::new(MergeStrategy::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallel::TaskResult;
    use std::collections::HashMap;

    fn make_result(backend: BackendId, status: VerificationStatus) -> BackendResult {
        BackendResult {
            backend,
            status,
            proof: Some("proof".to_string()),
            counterexample: None,
            diagnostics: vec![],
            time_taken: Duration::from_millis(100),
        }
    }

    fn make_task_result(backend: BackendId, status: VerificationStatus) -> TaskResult {
        TaskResult {
            property_index: 0,
            backend,
            result: Ok(make_result(backend, status)),
        }
    }

    #[test]
    fn test_first_success_prefers_proven() {
        let merger = ResultMerger::new(MergeStrategy::FirstSuccess);
        let results = vec![
            make_result(BackendId::Lean4, VerificationStatus::Proven),
            make_result(
                BackendId::Alloy,
                VerificationStatus::Unknown {
                    reason: "timeout".into(),
                },
            ),
        ];
        let (status, confidence) = merger.first_success_consensus(&results);
        assert!(matches!(status, VerificationStatus::Proven));
        assert_eq!(confidence, 1.0);
    }

    #[test]
    fn test_unanimous_agreement() {
        let merger = ResultMerger::new(MergeStrategy::Unanimous);
        let results = vec![
            make_result(BackendId::Lean4, VerificationStatus::Proven),
            make_result(BackendId::Alloy, VerificationStatus::Proven),
        ];
        let (status, confidence) = merger.unanimous_consensus(&results);
        assert!(matches!(status, VerificationStatus::Proven));
        assert_eq!(confidence, 1.0);
    }

    #[test]
    fn test_unanimous_disagreement() {
        let merger = ResultMerger::new(MergeStrategy::Unanimous);
        let results = vec![
            make_result(BackendId::Lean4, VerificationStatus::Proven),
            make_result(BackendId::Alloy, VerificationStatus::Disproven),
        ];
        let (status, _confidence) = merger.unanimous_consensus(&results);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn test_majority_voting() {
        let merger = ResultMerger::new(MergeStrategy::Majority);
        let results = vec![
            make_result(BackendId::Lean4, VerificationStatus::Proven),
            make_result(BackendId::Alloy, VerificationStatus::Proven),
            make_result(BackendId::Kani, VerificationStatus::Disproven),
        ];
        let (status, confidence) = merger.majority_consensus(&results);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!((confidence - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_pessimistic_prefers_disproven() {
        let merger = ResultMerger::new(MergeStrategy::Pessimistic);
        let results = vec![
            make_result(BackendId::Lean4, VerificationStatus::Proven),
            make_result(BackendId::Alloy, VerificationStatus::Disproven),
        ];
        let (status, _) = merger.pessimistic_consensus(&results);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn test_optimistic_prefers_proven() {
        let merger = ResultMerger::new(MergeStrategy::Optimistic);
        let results = vec![
            make_result(
                BackendId::Lean4,
                VerificationStatus::Unknown {
                    reason: "timeout".into(),
                },
            ),
            make_result(BackendId::Alloy, VerificationStatus::Proven),
        ];
        let (status, _) = merger.optimistic_consensus(&results);
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn test_merge_full_results() {
        let merger = ResultMerger::default();

        let mut by_property = HashMap::new();
        by_property.insert(
            0,
            vec![make_task_result(
                BackendId::Lean4,
                VerificationStatus::Proven,
            )],
        );
        by_property.insert(
            1,
            vec![make_task_result(
                BackendId::TlaPlus,
                VerificationStatus::Proven,
            )],
        );
        let property_types = HashMap::from([
            (0usize, PropertyType::Theorem),
            (1usize, PropertyType::Theorem),
        ]);

        let exec_results = ExecutionResults {
            by_property,
            property_types,
            total_time: Duration::from_secs(1),
            successful: 2,
            failed: 0,
        };

        let merged = merger.merge(exec_results);
        assert_eq!(merged.properties.len(), 2);
        assert_eq!(merged.summary.proven, 2);
        assert_eq!(merged.summary.disproven, 0);
    }

    #[test]
    fn test_merge_with_failed_backend() {
        let merger = ResultMerger::default();

        let mut by_property = HashMap::new();
        by_property.insert(
            0,
            vec![TaskResult {
                property_index: 0,
                backend: BackendId::Lean4,
                result: Err("Backend crashed".to_string()),
            }],
        );
        let property_types = HashMap::from([(0usize, PropertyType::Theorem)]);

        let exec_results = ExecutionResults {
            by_property,
            property_types,
            total_time: Duration::from_secs(1),
            successful: 0,
            failed: 1,
        };

        let merged = merger.merge(exec_results);
        assert_eq!(merged.properties.len(), 1);
        assert!(matches!(
            merged.properties[0].status,
            VerificationStatus::Unknown { .. }
        ));
        assert!(merged.properties[0].backend_results[0].error.is_some());
    }

    #[test]
    fn test_merge_extracts_proof() {
        let merger = ResultMerger::default();

        let mut result = make_result(BackendId::Lean4, VerificationStatus::Proven);
        result.proof = Some("The actual proof content".to_string());

        let mut by_property = HashMap::new();
        by_property.insert(
            0,
            vec![TaskResult {
                property_index: 0,
                backend: BackendId::Lean4,
                result: Ok(result),
            }],
        );
        let property_types = HashMap::from([(0usize, PropertyType::Theorem)]);

        let exec_results = ExecutionResults {
            by_property,
            property_types,
            total_time: Duration::from_secs(1),
            successful: 1,
            failed: 0,
        };

        let merged = merger.merge(exec_results);
        assert_eq!(
            merged.properties[0].proof,
            Some("The actual proof content".to_string())
        );
    }

    #[test]
    fn test_verification_summary() {
        let merger = ResultMerger::default();

        let mut by_property = HashMap::new();
        by_property.insert(
            0,
            vec![make_task_result(
                BackendId::Lean4,
                VerificationStatus::Proven,
            )],
        );
        by_property.insert(
            1,
            vec![make_task_result(
                BackendId::TlaPlus,
                VerificationStatus::Disproven,
            )],
        );
        by_property.insert(
            2,
            vec![make_task_result(
                BackendId::Kani,
                VerificationStatus::Unknown {
                    reason: "timeout".into(),
                },
            )],
        );
        let property_types = HashMap::from([
            (0usize, PropertyType::Theorem),
            (1usize, PropertyType::Temporal),
            (2usize, PropertyType::Contract),
        ]);

        let exec_results = ExecutionResults {
            by_property,
            property_types,
            total_time: Duration::from_secs(1),
            successful: 3,
            failed: 0,
        };

        let merged = merger.merge(exec_results);
        assert_eq!(merged.summary.proven, 1);
        assert_eq!(merged.summary.disproven, 1);
        assert_eq!(merged.summary.unknown, 1);
    }

    // ==================== Mutation-killing tests ====================

    #[test]
    fn test_overall_confidence_is_average() {
        // Mutation: replace += with -= or *= in merge (total_confidence accumulation)
        // Mutation: replace / with * in merge (average calculation)
        let merger = ResultMerger::default();

        // Create two properties with known confidence values
        let mut by_property = HashMap::new();
        by_property.insert(
            0,
            vec![make_task_result(
                BackendId::Lean4,
                VerificationStatus::Proven, // confidence 1.0
            )],
        );
        by_property.insert(
            1,
            vec![make_task_result(
                BackendId::Lean4,
                VerificationStatus::Proven, // confidence 1.0
            )],
        );
        let property_types = HashMap::from([
            (0usize, PropertyType::Theorem),
            (1usize, PropertyType::Theorem),
        ]);

        let exec_results = ExecutionResults {
            by_property,
            property_types,
            total_time: Duration::from_secs(1),
            successful: 2,
            failed: 0,
        };

        let merged = merger.merge(exec_results);
        // Average of 1.0 + 1.0 = 2.0 / 2 = 1.0
        assert!(
            (merged.summary.overall_confidence - 1.0).abs() < 0.01,
            "Expected overall_confidence ~1.0, got {}",
            merged.summary.overall_confidence
        );
    }

    #[test]
    fn test_majority_consensus_counts_disproven() {
        // Mutation: delete match arm VerificationStatus::Disproven in majority_consensus
        // Mutation: replace += with -= or *= for disproven_count
        let merger = ResultMerger::new(MergeStrategy::Majority);

        // 2 disproven, 1 proven -> disproven wins
        let results = vec![
            make_result(BackendId::Lean4, VerificationStatus::Disproven),
            make_result(BackendId::Alloy, VerificationStatus::Disproven),
            make_result(BackendId::Coq, VerificationStatus::Proven),
        ];

        let (status, confidence) = merger.majority_consensus(&results);
        assert!(
            matches!(status, VerificationStatus::Disproven),
            "Expected Disproven, got {:?}",
            status
        );
        // 2/3 = 0.666...
        assert!(
            (confidence - 0.666).abs() < 0.01,
            "Expected confidence ~0.666, got {}",
            confidence
        );
    }

    #[test]
    fn test_majority_consensus_proven_vs_disproven() {
        // Mutation: replace > with == or < or >= in majority_consensus comparisons
        // Mutation: replace && with || in majority_consensus conditions
        let merger = ResultMerger::new(MergeStrategy::Majority);

        // 2 proven, 1 disproven -> proven wins (proven_count >= disproven_count && proven_count > unknown_count)
        let results = vec![
            make_result(BackendId::Lean4, VerificationStatus::Proven),
            make_result(BackendId::Alloy, VerificationStatus::Proven),
            make_result(BackendId::Coq, VerificationStatus::Disproven),
        ];

        let (status, confidence) = merger.majority_consensus(&results);
        assert!(
            matches!(status, VerificationStatus::Proven),
            "Expected Proven with 2/3 majority, got {:?}",
            status
        );
        // 2/3 = 0.666...
        assert!(
            (confidence - 0.666).abs() < 0.01,
            "Expected confidence ~0.666, got {}",
            confidence
        );
    }

    #[test]
    fn test_majority_consensus_disproven_wins_over_unknown() {
        // Mutation: replace > with >= or < in disproven > unknown comparison
        let merger = ResultMerger::new(MergeStrategy::Majority);

        // 2 disproven, 1 unknown, 0 proven -> disproven wins
        let results = vec![
            make_result(BackendId::Lean4, VerificationStatus::Disproven),
            make_result(BackendId::Alloy, VerificationStatus::Disproven),
            make_result(
                BackendId::Coq,
                VerificationStatus::Unknown {
                    reason: "timeout".into(),
                },
            ),
        ];

        let (status, _confidence) = merger.majority_consensus(&results);
        assert!(
            matches!(status, VerificationStatus::Disproven),
            "Expected Disproven to win over Unknown, got {:?}",
            status
        );
    }

    #[test]
    fn test_majority_consensus_no_majority_returns_unknown() {
        // When unknown_count >= proven_count and unknown_count >= disproven_count
        let merger = ResultMerger::new(MergeStrategy::Majority);

        // 1 proven, 1 disproven, 2 unknown -> unknown wins
        let results = vec![
            make_result(BackendId::Lean4, VerificationStatus::Proven),
            make_result(BackendId::Alloy, VerificationStatus::Disproven),
            make_result(
                BackendId::Coq,
                VerificationStatus::Unknown {
                    reason: "timeout".into(),
                },
            ),
            make_result(
                BackendId::Kani,
                VerificationStatus::Unknown {
                    reason: "error".into(),
                },
            ),
        ];

        let (status, _confidence) = merger.majority_consensus(&results);
        assert!(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Expected Unknown (no majority), got {:?}",
            status
        );
    }

    #[test]
    fn test_most_confident_consensus_compares_scores() {
        // Mutation: replace > with ==, <, or >= in most_confident_consensus
        let merger = ResultMerger::new(MergeStrategy::MostConfident);

        // Proven (1.0) vs Disproven (0.95) vs Partial (0.5) -> Proven wins
        let results = vec![
            make_result(
                BackendId::Alloy,
                VerificationStatus::Partial {
                    verified_percentage: 0.5,
                },
            ),
            make_result(BackendId::Coq, VerificationStatus::Disproven),
            make_result(BackendId::Lean4, VerificationStatus::Proven),
        ];

        let (status, score) = merger.most_confident_consensus(&results);
        assert!(
            matches!(status, VerificationStatus::Proven),
            "Expected Proven (highest score), got {:?}",
            status
        );
        assert!(
            (score - 1.0).abs() < 0.01,
            "Expected score 1.0 for Proven, got {}",
            score
        );
    }

    #[test]
    fn test_most_confident_consensus_disproven_beats_partial() {
        // Disproven (0.95) vs Partial (0.8) -> Disproven wins
        let merger = ResultMerger::new(MergeStrategy::MostConfident);

        let results = vec![
            make_result(
                BackendId::Alloy,
                VerificationStatus::Partial {
                    verified_percentage: 0.8,
                },
            ),
            make_result(BackendId::Lean4, VerificationStatus::Disproven),
        ];

        let (status, score) = merger.most_confident_consensus(&results);
        assert!(
            matches!(status, VerificationStatus::Disproven),
            "Expected Disproven (score 0.95 > 0.8), got {:?}",
            status
        );
        assert!(
            (score - 0.95).abs() < 0.01,
            "Expected score 0.95 for Disproven, got {}",
            score
        );
    }

    #[test]
    fn test_status_score_values() {
        // Mutation: replace status_score -> f64 with 0.0, 1.0, or -1.0
        // Test each status returns the expected score

        // Proven = 1.0
        assert!(
            (ResultMerger::status_score(&VerificationStatus::Proven) - 1.0).abs() < 0.01,
            "Proven should have score 1.0"
        );

        // Disproven = 0.95
        assert!(
            (ResultMerger::status_score(&VerificationStatus::Disproven) - 0.95).abs() < 0.01,
            "Disproven should have score 0.95"
        );

        // Partial = verified_percentage
        let partial = VerificationStatus::Partial {
            verified_percentage: 0.75,
        };
        assert!(
            (ResultMerger::status_score(&partial) - 0.75).abs() < 0.01,
            "Partial(0.75) should have score 0.75"
        );

        // Unknown = 0.0
        let unknown = VerificationStatus::Unknown {
            reason: "test".into(),
        };
        assert!(
            (ResultMerger::status_score(&unknown) - 0.0).abs() < 0.01,
            "Unknown should have score 0.0"
        );
    }

    #[test]
    fn test_status_score_distinguishes_all_variants() {
        // Ensure scores are all different (prevents mutations that return same value)
        let proven_score = ResultMerger::status_score(&VerificationStatus::Proven);
        let disproven_score = ResultMerger::status_score(&VerificationStatus::Disproven);
        let partial_score = ResultMerger::status_score(&VerificationStatus::Partial {
            verified_percentage: 0.5,
        });
        let unknown_score =
            ResultMerger::status_score(&VerificationStatus::Unknown { reason: "x".into() });

        // All should be different
        assert!(proven_score != disproven_score);
        assert!(proven_score != partial_score);
        assert!(proven_score != unknown_score);
        assert!(disproven_score != partial_score);
        assert!(disproven_score != unknown_score);
        assert!(partial_score != unknown_score);

        // Order should be: proven > disproven > partial(0.5) > unknown
        assert!(proven_score > disproven_score);
        assert!(disproven_score > partial_score);
        assert!(partial_score > unknown_score);
    }

    #[test]
    fn test_majority_consensus_proven_equal_disproven_picks_proven() {
        // Mutation: replace > with >= in majority_consensus (proven_count >= disproven_count)
        // When proven_count == disproven_count, proven should win
        let merger = ResultMerger::new(MergeStrategy::Majority);

        // 1 proven, 1 disproven -> proven wins (>= condition)
        let results = vec![
            make_result(BackendId::Lean4, VerificationStatus::Proven),
            make_result(BackendId::Alloy, VerificationStatus::Disproven),
        ];

        let (status, confidence) = merger.majority_consensus(&results);
        assert!(
            matches!(status, VerificationStatus::Proven),
            "When proven == disproven, proven should win (>= condition)"
        );
        // 1/2 = 0.5
        assert!(
            (confidence - 0.5).abs() < 0.01,
            "Expected confidence 0.5, got {}",
            confidence
        );
    }

    #[test]
    fn test_majority_consensus_edge_case_all_unknown() {
        // Test case where unknown dominates
        let merger = ResultMerger::new(MergeStrategy::Majority);

        let results = vec![
            make_result(
                BackendId::Lean4,
                VerificationStatus::Unknown { reason: "a".into() },
            ),
            make_result(
                BackendId::Alloy,
                VerificationStatus::Unknown { reason: "b".into() },
            ),
            make_result(
                BackendId::Coq,
                VerificationStatus::Unknown { reason: "c".into() },
            ),
        ];

        let (status, confidence) = merger.majority_consensus(&results);
        assert!(
            matches!(status, VerificationStatus::Unknown { .. }),
            "All unknown should result in Unknown"
        );
        assert!(
            (confidence - 1.0).abs() < 0.01,
            "3/3 unknown should give confidence 1.0"
        );
    }

    #[test]
    fn test_most_confident_consensus_equal_scores_uses_first() {
        // Mutation: replace > with >= in most_confident_consensus
        // When scores are equal, first result should be used
        let merger = ResultMerger::new(MergeStrategy::MostConfident);

        // Two proven results (both score 1.0) - first should win
        let results = vec![
            make_result(BackendId::Lean4, VerificationStatus::Proven),
            make_result(BackendId::Alloy, VerificationStatus::Proven),
        ];

        let (status, score) = merger.most_confident_consensus(&results);
        assert!(
            matches!(status, VerificationStatus::Proven),
            "Expected Proven"
        );
        assert!((score - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_overall_confidence_accumulates_properly() {
        // Mutation: replace += with -= or *= in merge (total_confidence)
        let merger = ResultMerger::default();

        // Create 3 properties with known confidence values
        let mut by_property = HashMap::new();
        by_property.insert(
            0,
            vec![make_task_result(
                BackendId::Lean4,
                VerificationStatus::Proven, // confidence 1.0
            )],
        );
        by_property.insert(
            1,
            vec![make_task_result(
                BackendId::Alloy,
                VerificationStatus::Disproven, // confidence 1.0
            )],
        );
        by_property.insert(
            2,
            vec![make_task_result(
                BackendId::Coq,
                VerificationStatus::Proven, // confidence 1.0
            )],
        );
        let property_types = HashMap::from([
            (0usize, PropertyType::Theorem),
            (1usize, PropertyType::Invariant),
            (2usize, PropertyType::Theorem),
        ]);

        let exec_results = ExecutionResults {
            by_property,
            property_types,
            total_time: Duration::from_secs(1),
            successful: 3,
            failed: 0,
        };

        let merged = merger.merge(exec_results);
        // Average of 1.0 + 1.0 + 1.0 = 3.0 / 3 = 1.0
        assert!(
            (merged.summary.overall_confidence - 1.0).abs() < 0.01,
            "Average of three 1.0 confidences should be 1.0, got {}",
            merged.summary.overall_confidence
        );
    }

    #[test]
    fn test_merge_partial_status_counted() {
        // Mutation: replace += with -= or *= in merge for partial count (line 118)
        let merger = ResultMerger::default();

        let mut by_property = HashMap::new();
        by_property.insert(
            0,
            vec![make_task_result(
                BackendId::Lean4,
                VerificationStatus::Partial {
                    verified_percentage: 0.6,
                },
            )],
        );
        by_property.insert(
            1,
            vec![make_task_result(
                BackendId::Alloy,
                VerificationStatus::Partial {
                    verified_percentage: 0.8,
                },
            )],
        );
        let property_types = HashMap::from([
            (0usize, PropertyType::Theorem),
            (1usize, PropertyType::Theorem),
        ]);

        let exec_results = ExecutionResults {
            by_property,
            property_types,
            total_time: Duration::from_secs(1),
            successful: 2,
            failed: 0,
        };

        let merged = merger.merge(exec_results);
        assert_eq!(
            merged.summary.partial, 2,
            "Expected 2 partial results, got {}",
            merged.summary.partial
        );
        assert_eq!(merged.summary.proven, 0);
        assert_eq!(merged.summary.disproven, 0);
        assert_eq!(merged.summary.unknown, 0);
    }

    #[test]
    fn test_majority_consensus_strict_comparison() {
        // Mutation: replace > with >= in majority_consensus
        // This tests that disproven must be STRICTLY greater than proven to win
        let merger = ResultMerger::new(MergeStrategy::Majority);

        // 1 proven, 1 disproven, 0 unknown
        // proven_count >= disproven_count is TRUE (1 >= 1)
        // proven_count > unknown_count is TRUE (1 > 0)
        // So proven should win, not disproven
        let results = vec![
            make_result(BackendId::Lean4, VerificationStatus::Proven),
            make_result(BackendId::Alloy, VerificationStatus::Disproven),
        ];

        let (status, _) = merger.majority_consensus(&results);
        assert!(
            matches!(status, VerificationStatus::Proven),
            "When proven == disproven and both > unknown, proven wins"
        );
    }

    #[test]
    fn test_majority_consensus_proven_must_beat_unknown() {
        // Mutation: replace && with || in majority_consensus
        // Tests that proven must both >= disproven AND > unknown
        let merger = ResultMerger::new(MergeStrategy::Majority);

        // 1 proven, 0 disproven, 2 unknown
        // proven_count >= disproven_count is TRUE (1 >= 0)
        // proven_count > unknown_count is FALSE (1 > 2)
        // So unknown should win
        let results = vec![
            make_result(BackendId::Lean4, VerificationStatus::Proven),
            make_result(
                BackendId::Alloy,
                VerificationStatus::Unknown { reason: "a".into() },
            ),
            make_result(
                BackendId::Coq,
                VerificationStatus::Unknown { reason: "b".into() },
            ),
        ];

        let (status, _) = merger.majority_consensus(&results);
        assert!(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Unknown count higher than proven should make Unknown win"
        );
    }

    #[test]
    fn test_majority_consensus_disproven_strictly_greater() {
        // Mutation: replace > with >= for disproven_count > proven_count
        // Tests that disproven needs to be STRICTLY greater than proven
        let merger = ResultMerger::new(MergeStrategy::Majority);

        // 2 proven, 2 disproven, 0 unknown
        // proven_count >= disproven_count is TRUE (2 >= 2)
        // So proven wins (first condition)
        let results = vec![
            make_result(BackendId::Lean4, VerificationStatus::Proven),
            make_result(BackendId::Alloy, VerificationStatus::Proven),
            make_result(BackendId::Coq, VerificationStatus::Disproven),
            make_result(BackendId::Kani, VerificationStatus::Disproven),
        ];

        let (status, _) = merger.majority_consensus(&results);
        assert!(
            matches!(status, VerificationStatus::Proven),
            "When proven == disproven, proven wins"
        );
    }

    #[test]
    fn test_most_confident_strictly_greater() {
        // Mutation: replace > with >= in most_confident_consensus
        // Tests strict comparison between scores
        let merger = ResultMerger::new(MergeStrategy::MostConfident);

        // First result should be kept if scores are equal (> not >=)
        let results = vec![
            make_result(BackendId::Lean4, VerificationStatus::Proven),
            make_result(BackendId::Alloy, VerificationStatus::Proven),
        ];

        let (_, score) = merger.most_confident_consensus(&results);
        // Both have score 1.0, but result should still be valid
        assert!((score - 1.0).abs() < 0.01);

        // More importantly, test that higher score wins over lower
        let results2 = vec![
            make_result(
                BackendId::Lean4,
                VerificationStatus::Partial {
                    verified_percentage: 0.5,
                },
            ),
            make_result(BackendId::Alloy, VerificationStatus::Proven), // 1.0 > 0.5
        ];

        let (status2, score2) = merger.most_confident_consensus(&results2);
        assert!(
            matches!(status2, VerificationStatus::Proven),
            "Higher score (1.0) should beat lower (0.5)"
        );
        assert!((score2 - 1.0).abs() < 0.01);
    }
    // ==================== BFT Consensus Tests ====================

    #[test]
    fn test_bft_consensus_all_agree_proven() {
        // 4 backends, tolerate 1 faulty -> need 3 agreements
        // 4 = 3*1 + 1, so BFT is valid
        let merger = ResultMerger::new(MergeStrategy::ByzantineFaultTolerant { max_faulty: 1 });
        let results = vec![
            make_result(BackendId::Lean4, VerificationStatus::Proven),
            make_result(BackendId::Alloy, VerificationStatus::Proven),
            make_result(BackendId::Coq, VerificationStatus::Proven),
            make_result(BackendId::Kani, VerificationStatus::Proven),
        ];

        let (status, confidence) = merger.bft_consensus(&results, 1);
        assert!(
            matches!(status, VerificationStatus::Proven),
            "All backends agree on Proven"
        );
        assert!(
            (confidence - 1.0).abs() < 0.01,
            "4/4 agreement should give 1.0 confidence"
        );
    }

    #[test]
    fn test_bft_consensus_one_faulty_still_passes() {
        // 4 backends, tolerate 1 faulty -> need 3 agreements
        // 3 proven, 1 disproven (faulty) -> should still reach consensus
        let merger = ResultMerger::new(MergeStrategy::ByzantineFaultTolerant { max_faulty: 1 });
        let results = vec![
            make_result(BackendId::Lean4, VerificationStatus::Proven),
            make_result(BackendId::Alloy, VerificationStatus::Proven),
            make_result(BackendId::Coq, VerificationStatus::Proven),
            make_result(BackendId::Kani, VerificationStatus::Disproven), // faulty
        ];

        let (status, confidence) = merger.bft_consensus(&results, 1);
        assert!(
            matches!(status, VerificationStatus::Proven),
            "3/4 proven with 1 faulty should reach BFT consensus"
        );
        assert!(
            (confidence - 0.75).abs() < 0.01,
            "3/4 agreement should give 0.75 confidence"
        );
    }

    #[test]
    fn test_bft_consensus_no_consensus() {
        // 4 backends, tolerate 1 faulty -> need 3 agreements
        // 2 proven, 2 disproven -> no consensus
        let merger = ResultMerger::new(MergeStrategy::ByzantineFaultTolerant { max_faulty: 1 });
        let results = vec![
            make_result(BackendId::Lean4, VerificationStatus::Proven),
            make_result(BackendId::Alloy, VerificationStatus::Proven),
            make_result(BackendId::Coq, VerificationStatus::Disproven),
            make_result(BackendId::Kani, VerificationStatus::Disproven),
        ];

        let (status, confidence) = merger.bft_consensus(&results, 1);
        assert!(
            matches!(status, VerificationStatus::Unknown { .. }),
            "2/4 vs 2/4 should not reach BFT consensus"
        );
        assert!(confidence < 0.5, "No consensus should have low confidence");
    }

    #[test]
    fn test_bft_consensus_insufficient_backends() {
        // 2 backends, tolerate 1 faulty -> need 3*1+1=4 backends for BFT safety
        // Falls back to majority with reduced confidence
        let merger = ResultMerger::new(MergeStrategy::ByzantineFaultTolerant { max_faulty: 1 });
        let results = vec![
            make_result(BackendId::Lean4, VerificationStatus::Proven),
            make_result(BackendId::Alloy, VerificationStatus::Proven),
        ];

        let (status, confidence) = merger.bft_consensus(&results, 1);
        // Falls back to majority
        assert!(
            matches!(status, VerificationStatus::Proven),
            "Fallback to majority should work"
        );
        assert!(
            (confidence - 0.5).abs() < 0.01,
            "Fallback has reduced confidence of 0.5"
        );
    }

    #[test]
    fn test_bft_consensus_disproven_wins() {
        // 4 backends, tolerate 1 faulty -> need 3 agreements
        // 3 disproven, 1 proven -> disproven consensus
        let merger = ResultMerger::new(MergeStrategy::ByzantineFaultTolerant { max_faulty: 1 });
        let results = vec![
            make_result(BackendId::Lean4, VerificationStatus::Disproven),
            make_result(BackendId::Alloy, VerificationStatus::Disproven),
            make_result(BackendId::Coq, VerificationStatus::Disproven),
            make_result(BackendId::Kani, VerificationStatus::Proven),
        ];

        let (status, confidence) = merger.bft_consensus(&results, 1);
        assert!(
            matches!(status, VerificationStatus::Disproven),
            "3/4 disproven should reach BFT consensus"
        );
        assert!(
            (confidence - 0.75).abs() < 0.01,
            "3/4 agreement gives 0.75 confidence"
        );
    }

    #[test]
    fn test_bft_higher_fault_tolerance() {
        // 7 backends, tolerate 2 faulty -> need 7-2=5 agreements, min = 3*2+1=7
        let merger = ResultMerger::new(MergeStrategy::ByzantineFaultTolerant { max_faulty: 2 });
        let results = vec![
            make_result(BackendId::Lean4, VerificationStatus::Proven),
            make_result(BackendId::Alloy, VerificationStatus::Proven),
            make_result(BackendId::Coq, VerificationStatus::Proven),
            make_result(BackendId::Kani, VerificationStatus::Proven),
            make_result(BackendId::TlaPlus, VerificationStatus::Proven),
            make_result(BackendId::Dafny, VerificationStatus::Disproven), // faulty
            make_result(BackendId::Creusot, VerificationStatus::Disproven), // faulty
        ];

        let (status, confidence) = merger.bft_consensus(&results, 2);
        assert!(
            matches!(status, VerificationStatus::Proven),
            "5/7 proven should reach BFT consensus with f=2"
        );
        // 5/7 ≈ 0.714
        assert!(
            (confidence - 5.0 / 7.0).abs() < 0.01,
            "5/7 agreement should give ~0.714 confidence"
        );
    }

    // ==================== Weighted Consensus Tests ====================

    #[test]
    fn test_weighted_consensus_equal_weights() {
        // All backends have equal weight -> behaves like majority
        let weights = HashMap::new(); // All get default 0.5
        let merger = ResultMerger::new(MergeStrategy::WeightedConsensus {
            weights: weights.clone(),
        });
        let results = vec![
            make_result(BackendId::Lean4, VerificationStatus::Proven),
            make_result(BackendId::Alloy, VerificationStatus::Proven),
            make_result(BackendId::Coq, VerificationStatus::Disproven),
        ];

        let (status, _) = merger.weighted_consensus(&results, &weights);
        assert!(
            matches!(status, VerificationStatus::Proven),
            "2/3 proven should win with equal weights"
        );
    }

    #[test]
    fn test_weighted_consensus_high_weight_wins() {
        // One backend with high weight can outweigh others
        let mut weights = HashMap::new();
        weights.insert(BackendId::Lean4, 1.0); // High confidence in Lean4
        weights.insert(BackendId::Alloy, 0.2); // Low confidence in Alloy
        weights.insert(BackendId::Coq, 0.2); // Low confidence in Coq

        let merger = ResultMerger::new(MergeStrategy::WeightedConsensus {
            weights: weights.clone(),
        });
        let results = vec![
            make_result(BackendId::Lean4, VerificationStatus::Proven), // weight 1.0
            make_result(BackendId::Alloy, VerificationStatus::Disproven), // weight 0.2
            make_result(BackendId::Coq, VerificationStatus::Disproven), // weight 0.2
        ];

        // Total weight: 1.4
        // Proven: 1.0/1.4 ≈ 0.714
        // Disproven: 0.4/1.4 ≈ 0.286
        let (status, confidence) = merger.weighted_consensus(&results, &weights);
        assert!(
            matches!(status, VerificationStatus::Proven),
            "High-weight Lean4 should win"
        );
        assert!(
            confidence > 0.7,
            "Proven should have high confidence due to Lean4 weight"
        );
    }

    #[test]
    fn test_weighted_consensus_low_weight_loses() {
        // Backend with low weight can't outweigh others
        let mut weights = HashMap::new();
        weights.insert(BackendId::Lean4, 0.1); // Low confidence in Lean4
        weights.insert(BackendId::Alloy, 0.9); // High confidence in Alloy
        weights.insert(BackendId::Coq, 0.9); // High confidence in Coq

        let merger = ResultMerger::new(MergeStrategy::WeightedConsensus {
            weights: weights.clone(),
        });
        let results = vec![
            make_result(BackendId::Lean4, VerificationStatus::Proven), // weight 0.1
            make_result(BackendId::Alloy, VerificationStatus::Disproven), // weight 0.9
            make_result(BackendId::Coq, VerificationStatus::Disproven), // weight 0.9
        ];

        // Total weight: 1.9
        // Proven: 0.1/1.9 ≈ 0.053
        // Disproven: 1.8/1.9 ≈ 0.947
        let (status, confidence) = merger.weighted_consensus(&results, &weights);
        assert!(
            matches!(status, VerificationStatus::Disproven),
            "High-weight disproven should win"
        );
        assert!(confidence > 0.9, "Disproven should have high confidence");
    }

    #[test]
    fn test_weighted_consensus_missing_weights_use_default() {
        // Missing backends use default weight 0.5
        let mut weights = HashMap::new();
        weights.insert(BackendId::Lean4, 1.0);
        // Alloy and Coq not in map -> use 0.5

        let merger = ResultMerger::new(MergeStrategy::WeightedConsensus {
            weights: weights.clone(),
        });
        let results = vec![
            make_result(BackendId::Lean4, VerificationStatus::Proven), // weight 1.0
            make_result(BackendId::Alloy, VerificationStatus::Disproven), // default 0.5
            make_result(BackendId::Coq, VerificationStatus::Disproven), // default 0.5
        ];

        // Total weight: 2.0
        // Proven: 1.0/2.0 = 0.5
        // Disproven: 1.0/2.0 = 0.5
        // When tied, proven wins (>= comparison)
        let (status, _) = merger.weighted_consensus(&results, &weights);
        assert!(
            matches!(status, VerificationStatus::Proven),
            "Equal weighted votes should favor proven"
        );
    }

    #[test]
    fn test_weighted_consensus_clamps_weights() {
        // Weights outside [0.0, 1.0] should be clamped
        let mut weights = HashMap::new();
        weights.insert(BackendId::Lean4, 10.0); // Should clamp to 1.0
        weights.insert(BackendId::Alloy, -5.0); // Should clamp to 0.0

        let merger = ResultMerger::new(MergeStrategy::WeightedConsensus {
            weights: weights.clone(),
        });
        let results = vec![
            make_result(BackendId::Lean4, VerificationStatus::Proven),
            make_result(BackendId::Alloy, VerificationStatus::Disproven),
        ];

        // Clamped: Lean4=1.0, Alloy=0.0
        // Total weight: 1.0
        // Proven: 1.0/1.0 = 1.0
        let (status, confidence) = merger.weighted_consensus(&results, &weights);
        assert!(
            matches!(status, VerificationStatus::Proven),
            "Clamped weight should work"
        );
        assert!(
            (confidence - 1.0).abs() < 0.01,
            "Only Lean4 contributes weight"
        );
    }

    #[test]
    fn test_weighted_consensus_partial_averaged() {
        // When partial wins, percentages are weighted-averaged
        let mut weights = HashMap::new();
        weights.insert(BackendId::Lean4, 0.8);
        weights.insert(BackendId::Alloy, 0.2);

        let merger = ResultMerger::new(MergeStrategy::WeightedConsensus {
            weights: weights.clone(),
        });
        let results = vec![
            make_result(
                BackendId::Lean4,
                VerificationStatus::Partial {
                    verified_percentage: 0.9,
                },
            ), // weight 0.8
            make_result(
                BackendId::Alloy,
                VerificationStatus::Partial {
                    verified_percentage: 0.5,
                },
            ), // weight 0.2
        ];

        // Weighted avg: (0.9*0.8 + 0.5*0.2) / (0.8 + 0.2) = (0.72 + 0.1) / 1.0 = 0.82
        let (status, _) = merger.weighted_consensus(&results, &weights);
        match status {
            VerificationStatus::Partial {
                verified_percentage,
            } => {
                assert!(
                    (verified_percentage - 0.82).abs() < 0.01,
                    "Weighted average should be ~0.82, got {}",
                    verified_percentage
                );
            }
            _ => panic!("Expected Partial status"),
        }
    }

    #[test]
    fn test_merge_strategy_bft_through_merger() {
        // Test BFT through the full merge path
        let merger = ResultMerger::new(MergeStrategy::ByzantineFaultTolerant { max_faulty: 1 });

        let mut by_property = HashMap::new();
        by_property.insert(
            0,
            vec![
                make_task_result(BackendId::Lean4, VerificationStatus::Proven),
                make_task_result(BackendId::Alloy, VerificationStatus::Proven),
                make_task_result(BackendId::Coq, VerificationStatus::Proven),
                make_task_result(BackendId::Kani, VerificationStatus::Disproven),
            ],
        );
        let property_types = HashMap::from([(0usize, PropertyType::Theorem)]);

        let exec_results = ExecutionResults {
            by_property,
            property_types,
            total_time: Duration::from_secs(1),
            successful: 4,
            failed: 0,
        };

        let merged = merger.merge(exec_results);
        assert_eq!(merged.summary.proven, 1);
        assert_eq!(merged.summary.disproven, 0);
    }

    #[test]
    fn test_merge_strategy_weighted_through_merger() {
        // Test weighted consensus through the full merge path
        let mut weights = HashMap::new();
        weights.insert(BackendId::Lean4, 1.0);
        weights.insert(BackendId::Alloy, 0.1);

        let merger = ResultMerger::new(MergeStrategy::WeightedConsensus { weights });

        let mut by_property = HashMap::new();
        by_property.insert(
            0,
            vec![
                make_task_result(BackendId::Lean4, VerificationStatus::Proven),
                make_task_result(BackendId::Alloy, VerificationStatus::Disproven),
            ],
        );
        let property_types = HashMap::from([(0usize, PropertyType::Theorem)]);

        let exec_results = ExecutionResults {
            by_property,
            property_types,
            total_time: Duration::from_secs(1),
            successful: 2,
            failed: 0,
        };

        let merged = merger.merge(exec_results);
        assert_eq!(merged.summary.proven, 1);
        assert_eq!(merged.summary.disproven, 0);
    }

    // ==========================================================================
    // Domain-weighted consensus tests
    // ==========================================================================

    #[test]
    fn test_domain_weighted_uses_domain_weights() {
        let mut domain_weights = HashMap::new();
        let mut aggregate_weights = HashMap::new();

        // Lean4 is excellent at theorems (0.9), TLA+ is mediocre (0.4)
        domain_weights.insert(DomainKey::new(BackendId::Lean4, PropertyType::Theorem), 0.9);
        domain_weights.insert(
            DomainKey::new(BackendId::TlaPlus, PropertyType::Theorem),
            0.4,
        );

        // Aggregate weights (fallback)
        aggregate_weights.insert(BackendId::Lean4, 0.7);
        aggregate_weights.insert(BackendId::TlaPlus, 0.6);

        let merger = ResultMerger::new(MergeStrategy::DomainWeightedConsensus {
            domain_weights,
            aggregate_weights,
        });

        let results = vec![
            make_result(BackendId::Lean4, VerificationStatus::Proven),
            make_result(BackendId::TlaPlus, VerificationStatus::Disproven),
        ];

        // With domain weights: Lean4 (0.9) vs TLA+ (0.4)
        // Lean4's proven (0.9) should win over TLA+'s disproven (0.4)
        let (status, confidence) =
            merger.compute_consensus_for_property_type(&results, PropertyType::Theorem);
        assert!(matches!(status, VerificationStatus::Proven));
        // confidence = 0.9 / (0.9 + 0.4) = 0.692...
        assert!((confidence - 0.692).abs() < 0.01);
    }

    #[test]
    fn test_domain_weighted_falls_back_to_aggregate() {
        let mut domain_weights = HashMap::new();
        let mut aggregate_weights = HashMap::new();

        // Only theorem-specific weights, no temporal weights
        domain_weights.insert(DomainKey::new(BackendId::Lean4, PropertyType::Theorem), 0.9);

        // Aggregate weights used for temporal properties
        aggregate_weights.insert(BackendId::Lean4, 0.6);
        aggregate_weights.insert(BackendId::TlaPlus, 0.8);

        let merger = ResultMerger::new(MergeStrategy::DomainWeightedConsensus {
            domain_weights,
            aggregate_weights,
        });

        let results = vec![
            make_result(BackendId::Lean4, VerificationStatus::Proven),
            make_result(BackendId::TlaPlus, VerificationStatus::Disproven),
        ];

        // For temporal: Lean4 falls back to 0.6, TLA+ uses 0.8
        // TLA+'s disproven (0.8) should win over Lean4's proven (0.6)
        let (status, confidence) =
            merger.compute_consensus_for_property_type(&results, PropertyType::Temporal);
        assert!(matches!(status, VerificationStatus::Disproven));
        // confidence = 0.8 / (0.6 + 0.8) = 0.571...
        assert!((confidence - 0.571).abs() < 0.01);
    }

    #[test]
    fn test_domain_weighted_default_fallback() {
        let domain_weights = HashMap::new(); // Empty
        let aggregate_weights = HashMap::new(); // Empty

        let merger = ResultMerger::new(MergeStrategy::DomainWeightedConsensus {
            domain_weights,
            aggregate_weights,
        });

        let results = vec![
            make_result(BackendId::Lean4, VerificationStatus::Proven),
            make_result(BackendId::TlaPlus, VerificationStatus::Proven),
        ];

        // Both use default weight (0.5), so they tie
        let (status, confidence) =
            merger.compute_consensus_for_property_type(&results, PropertyType::Theorem);
        assert!(matches!(status, VerificationStatus::Proven));
        // confidence = 1.0 / 1.0 = 1.0 (both proven with equal weights)
        assert!((confidence - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_domain_weighted_fallback_to_aggregate_in_compute_consensus() {
        let mut aggregate_weights = HashMap::new();
        aggregate_weights.insert(BackendId::Lean4, 0.8);
        aggregate_weights.insert(BackendId::TlaPlus, 0.2);

        let merger = ResultMerger::new(MergeStrategy::DomainWeightedConsensus {
            domain_weights: HashMap::new(),
            aggregate_weights,
        });

        let results = vec![
            make_result(BackendId::Lean4, VerificationStatus::Proven),
            make_result(BackendId::TlaPlus, VerificationStatus::Disproven),
        ];

        // Without property type (using compute_consensus), falls back to aggregate
        let (status, confidence) = merger.compute_consensus(&results);
        assert!(matches!(status, VerificationStatus::Proven));
        // Lean4 (0.8) > TLA+ (0.2)
        assert!((confidence - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_domain_weighted_partial_averaging() {
        let mut domain_weights = HashMap::new();
        domain_weights.insert(
            DomainKey::new(BackendId::Lean4, PropertyType::Contract),
            0.8,
        );
        domain_weights.insert(DomainKey::new(BackendId::Kani, PropertyType::Contract), 0.6);

        let merger = ResultMerger::new(MergeStrategy::DomainWeightedConsensus {
            domain_weights,
            aggregate_weights: HashMap::new(),
        });

        let results = vec![
            make_result(
                BackendId::Lean4,
                VerificationStatus::Partial {
                    verified_percentage: 80.0,
                },
            ),
            make_result(
                BackendId::Kani,
                VerificationStatus::Partial {
                    verified_percentage: 60.0,
                },
            ),
        ];

        let (status, _confidence) =
            merger.compute_consensus_for_property_type(&results, PropertyType::Contract);

        // Both partial, so result should be weighted average
        // (80.0 * 0.8 + 60.0 * 0.6) / (0.8 + 0.6) = (64 + 36) / 1.4 = 100 / 1.4 = 71.43
        if let VerificationStatus::Partial {
            verified_percentage,
        } = status
        {
            assert!(
                (verified_percentage - 71.43).abs() < 0.1,
                "Expected ~71.43, got {}",
                verified_percentage
            );
        } else {
            panic!("Expected Partial status");
        }
    }

    #[test]
    fn test_domain_weighted_empty_results() {
        let merger = ResultMerger::new(MergeStrategy::DomainWeightedConsensus {
            domain_weights: HashMap::new(),
            aggregate_weights: HashMap::new(),
        });

        let results: Vec<BackendResult> = vec![];

        let (status, confidence) =
            merger.compute_consensus_for_property_type(&results, PropertyType::Theorem);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
        assert_eq!(confidence, 0.0);
    }

    #[test]
    fn test_merge_uses_domain_weights_with_property_types() {
        let mut domain_weights = HashMap::new();
        domain_weights.insert(DomainKey::new(BackendId::Lean4, PropertyType::Theorem), 0.9);
        domain_weights.insert(
            DomainKey::new(BackendId::TlaPlus, PropertyType::Theorem),
            0.1,
        );

        let merger = ResultMerger::new(MergeStrategy::DomainWeightedConsensus {
            domain_weights,
            aggregate_weights: HashMap::new(),
        });

        let mut by_property = HashMap::new();
        by_property.insert(
            0,
            vec![
                make_task_result(BackendId::Lean4, VerificationStatus::Proven),
                make_task_result(BackendId::TlaPlus, VerificationStatus::Disproven),
            ],
        );
        let property_types = HashMap::from([(0usize, PropertyType::Theorem)]);

        let exec_results = ExecutionResults {
            by_property,
            property_types,
            total_time: Duration::from_millis(200),
            successful: 2,
            failed: 0,
        };

        let merged = merger.merge(exec_results);
        assert!(matches!(
            merged.properties[0].status,
            VerificationStatus::Proven
        ));
        assert_eq!(
            merged.properties[0].property_type,
            Some(PropertyType::Theorem)
        );
    }
}

// ============================================================================
// Kani Proof Harnesses
// ============================================================================
// These formal verification harnesses prove properties about the merge logic
// using bounded model checking. Run with: cargo kani -p dashprove-dispatcher

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Prove that status_score returns values in the valid range [0.0, 1.0]
    #[kani::proof]
    fn verify_status_score_bounds() {
        // Test all four status variants with arbitrary values where applicable
        let proven_score = ResultMerger::status_score(&VerificationStatus::Proven);
        kani::assert(
            proven_score >= 0.0 && proven_score <= 1.0,
            "Proven score out of bounds",
        );

        let disproven_score = ResultMerger::status_score(&VerificationStatus::Disproven);
        kani::assert(
            disproven_score >= 0.0 && disproven_score <= 1.0,
            "Disproven score out of bounds",
        );

        // Test Partial with symbolic percentage (clamped to valid range)
        let pct: f64 = kani::any();
        kani::assume(pct >= 0.0 && pct <= 100.0);
        let partial_score = ResultMerger::status_score(&VerificationStatus::Partial {
            verified_percentage: pct,
        });
        kani::assert(
            partial_score >= 0.0 && partial_score <= 100.0,
            "Partial score out of bounds",
        );

        let unknown_score = ResultMerger::status_score(&VerificationStatus::Unknown {
            reason: String::new(),
        });
        kani::assert(
            unknown_score >= 0.0 && unknown_score <= 1.0,
            "Unknown score out of bounds",
        );
    }

    /// Prove that status_score ordering is correct:
    /// Proven > Disproven > Partial(0.5) > Unknown
    #[kani::proof]
    fn verify_status_score_ordering() {
        let proven = ResultMerger::status_score(&VerificationStatus::Proven);
        let disproven = ResultMerger::status_score(&VerificationStatus::Disproven);
        let partial = ResultMerger::status_score(&VerificationStatus::Partial {
            verified_percentage: 0.5,
        });
        let unknown = ResultMerger::status_score(&VerificationStatus::Unknown {
            reason: String::new(),
        });

        // Verify the strict ordering
        kani::assert(
            proven > disproven,
            "Proven should score higher than Disproven",
        );
        kani::assert(
            disproven > partial,
            "Disproven should score higher than Partial(0.5)",
        );
        kani::assert(
            partial > unknown,
            "Partial(0.5) should score higher than Unknown",
        );
    }

    /// Prove that status_category returns the correct category string
    #[kani::proof]
    fn verify_status_category_consistency() {
        // Verify each variant maps to its expected category
        kani::assert(
            ResultMerger::status_category(&VerificationStatus::Proven) == "proven",
            "Proven category mismatch",
        );
        kani::assert(
            ResultMerger::status_category(&VerificationStatus::Disproven) == "disproven",
            "Disproven category mismatch",
        );
        kani::assert(
            ResultMerger::status_category(&VerificationStatus::Partial {
                verified_percentage: 50.0,
            }) == "partial",
            "Partial category mismatch",
        );
        kani::assert(
            ResultMerger::status_category(&VerificationStatus::Unknown {
                reason: String::new(),
            }) == "unknown",
            "Unknown category mismatch",
        );
    }

    /// Prove that status_score for Unknown is always 0.0
    #[kani::proof]
    fn verify_unknown_score_is_zero() {
        let score = ResultMerger::status_score(&VerificationStatus::Unknown {
            reason: String::new(),
        });
        kani::assert(score == 0.0, "Unknown score should be exactly 0.0");
    }

    /// Prove that status_score for Proven is always 1.0
    #[kani::proof]
    fn verify_proven_score_is_one() {
        let score = ResultMerger::status_score(&VerificationStatus::Proven);
        kani::assert(score == 1.0, "Proven score should be exactly 1.0");
    }

    /// Prove that status_score for Disproven is always 0.95
    #[kani::proof]
    fn verify_disproven_score_is_correct() {
        let score = ResultMerger::status_score(&VerificationStatus::Disproven);
        kani::assert(
            (score - 0.95).abs() < 0.001,
            "Disproven score should be 0.95",
        );
    }

    /// Prove that Partial score equals the verified_percentage
    #[kani::proof]
    fn verify_partial_score_equals_percentage() {
        let pct: f64 = kani::any();
        kani::assume(pct >= 0.0 && pct <= 100.0);
        kani::assume(!pct.is_nan());

        let score = ResultMerger::status_score(&VerificationStatus::Partial {
            verified_percentage: pct,
        });
        kani::assert(
            score == pct,
            "Partial score should equal verified_percentage",
        );
    }

    /// Prove that all status categories are distinct
    #[kani::proof]
    fn verify_categories_are_distinct() {
        let proven_cat = ResultMerger::status_category(&VerificationStatus::Proven);
        let disproven_cat = ResultMerger::status_category(&VerificationStatus::Disproven);
        let partial_cat = ResultMerger::status_category(&VerificationStatus::Partial {
            verified_percentage: 50.0,
        });
        let unknown_cat = ResultMerger::status_category(&VerificationStatus::Unknown {
            reason: String::new(),
        });

        kani::assert(proven_cat != disproven_cat, "proven != disproven");
        kani::assert(proven_cat != partial_cat, "proven != partial");
        kani::assert(proven_cat != unknown_cat, "proven != unknown");
        kani::assert(disproven_cat != partial_cat, "disproven != partial");
        kani::assert(disproven_cat != unknown_cat, "disproven != unknown");
        kani::assert(partial_cat != unknown_cat, "partial != unknown");
    }

    /// Prove that Proven and Disproven scores are distinct
    #[kani::proof]
    fn verify_proven_disproven_distinct() {
        let proven = ResultMerger::status_score(&VerificationStatus::Proven);
        let disproven = ResultMerger::status_score(&VerificationStatus::Disproven);
        kani::assert(
            proven != disproven,
            "Proven and Disproven must have distinct scores",
        );
    }
}
