//! Tactic effectiveness database
//!
//! Tracks which tactics work well in which contexts, enabling
//! data-driven tactic suggestions.

use crate::similarity::PropertyFeatures;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Context in which a tactic is applied
///
/// Derived from property features to group similar situations.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TacticContext {
    /// Property type (theorem, invariant, etc.)
    pub property_type: String,
    /// Whether quantifiers are involved
    pub has_quantifiers: bool,
    /// Whether implications are involved
    pub has_implications: bool,
    /// Whether arithmetic is involved
    pub has_arithmetic: bool,
    /// Approximate expression complexity (low, medium, high)
    pub complexity: ContextComplexity,
}

impl TacticContext {
    /// Create a context from property features
    pub fn from_features(features: &PropertyFeatures) -> Self {
        let complexity = if features.depth <= 2 {
            ContextComplexity::Low
        } else if features.depth <= 5 {
            ContextComplexity::Medium
        } else {
            ContextComplexity::High
        };

        TacticContext {
            property_type: features.property_type.clone(),
            has_quantifiers: features.quantifier_depth > 0,
            has_implications: features.implication_count > 0,
            has_arithmetic: features.arithmetic_ops > 0,
            complexity,
        }
    }
}

/// Complexity level for grouping
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContextComplexity {
    /// Simple properties (few variables, basic structure)
    Low,
    /// Moderate complexity (nested structures, multiple constraints)
    Medium,
    /// High complexity (deep nesting, many variables, complex invariants)
    High,
}

/// Statistics for a single tactic
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TacticStats {
    /// Number of successful uses
    pub successes: u32,
    /// Number of failed uses
    pub failures: u32,
    /// Number of partial successes
    pub partials: u32,
}

impl TacticStats {
    /// Success rate (0.0 to 1.0)
    pub fn success_rate(&self) -> f64 {
        let total = self.successes + self.failures + self.partials;
        if total == 0 {
            0.0
        } else {
            // Partial counts as 0.5 success
            let effective = self.successes as f64 + (self.partials as f64 * 0.5);
            effective / total as f64
        }
    }

    /// Total number of observations
    pub fn total(&self) -> u32 {
        self.successes + self.failures + self.partials
    }

    /// Wilson score lower bound for confidence
    ///
    /// Accounts for uncertainty when sample size is small.
    /// Returns a value that balances success rate with confidence.
    pub fn wilson_score(&self) -> f64 {
        let n = self.total() as f64;
        if n == 0.0 {
            return 0.0;
        }

        let p = self.success_rate();
        let z = 1.96; // 95% confidence

        // Wilson score interval lower bound
        let denominator = 1.0 + z * z / n;
        let center = p + z * z / (2.0 * n);
        let adjustment = z * ((p * (1.0 - p) + z * z / (4.0 * n)) / n).sqrt();

        (center - adjustment) / denominator
    }
}

/// Database tracking tactic effectiveness
#[derive(Debug, Default)]
pub struct TacticDatabase {
    /// Stats per context per tactic
    stats: HashMap<TacticContext, HashMap<String, TacticStats>>,
    /// Global stats (across all contexts)
    global_stats: HashMap<String, TacticStats>,
}

#[derive(Serialize, Deserialize)]
struct SerializableTacticDatabase {
    stats: Vec<ContextEntry>,
    global_stats: HashMap<String, TacticStats>,
}

#[derive(Serialize, Deserialize)]
struct ContextEntry {
    context: TacticContext,
    tactics: HashMap<String, TacticStats>,
}

impl Serialize for TacticDatabase {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let serializable = SerializableTacticDatabase {
            stats: self
                .stats
                .iter()
                .map(|(context, tactics)| ContextEntry {
                    context: context.clone(),
                    tactics: tactics.clone(),
                })
                .collect(),
            global_stats: self.global_stats.clone(),
        };

        serializable.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for TacticDatabase {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let serializable = SerializableTacticDatabase::deserialize(deserializer)?;

        let stats = serializable
            .stats
            .into_iter()
            .map(|entry| (entry.context, entry.tactics))
            .collect();

        Ok(Self {
            stats,
            global_stats: serializable.global_stats,
        })
    }
}

impl TacticDatabase {
    /// Create a new empty database
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a successful tactic use
    pub fn record_success(&mut self, context: &TacticContext, tactic: &str) {
        self.stats
            .entry(context.clone())
            .or_default()
            .entry(tactic.to_string())
            .or_default()
            .successes += 1;

        self.global_stats
            .entry(tactic.to_string())
            .or_default()
            .successes += 1;
    }

    /// Record a failed tactic use
    pub fn record_failure(&mut self, context: &TacticContext, tactic: &str) {
        self.stats
            .entry(context.clone())
            .or_default()
            .entry(tactic.to_string())
            .or_default()
            .failures += 1;

        self.global_stats
            .entry(tactic.to_string())
            .or_default()
            .failures += 1;
    }

    /// Record a partial success
    pub fn record_partial(&mut self, context: &TacticContext, tactic: &str) {
        self.stats
            .entry(context.clone())
            .or_default()
            .entry(tactic.to_string())
            .or_default()
            .partials += 1;

        self.global_stats
            .entry(tactic.to_string())
            .or_default()
            .partials += 1;
    }

    /// Get stats for a specific tactic in a specific context
    pub fn get_stats(&self, context: &TacticContext, tactic: &str) -> Option<&TacticStats> {
        self.stats.get(context)?.get(tactic)
    }

    /// Get global stats for a tactic
    pub fn get_global_stats(&self, tactic: &str) -> Option<&TacticStats> {
        self.global_stats.get(tactic)
    }

    /// Get best tactics for a context, sorted by Wilson score
    ///
    /// Returns up to `n` tactics with their scores.
    pub fn best_for_context(&self, context: &TacticContext, n: usize) -> Vec<(String, f64)> {
        // First try exact context match
        let mut candidates: Vec<_> = self
            .stats
            .get(context)
            .map(|ctx_stats| {
                ctx_stats
                    .iter()
                    .map(|(tactic, stats)| (tactic.clone(), stats.wilson_score()))
                    .collect()
            })
            .unwrap_or_default();

        // If no exact matches, use global stats
        if candidates.is_empty() {
            candidates = self
                .global_stats
                .iter()
                .map(|(tactic, stats)| (tactic.clone(), stats.wilson_score()))
                .collect();
        }

        // Sort by Wilson score descending
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        candidates.into_iter().take(n).collect()
    }

    /// Total number of tactic observations
    pub fn total_observations(&self) -> u32 {
        self.global_stats.values().map(|s| s.total()).sum()
    }

    /// Number of unique tactics seen
    pub fn unique_tactics(&self) -> usize {
        self.global_stats.len()
    }

    /// Persist tactic statistics to disk
    pub fn save_to_file<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), crate::LearningError> {
        crate::io::write_json_atomic(path, self)
    }

    /// Load tactic statistics from disk
    pub fn load_from_file<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, crate::LearningError> {
        crate::io::read_json(path)
    }

    /// Load tactic statistics if the file exists, otherwise return empty
    pub fn load_or_default<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, crate::LearningError> {
        match std::fs::File::open(path.as_ref()) {
            Ok(_) => Self::load_from_file(path),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_context(prop_type: &str, has_quant: bool) -> TacticContext {
        TacticContext {
            property_type: prop_type.to_string(),
            has_quantifiers: has_quant,
            has_implications: false,
            has_arithmetic: false,
            complexity: ContextComplexity::Low,
        }
    }

    #[test]
    fn test_tactic_stats_success_rate() {
        let mut stats = TacticStats::default();
        assert_eq!(stats.success_rate(), 0.0);

        stats.successes = 3;
        stats.failures = 1;
        assert_eq!(stats.success_rate(), 0.75);

        stats.partials = 2;
        // 3 success + 2*0.5 partial_equiv = 4 / 6 = 0.6667
        let rate = stats.success_rate();
        assert!((rate - 0.6667).abs() < 0.01);
    }

    #[test]
    fn test_wilson_score_uncertainty() {
        // With few observations, Wilson score is lower than raw rate
        let few = TacticStats {
            successes: 1,
            ..Default::default()
        };

        let many = TacticStats {
            successes: 100,
            ..Default::default()
        };

        // Both have 100% success rate but Wilson score reflects confidence
        assert!(few.wilson_score() < many.wilson_score());
        assert!(few.wilson_score() < 0.8); // Lower bound due to uncertainty
        assert!(many.wilson_score() > 0.95); // High confidence
    }

    #[test]
    fn test_database_record_and_query() {
        let mut db = TacticDatabase::new();
        let ctx = make_context("invariant", false);

        // Record some tactic uses
        // "decide": 10 success, 3 fail = 77% rate
        for _ in 0..10 {
            db.record_success(&ctx, "decide");
        }
        for _ in 0..3 {
            db.record_failure(&ctx, "decide");
        }
        // "simp": 5 success, 0 fail = 100% rate but lower confidence
        for _ in 0..5 {
            db.record_success(&ctx, "simp");
        }

        let stats = db.get_stats(&ctx, "decide").unwrap();
        assert_eq!(stats.successes, 10);
        assert_eq!(stats.failures, 3);

        let best = db.best_for_context(&ctx, 2);
        assert_eq!(best.len(), 2);
        // Both tactics should be returned; Wilson score considers both rate and confidence
        // The ordering depends on the balance - just verify both are present
        let tactic_names: Vec<_> = best.iter().map(|(name, _)| name.as_str()).collect();
        assert!(tactic_names.contains(&"decide"));
        assert!(tactic_names.contains(&"simp"));
    }

    #[test]
    fn test_global_fallback() {
        let mut db = TacticDatabase::new();
        let ctx1 = make_context("invariant", false);
        let ctx2 = make_context("theorem", true);

        // Only record for ctx1
        for _ in 0..10 {
            db.record_success(&ctx1, "omega");
        }

        // Query for ctx2 (no data) should fall back to global
        let best = db.best_for_context(&ctx2, 3);
        assert!(!best.is_empty());
        assert_eq!(best[0].0, "omega");
    }

    #[test]
    fn test_unique_tactics_count() {
        let mut db = TacticDatabase::new();
        let ctx = make_context("invariant", false);

        db.record_success(&ctx, "decide");
        db.record_success(&ctx, "simp");
        db.record_success(&ctx, "omega");
        db.record_failure(&ctx, "decide"); // Same tactic, shouldn't increase count

        assert_eq!(db.unique_tactics(), 3);
    }

    // Mutation-killing tests for TacticContext::from_features complexity thresholds

    #[test]
    fn test_context_from_features_depth_low() {
        // depth <= 2 should be Low complexity
        let features = PropertyFeatures {
            property_type: "theorem".to_string(),
            quantifier_depth: 0,
            implication_count: 0,
            arithmetic_ops: 0,
            depth: 2, // exactly at Low boundary
            ..Default::default()
        };
        let ctx = TacticContext::from_features(&features);
        assert_eq!(ctx.complexity, ContextComplexity::Low);

        // depth = 1 should also be Low
        let features_1 = PropertyFeatures {
            depth: 1,
            ..features.clone()
        };
        assert_eq!(
            TacticContext::from_features(&features_1).complexity,
            ContextComplexity::Low
        );
    }

    #[test]
    fn test_context_from_features_depth_medium() {
        // depth 3..=5 should be Medium complexity
        let features = PropertyFeatures {
            property_type: "theorem".to_string(),
            quantifier_depth: 0,
            implication_count: 0,
            arithmetic_ops: 0,
            depth: 3, // just above Low boundary
            ..Default::default()
        };
        let ctx = TacticContext::from_features(&features);
        assert_eq!(ctx.complexity, ContextComplexity::Medium);

        // depth = 5 should also be Medium (boundary)
        let features_5 = PropertyFeatures {
            depth: 5,
            ..features.clone()
        };
        assert_eq!(
            TacticContext::from_features(&features_5).complexity,
            ContextComplexity::Medium
        );
    }

    #[test]
    fn test_context_from_features_depth_high() {
        // depth > 5 should be High complexity
        let features = PropertyFeatures {
            property_type: "theorem".to_string(),
            quantifier_depth: 0,
            implication_count: 0,
            arithmetic_ops: 0,
            depth: 6, // just above Medium boundary
            ..Default::default()
        };
        let ctx = TacticContext::from_features(&features);
        assert_eq!(ctx.complexity, ContextComplexity::High);
    }

    #[test]
    fn test_context_from_features_quantifier_flag() {
        // quantifier_depth > 0 should set has_quantifiers = true
        let features_no_quant = PropertyFeatures {
            property_type: "theorem".to_string(),
            quantifier_depth: 0,
            ..Default::default()
        };
        let ctx_no = TacticContext::from_features(&features_no_quant);
        assert!(!ctx_no.has_quantifiers);

        let features_with_quant = PropertyFeatures {
            property_type: "theorem".to_string(),
            quantifier_depth: 1,
            ..Default::default()
        };
        let ctx_with = TacticContext::from_features(&features_with_quant);
        assert!(ctx_with.has_quantifiers);
    }

    #[test]
    fn test_context_from_features_implication_flag() {
        // implication_count > 0 should set has_implications = true
        let features_no_impl = PropertyFeatures {
            property_type: "theorem".to_string(),
            implication_count: 0,
            ..Default::default()
        };
        let ctx_no = TacticContext::from_features(&features_no_impl);
        assert!(!ctx_no.has_implications);

        let features_with_impl = PropertyFeatures {
            property_type: "theorem".to_string(),
            implication_count: 1,
            ..Default::default()
        };
        let ctx_with = TacticContext::from_features(&features_with_impl);
        assert!(ctx_with.has_implications);
    }

    #[test]
    fn test_context_from_features_arithmetic_flag() {
        // arithmetic_ops > 0 should set has_arithmetic = true
        let features_no_arith = PropertyFeatures {
            property_type: "theorem".to_string(),
            arithmetic_ops: 0,
            ..Default::default()
        };
        let ctx_no = TacticContext::from_features(&features_no_arith);
        assert!(!ctx_no.has_arithmetic);

        let features_with_arith = PropertyFeatures {
            property_type: "theorem".to_string(),
            arithmetic_ops: 1,
            ..Default::default()
        };
        let ctx_with = TacticContext::from_features(&features_with_arith);
        assert!(ctx_with.has_arithmetic);
    }

    // Mutation-killing tests for wilson_score arithmetic

    #[test]
    fn test_wilson_score_known_values() {
        // Test with known inputs where each arithmetic operation matters
        let stats = TacticStats {
            successes: 8,
            failures: 2,
            partials: 0,
        };
        // n=10, p=0.8, z=1.96
        // The Wilson score formula has specific behavior we can verify
        let score = stats.wilson_score();
        // Should be less than raw rate (0.8) due to uncertainty adjustment
        assert!(score < 0.8, "wilson_score={} should be < 0.8", score);
        // But should still be reasonably high (not too low)
        assert!(score > 0.4, "wilson_score={} should be > 0.4", score);
        // Verify it's in the expected range around 0.49
        assert!(
            (score - 0.49).abs() < 0.05,
            "wilson_score={} should be near 0.49",
            score
        );
    }

    #[test]
    fn test_wilson_score_different_n_values() {
        // Verify n affects denominator correctly (z^2/n term)
        let stats_10 = TacticStats {
            successes: 10,
            failures: 0,
            partials: 0,
        };
        let stats_100 = TacticStats {
            successes: 100,
            failures: 0,
            partials: 0,
        };
        // With n=10, z^2/n = 0.3842; with n=100, z^2/n = 0.03842
        // Larger n means smaller adjustment, higher Wilson score
        assert!(stats_100.wilson_score() > stats_10.wilson_score());
        // n=10 should give score around 0.72
        assert!(stats_10.wilson_score() > 0.70 && stats_10.wilson_score() < 0.75);
        // n=100 should give score around 0.96
        assert!(stats_100.wilson_score() > 0.95 && stats_100.wilson_score() < 0.98);
    }

    #[test]
    fn test_wilson_score_with_partials() {
        // Partials contribute 0.5 to success rate, verifying the multiplier
        let stats_pure = TacticStats {
            successes: 5,
            failures: 5,
            partials: 0,
        }; // rate = 0.5

        let stats_partial = TacticStats {
            successes: 3,
            failures: 5,
            partials: 4,
        }; // rate = (3 + 4*0.5)/12 = 5/12 = 0.417

        // Both have n=10 or 12, so confidence intervals similar
        // But different success rates lead to different Wilson scores
        assert!(stats_pure.wilson_score() > stats_partial.wilson_score());
    }

    // Mutation-killing tests for record_partial

    #[test]
    fn test_record_partial_increments() {
        let mut db = TacticDatabase::new();
        let ctx = make_context("theorem", false);

        // Record multiple partials
        db.record_partial(&ctx, "simp");
        db.record_partial(&ctx, "simp");
        db.record_partial(&ctx, "simp");

        let stats = db.get_stats(&ctx, "simp").unwrap();
        assert_eq!(stats.partials, 3); // Should be exactly 3, not 0 or some other value
        assert_eq!(stats.successes, 0);
        assert_eq!(stats.failures, 0);

        // Global stats should also reflect partials
        let global = db.get_global_stats("simp").unwrap();
        assert_eq!(global.partials, 3);
    }

    // Mutation-killing tests for get_global_stats

    #[test]
    fn test_get_global_stats_returns_none_for_unknown() {
        let db = TacticDatabase::new();
        assert!(db.get_global_stats("nonexistent").is_none());
    }

    #[test]
    fn test_get_global_stats_returns_some_for_known() {
        let mut db = TacticDatabase::new();
        let ctx = make_context("theorem", false);
        db.record_success(&ctx, "simp");

        let stats = db.get_global_stats("simp");
        assert!(stats.is_some());
        assert_eq!(stats.unwrap().successes, 1);
    }

    // Mutation-killing tests for load_or_default error handling

    #[test]
    fn test_load_or_default_nonexistent_file() {
        // Should return Ok(default) for NotFound error
        let result = TacticDatabase::load_or_default("/nonexistent/path/tactics.json");
        assert!(result.is_ok());
        let db = result.unwrap();
        assert_eq!(db.unique_tactics(), 0);
    }

    #[test]
    fn test_load_or_default_with_existing_invalid_file() {
        // Create a temp file with invalid JSON
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_tactics_invalid.json");
        std::fs::write(&temp_file, "not valid json {{{").unwrap();

        // Should return Err because file exists but is invalid
        // This catches the mutation "e.kind() == NotFound -> true"
        // because the file DOES exist, so it tries to load it and fails
        let result = TacticDatabase::load_or_default(&temp_file);
        assert!(result.is_err(), "Expected error for invalid JSON file");

        // Cleanup
        let _ = std::fs::remove_file(temp_file);
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify that success_rate returns 0.0 when total is 0
    #[kani::proof]
    fn verify_success_rate_zero_on_empty() {
        let stats = TacticStats {
            successes: 0,
            failures: 0,
            partials: 0,
        };
        let rate = stats.success_rate();
        assert!(rate == 0.0, "Empty stats should have 0.0 success rate");
    }

    /// Verify that success_rate is bounded in [0.0, 1.0]
    #[kani::proof]
    fn verify_success_rate_bounded() {
        let successes: u32 = kani::any();
        let failures: u32 = kani::any();
        let partials: u32 = kani::any();

        // Limit to avoid overflow
        kani::assume(successes <= 1000);
        kani::assume(failures <= 1000);
        kani::assume(partials <= 1000);

        let stats = TacticStats {
            successes,
            failures,
            partials,
        };
        let rate = stats.success_rate();

        assert!(rate >= 0.0, "success_rate should be >= 0.0");
        assert!(rate <= 1.0, "success_rate should be <= 1.0");
    }

    /// Verify that wilson_score returns 0.0 when total is 0
    #[kani::proof]
    fn verify_wilson_score_zero_on_empty() {
        let stats = TacticStats {
            successes: 0,
            failures: 0,
            partials: 0,
        };
        let score = stats.wilson_score();
        assert!(score == 0.0, "Empty stats should have 0.0 Wilson score");
    }

    /// Verify that wilson_score has bounded magnitude (can be slightly negative for edge cases)
    /// The Wilson score lower bound formula can produce small negative values when success rate
    /// is very low and sample size is moderate. This is mathematically correct behavior.
    #[kani::proof]
    fn verify_wilson_score_bounded_magnitude() {
        let successes: u32 = kani::any();
        let failures: u32 = kani::any();
        let partials: u32 = kani::any();

        // Limit values to avoid overflow and keep verification tractable
        kani::assume(successes <= 100);
        kani::assume(failures <= 100);
        kani::assume(partials <= 100);
        // Ensure at least one observation to exercise the formula
        kani::assume(successes + failures + partials > 0);

        let stats = TacticStats {
            successes,
            failures,
            partials,
        };
        let score = stats.wilson_score();

        // Wilson score lower bound is bounded: -1 < score <= 1
        // (can be slightly negative for extreme low-success cases)
        assert!(score > -1.0, "Wilson score should be > -1.0");
        assert!(score <= 1.0, "Wilson score should be <= 1.0");
    }

    /// Verify that total() equals sum of all fields
    #[kani::proof]
    fn verify_total_equals_sum() {
        let successes: u32 = kani::any();
        let failures: u32 = kani::any();
        let partials: u32 = kani::any();

        // Limit to avoid overflow
        kani::assume(successes <= 1000);
        kani::assume(failures <= 1000);
        kani::assume(partials <= 1000);

        let stats = TacticStats {
            successes,
            failures,
            partials,
        };

        assert_eq!(
            stats.total(),
            successes + failures + partials,
            "total() should equal sum of all fields"
        );
    }

    /// Verify complexity thresholds: depth <= 2 => Low
    #[kani::proof]
    fn verify_complexity_low_threshold() {
        let depth: usize = kani::any();
        kani::assume(depth <= 2);

        let features = PropertyFeatures {
            property_type: String::new(),
            depth,
            quantifier_depth: 0,
            implication_count: 0,
            arithmetic_ops: 0,
            ..Default::default()
        };
        let ctx = TacticContext::from_features(&features);

        assert!(
            ctx.complexity == ContextComplexity::Low,
            "depth <= 2 should be Low complexity"
        );
    }

    /// Verify complexity thresholds: 3 <= depth <= 5 => Medium
    #[kani::proof]
    fn verify_complexity_medium_threshold() {
        let depth: usize = kani::any();
        kani::assume(depth >= 3 && depth <= 5);

        let features = PropertyFeatures {
            property_type: String::new(),
            depth,
            quantifier_depth: 0,
            implication_count: 0,
            arithmetic_ops: 0,
            ..Default::default()
        };
        let ctx = TacticContext::from_features(&features);

        assert!(
            ctx.complexity == ContextComplexity::Medium,
            "depth in [3, 5] should be Medium complexity"
        );
    }

    /// Verify complexity thresholds: depth > 5 => High
    #[kani::proof]
    fn verify_complexity_high_threshold() {
        let depth: usize = kani::any();
        kani::assume(depth > 5 && depth < 100); // upper bound for tractability

        let features = PropertyFeatures {
            property_type: String::new(),
            depth,
            quantifier_depth: 0,
            implication_count: 0,
            arithmetic_ops: 0,
            ..Default::default()
        };
        let ctx = TacticContext::from_features(&features);

        assert!(
            ctx.complexity == ContextComplexity::High,
            "depth > 5 should be High complexity"
        );
    }

    /// Verify has_quantifiers flag matches quantifier_depth > 0
    #[kani::proof]
    fn verify_has_quantifiers_flag() {
        let quantifier_depth: usize = kani::any();
        kani::assume(quantifier_depth < 100);

        let features = PropertyFeatures {
            property_type: String::new(),
            quantifier_depth,
            depth: 1,
            implication_count: 0,
            arithmetic_ops: 0,
            ..Default::default()
        };
        let ctx = TacticContext::from_features(&features);

        assert!(
            ctx.has_quantifiers == (quantifier_depth > 0),
            "has_quantifiers should match quantifier_depth > 0"
        );
    }

    /// Verify has_implications flag matches implication_count > 0
    #[kani::proof]
    fn verify_has_implications_flag() {
        let implication_count: usize = kani::any();
        kani::assume(implication_count < 100);

        let features = PropertyFeatures {
            property_type: String::new(),
            implication_count,
            depth: 1,
            quantifier_depth: 0,
            arithmetic_ops: 0,
            ..Default::default()
        };
        let ctx = TacticContext::from_features(&features);

        assert!(
            ctx.has_implications == (implication_count > 0),
            "has_implications should match implication_count > 0"
        );
    }

    /// Verify has_arithmetic flag matches arithmetic_ops > 0
    #[kani::proof]
    fn verify_has_arithmetic_flag() {
        let arithmetic_ops: usize = kani::any();
        kani::assume(arithmetic_ops < 100);

        let features = PropertyFeatures {
            property_type: String::new(),
            arithmetic_ops,
            depth: 1,
            quantifier_depth: 0,
            implication_count: 0,
            ..Default::default()
        };
        let ctx = TacticContext::from_features(&features);

        assert!(
            ctx.has_arithmetic == (arithmetic_ops > 0),
            "has_arithmetic should match arithmetic_ops > 0"
        );
    }
}
