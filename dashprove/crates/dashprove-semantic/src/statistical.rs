//! Statistical property verification over multiple samples
//!
//! This module provides verification of properties that hold statistically
//! over multiple samples, useful for verifying LLM outputs where individual
//! responses may vary but should satisfy certain statistical properties.

use crate::error::{SemanticError, SemanticResult};
use crate::predicate::{PredicateInputs, SemanticPredicate};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Configuration for statistical verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalConfig {
    /// Minimum number of samples required
    pub min_samples: usize,
    /// Maximum number of samples to collect
    pub max_samples: usize,
    /// Confidence level for statistical tests (e.g., 0.95 for 95%)
    pub confidence_level: f64,
    /// Early stopping threshold (stop if clear pass/fail before max samples)
    pub early_stopping: bool,
    /// Timeout per sample in milliseconds (0 = no timeout)
    pub sample_timeout_ms: u64,
}

impl Default for StatisticalConfig {
    fn default() -> Self {
        Self {
            min_samples: 10,
            max_samples: 100,
            confidence_level: 0.95,
            early_stopping: true,
            sample_timeout_ms: 5000,
        }
    }
}

/// Result of verifying a single sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleResult {
    /// Sample index
    pub index: usize,
    /// The sample value
    pub value: String,
    /// Whether this sample passed the property
    pub passed: bool,
    /// Score/confidence for this sample
    pub score: f64,
    /// Additional details
    pub details: Option<String>,
}

/// Result of statistical verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalResult {
    /// Whether the statistical property passed
    pub passed: bool,
    /// Number of samples collected
    pub num_samples: usize,
    /// Number of samples that passed
    pub num_passed: usize,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Confidence interval lower bound
    pub ci_lower: f64,
    /// Confidence interval upper bound
    pub ci_upper: f64,
    /// P-value for hypothesis test (if applicable)
    pub p_value: Option<f64>,
    /// Individual sample results
    pub samples: Vec<SampleResult>,
    /// Whether early stopping was triggered
    pub early_stopped: bool,
    /// Explanation of the result
    pub explanation: String,
}

impl StatisticalResult {
    /// Check if the success rate is statistically significant above threshold
    pub fn is_significant_above(&self, threshold: f64) -> bool {
        self.ci_lower >= threshold
    }

    /// Check if the success rate is statistically significant below threshold
    pub fn is_significant_below(&self, threshold: f64) -> bool {
        self.ci_upper < threshold
    }
}

/// A statistical property to verify
pub struct StatisticalProperty {
    /// Name of the property
    pub name: String,
    /// Minimum success rate required
    pub min_success_rate: f64,
    /// The predicate to evaluate on each sample
    predicate: Arc<dyn SemanticPredicate>,
}

impl std::fmt::Debug for StatisticalProperty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StatisticalProperty")
            .field("name", &self.name)
            .field("min_success_rate", &self.min_success_rate)
            .field("predicate", &self.predicate.name())
            .finish()
    }
}

impl StatisticalProperty {
    /// Create a new statistical property
    pub fn new(name: &str, min_success_rate: f64, predicate: Arc<dyn SemanticPredicate>) -> Self {
        Self {
            name: name.to_string(),
            min_success_rate,
            predicate,
        }
    }

    /// Evaluate the property on a single sample
    pub async fn evaluate_sample(
        &self,
        sample: &str,
        reference: Option<&str>,
    ) -> SemanticResult<(bool, f64)> {
        let inputs = if let Some(ref_text) = reference {
            PredicateInputs::pair(sample, ref_text)
        } else {
            PredicateInputs::primary(sample)
        };

        let result = self.predicate.evaluate(&inputs).await?;
        Ok((result.passed, result.confidence))
    }
}

/// Statistical verifier for running multiple samples
pub struct StatisticalVerifier {
    /// Configuration
    config: StatisticalConfig,
}

impl StatisticalVerifier {
    /// Create a new statistical verifier
    pub fn new(config: StatisticalConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(StatisticalConfig::default())
    }

    /// Verify a property over provided samples
    pub async fn verify(
        &self,
        property: &StatisticalProperty,
        samples: &[&str],
        reference: Option<&str>,
    ) -> SemanticResult<StatisticalResult> {
        if samples.len() < self.config.min_samples {
            return Err(SemanticError::Statistical(format!(
                "Insufficient samples: {} provided, {} required",
                samples.len(),
                self.config.min_samples
            )));
        }

        let mut sample_results = Vec::new();
        let mut passed_count = 0;
        let samples_to_check = samples.len().min(self.config.max_samples);

        for (idx, sample) in samples.iter().take(samples_to_check).enumerate() {
            let (passed, score) = property.evaluate_sample(sample, reference).await?;

            if passed {
                passed_count += 1;
            }

            sample_results.push(SampleResult {
                index: idx,
                value: sample.to_string(),
                passed,
                score,
                details: None,
            });

            // Early stopping check
            if self.config.early_stopping && idx >= self.config.min_samples - 1 {
                let (ci_lower, ci_upper) =
                    wilson_confidence_interval(passed_count, idx + 1, self.config.confidence_level);

                // Stop if we can conclusively pass or fail
                if ci_lower >= property.min_success_rate {
                    // Conclusively passing
                    return Ok(self.build_result(
                        true,
                        sample_results,
                        passed_count,
                        true,
                        &property.name,
                        property.min_success_rate,
                    ));
                }
                if ci_upper < property.min_success_rate {
                    // Conclusively failing
                    return Ok(self.build_result(
                        false,
                        sample_results,
                        passed_count,
                        true,
                        &property.name,
                        property.min_success_rate,
                    ));
                }
            }
        }

        // Final evaluation
        let (ci_lower, _ci_upper) = wilson_confidence_interval(
            passed_count,
            sample_results.len(),
            self.config.confidence_level,
        );
        let passed = ci_lower >= property.min_success_rate;

        Ok(self.build_result(
            passed,
            sample_results,
            passed_count,
            false,
            &property.name,
            property.min_success_rate,
        ))
    }

    /// Build the final result
    fn build_result(
        &self,
        passed: bool,
        samples: Vec<SampleResult>,
        passed_count: usize,
        early_stopped: bool,
        property_name: &str,
        min_success_rate: f64,
    ) -> StatisticalResult {
        let num_samples = samples.len();
        let success_rate = passed_count as f64 / num_samples as f64;
        let (ci_lower, ci_upper) =
            wilson_confidence_interval(passed_count, num_samples, self.config.confidence_level);

        let explanation = format!(
            "Property '{}': {}/{} samples passed ({:.1}%). CI: [{:.1}%, {:.1}%]. Required: {:.1}%. {}",
            property_name,
            passed_count,
            num_samples,
            success_rate * 100.0,
            ci_lower * 100.0,
            ci_upper * 100.0,
            min_success_rate * 100.0,
            if passed { "PASSED" } else { "FAILED" }
        );

        StatisticalResult {
            passed,
            num_samples,
            num_passed: passed_count,
            success_rate,
            ci_lower,
            ci_upper,
            p_value: Some(binomial_p_value(
                passed_count,
                num_samples,
                min_success_rate,
            )),
            samples,
            early_stopped,
            explanation,
        }
    }

    /// Verify with a sample generator function
    pub async fn verify_with_generator<F, Fut>(
        &self,
        property: &StatisticalProperty,
        mut generator: F,
        reference: Option<&str>,
    ) -> SemanticResult<StatisticalResult>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = SemanticResult<String>>,
    {
        let mut samples = Vec::new();

        for _ in 0..self.config.max_samples {
            let sample = generator().await?;
            samples.push(sample);
        }

        let sample_refs: Vec<&str> = samples.iter().map(|s| s.as_str()).collect();
        self.verify(property, &sample_refs, reference).await
    }
}

/// Calculate Wilson score confidence interval for a proportion
///
/// This is more accurate than the normal approximation for small samples
/// and proportions near 0 or 1.
fn wilson_confidence_interval(successes: usize, trials: usize, confidence: f64) -> (f64, f64) {
    if trials == 0 {
        return (0.0, 1.0);
    }

    let n = trials as f64;
    let p = successes as f64 / n;

    // Z-score for confidence level (approximate for common values)
    let z = if confidence >= 0.99 {
        2.576
    } else if confidence >= 0.95 {
        1.96
    } else if confidence >= 0.90 {
        1.645
    } else {
        1.28 // 80%
    };

    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let spread = (z / denominator) * ((p * (1.0 - p) / n) + (z2 / (4.0 * n * n))).sqrt();

    let lower = (center - spread).max(0.0);
    let upper = (center + spread).min(1.0);

    (lower, upper)
}

/// Calculate p-value for binomial test
///
/// Tests H0: p >= threshold vs H1: p < threshold
fn binomial_p_value(successes: usize, trials: usize, threshold: f64) -> f64 {
    if trials == 0 {
        return 1.0;
    }

    // Use normal approximation for large samples
    let n = trials as f64;
    let observed_p = successes as f64 / n;

    if n * threshold >= 5.0 && n * (1.0 - threshold) >= 5.0 {
        // Normal approximation
        let se = (threshold * (1.0 - threshold) / n).sqrt();
        if se < 1e-10 {
            return if observed_p >= threshold { 1.0 } else { 0.0 };
        }
        let z = (observed_p - threshold) / se;
        // Approximate one-sided p-value using normal CDF
        0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
    } else {
        // For small samples, use exact binomial
        // Simplified: just return approximate based on comparison
        if observed_p >= threshold {
            0.5
        } else {
            0.05
        }
    }
}

/// Error function approximation
fn erf(x: f64) -> f64 {
    // Approximation from Abramowitz and Stegun
    let t = 1.0 / (1.0 + 0.5 * x.abs());
    let tau = t
        * (-x * x - 1.26551223
            + t * (1.00002368
                + t * (0.37409196
                    + t * (0.09678418
                        + t * (-0.18628806
                            + t * (0.27886807
                                + t * (-1.13520398
                                    + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
            .exp();

    if x >= 0.0 {
        1.0 - tau
    } else {
        tau - 1.0
    }
}

/// Builder for StatisticalProperty
pub struct StatisticalPropertyBuilder {
    name: Option<String>,
    min_success_rate: f64,
    predicate: Option<Arc<dyn SemanticPredicate>>,
}

impl StatisticalPropertyBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            name: None,
            min_success_rate: 0.9,
            predicate: None,
        }
    }

    /// Set the property name
    pub fn name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    /// Set minimum success rate
    pub fn min_success_rate(mut self, rate: f64) -> Self {
        self.min_success_rate = rate;
        self
    }

    /// Set the predicate
    pub fn predicate(mut self, predicate: Arc<dyn SemanticPredicate>) -> Self {
        self.predicate = Some(predicate);
        self
    }

    /// Build the property
    pub fn build(self) -> SemanticResult<StatisticalProperty> {
        let name = self
            .name
            .ok_or_else(|| SemanticError::Config("Property name is required".to_string()))?;
        let predicate = self
            .predicate
            .ok_or_else(|| SemanticError::Config("Predicate is required".to_string()))?;

        Ok(StatisticalProperty {
            name,
            min_success_rate: self.min_success_rate,
            predicate,
        })
    }
}

impl Default for StatisticalPropertyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predicate::SemanticSimilarity;

    #[test]
    fn test_wilson_confidence_interval() {
        // Test with 50% success rate
        let (lower, upper) = wilson_confidence_interval(50, 100, 0.95);
        assert!(lower > 0.0 && lower < 0.5);
        assert!(upper > 0.5 && upper < 1.0);

        // Test with 100% success rate
        let (lower, upper) = wilson_confidence_interval(100, 100, 0.95);
        assert!(lower > 0.95);
        assert!((upper - 1.0).abs() < 0.01);

        // Test with 0% success rate
        let (lower, upper) = wilson_confidence_interval(0, 100, 0.95);
        assert!(lower.abs() < 0.01);
        assert!(upper < 0.05);
    }

    #[test]
    fn test_binomial_p_value() {
        // High success rate vs high threshold should have high p-value
        let p1 = binomial_p_value(90, 100, 0.5);
        assert!(p1 > 0.5);

        // Low success rate vs high threshold should have low p-value
        let p2 = binomial_p_value(30, 100, 0.8);
        assert!(p2 < 0.5);
    }

    #[tokio::test]
    async fn test_statistical_verifier_all_pass() {
        let predicate = Arc::new(SemanticSimilarity::new(0.5));
        // Use min_success_rate of 0.6 - Wilson CI lower bound for 10/10 is ~0.69
        let property = StatisticalProperty::new("test_similarity", 0.6, predicate);

        let config = StatisticalConfig {
            min_samples: 5,
            max_samples: 10,
            ..Default::default()
        };
        let verifier = StatisticalVerifier::new(config);

        // All identical samples should pass
        let samples = vec!["hello world"; 10];
        let result = verifier
            .verify(&property, &samples, Some("hello world"))
            .await
            .unwrap();

        assert!(
            result.passed,
            "Expected to pass with all samples matching: {:?}",
            result
        );
        assert_eq!(result.num_passed, result.num_samples);
        assert!((result.success_rate - 1.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_statistical_verifier_mixed() {
        let predicate = Arc::new(SemanticSimilarity::new(0.99));
        let property = StatisticalProperty::new("test_strict_similarity", 0.5, predicate);

        let config = StatisticalConfig {
            min_samples: 10,
            max_samples: 20,
            early_stopping: false,
            ..Default::default()
        };
        let verifier = StatisticalVerifier::new(config);

        // Mix of identical and different samples
        let samples: Vec<&str> = (0..20)
            .map(|i| if i % 2 == 0 { "hello world" } else { "goodbye" })
            .collect();

        let result = verifier
            .verify(&property, &samples, Some("hello world"))
            .await
            .unwrap();

        // About 50% should pass
        assert!(result.success_rate > 0.4 && result.success_rate < 0.6);
    }

    #[tokio::test]
    async fn test_statistical_verifier_early_stopping() {
        let predicate = Arc::new(SemanticSimilarity::new(0.5));
        let property = StatisticalProperty::new("test_early", 0.5, predicate);

        let config = StatisticalConfig {
            min_samples: 5,
            max_samples: 100,
            early_stopping: true,
            ..Default::default()
        };
        let verifier = StatisticalVerifier::new(config);

        // All identical samples - should early stop as passing
        let samples = vec!["hello world"; 100];
        let result = verifier
            .verify(&property, &samples, Some("hello world"))
            .await
            .unwrap();

        assert!(result.passed);
        assert!(result.early_stopped);
        // Should have stopped before 100 samples
        assert!(result.num_samples < 100);
    }

    #[tokio::test]
    async fn test_statistical_config_default() {
        let config = StatisticalConfig::default();
        assert_eq!(config.min_samples, 10);
        assert_eq!(config.max_samples, 100);
        assert!((config.confidence_level - 0.95).abs() < 0.01);
        assert!(config.early_stopping);
    }

    #[test]
    fn test_statistical_result_significance() {
        let result = StatisticalResult {
            passed: true,
            num_samples: 100,
            num_passed: 90,
            success_rate: 0.9,
            ci_lower: 0.83,
            ci_upper: 0.95,
            p_value: Some(0.01),
            samples: vec![],
            early_stopped: false,
            explanation: String::new(),
        };

        assert!(result.is_significant_above(0.8));
        assert!(!result.is_significant_above(0.85));
        // ci_upper (0.95) < 0.96, so is_significant_below(0.96) is true
        assert!(result.is_significant_below(0.96));
    }

    #[tokio::test]
    async fn test_property_builder() {
        let predicate = Arc::new(SemanticSimilarity::new(0.8));
        let property = StatisticalPropertyBuilder::new()
            .name("test_property")
            .min_success_rate(0.9)
            .predicate(predicate)
            .build()
            .unwrap();

        assert_eq!(property.name, "test_property");
        assert!((property.min_success_rate - 0.9).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_insufficient_samples_error() {
        let predicate = Arc::new(SemanticSimilarity::new(0.8));
        let property = StatisticalProperty::new("test", 0.9, predicate);

        let config = StatisticalConfig {
            min_samples: 20,
            ..Default::default()
        };
        let verifier = StatisticalVerifier::new(config);

        let samples = vec!["hello"; 5]; // Only 5 samples
        let result = verifier.verify(&property, &samples, None).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), SemanticError::Statistical(_)));
    }

    // Mutation-killing tests for erf function
    #[test]
    fn test_erf_known_values() {
        // Test known values of error function
        // erf(0) = 0
        let erf_0 = erf(0.0);
        assert!(erf_0.abs() < 1e-6, "erf(0) should be 0, got {}", erf_0);

        // erf(1) ≈ 0.8427
        let erf_1 = erf(1.0);
        assert!(
            (erf_1 - 0.8427).abs() < 0.01,
            "erf(1) should be ~0.8427, got {}",
            erf_1
        );

        // erf(-1) ≈ -0.8427 (odd function)
        let erf_neg1 = erf(-1.0);
        assert!(
            (erf_neg1 + 0.8427).abs() < 0.01,
            "erf(-1) should be ~-0.8427, got {}",
            erf_neg1
        );

        // erf(2) ≈ 0.9953
        let erf_2 = erf(2.0);
        assert!(
            (erf_2 - 0.9953).abs() < 0.01,
            "erf(2) should be ~0.9953, got {}",
            erf_2
        );

        // erf(0.5) ≈ 0.5205
        let erf_05 = erf(0.5);
        assert!(
            (erf_05 - 0.5205).abs() < 0.01,
            "erf(0.5) should be ~0.5205, got {}",
            erf_05
        );
    }

    #[test]
    fn test_erf_symmetry() {
        // erf is an odd function: erf(-x) = -erf(x)
        for x in [0.1, 0.5, 1.0, 1.5, 2.0, 3.0] {
            let pos = erf(x);
            let neg = erf(-x);
            assert!(
                (pos + neg).abs() < 1e-6,
                "erf({}) + erf({}) should be 0, got {} + {} = {}",
                x,
                -x,
                pos,
                neg,
                pos + neg
            );
        }
    }

    #[test]
    fn test_erf_bounds() {
        // erf is bounded by [-1, 1]
        for x in [-10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0] {
            let result = erf(x);
            assert!(
                (-1.0..=1.0).contains(&result),
                "erf({}) = {} should be in [-1, 1]",
                x,
                result
            );
        }
    }

    #[test]
    fn test_erf_monotonicity() {
        // erf is monotonically increasing
        let mut prev = erf(-5.0);
        for x in (-40..=50).map(|i| i as f64 * 0.1) {
            let curr = erf(x);
            assert!(
                curr >= prev - 1e-10,
                "erf({}) = {} should be >= erf(previous) = {}",
                x,
                curr,
                prev
            );
            prev = curr;
        }
    }

    // Mutation-killing tests for wilson_confidence_interval
    #[test]
    fn test_wilson_ci_zero_trials() {
        let (lower, upper) = wilson_confidence_interval(0, 0, 0.95);
        assert_eq!(lower, 0.0);
        assert_eq!(upper, 1.0);
    }

    #[test]
    fn test_wilson_ci_formula_correctness() {
        // Test specific values to verify formula correctness
        // With 80/100 successes at 95% confidence:
        let (lower, upper) = wilson_confidence_interval(80, 100, 0.95);
        // The Wilson interval for p=0.8, n=100 should be approximately [0.71, 0.87]
        assert!(
            lower > 0.70 && lower < 0.75,
            "Wilson CI lower bound for 80/100 should be ~0.71-0.73, got {}",
            lower
        );
        assert!(
            upper > 0.85 && upper < 0.90,
            "Wilson CI upper bound for 80/100 should be ~0.86-0.88, got {}",
            upper
        );
    }

    #[test]
    fn test_wilson_ci_different_confidence_levels() {
        // Higher confidence = wider interval
        let (lower_95, upper_95) = wilson_confidence_interval(50, 100, 0.95);
        let (lower_99, upper_99) = wilson_confidence_interval(50, 100, 0.99);
        let (lower_90, upper_90) = wilson_confidence_interval(50, 100, 0.90);

        // 99% CI should be wider than 95% CI
        assert!(
            lower_99 < lower_95,
            "99% CI lower ({}) should be < 95% CI lower ({})",
            lower_99,
            lower_95
        );
        assert!(
            upper_99 > upper_95,
            "99% CI upper ({}) should be > 95% CI upper ({})",
            upper_99,
            upper_95
        );

        // 90% CI should be narrower than 95% CI
        assert!(
            lower_90 > lower_95,
            "90% CI lower ({}) should be > 95% CI lower ({})",
            lower_90,
            lower_95
        );
        assert!(
            upper_90 < upper_95,
            "90% CI upper ({}) should be < 95% CI upper ({})",
            upper_90,
            upper_95
        );
    }

    #[test]
    fn test_wilson_ci_bounds_clamping() {
        // Test that bounds are properly clamped to [0, 1]
        let (lower, upper) = wilson_confidence_interval(100, 100, 0.95);
        assert!(lower >= 0.0, "Lower bound should be >= 0.0");
        assert!(upper <= 1.0, "Upper bound should be <= 1.0");

        let (lower2, upper2) = wilson_confidence_interval(0, 100, 0.95);
        assert!(lower2 >= 0.0, "Lower bound should be >= 0.0");
        assert!(upper2 <= 1.0, "Upper bound should be <= 1.0");
    }

    #[test]
    fn test_wilson_ci_z_score_selection() {
        // Test z-score selection for different confidence levels
        // z=2.576 for 99%, z=1.96 for 95%, z=1.645 for 90%, z=1.28 for 80%

        // These tests verify that z-score differences produce expected CI widths
        let (l_80, u_80) = wilson_confidence_interval(50, 100, 0.80);
        let (l_90, u_90) = wilson_confidence_interval(50, 100, 0.90);
        let (l_95, u_95) = wilson_confidence_interval(50, 100, 0.95);
        let (l_99, u_99) = wilson_confidence_interval(50, 100, 0.99);

        let width_80 = u_80 - l_80;
        let width_90 = u_90 - l_90;
        let width_95 = u_95 - l_95;
        let width_99 = u_99 - l_99;

        assert!(width_80 < width_90, "80% CI width should be < 90% CI width");
        assert!(width_90 < width_95, "90% CI width should be < 95% CI width");
        assert!(width_95 < width_99, "95% CI width should be < 99% CI width");
    }

    // Mutation-killing tests for binomial_p_value
    #[test]
    fn test_binomial_pvalue_zero_trials() {
        let p = binomial_p_value(0, 0, 0.5);
        assert_eq!(p, 1.0);
    }

    #[test]
    fn test_binomial_pvalue_normal_approx_condition() {
        // Normal approximation kicks in when n*p >= 5 and n*(1-p) >= 5
        // With threshold=0.5 and n=100, both conditions are met (50 >= 5)

        // High success rate vs low threshold should give high p-value
        let p_high = binomial_p_value(90, 100, 0.5);
        assert!(
            p_high > 0.99,
            "p-value for 90/100 vs threshold 0.5 should be very high, got {}",
            p_high
        );

        // Low success rate vs high threshold should give low p-value
        let p_low = binomial_p_value(10, 100, 0.5);
        assert!(
            p_low < 0.01,
            "p-value for 10/100 vs threshold 0.5 should be very low, got {}",
            p_low
        );
    }

    #[test]
    fn test_binomial_pvalue_small_sample_fallback() {
        // When n*p < 5 or n*(1-p) < 5, use fallback
        // n=6, threshold=0.9 -> n*threshold=5.4 >= 5, n*(1-threshold)=0.6 < 5
        // This should use the fallback path

        let p_above = binomial_p_value(6, 6, 0.9);
        // observed_p = 1.0 >= threshold=0.9, so should return 0.5
        assert!(
            (p_above - 0.5).abs() < 0.01,
            "p-value should be 0.5 for small sample with observed >= threshold, got {}",
            p_above
        );

        let p_below = binomial_p_value(4, 6, 0.9);
        // observed_p = 0.67 < threshold=0.9, so should return 0.05
        assert!(
            (p_below - 0.05).abs() < 0.01,
            "p-value should be 0.05 for small sample with observed < threshold, got {}",
            p_below
        );
    }

    #[test]
    fn test_binomial_pvalue_se_near_zero() {
        // When standard error is very small (near 0), use simple comparison
        // This happens when threshold is very close to 0 or 1
        // BUT only if we pass the n*threshold >= 5 && n*(1-threshold) >= 5 check

        // For threshold = 1.0, n*(1-threshold) = 0 < 5, so it uses the fallback path
        // The fallback path returns 0.5 if observed >= threshold
        let p_exact_threshold_1 = binomial_p_value(100, 100, 1.0);
        // observed_p = 1.0 >= threshold=1.0, fallback returns 0.5
        assert!(
            (p_exact_threshold_1 - 0.5).abs() < 0.01,
            "p-value should be 0.5 (fallback) when threshold=1.0, got {}",
            p_exact_threshold_1
        );

        // For the se < 1e-10 path to trigger, we need:
        // 1. n*threshold >= 5 AND n*(1-threshold) >= 5
        // 2. se = sqrt(threshold * (1-threshold) / n) < 1e-10
        // This would require threshold*(1-threshold)/n < 1e-20
        // With n=100000000000 (1e11), threshold=0.5: se = sqrt(0.25/1e11) ≈ 1.58e-6, still > 1e-10
        // The se < 1e-10 branch is effectively unreachable for practical inputs

        // Test the normal path with moderate threshold
        let p_normal = binomial_p_value(50, 100, 0.5);
        // observed_p = 0.5 = threshold, z = 0, erf(0) = 0, p = 0.5*(1+0) = 0.5
        assert!(
            (p_normal - 0.5).abs() < 0.01,
            "p-value should be ~0.5 when observed equals threshold, got {}",
            p_normal
        );
    }

    #[test]
    fn test_binomial_pvalue_monotonicity() {
        // For fixed n and threshold, higher success count should give higher p-value
        let mut prev_p = 0.0;
        for successes in 0..=100 {
            let p = binomial_p_value(successes, 100, 0.5);
            assert!(
                p >= prev_p - 1e-10,
                "p-value should increase with success count: p({})={} < p({})={}",
                successes - 1,
                prev_p,
                successes,
                p
            );
            prev_p = p;
        }
    }

    // Mutation-killing tests for is_significant_below
    #[test]
    fn test_is_significant_below_boundary() {
        // Test the boundary condition: ci_upper < threshold
        let result = StatisticalResult {
            passed: false,
            num_samples: 100,
            num_passed: 50,
            success_rate: 0.5,
            ci_lower: 0.4,
            ci_upper: 0.6, // exactly at threshold
            p_value: Some(0.5),
            samples: vec![],
            early_stopped: false,
            explanation: String::new(),
        };

        // ci_upper (0.6) is NOT < 0.6, so should be false
        assert!(
            !result.is_significant_below(0.6),
            "is_significant_below(0.6) should be false when ci_upper = 0.6"
        );

        // ci_upper (0.6) < 0.61, so should be true
        assert!(
            result.is_significant_below(0.61),
            "is_significant_below(0.61) should be true when ci_upper = 0.6"
        );

        // ci_upper (0.6) is NOT < 0.59, so should be false
        assert!(
            !result.is_significant_below(0.59),
            "is_significant_below(0.59) should be false when ci_upper = 0.6"
        );
    }

    #[test]
    fn test_is_significant_below_strict_less_than() {
        // Verify that is_significant_below uses strict < not <=
        let result = StatisticalResult {
            passed: false,
            num_samples: 100,
            num_passed: 10,
            success_rate: 0.1,
            ci_lower: 0.05,
            ci_upper: 0.15,
            p_value: Some(0.01),
            samples: vec![],
            early_stopped: false,
            explanation: String::new(),
        };

        // ci_upper = 0.15
        assert!(result.is_significant_below(0.16)); // 0.15 < 0.16
        assert!(result.is_significant_below(0.20)); // 0.15 < 0.20
        assert!(!result.is_significant_below(0.15)); // 0.15 is NOT < 0.15
        assert!(!result.is_significant_below(0.14)); // 0.15 is NOT < 0.14
    }

    // Mutation-killing tests for StatisticalProperty Debug impl
    #[test]
    fn test_statistical_property_debug() {
        let predicate = Arc::new(SemanticSimilarity::new(0.8));
        let property = StatisticalProperty::new("test_prop", 0.9, predicate);

        let debug_str = format!("{:?}", property);
        assert!(
            debug_str.contains("StatisticalProperty"),
            "Debug output should contain 'StatisticalProperty'"
        );
        assert!(
            debug_str.contains("test_prop"),
            "Debug output should contain property name"
        );
        assert!(
            debug_str.contains("0.9"),
            "Debug output should contain min_success_rate"
        );
    }

    // Mutation-killing tests for early stopping logic
    #[tokio::test]
    async fn test_early_stopping_fail_path() {
        let predicate = Arc::new(SemanticSimilarity::new(0.99)); // Very high threshold - will fail
        let property = StatisticalProperty::new("fail_test", 0.95, predicate);

        let config = StatisticalConfig {
            min_samples: 5,
            max_samples: 100,
            early_stopping: true,
            ..Default::default()
        };
        let verifier = StatisticalVerifier::new(config);

        // All different samples - none will pass the 0.99 similarity threshold
        let samples: Vec<&str> = (0..100)
            .map(|i| match i % 4 {
                0 => "apple",
                1 => "banana",
                2 => "cherry",
                _ => "date",
            })
            .collect();

        let result = verifier
            .verify(&property, &samples, Some("elephant"))
            .await
            .unwrap();

        // Should fail early since no samples will pass
        assert!(!result.passed, "Should fail with 0% success rate");
        assert!(
            result.early_stopped,
            "Should trigger early stopping on failure"
        );
        assert!(
            result.num_samples < 100,
            "Should have stopped before 100 samples"
        );
    }

    #[tokio::test]
    async fn test_verify_checks_min_samples_index() {
        // Test that early stopping only kicks in after min_samples - 1
        let predicate = Arc::new(SemanticSimilarity::new(0.5));
        let property = StatisticalProperty::new("test", 0.5, predicate);

        let config = StatisticalConfig {
            min_samples: 10,
            max_samples: 15,
            early_stopping: true,
            ..Default::default()
        };
        let verifier = StatisticalVerifier::new(config);

        // All same samples - should pass
        let samples = vec!["hello"; 15];
        let result = verifier
            .verify(&property, &samples, Some("hello"))
            .await
            .unwrap();

        // Should have at least min_samples evaluated
        assert!(
            result.num_samples >= 10,
            "Should evaluate at least min_samples (10), got {}",
            result.num_samples
        );
    }

    #[tokio::test]
    async fn test_verify_final_evaluation_uses_ci_lower() {
        // Test that final pass/fail decision uses ci_lower >= min_success_rate
        let predicate = Arc::new(SemanticSimilarity::new(0.5));
        // Set a very low min_success_rate that should pass even with some failures
        let property = StatisticalProperty::new("test", 0.3, predicate);

        let config = StatisticalConfig {
            min_samples: 10,
            max_samples: 10,
            early_stopping: false, // Disable early stopping to test final evaluation
            ..Default::default()
        };
        let verifier = StatisticalVerifier::new(config);

        // 7/10 same samples - 70% success rate
        let samples: Vec<&str> = (0..10)
            .map(|i| if i < 7 { "hello" } else { "different_text_xyz" })
            .collect();
        let result = verifier
            .verify(&property, &samples, Some("hello"))
            .await
            .unwrap();

        // With 70% success rate and 0.3 threshold, Wilson CI lower bound should be > 0.3
        assert!(
            result.passed,
            "Should pass: ci_lower ({}) >= threshold (0.3)",
            result.ci_lower
        );
    }
}
