//! Domain Clipping: Tighten intermediate bounds using activation statistics.
//!
//! Domain clipping is a sound bound tightening technique that uses empirical
//! activation statistics to clip intermediate bounds to realistic ranges.
//! By observing what values activations actually take on concrete inputs,
//! we can safely tighten bounds that would otherwise explode through deep networks.
//!
//! ## Algorithm
//!
//! 1. **Statistics Collection**: Run concrete forward passes on representative inputs
//!    to collect per-layer activation statistics (mean μ, std σ, min, max).
//!
//! 2. **Bound Clipping**: During abstract propagation, clip each layer's output bounds:
//!    - `clip_lower = max(original_lower, observed_min - margin)`
//!    - `clip_upper = min(original_upper, observed_max + margin)`
//!
//!    The margin ensures soundness by extending beyond observed values.
//!
//! ## Soundness Guarantee
//!
//! Clipping is sound (never excludes reachable values) when:
//! - The margin is large enough to cover unobserved but reachable values
//! - The clipping range contains all values the network can actually produce
//!
//! We provide two margin strategies:
//! - **Statistical**: μ ± k*σ (k=6 gives 99.99% coverage for Gaussian distributions)
//! - **Empirical**: observed_min/max ± empirical_margin (based on sample extremes)
//!
//! The empirical strategy is more robust for non-Gaussian activations.
//!
//! ## References
//!
//! - arxiv:2512.11087 - Domain-specific bound tightening for neural network verification

use gamma_core::{GammaError, Result};
use gamma_tensor::BoundedTensor;
use ndarray::{ArrayD, IxDyn, Zip};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, trace};

/// Statistics for a single layer's activations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerStatistics {
    /// Layer identifier (name or index).
    pub layer_id: String,
    /// Per-element mean across samples.
    pub mean: ArrayD<f32>,
    /// Per-element standard deviation across samples.
    pub std: ArrayD<f32>,
    /// Per-element minimum observed value.
    pub min_observed: ArrayD<f32>,
    /// Per-element maximum observed value.
    pub max_observed: ArrayD<f32>,
    /// Number of samples used to compute statistics.
    pub num_samples: usize,
    /// Shape of the activation tensor.
    pub shape: Vec<usize>,
}

impl LayerStatistics {
    /// Create new statistics initialized to empty state.
    pub fn new(layer_id: impl Into<String>, shape: Vec<usize>) -> Self {
        let dim = IxDyn(&shape);
        Self {
            layer_id: layer_id.into(),
            mean: ArrayD::zeros(dim.clone()),
            std: ArrayD::zeros(dim.clone()),
            min_observed: ArrayD::from_elem(dim.clone(), f32::INFINITY),
            max_observed: ArrayD::from_elem(dim, f32::NEG_INFINITY),
            num_samples: 0,
            shape,
        }
    }

    /// Update statistics with a new sample (Welford's online algorithm).
    pub fn update(&mut self, sample: &ArrayD<f32>) -> Result<()> {
        if sample.shape() != self.shape.as_slice() {
            return Err(GammaError::shape_mismatch(
                self.shape.clone(),
                sample.shape().to_vec(),
            ));
        }

        self.num_samples += 1;
        let n = self.num_samples as f32;

        // Size-adaptive parallelization: use parallel ops for very large tensors only.
        // Domain clip operations are memory-bound; parallelization overhead exceeds benefit
        // for tensors below ~1M elements (benchmarked on M-series Mac).
        const PARALLEL_THRESHOLD: usize = 1_000_000;
        let use_parallel = sample.len() >= PARALLEL_THRESHOLD;

        // Update min/max
        if use_parallel {
            Zip::from(&mut self.min_observed)
                .and(sample)
                .par_for_each(|min_val, &s| {
                    *min_val = min_val.min(s);
                });
            Zip::from(&mut self.max_observed)
                .and(sample)
                .par_for_each(|max_val, &s| {
                    *max_val = max_val.max(s);
                });
        } else {
            Zip::from(&mut self.min_observed)
                .and(sample)
                .for_each(|min_val, &s| {
                    *min_val = min_val.min(s);
                });
            Zip::from(&mut self.max_observed)
                .and(sample)
                .for_each(|max_val, &s| {
                    *max_val = max_val.max(s);
                });
        }

        // Welford's online mean/variance update
        let delta = sample - &self.mean;
        self.mean = &self.mean + &(&delta / n);
        let delta2 = sample - &self.mean;

        // For variance: M2 = M2 + delta * delta2
        // We store sqrt(M2/(n-1)) as std after sufficient samples
        if self.num_samples > 1 {
            // Update variance estimate (using Bessel's correction)
            let variance_update = &delta * &delta2;
            // Running variance: new_var = old_var * (n-2)/(n-1) + delta*delta2/(n-1)
            let old_var = &self.std * &self.std;
            let new_var = &old_var * ((n - 2.0) / (n - 1.0)) + &variance_update / (n - 1.0);
            self.std = new_var.mapv(|v| v.max(0.0).sqrt());
        }

        Ok(())
    }

    /// Get the clipping bounds using statistical margin (μ ± k*σ).
    pub fn statistical_bounds(&self, clip_factor: f32) -> (ArrayD<f32>, ArrayD<f32>) {
        let margin = &self.std * clip_factor;
        let lower = &self.mean - &margin;
        let upper = &self.mean + &margin;
        (lower, upper)
    }

    /// Get the clipping bounds using empirical margin (observed ± margin).
    pub fn empirical_bounds(&self, margin_factor: f32) -> (ArrayD<f32>, ArrayD<f32>) {
        let range = &self.max_observed - &self.min_observed;
        let margin = &range * margin_factor;
        let lower = &self.min_observed - &margin;
        let upper = &self.max_observed + &margin;
        (lower, upper)
    }

    /// Get the tighter of statistical and empirical bounds.
    ///
    /// Uses the intersection of both bounds, which is sound if both are sound.
    pub fn combined_bounds(
        &self,
        statistical_factor: f32,
        empirical_factor: f32,
    ) -> (ArrayD<f32>, ArrayD<f32>) {
        let (stat_lower, stat_upper) = self.statistical_bounds(statistical_factor);
        let (emp_lower, emp_upper) = self.empirical_bounds(empirical_factor);

        // Size-adaptive parallelization: use parallel ops for very large tensors only.
        // Domain clip operations are memory-bound; parallelization overhead exceeds benefit
        // for tensors below ~1M elements (benchmarked on M-series Mac).
        const PARALLEL_THRESHOLD: usize = 1_000_000;
        let use_parallel = stat_lower.len() >= PARALLEL_THRESHOLD;

        // Take the tighter (inner) bounds
        let (lower, upper) = if use_parallel {
            let mut lower_out = ArrayD::zeros(stat_lower.raw_dim());
            let mut upper_out = ArrayD::zeros(stat_upper.raw_dim());
            Zip::from(&mut lower_out)
                .and(&stat_lower)
                .and(&emp_lower)
                .par_for_each(|out, &s, &e| *out = s.max(e));
            Zip::from(&mut upper_out)
                .and(&stat_upper)
                .and(&emp_upper)
                .par_for_each(|out, &s, &e| *out = s.min(e));
            (lower_out, upper_out)
        } else {
            let lower = Zip::from(&stat_lower)
                .and(&emp_lower)
                .map_collect(|&s, &e| s.max(e));
            let upper = Zip::from(&stat_upper)
                .and(&emp_upper)
                .map_collect(|&s, &e| s.min(e));
            (lower, upper)
        };

        (lower, upper)
    }
}

/// Strategy for computing clipping margins.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ClipStrategy {
    /// Use statistical bounds: μ ± k*σ
    /// Good for Gaussian-like distributions (most activations).
    Statistical {
        /// Number of standard deviations (default: 6.0 for 99.99% coverage)
        k: f32,
    },

    /// Use empirical bounds: observed_min/max ± margin*range
    /// More robust for heavy-tailed or bounded distributions.
    Empirical {
        /// Fraction of observed range to add as margin (default: 0.1 = 10%)
        margin_factor: f32,
    },

    /// Use the tighter of statistical and empirical bounds.
    /// Best overall tightness while maintaining soundness.
    Combined {
        /// Statistical factor (k in μ ± k*σ)
        statistical_k: f32,
        /// Empirical margin factor
        empirical_margin: f32,
    },
}

impl Default for ClipStrategy {
    fn default() -> Self {
        // Combined strategy provides best tightness
        Self::Combined {
            statistical_k: 6.0,
            empirical_margin: 0.1,
        }
    }
}

/// Configuration for domain clipping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainClipConfig {
    /// Clipping strategy to use.
    pub strategy: ClipStrategy,
    /// Minimum number of samples before clipping is applied.
    /// Too few samples may give unreliable statistics.
    pub min_samples: usize,
    /// Whether to apply clipping (can be disabled for soundness testing).
    pub enabled: bool,
    /// Layer name patterns to exclude from clipping (e.g., output layers).
    pub exclude_patterns: Vec<String>,
    /// Maximum tightening factor: if clipping would reduce width by more than
    /// this factor, limit the tightening. Prevents over-aggressive clipping.
    pub max_tightening_factor: f32,
}

impl Default for DomainClipConfig {
    fn default() -> Self {
        Self {
            strategy: ClipStrategy::default(),
            min_samples: 10,
            enabled: true,
            exclude_patterns: vec![],
            max_tightening_factor: 100.0, // Allow up to 100x tightening
        }
    }
}

impl DomainClipConfig {
    /// Create a conservative configuration with wide margins.
    pub fn conservative() -> Self {
        Self {
            strategy: ClipStrategy::Statistical { k: 10.0 },
            min_samples: 100,
            enabled: true,
            exclude_patterns: vec![],
            max_tightening_factor: 10.0,
        }
    }

    /// Create an aggressive configuration for tighter bounds.
    /// Use only when soundness has been verified via sampling.
    pub fn aggressive() -> Self {
        Self {
            strategy: ClipStrategy::Combined {
                statistical_k: 4.0,
                empirical_margin: 0.05,
            },
            min_samples: 10,
            enabled: true,
            exclude_patterns: vec![],
            max_tightening_factor: 1000.0,
        }
    }
}

/// Domain clipper that stores and applies activation statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainClipper {
    /// Configuration for clipping behavior.
    pub config: DomainClipConfig,
    /// Per-layer statistics, keyed by layer identifier.
    pub statistics: HashMap<String, LayerStatistics>,
    /// Number of times clipping has been applied.
    pub clip_count: usize,
    /// Total amount of bound width reduction from clipping.
    pub total_width_reduction: f64,
}

impl DomainClipper {
    /// Create a new domain clipper with the given configuration.
    pub fn new(config: DomainClipConfig) -> Self {
        Self {
            config,
            statistics: HashMap::new(),
            clip_count: 0,
            total_width_reduction: 0.0,
        }
    }

    /// Create a new domain clipper with default configuration.
    pub fn default_config() -> Self {
        Self::new(DomainClipConfig::default())
    }

    /// Check if a layer should be excluded from clipping.
    fn is_excluded(&self, layer_id: &str) -> bool {
        self.config
            .exclude_patterns
            .iter()
            .any(|pattern| layer_id.contains(pattern))
    }

    /// Update statistics for a layer with a concrete sample.
    pub fn observe(&mut self, layer_id: &str, sample: &ArrayD<f32>) -> Result<()> {
        let stats = self
            .statistics
            .entry(layer_id.to_string())
            .or_insert_with(|| LayerStatistics::new(layer_id, sample.shape().to_vec()));

        stats.update(sample)
    }

    /// Observe multiple samples at once for a layer.
    pub fn observe_batch(&mut self, layer_id: &str, samples: &[ArrayD<f32>]) -> Result<()> {
        for sample in samples {
            self.observe(layer_id, sample)?;
        }
        Ok(())
    }

    /// Get statistics for a layer, if available.
    pub fn get_statistics(&self, layer_id: &str) -> Option<&LayerStatistics> {
        self.statistics.get(layer_id)
    }

    /// Get the clipping bounds for a layer based on its statistics.
    pub fn get_clip_bounds(&self, layer_id: &str) -> Option<(ArrayD<f32>, ArrayD<f32>)> {
        let stats = self.statistics.get(layer_id)?;

        if stats.num_samples < self.config.min_samples {
            trace!(
                "Layer {} has insufficient samples ({} < {}), skipping clip",
                layer_id,
                stats.num_samples,
                self.config.min_samples
            );
            return None;
        }

        let bounds = match self.config.strategy {
            ClipStrategy::Statistical { k } => stats.statistical_bounds(k),
            ClipStrategy::Empirical { margin_factor } => stats.empirical_bounds(margin_factor),
            ClipStrategy::Combined {
                statistical_k,
                empirical_margin,
            } => stats.combined_bounds(statistical_k, empirical_margin),
        };

        Some(bounds)
    }

    /// Apply clipping to a bounded tensor for a specific layer.
    ///
    /// Returns the clipped tensor and the amount of width reduction achieved.
    pub fn clip_bounds(
        &mut self,
        layer_id: &str,
        bounds: &BoundedTensor,
    ) -> Result<(BoundedTensor, f32)> {
        if !self.config.enabled {
            return Ok((bounds.clone(), 0.0));
        }

        if self.is_excluded(layer_id) {
            trace!("Layer {} is excluded from clipping", layer_id);
            return Ok((bounds.clone(), 0.0));
        }

        let Some((clip_lower, clip_upper)) = self.get_clip_bounds(layer_id) else {
            return Ok((bounds.clone(), 0.0));
        };

        // Verify shapes match
        if clip_lower.shape() != bounds.shape() {
            return Err(GammaError::shape_mismatch(
                clip_lower.shape().to_vec(),
                bounds.shape().to_vec(),
            ));
        }

        let original_width = bounds.max_width();

        // Apply clipping: intersection of original bounds and clip bounds
        // Size-adaptive parallelization: use parallel ops for very large tensors only.
        // Domain clip operations are memory-bound; parallelization overhead exceeds benefit
        // for tensors below ~1M elements (benchmarked on M-series Mac).
        const PARALLEL_THRESHOLD: usize = 1_000_000;
        let use_parallel = bounds.lower.len() >= PARALLEL_THRESHOLD;

        let (clipped_lower, clipped_upper) = if use_parallel {
            let mut lower_out = ArrayD::zeros(bounds.lower.raw_dim());
            let mut upper_out = ArrayD::zeros(bounds.upper.raw_dim());
            Zip::from(&mut lower_out)
                .and(&bounds.lower)
                .and(&clip_lower)
                .par_for_each(|out, &orig, &clip| *out = orig.max(clip));
            Zip::from(&mut upper_out)
                .and(&bounds.upper)
                .and(&clip_upper)
                .par_for_each(|out, &orig, &clip| *out = orig.min(clip));
            (lower_out, upper_out)
        } else {
            let clipped_lower = Zip::from(&bounds.lower)
                .and(&clip_lower)
                .map_collect(|&orig, &clip| orig.max(clip));
            let clipped_upper = Zip::from(&bounds.upper)
                .and(&clip_upper)
                .map_collect(|&orig, &clip| orig.min(clip));
            (clipped_lower, clipped_upper)
        };

        // Ensure bounds remain valid (lower <= upper)
        // If clipping inverts bounds, keep original (indicates our statistics are off)
        let (final_lower, final_upper) =
            Self::ensure_valid_bounds(&bounds.lower, &bounds.upper, clipped_lower, clipped_upper);

        let clipped = BoundedTensor::new(final_lower, final_upper)?;
        let clipped_width = clipped.max_width();

        // Check tightening factor limit
        let tightening_factor = if clipped_width > 1e-10 {
            original_width / clipped_width
        } else {
            self.config.max_tightening_factor + 1.0
        };

        if tightening_factor > self.config.max_tightening_factor {
            debug!(
                "Layer {} clipping would exceed max tightening factor ({:.1}x > {:.1}x), limiting",
                layer_id, tightening_factor, self.config.max_tightening_factor
            );
            // Limit the tightening
            let target_width = original_width / self.config.max_tightening_factor;
            let center = (&bounds.lower + &bounds.upper) / 2.0;
            let half_width = target_width / 2.0;
            let limited_lower = &center - half_width;
            let limited_upper = &center + half_width;
            let limited = BoundedTensor::new(limited_lower, limited_upper)?;
            let width_reduction = original_width - limited.max_width();
            self.clip_count += 1;
            self.total_width_reduction += width_reduction as f64;
            return Ok((limited, width_reduction));
        }

        let width_reduction = original_width - clipped_width;
        if width_reduction > 0.0 {
            self.clip_count += 1;
            self.total_width_reduction += width_reduction as f64;
            debug!(
                "Layer {} clipped: width {:.4} -> {:.4} ({:.1}x tighter)",
                layer_id, original_width, clipped_width, tightening_factor
            );
        }

        Ok((clipped, width_reduction))
    }

    /// Ensure clipped bounds are valid (lower <= upper).
    /// If clipping inverts bounds at any position, keep original bounds there.
    fn ensure_valid_bounds(
        orig_lower: &ArrayD<f32>,
        orig_upper: &ArrayD<f32>,
        clipped_lower: ArrayD<f32>,
        clipped_upper: ArrayD<f32>,
    ) -> (ArrayD<f32>, ArrayD<f32>) {
        // Size-adaptive parallelization: use parallel ops for very large tensors only.
        // Domain clip operations are memory-bound; parallelization overhead exceeds benefit
        // for tensors below ~1M elements (benchmarked on M-series Mac).
        const PARALLEL_THRESHOLD: usize = 1_000_000;
        let use_parallel = clipped_lower.len() >= PARALLEL_THRESHOLD;

        let mut final_lower = ArrayD::zeros(clipped_lower.raw_dim());
        let mut final_upper = ArrayD::zeros(clipped_upper.raw_dim());

        if use_parallel {
            Zip::from(&mut final_lower)
                .and(&clipped_lower)
                .and(&clipped_upper)
                .and(orig_lower)
                .par_for_each(|out, &clip_l, &clip_u, &orig_l| {
                    *out = if clip_l <= clip_u { clip_l } else { orig_l };
                });
            Zip::from(&mut final_upper)
                .and(&clipped_lower)
                .and(&clipped_upper)
                .and(orig_upper)
                .par_for_each(|out, &clip_l, &clip_u, &orig_u| {
                    *out = if clip_l <= clip_u { clip_u } else { orig_u };
                });
        } else {
            Zip::from(&mut final_lower)
                .and(&clipped_lower)
                .and(&clipped_upper)
                .and(orig_lower)
                .for_each(|out, &clip_l, &clip_u, &orig_l| {
                    *out = if clip_l <= clip_u { clip_l } else { orig_l };
                });
            Zip::from(&mut final_upper)
                .and(&clipped_lower)
                .and(&clipped_upper)
                .and(orig_upper)
                .for_each(|out, &clip_l, &clip_u, &orig_u| {
                    *out = if clip_l <= clip_u { clip_u } else { orig_u };
                });
        }

        (final_lower, final_upper)
    }

    /// Get a summary of clipping statistics.
    pub fn summary(&self) -> ClipperSummary {
        let total_layers = self.statistics.len();
        let layers_with_sufficient_samples = self
            .statistics
            .values()
            .filter(|s| s.num_samples >= self.config.min_samples)
            .count();

        ClipperSummary {
            total_layers,
            layers_with_sufficient_samples,
            total_samples: self.statistics.values().map(|s| s.num_samples).sum(),
            clip_count: self.clip_count,
            total_width_reduction: self.total_width_reduction,
            config: self.config.clone(),
        }
    }

    /// Reset clipping counters (but keep statistics).
    pub fn reset_counters(&mut self) {
        self.clip_count = 0;
        self.total_width_reduction = 0.0;
    }

    /// Clear all statistics.
    pub fn clear(&mut self) {
        self.statistics.clear();
        self.clip_count = 0;
        self.total_width_reduction = 0.0;
    }

    /// Merge statistics from another clipper.
    pub fn merge(&mut self, other: &DomainClipper) -> Result<()> {
        // Size-adaptive parallelization: use parallel ops for very large tensors only.
        // Domain clip operations are memory-bound; parallelization overhead exceeds benefit
        // for tensors below ~1M elements (benchmarked on M-series Mac).
        const PARALLEL_THRESHOLD: usize = 1_000_000;

        for (layer_id, other_stats) in &other.statistics {
            if let Some(self_stats) = self.statistics.get_mut(layer_id) {
                // Merge by weighted combination
                let total_n = (self_stats.num_samples + other_stats.num_samples).max(1) as f32;
                let self_weight = self_stats.num_samples as f32 / total_n;
                let other_weight = other_stats.num_samples as f32 / total_n;

                self_stats.mean =
                    &(&self_stats.mean * self_weight) + &(&other_stats.mean * other_weight);
                self_stats.std =
                    &(&self_stats.std * self_weight) + &(&other_stats.std * other_weight);

                let use_parallel = self_stats.min_observed.len() >= PARALLEL_THRESHOLD;
                if use_parallel {
                    let mut new_min = ArrayD::zeros(self_stats.min_observed.raw_dim());
                    let mut new_max = ArrayD::zeros(self_stats.max_observed.raw_dim());
                    Zip::from(&mut new_min)
                        .and(&self_stats.min_observed)
                        .and(&other_stats.min_observed)
                        .par_for_each(|out, &a, &b| *out = a.min(b));
                    Zip::from(&mut new_max)
                        .and(&self_stats.max_observed)
                        .and(&other_stats.max_observed)
                        .par_for_each(|out, &a, &b| *out = a.max(b));
                    self_stats.min_observed = new_min;
                    self_stats.max_observed = new_max;
                } else {
                    self_stats.min_observed = Zip::from(&self_stats.min_observed)
                        .and(&other_stats.min_observed)
                        .map_collect(|&a, &b| a.min(b));
                    self_stats.max_observed = Zip::from(&self_stats.max_observed)
                        .and(&other_stats.max_observed)
                        .map_collect(|&a, &b| a.max(b));
                }

                self_stats.num_samples += other_stats.num_samples;
            } else {
                self.statistics
                    .insert(layer_id.clone(), other_stats.clone());
            }
        }
        Ok(())
    }
}

impl Default for DomainClipper {
    fn default() -> Self {
        Self::default_config()
    }
}

/// Summary of domain clipping statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipperSummary {
    /// Total number of layers with statistics.
    pub total_layers: usize,
    /// Layers with enough samples for clipping.
    pub layers_with_sufficient_samples: usize,
    /// Total samples collected across all layers.
    pub total_samples: usize,
    /// Number of times clipping was applied.
    pub clip_count: usize,
    /// Total bound width reduction from clipping.
    pub total_width_reduction: f64,
    /// Configuration used.
    pub config: DomainClipConfig,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_layer_statistics_update() {
        let mut stats = LayerStatistics::new("test", vec![3]);

        // Add some samples
        let samples = vec![
            array![1.0, 2.0, 3.0].into_dyn(),
            array![2.0, 3.0, 4.0].into_dyn(),
            array![3.0, 4.0, 5.0].into_dyn(),
        ];

        for sample in &samples {
            stats.update(sample).unwrap();
        }

        assert_eq!(stats.num_samples, 3);

        // Check mean (should be [2, 3, 4])
        let mean = stats.mean.as_slice().unwrap();
        assert!((mean[0] - 2.0).abs() < 1e-5);
        assert!((mean[1] - 3.0).abs() < 1e-5);
        assert!((mean[2] - 4.0).abs() < 1e-5);

        // Check min/max
        let min = stats.min_observed.as_slice().unwrap();
        let max = stats.max_observed.as_slice().unwrap();
        assert!((min[0] - 1.0).abs() < 1e-5);
        assert!((max[0] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_statistical_bounds() {
        let mut stats = LayerStatistics::new("test", vec![2]);

        // Add samples with known mean and std
        for i in 0..100 {
            let val = (i as f32 - 50.0) / 10.0; // Range [-5, 4.9]
            stats.update(&array![val, val * 2.0].into_dyn()).unwrap();
        }

        let (lower, upper) = stats.statistical_bounds(3.0);

        // Bounds should be approximately μ ± 3σ
        // For uniform distribution, σ ≈ range/√12 ≈ 2.87
        assert!(lower[[0]] < -5.0);
        assert!(upper[[0]] > 5.0);
    }

    #[test]
    fn test_empirical_bounds() {
        let mut stats = LayerStatistics::new("test", vec![2]);

        stats.update(&array![0.0, 10.0].into_dyn()).unwrap();
        stats.update(&array![10.0, 20.0].into_dyn()).unwrap();

        let (lower, upper) = stats.empirical_bounds(0.1); // 10% margin

        // Range is [0, 10] with margin of 10 * 0.1 = 1
        let lower_slice = lower.as_slice().unwrap();
        let upper_slice = upper.as_slice().unwrap();
        assert!((lower_slice[0] - (-1.0)).abs() < 1e-5);
        assert!((upper_slice[0] - 11.0).abs() < 1e-5);
    }

    #[test]
    fn test_domain_clipper_clip_bounds() {
        let mut clipper = DomainClipper::new(DomainClipConfig {
            strategy: ClipStrategy::Empirical { margin_factor: 0.1 },
            min_samples: 1,
            enabled: true,
            exclude_patterns: vec![],
            max_tightening_factor: 100.0,
        });

        // Observe some concrete values
        for _ in 0..10 {
            clipper
                .observe("layer1", &array![5.0, 5.0, 5.0].into_dyn())
                .unwrap();
        }

        // Create bounds that are wider than observed
        let wide_bounds = BoundedTensor::new(
            array![-100.0, -100.0, -100.0].into_dyn(),
            array![100.0, 100.0, 100.0].into_dyn(),
        )
        .unwrap();

        let (clipped, reduction) = clipper.clip_bounds("layer1", &wide_bounds).unwrap();

        // Bounds should be clipped to approximately [5-0.1*0, 5+0.1*0] = [5, 5]
        // since range is 0 (all same values), empirical bounds are [5, 5]
        assert!(clipped.max_width() < wide_bounds.max_width());
        assert!(reduction > 0.0);
    }

    #[test]
    fn test_excluded_layers() {
        let clipper = DomainClipper::new(DomainClipConfig {
            exclude_patterns: vec!["output".to_string(), "final".to_string()],
            ..Default::default()
        });

        assert!(clipper.is_excluded("output_layer"));
        assert!(clipper.is_excluded("model/final_norm"));
        assert!(!clipper.is_excluded("hidden_layer"));
    }

    #[test]
    fn test_insufficient_samples() {
        let mut clipper = DomainClipper::new(DomainClipConfig {
            min_samples: 100,
            ..Default::default()
        });

        // Only add 10 samples
        for _ in 0..10 {
            clipper.observe("layer1", &array![1.0].into_dyn()).unwrap();
        }

        // Should return None (insufficient samples)
        assert!(clipper.get_clip_bounds("layer1").is_none());
    }

    #[test]
    fn test_max_tightening_factor() {
        let mut clipper = DomainClipper::new(DomainClipConfig {
            strategy: ClipStrategy::Empirical { margin_factor: 0.0 },
            min_samples: 1,
            max_tightening_factor: 2.0, // Only allow 2x tightening
            enabled: true,
            exclude_patterns: vec![],
        });

        // Observe a narrow range
        clipper.observe("layer1", &array![0.0].into_dyn()).unwrap();

        // Try to clip very wide bounds
        let wide_bounds =
            BoundedTensor::new(array![-100.0].into_dyn(), array![100.0].into_dyn()).unwrap();

        let (clipped, _) = clipper.clip_bounds("layer1", &wide_bounds).unwrap();

        // Should be limited to 2x tightening (width 200 -> 100)
        assert!(clipped.max_width() >= 100.0);
    }

    #[test]
    fn test_inverted_bounds_protection() {
        // Test that clipping doesn't invert bounds
        let mut clipper = DomainClipper::new(DomainClipConfig {
            strategy: ClipStrategy::Statistical { k: 0.1 }, // Very tight
            min_samples: 1,
            enabled: true,
            exclude_patterns: vec![],
            max_tightening_factor: 1000.0,
        });

        // Observe values around 0
        for _ in 0..10 {
            clipper
                .observe("layer1", &array![0.0, 0.0].into_dyn())
                .unwrap();
        }

        // Bounds that are already very tight
        let tight_bounds = BoundedTensor::new(
            array![10.0, 10.0].into_dyn(), // Far from observed values!
            array![11.0, 11.0].into_dyn(),
        )
        .unwrap();

        let (clipped, _) = clipper.clip_bounds("layer1", &tight_bounds).unwrap();

        // Should preserve original bounds since clipping would invert them
        assert!(clipped.lower[[0]] <= clipped.upper[[0]]);
    }

    // ========== Mutation-killing tests ==========

    /// Test that Welford's mean update uses subtraction (delta - not delta +)
    /// Kills: domain_clip.rs:122:29: replace - with + in LayerStatistics::update
    #[test]
    fn test_welford_mean_uses_subtraction() {
        let mut stats = LayerStatistics::new("test", vec![1]);

        // Add samples with known mean
        stats.update(&array![10.0].into_dyn()).unwrap();
        stats.update(&array![20.0].into_dyn()).unwrap();
        stats.update(&array![30.0].into_dyn()).unwrap();

        // Mean should be 20.0, not some other value
        let mean = stats.mean[[0]];
        assert!(
            (mean - 20.0).abs() < 1e-5,
            "Mean should be 20.0 but got {}",
            mean
        );
    }

    /// Test variance update only happens after first sample
    /// Kills: domain_clip.rs:126:29: replace > with >= in LayerStatistics::update
    #[test]
    fn test_variance_update_after_first_sample() {
        let mut stats = LayerStatistics::new("test", vec![1]);

        // After first sample, std should still be 0
        stats.update(&array![10.0].into_dyn()).unwrap();
        assert_eq!(stats.std[[0]], 0.0, "Std should be 0 after first sample");

        // After second sample, std should be non-zero for different values
        stats.update(&array![20.0].into_dyn()).unwrap();
        assert!(
            stats.std[[0]] > 0.0,
            "Std should be positive after two different samples"
        );
    }

    /// Test variance update arithmetic operations
    /// Kills multiple mutations in line 128-131
    #[test]
    fn test_welford_variance_formula() {
        let mut stats = LayerStatistics::new("test", vec![1]);

        // Add samples with known variance
        // Values: 0, 10, 20 have mean=10 and population std dev ~= 8.165
        stats.update(&array![0.0].into_dyn()).unwrap();
        stats.update(&array![10.0].into_dyn()).unwrap();
        stats.update(&array![20.0].into_dyn()).unwrap();

        let std = stats.std[[0]];
        // Sample std dev should be close to sqrt(100) = 10
        assert!(
            (std - 10.0).abs() < 1.0,
            "Std should be approximately 10.0 but got {}",
            std
        );
    }

    /// Test statistical bounds margin calculation uses multiplication
    /// Kills: domain_clip.rs:140:32: replace * with + in LayerStatistics::statistical_bounds
    #[test]
    fn test_statistical_bounds_multiplication() {
        let mut stats = LayerStatistics::new("test", vec![1]);

        // Create stats with known std
        stats.update(&array![0.0].into_dyn()).unwrap();
        stats.update(&array![10.0].into_dyn()).unwrap();

        // Get bounds with clip_factor = 2.0
        let (lower, upper) = stats.statistical_bounds(2.0);

        // Mean is 5.0, std should be ~7.07 (sample std for 0,10)
        // With multiplication: margin = std * 2.0 = ~14.14
        // With addition: margin = std + 2.0 = ~9.07
        // Bounds should be significantly different

        let width = upper[[0]] - lower[[0]];
        // Width should be 2 * margin = 2 * std * 2.0 = 4 * std
        // std ~= 7.07, so width ~= 28.28
        assert!(
            width > 20.0,
            "Width should be > 20 (using multiplication) but got {}",
            width
        );
    }

    /// Test empirical bounds range calculation uses subtraction
    /// Kills: domain_clip.rs:148:40: replace - with + in LayerStatistics::empirical_bounds
    #[test]
    fn test_empirical_bounds_subtraction() {
        let mut stats = LayerStatistics::new("test", vec![1]);

        stats.update(&array![10.0].into_dyn()).unwrap();
        stats.update(&array![20.0].into_dyn()).unwrap();

        let (lower, upper) = stats.empirical_bounds(0.1);

        // Range is max - min = 20 - 10 = 10
        // Margin = range * 0.1 = 1.0
        // With subtraction: range = 10, margin = 1, lower = 10-1=9, upper = 20+1=21
        // With addition: range = 30, margin = 3, lower = 10-3=7, upper = 20+3=23

        let lower_val = lower[[0]];
        let upper_val = upper[[0]];

        // With subtraction (correct): lower should be 9.0
        assert!(
            (lower_val - 9.0).abs() < 1e-5,
            "Lower should be 9.0 but got {}",
            lower_val
        );
        assert!(
            (upper_val - 21.0).abs() < 1e-5,
            "Upper should be 21.0 but got {}",
            upper_val
        );
    }

    /// Test conservative config differs from default
    /// Kills: domain_clip.rs:268:9: replace DomainClipConfig::conservative -> Self with Default::default()
    #[test]
    fn test_conservative_config_differs_from_default() {
        let conservative = DomainClipConfig::conservative();
        let default = DomainClipConfig::default();

        // Conservative should have higher min_samples
        assert!(
            conservative.min_samples > default.min_samples,
            "Conservative should require more samples: {} vs {}",
            conservative.min_samples,
            default.min_samples
        );

        // Conservative should have lower max_tightening_factor
        assert!(
            conservative.max_tightening_factor < default.max_tightening_factor,
            "Conservative should have lower max tightening: {} vs {}",
            conservative.max_tightening_factor,
            default.max_tightening_factor
        );
    }

    /// Test aggressive config differs from default
    /// Kills: domain_clip.rs:280:9: replace DomainClipConfig::aggressive -> Self with Default::default()
    #[test]
    fn test_aggressive_config_differs_from_default() {
        let aggressive = DomainClipConfig::aggressive();
        let default = DomainClipConfig::default();

        // Aggressive should have higher max_tightening_factor
        assert!(
            aggressive.max_tightening_factor > default.max_tightening_factor,
            "Aggressive should have higher max tightening: {} vs {}",
            aggressive.max_tightening_factor,
            default.max_tightening_factor
        );
    }

    /// Test observe_batch actually processes samples
    /// Kills: domain_clip.rs:342:9: replace DomainClipper::observe_batch -> Result<()> with Ok(())
    #[test]
    fn test_observe_batch_processes_samples() {
        let mut clipper = DomainClipper::default();

        let samples = vec![
            array![1.0, 2.0].into_dyn(),
            array![3.0, 4.0].into_dyn(),
            array![5.0, 6.0].into_dyn(),
        ];

        clipper.observe_batch("layer1", &samples).unwrap();

        let stats = clipper.get_statistics("layer1").unwrap();
        assert_eq!(
            stats.num_samples, 3,
            "observe_batch should have processed 3 samples, got {}",
            stats.num_samples
        );
    }

    /// Test get_statistics returns actual statistics
    /// Kills: domain_clip.rs:350:9: replace DomainClipper::get_statistics -> Option<&LayerStatistics> with None
    #[test]
    fn test_get_statistics_returns_data() {
        let mut clipper = DomainClipper::default();

        clipper
            .observe("layer1", &array![1.0, 2.0].into_dyn())
            .unwrap();

        let stats = clipper.get_statistics("layer1");
        assert!(
            stats.is_some(),
            "get_statistics should return Some after observe"
        );
        assert_eq!(stats.unwrap().num_samples, 1);
    }

    /// Test min_samples boundary condition (< vs <=)
    /// Kills: domain_clip.rs:357:30: replace < with <= in DomainClipper::get_clip_bounds
    #[test]
    fn test_min_samples_boundary() {
        let mut clipper = DomainClipper::new(DomainClipConfig {
            min_samples: 5,
            ..Default::default()
        });

        // Add exactly 5 samples
        for _ in 0..5 {
            clipper.observe("layer1", &array![1.0].into_dyn()).unwrap();
        }

        // With < (correct): 5 < 5 is false, so clipping should work
        // With <= (wrong): 5 <= 5 is true, so clipping would be skipped
        let bounds = clipper.get_clip_bounds("layer1");
        assert!(
            bounds.is_some(),
            "Should get clip bounds when num_samples == min_samples"
        );
    }

    /// Test variance update delta multiplication (not addition)
    /// Kills: domain_clip.rs:128:42: replace * with + in LayerStatistics::update
    #[test]
    fn test_variance_delta_multiplication() {
        let mut stats = LayerStatistics::new("test", vec![1]);

        // With very different values, variance should be large
        stats.update(&array![0.0].into_dyn()).unwrap();
        stats.update(&array![100.0].into_dyn()).unwrap();

        // With multiplication: variance_update = delta * delta2 (large)
        // With addition: variance_update = delta + delta2 (smaller)
        let std = stats.std[[0]];
        assert!(
            std > 30.0,
            "Std for [0, 100] should be large (>30) but got {}",
            std
        );
    }

    /// Test variance formula division operations
    /// Kills: domain_clip.rs:131:49: replace / with % or *
    #[test]
    fn test_variance_division_not_modulo() {
        let mut stats = LayerStatistics::new("test", vec![1]);

        // Add samples to test variance calculation
        stats.update(&array![1.0].into_dyn()).unwrap();
        stats.update(&array![2.0].into_dyn()).unwrap();
        stats.update(&array![3.0].into_dyn()).unwrap();
        stats.update(&array![4.0].into_dyn()).unwrap();
        stats.update(&array![5.0].into_dyn()).unwrap();

        // Mean should be 3.0
        assert!((stats.mean[[0]] - 3.0).abs() < 1e-5);

        // Sample variance for [1,2,3,4,5] = 2.5, so std = sqrt(2.5) ≈ 1.58
        let std = stats.std[[0]];
        assert!(
            std > 1.0 && std < 2.0,
            "Std should be ~1.58 but got {}",
            std
        );
    }

    /// Test old_var multiplication in variance update
    /// Kills: domain_clip.rs:130:37 and 131:36: replace * with + in old_var calculations
    #[test]
    fn test_variance_old_var_multiplication() {
        let mut stats = LayerStatistics::new("test", vec![1]);

        // Add many samples to test running variance
        for i in 0..20 {
            stats.update(&array![i as f32].into_dyn()).unwrap();
        }

        // For 0..19, mean=9.5, variance should be consistent
        // If * were replaced with +, the variance would explode
        let std = stats.std[[0]];
        assert!(
            std > 5.0 && std < 10.0,
            "Std for 0..19 should be ~5.92 but got {}",
            std
        );
    }

    /// Test variance subtraction operations
    /// Kills: domain_clip.rs:131:42, 131:54, 131:86: replace - with + or /
    #[test]
    fn test_variance_subtraction_operations() {
        let mut stats = LayerStatistics::new("test", vec![1]);

        // Add samples with known variance
        stats.update(&array![10.0].into_dyn()).unwrap();
        stats.update(&array![12.0].into_dyn()).unwrap();
        stats.update(&array![14.0].into_dyn()).unwrap();

        // Mean should be 12.0
        assert!((stats.mean[[0]] - 12.0).abs() < 1e-4);

        // Sample std for [10, 12, 14] should be 2.0
        let std = stats.std[[0]];
        assert!(
            (std - 2.0).abs() < 0.5,
            "Std for [10,12,14] should be ~2.0 but got {}",
            std
        );
    }

    /// Test variance formula with (n-2)/(n-1) ratio
    /// Kills: domain_clip.rs:131:81: replace / with % or *
    #[test]
    fn test_variance_n_ratio_division() {
        let mut stats = LayerStatistics::new("test", vec![1]);

        // Need at least 3 samples to get (n-2)/(n-1) = 1/2
        stats.update(&array![0.0].into_dyn()).unwrap();
        stats.update(&array![10.0].into_dyn()).unwrap();
        stats.update(&array![20.0].into_dyn()).unwrap();
        stats.update(&array![30.0].into_dyn()).unwrap();

        // If division were replaced with modulo or multiplication,
        // the variance calculation would be completely wrong
        let std = stats.std[[0]];
        // For [0,10,20,30]: mean=15, sample var = 166.67, std ≈ 12.91
        assert!(
            std > 10.0 && std < 16.0,
            "Std for [0,10,20,30] should be ~12.91 but got {}",
            std
        );
    }

    /// Test that variance update check uses > (not >=)
    /// Kills: domain_clip.rs:126:29: replace > with >= in LayerStatistics::update
    ///
    /// With >=: variance would be updated on first sample where n=1
    /// This would cause (n-2)/(n-1) = -1/0 = -inf or NaN
    #[test]
    fn test_variance_check_strictly_greater_than_one() {
        let mut stats = LayerStatistics::new("test", vec![1]);

        // Add exactly one sample
        stats.update(&array![42.0].into_dyn()).unwrap();

        // With > (correct): variance not updated, std stays at 0
        // With >= (wrong): variance updated with n=1, causing (n-2)/(n-1) = -1/0
        // This would result in NaN or infinity

        let std = stats.std[[0]];
        assert!(!std.is_nan(), "Std should not be NaN after one sample");
        assert!(
            !std.is_infinite(),
            "Std should not be infinite after one sample"
        );
        assert_eq!(
            std, 0.0,
            "Std should be exactly 0.0 after one sample, got {}",
            std
        );
    }

    /// Test variance formula subtraction vs division
    /// Kills: domain_clip.rs:131:54: replace - with / in LayerStatistics::update
    ///
    /// The formula has: (n - 1.0), if replaced with (n / 1.0) = n, the Bessel correction breaks
    #[test]
    fn test_variance_bessel_correction_subtraction() {
        let mut stats = LayerStatistics::new("test", vec![1]);

        // Add exactly 2 samples: 0 and 10
        stats.update(&array![0.0].into_dyn()).unwrap();
        stats.update(&array![10.0].into_dyn()).unwrap();

        // With n=2:
        // Correct: (n-1) = 1, variance_update / 1 = variance_update
        // Wrong: (n/1) = 2, variance_update / 2 = half the variance

        // For [0, 10]: mean=5, sample variance = (25+25)/1 = 50, std = 7.07
        // With division bug: sample variance = (25+25)/2 = 25, std = 5.0

        let std = stats.std[[0]];
        assert!(
            std > 6.0,
            "Std for [0, 10] should be >6 (around 7.07), got {}. If ~5.0, the Bessel correction is broken",
            std
        );
    }
}
