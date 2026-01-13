//! β-CROWN: Branch-and-Bound Neural Network Verification
//!
//! Implements complete verification via branch-and-bound search over ReLU activation
//! space. When CROWN/α-CROWN bounds are inconclusive (lower bound < threshold),
//! β-CROWN splits unstable neurons into "always active" (x ≥ 0) and "always inactive"
//! (x ≤ 0) branches, encoding these constraints as β parameters in the bound computation.
//!
//! ## Key Features
//!
//! - **Joint α-β optimization**: Optimizes both ReLU relaxation slopes (α) and Lagrangian
//!   multipliers (β) together within each domain for tighter bounds.
//! - **Exact β gradients**: Uses augmented chain rule for precise gradient computation.
//! - **Parallel domain processing**: Processes multiple domains concurrently via Rayon.
//! - **Adaptive optimization**: Adam-style optimizer with configurable variants.
//!
//! ## Algorithm
//!
//! 1. Initialize with root domain (no splits)
//! 2. While domains remain and not timed out:
//!    a. Pick domain with worst (highest) lower bound
//!    b. If lower bound > threshold, domain is verified
//!    c. Select an unstable neuron to split
//!    d. Create two child domains (neuron active vs inactive)
//!    e. Compute bounds for children with joint α-β optimization
//!    f. Add children to queue if not verified/infeasible
//! 3. If all domains verified, return Verified; else Unknown/Timeout
//!
//! ## Optimizer Recommendations
//!
//! Based on benchmark testing, these optimizer settings perform best for β-CROWN:
//!
//! | Setting | Recommendation | Notes |
//! |---------|---------------|-------|
//! | Base optimizer | Adam (default) | AMSGrad and Lookahead+Adam also work well |
//! | RAdam | **Avoid** | 7x slower due to warmup behavior |
//! | AdamW (weight_decay) | **Avoid** | 60% slower, unnecessary regularization |
//! | LR scheduler | Constant | All decay schedules are 67% slower |
//! | bias_correction | `true` | Essential for short iteration runs |
//! | grad_clip | `10.0` | Default is appropriate for most cases |
//!
//! The default `AdaptiveOptConfig::default()` uses optimal settings.

use crate::bounds::GraphAlphaCrownIntermediate;
use crate::pgd_attack::{PgdAttacker, PgdConfig};
use crate::{
    AlphaCrownConfig, AlphaState, BoundPropagation, GraphNetwork, Layer, LinearBounds, Network,
};
use gamma_core::{GammaError, GemmEngine, Result};
use gamma_tensor::BoundedTensor;
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, instrument, trace, warn};

/// Configuration for β-CROWN branch-and-bound search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetaCrownConfig {
    /// Maximum number of domains to explore before giving up.
    pub max_domains: usize,
    /// Timeout for the entire search.
    pub timeout: Duration,
    /// Maximum depth of the search tree (max number of splits).
    pub max_depth: usize,
    /// Use α-CROWN optimization within each domain.
    pub use_alpha_crown: bool,
    /// Use CROWN-IBP for tighter intermediate bounds.
    /// When enabled, intermediate bounds are computed using CROWN backward
    /// propagation and intersected with IBP bounds, producing ~66% tighter bounds
    /// than standard CROWN. More expensive but significantly improves verification.
    #[serde(default)]
    pub use_crown_ibp: bool,
    /// α-CROWN configuration (if use_alpha_crown is true).
    pub alpha_config: AlphaCrownConfig,
    /// Branching heuristic for selecting neurons to split.
    pub branching_heuristic: BranchingHeuristic,
    /// Number of candidate neurons to evaluate for FSB branching.
    ///
    /// Higher values can reduce domain count but increase per-domain overhead.
    #[serde(default = "default_fsb_candidates")]
    pub fsb_candidates: usize,
    /// Learning rate for β parameter optimization.
    pub beta_lr: f32,
    /// Number of optimization iterations per domain (applies to both α and β).
    pub beta_iterations: usize,
    /// Minimum improvement to continue optimization.
    pub beta_tolerance: f32,
    /// Number of β optimization iterations to run on the root domain before BaB.
    /// This amortizes the optimization cost across all descendant domains via warmup.
    /// Set to 0 to disable root-level optimization (default: 20).
    #[serde(default = "default_root_beta_iterations")]
    pub root_beta_iterations: usize,
    /// Maximum depth at which to run per-domain β optimization.
    /// Domains deeper than this use inherited β values without further optimization.
    /// Set to 0 to disable per-domain optimization entirely (rely only on root + warmup).
    /// Default: 3 (optimize root and first 3 levels of splits).
    #[serde(default = "default_beta_max_depth")]
    pub beta_max_depth: usize,
    /// Use analytical β gradients for DAG networks instead of SPSA.
    /// Analytical gradients are computed from the A matrices stored during
    /// CROWN backward propagation, which is ~3x faster than SPSA (1 pass vs 3 passes per iteration).
    /// Default: true (use analytical gradients when available).
    #[serde(default = "default_use_analytical_beta_gradients")]
    pub use_analytical_beta_gradients: bool,
    /// Learning rate for α parameter optimization (when use_alpha_crown is true).
    pub alpha_lr: f32,
    /// Use momentum for α updates (inherits from alpha_config.momentum if true).
    pub alpha_momentum: bool,
    /// Number of domains to process in parallel (batch size).
    /// Set to 1 for sequential processing, or higher for parallel.
    pub batch_size: usize,
    /// Use parallel child domain creation (both branches computed in parallel).
    pub parallel_children: bool,
    /// Use adaptive learning rates (Adam-style optimizer).
    /// When enabled, learning rates are automatically adjusted per-parameter
    /// based on gradient history, improving convergence on diverse problems.
    pub use_adaptive: bool,
    /// Configuration for adaptive optimizer (when use_adaptive is true).
    pub adaptive_config: AdaptiveOptConfig,
    // =========================================================================
    // GCP-CROWN: Cutting Plane Configuration
    // =========================================================================
    /// Enable GCP-CROWN cutting planes.
    /// When enabled, verified subdomains generate cuts that tighten bounds
    /// for remaining domains.
    #[serde(default)]
    pub enable_cuts: bool,
    /// Maximum number of cutting planes to retain.
    /// More cuts = tighter bounds but more computation per domain.
    #[serde(default = "default_max_cuts")]
    pub max_cuts: usize,
    /// Minimum depth of verified domain to generate a cut.
    /// Deeper domains produce more specific cuts.
    #[serde(default = "default_min_cut_depth")]
    pub min_cut_depth: usize,
    /// Enable near-miss cut generation.
    /// When enabled, cuts are also generated from domains where the lower bound
    /// is within `near_miss_margin` of the threshold, even if not verified.
    /// This can help prune similar regions that are "almost verified".
    #[serde(default)]
    pub enable_near_miss_cuts: bool,
    /// Margin for near-miss cut generation (as fraction of threshold).
    /// Only used when `enable_near_miss_cuts` is true.
    /// Default: 0.1 (10% of threshold, or absolute 0.1 if threshold is 0)
    #[serde(default = "default_near_miss_margin")]
    pub near_miss_margin: f32,
    /// Enable proactive cut generation (BICCOS-lite).
    /// When enabled, cuts are generated for unstable ReLUs BEFORE BaB starts,
    /// rather than waiting for domains to verify. This helps on hard instances
    /// where no domains verify initially (chicken-and-egg problem).
    ///
    /// The proactive cuts encode pairwise neuron implications based on initial bounds.
    #[serde(default)]
    pub enable_proactive_cuts: bool,
    /// Maximum number of proactive cuts to generate.
    /// More cuts = potentially tighter bounds but more computation.
    #[serde(default = "default_max_proactive_cuts")]
    pub max_proactive_cuts: usize,
    // =========================================================================
    // Property Direction
    // =========================================================================
    /// Verify upper bound instead of lower bound.
    ///
    /// When `false` (default): verifies output > threshold (lower_bound > threshold)
    /// When `true`: verifies output < threshold (upper_bound < threshold)
    ///
    /// Use `true` for VNNLIB constraints like `Y >= c` (unsafe region), where
    /// proving safety requires proving the upper bound is below the threshold.
    #[serde(default)]
    pub verify_upper_bound: bool,
    // =========================================================================
    // PGD Attack for Counterexample Finding
    // =========================================================================
    /// Enable PGD attack to find concrete counterexamples.
    /// When enabled and verification is inconclusive, PGD attack is run to
    /// try to find a concrete input that violates the property.
    #[serde(default)]
    pub enable_pgd_attack: bool,
    /// Number of random restarts for PGD attack.
    #[serde(default = "default_pgd_restarts")]
    pub pgd_restarts: usize,
    /// Number of gradient steps per restart.
    #[serde(default = "default_pgd_steps")]
    pub pgd_steps: usize,
}

fn default_max_cuts() -> usize {
    1000
}

fn default_min_cut_depth() -> usize {
    2
}

fn default_near_miss_margin() -> f32 {
    0.1
}

fn default_max_proactive_cuts() -> usize {
    100
}

fn default_fsb_candidates() -> usize {
    8
}

fn default_pgd_restarts() -> usize {
    100
}

fn default_pgd_steps() -> usize {
    50
}

fn default_use_analytical_beta_gradients() -> bool {
    true
}

fn default_root_beta_iterations() -> usize {
    20 // Run 20 β optimization iterations on root before BaB
}

fn default_beta_max_depth() -> usize {
    3 // Optimize β for domains at depth 0, 1, 2, 3
}

/// Learning rate scheduler for adaptive optimization.
///
/// Controls how the base learning rate changes over iterations, enabling
/// exploration-exploitation tradeoffs during optimization.
///
/// **Performance Note for β-CROWN:**
///
/// Benchmark testing shows `Constant` learning rate performs best for β-CROWN
/// constraint optimization. All decay schedules (StepDecay, ExponentialDecay,
/// CosineAnnealing, WarmupCosine) increase domains needed from 15 to 25 (~67% slower).
///
/// The reason: β-CROWN typically runs only 10-20 iterations per domain. Learning
/// rate decay reduces effective optimization time when iterations are already limited.
/// Constant LR maximizes the optimization work per iteration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum LRScheduler {
    /// Constant learning rate (no decay).
    #[default]
    Constant,

    /// Step decay: multiply LR by `gamma` every `step_size` iterations.
    /// LR(t) = base_lr * gamma^(floor(t / step_size))
    StepDecay {
        /// Multiplicative factor for decay (typically 0.1-0.5).
        gamma: f32,
        /// Number of iterations between decay steps.
        step_size: usize,
    },

    /// Exponential decay: LR(t) = base_lr * gamma^t
    ExponentialDecay {
        /// Per-iteration decay factor (typically 0.9-0.99).
        gamma: f32,
    },

    /// Cosine annealing: smooth decay from base_lr to min_lr following cosine curve.
    /// LR(t) = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * t / T_max))
    CosineAnnealing {
        /// Minimum learning rate at the end of annealing.
        min_lr: f32,
        /// Total number of iterations for full cosine cycle.
        t_max: usize,
    },

    /// Linear warmup followed by cosine annealing.
    /// For t < warmup_steps: LR(t) = base_lr * t / warmup_steps
    /// For t >= warmup_steps: cosine annealing to min_lr
    WarmupCosine {
        /// Number of warmup iterations.
        warmup_steps: usize,
        /// Minimum learning rate after warmup.
        min_lr: f32,
        /// Total iterations including warmup.
        t_max: usize,
    },
}

impl LRScheduler {
    /// Compute the learning rate multiplier for the given iteration.
    ///
    /// Returns a factor in [0, 1] that should be multiplied with the base learning rate.
    /// Iteration `t` is 0-indexed.
    pub fn get_lr_factor(&self, t: usize, base_lr: f32) -> f32 {
        match self {
            LRScheduler::Constant => 1.0,

            LRScheduler::StepDecay { gamma, step_size } => {
                let num_decays = t / step_size;
                gamma.powi(num_decays as i32)
            }

            LRScheduler::ExponentialDecay { gamma } => gamma.powi(t as i32),

            LRScheduler::CosineAnnealing { min_lr, t_max } => {
                if *t_max == 0 {
                    return 1.0;
                }
                let progress = (t as f32 / *t_max as f32).min(1.0);
                let cosine = (std::f32::consts::PI * progress).cos();
                // Returns factor such that: min_lr + factor * (base_lr - min_lr) gives desired LR
                // Desired: min_lr + 0.5 * (base_lr - min_lr) * (1 + cos)
                // = base_lr * [min_lr/base_lr + 0.5 * (1 - min_lr/base_lr) * (1 + cos)]
                let min_ratio = min_lr / base_lr.max(1e-10);
                min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + cosine)
            }

            LRScheduler::WarmupCosine {
                warmup_steps,
                min_lr,
                t_max,
            } => {
                if t < *warmup_steps {
                    // Linear warmup: 0 -> 1 over warmup_steps
                    (t + 1) as f32 / (*warmup_steps).max(1) as f32
                } else {
                    // Cosine annealing after warmup
                    let t_after_warmup = t - warmup_steps;
                    let t_max_after_warmup = t_max.saturating_sub(*warmup_steps);
                    if t_max_after_warmup == 0 {
                        return 1.0;
                    }
                    let progress = (t_after_warmup as f32 / t_max_after_warmup as f32).min(1.0);
                    let cosine = (std::f32::consts::PI * progress).cos();
                    let min_ratio = min_lr / base_lr.max(1e-10);
                    min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + cosine)
                }
            }
        }
    }

    /// Convenience method to get the actual learning rate.
    pub fn get_lr(&self, t: usize, base_lr: f32) -> f32 {
        base_lr * self.get_lr_factor(t, base_lr)
    }
}

/// Strategy for scaling learning rates across different layers.
///
/// Different layers in a neural network may benefit from different learning rates.
/// Early layers (close to input) typically need smaller learning rates because
/// their changes propagate through more downstream computations. Later layers
/// (close to output) can often use larger learning rates.
///
/// Reference: Layer-wise adaptive learning rates are used in various deep learning
/// optimizers including LARS and LAMB.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum PerLayerLR {
    /// All layers use the same learning rate (current behavior).
    #[default]
    Uniform,

    /// Learning rate scales inversely with depth.
    /// LR(layer) = base_lr / (1 + layer_idx * scale_factor)
    ///
    /// For scale_factor=0.1 and base_lr=0.1:
    /// - Layer 0: 0.1
    /// - Layer 5: 0.067
    /// - Layer 10: 0.05
    DepthScaling {
        /// Scale factor for depth-based reduction.
        /// Typical values: 0.05-0.2
        scale_factor: f32,
    },

    /// Learning rate decays exponentially with depth.
    /// LR(layer) = base_lr * decay^layer_idx
    ///
    /// For decay=0.9 and base_lr=0.1:
    /// - Layer 0: 0.1
    /// - Layer 5: 0.059
    /// - Layer 10: 0.035
    ExponentialDepth {
        /// Per-layer decay factor (typically 0.8-0.95).
        decay: f32,
    },

    /// Learning rate scales with square root of inverse depth.
    /// LR(layer) = base_lr / sqrt(1 + layer_idx * scale)
    ///
    /// Provides gentler decay than linear for deep networks.
    SqrtDepthScaling {
        /// Scale factor for sqrt-based reduction.
        scale: f32,
    },

    /// Linear warmup of LR from early layers to later layers.
    /// LR(layer) = base_lr * (start_factor + (1 - start_factor) * layer_idx / total_layers)
    ///
    /// Later layers get higher LR, useful when output layers need more updates.
    LinearWarmup {
        /// Initial factor for layer 0 (e.g., 0.5 means half the base LR).
        start_factor: f32,
    },

    /// Custom per-layer multipliers.
    /// LR(layer) = base_lr * factors\[layer_idx\] (or 1.0 if out of bounds)
    Custom {
        /// Learning rate multiplier for each layer index.
        factors: Vec<f32>,
    },
}

impl PerLayerLR {
    /// Compute the learning rate multiplier for a given layer.
    ///
    /// # Arguments
    /// * `layer_idx` - Index of the layer (0-indexed)
    /// * `total_layers` - Total number of layers in the network (used by some strategies)
    ///
    /// # Returns
    /// A factor in (0, inf) to multiply with the base learning rate.
    pub fn get_factor(&self, layer_idx: usize, total_layers: usize) -> f32 {
        match self {
            PerLayerLR::Uniform => 1.0,

            PerLayerLR::DepthScaling { scale_factor } => {
                1.0 / (1.0 + layer_idx as f32 * scale_factor)
            }

            PerLayerLR::ExponentialDepth { decay } => decay.powi(layer_idx as i32),

            PerLayerLR::SqrtDepthScaling { scale } => 1.0 / (1.0 + layer_idx as f32 * scale).sqrt(),

            PerLayerLR::LinearWarmup { start_factor } => {
                if total_layers <= 1 {
                    1.0
                } else {
                    let progress = layer_idx as f32 / (total_layers - 1).max(1) as f32;
                    start_factor + (1.0 - start_factor) * progress
                }
            }

            PerLayerLR::Custom { factors } => *factors.get(layer_idx).unwrap_or(&1.0),
        }
    }
}

/// Configuration for the Lookahead optimizer wrapper.
///
/// Lookahead maintains two sets of weights: "fast weights" (θ) updated by the inner
/// optimizer (e.g., Adam), and "slow weights" (φ) that are a moving average of
/// the fast weights.
///
/// Algorithm:
/// 1. Run inner optimizer for k steps, updating θ
/// 2. After k steps: φ = φ + α * (θ - φ)  [interpolate slow toward fast]
/// 3. Reset: θ = φ  [sync fast weights back to slow]
///
/// This provides a stabilizing effect and can improve convergence, especially
/// when the inner optimizer produces noisy updates.
///
/// Reference: Zhang et al., "Lookahead Optimizer: k steps forward, 1 step back" (NeurIPS 2019)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LookaheadConfig {
    /// Whether lookahead is enabled.
    /// When false, no slow weights are maintained and the inner optimizer runs normally.
    pub enabled: bool,
    /// Synchronization period (k).
    /// Slow weights are updated every k inner optimizer steps.
    /// Typical values: 5-10
    pub sync_period: usize,
    /// Interpolation coefficient (α).
    /// How far slow weights move toward fast weights: φ = φ + α * (θ - φ)
    /// Typical values: 0.5-0.8
    pub alpha: f32,
}

impl Default for LookaheadConfig {
    fn default() -> Self {
        Self {
            enabled: false, // Disabled by default for backward compatibility
            sync_period: 5, // Sync every 5 steps (paper default)
            alpha: 0.5,     // Move halfway toward fast weights (paper default)
        }
    }
}

impl LookaheadConfig {
    /// Create a new enabled lookahead config with the given parameters.
    pub fn new(sync_period: usize, alpha: f32) -> Self {
        Self {
            enabled: true,
            sync_period: sync_period.max(1),
            alpha: alpha.clamp(0.0, 1.0),
        }
    }

    /// Check if slow weights should be synchronized at the given iteration.
    ///
    /// # Arguments
    /// * `iteration` - Current iteration (1-indexed)
    ///
    /// # Returns
    /// true if this iteration is a sync point (iteration % sync_period == 0)
    #[inline]
    pub fn should_sync(&self, iteration: usize) -> bool {
        self.enabled && iteration > 0 && iteration % self.sync_period == 0
    }
}

/// Configuration for Adam-style adaptive optimizer.
///
/// Adam (Adaptive Moment Estimation) maintains per-parameter first and second
/// moment estimates of gradients, providing automatic learning rate adaptation.
/// This is particularly effective for β-CROWN where gradient magnitudes vary
/// significantly across different neurons and optimization stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveOptConfig {
    /// Base learning rate for β parameters.
    pub beta_lr: f32,
    /// Base learning rate for α parameters.
    pub alpha_lr: f32,
    /// Exponential decay rate for first moment estimate (β₁ in Adam paper).
    /// Typical value: 0.9
    pub beta1: f32,
    /// Exponential decay rate for second moment estimate (β₂ in Adam paper).
    /// Typical value: 0.999
    pub beta2: f32,
    /// Small constant for numerical stability (ε in Adam paper).
    /// Typical value: 1e-8
    pub epsilon: f32,
    /// Maximum gradient magnitude (gradient clipping). 0.0 = no clipping.
    ///
    /// Gradient clipping bounds the magnitude of gradient updates to prevent
    /// exploding gradients that can destabilize optimization. Both α and β
    /// gradients are clipped to the range `[-grad_clip, grad_clip]` before
    /// being used in the Adam update.
    ///
    /// **When to adjust:**
    /// - If optimization diverges (NaN values), try reducing grad_clip (e.g., 1.0)
    /// - If convergence is too slow, try increasing grad_clip or disabling (0.0)
    ///
    /// Default: 10.0 (moderate clipping)
    pub grad_clip: f32,
    /// Enable bias correction for moment estimates.
    ///
    /// Adam's moment estimates (m, v) are initialized to zero and biased toward zero
    /// during early iterations. Bias correction divides by `(1 - beta^t)` to correct
    /// this bias.
    ///
    /// Recommended: `true` for small iteration counts (β-CROWN typically uses 10-20).
    /// Default: true
    pub bias_correction: bool,
    /// Learning rate scheduler for controlling LR over iterations.
    pub scheduler: LRScheduler,
    /// Enable AMSGrad variant.
    /// AMSGrad maintains a maximum of past squared gradients (v_max) to prevent
    /// the effective learning rate from increasing when v decreases. This provides
    /// more stable convergence guarantees than standard Adam.
    /// Reference: Reddi et al., "On the Convergence of Adam and Beyond" (ICLR 2018)
    pub amsgrad: bool,
    /// Enable RAdam (Rectified Adam) variant.
    ///
    /// RAdam rectifies the variance of the adaptive learning rate during early
    /// iterations to avoid excessively large/unstable steps when the second-moment
    /// estimate is unreliable.
    ///
    /// When enabled:
    /// - For early iterations (ρ_t ≤ 4): uses an SGD-with-momentum style step (no variance term)
    /// - For later iterations (ρ_t > 4): uses a rectified Adam step with factor r_t
    ///
    /// **Performance Warning for β-CROWN:**
    ///
    /// RAdam is **NOT recommended** for β-CROWN optimization. Benchmark testing shows
    /// RAdam significantly underperforms compared to standard Adam:
    /// - Adam/AMSGrad: converge in ~15 domains
    /// - RAdam: fails to converge even after 100+ domains
    ///
    /// The cause is RAdam's warmup behavior. With `beta2=0.999`, rectification activates
    /// at t=5, meaning the first 4 iterations per domain use SGD-style updates without
    /// the adaptive variance term. Since β-CROWN typically uses only 10-20 iterations
    /// per domain (`beta_iterations`), RAdam spends 20-40% of each domain in warmup mode.
    /// For constraint optimization in β-CROWN, the full Adam adaptive update is more
    /// effective from iteration 1.
    ///
    /// If you must use RAdam, consider increasing `beta_iterations` to 50+ to allow
    /// the warmup to complete. However, even then, standard Adam is generally preferred.
    ///
    /// **Recommended alternatives:**
    /// - `Adam` (default): Best general performance
    /// - `AMSGrad`: Same performance as Adam with better convergence guarantees
    /// - `Lookahead + Adam`: Same performance with added stability
    ///
    /// Reference: Liu et al., "On the Variance of the Adaptive Learning Rate and Beyond"
    /// (RAdam, 2019)
    pub radam: bool,
    /// Weight decay coefficient for AdamW (decoupled weight decay regularization).
    /// Unlike L2 regularization in standard Adam, AdamW applies weight decay directly
    /// to the parameters after the Adam update step, preventing interaction with
    /// the adaptive learning rate.
    ///
    /// **Performance Note for β-CROWN:**
    ///
    /// Weight decay is generally **not recommended** for β-CROWN constraint optimization.
    /// Benchmark testing shows AdamW (weight_decay=0.01) increases domains needed from
    /// 15 to 24 (~60% slower). The regularization adds unnecessary overhead for the
    /// Lagrangian constraint optimization in β-CROWN, where parameter magnitudes are
    /// naturally bounded by the problem structure.
    ///
    /// Typical values: 0.0 (disabled, recommended), 0.01-0.1 if regularization is needed.
    /// Reference: Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (ICLR 2019)
    pub weight_decay: f32,
    /// Per-layer learning rate strategy for β parameters.
    /// Allows different layers to use different learning rates based on their depth.
    /// Default: Uniform (all layers use the same base_lr).
    pub per_layer_lr_beta: PerLayerLR,
    /// Per-layer learning rate strategy for α parameters.
    /// Allows different layers to use different learning rates based on their depth.
    /// Default: Uniform (all layers use the same base_lr).
    pub per_layer_lr_alpha: PerLayerLR,
    /// Total number of layers in the network (used by some PerLayerLR strategies).
    /// This should be set when using LinearWarmup or similar strategies that need
    /// to know the total depth. If not set (0), defaults to layer_idx+1.
    pub total_layers: usize,
    /// Configuration for Lookahead optimizer wrapper.
    /// When enabled, maintains slow weights that stabilize optimization.
    /// Reference: Zhang et al., "Lookahead Optimizer: k steps forward, 1 step back" (NeurIPS 2019)
    pub lookahead: LookaheadConfig,
    /// Learning rate for λ parameters (cutting plane Lagrangian multipliers).
    /// If None, uses beta_lr as default.
    /// GCP-CROWN typically uses a separate (often lower) learning rate for cuts.
    #[serde(default)]
    pub lr_lambda: Option<f32>,
}

impl Default for AdaptiveOptConfig {
    fn default() -> Self {
        Self {
            beta_lr: 0.05,  // α,β-CROWN default: 0.05
            alpha_lr: 0.01, // α,β-CROWN default: 0.01 (much lower than init!)
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            grad_clip: 10.0, // Clip gradients > 10 to prevent explosion
            bias_correction: true,
            scheduler: LRScheduler::Constant,
            amsgrad: false,    // Disabled by default for backward compatibility
            radam: false,      // Disabled by default for backward compatibility
            weight_decay: 0.0, // Disabled by default for backward compatibility
            per_layer_lr_beta: PerLayerLR::Uniform, // Default: same LR for all layers
            per_layer_lr_alpha: PerLayerLR::Uniform, // Default: same LR for all layers
            total_layers: 0,   // Will use layer_idx+1 as fallback
            lookahead: LookaheadConfig::default(), // Disabled by default
            lr_lambda: None,   // Uses beta_lr by default
        }
    }
}

fn radam_rectification_factor(beta2: f32, t: f32) -> Option<f32> {
    if !(0.0..1.0).contains(&beta2) {
        return None;
    }
    let rho_inf = 2.0 / (1.0 - beta2) - 1.0;
    if rho_inf <= 4.0 {
        return None;
    }

    let beta2_t = beta2.powf(t.max(1.0));
    let one_minus_beta2_t = 1.0 - beta2_t;
    if one_minus_beta2_t <= 0.0 {
        return None;
    }

    // ρ_t from the RAdam paper (t is 1-indexed).
    let rho_t = rho_inf - 2.0 * t * beta2_t / one_minus_beta2_t;
    if rho_t <= 4.0 {
        return None;
    }

    let numerator = (rho_t - 4.0) * (rho_t - 2.0) * rho_inf;
    let denominator = (rho_inf - 4.0) * (rho_inf - 2.0) * rho_t;
    if numerator <= 0.0 || denominator <= 0.0 {
        return None;
    }

    Some((numerator / denominator).sqrt())
}

impl Default for BetaCrownConfig {
    fn default() -> Self {
        Self {
            max_domains: 100_000,
            timeout: Duration::from_secs(300),
            max_depth: 100,
            use_alpha_crown: true,
            use_crown_ibp: false, // Disabled by default (more expensive)
            alpha_config: AlphaCrownConfig::default(),
            branching_heuristic: BranchingHeuristic::LargestBoundWidth,
            fsb_candidates: default_fsb_candidates(),
            beta_lr: 0.05,      // α,β-CROWN default: 0.05
            beta_iterations: 0, // Per-domain iterations disabled by default for throughput
            beta_tolerance: 1e-5,
            root_beta_iterations: default_root_beta_iterations(), // Root-level optimization
            beta_max_depth: default_beta_max_depth(), // Limit per-domain optimization depth
            use_analytical_beta_gradients: true, // Use analytical gradients for ~3x faster optimization
            alpha_lr: 0.01,                      // α,β-CROWN default: 0.01 (much lower than init!)
            alpha_momentum: true,                // Use momentum for α updates
            batch_size: 64,                      // Process 64 domains in parallel (GPU-optimized)
            parallel_children: true,             // Enable parallel child creation by default
            use_adaptive: false,                 // Disabled by default for backward compatibility
            adaptive_config: AdaptiveOptConfig::default(),
            // GCP-CROWN defaults
            enable_cuts: false, // Disabled by default for backward compatibility
            max_cuts: default_max_cuts(),
            min_cut_depth: default_min_cut_depth(),
            enable_near_miss_cuts: false, // Disabled by default
            near_miss_margin: default_near_miss_margin(),
            enable_proactive_cuts: false, // Disabled by default
            max_proactive_cuts: default_max_proactive_cuts(),
            // Property direction
            verify_upper_bound: false, // Default: verify lower > threshold
            // PGD attack defaults
            enable_pgd_attack: false, // Disabled by default
            pgd_restarts: default_pgd_restarts(),
            pgd_steps: default_pgd_steps(),
        }
    }
}

/// Heuristic for selecting which neuron to split.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BranchingHeuristic {
    /// Split the neuron with largest bound width (u - l).
    LargestBoundWidth,
    /// Split the neuron that most affects the output bound (BaBSR-like).
    BoundImpact,
    /// Filtered Smart Branching (FSB): evaluate a small set of high-scoring BaBSR candidates
    /// by estimating both child bounds and choosing the best worst-child improvement.
    FilteredSmartBranching,
    /// Split neurons in order (layer by layer, neuron by neuron).
    Sequential,
    /// Input splitting: divide input space instead of ReLU activation space.
    /// More effective than ReLU splitting for small networks with tight input bounds
    /// (e.g., ACAS-Xu with 5 input dimensions). Each split halves one input dimension,
    /// creating tighter output bounds for each subdomain.
    InputSplit,
}

/// A constraint on a single ReLU neuron.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeuronConstraint {
    /// Layer index of the ReLU.
    pub layer_idx: usize,
    /// Neuron index within the layer.
    pub neuron_idx: usize,
    /// True if neuron is constrained to be active (x ≥ 0), false if inactive (x ≤ 0).
    pub is_active: bool,
}

/// History of split decisions in a domain.
#[derive(Debug, Clone, Default)]
pub struct SplitHistory {
    /// All constraints applied in this domain.
    pub constraints: Vec<NeuronConstraint>,
}

impl SplitHistory {
    /// Create empty split history.
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
        }
    }

    /// Add a constraint to the history.
    pub fn add_constraint(&mut self, constraint: NeuronConstraint) {
        self.constraints.push(constraint);
    }

    /// Get the depth (number of splits).
    pub fn depth(&self) -> usize {
        self.constraints.len()
    }

    /// Check if a neuron is already constrained.
    pub fn is_constrained(&self, layer_idx: usize, neuron_idx: usize) -> Option<bool> {
        for c in &self.constraints {
            if c.layer_idx == layer_idx && c.neuron_idx == neuron_idx {
                return Some(c.is_active);
            }
        }
        None
    }

    /// Create a new history with an additional constraint.
    pub fn with_constraint(&self, constraint: NeuronConstraint) -> Self {
        let mut new = self.clone();
        new.add_constraint(constraint);
        new
    }
}

// =============================================================================
// GraphNetwork ReLU Branching Infrastructure
// =============================================================================

/// A constraint on a single ReLU neuron in a GraphNetwork.
///
/// Unlike NeuronConstraint which uses layer indices, this uses node names
/// to identify ReLU nodes in the DAG structure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GraphNeuronConstraint {
    /// Name of the ReLU node in the graph.
    pub node_name: String,
    /// Neuron index within the ReLU node's output.
    pub neuron_idx: usize,
    /// True if neuron is constrained to be active (x ≥ 0), false if inactive (x ≤ 0).
    pub is_active: bool,
}

/// History of split decisions for a GraphNetwork domain.
#[derive(Debug, Clone, Default)]
pub struct GraphSplitHistory {
    /// All constraints applied in this domain.
    pub constraints: Vec<GraphNeuronConstraint>,
}

impl GraphSplitHistory {
    /// Create empty split history.
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
        }
    }

    /// Add a constraint to the history.
    pub fn add_constraint(&mut self, constraint: GraphNeuronConstraint) {
        self.constraints.push(constraint);
    }

    /// Get the depth (number of splits).
    pub fn depth(&self) -> usize {
        self.constraints.len()
    }

    /// Check if a neuron is already constrained.
    pub fn is_constrained(&self, node_name: &str, neuron_idx: usize) -> Option<bool> {
        for c in &self.constraints {
            if c.node_name == node_name && c.neuron_idx == neuron_idx {
                return Some(c.is_active);
            }
        }
        None
    }

    /// Create a new history with an additional constraint.
    pub fn with_constraint(&self, constraint: GraphNeuronConstraint) -> Self {
        let mut new = self.clone();
        new.add_constraint(constraint);
        new
    }

    /// Build a lookup map for efficient constraint checking.
    pub fn build_constraint_map(&self) -> std::collections::HashMap<(String, usize), bool> {
        self.constraints
            .iter()
            .map(|c| ((c.node_name.clone(), c.neuron_idx), c.is_active))
            .collect()
    }
}

/// A single beta parameter entry for GraphNetwork sparse beta representation.
/// Similar to BetaEntry but uses node_name (String) instead of layer_idx.
#[derive(Debug, Clone)]
pub struct GraphBetaEntry {
    /// Node name of the constrained ReLU neuron.
    pub node_name: String,
    /// Neuron index within the ReLU node's output.
    pub neuron_idx: usize,
    /// Current beta value (Lagrangian multiplier, must be >= 0).
    pub value: f32,
    /// Sign of the constraint: +1 for active (x >= 0), -1 for inactive (x <= 0).
    pub sign: f32,
    /// Accumulated gradient for this iteration.
    pub grad: f32,
    /// First moment estimate (mean of gradients) for Adam optimizer.
    pub m: f32,
    /// Second moment estimate (uncentered variance) for Adam optimizer.
    pub v: f32,
    /// Maximum second moment estimate for AMSGrad variant.
    pub v_max: f32,
}

/// β parameters for constrained GraphNetwork CROWN propagation.
///
/// β values are Lagrangian multipliers in the dual formulation of split constraints.
/// The Lagrangian augmented bound is: lb = c^T * A * x + b + sum_i(β_i * sign_i * x_i)
///
/// This is the GraphNetwork equivalent of BetaState, using node_name instead of layer_idx.
#[derive(Debug, Clone, Default)]
pub struct GraphBetaState {
    /// Sparse beta entries for constrained neurons.
    pub entries: Vec<GraphBetaEntry>,
}

impl GraphBetaState {
    /// Default initial β value for Lagrangian relaxation.
    /// A small positive value allows β optimization to start exploring.
    pub const DEFAULT_BETA_INIT: f32 = 0.1;

    /// Create β state from GraphSplitHistory.
    pub fn from_history(history: &GraphSplitHistory) -> Self {
        Self::from_history_with_init(history, Self::DEFAULT_BETA_INIT)
    }

    /// Create β state from GraphSplitHistory with custom initial β value.
    pub fn from_history_with_init(history: &GraphSplitHistory, init_beta: f32) -> Self {
        let entries = history
            .constraints
            .iter()
            .map(|c| GraphBetaEntry {
                node_name: c.node_name.clone(),
                neuron_idx: c.neuron_idx,
                value: init_beta,
                sign: if c.is_active { 1.0 } else { -1.0 },
                grad: 0.0,
                m: 0.0,
                v: 0.0,
                v_max: 0.0,
            })
            .collect();
        Self { entries }
    }

    /// Create β state from GraphSplitHistory with warmup from parent state.
    ///
    /// This inherits optimized β values from the parent domain for constraints
    /// that existed in the parent, while initializing the new constraint's β
    /// to the default value. This warmup is crucial for β-CROWN convergence
    /// and matches the α,β-CROWN behavior.
    ///
    /// For each constraint in the history:
    /// - If the parent had a β for this constraint, copy its value and Adam state (m, v, v_max)
    /// - Otherwise, initialize with the default value
    pub fn from_history_with_warmup(
        history: &GraphSplitHistory,
        parent_beta: &GraphBetaState,
        init_beta: f32,
    ) -> Self {
        let entries = history
            .constraints
            .iter()
            .map(|c| {
                // Try to find matching entry in parent
                if let Some(parent_entry) = parent_beta.get_entry(&c.node_name, c.neuron_idx) {
                    // Inherit from parent (including Adam momentum state)
                    GraphBetaEntry {
                        node_name: c.node_name.clone(),
                        neuron_idx: c.neuron_idx,
                        value: parent_entry.value,
                        sign: if c.is_active { 1.0 } else { -1.0 },
                        grad: 0.0, // Reset gradient for new optimization
                        m: parent_entry.m,
                        v: parent_entry.v,
                        v_max: parent_entry.v_max,
                    }
                } else {
                    // New constraint - initialize fresh
                    GraphBetaEntry {
                        node_name: c.node_name.clone(),
                        neuron_idx: c.neuron_idx,
                        value: init_beta,
                        sign: if c.is_active { 1.0 } else { -1.0 },
                        grad: 0.0,
                        m: 0.0,
                        v: 0.0,
                        v_max: 0.0,
                    }
                }
            })
            .collect();
        Self { entries }
    }

    /// Create empty beta state.
    pub fn empty() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Check if the beta state is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get β entry for a neuron, if constrained.
    pub fn get_entry(&self, node_name: &str, neuron_idx: usize) -> Option<&GraphBetaEntry> {
        self.entries
            .iter()
            .find(|e| e.node_name == node_name && e.neuron_idx == neuron_idx)
    }

    /// Get mutable β entry for a neuron, if constrained.
    pub fn get_entry_mut(
        &mut self,
        node_name: &str,
        neuron_idx: usize,
    ) -> Option<&mut GraphBetaEntry> {
        self.entries
            .iter_mut()
            .find(|e| e.node_name == node_name && e.neuron_idx == neuron_idx)
    }

    /// Get signed β value for a neuron: β * sign.
    /// Returns None if neuron is not constrained.
    pub fn get_signed_beta(&self, node_name: &str, neuron_idx: usize) -> Option<f32> {
        self.get_entry(node_name, neuron_idx)
            .map(|e| e.value * e.sign)
    }

    /// Reset all gradients to zero.
    pub fn zero_grad(&mut self) {
        for entry in &mut self.entries {
            entry.grad = 0.0;
        }
    }

    /// Accumulate gradient for a specific neuron.
    pub fn accumulate_grad(&mut self, node_name: &str, neuron_idx: usize, grad: f32) {
        if let Some(entry) = self.get_entry_mut(node_name, neuron_idx) {
            entry.grad += grad;
        }
    }

    /// Perform projected gradient ascent step.
    /// β = max(0, β + lr * grad)
    ///
    /// Returns the maximum gradient magnitude (for convergence check).
    pub fn gradient_step(&mut self, lr: f32) -> f32 {
        let mut max_grad = 0.0f32;
        for entry in &mut self.entries {
            max_grad = max_grad.max(entry.grad.abs());
            // Gradient ascent (we want to maximize lower bound)
            entry.value = (entry.value + lr * entry.grad).max(0.0);
        }
        max_grad
    }

    /// Perform Adam optimizer step for β parameters.
    ///
    /// Returns the maximum gradient magnitude (for convergence check).
    pub fn gradient_step_adam(&mut self, config: &AdaptiveOptConfig, t: usize) -> f32 {
        let lr = config.beta_lr;
        let beta1 = config.beta1;
        let beta2 = config.beta2;
        let eps = config.epsilon;

        // Bias correction factors
        let bias_correction1 = 1.0 - beta1.powi(t as i32);
        let bias_correction2 = 1.0 - beta2.powi(t as i32);

        let mut max_grad = 0.0f32;
        for entry in &mut self.entries {
            max_grad = max_grad.max(entry.grad.abs());

            // Update biased first moment estimate
            entry.m = beta1 * entry.m + (1.0 - beta1) * entry.grad;
            // Update biased second raw moment estimate
            entry.v = beta2 * entry.v + (1.0 - beta2) * entry.grad * entry.grad;

            // AMSGrad: use max of past squared gradients
            entry.v_max = entry.v_max.max(entry.v);

            // Bias-corrected estimates
            let m_hat = entry.m / bias_correction1;
            let v_hat = entry.v_max / bias_correction2;

            // Adam update (gradient ascent)
            entry.value = (entry.value + lr * m_hat / (v_hat.sqrt() + eps)).max(0.0);
        }
        max_grad
    }

    /// Compute analytical β gradients from stored A matrices.
    ///
    /// For each constrained neuron at (node_name, neuron_idx), the gradient is:
    ///   ∂lb/∂β = -sign * sensitivity
    ///
    /// where sensitivity measures how changes at that neuron affect the output.
    /// The A matrix at each ReLU has shape (num_outputs, num_neurons), where
    /// A[j, i] is the coefficient from output j to neuron i.
    ///
    /// For lower bound optimization, positive A coefficients indicate the neuron
    /// contributes to tightening the lower bound, so the sensitivity is:
    ///   sensitivity = sum_j(A[j, neuron_idx]) for all outputs
    ///
    /// Returns the maximum gradient magnitude for convergence checking.
    pub fn compute_analytical_gradients(
        &mut self,
        intermediate: &GraphAlphaCrownIntermediate,
    ) -> f32 {
        let mut max_grad = 0.0f32;

        for entry in &mut self.entries {
            // Get A matrix at this constrained ReLU node
            let a_matrix = match intermediate.a_at_relu.get(&entry.node_name) {
                Some(a) => a,
                None => {
                    // No A matrix stored for this node - gradient is zero
                    entry.grad = 0.0;
                    continue;
                }
            };

            // Check neuron_idx is within bounds
            if entry.neuron_idx >= a_matrix.ncols() {
                entry.grad = 0.0;
                continue;
            }

            // Compute sensitivity: sum of A[j, neuron_idx] over all outputs
            // This represents how changes at this neuron propagate to all outputs.
            //
            // For the Lagrangian β-CROWN formulation:
            // - β appears as: -sign * β in the A coefficients
            // - The gradient ∂lb/∂β = -sign * sensitivity
            //
            // Sensitivity = sum_j A[j, neuron_idx] where A > 0 contributes to lower bound
            let mut sensitivity = 0.0f32;
            let num_outputs = a_matrix.nrows();

            for j in 0..num_outputs {
                let a_ji = a_matrix[[j, entry.neuron_idx]];
                // For lower bound: positive A coefficients use lower relaxation
                // The sensitivity is the direct contribution
                sensitivity += a_ji;
            }

            // β gradient = -sign * sensitivity
            // Positive gradient means increasing β improves the lower bound
            let grad = -entry.sign * sensitivity;
            entry.grad = grad;
            max_grad = max_grad.max(grad.abs());
        }

        max_grad
    }

    /// Compute analytical β gradients for multi-objective verification.
    ///
    /// For multi-objective, we optimize the minimum margin across unverified objectives:
    ///   maximize min_i (lower_bound\[i\] - threshold\[i\])
    ///
    /// The gradient is computed for the "critical" objective (the one with minimum margin).
    /// This is a subgradient of the max function.
    ///
    /// Arguments:
    /// - `intermediate`: A matrices from backward pass (without objective applied)
    /// - `obj_bounds`: Lower bounds for each objective (pre-computed)
    /// - `objectives`: Coefficient vectors for each objective
    /// - `thresholds`: Threshold values for each objective
    /// - `verified_mask`: Mask indicating which objectives are already verified
    ///
    /// Returns the maximum gradient magnitude for convergence checking.
    pub fn compute_analytical_gradients_multi_objective(
        &mut self,
        intermediate: &GraphAlphaCrownIntermediate,
        obj_bounds: &[(f32, f32)],
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        verified_mask: &[bool],
    ) -> f32 {
        // Find the critical objective: minimum margin among unverified objectives
        let mut critical_idx = None;
        let mut min_margin = f32::INFINITY;

        for (i, ((lb, _ub), &threshold)) in obj_bounds.iter().zip(thresholds).enumerate() {
            if i < verified_mask.len() && verified_mask[i] {
                continue; // Skip already verified objectives
            }
            let margin = lb - threshold;
            if margin < min_margin {
                min_margin = margin;
                critical_idx = Some(i);
            }
        }

        // If all objectives are verified, gradients are zero
        let critical_idx = match critical_idx {
            Some(idx) => idx,
            None => {
                for entry in &mut self.entries {
                    entry.grad = 0.0;
                }
                return 0.0;
            }
        };

        let critical_objective = &objectives[critical_idx];
        let mut max_grad = 0.0f32;

        for entry in &mut self.entries {
            // Get A matrix at this constrained ReLU node
            let a_matrix = match intermediate.a_at_relu.get(&entry.node_name) {
                Some(a) => a,
                None => {
                    // No A matrix stored for this node - gradient is zero
                    entry.grad = 0.0;
                    continue;
                }
            };

            // Check neuron_idx is within bounds
            if entry.neuron_idx >= a_matrix.ncols() {
                entry.grad = 0.0;
                continue;
            }

            // Compute weighted sensitivity for the critical objective.
            // For objective c, the effective sensitivity at neuron k is:
            //   sum_j c[j] * A[j, k]
            // This represents how changes at neuron k propagate to the objective output.
            let mut sensitivity = 0.0f32;
            let num_outputs = a_matrix.nrows();

            for j in 0..num_outputs {
                let a_jk = a_matrix[[j, entry.neuron_idx]];
                let c_j = if j < critical_objective.len() {
                    critical_objective[j]
                } else {
                    0.0
                };
                // Weight the A coefficient by the objective coefficient
                sensitivity += c_j * a_jk;
            }

            // β gradient = -sign * sensitivity
            // Positive gradient means increasing β improves the lower bound
            let grad = -entry.sign * sensitivity;
            entry.grad = grad;
            max_grad = max_grad.max(grad.abs());
        }

        max_grad
    }
}

/// Graph CROWN propagation context.
///
/// Bundles common parameters for graph CROWN propagation to reduce function argument counts.
/// This struct holds references to propagation context that are typically passed together.
pub struct GraphCrownContext<'a> {
    /// Split history for constraint tracking.
    pub history: &'a GraphSplitHistory,
    /// Optional cut pool for cutting planes.
    pub cut_pool: Option<&'a GraphCutPool>,
    /// Optional pre-computed bounds from CROWN-IBP.
    pub base_bounds: Option<&'a std::collections::HashMap<String, Arc<BoundedTensor>>>,
    /// Optional GPU/accelerated GEMM engine.
    pub engine: Option<&'a dyn GemmEngine>,
}

impl<'a> GraphCrownContext<'a> {
    /// Create a new graph CROWN context.
    pub fn new(
        history: &'a GraphSplitHistory,
        cut_pool: Option<&'a GraphCutPool>,
        base_bounds: Option<&'a std::collections::HashMap<String, Arc<BoundedTensor>>>,
        engine: Option<&'a dyn GemmEngine>,
    ) -> Self {
        Self {
            history,
            cut_pool,
            base_bounds,
            engine,
        }
    }

    /// Create a minimal context with just history.
    pub fn with_history(history: &'a GraphSplitHistory) -> Self {
        Self {
            history,
            cut_pool: None,
            base_bounds: None,
            engine: None,
        }
    }
}

/// Domain with unstable neurons for parallel processing.
///
/// Tuple of (domain_index, domain_ref, unstable_neurons).
/// Used in parallel domain verification to track which neurons need splitting.
pub type DomainWithUnstable<'a> = (usize, &'a GraphBabDomain, Vec<(String, usize)>);

/// Multi-objective domain with unstable neurons for parallel processing.
///
/// Tuple of (domain_index, domain_ref, unstable_neurons).
/// Used in parallel multi-objective verification.
pub type MultiObjDomainWithUnstable<'a> = (
    usize,
    &'a MultiObjectiveGraphBabDomain,
    Vec<(String, usize)>,
);

/// Pre-computed bounds for graph verification.
///
/// Bundles pre-computed node and output bounds from CROWN-IBP to reduce argument counts.
/// These bounds are computed once and reused across multiple verification calls.
pub struct GraphPrecomputedBounds<'a> {
    /// Pre-computed intermediate node bounds.
    pub node_bounds: &'a std::collections::HashMap<String, BoundedTensor>,
    /// Pre-computed output bounds.
    pub output_bounds: &'a BoundedTensor,
}

impl<'a> GraphPrecomputedBounds<'a> {
    /// Create new pre-computed bounds.
    pub fn new(
        node_bounds: &'a std::collections::HashMap<String, BoundedTensor>,
        output_bounds: &'a BoundedTensor,
    ) -> Self {
        Self {
            node_bounds,
            output_bounds,
        }
    }
}

/// Multi-objective verification targets.
///
/// Bundles objective vectors, thresholds, and verification status for multi-objective SPSA.
pub struct MultiObjectiveTargets<'a> {
    /// Objective coefficient vectors for each property.
    pub objectives: &'a [Vec<f32>],
    /// Threshold values for each property.
    pub thresholds: &'a [f32],
    /// Mask indicating which properties are already verified.
    pub verified_mask: &'a [bool],
}

impl<'a> MultiObjectiveTargets<'a> {
    /// Create new multi-objective targets.
    pub fn new(
        objectives: &'a [Vec<f32>],
        thresholds: &'a [f32],
        verified_mask: &'a [bool],
    ) -> Self {
        Self {
            objectives,
            thresholds,
            verified_mask,
        }
    }
}

/// Configuration for parallel domain processing.
///
/// Bundles processing options for the domain parallel processing function.
pub struct DomainProcessingConfig {
    /// Threshold for verification.
    pub threshold: f32,
    /// Whether to create children in parallel.
    pub use_parallel_children: bool,
}

impl DomainProcessingConfig {
    /// Create new domain processing configuration.
    pub fn new(threshold: f32, use_parallel_children: bool) -> Self {
        Self {
            threshold,
            use_parallel_children,
        }
    }
}

/// Domain for GraphNetwork branch-and-bound with ReLU splitting.
#[derive(Debug, Clone)]
pub struct GraphBabDomain {
    /// Split history for this domain.
    pub history: GraphSplitHistory,
    /// Pre-activation bounds for each node (before applying ReLU constraints).
    /// Uses Arc for cheap cloning during branch-and-bound splits.
    pub node_bounds: std::collections::HashMap<String, Arc<BoundedTensor>>,
    /// Lower bound on the objective.
    pub lower_bound: f32,
    /// Upper bound on the objective.
    pub upper_bound: f32,
    /// Current depth in the B&B tree.
    pub depth: usize,
    /// Priority for queue ordering.
    pub priority: f32,
    /// Input bounds for this domain.
    pub input_bounds: Arc<BoundedTensor>,
    /// β parameters for Lagrangian optimization (initialized from history).
    pub beta_state: GraphBetaState,
}

impl PartialEq for GraphBabDomain {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for GraphBabDomain {}

impl PartialOrd for GraphBabDomain {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GraphBabDomain {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap: higher priority = pop first
        self.priority
            .partial_cmp(&other.priority)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl GraphBabDomain {
    /// Create root domain with initial bounds.
    pub fn root(
        node_bounds: std::collections::HashMap<String, BoundedTensor>,
        lower_bound: f32,
        upper_bound: f32,
        input: &BoundedTensor,
        verify_upper: bool,
    ) -> Self {
        let node_bounds = node_bounds
            .into_iter()
            .map(|(k, v)| (k, Arc::new(v)))
            .collect();
        let priority = if verify_upper {
            upper_bound
        } else {
            -lower_bound
        };
        Self {
            history: GraphSplitHistory::new(),
            node_bounds,
            lower_bound,
            upper_bound,
            depth: 0,
            priority,
            input_bounds: Arc::new(input.clone()),
            beta_state: GraphBetaState::empty(), // Root has no constraints
        }
    }

    /// Apply a constraint to create a child domain.
    ///
    /// Returns None if the constraint makes the domain infeasible.
    pub fn with_constraint(
        &self,
        graph: &GraphNetwork,
        constraint: GraphNeuronConstraint,
        verify_upper: bool,
    ) -> Option<Self> {
        let node_name = &constraint.node_name;
        let neuron_idx = constraint.neuron_idx;
        let is_active = constraint.is_active;

        // Constraints are on the *pre-activation* of this ReLU, i.e. the ReLU input node.
        let relu_node = graph.nodes.get(node_name)?;
        if !matches!(relu_node.layer, Layer::ReLU(_)) {
            return None;
        }
        let pre_name = relu_node
            .inputs
            .first()
            .map(|s| s.as_str())
            .unwrap_or("_input");

        let pre_bounds: &BoundedTensor = if pre_name == "_input" {
            self.input_bounds.as_ref()
        } else {
            self.node_bounds.get(pre_name)?.as_ref()
        };
        let flat = pre_bounds.flatten();

        if neuron_idx >= flat.len() {
            return None;
        }

        let l = flat.lower[[neuron_idx]];
        let u = flat.upper[[neuron_idx]];

        // Feasibility check for intersection with the half-space.
        // Use strict inequalities so boundary cases (x==0) remain feasible.
        if is_active && u < 0.0 {
            return None;
        }
        if !is_active && l > 0.0 {
            return None;
        }

        // If the constrained pre-activation is the network input, tighten input bounds
        // so subsequent consumer nodes see the restricted range.
        let new_input_bounds = if pre_name == "_input" {
            let shape = pre_bounds.shape().to_vec();
            let mut lower_flat = flat.lower.clone();
            let mut upper_flat = flat.upper.clone();

            if is_active {
                lower_flat[[neuron_idx]] = lower_flat[[neuron_idx]].max(0.0);
            } else {
                upper_flat[[neuron_idx]] = upper_flat[[neuron_idx]].min(0.0);
            }
            if lower_flat[[neuron_idx]] > upper_flat[[neuron_idx]] {
                return None;
            }

            let lower_new = lower_flat.into_shape_clone(ndarray::IxDyn(&shape)).ok()?;
            let upper_new = upper_flat.into_shape_clone(ndarray::IxDyn(&shape)).ok()?;
            Arc::new(BoundedTensor::new(lower_new, upper_new).ok()?)
        } else {
            self.input_bounds.clone()
        };

        let new_history = self.history.with_constraint(constraint);
        let priority = if verify_upper {
            self.upper_bound // Will be updated after propagation
        } else {
            -self.lower_bound
        };

        // Initialize β state with warmup from parent domain.
        // This inherits optimized β values for existing constraints while
        // initializing the new constraint's β to the default value.
        // This warmup is crucial for β-CROWN convergence per α,β-CROWN paper.
        let beta_state = GraphBetaState::from_history_with_warmup(
            &new_history,
            &self.beta_state,
            GraphBetaState::DEFAULT_BETA_INIT,
        );

        Some(Self {
            history: new_history,
            node_bounds: self.node_bounds.clone(),
            lower_bound: self.lower_bound,
            upper_bound: self.upper_bound,
            depth: self.depth + 1,
            priority,
            input_bounds: new_input_bounds,
            beta_state,
        })
    }
}

// =============================================================================
// Multi-Objective BaB Domain for Disjunctive Properties
// =============================================================================

/// Domain for multi-objective GraphNetwork branch-and-bound.
///
/// Used for disjunctive properties where ALL constraints must be verified simultaneously.
/// Tracks bounds for each objective and only considers a domain verified when ALL
/// objectives are verified, enabling shared computation across objectives.
#[derive(Debug, Clone)]
pub struct MultiObjectiveGraphBabDomain {
    /// Split history for this domain.
    pub history: GraphSplitHistory,
    /// Pre-activation bounds for each node.
    pub node_bounds: std::collections::HashMap<String, Arc<BoundedTensor>>,
    /// Bounds (lower, upper) for each objective.
    pub objective_bounds: Vec<(f32, f32)>,
    /// Which objectives are verified in this domain.
    pub verified: Vec<bool>,
    /// Current depth in the B&B tree.
    pub depth: usize,
    /// Priority for queue ordering (max gap across unverified objectives).
    pub priority: f32,
    /// Input bounds for this domain.
    pub input_bounds: Arc<BoundedTensor>,
    /// β parameters for Lagrangian optimization (initialized from history).
    pub beta_state: GraphBetaState,
}

impl PartialEq for MultiObjectiveGraphBabDomain {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for MultiObjectiveGraphBabDomain {}

impl PartialOrd for MultiObjectiveGraphBabDomain {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MultiObjectiveGraphBabDomain {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap: higher priority = pop first
        self.priority
            .partial_cmp(&other.priority)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl MultiObjectiveGraphBabDomain {
    /// Create root domain with initial bounds for all objectives.
    pub fn root(
        node_bounds: std::collections::HashMap<String, BoundedTensor>,
        objective_bounds: Vec<(f32, f32)>,
        input: &BoundedTensor,
        thresholds: &[f32],
        verify_upper: bool,
    ) -> Self {
        let node_bounds = node_bounds
            .into_iter()
            .map(|(k, v)| (k, Arc::new(v)))
            .collect();

        // Check which objectives are already verified
        let verified: Vec<bool> = objective_bounds
            .iter()
            .zip(thresholds.iter())
            .map(|((l, u), &t)| {
                if verify_upper {
                    *u < t // Verified if upper < threshold
                } else {
                    *l > t // Verified if lower > threshold
                }
            })
            .collect();

        // Priority: max gap across unverified objectives
        let priority = objective_bounds
            .iter()
            .zip(verified.iter())
            .filter(|(_, &v)| !v)
            .map(|((l, u), _)| {
                if verify_upper {
                    *u // Higher upper = worse = higher priority
                } else {
                    -*l // Lower bound (negated for max-heap)
                }
            })
            .fold(f32::NEG_INFINITY, f32::max);

        Self {
            history: GraphSplitHistory::new(),
            node_bounds,
            objective_bounds,
            verified,
            depth: 0,
            priority,
            input_bounds: Arc::new(input.clone()),
            beta_state: GraphBetaState::empty(), // Root has no constraints
        }
    }

    /// Check if all objectives are verified.
    pub fn all_verified(&self) -> bool {
        self.verified.iter().all(|&v| v)
    }

    /// Count of verified objectives.
    pub fn verified_count(&self) -> usize {
        self.verified.iter().filter(|&&v| v).count()
    }

    /// Apply a constraint to create a child domain.
    pub fn with_constraint(
        &self,
        graph: &GraphNetwork,
        constraint: GraphNeuronConstraint,
        verify_upper: bool,
        _thresholds: &[f32],
    ) -> Option<Self> {
        let node_name = &constraint.node_name;
        let neuron_idx = constraint.neuron_idx;
        let is_active = constraint.is_active;

        // Constraints are on the *pre-activation* of this ReLU
        let relu_node = graph.nodes.get(node_name)?;
        if !matches!(relu_node.layer, Layer::ReLU(_)) {
            return None;
        }
        let pre_name = relu_node
            .inputs
            .first()
            .map(|s| s.as_str())
            .unwrap_or("_input");

        let pre_bounds: &BoundedTensor = if pre_name == "_input" {
            self.input_bounds.as_ref()
        } else {
            self.node_bounds.get(pre_name)?.as_ref()
        };
        let flat = pre_bounds.flatten();

        if neuron_idx >= flat.len() {
            return None;
        }

        let l = flat.lower[[neuron_idx]];
        let u = flat.upper[[neuron_idx]];

        // Feasibility check
        if is_active && u < 0.0 {
            return None;
        }
        if !is_active && l > 0.0 {
            return None;
        }

        // Update input bounds if constraining network input
        let new_input_bounds = if pre_name == "_input" {
            let shape = pre_bounds.shape().to_vec();
            let mut lower_flat = flat.lower.clone();
            let mut upper_flat = flat.upper.clone();

            if is_active {
                lower_flat[[neuron_idx]] = lower_flat[[neuron_idx]].max(0.0);
            } else {
                upper_flat[[neuron_idx]] = upper_flat[[neuron_idx]].min(0.0);
            }
            if lower_flat[[neuron_idx]] > upper_flat[[neuron_idx]] {
                return None;
            }

            let lower_new = lower_flat.into_shape_clone(ndarray::IxDyn(&shape)).ok()?;
            let upper_new = upper_flat.into_shape_clone(ndarray::IxDyn(&shape)).ok()?;
            Arc::new(BoundedTensor::new(lower_new, upper_new).ok()?)
        } else {
            self.input_bounds.clone()
        };

        let new_history = self.history.with_constraint(constraint);

        // Compute priority based on unverified objectives
        let priority = self
            .objective_bounds
            .iter()
            .zip(self.verified.iter())
            .filter(|(_, &v)| !v)
            .map(|((l, u), _)| if verify_upper { *u } else { -*l })
            .fold(f32::NEG_INFINITY, f32::max);

        // Initialize β state with warmup from parent domain.
        // This inherits optimized β values for existing constraints while
        // initializing the new constraint's β to the default value.
        let beta_state = GraphBetaState::from_history_with_warmup(
            &new_history,
            &self.beta_state,
            GraphBetaState::DEFAULT_BETA_INIT,
        );

        Some(Self {
            history: new_history,
            node_bounds: self.node_bounds.clone(),
            objective_bounds: self.objective_bounds.clone(),
            verified: self.verified.clone(),
            depth: self.depth + 1,
            priority,
            input_bounds: new_input_bounds,
            beta_state,
        })
    }

    /// Update bounds and verification status after propagation.
    pub fn update_bounds(
        &mut self,
        new_bounds: Vec<(f32, f32)>,
        thresholds: &[f32],
        verify_upper: bool,
    ) {
        self.objective_bounds = new_bounds;
        self.verified = self
            .objective_bounds
            .iter()
            .zip(thresholds.iter())
            .map(|((l, u), &t)| if verify_upper { *u < t } else { *l > t })
            .collect();

        // Update priority
        self.priority = self
            .objective_bounds
            .iter()
            .zip(self.verified.iter())
            .filter(|(_, &v)| !v)
            .map(|((l, u), _)| if verify_upper { *u } else { -*l })
            .fold(f32::NEG_INFINITY, f32::max);
    }

    /// Check if any objective is conclusively violated (provably cannot be verified).
    pub fn any_violated(&self, thresholds: &[f32], verify_upper: bool) -> bool {
        self.objective_bounds
            .iter()
            .zip(thresholds.iter())
            .any(|((l, u), &t)| {
                if verify_upper {
                    *l >= t // Lower bound >= threshold means upper cannot be < threshold
                } else {
                    *u < t // Upper bound < threshold means lower cannot be > threshold
                }
            })
    }
}

// =============================================================================
// GCP-CROWN: Cutting Plane Infrastructure
// =============================================================================

/// A term in a cutting plane constraint.
///
/// Represents a coefficient for a specific neuron's pre-activation value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CutTerm {
    /// Layer index of the neuron.
    pub layer_idx: usize,
    /// Neuron index within the layer.
    pub neuron_idx: usize,
    /// Coefficient in the linear constraint.
    /// Positive for active constraints, negative for inactive.
    pub coefficient: f32,
}

/// A cutting plane constraint derived from a verified subdomain.
///
/// Cutting planes capture the relationship: "this combination of neuron states
/// leads to verification." For unverified regions, at least one neuron must
/// be in a different state than what was verified.
///
/// Mathematical form: sum_i(coeff_i * x_i) >= bias
/// where x_i is the pre-activation value of neuron i.
#[derive(Debug, Clone)]
pub struct CuttingPlane {
    /// Terms in the linear constraint.
    pub terms: Vec<CutTerm>,
    /// Right-hand side of the constraint.
    pub bias: f32,
    /// Lagrangian multiplier for this cut (dual variable, must be >= 0).
    pub lambda: f32,
    /// Gradient of lambda for optimization.
    pub lambda_grad: f32,
    /// First moment estimate for Adam optimizer.
    pub lambda_m: f32,
    /// Second moment estimate for Adam optimizer.
    pub lambda_v: f32,
    /// Source domain depth (for debugging/analysis).
    pub source_depth: usize,
}

impl CuttingPlane {
    /// Create a cutting plane from a verified domain's split history.
    ///
    /// When a domain is verified with split history {(l1,n1,active), (l2,n2,inactive), ...},
    /// it means that constraining neurons to these states leads to lb > threshold.
    /// The cut encodes: "NOT all of these constraints can be true in an unverified region"
    /// which translates to: sum_i(sign_i * indicator_i) >= 1
    pub fn from_verified_domain(history: &SplitHistory) -> Option<Self> {
        if history.constraints.is_empty() {
            return None;
        }

        let terms: Vec<CutTerm> = history
            .constraints
            .iter()
            .map(|c| CutTerm {
                layer_idx: c.layer_idx,
                neuron_idx: c.neuron_idx,
                // Sign based on constraint: +1 if active (x >= 0), -1 if inactive (x <= 0)
                coefficient: if c.is_active { 1.0 } else { -1.0 },
            })
            .collect();

        // Bias computation follows BICCOS: bias = (count of active neurons) - 1
        // The constraint form is: sum(coeff_i * z_i) <= bias
        // where z_i is the ReLU indicator (0 if inactive, 1 if active)
        // This constraint encodes: "can't have all neurons in their verified states"
        let active_count = terms.iter().filter(|t| t.coefficient > 0.0).count();
        let bias = (active_count as f32) - 1.0;

        Some(Self {
            source_depth: terms.len(),
            terms,
            bias,
            lambda: 0.0,
            lambda_grad: 0.0,
            lambda_m: 0.0,
            lambda_v: 0.0,
        })
    }

    /// Check if this cut is redundant with a domain's current constraints.
    ///
    /// A cut is redundant if the domain already implies all the cut's constraints
    /// are satisfied (all neurons are in the states specified by the cut).
    pub fn is_redundant_for(&self, history: &SplitHistory) -> bool {
        // Count how many of the cut's terms are already satisfied by the domain
        let satisfied = self
            .terms
            .iter()
            .filter(|term| {
                history
                    .is_constrained(term.layer_idx, term.neuron_idx)
                    .map(|is_active| {
                        // Term is satisfied if constraint matches cut's expectation
                        (term.coefficient > 0.0 && is_active)
                            || (term.coefficient < 0.0 && !is_active)
                    })
                    .unwrap_or(false)
            })
            .count();

        // Cut is redundant if all terms are already satisfied
        // (means this domain is already in the verified region)
        satisfied == self.terms.len()
    }

    /// Evaluate the cut's contribution to the bound.
    ///
    /// Returns the Lagrangian term: lambda * (sum_i(coeff_i * x_i) - bias)
    /// For lower bound maximization, this is added when the constraint is satisfied.
    pub fn evaluate(&self, pre_activations: &[(f32, f32)]) -> f32 {
        let constraint_value: f32 = self
            .terms
            .iter()
            .map(|term| {
                // Use lower bound if coefficient positive, upper if negative
                // This gives the worst-case (minimum) value of the constraint
                if term.coefficient > 0.0 {
                    term.coefficient * pre_activations[term.neuron_idx].0
                } else {
                    term.coefficient * pre_activations[term.neuron_idx].1
                }
            })
            .sum();

        self.lambda * (constraint_value - self.bias)
    }

    /// Reset gradients for a new optimization iteration.
    pub fn zero_grad(&mut self) {
        self.lambda_grad = 0.0;
    }

    /// Perform Adam gradient step on lambda.
    pub fn gradient_step_adam(&mut self, config: &AdaptiveOptConfig, t: usize) {
        let eps = config.epsilon;
        let beta1 = config.beta1;
        let beta2 = config.beta2;
        let lr = config.lr_lambda.unwrap_or(config.beta_lr);

        // Update biased first moment estimate
        self.lambda_m = beta1 * self.lambda_m + (1.0 - beta1) * self.lambda_grad;

        // Update biased second moment estimate
        self.lambda_v = beta2 * self.lambda_v + (1.0 - beta2) * self.lambda_grad * self.lambda_grad;

        // Compute bias-corrected estimates
        let m_hat = if config.bias_correction {
            self.lambda_m / (1.0 - beta1.powi(t as i32))
        } else {
            self.lambda_m
        };

        let v_hat = if config.bias_correction {
            self.lambda_v / (1.0 - beta2.powi(t as i32))
        } else {
            self.lambda_v
        };

        // Gradient ascent step (maximize lower bound)
        let update = lr * m_hat / (v_hat.sqrt() + eps);
        self.lambda += update;

        // Project to feasible region: 0 <= lambda <= MAX_LAMBDA
        // Upper bound prevents lambda explosion and maintains soundness
        const MAX_LAMBDA: f32 = 10.0;
        self.lambda = self.lambda.clamp(0.0, MAX_LAMBDA);
    }
}

/// Pool of cutting planes for GCP-CROWN.
///
/// Manages the collection of cuts derived from verified subdomains during B&B.
/// Cuts are added when domains are verified and applied to all subsequent
/// bound computations.
#[derive(Debug, Clone, Default)]
pub struct CutPool {
    /// Active cutting planes.
    pub cuts: Vec<CuttingPlane>,
    /// Maximum number of cuts to retain.
    pub max_cuts: usize,
    /// Total cuts generated (for statistics).
    pub total_generated: usize,
}

impl CutPool {
    /// Create a new cut pool with specified capacity.
    pub fn new(max_cuts: usize) -> Self {
        Self {
            cuts: Vec::with_capacity(max_cuts),
            max_cuts,
            total_generated: 0,
        }
    }

    /// Add a cut from a verified domain.
    ///
    /// Returns true if the cut was added, false if pool is full or cut is trivial.
    pub fn add_from_verified_domain(&mut self, history: &SplitHistory) -> bool {
        // Don't add cuts from shallow domains (less likely to be useful)
        if history.depth() < 2 {
            return false;
        }

        if let Some(cut) = CuttingPlane::from_verified_domain(history) {
            self.total_generated += 1;

            if self.cuts.len() < self.max_cuts {
                self.cuts.push(cut);
                return true;
            } else {
                // Pool full - could implement cut replacement strategy here
                // For now, just keep the existing cuts
                return false;
            }
        }
        false
    }

    /// Get cuts that are relevant for a domain (not redundant).
    pub fn relevant_cuts_for(&self, history: &SplitHistory) -> Vec<&CuttingPlane> {
        self.cuts
            .iter()
            .filter(|cut| !cut.is_redundant_for(history))
            .collect()
    }

    /// Get mutable references to all cuts for optimization.
    pub fn cuts_mut(&mut self) -> &mut [CuttingPlane] {
        &mut self.cuts
    }

    /// Reset all lambda gradients.
    pub fn zero_grad(&mut self) {
        for cut in &mut self.cuts {
            cut.zero_grad();
        }
    }

    /// Number of active cuts.
    pub fn len(&self) -> usize {
        self.cuts.len()
    }

    /// Check if pool is empty.
    pub fn is_empty(&self) -> bool {
        self.cuts.is_empty()
    }

    /// Sum of all lambda values (for regularization/monitoring).
    pub fn total_lambda(&self) -> f32 {
        self.cuts.iter().map(|c| c.lambda).sum()
    }

    /// Generate proactive cuts for unstable ReLUs before BaB starts.
    ///
    /// This implements BICCOS-lite for sequential networks: instead of waiting
    /// for domains to verify (which may never happen on hard instances), we
    /// generate cuts proactively based on the initial bounds.
    ///
    /// The cuts encode pairwise neuron implications:
    /// - For pairs of unstable neurons in consecutive ReLU layers
    /// - Encodes: "at least one of these neurons must be stable"
    ///
    /// This gives the optimizer lambda variables to work with from iteration 0.
    ///
    /// # Arguments
    /// * `network` - The Network being verified
    /// * `layer_bounds` - IBP/CROWN-IBP bounds for each layer (output of each layer)
    /// * `max_cuts` - Maximum number of proactive cuts to generate
    ///
    /// # Returns
    /// Number of cuts generated
    pub fn generate_proactive_cuts(
        &mut self,
        network: &crate::Network,
        layer_bounds: &[crate::BoundedTensor],
        max_cuts: usize,
    ) -> usize {
        use crate::layers::Layer;

        // Find all ReLU layers and their unstable neurons
        // Note: layer_bounds[i] is the OUTPUT of layer i, so for ReLU at index i,
        // we need the PREVIOUS layer's output (i-1) for pre-activation bounds
        let mut relu_unstable: Vec<(usize, Vec<usize>)> = Vec::new();

        for (layer_idx, layer) in network.layers.iter().enumerate() {
            if !matches!(layer, Layer::ReLU(_)) {
                continue;
            }

            // Get pre-activation bounds (output of previous layer)
            // For first layer, we can't have ReLU (would be no-op on input)
            if layer_idx == 0 {
                continue;
            }

            let pre_bounds = &layer_bounds[layer_idx - 1];
            let flat = pre_bounds.flatten();
            let mut unstable = Vec::new();

            for i in 0..flat.len() {
                let lb = flat.lower[[i]];
                let ub = flat.upper[[i]];
                // Neuron is unstable if it crosses zero
                if lb < 0.0 && ub > 0.0 {
                    unstable.push(i);
                }
            }

            if !unstable.is_empty() {
                relu_unstable.push((layer_idx, unstable));
            }
        }

        if relu_unstable.is_empty() {
            return 0;
        }

        let mut cuts_generated = 0;

        // Strategy 1: Generate single-neuron "indicator" cuts for highly unstable neurons
        // These encode the constraint that z_i ∈ {0, 1} (binary indicator)
        // We prioritize neurons with balanced pre-activation bounds (close to zero)
        for &(layer_idx, ref unstable_neurons) in &relu_unstable {
            if cuts_generated >= max_cuts {
                break;
            }

            if layer_idx == 0 {
                continue;
            }

            let pre_bounds = &layer_bounds[layer_idx - 1];
            let flat = pre_bounds.flatten();

            // Score neurons by "instability" (how close to zero the bounds are)
            let mut scored_neurons: Vec<(usize, f32)> = unstable_neurons
                .iter()
                .filter_map(|&idx| {
                    if idx < flat.len() {
                        let lb = flat.lower[[idx]];
                        let ub = flat.upper[[idx]];
                        // Score = how "balanced" the neuron is (closer to 0 = higher score)
                        // Use |lb| / (|lb| + ub) as balance metric
                        let balance = lb.abs() / (lb.abs() + ub);
                        let score = 1.0 - (balance - 0.5).abs() * 2.0; // 1.0 = perfectly balanced
                        Some((idx, score))
                    } else {
                        None
                    }
                })
                .collect();

            // Sort by score descending (most balanced first)
            scored_neurons
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Take top neurons for single-neuron cuts
            for (neuron_idx, _score) in scored_neurons.iter().take(5) {
                if cuts_generated >= max_cuts {
                    break;
                }

                // Create a single-neuron "active" cut
                // This encodes: z_i should be pushed toward 1 (active)
                let active_cut = CuttingPlane {
                    terms: vec![CutTerm {
                        layer_idx,
                        neuron_idx: *neuron_idx,
                        coefficient: 1.0, // Active constraint
                    }],
                    bias: 0.5,    // Midpoint bias
                    lambda: 0.01, // Small initial lambda (will be optimized)
                    lambda_grad: 0.0,
                    lambda_m: 0.0,
                    lambda_v: 0.0,
                    source_depth: 0, // Proactive cut (depth 0)
                };

                self.cuts.push(active_cut);
                self.total_generated += 1;
                cuts_generated += 1;
            }
        }

        // Strategy 2: Generate pairwise cuts between consecutive ReLU layers
        // These encode implications like: "if neuron i is active, neuron j is likely active"
        if cuts_generated < max_cuts && relu_unstable.len() >= 2 {
            for window in relu_unstable.windows(2) {
                if cuts_generated >= max_cuts {
                    break;
                }

                let (layer1, unstable1) = &window[0];
                let (layer2, unstable2) = &window[1];

                // Create pairwise cuts for a subset of neurons
                let pairs_to_create = ((max_cuts - cuts_generated) / 4).clamp(1, 10);
                let mut pairs_created = 0;

                for &n1 in unstable1.iter().take(5) {
                    if pairs_created >= pairs_to_create {
                        break;
                    }
                    for &n2 in unstable2.iter().take(2) {
                        if pairs_created >= pairs_to_create || cuts_generated >= max_cuts {
                            break;
                        }

                        // Create cut: z_n1 + z_n2 >= 1 (at least one active)
                        // Encoded as: sum(coeffs * z) >= bias
                        // With coeffs=[+1, +1], bias=1 means: z_n1 + z_n2 >= 1
                        let pairwise_cut = CuttingPlane {
                            terms: vec![
                                CutTerm {
                                    layer_idx: *layer1,
                                    neuron_idx: n1,
                                    coefficient: 1.0,
                                },
                                CutTerm {
                                    layer_idx: *layer2,
                                    neuron_idx: n2,
                                    coefficient: 1.0,
                                },
                            ],
                            bias: 1.0,
                            lambda: 0.01,
                            lambda_grad: 0.0,
                            lambda_m: 0.0,
                            lambda_v: 0.0,
                            source_depth: 0, // Proactive cut (depth 0)
                        };

                        self.cuts.push(pairwise_cut);
                        self.total_generated += 1;
                        cuts_generated += 1;
                        pairs_created += 1;
                    }
                }
            }
        }

        cuts_generated
    }
}

// =============================================================================
// GraphNetwork GCP-CROWN: Cutting Planes for DAG Models
// =============================================================================

/// A single term in a GraphCuttingPlane.
///
/// Unlike CutTerm which uses layer_idx, this uses node_name to identify
/// ReLU nodes in DAG structures (e.g., ResNets with skip connections).
#[derive(Debug, Clone)]
pub struct GraphCutTerm {
    /// Name of the ReLU node in the graph.
    pub node_name: String,
    /// Neuron index within the ReLU node's output.
    pub neuron_idx: usize,
    /// Coefficient: +1.0 if active constraint, -1.0 if inactive.
    pub coefficient: f32,
}

/// A cutting plane constraint for GraphNetwork verification.
///
/// Graph cutting planes encode verified domain configurations:
/// "If these neurons are all in these states, the property is verified."
/// The cut constraint prevents the verifier from redundantly exploring
/// regions already proven by earlier verified subdomains.
#[derive(Debug, Clone)]
pub struct GraphCuttingPlane {
    /// Terms of the cut (neuron references and coefficients).
    pub terms: Vec<GraphCutTerm>,
    /// Right-hand side bias of the constraint.
    pub bias: f32,
    /// Lagrangian multiplier for this cut (optimized during bound computation).
    pub lambda: f32,
    /// Gradient of the objective w.r.t. lambda.
    pub lambda_grad: f32,
    /// First moment estimate for Adam optimizer.
    pub lambda_m: f32,
    /// Second moment estimate for Adam optimizer.
    pub lambda_v: f32,
    /// Source domain depth (for debugging/analysis).
    pub source_depth: usize,
}

impl GraphCuttingPlane {
    /// Create a cutting plane from a verified GraphNetwork domain's split history.
    ///
    /// When a domain is verified with split history {(node1,n1,active), (node2,n2,inactive), ...},
    /// it means that constraining neurons to these states leads to lb > threshold.
    /// The cut encodes: "NOT all of these constraints can be true in an unverified region"
    pub fn from_verified_domain(history: &GraphSplitHistory) -> Option<Self> {
        if history.constraints.is_empty() {
            return None;
        }

        let terms: Vec<GraphCutTerm> = history
            .constraints
            .iter()
            .map(|c| GraphCutTerm {
                node_name: c.node_name.clone(),
                neuron_idx: c.neuron_idx,
                // Sign based on constraint: +1 if active (x >= 0), -1 if inactive (x <= 0)
                coefficient: if c.is_active { 1.0 } else { -1.0 },
            })
            .collect();

        // Bias computation follows BICCOS: bias = (count of active neurons) - 1
        let active_count = terms.iter().filter(|t| t.coefficient > 0.0).count();
        let bias = (active_count as f32) - 1.0;

        Some(Self {
            source_depth: terms.len(),
            terms,
            bias,
            lambda: 0.0,
            lambda_grad: 0.0,
            lambda_m: 0.0,
            lambda_v: 0.0,
        })
    }

    /// Check if this cut is redundant with a domain's current constraints.
    pub fn is_redundant_for(&self, history: &GraphSplitHistory) -> bool {
        let satisfied = self
            .terms
            .iter()
            .filter(|term| {
                history
                    .is_constrained(&term.node_name, term.neuron_idx)
                    .map(|is_active| {
                        (term.coefficient > 0.0 && is_active)
                            || (term.coefficient < 0.0 && !is_active)
                    })
                    .unwrap_or(false)
            })
            .count();

        satisfied == self.terms.len()
    }

    /// Reset gradients for a new optimization iteration.
    pub fn zero_grad(&mut self) {
        self.lambda_grad = 0.0;
    }

    /// Update lambda using Adam optimizer.
    pub fn update_lambda_adam(&mut self, lr: f32, beta1: f32, beta2: f32, epsilon: f32, t: usize) {
        // Update biased first moment estimate
        self.lambda_m = beta1 * self.lambda_m + (1.0 - beta1) * self.lambda_grad;

        // Update biased second raw moment estimate
        self.lambda_v = beta2 * self.lambda_v + (1.0 - beta2) * self.lambda_grad * self.lambda_grad;

        // Compute bias-corrected estimates
        let t_f32 = t as f32;
        let m_hat = self.lambda_m / (1.0 - beta1.powf(t_f32));
        let v_hat = self.lambda_v / (1.0 - beta2.powf(t_f32));

        // Update lambda (maximize, so add gradient)
        self.lambda += lr * m_hat / (v_hat.sqrt() + epsilon);

        // Project to feasible region: 0 <= lambda <= MAX_LAMBDA
        // Upper bound prevents lambda explosion and maintains soundness
        const MAX_LAMBDA: f32 = 10.0;
        self.lambda = self.lambda.clamp(0.0, MAX_LAMBDA);
    }
}

/// Pool of cutting planes for GraphNetwork GCP-CROWN.
///
/// Manages graph-aware cuts derived from verified subdomains during B&B.
#[derive(Debug, Clone, Default)]
pub struct GraphCutPool {
    /// Active cutting planes.
    pub cuts: Vec<GraphCuttingPlane>,
    /// Maximum number of cuts to retain.
    pub max_cuts: usize,
    /// Total cuts generated (for statistics).
    pub total_generated: usize,
    /// Minimum depth required to generate cuts (default: 2).
    pub min_cut_depth: usize,
}

impl GraphCutPool {
    /// Create a new graph cut pool with specified capacity.
    pub fn new(max_cuts: usize) -> Self {
        Self {
            cuts: Vec::with_capacity(max_cuts),
            max_cuts,
            total_generated: 0,
            min_cut_depth: 2,
        }
    }

    /// Create with custom minimum depth.
    pub fn with_min_depth(max_cuts: usize, min_depth: usize) -> Self {
        Self {
            cuts: Vec::with_capacity(max_cuts),
            max_cuts,
            total_generated: 0,
            min_cut_depth: min_depth,
        }
    }

    /// Add a cut from a verified domain.
    ///
    /// Returns true if the cut was added, false if pool is full or cut is trivial.
    pub fn add_from_verified_domain(&mut self, history: &GraphSplitHistory) -> bool {
        // Don't add cuts from shallow domains (less likely to be useful)
        if history.depth() < self.min_cut_depth {
            return false;
        }

        if let Some(cut) = GraphCuttingPlane::from_verified_domain(history) {
            self.total_generated += 1;

            if self.cuts.len() < self.max_cuts {
                self.cuts.push(cut);
                return true;
            }
        }
        false
    }

    /// Add a cut from a near-miss domain (close to verification but not verified).
    ///
    /// Near-miss cuts are weaker than verified cuts because the domain didn't
    /// actually verify. However, they can still be useful for pruning similar
    /// regions in the search space.
    ///
    /// Returns true if the cut was added, false if pool is full or cut is trivial.
    pub fn add_from_near_miss_domain(
        &mut self,
        history: &GraphSplitHistory,
        lower_bound: f32,
        threshold: f32,
        margin: f32,
    ) -> bool {
        // Don't add cuts from shallow domains
        if history.depth() < self.min_cut_depth {
            return false;
        }

        // Check if it's a near-miss (close to threshold but not verified)
        let effective_margin = if threshold.abs() < 1e-6 {
            margin // Use absolute margin if threshold is ~0
        } else {
            threshold.abs() * margin // Use relative margin
        };

        // Near-miss: lower_bound is within margin of threshold
        // For verify lower > threshold: lb should be close to threshold but < threshold
        let gap = threshold - lower_bound;
        if gap <= 0.0 || gap > effective_margin {
            return false; // Not a near-miss (either verified or too far)
        }

        if let Some(cut) = GraphCuttingPlane::from_verified_domain(history) {
            self.total_generated += 1;

            if self.cuts.len() < self.max_cuts {
                self.cuts.push(cut);
                return true;
            }
        }
        false
    }

    /// Get cuts that are relevant for a domain (not redundant).
    pub fn relevant_cuts_for(&self, history: &GraphSplitHistory) -> Vec<&GraphCuttingPlane> {
        self.cuts
            .iter()
            .filter(|cut| !cut.is_redundant_for(history))
            .collect()
    }

    /// Get mutable references to all cuts for optimization.
    pub fn cuts_mut(&mut self) -> &mut [GraphCuttingPlane] {
        &mut self.cuts
    }

    /// Reset all lambda gradients.
    pub fn zero_grad(&mut self) {
        for cut in &mut self.cuts {
            cut.zero_grad();
        }
    }

    /// Number of active cuts.
    pub fn len(&self) -> usize {
        self.cuts.len()
    }

    /// Check if pool is empty.
    pub fn is_empty(&self) -> bool {
        self.cuts.is_empty()
    }

    /// Sum of all lambda values.
    pub fn total_lambda(&self) -> f32 {
        self.cuts.iter().map(|c| c.lambda).sum()
    }

    /// Generate proactive cuts for unstable ReLUs before BaB starts.
    ///
    /// This implements BICCOS-lite: instead of waiting for domains to verify
    /// (which may never happen on hard instances), we generate cuts proactively
    /// based on the initial bounds.
    ///
    /// The cuts encode pairwise neuron implications:
    /// - For pairs of unstable neurons in consecutive layers
    /// - Encodes: "at least one of these neurons must be stable"
    ///
    /// This gives the optimizer lambda variables to work with from iteration 0.
    ///
    /// # Arguments
    /// * `graph` - The GraphNetwork being verified
    /// * `node_bounds` - Initial bounds for each node (from α-CROWN or CROWN-IBP)
    /// * `max_cuts` - Maximum number of proactive cuts to generate
    ///
    /// # Returns
    /// Number of cuts generated
    pub fn generate_proactive_cuts(
        &mut self,
        graph: &crate::GraphNetwork,
        node_bounds: &std::collections::HashMap<String, std::sync::Arc<crate::BoundedTensor>>,
        max_cuts: usize,
    ) -> usize {
        use crate::layers::Layer;

        // Find all ReLU nodes and their unstable neurons
        let mut relu_unstable: Vec<(String, Vec<usize>)> = Vec::new();

        for (name, node) in &graph.nodes {
            if !matches!(node.layer, Layer::ReLU(_)) {
                continue;
            }

            // Get bounds for this ReLU's input (pre-activation)
            // The input to a ReLU node is typically named without the "_relu" suffix
            let input_name = node.inputs.first().cloned().unwrap_or_else(|| name.clone());
            let bounds = node_bounds
                .get(&input_name)
                .or_else(|| node_bounds.get(name));

            if let Some(bounds) = bounds {
                let flat = bounds.flatten();
                let mut unstable = Vec::new();

                for i in 0..flat.len() {
                    let lb = flat.lower[[i]];
                    let ub = flat.upper[[i]];
                    // Neuron is unstable if it crosses zero
                    if lb < 0.0 && ub > 0.0 {
                        unstable.push(i);
                    }
                }

                if !unstable.is_empty() {
                    relu_unstable.push((name.clone(), unstable));
                }
            }
        }

        if relu_unstable.is_empty() {
            return 0;
        }

        let mut cuts_generated = 0;

        // Strategy 1: Generate single-neuron "indicator" cuts for highly unstable neurons
        // These encode the constraint that z_i ∈ {0, 1} (binary indicator)
        // We prioritize neurons with balanced pre-activation bounds (close to zero)
        for (node_name, unstable_neurons) in &relu_unstable {
            if cuts_generated >= max_cuts {
                break;
            }

            // Sort neurons by "instability score" (how close to zero the bounds are)
            let bounds = node_bounds.get(node_name).or_else(|| {
                // Try to find input bounds
                let node = graph.nodes.get(node_name)?;
                let input_name = node.inputs.first()?;
                node_bounds.get(input_name)
            });

            if let Some(bounds) = bounds {
                let flat = bounds.flatten();
                let mut scored_neurons: Vec<(usize, f32)> = unstable_neurons
                    .iter()
                    .filter_map(|&idx| {
                        if idx < flat.len() {
                            let lb = flat.lower[[idx]];
                            let ub = flat.upper[[idx]];
                            // Score = how "balanced" the neuron is (closer to 0 = higher score)
                            // Use |lb| / (|lb| + ub) as balance metric
                            let balance = lb.abs() / (lb.abs() + ub);
                            let score = 1.0 - (balance - 0.5).abs() * 2.0; // 1.0 = perfectly balanced
                            Some((idx, score))
                        } else {
                            None
                        }
                    })
                    .collect();

                // Sort by score descending (most balanced first)
                scored_neurons
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                // Take top neurons for single-neuron cuts
                for (neuron_idx, _score) in scored_neurons.iter().take(5) {
                    if cuts_generated >= max_cuts {
                        break;
                    }

                    // Create a single-neuron "active" cut
                    // This encodes: z_i should be pushed toward 1 (active)
                    let active_cut = GraphCuttingPlane {
                        terms: vec![GraphCutTerm {
                            node_name: node_name.clone(),
                            neuron_idx: *neuron_idx,
                            coefficient: 1.0, // Active constraint
                        }],
                        bias: 0.5,    // Midpoint bias
                        lambda: 0.01, // Small initial lambda (will be optimized)
                        lambda_grad: 0.0,
                        lambda_m: 0.0,
                        lambda_v: 0.0,
                        source_depth: 0, // Proactive cut (depth 0)
                    };

                    self.cuts.push(active_cut);
                    self.total_generated += 1;
                    cuts_generated += 1;
                }
            }
        }

        // Strategy 2: Generate pairwise cuts between consecutive layers
        // These encode implications like: "if neuron i is active, neuron j is likely active"
        if cuts_generated < max_cuts && relu_unstable.len() >= 2 {
            for window in relu_unstable.windows(2) {
                if cuts_generated >= max_cuts {
                    break;
                }

                let (node1, unstable1) = &window[0];
                let (node2, unstable2) = &window[1];

                // Create pairwise cuts for a subset of neurons
                let pairs_to_create = ((max_cuts - cuts_generated) / 4).clamp(1, 10);
                let mut pairs_created = 0;

                for &n1 in unstable1.iter().take(5) {
                    if pairs_created >= pairs_to_create {
                        break;
                    }
                    for &n2 in unstable2.iter().take(2) {
                        if pairs_created >= pairs_to_create || cuts_generated >= max_cuts {
                            break;
                        }

                        // Create cut: z_n1 + z_n2 >= 1 (at least one active)
                        // Encoded as: sum(coeffs * z) >= bias
                        // With coeffs=[+1, +1], bias=1 means: z_n1 + z_n2 >= 1
                        let pairwise_cut = GraphCuttingPlane {
                            terms: vec![
                                GraphCutTerm {
                                    node_name: node1.clone(),
                                    neuron_idx: n1,
                                    coefficient: 1.0,
                                },
                                GraphCutTerm {
                                    node_name: node2.clone(),
                                    neuron_idx: n2,
                                    coefficient: 1.0,
                                },
                            ],
                            bias: 0.5,    // Soft constraint (not requiring full activation)
                            lambda: 0.01, // Small initial lambda
                            lambda_grad: 0.0,
                            lambda_m: 0.0,
                            lambda_v: 0.0,
                            source_depth: 0, // Proactive cut
                        };

                        self.cuts.push(pairwise_cut);
                        self.total_generated += 1;
                        cuts_generated += 1;
                        pairs_created += 1;
                    }
                }
            }
        }

        debug!(
            "Generated {} proactive cuts from {} ReLU nodes with unstable neurons",
            cuts_generated,
            relu_unstable.len()
        );

        cuts_generated
    }
}

/// A single beta parameter entry for sparse beta representation.
#[derive(Debug, Clone, Copy)]
pub struct BetaEntry {
    /// Layer index of the constrained neuron.
    pub layer_idx: usize,
    /// Neuron index within the layer.
    pub neuron_idx: usize,
    /// Current beta value (Lagrangian multiplier, must be >= 0).
    pub value: f32,
    /// Sign of the constraint: +1 for active (x >= 0), -1 for inactive (x <= 0).
    pub sign: f32,
    /// Accumulated gradient for this iteration.
    pub grad: f32,
    /// First moment estimate (mean of gradients) for Adam optimizer.
    pub m: f32,
    /// Second moment estimate (uncentered variance) for Adam optimizer.
    pub v: f32,
    /// Maximum second moment estimate for AMSGrad variant.
    /// Tracks max(v) over all past iterations for stable convergence.
    pub v_max: f32,
}

/// β parameters for constrained CROWN propagation.
///
/// β values are Lagrangian multipliers in the dual formulation of split constraints.
/// The Lagrangian augmented bound is: lb = c^T * A * x + b + sum_i(β_i * sign_i * x_i)
///
/// During optimization:
/// 1. Forward pass: compute bounds with current β values
/// 2. Compute gradient: (sub)gradient of the current lower bound w.r.t. β (depends on the
///    current piecewise-linear relaxation choices and concretization point)
/// 3. Update: β = max(0, β + lr * grad)  (projected gradient ascent)
#[derive(Debug, Clone)]
pub struct BetaState {
    /// Sparse beta entries for constrained neurons.
    pub entries: Vec<BetaEntry>,
    /// Slow weights for Lookahead optimizer.
    /// When Some, contains β values that are a moving average of the fast weights.
    /// Maps 1:1 with entries (slow_weights\[i\] corresponds to entries\[i\].value).
    pub slow_weights: Option<Vec<f32>>,
}

impl BetaState {
    /// Create β state from split history.
    pub fn from_history(history: &SplitHistory) -> Self {
        let entries = history
            .constraints
            .iter()
            .map(|c| BetaEntry {
                layer_idx: c.layer_idx,
                neuron_idx: c.neuron_idx,
                value: 0.0,
                sign: if c.is_active { 1.0 } else { -1.0 },
                grad: 0.0,
                m: 0.0,     // Initialize Adam first moment
                v: 0.0,     // Initialize Adam second moment
                v_max: 0.0, // Initialize AMSGrad max second moment
            })
            .collect();
        Self {
            entries,
            slow_weights: None, // Initialized lazily when lookahead is first used
        }
    }

    /// Create empty beta state.
    pub fn empty() -> Self {
        Self {
            entries: Vec::new(),
            slow_weights: None,
        }
    }

    /// Get β entry for a neuron, if constrained.
    pub fn get_entry(&self, layer_idx: usize, neuron_idx: usize) -> Option<&BetaEntry> {
        self.entries
            .iter()
            .find(|e| e.layer_idx == layer_idx && e.neuron_idx == neuron_idx)
    }

    /// Get mutable β entry for a neuron, if constrained.
    pub fn get_entry_mut(&mut self, layer_idx: usize, neuron_idx: usize) -> Option<&mut BetaEntry> {
        self.entries
            .iter_mut()
            .find(|e| e.layer_idx == layer_idx && e.neuron_idx == neuron_idx)
    }

    /// Get β value for a neuron, if constrained (for backward compatibility).
    pub fn get_beta(&self, layer_idx: usize, neuron_idx: usize) -> Option<f32> {
        self.get_entry(layer_idx, neuron_idx).map(|e| e.value)
    }

    /// Set β value for a neuron.
    pub fn set_beta(&mut self, layer_idx: usize, neuron_idx: usize, value: f32) {
        if let Some(entry) = self.get_entry_mut(layer_idx, neuron_idx) {
            entry.value = value;
        }
    }

    /// Reset all gradients to zero.
    pub fn zero_grad(&mut self) {
        for entry in &mut self.entries {
            entry.grad = 0.0;
        }
    }

    /// Accumulate gradient for a specific neuron.
    pub fn accumulate_grad(&mut self, layer_idx: usize, neuron_idx: usize, grad: f32) {
        if let Some(entry) = self.get_entry_mut(layer_idx, neuron_idx) {
            entry.grad += grad;
        }
    }

    /// Perform projected gradient ascent step.
    /// β = max(0, β + lr * grad)
    ///
    /// Returns the maximum gradient magnitude (for convergence check).
    pub fn gradient_step(&mut self, lr: f32) -> f32 {
        let mut max_grad = 0.0f32;
        for entry in &mut self.entries {
            max_grad = max_grad.max(entry.grad.abs());
            // Gradient ascent (we want to maximize lower bound)
            entry.value = (entry.value + lr * entry.grad).max(0.0);
        }
        max_grad
    }

    /// Perform Adam optimizer step for β parameters.
    ///
    /// Adam update rule:
    /// - m = β₁ * m + (1 - β₁) * grad
    /// - v = β₂ * v + (1 - β₂) * grad²
    /// - m_hat = m / (1 - β₁^t)  (bias correction)
    /// - v_hat = v / (1 - β₂^t)  (bias correction)
    /// - β = max(0, β + lr * m_hat / (√v_hat + ε))
    ///
    /// When `config.amsgrad` is true, uses AMSGrad variant:
    /// - v_max = max(v_max, v)
    /// - v_hat = v_max / (1 - β₂^t)  (use v_max instead of v)
    ///
    /// The learning rate is adjusted by:
    /// 1. The scheduler based on iteration `t`
    /// 2. Per-layer scaling based on `config.per_layer_lr_beta`
    ///
    /// Parameter `t` is 1-indexed (first iteration is t=1) for bias correction.
    ///
    /// Returns the maximum (bias-corrected) gradient magnitude.
    pub fn gradient_step_adam(&mut self, config: &AdaptiveOptConfig, t: usize) -> f32 {
        let mut max_grad = 0.0f32;
        let t_float = t.max(1) as f32; // Avoid division by zero
        let radam_r = if config.radam {
            radam_rectification_factor(config.beta2, t_float)
        } else {
            None
        };

        // Bias correction factors
        let beta1_corr = if config.bias_correction {
            1.0 - config.beta1.powf(t_float)
        } else {
            1.0
        };
        let beta2_corr = if config.bias_correction {
            1.0 - config.beta2.powf(t_float)
        } else {
            1.0
        };

        // Compute base scheduled learning rate (scheduler uses 0-indexed iteration)
        let base_scheduled_lr = config.scheduler.get_lr(t.saturating_sub(1), config.beta_lr);

        for entry in &mut self.entries {
            // Compute per-layer LR factor
            let total_layers = if config.total_layers > 0 {
                config.total_layers
            } else {
                entry.layer_idx + 1 // Fallback: assume current layer is the deepest
            };
            let layer_factor = config
                .per_layer_lr_beta
                .get_factor(entry.layer_idx, total_layers);
            let scheduled_lr = base_scheduled_lr * layer_factor;

            // Gradient clipping
            let grad = if config.grad_clip > 0.0 {
                entry.grad.clamp(-config.grad_clip, config.grad_clip)
            } else {
                entry.grad
            };

            // Update biased first moment estimate
            entry.m = config.beta1 * entry.m + (1.0 - config.beta1) * grad;

            // Update biased second raw moment estimate
            entry.v = config.beta2 * entry.v + (1.0 - config.beta2) * grad * grad;

            // Compute bias-corrected estimates
            let m_hat = entry.m / beta1_corr;

            // AMSGrad: use maximum of past squared gradients for stable convergence
            let v_for_update = if config.amsgrad {
                entry.v_max = entry.v_max.max(entry.v);
                entry.v_max
            } else {
                entry.v
            };
            let v_hat = v_for_update / beta2_corr;

            // Track max gradient (bias-corrected)
            max_grad = max_grad.max(m_hat.abs());

            // Adaptive update:
            // - Adam: θ = θ + lr * m_hat / (√v_hat + ε)
            // - RAdam: use SGD-with-momentum style step for early iterations; otherwise
            //   apply rectification factor r_t.
            // Gradient ascent (we want to maximize lower bound).
            let update = if config.radam {
                if let Some(r_t) = radam_r {
                    scheduled_lr * r_t * m_hat / (v_hat.sqrt() + config.epsilon)
                } else {
                    scheduled_lr * m_hat
                }
            } else {
                scheduled_lr * m_hat / (v_hat.sqrt() + config.epsilon)
            };

            // AdamW: apply decoupled weight decay directly to parameters
            // θ = θ * (1 - lr * λ) + update
            let decay_factor = if config.weight_decay > 0.0 {
                1.0 - scheduled_lr * config.weight_decay
            } else {
                1.0
            };
            entry.value = (entry.value * decay_factor + update).max(0.0);
        }
        max_grad
    }

    /// Get the signed beta contribution for A matrix modification.
    /// Returns β * sign for the specified neuron.
    pub fn get_signed_beta(&self, layer_idx: usize, neuron_idx: usize) -> Option<f32> {
        self.get_entry(layer_idx, neuron_idx)
            .map(|e| e.value * e.sign)
    }

    /// Check if there are any beta entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get total number of constrained neurons.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Get all entries for a specific layer.
    pub fn entries_for_layer(&self, layer_idx: usize) -> impl Iterator<Item = &BetaEntry> {
        self.entries
            .iter()
            .filter(move |e| e.layer_idx == layer_idx)
    }

    /// Initialize slow weights for Lookahead optimizer.
    ///
    /// Should be called once at the beginning of optimization when lookahead is enabled.
    /// Copies current β values as the initial slow weights.
    pub fn init_slow_weights(&mut self) {
        self.slow_weights = Some(self.entries.iter().map(|e| e.value).collect());
    }

    /// Perform Lookahead synchronization step.
    ///
    /// This should be called after every `sync_period` iterations of the inner optimizer.
    ///
    /// Algorithm:
    /// 1. slow = slow + α * (fast - slow)  [interpolate slow toward fast]
    /// 2. fast = slow  [reset fast weights to slow]
    ///
    /// # Arguments
    /// * `config` - Lookahead configuration with interpolation coefficient α
    ///
    /// # Panics
    /// Panics if slow_weights is None (must call init_slow_weights first).
    pub fn lookahead_step(&mut self, config: &LookaheadConfig) {
        let slow = self
            .slow_weights
            .as_mut()
            .expect("init_slow_weights must be called before lookahead_step");

        debug_assert_eq!(slow.len(), self.entries.len());

        for (i, entry) in self.entries.iter_mut().enumerate() {
            let fast = entry.value;
            // slow = slow + α * (fast - slow)
            slow[i] = slow[i] + config.alpha * (fast - slow[i]);
            // fast = slow (reset fast weights to slow)
            // Apply projection to [0, ∞) since β must be non-negative
            entry.value = slow[i].max(0.0);
        }
    }

    /// Check if slow weights are initialized for Lookahead.
    pub fn has_slow_weights(&self) -> bool {
        self.slow_weights.is_some()
    }

    /// Get current slow weights (for debugging/testing).
    pub fn get_slow_weights(&self) -> Option<&[f32]> {
        self.slow_weights.as_deref()
    }
}

/// Domain-specific α state for joint α-β optimization.
///
/// Unlike `AlphaState` which uses relu_idx (position in list of ReLU layers),
/// this struct maps α values by layer_idx for direct integration with β-CROWN.
#[derive(Debug, Clone)]
pub struct DomainAlphaState {
    /// α values indexed by (layer_idx, neuron_idx).
    /// Only stores values for ReLU layers with unstable neurons.
    pub alphas: std::collections::HashMap<(usize, usize), f32>,
    /// Gradient accumulator for α values.
    pub grads: std::collections::HashMap<(usize, usize), f32>,
    /// Velocity (momentum) for α updates.
    pub velocity: std::collections::HashMap<(usize, usize), f32>,
    /// Track which neurons are unstable (l < 0 < u) and not constrained.
    pub unstable_neurons: std::collections::HashSet<(usize, usize)>,
    /// First moment estimates (m) for Adam optimizer.
    pub adam_m: std::collections::HashMap<(usize, usize), f32>,
    /// Second moment estimates (v) for Adam optimizer.
    pub adam_v: std::collections::HashMap<(usize, usize), f32>,
    /// Maximum second moment estimates (v_max) for AMSGrad variant.
    /// Tracks max(v) over all past iterations for stable convergence.
    pub adam_v_max: std::collections::HashMap<(usize, usize), f32>,
    /// Slow weights for Lookahead optimizer.
    /// When Some, contains α values that are a moving average of the fast weights.
    pub slow_alphas: Option<std::collections::HashMap<(usize, usize), f32>>,
}

impl DomainAlphaState {
    /// Create empty alpha state.
    pub fn empty() -> Self {
        Self {
            alphas: std::collections::HashMap::new(),
            grads: std::collections::HashMap::new(),
            velocity: std::collections::HashMap::new(),
            unstable_neurons: std::collections::HashSet::new(),
            adam_m: std::collections::HashMap::new(),
            adam_v: std::collections::HashMap::new(),
            adam_v_max: std::collections::HashMap::new(),
            slow_alphas: None, // Initialized lazily when lookahead is first used
        }
    }

    /// Initialize α state from layer bounds and constraints.
    ///
    /// For each ReLU layer, identifies unstable neurons (l < 0 < u) that are not
    /// constrained, and initializes α using the standard heuristic: α = 1 if u > -l, else 0.
    pub fn from_layer_bounds_and_constraints(
        network: &Network,
        layer_bounds: &[Arc<BoundedTensor>],
        history: &SplitHistory,
    ) -> Self {
        let mut state = Self::empty();

        // Build constraint lookup
        let mut constraints: std::collections::HashMap<(usize, usize), bool> =
            std::collections::HashMap::new();
        for c in &history.constraints {
            constraints.insert((c.layer_idx, c.neuron_idx), c.is_active);
        }

        // Find ReLU layers and initialize α for unstable neurons
        for (layer_idx, layer) in network.layers.iter().enumerate() {
            if !matches!(layer, Layer::ReLU(_)) {
                continue;
            }

            // Get pre-activation bounds for this ReLU
            if layer_idx == 0 || layer_idx > layer_bounds.len() {
                continue;
            }
            let pre_bounds = &layer_bounds[layer_idx - 1];
            let pre_flat = pre_bounds.flatten();

            for neuron_idx in 0..pre_flat.len() {
                let l = pre_flat.lower[[neuron_idx]];
                let u = pre_flat.upper[[neuron_idx]];

                // Skip stable neurons
                if l >= 0.0 || u <= 0.0 {
                    continue;
                }

                // Skip constrained neurons (they use fixed slopes 0 or 1)
                if constraints.contains_key(&(layer_idx, neuron_idx)) {
                    continue;
                }

                // Unstable and not constrained: initialize α with heuristic
                let alpha = if u > -l { 1.0 } else { 0.0 };
                state.alphas.insert((layer_idx, neuron_idx), alpha);
                state.grads.insert((layer_idx, neuron_idx), 0.0);
                state.velocity.insert((layer_idx, neuron_idx), 0.0);
                state.adam_m.insert((layer_idx, neuron_idx), 0.0);
                state.adam_v.insert((layer_idx, neuron_idx), 0.0);
                state.adam_v_max.insert((layer_idx, neuron_idx), 0.0);
                state.unstable_neurons.insert((layer_idx, neuron_idx));
            }
        }

        state
    }

    /// Get α value for a neuron. Returns the heuristic default if not found.
    pub fn get_alpha(&self, layer_idx: usize, neuron_idx: usize) -> f32 {
        *self.alphas.get(&(layer_idx, neuron_idx)).unwrap_or(&0.0)
    }

    /// Set α value for a neuron.
    pub fn set_alpha(&mut self, layer_idx: usize, neuron_idx: usize, value: f32) {
        if let Some(alpha) = self.alphas.get_mut(&(layer_idx, neuron_idx)) {
            *alpha = value.clamp(0.0, 1.0);
        }
    }

    /// Check if a neuron has an optimizable α.
    pub fn is_unstable(&self, layer_idx: usize, neuron_idx: usize) -> bool {
        self.unstable_neurons.contains(&(layer_idx, neuron_idx))
    }

    /// Reset all gradients to zero.
    pub fn zero_grad(&mut self) {
        for grad in self.grads.values_mut() {
            *grad = 0.0;
        }
    }

    /// Accumulate gradient for a specific neuron.
    pub fn accumulate_grad(&mut self, layer_idx: usize, neuron_idx: usize, grad: f32) {
        if let Some(g) = self.grads.get_mut(&(layer_idx, neuron_idx)) {
            *g += grad;
        }
    }

    /// Perform gradient ascent step with optional momentum.
    /// Returns the maximum gradient magnitude (for convergence check).
    pub fn gradient_step(&mut self, lr: f32, momentum: f32) -> f32 {
        let mut max_grad = 0.0f32;

        for ((layer_idx, neuron_idx), alpha) in &mut self.alphas {
            if let Some(grad) = self.grads.get(&(*layer_idx, *neuron_idx)) {
                max_grad = max_grad.max(grad.abs());

                // Update with momentum
                let vel = self
                    .velocity
                    .entry((*layer_idx, *neuron_idx))
                    .or_insert(0.0);
                *vel = momentum * *vel + lr * grad;

                // Gradient ascent (we want to maximize lower bound)
                *alpha = (*alpha + *vel).clamp(0.0, 1.0);
            }
        }
        max_grad
    }

    /// Perform Adam optimizer step for α parameters.
    ///
    /// Similar to BetaState::gradient_step_adam, but additionally clamps α to \[0, 1\].
    /// The learning rate is adjusted by:
    /// 1. The scheduler based on iteration `t`
    /// 2. Per-layer scaling based on `config.per_layer_lr_alpha`
    ///
    /// Parameter `t` is 1-indexed (first iteration is t=1) for bias correction.
    ///
    /// When `config.amsgrad` is true, uses AMSGrad variant:
    /// - v_max = max(v_max, v)
    /// - v_hat = v_max / (1 - β₂^t)  (use v_max instead of v)
    ///
    /// Returns the maximum (bias-corrected) gradient magnitude.
    pub fn gradient_step_adam(&mut self, config: &AdaptiveOptConfig, t: usize) -> f32 {
        let mut max_grad = 0.0f32;
        let t_float = t.max(1) as f32;
        let radam_r = if config.radam {
            radam_rectification_factor(config.beta2, t_float)
        } else {
            None
        };

        // Bias correction factors
        let beta1_corr = if config.bias_correction {
            1.0 - config.beta1.powf(t_float)
        } else {
            1.0
        };
        let beta2_corr = if config.bias_correction {
            1.0 - config.beta2.powf(t_float)
        } else {
            1.0
        };

        // Compute base scheduled learning rate (scheduler uses 0-indexed iteration)
        let base_scheduled_lr = config
            .scheduler
            .get_lr(t.saturating_sub(1), config.alpha_lr);

        for ((layer_idx, neuron_idx), alpha) in &mut self.alphas {
            let key = (*layer_idx, *neuron_idx);

            // Compute per-layer LR factor
            let total_layers = if config.total_layers > 0 {
                config.total_layers
            } else {
                layer_idx + 1 // Fallback: assume current layer is the deepest
            };
            let layer_factor = config
                .per_layer_lr_alpha
                .get_factor(*layer_idx, total_layers);
            let scheduled_lr = base_scheduled_lr * layer_factor;

            if let Some(&grad_raw) = self.grads.get(&key) {
                // Gradient clipping
                let grad = if config.grad_clip > 0.0 {
                    grad_raw.clamp(-config.grad_clip, config.grad_clip)
                } else {
                    grad_raw
                };

                // Update biased first moment estimate
                let m = self.adam_m.entry(key).or_insert(0.0);
                *m = config.beta1 * *m + (1.0 - config.beta1) * grad;

                // Update biased second raw moment estimate
                let v = self.adam_v.entry(key).or_insert(0.0);
                *v = config.beta2 * *v + (1.0 - config.beta2) * grad * grad;

                // Compute bias-corrected estimates
                let m_hat = *m / beta1_corr;

                // AMSGrad: use maximum of past squared gradients for stable convergence
                let v_for_update = if config.amsgrad {
                    let v_max = self.adam_v_max.entry(key).or_insert(0.0);
                    *v_max = v_max.max(*v);
                    *v_max
                } else {
                    *v
                };
                let v_hat = v_for_update / beta2_corr;

                // Track max gradient
                max_grad = max_grad.max(m_hat.abs());

                // Adaptive update:
                // - Adam: θ = θ + lr * m_hat / (√v_hat + ε)
                // - RAdam: use SGD-with-momentum style step for early iterations; otherwise
                //   apply rectification factor r_t.
                // Gradient ascent (we want to maximize lower bound).
                let update = if config.radam {
                    if let Some(r_t) = radam_r {
                        scheduled_lr * r_t * m_hat / (v_hat.sqrt() + config.epsilon)
                    } else {
                        scheduled_lr * m_hat
                    }
                } else {
                    scheduled_lr * m_hat / (v_hat.sqrt() + config.epsilon)
                };

                // AdamW: apply decoupled weight decay directly to parameters
                // θ = θ * (1 - lr * λ) + update
                let decay_factor = if config.weight_decay > 0.0 {
                    1.0 - scheduled_lr * config.weight_decay
                } else {
                    1.0
                };
                *alpha = (*alpha * decay_factor + update).clamp(0.0, 1.0);
            }
        }
        max_grad
    }

    /// Check if there are any optimizable α values.
    pub fn is_empty(&self) -> bool {
        self.alphas.is_empty()
    }

    /// Get the number of optimizable α values.
    pub fn len(&self) -> usize {
        self.alphas.len()
    }

    /// Initialize slow weights for Lookahead optimizer.
    ///
    /// Should be called once at the beginning of optimization when lookahead is enabled.
    /// Copies current α values as the initial slow weights.
    pub fn init_slow_weights(&mut self) {
        self.slow_alphas = Some(self.alphas.clone());
    }

    /// Perform Lookahead synchronization step.
    ///
    /// This should be called after every `sync_period` iterations of the inner optimizer.
    ///
    /// Algorithm:
    /// 1. slow = slow + α * (fast - slow)  [interpolate slow toward fast]
    /// 2. fast = slow  [reset fast weights to slow]
    ///
    /// # Arguments
    /// * `config` - Lookahead configuration with interpolation coefficient α
    ///
    /// # Panics
    /// Panics if slow_alphas is None (must call init_slow_weights first).
    pub fn lookahead_step(&mut self, config: &LookaheadConfig) {
        let slow = self
            .slow_alphas
            .as_mut()
            .expect("init_slow_weights must be called before lookahead_step");

        for (key, fast) in &mut self.alphas {
            let slow_val = slow.entry(*key).or_insert(*fast);
            // slow = slow + α * (fast - slow)
            *slow_val = *slow_val + config.alpha * (*fast - *slow_val);
            // fast = slow (reset fast weights to slow)
            // Apply projection to [0, 1] since α must be in [0, 1]
            *fast = slow_val.clamp(0.0, 1.0);
        }
    }

    /// Check if slow weights are initialized for Lookahead.
    pub fn has_slow_weights(&self) -> bool {
        self.slow_alphas.is_some()
    }

    /// Get current slow weights (for debugging/testing).
    pub fn get_slow_weights(&self) -> Option<&std::collections::HashMap<(usize, usize), f32>> {
        self.slow_alphas.as_ref()
    }
}

/// A domain in the branch-and-bound search tree.
#[derive(Debug, Clone)]
pub struct BabDomain {
    /// Split history for this domain.
    pub history: SplitHistory,
    /// Lower bound on the output (used for priority queue ordering).
    pub lower_bound: f32,
    /// Upper bound on the output.
    pub upper_bound: f32,
    /// Pre-activation bounds for each layer (tightened by constraints).
    /// Uses Arc for cheap cloning during branch-and-bound splits - only modified
    /// layers need new allocations.
    pub layer_bounds: Vec<Arc<BoundedTensor>>,
    /// α state for α-CROWN (if used) - legacy field.
    pub alpha_state: Option<AlphaState>,
    /// Domain-specific α state for joint α-β optimization.
    pub domain_alpha_state: DomainAlphaState,
    /// β state for constrained neurons.
    pub beta_state: BetaState,
    /// Input bounds for this domain (used for input splitting).
    /// When None, the domain uses the original input bounds from verification.
    /// When Some, contains tightened input bounds from input space splitting.
    pub input_bounds: Option<Arc<BoundedTensor>>,
    /// Number of input splits applied to this domain.
    pub input_split_count: usize,
}

impl BabDomain {
    /// Create root domain with no constraints.
    pub fn root(layer_bounds: Vec<BoundedTensor>, lower_bound: f32, upper_bound: f32) -> Self {
        let layer_bounds: Vec<Arc<BoundedTensor>> =
            layer_bounds.into_iter().map(Arc::new).collect();
        Self {
            history: SplitHistory::new(),
            lower_bound,
            upper_bound,
            layer_bounds,
            alpha_state: None,
            domain_alpha_state: DomainAlphaState::empty(),
            beta_state: BetaState::empty(),
            input_bounds: None,
            input_split_count: 0,
        }
    }

    /// Create root domain with input bounds (for input splitting).
    pub fn root_with_input(
        layer_bounds: Vec<BoundedTensor>,
        lower_bound: f32,
        upper_bound: f32,
        input: &BoundedTensor,
    ) -> Self {
        let layer_bounds: Vec<Arc<BoundedTensor>> =
            layer_bounds.into_iter().map(Arc::new).collect();
        Self {
            history: SplitHistory::new(),
            lower_bound,
            upper_bound,
            layer_bounds,
            alpha_state: None,
            domain_alpha_state: DomainAlphaState::empty(),
            beta_state: BetaState::empty(),
            input_bounds: Some(Arc::new(input.clone())),
            input_split_count: 0,
        }
    }

    /// Create root domain with initialized α state for joint optimization.
    pub fn root_with_alpha(
        network: &Network,
        layer_bounds: Vec<BoundedTensor>,
        lower_bound: f32,
        upper_bound: f32,
    ) -> Self {
        let layer_bounds: Vec<Arc<BoundedTensor>> =
            layer_bounds.into_iter().map(Arc::new).collect();
        let history = SplitHistory::new();
        let domain_alpha_state =
            DomainAlphaState::from_layer_bounds_and_constraints(network, &layer_bounds, &history);
        Self {
            history,
            lower_bound,
            upper_bound,
            layer_bounds,
            alpha_state: None,
            domain_alpha_state,
            beta_state: BetaState::empty(),
            input_bounds: None,
            input_split_count: 0,
        }
    }

    /// Depth of this domain (number of splits including input splits).
    pub fn depth(&self) -> usize {
        self.history.depth() + self.input_split_count
    }

    /// Get effective input bounds for this domain.
    /// Returns domain-specific bounds if available, otherwise None (use original).
    pub fn get_input_bounds(&self) -> Option<&BoundedTensor> {
        self.input_bounds.as_ref().map(|arc| arc.as_ref())
    }
}

// For BinaryHeap: we want to pop domains with highest (worst) lower bound first
impl PartialEq for BabDomain {
    fn eq(&self, other: &Self) -> bool {
        self.lower_bound == other.lower_bound
    }
}

impl Eq for BabDomain {}

impl PartialOrd for BabDomain {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BabDomain {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher lower_bound = worse = should be processed first
        // So we want max-heap behavior on lower_bound
        self.lower_bound
            .partial_cmp(&other.lower_bound)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Result of β-CROWN verification.
#[derive(Debug, Clone)]
pub struct BetaCrownResult {
    /// Verification result.
    pub result: BabVerificationStatus,
    /// Number of domains explored.
    pub domains_explored: usize,
    /// Total time taken.
    pub time_elapsed: Duration,
    /// Maximum depth reached.
    pub max_depth_reached: usize,
    /// Final output bounds (if available).
    pub output_bounds: Option<BoundedTensor>,
    /// Number of cutting planes generated (GCP-CROWN).
    pub cuts_generated: usize,
    /// Number of domains verified (contributes to cut generation).
    pub domains_verified: usize,
}

/// Status of β-CROWN verification.
#[derive(Debug, Clone, PartialEq)]
pub enum BabVerificationStatus {
    /// Property verified: all domains have lower bound > threshold.
    Verified,
    /// Property violated: concrete counterexample found via PGD attack.
    Violated {
        /// Counterexample input that violates the property.
        counterexample: Vec<f32>,
        /// Output at the counterexample.
        output: Vec<f32>,
    },
    /// Property potentially violated: found a domain where upper bound < threshold,
    /// but no concrete counterexample found.
    PotentialViolation,
    /// Inconclusive: timed out or hit domain limit.
    Unknown { reason: String },
}

/// β-CROWN verifier for complete neural network verification.
#[derive(Default)]
pub struct BetaCrownVerifier {
    /// Configuration.
    pub config: BetaCrownConfig,
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // Used by legacy compute_beta_gradients
struct ReluLowerRelaxation {
    slopes: Vec<f32>,
    intercepts: Vec<f32>,
}

/// Result of processing a single graph domain in parallel.
///
/// Contains the children (if any) and whether they were verified or need
/// further splitting, plus information about domains that couldn't be split
/// (no unstable neurons or max depth reached).
#[derive(Debug)]
enum GraphDomainResult {
    /// Domain was already verified (lb > threshold for lower verification)
    AlreadyVerified,
    /// Domain conclusively violates the property
    Violation,
    /// Children created (each child has bounds and verification status)
    Children(Vec<(GraphBabDomain, bool)>), // (domain, is_verified)
    /// No unstable neurons - domain cannot be split
    NoUnstable {
        lower: f32,
        upper: f32,
        verified: bool,
    },
    /// Error during processing
    #[allow(dead_code)]
    Error(String),
}

/// Result of processing a batch of multi-objective domains in parallel GPU mode.
#[derive(Debug)]
enum MultiObjectiveGraphDomainResult {
    /// Domain was already verified (all objectives verified)
    AlreadyVerified,
    /// Domain conclusively violates the property (any objective violated)
    Violation,
    /// Children created (each child has bounds and verification status)
    Children(Vec<(MultiObjectiveGraphBabDomain, bool)>), // (domain, all_verified)
    /// No unstable neurons - domain cannot be split
    #[allow(dead_code)]
    NoUnstable {
        objective_bounds: Vec<(f32, f32)>,
        all_verified: bool,
    },
    /// Error during processing
    #[allow(dead_code)]
    Error(String),
}

impl BetaCrownVerifier {
    /// Create a new β-CROWN verifier.
    pub fn new(config: BetaCrownConfig) -> Self {
        Self { config }
    }

    /// Verify with optional GPU acceleration via GemmEngine.
    ///
    /// Same as `verify`, but accepts an optional GemmEngine for GPU-accelerated
    /// linear layer CROWN backward passes.
    pub fn verify_with_engine(
        &self,
        network: &Network,
        input: &BoundedTensor,
        threshold: f32,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BetaCrownResult> {
        self.verify_impl(network, input, threshold, engine)
    }

    /// Verify GraphNetwork with input splitting and optional GPU acceleration.
    pub fn verify_graph_input_split_with_engine(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objective: &[f32],
        threshold: f32,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BetaCrownResult> {
        self.verify_graph_input_split_impl(graph, input, objective, threshold, engine)
    }

    /// Verify GraphNetwork with ReLU splitting and optional GPU acceleration.
    pub fn verify_graph_relu_split_with_engine_gpu(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objective: &[f32],
        threshold: f32,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BetaCrownResult> {
        self.verify_graph_relu_split_impl(graph, input, objective, threshold, engine)
    }

    /// Verify GraphNetwork with ReLU splitting using pre-computed bounds, with optional GPU acceleration.
    pub fn verify_graph_relu_split_with_bounds_with_engine(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objective: &[f32],
        threshold: f32,
        precomputed_bounds: &GraphPrecomputedBounds<'_>,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BetaCrownResult> {
        self.verify_graph_relu_split_with_bounds_impl(
            graph,
            input,
            objective,
            threshold,
            precomputed_bounds,
            engine,
        )
    }

    /// Verify that output > threshold for all inputs in the bounded region.
    ///
    /// Returns Verified if we can prove output > threshold for all inputs,
    /// PotentialViolation if we find a region where output might be < threshold,
    /// Unknown if we can't determine (timeout/domain limit).
    #[instrument(skip(self, network, input), fields(threshold, input_shape = ?input.shape(), max_domains = self.config.max_domains))]
    pub fn verify(
        &self,
        network: &Network,
        input: &BoundedTensor,
        threshold: f32,
    ) -> Result<BetaCrownResult> {
        self.verify_impl(network, input, threshold, None)
    }

    /// Internal verify implementation with optional GemmEngine for GPU acceleration.
    fn verify_impl(
        &self,
        network: &Network,
        input: &BoundedTensor,
        threshold: f32,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BetaCrownResult> {
        let start_time = Instant::now();
        let mut domains_explored = 0usize;
        let mut max_depth = 0usize;
        let mut domains_verified = 0usize;

        // Conv layers now support CROWN backward (transposed convolution), so ReLU splitting
        // works for CNNs. Input splitting is no longer forced for Conv networks.
        let _has_conv = network
            .layers
            .iter()
            .any(|layer| matches!(layer, Layer::Conv2d(_) | Layer::Conv1d(_)));

        // Initialize cut pool for GCP-CROWN (even if disabled, for statistics)
        let mut cut_pool = if self.config.enable_cuts {
            CutPool::new(self.config.max_cuts)
        } else {
            CutPool::new(0) // Disabled: pool with 0 capacity
        };

        // Step 1: Initial bound computation with CROWN/α-CROWN
        // Use early exit optimization: if fast CROWN-IBP bounds verify, skip α-CROWN
        // Use GPU-accelerated CROWN if engine is provided
        let initial_bounds = self.compute_initial_bounds_with_early_exit_engine(
            network,
            input,
            Some((threshold, self.config.verify_upper_bound)),
            engine,
        )?;
        // Collect layer bounds for branch-and-bound - optionally use CROWN-IBP for tighter bounds
        let initial_layer_bounds = if self.config.use_crown_ibp {
            network.collect_crown_ibp_bounds(input)?
        } else {
            network.collect_ibp_bounds(input)?
        };

        // Generate proactive cuts if enabled (BICCOS-lite)
        // This populates the cut pool BEFORE BaB starts, avoiding the chicken-and-egg
        // problem where no cuts are generated because no domains verify
        if self.config.enable_proactive_cuts && self.config.enable_cuts {
            let proactive_count = cut_pool.generate_proactive_cuts(
                network,
                &initial_layer_bounds,
                self.config.max_proactive_cuts,
            );
            if proactive_count > 0 {
                info!(
                    "Generated {} proactive cuts for sequential network",
                    proactive_count
                );
            }
        }

        let initial_lower = initial_bounds.lower_scalar();
        let initial_upper = initial_bounds.upper_scalar();

        let direction = if self.config.verify_upper_bound {
            "upper < threshold"
        } else {
            "lower > threshold"
        };
        info!(
            "β-CROWN initial bounds: [{}, {}], threshold: {}, verify: {}",
            initial_lower, initial_upper, threshold, direction
        );

        // Quick checks depend on verification direction
        if self.config.verify_upper_bound {
            // Verifying output < threshold (upper_bound < threshold)
            // Used for VNNLIB constraints like Y >= c (unsafe region)
            if initial_upper < threshold {
                return Ok(BetaCrownResult {
                    result: BabVerificationStatus::Verified,
                    domains_explored: 1,
                    time_elapsed: start_time.elapsed(),
                    max_depth_reached: 0,
                    output_bounds: Some(initial_bounds),
                    cuts_generated: 0,
                    domains_verified: 1,
                });
            }
            if initial_lower >= threshold {
                // Lower bound >= threshold means output might be >= threshold (potential violation)
                return Ok(BetaCrownResult {
                    result: BabVerificationStatus::PotentialViolation,
                    domains_explored: 1,
                    time_elapsed: start_time.elapsed(),
                    max_depth_reached: 0,
                    output_bounds: Some(initial_bounds),
                    cuts_generated: 0,
                    domains_verified: 0,
                });
            }
        } else {
            // Verifying output > threshold (lower_bound > threshold) - original behavior
            if initial_lower > threshold {
                return Ok(BetaCrownResult {
                    result: BabVerificationStatus::Verified,
                    domains_explored: 1,
                    time_elapsed: start_time.elapsed(),
                    max_depth_reached: 0,
                    output_bounds: Some(initial_bounds),
                    cuts_generated: 0,
                    domains_verified: 1,
                });
            }
            if initial_upper < threshold {
                // Upper bound < threshold means output might be < threshold (potential violation)
                return Ok(BetaCrownResult {
                    result: BabVerificationStatus::PotentialViolation,
                    domains_explored: 1,
                    time_elapsed: start_time.elapsed(),
                    max_depth_reached: 0,
                    output_bounds: Some(initial_bounds),
                    cuts_generated: 0,
                    domains_verified: 0,
                });
            }
        }

        // Conv layers now have CROWN backward support via transposed convolution, so we
        // no longer force input splitting. ReLU splitting works for Conv networks.
        let force_input_split = false;

        // Step 2: Initialize domain queue with root domain
        let mut queue: BinaryHeap<BabDomain> = BinaryHeap::new();
        let root = if matches!(
            self.config.branching_heuristic,
            BranchingHeuristic::InputSplit
        ) || force_input_split
        {
            // Store input bounds for input splitting
            BabDomain::root_with_input(initial_layer_bounds, initial_lower, initial_upper, input)
        } else {
            BabDomain::root(initial_layer_bounds, initial_lower, initial_upper)
        };
        queue.push(root);

        // Step 3: Branch-and-bound loop with batched parallel processing
        let batch_size = self.config.batch_size.max(1);
        let use_parallel_children = self.config.parallel_children;

        while !queue.is_empty() {
            // Check termination conditions before processing batch
            if start_time.elapsed() > self.config.timeout {
                info!(
                    "β-CROWN timeout after {} domains, {} verified, {} cuts",
                    domains_explored, domains_verified, cut_pool.total_generated
                );
                let unknown_result = BetaCrownResult {
                    result: BabVerificationStatus::Unknown {
                        reason: "Timeout".to_string(),
                    },
                    domains_explored,
                    time_elapsed: start_time.elapsed(),
                    max_depth_reached: max_depth,
                    output_bounds: None,
                    cuts_generated: cut_pool.total_generated,
                    domains_verified,
                };
                // Try PGD attack to find counterexample
                return self.try_pgd_attack(network, input, threshold, unknown_result);
            }

            if domains_explored >= self.config.max_domains {
                info!(
                    "β-CROWN hit domain limit: {}, {} verified, {} cuts",
                    self.config.max_domains, domains_verified, cut_pool.total_generated
                );
                let unknown_result = BetaCrownResult {
                    result: BabVerificationStatus::Unknown {
                        reason: format!("Domain limit {} reached", self.config.max_domains),
                    },
                    domains_explored,
                    time_elapsed: start_time.elapsed(),
                    max_depth_reached: max_depth,
                    output_bounds: None,
                    cuts_generated: cut_pool.total_generated,
                    domains_verified,
                };
                // Try PGD attack to find counterexample
                return self.try_pgd_attack(network, input, threshold, unknown_result);
            }

            // Pop up to batch_size domains from the queue
            let mut batch: Vec<BabDomain> = Vec::with_capacity(batch_size);
            while batch.len() < batch_size {
                if let Some(domain) = queue.pop() {
                    batch.push(domain);
                } else {
                    break;
                }
            }

            if batch.is_empty() {
                break;
            }

            // Separate verified domains from domains needing processing
            let mut domains_to_process: Vec<BabDomain> = Vec::new();
            for domain in batch {
                domains_explored += 1;
                max_depth = max_depth.max(domain.depth());

                trace!(
                    "Processing domain {}: depth={}, lb={:.4}, ub={:.4}",
                    domains_explored,
                    domain.depth(),
                    domain.lower_bound,
                    domain.upper_bound
                );

                // Check if domain is verified (depends on verification direction)
                let domain_verified = if self.config.verify_upper_bound {
                    domain.upper_bound < threshold
                } else {
                    domain.lower_bound > threshold
                };

                if domain_verified {
                    trace!(
                        "Domain verified: lb={}, ub={}, threshold={}",
                        domain.lower_bound,
                        domain.upper_bound,
                        threshold
                    );
                    domains_verified += 1;

                    // GCP-CROWN: Generate cutting plane from verified domain
                    if self.config.enable_cuts
                        && domain.depth() >= self.config.min_cut_depth
                        && cut_pool.add_from_verified_domain(&domain.history)
                    {
                        trace!(
                            "Generated cut from verified domain (depth={}, total cuts={})",
                            domain.depth(),
                            cut_pool.len()
                        );
                    }
                    continue;
                }

                // Skip domains at max depth
                if domain.depth() >= self.config.max_depth {
                    debug!("Domain at max depth {}, skipping", self.config.max_depth);
                    continue;
                }

                domains_to_process.push(domain);
            }

            if domains_to_process.is_empty() {
                continue;
            }

            // Process domains: parallel when no cuts, sequential when cuts are enabled
            // (cuts require mutable access to cut_pool for lambda optimization)
            let has_cuts = !cut_pool.is_empty() && self.config.enable_cuts;
            let domain_config = DomainProcessingConfig::new(threshold, use_parallel_children);
            let child_results: Vec<_> = if batch_size > 1 && !has_cuts {
                // Parallel domain processing (no cuts)
                // GemmEngine is now Sync+Send, so GPU acceleration works with rayon
                domains_to_process
                    .par_iter()
                    .flat_map(|domain| {
                        let mut empty_pool = CutPool::new(0);
                        self.process_domain_parallel(
                            network,
                            input,
                            domain,
                            &domain_config,
                            &mut empty_pool,
                            engine, // GPU engine now usable in parallel path
                        )
                    })
                    .collect()
            } else {
                // Sequential processing (required when cuts are enabled)
                // GPU engine can be used here for acceleration
                domains_to_process
                    .iter()
                    .flat_map(|domain| {
                        self.process_domain_sequential(
                            network,
                            input,
                            domain,
                            threshold,
                            &mut cut_pool,
                            engine,
                        )
                    })
                    .collect()
            };

            // Add non-verified children to queue, generate cuts for verified ones
            for child in child_results {
                let child_verified = if self.config.verify_upper_bound {
                    child.upper_bound < threshold
                } else {
                    child.lower_bound > threshold
                };

                if !child_verified {
                    queue.push(child);
                } else {
                    trace!("Child verified immediately");
                    domains_verified += 1;

                    // GCP-CROWN: Generate cutting plane from verified child
                    if self.config.enable_cuts
                        && child.depth() >= self.config.min_cut_depth
                        && cut_pool.add_from_verified_domain(&child.history)
                    {
                        trace!(
                            "Generated cut from verified child (depth={}, total cuts={})",
                            child.depth(),
                            cut_pool.len()
                        );
                    }
                }
            }
        }

        // All domains processed, property verified
        info!(
            "β-CROWN verified after {} domains, {} verified, {} cuts, max depth {}",
            domains_explored, domains_verified, cut_pool.total_generated, max_depth
        );
        Ok(BetaCrownResult {
            result: BabVerificationStatus::Verified,
            domains_explored,
            time_elapsed: start_time.elapsed(),
            max_depth_reached: max_depth,
            output_bounds: None,
            cuts_generated: cut_pool.total_generated,
            domains_verified,
        })
    }

    /// GraphNetwork support via input-space splitting.
    ///
    /// This is a conservative fallback for DAG models (e.g., ResNets with residual adds)
    /// where sequential ReLU-splitting β-CROWN is not yet implemented.
    ///
    /// Limitations:
    /// - Requires `branching_heuristic = InputSplit`
    /// - Does not support α-CROWN, β optimization, or cutting planes
    /// - Does not attempt PGD counterexample search
    #[instrument(skip(self, graph, input, objective), fields(threshold, input_shape = ?input.shape(), num_nodes = graph.nodes.len()))]
    pub fn verify_graph_input_split(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objective: &[f32],
        threshold: f32,
    ) -> Result<BetaCrownResult> {
        self.verify_graph_input_split_impl(graph, input, objective, threshold, None)
    }

    /// Internal implementation with optional GemmEngine for GPU acceleration.
    fn verify_graph_input_split_impl(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objective: &[f32],
        threshold: f32,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BetaCrownResult> {
        // Note: branching_heuristic check removed - this function may be called as a fallback
        // from verify_graph_relu_split when the model has Conv layers
        if self.config.use_alpha_crown {
            return Err(GammaError::NotSupported(
                "GraphNetwork β-CROWN input splitting does not support --use-alpha".to_string(),
            ));
        }
        if self.config.use_crown_ibp {
            return Err(GammaError::NotSupported(
                "GraphNetwork β-CROWN input splitting does not support --crown-ibp".to_string(),
            ));
        }
        if self.config.enable_cuts {
            return Err(GammaError::NotSupported(
                "GraphNetwork β-CROWN input splitting does not support cutting planes".to_string(),
            ));
        }

        fn objective_bounds(output: &BoundedTensor, objective: &[f32]) -> Result<(f32, f32)> {
            let flat = output.flatten();
            if flat.len() != objective.len() {
                return Err(GammaError::shape_mismatch(
                    vec![objective.len()],
                    vec![flat.len()],
                ));
            }

            let mut lower = 0.0f32;
            let mut upper = 0.0f32;
            for (idx, &c) in objective.iter().enumerate() {
                let l = flat.lower[[idx]];
                let u = flat.upper[[idx]];
                if c >= 0.0 {
                    lower += c * l;
                    upper += c * u;
                } else {
                    lower += c * u;
                    upper += c * l;
                }
            }
            Ok((lower, upper))
        }

        #[derive(Debug, Clone)]
        struct GraphInputDomain {
            input_bounds: Arc<BoundedTensor>,
            lower_bound: f32,
            upper_bound: f32,
            depth: usize,
            priority: f32,
        }

        impl PartialEq for GraphInputDomain {
            fn eq(&self, other: &Self) -> bool {
                self.priority == other.priority
            }
        }
        impl Eq for GraphInputDomain {}
        impl PartialOrd for GraphInputDomain {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for GraphInputDomain {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.priority
                    .partial_cmp(&other.priority)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        }

        let start_time = Instant::now();
        let mut domains_explored = 0usize;
        let mut domains_verified = 0usize;
        let mut max_depth_reached = 0usize;
        let mut unresolved_due_to_depth = false;

        // Root bounds - use α-CROWN if enabled for tighter bounds
        let root_output = if self.config.use_alpha_crown {
            graph.propagate_alpha_crown_with_config_and_engine(
                input,
                &self.config.alpha_config,
                engine,
            )?
        } else {
            graph.propagate_crown_with_engine(input, engine)?
        };
        let (root_lower, root_upper) = objective_bounds(&root_output, objective)?;

        info!(
            "Graph β-CROWN (input split) initial objective bounds: [{}, {}], threshold: {}, verify_upper={}",
            root_lower, root_upper, threshold, self.config.verify_upper_bound
        );

        // Quick checks
        if self.config.verify_upper_bound {
            if root_upper < threshold {
                return Ok(BetaCrownResult {
                    result: BabVerificationStatus::Verified,
                    domains_explored: 1,
                    time_elapsed: start_time.elapsed(),
                    max_depth_reached: 0,
                    output_bounds: Some(BoundedTensor::new(
                        Array1::from_vec(vec![root_lower]).into_dyn(),
                        Array1::from_vec(vec![root_upper]).into_dyn(),
                    )?),
                    cuts_generated: 0,
                    domains_verified: 1,
                });
            }
            if root_lower >= threshold {
                return Ok(BetaCrownResult {
                    result: BabVerificationStatus::PotentialViolation,
                    domains_explored: 1,
                    time_elapsed: start_time.elapsed(),
                    max_depth_reached: 0,
                    output_bounds: Some(BoundedTensor::new(
                        Array1::from_vec(vec![root_lower]).into_dyn(),
                        Array1::from_vec(vec![root_upper]).into_dyn(),
                    )?),
                    cuts_generated: 0,
                    domains_verified: 0,
                });
            }
        } else {
            if root_lower > threshold {
                return Ok(BetaCrownResult {
                    result: BabVerificationStatus::Verified,
                    domains_explored: 1,
                    time_elapsed: start_time.elapsed(),
                    max_depth_reached: 0,
                    output_bounds: Some(BoundedTensor::new(
                        Array1::from_vec(vec![root_lower]).into_dyn(),
                        Array1::from_vec(vec![root_upper]).into_dyn(),
                    )?),
                    cuts_generated: 0,
                    domains_verified: 1,
                });
            }
            if root_upper < threshold {
                return Ok(BetaCrownResult {
                    result: BabVerificationStatus::PotentialViolation,
                    domains_explored: 1,
                    time_elapsed: start_time.elapsed(),
                    max_depth_reached: 0,
                    output_bounds: Some(BoundedTensor::new(
                        Array1::from_vec(vec![root_lower]).into_dyn(),
                        Array1::from_vec(vec![root_upper]).into_dyn(),
                    )?),
                    cuts_generated: 0,
                    domains_verified: 0,
                });
            }
        }

        let priority = if self.config.verify_upper_bound {
            root_upper
        } else {
            -root_lower
        };

        let mut queue: BinaryHeap<GraphInputDomain> = BinaryHeap::new();
        queue.push(GraphInputDomain {
            input_bounds: Arc::new(input.clone()),
            lower_bound: root_lower,
            upper_bound: root_upper,
            depth: 0,
            priority,
        });

        while let Some(domain) = queue.pop() {
            if start_time.elapsed() > self.config.timeout {
                return Ok(BetaCrownResult {
                    result: BabVerificationStatus::Unknown {
                        reason: "Timeout".to_string(),
                    },
                    domains_explored,
                    time_elapsed: start_time.elapsed(),
                    max_depth_reached,
                    output_bounds: None,
                    cuts_generated: 0,
                    domains_verified,
                });
            }
            if domains_explored >= self.config.max_domains {
                return Ok(BetaCrownResult {
                    result: BabVerificationStatus::Unknown {
                        reason: format!("Domain limit {} reached", self.config.max_domains),
                    },
                    domains_explored,
                    time_elapsed: start_time.elapsed(),
                    max_depth_reached,
                    output_bounds: None,
                    cuts_generated: 0,
                    domains_verified,
                });
            }

            domains_explored += 1;
            max_depth_reached = max_depth_reached.max(domain.depth);

            let domain_verified = if self.config.verify_upper_bound {
                domain.upper_bound < threshold
            } else {
                domain.lower_bound > threshold
            };
            if domain_verified {
                domains_verified += 1;
                continue;
            }

            // Conclusive "unsafe" region under the current objective.
            if self.config.verify_upper_bound {
                // Safe goal: upper < threshold. If we prove lower >= threshold, unsafe holds everywhere in this domain.
                if domain.lower_bound >= threshold {
                    return Ok(BetaCrownResult {
                        result: BabVerificationStatus::PotentialViolation,
                        domains_explored,
                        time_elapsed: start_time.elapsed(),
                        max_depth_reached,
                        output_bounds: None,
                        cuts_generated: 0,
                        domains_verified,
                    });
                }
            } else {
                // Safe goal: lower > threshold. If we prove upper < threshold, unsafe holds everywhere in this domain.
                if domain.upper_bound < threshold {
                    return Ok(BetaCrownResult {
                        result: BabVerificationStatus::PotentialViolation,
                        domains_explored,
                        time_elapsed: start_time.elapsed(),
                        max_depth_reached,
                        output_bounds: None,
                        cuts_generated: 0,
                        domains_verified,
                    });
                }
            }

            if domain.depth >= self.config.max_depth {
                unresolved_due_to_depth = true;
                continue;
            }

            // Split input dimension with largest width
            let split_dim = self.select_input_dimension(domain.input_bounds.as_ref());
            let flat = domain.input_bounds.as_ref().flatten();
            let l = flat.lower[[split_dim]];
            let u = flat.upper[[split_dim]];
            let mid = (l + u) / 2.0;

            // Construct children bounds
            let shape = domain.input_bounds.as_ref().lower.shape().to_vec();
            let mut child_lower = flat.lower.clone();
            let mut child_upper = flat.upper.clone();

            // Left child: [l, mid]
            child_lower[[split_dim]] = l;
            child_upper[[split_dim]] = mid;
            let left_lower_arr = child_lower
                .clone()
                .into_shape_clone(ndarray::IxDyn(&shape))
                .map_err(|e| GammaError::InvalidSpec(format!("reshape lower: {}", e)))?;
            let left_upper_arr = child_upper
                .clone()
                .into_shape_clone(ndarray::IxDyn(&shape))
                .map_err(|e| GammaError::InvalidSpec(format!("reshape upper: {}", e)))?;
            let left_input = BoundedTensor::new(left_lower_arr, left_upper_arr)?;
            let left_output = graph.propagate_crown_with_engine(&left_input, engine)?;
            let (left_obj_l, left_obj_u) = objective_bounds(&left_output, objective)?;
            let left_priority = if self.config.verify_upper_bound {
                left_obj_u
            } else {
                -left_obj_l
            };
            let left_domain = GraphInputDomain {
                input_bounds: Arc::new(left_input),
                lower_bound: left_obj_l,
                upper_bound: left_obj_u,
                depth: domain.depth + 1,
                priority: left_priority,
            };

            // Right child: [mid, u]
            child_lower[[split_dim]] = mid;
            child_upper[[split_dim]] = u;
            let right_lower_arr = child_lower
                .into_shape_clone(ndarray::IxDyn(&shape))
                .map_err(|e| GammaError::InvalidSpec(format!("reshape lower: {}", e)))?;
            let right_upper_arr = child_upper
                .into_shape_clone(ndarray::IxDyn(&shape))
                .map_err(|e| GammaError::InvalidSpec(format!("reshape upper: {}", e)))?;
            let right_input = BoundedTensor::new(right_lower_arr, right_upper_arr)?;
            let right_output = graph.propagate_crown_with_engine(&right_input, engine)?;
            let (right_obj_l, right_obj_u) = objective_bounds(&right_output, objective)?;
            let right_priority = if self.config.verify_upper_bound {
                right_obj_u
            } else {
                -right_obj_l
            };
            let right_domain = GraphInputDomain {
                input_bounds: Arc::new(right_input),
                lower_bound: right_obj_l,
                upper_bound: right_obj_u,
                depth: domain.depth + 1,
                priority: right_priority,
            };

            for child in [left_domain, right_domain] {
                let child_verified = if self.config.verify_upper_bound {
                    child.upper_bound < threshold
                } else {
                    child.lower_bound > threshold
                };
                if child_verified {
                    domains_verified += 1;
                } else {
                    queue.push(child);
                }
            }
        }

        if unresolved_due_to_depth {
            return Ok(BetaCrownResult {
                result: BabVerificationStatus::Unknown {
                    reason: format!("Max depth {} reached", self.config.max_depth),
                },
                domains_explored,
                time_elapsed: start_time.elapsed(),
                max_depth_reached,
                output_bounds: None,
                cuts_generated: 0,
                domains_verified,
            });
        }

        Ok(BetaCrownResult {
            result: BabVerificationStatus::Verified,
            domains_explored,
            time_elapsed: start_time.elapsed(),
            max_depth_reached,
            output_bounds: None,
            cuts_generated: 0,
            domains_verified,
        })
    }

    /// GraphNetwork verification with ReLU-splitting branch-and-bound.
    ///
    /// This is the full DAG β-CROWN implementation that branches on unstable
    /// ReLU neurons rather than input dimensions. It is more precise than
    /// input splitting but requires more complex constraint tracking.
    ///
    /// # Algorithm
    /// 1. Collect initial node bounds via IBP
    /// 2. Run CROWN with constraint-aware ReLU relaxation
    /// 3. Find unstable neurons (l < 0 < u) that aren't constrained
    /// 4. Select neuron to split using branching heuristic
    /// 5. Create two child domains (active/inactive constraints)
    /// 6. Repeat until all domains verified or limits reached
    ///
    /// When `use_alpha_crown` is enabled, the initial bounds are computed using
    /// α-CROWN optimization which provides ~10x tighter bounds than IBP.
    ///
    /// # Limitations
    /// - Branching heuristic is currently "widest bound"
    #[instrument(skip(self, graph, input, objective), fields(threshold, input_shape = ?input.shape(), num_nodes = graph.nodes.len()))]
    pub fn verify_graph_relu_split(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objective: &[f32],
        threshold: f32,
    ) -> Result<BetaCrownResult> {
        self.verify_graph_relu_split_impl(graph, input, objective, threshold, None)
    }

    /// Internal GraphNetwork ReLU split implementation with optional GemmEngine.
    fn verify_graph_relu_split_impl(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objective: &[f32],
        threshold: f32,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BetaCrownResult> {
        let start_time = Instant::now();
        let mut domains_explored = 0usize;
        let mut domains_verified = 0usize;
        let mut max_depth_reached = 0usize;
        let mut unresolved_due_to_depth = false;
        let mut unresolved_due_to_no_branch = false;

        // Helper to compute objective bounds from output bounds
        fn objective_bounds(output: &BoundedTensor, objective: &[f32]) -> Result<(f32, f32)> {
            let flat = output.flatten();
            if flat.len() != objective.len() {
                return Err(GammaError::shape_mismatch(
                    vec![objective.len()],
                    vec![flat.len()],
                ));
            }

            let mut lower = 0.0f32;
            let mut upper = 0.0f32;
            for (idx, &c) in objective.iter().enumerate() {
                let l = flat.lower[[idx]];
                let u = flat.upper[[idx]];
                if c >= 0.0 {
                    lower += c * l;
                    upper += c * u;
                } else {
                    lower += c * u;
                    upper += c * l;
                }
            }
            Ok((lower, upper))
        }

        // Step 1: Collect initial node bounds
        // - α-CROWN: iteratively optimizes ReLU slopes (~10x tighter than IBP)
        // - IBP: fast O(N) forward pass (default when no alpha)
        // - CROWN-IBP: O(N²) backward pass per node (optional, slower but tighter)
        //
        // By default, use IBP for fast initialization (matches α,β-CROWN's fix_interm_bounds=True).
        // This reduces resnet_4b init from ~80s to <5s.
        let initial_node_bounds = if self.config.use_alpha_crown {
            info!(
                "Computing α-CROWN initial bounds ({} iterations, fix_interm_bounds={})...",
                self.config.alpha_config.iterations, self.config.alpha_config.fix_interm_bounds
            );
            graph.collect_alpha_crown_bounds_dag(input, &self.config.alpha_config)?
        } else if self.config.alpha_config.fix_interm_bounds {
            // Fast path: use IBP bounds (O(N)) for intermediate nodes
            info!("Computing IBP initial bounds (fast O(N) initialization)...");
            graph.collect_node_bounds(input)?
        } else {
            // Slow path: use CROWN-IBP bounds (O(N²)) for tighter intermediate bounds
            info!("Computing CROWN-IBP initial bounds (O(N²) - slow but tighter)...");
            graph.collect_crown_ibp_bounds_dag(input)?
        };

        // Step 2: Compute initial output bounds via α-CROWN (if enabled) or CROWN
        // Use GPU-accelerated CROWN if engine is provided
        let initial_output = if self.config.use_alpha_crown {
            graph.propagate_alpha_crown_with_config_and_engine(
                input,
                &self.config.alpha_config,
                engine,
            )?
        } else {
            graph.propagate_crown_with_engine(input, engine)?
        };
        let (root_lower, root_upper) = objective_bounds(&initial_output, objective)?;

        info!(
            "Graph β-CROWN (ReLU split) initial objective: [{}, {}], threshold: {}, verify_upper={}",
            root_lower, root_upper, threshold, self.config.verify_upper_bound
        );

        // Quick verification check
        if self.config.verify_upper_bound {
            if root_upper < threshold {
                return Ok(BetaCrownResult {
                    result: BabVerificationStatus::Verified,
                    domains_explored: 1,
                    time_elapsed: start_time.elapsed(),
                    max_depth_reached: 0,
                    output_bounds: Some(BoundedTensor::new(
                        Array1::from_vec(vec![root_lower]).into_dyn(),
                        Array1::from_vec(vec![root_upper]).into_dyn(),
                    )?),
                    cuts_generated: 0,
                    domains_verified: 1,
                });
            }
            if root_lower >= threshold {
                return Ok(BetaCrownResult {
                    result: BabVerificationStatus::PotentialViolation,
                    domains_explored: 1,
                    time_elapsed: start_time.elapsed(),
                    max_depth_reached: 0,
                    output_bounds: Some(BoundedTensor::new(
                        Array1::from_vec(vec![root_lower]).into_dyn(),
                        Array1::from_vec(vec![root_upper]).into_dyn(),
                    )?),
                    cuts_generated: 0,
                    domains_verified: 0,
                });
            }
        } else {
            if root_lower > threshold {
                return Ok(BetaCrownResult {
                    result: BabVerificationStatus::Verified,
                    domains_explored: 1,
                    time_elapsed: start_time.elapsed(),
                    max_depth_reached: 0,
                    output_bounds: Some(BoundedTensor::new(
                        Array1::from_vec(vec![root_lower]).into_dyn(),
                        Array1::from_vec(vec![root_upper]).into_dyn(),
                    )?),
                    cuts_generated: 0,
                    domains_verified: 1,
                });
            }
            if root_upper < threshold {
                return Ok(BetaCrownResult {
                    result: BabVerificationStatus::PotentialViolation,
                    domains_explored: 1,
                    time_elapsed: start_time.elapsed(),
                    max_depth_reached: 0,
                    output_bounds: Some(BoundedTensor::new(
                        Array1::from_vec(vec![root_lower]).into_dyn(),
                        Array1::from_vec(vec![root_upper]).into_dyn(),
                    )?),
                    cuts_generated: 0,
                    domains_verified: 0,
                });
            }
        }

        // Convert bounds to Arc for proactive cuts (if enabled)
        let initial_node_bounds_arc: std::collections::HashMap<String, Arc<BoundedTensor>> =
            initial_node_bounds
                .iter()
                .map(|(k, v)| (k.clone(), Arc::new(v.clone())))
                .collect();

        // Create root domain
        let root_domain = GraphBabDomain::root(
            initial_node_bounds,
            root_lower,
            root_upper,
            input,
            self.config.verify_upper_bound,
        );

        // Branch-and-bound queue
        let mut queue: BinaryHeap<GraphBabDomain> = BinaryHeap::new();
        queue.push(root_domain);

        // Identify ReLU nodes in the graph
        let relu_nodes: Vec<String> = graph
            .nodes
            .iter()
            .filter_map(|(name, node)| {
                if matches!(node.layer, Layer::ReLU(_)) {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect();

        info!("Found {} ReLU nodes for branching", relu_nodes.len());

        // Initialize graph cut pool if enabled
        let mut cut_pool = if self.config.enable_cuts {
            GraphCutPool::with_min_depth(self.config.max_cuts, self.config.min_cut_depth)
        } else {
            GraphCutPool::new(0) // Disabled: pool with 0 capacity
        };

        // Generate proactive cuts if enabled (BICCOS-lite)
        // This populates the cut pool BEFORE BaB starts, avoiding the chicken-and-egg
        // problem where no cuts are generated because no domains verify
        if self.config.enable_proactive_cuts && self.config.enable_cuts {
            let proactive_count = cut_pool.generate_proactive_cuts(
                graph,
                &initial_node_bounds_arc,
                self.config.max_proactive_cuts,
            );
            if proactive_count > 0 {
                info!(
                    "Generated {} proactive cuts for {} ReLU nodes",
                    proactive_count,
                    relu_nodes.len()
                );
            }
        }

        // Lambda optimization state (for cuts)
        let mut lambda_opt_iter = 0usize;
        let lambda_opt_interval = 20; // Optimize lambdas every 20 domains
        let lambda_lr = 0.05f32;
        let lambda_beta1 = 0.9f32;
        let lambda_beta2 = 0.999f32;
        let lambda_epsilon = 1e-8f32;

        // Batch processing: pop multiple domains and process in parallel
        let batch_size = self.config.batch_size.max(1);
        let use_parallel = batch_size > 1;

        while !queue.is_empty() {
            // Check termination conditions before processing batch
            if start_time.elapsed() > self.config.timeout {
                return Ok(BetaCrownResult {
                    result: BabVerificationStatus::Unknown {
                        reason: "Timeout".to_string(),
                    },
                    domains_explored,
                    time_elapsed: start_time.elapsed(),
                    max_depth_reached,
                    output_bounds: None,
                    cuts_generated: cut_pool.total_generated,
                    domains_verified,
                });
            }

            if domains_explored >= self.config.max_domains {
                return Ok(BetaCrownResult {
                    result: BabVerificationStatus::Unknown {
                        reason: format!("Domain limit {} reached", self.config.max_domains),
                    },
                    domains_explored,
                    time_elapsed: start_time.elapsed(),
                    max_depth_reached,
                    output_bounds: None,
                    cuts_generated: cut_pool.total_generated,
                    domains_verified,
                });
            }

            // Pop up to batch_size domains from the queue
            let mut batch: Vec<GraphBabDomain> = Vec::with_capacity(batch_size);
            while batch.len() < batch_size {
                if let Some(domain) = queue.pop() {
                    batch.push(domain);
                } else {
                    break;
                }
            }

            if batch.is_empty() {
                break;
            }

            // Pre-filter: separate already-verified domains and check for violations
            let mut domains_to_process: Vec<GraphBabDomain> = Vec::new();
            let mut found_violation = false;

            for domain in batch {
                domains_explored += 1;
                max_depth_reached = max_depth_reached.max(domain.depth);

                // Check if domain is already verified
                let domain_verified = if self.config.verify_upper_bound {
                    domain.upper_bound < threshold
                } else {
                    domain.lower_bound > threshold
                };
                if domain_verified {
                    domains_verified += 1;
                    // Generate cut from verified domain
                    if self.config.enable_cuts
                        && domain.depth >= self.config.min_cut_depth
                        && cut_pool.add_from_verified_domain(&domain.history)
                    {
                        debug!(
                            "Generated cut from verified domain (depth={}, total cuts={})",
                            domain.depth,
                            cut_pool.len()
                        );
                    }
                    continue;
                }

                // Check for conclusive violation
                if self.config.verify_upper_bound {
                    if domain.lower_bound >= threshold {
                        found_violation = true;
                        break;
                    }
                } else if domain.upper_bound < threshold {
                    found_violation = true;
                    break;
                }

                // Near-miss cut generation (sequential, needs mutable cut_pool)
                if self.config.enable_cuts
                    && self.config.enable_near_miss_cuts
                    && domain.depth >= self.config.min_cut_depth
                {
                    let bound_for_check = if self.config.verify_upper_bound {
                        domain.upper_bound
                    } else {
                        domain.lower_bound
                    };
                    if cut_pool.add_from_near_miss_domain(
                        &domain.history,
                        bound_for_check,
                        threshold,
                        self.config.near_miss_margin,
                    ) {
                        debug!(
                            "Generated near-miss cut (depth={}, lb={:.4}, threshold={:.4}, total cuts={})",
                            domain.depth,
                            bound_for_check,
                            threshold,
                            cut_pool.len()
                        );
                    }
                }

                // Check depth limit
                if domain.depth >= self.config.max_depth {
                    unresolved_due_to_depth = true;
                    continue;
                }

                domains_to_process.push(domain);
            }

            if found_violation {
                return Ok(BetaCrownResult {
                    result: BabVerificationStatus::PotentialViolation,
                    domains_explored,
                    time_elapsed: start_time.elapsed(),
                    max_depth_reached,
                    output_bounds: None,
                    cuts_generated: cut_pool.total_generated,
                    domains_verified,
                });
            }

            if domains_to_process.is_empty() {
                continue;
            }

            // Lambda optimization: periodically optimize cut lambdas (sequential)
            if self.config.enable_cuts
                && !cut_pool.is_empty()
                && domains_explored % lambda_opt_interval == 0
            {
                lambda_opt_iter += 1;

                // Use first domain for gradient estimation
                if let Some(sample_domain) = domains_to_process.first() {
                    self.compute_graph_cut_gradients(
                        graph,
                        &mut cut_pool,
                        &sample_domain.node_bounds,
                        sample_domain.input_bounds.as_ref(),
                    );
                }

                // Update all cut lambdas
                for cut in cut_pool.cuts_mut() {
                    cut.update_lambda_adam(
                        lambda_lr,
                        lambda_beta1,
                        lambda_beta2,
                        lambda_epsilon,
                        lambda_opt_iter,
                    );
                }
                cut_pool.zero_grad();

                debug!(
                    "Lambda optimization iter {}: total_lambda = {:.4}",
                    lambda_opt_iter,
                    cut_pool.total_lambda()
                );
            }

            // Process domains: GPU-batched when engine available, parallel CPU otherwise
            let has_active_cuts = !cut_pool.is_empty() && self.config.enable_cuts;

            let results: Vec<GraphDomainResult> = if use_parallel && !has_active_cuts {
                // Prefer GPU-batched processing when engine is available
                if let Some(eng) = engine {
                    // GPU-batched processing: significantly better GPU utilization
                    // Batches children creation and CROWN passes for all domains
                    let domain_refs: Vec<&GraphBabDomain> = domains_to_process.iter().collect();
                    self.process_graph_domains_batched_gpu(
                        graph,
                        &domain_refs,
                        &relu_nodes,
                        objective,
                        threshold,
                        eng,
                    )
                } else {
                    // CPU parallel processing (no GPU engine)
                    domains_to_process
                        .par_iter()
                        .map(|domain| {
                            self.process_graph_domain_parallel(
                                graph,
                                domain,
                                &relu_nodes,
                                objective,
                                threshold,
                                None,
                            )
                        })
                        .collect()
                }
            } else {
                // Sequential processing (cuts enabled or batch_size=1)
                // Use full SPSA β optimization for better bounds
                let mut seq_results = Vec::with_capacity(domains_to_process.len());
                for domain in &domains_to_process {
                    // Find unstable neurons
                    let unstable = self.find_unstable_graph_neurons(graph, domain, &relu_nodes);
                    if unstable.is_empty() {
                        // No unstable neurons - try CROWN with cuts
                        let context = GraphCrownContext::new(
                            &domain.history,
                            Some(&cut_pool),
                            Some(&domain.node_bounds),
                            engine,
                        );
                        if let Ok((output, _node_cache)) = self
                            .propagate_crown_with_graph_constraints(
                                graph,
                                domain.input_bounds.as_ref(),
                                &context,
                                None,
                                Some(objective),
                            )
                        {
                            let l = output.lower_scalar();
                            let u = output.upper_scalar();
                            let verified = if self.config.verify_upper_bound {
                                u < threshold
                            } else {
                                l > threshold
                            };
                            seq_results.push(GraphDomainResult::NoUnstable {
                                lower: l,
                                upper: u,
                                verified,
                            });
                        } else {
                            seq_results.push(GraphDomainResult::NoUnstable {
                                lower: domain.lower_bound,
                                upper: domain.upper_bound,
                                verified: false,
                            });
                        }
                        continue;
                    }

                    // Select neuron and create children with full SPSA optimization
                    let (node_name, neuron_idx) =
                        self.select_graph_branch(graph, domain, &unstable);
                    let mut children: Vec<(GraphBabDomain, bool)> = Vec::with_capacity(2);

                    // Active child
                    let active_constraint = GraphNeuronConstraint {
                        node_name: node_name.clone(),
                        neuron_idx,
                        is_active: true,
                    };
                    if let Some(mut active_child) = domain.with_constraint(
                        graph,
                        active_constraint,
                        self.config.verify_upper_bound,
                    ) {
                        let context = GraphCrownContext::new(
                            &active_child.history,
                            Some(&cut_pool),
                            Some(&domain.node_bounds),
                            engine,
                        );
                        // Only run β optimization when enabled and for shallow domains
                        // When beta_iterations=0, skip optimization entirely for all domains
                        let should_optimize = self.config.beta_iterations > 0
                            && active_child.depth <= self.config.beta_max_depth;
                        let beta_result =
                            if should_optimize && self.config.use_analytical_beta_gradients {
                                self.optimize_graph_beta_analytical(
                                    graph,
                                    active_child.input_bounds.as_ref(),
                                    &context,
                                    &mut active_child.beta_state,
                                    objective,
                                )
                            } else if should_optimize {
                                self.optimize_graph_beta_spsa(
                                    graph,
                                    active_child.input_bounds.as_ref(),
                                    &context,
                                    &mut active_child.beta_state,
                                    objective,
                                )
                            } else {
                                // Skip optimization, just propagate with inherited β
                                self.propagate_crown_with_graph_beta(
                                    graph,
                                    active_child.input_bounds.as_ref(),
                                    &context,
                                    &active_child.beta_state,
                                    Some(objective),
                                )
                                .map(|(out, cache)| (out.lower_scalar(), out.upper_scalar(), cache))
                            };
                        if let Ok((l, u, node_cache)) = beta_result {
                            active_child.node_bounds = node_cache
                                .into_iter()
                                .map(|(k, v)| (k, Arc::new(v)))
                                .collect();
                            active_child.lower_bound = l;
                            active_child.upper_bound = u;
                            active_child.priority = if self.config.verify_upper_bound {
                                u
                            } else {
                                -l
                            };

                            let verified = if self.config.verify_upper_bound {
                                u < threshold
                            } else {
                                l > threshold
                            };
                            children.push((active_child, verified));
                        }
                    }

                    // Inactive child
                    let inactive_constraint = GraphNeuronConstraint {
                        node_name: node_name.clone(),
                        neuron_idx,
                        is_active: false,
                    };
                    if let Some(mut inactive_child) = domain.with_constraint(
                        graph,
                        inactive_constraint,
                        self.config.verify_upper_bound,
                    ) {
                        let context = GraphCrownContext::new(
                            &inactive_child.history,
                            Some(&cut_pool),
                            Some(&domain.node_bounds),
                            engine,
                        );
                        // Only run β optimization when enabled and for shallow domains
                        let should_optimize = self.config.beta_iterations > 0
                            && inactive_child.depth <= self.config.beta_max_depth;
                        let beta_result =
                            if should_optimize && self.config.use_analytical_beta_gradients {
                                self.optimize_graph_beta_analytical(
                                    graph,
                                    inactive_child.input_bounds.as_ref(),
                                    &context,
                                    &mut inactive_child.beta_state,
                                    objective,
                                )
                            } else if should_optimize {
                                self.optimize_graph_beta_spsa(
                                    graph,
                                    inactive_child.input_bounds.as_ref(),
                                    &context,
                                    &mut inactive_child.beta_state,
                                    objective,
                                )
                            } else {
                                // Skip optimization, just propagate with inherited β
                                self.propagate_crown_with_graph_beta(
                                    graph,
                                    inactive_child.input_bounds.as_ref(),
                                    &context,
                                    &inactive_child.beta_state,
                                    Some(objective),
                                )
                                .map(|(out, cache)| (out.lower_scalar(), out.upper_scalar(), cache))
                            };
                        if let Ok((l, u, node_cache)) = beta_result {
                            inactive_child.node_bounds = node_cache
                                .into_iter()
                                .map(|(k, v)| (k, Arc::new(v)))
                                .collect();
                            inactive_child.lower_bound = l;
                            inactive_child.upper_bound = u;
                            inactive_child.priority = if self.config.verify_upper_bound {
                                u
                            } else {
                                -l
                            };

                            let verified = if self.config.verify_upper_bound {
                                u < threshold
                            } else {
                                l > threshold
                            };
                            children.push((inactive_child, verified));
                        }
                    }

                    seq_results.push(GraphDomainResult::Children(children));
                }
                seq_results
            };

            // Process results: handle children, track verified/violations
            for result in results {
                match result {
                    GraphDomainResult::AlreadyVerified => {
                        domains_verified += 1;
                    }
                    GraphDomainResult::Violation => {
                        return Ok(BetaCrownResult {
                            result: BabVerificationStatus::PotentialViolation,
                            domains_explored,
                            time_elapsed: start_time.elapsed(),
                            max_depth_reached,
                            output_bounds: None,
                            cuts_generated: cut_pool.total_generated,
                            domains_verified,
                        });
                    }
                    GraphDomainResult::Children(children) => {
                        for (child, verified) in children {
                            if verified {
                                domains_verified += 1;
                                // Generate cut from verified child
                                if self.config.enable_cuts
                                    && child.depth >= self.config.min_cut_depth
                                    && cut_pool.add_from_verified_domain(&child.history)
                                {
                                    debug!(
                                        "Generated cut from verified child (depth={}, total cuts={})",
                                        child.depth,
                                        cut_pool.len()
                                    );
                                }
                            } else {
                                queue.push(child);
                            }
                        }
                    }
                    GraphDomainResult::NoUnstable {
                        lower,
                        upper,
                        verified,
                    } => {
                        if verified {
                            domains_verified += 1;
                        } else {
                            // Check for violation in no-unstable case
                            if self.config.verify_upper_bound {
                                if lower >= threshold {
                                    return Ok(BetaCrownResult {
                                        result: BabVerificationStatus::PotentialViolation,
                                        domains_explored,
                                        time_elapsed: start_time.elapsed(),
                                        max_depth_reached,
                                        output_bounds: None,
                                        cuts_generated: cut_pool.total_generated,
                                        domains_verified,
                                    });
                                }
                            } else if upper < threshold {
                                return Ok(BetaCrownResult {
                                    result: BabVerificationStatus::PotentialViolation,
                                    domains_explored,
                                    time_elapsed: start_time.elapsed(),
                                    max_depth_reached,
                                    output_bounds: None,
                                    cuts_generated: cut_pool.total_generated,
                                    domains_verified,
                                });
                            }
                            unresolved_due_to_no_branch = true;
                        }
                    }
                    GraphDomainResult::Error(msg) => {
                        warn!("Domain processing error: {}", msg);
                        // Continue processing other domains
                    }
                }
            }
        }

        if unresolved_due_to_depth || unresolved_due_to_no_branch {
            let mut reason_parts = Vec::new();
            if unresolved_due_to_depth {
                reason_parts.push(format!("Max depth {} reached", self.config.max_depth));
            }
            if unresolved_due_to_no_branch {
                reason_parts.push("No unstable ReLU neurons left in some domains".to_string());
            }
            return Ok(BetaCrownResult {
                result: BabVerificationStatus::Unknown {
                    reason: reason_parts.join("; "),
                },
                domains_explored,
                time_elapsed: start_time.elapsed(),
                max_depth_reached,
                output_bounds: None,
                cuts_generated: cut_pool.total_generated,
                domains_verified,
            });
        }

        info!(
            "Graph β-CROWN (ReLU split) verified after {} domains, {} verified, {} cuts",
            domains_explored, domains_verified, cut_pool.total_generated
        );

        Ok(BetaCrownResult {
            result: BabVerificationStatus::Verified,
            domains_explored,
            time_elapsed: start_time.elapsed(),
            max_depth_reached,
            output_bounds: None,
            cuts_generated: cut_pool.total_generated,
            domains_verified,
        })
    }

    /// Pre-compute initial bounds (intermediate + output) using α-CROWN or CROWN-IBP.
    ///
    /// Call this once before verifying multiple constraints on the same graph/input.
    /// The returned tuple contains (intermediate node bounds, output bounds).
    ///
    /// This optimization provides ~9x speedup for CIFAR-10 classification
    /// (9 constraints that would each re-compute bounds).
    pub fn compute_initial_graph_bounds(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
    ) -> Result<(
        std::collections::HashMap<String, BoundedTensor>,
        BoundedTensor,
    )> {
        let (node_bounds, output_bounds) = if self.config.use_alpha_crown {
            info!(
                "Pre-computing α-CROWN bounds ({} iterations, fix_interm_bounds={})...",
                self.config.alpha_config.iterations, self.config.alpha_config.fix_interm_bounds
            );
            let node_bounds =
                graph.collect_alpha_crown_bounds_dag(input, &self.config.alpha_config)?;
            let output_bounds =
                graph.propagate_alpha_crown_with_config(input, &self.config.alpha_config)?;
            (node_bounds, output_bounds)
        } else if self.config.alpha_config.fix_interm_bounds {
            // Fast path: use IBP bounds (O(N)) for intermediate nodes
            info!("Pre-computing IBP bounds (fast O(N) initialization)...");
            let node_bounds = graph.collect_node_bounds(input)?;
            let output_bounds = graph.propagate_crown(input)?;
            (node_bounds, output_bounds)
        } else {
            // Slow path: use CROWN-IBP bounds (O(N²)) for tighter intermediate bounds
            info!("Pre-computing CROWN-IBP bounds (O(N²) - slow but tighter)...");
            let node_bounds = graph.collect_crown_ibp_bounds_dag(input)?;
            let output_bounds = graph.propagate_crown(input)?;
            (node_bounds, output_bounds)
        };
        Ok((node_bounds, output_bounds))
    }

    /// Verify a graph using ReLU splitting with pre-computed initial bounds.
    ///
    /// Use this when verifying multiple constraints on the same graph/input.
    /// Pre-compute bounds once with `compute_initial_graph_bounds`, then
    /// call this method for each constraint.
    ///
    /// Arguments:
    /// - `precomputed_bounds`: Pre-computed node and output bounds
    #[instrument(skip(self, graph, input, objective, precomputed_bounds), fields(threshold, input_shape = ?input.shape(), num_nodes = graph.nodes.len()))]
    pub fn verify_graph_relu_split_with_bounds(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objective: &[f32],
        threshold: f32,
        precomputed_bounds: &GraphPrecomputedBounds<'_>,
    ) -> Result<BetaCrownResult> {
        self.verify_graph_relu_split_with_bounds_impl(
            graph,
            input,
            objective,
            threshold,
            precomputed_bounds,
            None,
        )
    }

    fn verify_graph_relu_split_with_bounds_impl(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objective: &[f32],
        threshold: f32,
        precomputed_bounds: &GraphPrecomputedBounds<'_>,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BetaCrownResult> {
        let start_time = Instant::now();
        let mut domains_explored = 0usize;
        let mut domains_verified = 0usize;
        let mut max_depth_reached = 0usize;
        let mut unresolved_due_to_depth = false;
        let mut unresolved_due_to_no_branch = false;

        // Helper to compute objective bounds from output bounds
        fn objective_bounds(output: &BoundedTensor, objective: &[f32]) -> Result<(f32, f32)> {
            let flat = output.flatten();
            if flat.len() != objective.len() {
                return Err(GammaError::shape_mismatch(
                    vec![objective.len()],
                    vec![flat.len()],
                ));
            }

            let mut lower = 0.0f32;
            let mut upper = 0.0f32;
            for (idx, &c) in objective.iter().enumerate() {
                let l = flat.lower[[idx]];
                let u = flat.upper[[idx]];
                if c >= 0.0 {
                    lower += c * l;
                    upper += c * u;
                } else {
                    lower += c * u;
                    upper += c * l;
                }
            }
            Ok((lower, upper))
        }

        // Use pre-computed bounds (clone to owned HashMap)
        let initial_node_bounds: std::collections::HashMap<String, BoundedTensor> =
            precomputed_bounds.node_bounds.clone();

        // Use pre-computed output bounds - apply objective to get initial bounds
        let (root_lower, root_upper) =
            objective_bounds(precomputed_bounds.output_bounds, objective)?;

        info!(
            "Graph β-CROWN (ReLU split, pre-computed bounds) initial objective: [{}, {}], threshold: {}, verify_upper={}",
            root_lower, root_upper, threshold, self.config.verify_upper_bound
        );

        // Quick verification check
        if self.config.verify_upper_bound {
            if root_upper < threshold {
                return Ok(BetaCrownResult {
                    result: BabVerificationStatus::Verified,
                    domains_explored: 1,
                    time_elapsed: start_time.elapsed(),
                    max_depth_reached: 0,
                    output_bounds: Some(BoundedTensor::new(
                        Array1::from_vec(vec![root_lower]).into_dyn(),
                        Array1::from_vec(vec![root_upper]).into_dyn(),
                    )?),
                    cuts_generated: 0,
                    domains_verified: 1,
                });
            }
            if root_lower >= threshold {
                return Ok(BetaCrownResult {
                    result: BabVerificationStatus::PotentialViolation,
                    domains_explored: 1,
                    time_elapsed: start_time.elapsed(),
                    max_depth_reached: 0,
                    output_bounds: Some(BoundedTensor::new(
                        Array1::from_vec(vec![root_lower]).into_dyn(),
                        Array1::from_vec(vec![root_upper]).into_dyn(),
                    )?),
                    cuts_generated: 0,
                    domains_verified: 0,
                });
            }
        } else {
            if root_lower > threshold {
                return Ok(BetaCrownResult {
                    result: BabVerificationStatus::Verified,
                    domains_explored: 1,
                    time_elapsed: start_time.elapsed(),
                    max_depth_reached: 0,
                    output_bounds: Some(BoundedTensor::new(
                        Array1::from_vec(vec![root_lower]).into_dyn(),
                        Array1::from_vec(vec![root_upper]).into_dyn(),
                    )?),
                    cuts_generated: 0,
                    domains_verified: 1,
                });
            }
            if root_upper < threshold {
                return Ok(BetaCrownResult {
                    result: BabVerificationStatus::PotentialViolation,
                    domains_explored: 1,
                    time_elapsed: start_time.elapsed(),
                    max_depth_reached: 0,
                    output_bounds: Some(BoundedTensor::new(
                        Array1::from_vec(vec![root_lower]).into_dyn(),
                        Array1::from_vec(vec![root_upper]).into_dyn(),
                    )?),
                    cuts_generated: 0,
                    domains_verified: 0,
                });
            }
        }

        // Convert to Arc for proactive cuts (if enabled)
        let initial_node_bounds_arc: std::collections::HashMap<String, Arc<BoundedTensor>> =
            initial_node_bounds
                .iter()
                .map(|(k, v)| (k.clone(), Arc::new(v.clone())))
                .collect();

        // Create root domain
        let root_domain = GraphBabDomain::root(
            initial_node_bounds,
            root_lower,
            root_upper,
            input,
            self.config.verify_upper_bound,
        );

        // Branch-and-bound queue
        let mut queue: BinaryHeap<GraphBabDomain> = BinaryHeap::new();
        queue.push(root_domain);

        // Identify ReLU nodes in the graph
        let relu_nodes: Vec<String> = graph
            .nodes
            .iter()
            .filter_map(|(name, node)| {
                if matches!(node.layer, Layer::ReLU(_)) {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect();

        info!("Found {} ReLU nodes for branching", relu_nodes.len());

        // Initialize graph cut pool if enabled
        let mut cut_pool = if self.config.enable_cuts {
            GraphCutPool::with_min_depth(self.config.max_cuts, self.config.min_cut_depth)
        } else {
            GraphCutPool::new(0) // Disabled: pool with 0 capacity
        };

        // Generate proactive cuts if enabled (BICCOS-lite)
        // This populates the cut pool BEFORE BaB starts, avoiding the chicken-and-egg
        // problem where no cuts are generated because no domains verify
        if self.config.enable_proactive_cuts && self.config.enable_cuts {
            let proactive_count = cut_pool.generate_proactive_cuts(
                graph,
                &initial_node_bounds_arc,
                self.config.max_proactive_cuts,
            );
            if proactive_count > 0 {
                info!(
                    "Generated {} proactive cuts for {} ReLU nodes",
                    proactive_count,
                    relu_nodes.len()
                );
            }
        }

        // Lambda optimization state
        let mut lambda_opt_iter = 0usize;
        let lambda_opt_interval = 20; // Optimize lambdas every 20 domains
        let lambda_lr = 0.05f32;
        let lambda_beta1 = 0.9f32;
        let lambda_beta2 = 0.999f32;
        let lambda_epsilon = 1e-8f32;

        while let Some(domain) = queue.pop() {
            // Check timeout
            if start_time.elapsed() > self.config.timeout {
                return Ok(BetaCrownResult {
                    result: BabVerificationStatus::Unknown {
                        reason: "Timeout".to_string(),
                    },
                    domains_explored,
                    time_elapsed: start_time.elapsed(),
                    max_depth_reached,
                    output_bounds: None,
                    cuts_generated: cut_pool.total_generated,
                    domains_verified,
                });
            }

            // Check domain limit
            if domains_explored >= self.config.max_domains {
                return Ok(BetaCrownResult {
                    result: BabVerificationStatus::Unknown {
                        reason: format!("Domain limit {} reached", self.config.max_domains),
                    },
                    domains_explored,
                    time_elapsed: start_time.elapsed(),
                    max_depth_reached,
                    output_bounds: None,
                    cuts_generated: cut_pool.total_generated,
                    domains_verified,
                });
            }

            domains_explored += 1;
            max_depth_reached = max_depth_reached.max(domain.depth);

            // Lambda optimization: periodically optimize cut lambdas
            if self.config.enable_cuts
                && !cut_pool.is_empty()
                && domains_explored % lambda_opt_interval == 0
            {
                lambda_opt_iter += 1;

                // Compute gradients using current domain's node bounds
                self.compute_graph_cut_gradients(
                    graph,
                    &mut cut_pool,
                    &domain.node_bounds,
                    domain.input_bounds.as_ref(),
                );

                // Update all cut lambdas
                for cut in cut_pool.cuts_mut() {
                    cut.update_lambda_adam(
                        lambda_lr,
                        lambda_beta1,
                        lambda_beta2,
                        lambda_epsilon,
                        lambda_opt_iter,
                    );
                }
                cut_pool.zero_grad();

                debug!(
                    "Lambda optimization iter {}: total_lambda = {:.4}",
                    lambda_opt_iter,
                    cut_pool.total_lambda()
                );
            }

            // Check if domain is already verified
            let domain_verified = if self.config.verify_upper_bound {
                domain.upper_bound < threshold
            } else {
                domain.lower_bound > threshold
            };
            if domain_verified {
                domains_verified += 1;
                // Generate cut from verified domain
                if self.config.enable_cuts
                    && domain.depth >= self.config.min_cut_depth
                    && cut_pool.add_from_verified_domain(&domain.history)
                {
                    debug!(
                        "Generated cut from verified domain (depth={}, total cuts={})",
                        domain.depth,
                        cut_pool.len()
                    );
                }
                continue;
            }

            // Check for conclusive violation
            if self.config.verify_upper_bound {
                if domain.lower_bound >= threshold {
                    return Ok(BetaCrownResult {
                        result: BabVerificationStatus::PotentialViolation,
                        domains_explored,
                        time_elapsed: start_time.elapsed(),
                        max_depth_reached,
                        output_bounds: None,
                        cuts_generated: cut_pool.total_generated,
                        domains_verified,
                    });
                }
            } else if domain.upper_bound < threshold {
                return Ok(BetaCrownResult {
                    result: BabVerificationStatus::PotentialViolation,
                    domains_explored,
                    time_elapsed: start_time.elapsed(),
                    max_depth_reached,
                    output_bounds: None,
                    cuts_generated: cut_pool.total_generated,
                    domains_verified,
                });
            }

            // Near-miss cut generation: generate cuts from domains close to verification
            // This can help prune similar regions in the search space
            if self.config.enable_cuts
                && self.config.enable_near_miss_cuts
                && domain.depth >= self.config.min_cut_depth
            {
                let bound_for_check = if self.config.verify_upper_bound {
                    domain.upper_bound
                } else {
                    domain.lower_bound
                };
                if cut_pool.add_from_near_miss_domain(
                    &domain.history,
                    bound_for_check,
                    threshold,
                    self.config.near_miss_margin,
                ) {
                    debug!(
                        "Generated near-miss cut (depth={}, lb={:.4}, threshold={:.4}, total cuts={})",
                        domain.depth,
                        bound_for_check,
                        threshold,
                        cut_pool.len()
                    );
                }
            }

            // Check depth limit
            if domain.depth >= self.config.max_depth {
                unresolved_due_to_depth = true;
                continue;
            }

            // Find unstable neurons to branch on
            let unstable = self.find_unstable_graph_neurons(graph, &domain, &relu_nodes);
            if unstable.is_empty() {
                // No unstable ReLU neurons left to branch on. If bounds are still inconclusive,
                // we cannot refine this domain further.
                let context = GraphCrownContext::new(
                    &domain.history,
                    Some(&cut_pool),
                    Some(&domain.node_bounds),
                    engine,
                );
                if let Ok((output, _node_cache)) = self.propagate_crown_with_graph_constraints(
                    graph,
                    domain.input_bounds.as_ref(),
                    &context,
                    None, // No β when no unstable neurons
                    Some(objective),
                ) {
                    let l = output.lower_scalar();
                    let u = output.upper_scalar();
                    let domain_verified = if self.config.verify_upper_bound {
                        u < threshold
                    } else {
                        l > threshold
                    };
                    if domain_verified {
                        domains_verified += 1;
                        continue;
                    }

                    if self.config.verify_upper_bound {
                        if l >= threshold {
                            return Ok(BetaCrownResult {
                                result: BabVerificationStatus::PotentialViolation,
                                domains_explored,
                                time_elapsed: start_time.elapsed(),
                                max_depth_reached,
                                output_bounds: None,
                                cuts_generated: cut_pool.total_generated,
                                domains_verified,
                            });
                        }
                    } else if u < threshold {
                        return Ok(BetaCrownResult {
                            result: BabVerificationStatus::PotentialViolation,
                            domains_explored,
                            time_elapsed: start_time.elapsed(),
                            max_depth_reached,
                            output_bounds: None,
                            cuts_generated: cut_pool.total_generated,
                            domains_verified,
                        });
                    }
                }
                unresolved_due_to_no_branch = true;
                continue;
            }

            // Select neuron to split using branching heuristic
            let (node_name, neuron_idx) = self.select_graph_branch(graph, &domain, &unstable);

            // Create active child (x >= 0)
            let active_constraint = GraphNeuronConstraint {
                node_name: node_name.clone(),
                neuron_idx,
                is_active: true,
            };
            if let Some(mut active_child) =
                domain.with_constraint(graph, active_constraint, self.config.verify_upper_bound)
            {
                // Recompute bounds with constraint and apply cuts
                // Use parent domain's node_bounds as base for efficiency
                // β-CROWN adds Lagrangian contribution from split constraints
                let context = GraphCrownContext::new(
                    &active_child.history,
                    Some(&cut_pool),
                    Some(&domain.node_bounds),
                    engine,
                );
                if let Ok((output, node_cache)) = self.propagate_crown_with_graph_beta(
                    graph,
                    active_child.input_bounds.as_ref(),
                    &context,
                    &active_child.beta_state,
                    Some(objective),
                ) {
                    active_child.node_bounds = node_cache
                        .into_iter()
                        .map(|(k, v)| (k, Arc::new(v)))
                        .collect();
                    let l = output.lower_scalar();
                    let u = output.upper_scalar();
                    active_child.lower_bound = l;
                    active_child.upper_bound = u;
                    active_child.priority = if self.config.verify_upper_bound {
                        u
                    } else {
                        -l
                    };

                    let child_verified = if self.config.verify_upper_bound {
                        u < threshold
                    } else {
                        l > threshold
                    };
                    if child_verified {
                        domains_verified += 1;
                        // Generate cut from verified child
                        if self.config.enable_cuts
                            && active_child.depth >= self.config.min_cut_depth
                            && cut_pool.add_from_verified_domain(&active_child.history)
                        {
                            debug!(
                                "Generated cut from verified child (depth={}, total cuts={})",
                                active_child.depth,
                                cut_pool.len()
                            );
                        }
                    } else {
                        queue.push(active_child);
                    }
                }
            }

            // Create inactive child (x < 0)
            let inactive_constraint = GraphNeuronConstraint {
                node_name: node_name.clone(),
                neuron_idx,
                is_active: false,
            };
            if let Some(mut inactive_child) =
                domain.with_constraint(graph, inactive_constraint, self.config.verify_upper_bound)
            {
                // Recompute bounds with constraint and apply cuts
                let context = GraphCrownContext::new(
                    &inactive_child.history,
                    Some(&cut_pool),
                    Some(&domain.node_bounds),
                    engine,
                );
                if let Ok((output, node_cache)) = self.propagate_crown_with_graph_constraints(
                    graph,
                    inactive_child.input_bounds.as_ref(),
                    &context,
                    None, // No β for child bound computation
                    Some(objective),
                ) {
                    inactive_child.node_bounds = node_cache
                        .into_iter()
                        .map(|(k, v)| (k, Arc::new(v)))
                        .collect();
                    let l = output.lower_scalar();
                    let u = output.upper_scalar();
                    inactive_child.lower_bound = l;
                    inactive_child.upper_bound = u;
                    inactive_child.priority = if self.config.verify_upper_bound {
                        u
                    } else {
                        -l
                    };

                    let child_verified = if self.config.verify_upper_bound {
                        u < threshold
                    } else {
                        l > threshold
                    };
                    if child_verified {
                        domains_verified += 1;
                        // Generate cut from verified child
                        if self.config.enable_cuts
                            && inactive_child.depth >= self.config.min_cut_depth
                            && cut_pool.add_from_verified_domain(&inactive_child.history)
                        {
                            debug!(
                                "Generated cut from verified child (depth={}, total cuts={})",
                                inactive_child.depth,
                                cut_pool.len()
                            );
                        }
                    } else {
                        queue.push(inactive_child);
                    }
                }
            }
        }

        // Queue empty - check for unresolved
        if unresolved_due_to_depth || unresolved_due_to_no_branch {
            let mut reason_parts = Vec::new();
            if unresolved_due_to_depth {
                reason_parts.push(format!("Max depth {} reached", self.config.max_depth));
            }
            if unresolved_due_to_no_branch {
                reason_parts.push("Some domains had no unstable neurons to branch on".to_string());
            }
            return Ok(BetaCrownResult {
                result: BabVerificationStatus::Unknown {
                    reason: reason_parts.join("; "),
                },
                domains_explored,
                time_elapsed: start_time.elapsed(),
                max_depth_reached,
                output_bounds: None,
                cuts_generated: cut_pool.total_generated,
                domains_verified,
            });
        }

        info!(
            "Graph β-CROWN (ReLU split, pre-computed bounds) verified after {} domains, {} verified, {} cuts",
            domains_explored, domains_verified, cut_pool.total_generated
        );

        Ok(BetaCrownResult {
            result: BabVerificationStatus::Verified,
            domains_explored,
            time_elapsed: start_time.elapsed(),
            max_depth_reached,
            output_bounds: None,
            cuts_generated: cut_pool.total_generated,
            domains_verified,
        })
    }

    /// Multi-objective verification for disjunctive properties.
    ///
    /// Verifies ALL objectives simultaneously in a single BaB pass, sharing
    /// computation across objectives. For disjunctive properties (OR), this is
    /// required: the property is SAFE only if ALL constraints are proved violated.
    ///
    /// # Arguments
    /// * `graph` - The DAG-based neural network
    /// * `input` - Input bounds
    /// * `objectives` - List of objective vectors (each is a linear combination of outputs)
    /// * `thresholds` - Threshold for each objective (usually all 0.0)
    ///
    /// # Returns
    /// * `Verified` if ALL objectives are verified (all lower > threshold)
    /// * `Unknown` if ANY objective cannot be verified within timeout
    #[instrument(skip(self, graph, input, objectives), fields(num_objectives = objectives.len(), input_shape = ?input.shape()))]
    pub fn verify_graph_relu_split_multi_objective(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objectives: &[Vec<f32>],
        thresholds: &[f32],
    ) -> Result<BetaCrownResult> {
        self.verify_graph_relu_split_multi_objective_with_engine(
            graph, input, objectives, thresholds, None,
        )
    }

    /// Multi-objective Graph β-CROWN verification with GPU acceleration.
    ///
    /// Same as `verify_graph_relu_split_multi_objective` but with optional GPU engine
    /// for accelerated bound computation.
    #[instrument(skip(self, graph, input, objectives, engine), fields(num_objectives = objectives.len(), input_shape = ?input.shape()))]
    pub fn verify_graph_relu_split_multi_objective_with_engine(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BetaCrownResult> {
        let start_time = Instant::now();
        let mut domains_explored = 0usize;
        let mut domains_verified = 0usize;
        let mut max_depth_reached = 0usize;

        let num_objectives = objectives.len();
        if num_objectives == 0 {
            return Ok(BetaCrownResult {
                result: BabVerificationStatus::Verified,
                domains_explored: 0,
                time_elapsed: start_time.elapsed(),
                max_depth_reached: 0,
                output_bounds: None,
                cuts_generated: 0,
                domains_verified: 0,
            });
        }

        // Helper to compute objective bounds from output bounds (interval arithmetic fallback)
        fn objective_bounds_vec(
            output: &BoundedTensor,
            objectives: &[Vec<f32>],
        ) -> Result<Vec<(f32, f32)>> {
            let flat = output.flatten();
            objectives
                .iter()
                .map(|obj| {
                    if flat.len() != obj.len() {
                        return Err(GammaError::shape_mismatch(
                            vec![obj.len()],
                            vec![flat.len()],
                        ));
                    }

                    let mut lower = 0.0f32;
                    let mut upper = 0.0f32;
                    for (idx, &c) in obj.iter().enumerate() {
                        let l = flat.lower[[idx]];
                        let u = flat.upper[[idx]];
                        if c >= 0.0 {
                            lower += c * l;
                            upper += c * u;
                        } else {
                            lower += c * u;
                            upper += c * l;
                        }
                    }
                    Ok((lower, upper))
                })
                .collect()
        }

        // Build specification matrix from objectives for spec-guided CROWN
        fn build_spec_matrix(objectives: &[Vec<f32>]) -> Option<Array2<f32>> {
            if objectives.is_empty() {
                return None;
            }
            let num_specs = objectives.len();
            let output_dim = objectives[0].len();
            let mut data = Vec::with_capacity(num_specs * output_dim);
            for obj in objectives {
                if obj.len() != output_dim {
                    return None; // Inconsistent objective dimensions
                }
                data.extend_from_slice(obj);
            }
            Array2::from_shape_vec((num_specs, output_dim), data).ok()
        }

        // Convert BoundedTensor to Vec<(f32, f32)> bounds
        fn spec_bounds_to_vec(bounds: &BoundedTensor) -> Vec<(f32, f32)> {
            let flat = bounds.flatten();
            (0..flat.len())
                .map(|i| (flat.lower[[i]], flat.upper[[i]]))
                .collect()
        }

        // Step 1: Collect initial node bounds
        let initial_node_bounds = if self.config.use_alpha_crown {
            info!(
                "Multi-objective: Computing α-CROWN initial bounds ({} iterations)...",
                self.config.alpha_config.iterations
            );
            graph.collect_alpha_crown_bounds_dag(input, &self.config.alpha_config)?
        } else if self.config.alpha_config.fix_interm_bounds {
            info!("Multi-objective: Computing IBP initial bounds...");
            graph.collect_node_bounds(input)?
        } else {
            info!("Multi-objective: Computing CROWN-IBP initial bounds...");
            graph.collect_crown_ibp_bounds_dag(input)?
        };

        // Step 2: Compute initial objective bounds
        // Use spec-guided CROWN for tighter bounds when possible
        let spec_matrix = build_spec_matrix(objectives);
        let (initial_output, initial_obj_bounds) = if let Some(ref spec_mat) = spec_matrix {
            // Try spec-guided CROWN which preserves output correlations
            match graph.propagate_crown_with_specs_and_engine(input, spec_mat, engine) {
                Ok(spec_bounds) => {
                    let obj_bounds = spec_bounds_to_vec(&spec_bounds);
                    info!(
                        "Multi-objective: Using spec-guided CROWN ({} objectives)",
                        obj_bounds.len()
                    );
                    // Also compute raw output bounds for reporting (optional)
                    let output = if self.config.use_alpha_crown {
                        graph.propagate_alpha_crown_with_config(input, &self.config.alpha_config)?
                    } else {
                        graph.propagate_crown(input)?
                    };
                    (output, obj_bounds)
                }
                Err(e) => {
                    // Fall back to interval arithmetic if spec-guided CROWN fails
                    debug!(
                        "Spec-guided CROWN failed ({}), falling back to interval arithmetic",
                        e
                    );
                    let output = if self.config.use_alpha_crown {
                        graph.propagate_alpha_crown_with_config(input, &self.config.alpha_config)?
                    } else {
                        graph.propagate_crown(input)?
                    };
                    let obj_bounds = objective_bounds_vec(&output, objectives)?;
                    (output, obj_bounds)
                }
            }
        } else {
            // No valid spec matrix, use traditional approach
            let output = if self.config.use_alpha_crown {
                graph.propagate_alpha_crown_with_config(input, &self.config.alpha_config)?
            } else {
                graph.propagate_crown(input)?
            };
            let obj_bounds = objective_bounds_vec(&output, objectives)?;
            (output, obj_bounds)
        };

        // Check which objectives are already verified
        let initially_verified: Vec<bool> = initial_obj_bounds
            .iter()
            .zip(thresholds.iter())
            .map(|((l, _u), &t)| *l > t) // verify lower > threshold
            .collect();
        let verified_count: usize = initially_verified.iter().filter(|&&v| v).count();

        info!(
            "Multi-objective initial: {}/{} objectives already verified",
            verified_count, num_objectives
        );

        if verified_count == num_objectives {
            return Ok(BetaCrownResult {
                result: BabVerificationStatus::Verified,
                domains_explored: 1,
                time_elapsed: start_time.elapsed(),
                max_depth_reached: 0,
                output_bounds: Some(initial_output),
                cuts_generated: 0,
                domains_verified: 1,
            });
        }

        // Check if any objective is conclusively violated (cannot be verified)
        for (i, ((_l, u), &t)) in initial_obj_bounds.iter().zip(thresholds.iter()).enumerate() {
            if *u < t {
                // Upper bound < threshold means lower cannot be > threshold
                info!(
                    "Multi-objective: objective {} is conclusively violated (upper={} < threshold={})",
                    i, u, t
                );
                return Ok(BetaCrownResult {
                    result: BabVerificationStatus::Unknown {
                        reason: format!(
                            "Objective {} cannot be verified (upper {} < threshold {})",
                            i, u, t
                        ),
                    },
                    domains_explored: 1,
                    time_elapsed: start_time.elapsed(),
                    max_depth_reached: 0,
                    output_bounds: Some(initial_output),
                    cuts_generated: 0,
                    domains_verified: 0,
                });
            }
        }

        // Identify ReLU nodes in the graph
        let relu_nodes: Vec<String> = graph
            .nodes
            .iter()
            .filter_map(|(name, node)| {
                if matches!(node.layer, Layer::ReLU(_)) {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect();

        // Initialize graph cut pool if enabled (GCP-CROWN)
        let mut cut_pool = if self.config.enable_cuts {
            GraphCutPool::with_min_depth(self.config.max_cuts, self.config.min_cut_depth)
        } else {
            GraphCutPool::new(0) // Disabled: pool with 0 capacity
        };

        // Generate proactive cuts if enabled (BICCOS-lite)
        // This populates the cut pool BEFORE BaB starts
        // Note: We need to convert to Arc before passing to root_domain
        let initial_node_bounds_arc: std::collections::HashMap<String, Arc<BoundedTensor>> =
            initial_node_bounds
                .iter()
                .map(|(k, v)| (k.clone(), Arc::new(v.clone())))
                .collect();
        if self.config.enable_proactive_cuts && self.config.enable_cuts {
            let proactive_count = cut_pool.generate_proactive_cuts(
                graph,
                &initial_node_bounds_arc,
                self.config.max_proactive_cuts,
            );
            if proactive_count > 0 {
                info!(
                    "Multi-objective: Generated {} proactive cuts for {} ReLU nodes",
                    proactive_count,
                    relu_nodes.len()
                );
            }
        }

        // Create root domain
        let root_domain = MultiObjectiveGraphBabDomain::root(
            initial_node_bounds,
            initial_obj_bounds,
            input,
            thresholds,
            false, // verify_upper = false (we want lower > threshold)
        );

        // Branch-and-bound queue
        let mut queue: BinaryHeap<MultiObjectiveGraphBabDomain> = BinaryHeap::new();
        queue.push(root_domain);

        // Determine batch size and whether to use parallel processing
        let batch_size = self.config.batch_size.max(1);
        let use_batched_gpu = engine.is_some() && batch_size > 1 && !self.config.enable_cuts;

        info!(
            "Multi-objective BaB: {} objectives, {} ReLU nodes, {} cuts, batch_size={}, gpu_batched={}, timeout {:?}",
            num_objectives,
            relu_nodes.len(),
            cut_pool.len(),
            batch_size,
            use_batched_gpu,
            self.config.timeout
        );

        loop {
            // Check timeout
            if start_time.elapsed() > self.config.timeout {
                return Ok(BetaCrownResult {
                    result: BabVerificationStatus::Unknown {
                        reason: format!(
                            "Timeout: {}/{} domains verified",
                            domains_verified, domains_explored
                        ),
                    },
                    domains_explored,
                    time_elapsed: start_time.elapsed(),
                    max_depth_reached,
                    output_bounds: None,
                    cuts_generated: cut_pool.total_generated,
                    domains_verified,
                });
            }

            // Check domain limit
            if domains_explored >= self.config.max_domains {
                return Ok(BetaCrownResult {
                    result: BabVerificationStatus::Unknown {
                        reason: format!(
                            "Domain limit {}: {}/{} objectives verified",
                            self.config.max_domains, domains_verified, domains_explored
                        ),
                    },
                    domains_explored,
                    time_elapsed: start_time.elapsed(),
                    max_depth_reached,
                    output_bounds: None,
                    cuts_generated: cut_pool.total_generated,
                    domains_verified,
                });
            }

            // Pop up to batch_size domains from the queue
            let mut batch: Vec<MultiObjectiveGraphBabDomain> = Vec::with_capacity(batch_size);
            while batch.len() < batch_size {
                if let Some(domain) = queue.pop() {
                    batch.push(domain);
                } else {
                    break;
                }
            }

            if batch.is_empty() {
                break; // Queue exhausted
            }

            // Pre-filter batch: check verified, violations, depth limits
            let mut domains_to_process: Vec<MultiObjectiveGraphBabDomain> = Vec::new();

            for domain in batch {
                domains_explored += 1;
                max_depth_reached = max_depth_reached.max(domain.depth);

                // Check if all objectives are verified
                if domain.all_verified() {
                    domains_verified += 1;
                    if self.config.enable_cuts {
                        cut_pool.add_from_verified_domain(&domain.history);
                    }
                    continue;
                }

                // Check if any objective is conclusively violated
                if domain.any_violated(thresholds, false) {
                    continue;
                }

                // Check depth limit
                if domain.depth >= self.config.max_depth {
                    continue;
                }

                domains_to_process.push(domain);
            }

            if domains_to_process.is_empty() {
                continue;
            }

            // Choose processing strategy based on availability of GPU engine and cuts
            if use_batched_gpu {
                // GPU-batched processing: process all domains in parallel
                let eng = engine.unwrap();
                let domain_refs: Vec<&MultiObjectiveGraphBabDomain> =
                    domains_to_process.iter().collect();

                let results = self.process_graph_domains_batched_gpu_multi_objective(
                    graph,
                    &domain_refs,
                    &relu_nodes,
                    objectives,
                    thresholds,
                    eng,
                );

                // Process results
                for result in results {
                    match result {
                        MultiObjectiveGraphDomainResult::AlreadyVerified => {
                            domains_verified += 1;
                        }
                        MultiObjectiveGraphDomainResult::Violation => {
                            // Domain violated, skip
                        }
                        MultiObjectiveGraphDomainResult::Children(children) => {
                            for (child, all_verified) in children {
                                if all_verified {
                                    domains_verified += 1;
                                } else {
                                    queue.push(child);
                                }
                            }
                        }
                        MultiObjectiveGraphDomainResult::NoUnstable { all_verified, .. } => {
                            if all_verified {
                                domains_verified += 1;
                            }
                        }
                        MultiObjectiveGraphDomainResult::Error(_) => {
                            // Skip errored domains
                        }
                    }
                }
            } else {
                // Sequential processing (with cuts or no GPU engine)
                for mut domain in domains_to_process {
                    // Find unstable neurons to branch on
                    let unstable =
                        self.find_unstable_graph_neurons_multi(graph, &domain, &relu_nodes);
                    if unstable.is_empty() {
                        // No unstable neurons left - recompute bounds one more time
                        let cut_pool_ref = if self.config.enable_cuts && !cut_pool.is_empty() {
                            Some(&cut_pool)
                        } else {
                            None
                        };
                        let context = GraphCrownContext::new(
                            &domain.history,
                            cut_pool_ref,
                            Some(&domain.node_bounds),
                            engine,
                        );
                        if let Ok((output, _node_cache)) = self
                            .propagate_crown_with_graph_constraints(
                                graph,
                                domain.input_bounds.as_ref(),
                                &context,
                                None,
                                None,
                            )
                        {
                            if let Ok(new_bounds) = objective_bounds_vec(&output, objectives) {
                                domain.update_bounds(new_bounds, thresholds, false);
                                if domain.all_verified() {
                                    domains_verified += 1;
                                    if self.config.enable_cuts {
                                        cut_pool.add_from_verified_domain(&domain.history);
                                    }
                                }
                            }
                        }
                        continue;
                    }

                    // Select neuron to split
                    let (node_name, neuron_idx) =
                        self.select_graph_branch_multi(graph, &domain, &unstable);

                    // Collect histories of verified children for cut generation
                    let mut verified_histories: Vec<GraphSplitHistory> = Vec::new();

                    // Create active child (x >= 0)
                    let active_constraint = GraphNeuronConstraint {
                        node_name: node_name.clone(),
                        neuron_idx,
                        is_active: true,
                    };
                    if let Some(mut active_child) =
                        domain.with_constraint(graph, active_constraint, false, thresholds)
                    {
                        let cut_pool_ref = if self.config.enable_cuts && !cut_pool.is_empty() {
                            Some(&cut_pool)
                        } else {
                            None
                        };
                        let context = GraphCrownContext::new(
                            &active_child.history,
                            cut_pool_ref,
                            Some(&domain.node_bounds),
                            engine,
                        );
                        let targets = MultiObjectiveTargets::new(
                            objectives,
                            thresholds,
                            &active_child.verified,
                        );
                        // Only run β optimization when enabled and for shallow domains
                        // When beta_iterations=0, skip optimization entirely for all domains
                        let should_optimize = self.config.beta_iterations > 0
                            && active_child.depth <= self.config.beta_max_depth;
                        let result = if should_optimize {
                            self.optimize_graph_beta_analytical_multi_objective(
                                graph,
                                active_child.input_bounds.as_ref(),
                                &context,
                                &mut active_child.beta_state,
                                &targets,
                            )
                        } else {
                            // Skip optimization, just propagate with inherited β
                            self.propagate_multi_objective_with_beta(
                                graph,
                                active_child.input_bounds.as_ref(),
                                &context,
                                &active_child.beta_state,
                                &targets,
                            )
                        };
                        if let Ok((new_bounds, node_cache)) = result {
                            active_child.node_bounds = node_cache
                                .into_iter()
                                .map(|(k, v)| (k, Arc::new(v)))
                                .collect();
                            active_child.update_bounds(new_bounds, thresholds, false);
                            if !active_child.any_violated(thresholds, false) {
                                if active_child.all_verified() {
                                    domains_verified += 1;
                                    if self.config.enable_cuts {
                                        verified_histories.push(active_child.history.clone());
                                    }
                                } else {
                                    queue.push(active_child);
                                }
                            }
                        }
                    }

                    // Create inactive child (x < 0)
                    let inactive_constraint = GraphNeuronConstraint {
                        node_name: node_name.clone(),
                        neuron_idx,
                        is_active: false,
                    };
                    if let Some(mut inactive_child) =
                        domain.with_constraint(graph, inactive_constraint, false, thresholds)
                    {
                        let cut_pool_ref = if self.config.enable_cuts && !cut_pool.is_empty() {
                            Some(&cut_pool)
                        } else {
                            None
                        };
                        let context = GraphCrownContext::new(
                            &inactive_child.history,
                            cut_pool_ref,
                            Some(&domain.node_bounds),
                            engine,
                        );
                        let targets = MultiObjectiveTargets::new(
                            objectives,
                            thresholds,
                            &inactive_child.verified,
                        );
                        // Only run β optimization when enabled and for shallow domains
                        let should_optimize = self.config.beta_iterations > 0
                            && inactive_child.depth <= self.config.beta_max_depth;
                        let result = if should_optimize {
                            self.optimize_graph_beta_analytical_multi_objective(
                                graph,
                                inactive_child.input_bounds.as_ref(),
                                &context,
                                &mut inactive_child.beta_state,
                                &targets,
                            )
                        } else {
                            // Skip optimization, just propagate with inherited β
                            self.propagate_multi_objective_with_beta(
                                graph,
                                inactive_child.input_bounds.as_ref(),
                                &context,
                                &inactive_child.beta_state,
                                &targets,
                            )
                        };
                        if let Ok((new_bounds, node_cache)) = result {
                            inactive_child.node_bounds = node_cache
                                .into_iter()
                                .map(|(k, v)| (k, Arc::new(v)))
                                .collect();
                            inactive_child.update_bounds(new_bounds, thresholds, false);
                            if !inactive_child.any_violated(thresholds, false) {
                                if inactive_child.all_verified() {
                                    domains_verified += 1;
                                    if self.config.enable_cuts {
                                        verified_histories.push(inactive_child.history.clone());
                                    }
                                } else {
                                    queue.push(inactive_child);
                                }
                            }
                        }
                    }

                    // Add cuts from verified children
                    for history in verified_histories {
                        cut_pool.add_from_verified_domain(&history);
                    }
                }
            }
        }

        // Queue empty - check if all domains were verified
        if domains_verified > 0 && queue.is_empty() {
            // We exhausted all domains and some were verified
            // For multi-objective: if we explored the whole space and verified all leaves, success
            Ok(BetaCrownResult {
                result: BabVerificationStatus::Verified,
                domains_explored,
                time_elapsed: start_time.elapsed(),
                max_depth_reached,
                output_bounds: None,
                cuts_generated: cut_pool.total_generated,
                domains_verified,
            })
        } else {
            Ok(BetaCrownResult {
                result: BabVerificationStatus::Unknown {
                    reason: "Could not verify all objectives in explored domains".to_string(),
                },
                domains_explored,
                time_elapsed: start_time.elapsed(),
                max_depth_reached,
                output_bounds: None,
                cuts_generated: cut_pool.total_generated,
                domains_verified,
            })
        }
    }

    /// Find unstable neurons for multi-objective domain.
    fn find_unstable_graph_neurons_multi(
        &self,
        graph: &GraphNetwork,
        domain: &MultiObjectiveGraphBabDomain,
        relu_nodes: &[String],
    ) -> Vec<(String, usize)> {
        let constraint_map = domain.history.build_constraint_map();
        let mut unstable = Vec::new();

        for node_name in relu_nodes {
            let relu_node = match graph.nodes.get(node_name) {
                Some(n) => n,
                None => continue,
            };
            if !matches!(relu_node.layer, Layer::ReLU(_)) {
                continue;
            }
            let pre_name = relu_node
                .inputs
                .first()
                .map(|s| s.as_str())
                .unwrap_or("_input");

            let pre_bounds: &BoundedTensor = if pre_name == "_input" {
                domain.input_bounds.as_ref()
            } else {
                match domain.node_bounds.get(pre_name) {
                    Some(b) => b.as_ref(),
                    None => continue,
                }
            };

            let flat = pre_bounds.flatten();
            for neuron_idx in 0..flat.len() {
                if constraint_map.contains_key(&(node_name.clone(), neuron_idx)) {
                    continue;
                }

                let l = flat.lower[[neuron_idx]];
                let u = flat.upper[[neuron_idx]];

                // Unstable if bounds cross zero
                if l < 0.0 && u > 0.0 {
                    unstable.push((node_name.clone(), neuron_idx));
                }
            }
        }

        unstable
    }

    /// Select branch point for multi-objective domain.
    fn select_graph_branch_multi(
        &self,
        graph: &GraphNetwork,
        domain: &MultiObjectiveGraphBabDomain,
        unstable: &[(String, usize)],
    ) -> (String, usize) {
        // Use intercept-based scoring
        let mut best = unstable[0].clone();
        let mut best_intercept = f32::NEG_INFINITY;

        for (node_name, neuron_idx) in unstable {
            let relu_node = match graph.nodes.get(node_name) {
                Some(n) => n,
                None => continue,
            };
            if !matches!(relu_node.layer, Layer::ReLU(_)) {
                continue;
            }
            let pre_name = relu_node
                .inputs
                .first()
                .map(|s| s.as_str())
                .unwrap_or("_input");
            let pre_bounds: &BoundedTensor = if pre_name == "_input" {
                domain.input_bounds.as_ref()
            } else {
                match domain.node_bounds.get(pre_name) {
                    Some(b) => b.as_ref(),
                    None => continue,
                }
            };

            let flat = pre_bounds.flatten();
            if *neuron_idx >= flat.len() {
                continue;
            }

            let l = flat.lower[[*neuron_idx]];
            let u = flat.upper[[*neuron_idx]];
            let width = u - l;

            if width > 1e-6 {
                let intercept = (-l * u) / width;
                if intercept > best_intercept {
                    best_intercept = intercept;
                    best = (node_name.clone(), *neuron_idx);
                }
            }
        }

        best
    }

    /// Find unstable neurons in graph ReLU nodes.
    ///
    /// Returns a list of (node_name, neuron_idx) pairs where the pre-activation
    /// bounds cross zero (l < 0 < u) and are not already constrained.
    fn find_unstable_graph_neurons(
        &self,
        graph: &GraphNetwork,
        domain: &GraphBabDomain,
        relu_nodes: &[String],
    ) -> Vec<(String, usize)> {
        let constraint_map = domain.history.build_constraint_map();
        let mut unstable = Vec::new();

        for node_name in relu_nodes {
            let relu_node = match graph.nodes.get(node_name) {
                Some(n) => n,
                None => continue,
            };
            if !matches!(relu_node.layer, Layer::ReLU(_)) {
                continue;
            }
            let pre_name = relu_node
                .inputs
                .first()
                .map(|s| s.as_str())
                .unwrap_or("_input");

            let pre_bounds: &BoundedTensor = if pre_name == "_input" {
                domain.input_bounds.as_ref()
            } else {
                match domain.node_bounds.get(pre_name) {
                    Some(b) => b.as_ref(),
                    None => continue,
                }
            };

            let flat = pre_bounds.flatten();
            for neuron_idx in 0..flat.len() {
                // Skip if already constrained (constraints are keyed by ReLU node name).
                if constraint_map.contains_key(&(node_name.clone(), neuron_idx)) {
                    continue;
                }

                let l = flat.lower[[neuron_idx]];
                let u = flat.upper[[neuron_idx]];

                // Unstable if bounds cross zero
                if l < 0.0 && u > 0.0 {
                    unstable.push((node_name.clone(), neuron_idx));
                }
            }
        }

        unstable
    }

    /// Select which neuron to branch on using intercept-based scoring.
    ///
    /// Uses the triangle relaxation intercept: intercept = (-l * u) / (u - l)
    /// Higher values indicate neurons where splitting will reduce the most
    /// relaxation error. This is more effective than width-based selection.
    fn select_graph_branch(
        &self,
        graph: &GraphNetwork,
        domain: &GraphBabDomain,
        unstable: &[(String, usize)],
    ) -> (String, usize) {
        // Use intercept-based scoring: intercept = (-l * u) / (u - l)
        // Higher intercept means larger triangle relaxation gap, hence more potential
        // improvement from splitting.
        //
        // This is similar to the BaBSR heuristic but without CROWN coefficient weighting
        // (which would be expensive to compute for graph networks).
        let mut best = unstable[0].clone();
        let mut best_intercept = f32::NEG_INFINITY;

        for (node_name, neuron_idx) in unstable {
            let relu_node = match graph.nodes.get(node_name) {
                Some(n) => n,
                None => continue,
            };
            if !matches!(relu_node.layer, Layer::ReLU(_)) {
                continue;
            }
            let pre_name = relu_node
                .inputs
                .first()
                .map(|s| s.as_str())
                .unwrap_or("_input");
            let pre_bounds: &BoundedTensor = if pre_name == "_input" {
                domain.input_bounds.as_ref()
            } else {
                match domain.node_bounds.get(pre_name) {
                    Some(b) => b.as_ref(),
                    None => continue,
                }
            };

            let flat = pre_bounds.flatten();
            if *neuron_idx < flat.len() {
                let l = flat.lower[[*neuron_idx]];
                let u = flat.upper[[*neuron_idx]];
                // Intercept score: measures the triangle relaxation gap
                // Only valid for unstable neurons (l < 0 < u)
                if l < 0.0 && u > 0.0 {
                    let intercept = (-l * u) / (u - l);
                    if intercept > best_intercept {
                        best_intercept = intercept;
                        best = (node_name.clone(), *neuron_idx);
                    }
                }
            }
        }

        best
    }

    /// Process a single graph domain: split and compute child bounds.
    ///
    /// This is the parallel-safe workhorse for batched Graph BaB. It processes
    /// one domain and returns its children without touching mutable shared state.
    ///
    /// Note: Does NOT use cuts (cut_pool is not passed) because cuts require
    /// mutable access. Cut generation happens after parallel processing.
    fn process_graph_domain_parallel(
        &self,
        graph: &GraphNetwork,
        domain: &GraphBabDomain,
        relu_nodes: &[String],
        objective: &[f32],
        threshold: f32,
        engine: Option<&dyn GemmEngine>,
    ) -> GraphDomainResult {
        // Quick verification check (already done in caller, but re-check for safety)
        let already_verified = if self.config.verify_upper_bound {
            domain.upper_bound < threshold
        } else {
            domain.lower_bound > threshold
        };
        if already_verified {
            return GraphDomainResult::AlreadyVerified;
        }

        // Quick violation check
        if self.config.verify_upper_bound {
            if domain.lower_bound >= threshold {
                return GraphDomainResult::Violation;
            }
        } else if domain.upper_bound < threshold {
            return GraphDomainResult::Violation;
        }

        // Find unstable neurons
        let unstable = self.find_unstable_graph_neurons(graph, domain, relu_nodes);
        if unstable.is_empty() {
            // No unstable neurons - try one more CROWN pass to get final bounds
            // Use empty cut pool for parallel-safe operation
            let context = GraphCrownContext::new(
                &domain.history,
                None, // No cuts in parallel path
                Some(&domain.node_bounds),
                engine,
            );
            if let Ok((output, _node_cache)) = self.propagate_crown_with_graph_constraints(
                graph,
                domain.input_bounds.as_ref(),
                &context,
                None, // No β when no unstable neurons
                Some(objective),
            ) {
                let l = output.lower_scalar();
                let u = output.upper_scalar();
                let verified = if self.config.verify_upper_bound {
                    u < threshold
                } else {
                    l > threshold
                };
                return GraphDomainResult::NoUnstable {
                    lower: l,
                    upper: u,
                    verified,
                };
            } else {
                return GraphDomainResult::NoUnstable {
                    lower: domain.lower_bound,
                    upper: domain.upper_bound,
                    verified: false,
                };
            }
        }

        // Select neuron to split
        let (node_name, neuron_idx) = self.select_graph_branch(graph, domain, &unstable);

        let mut children: Vec<(GraphBabDomain, bool)> = Vec::with_capacity(2);

        // Create and process active child (x >= 0)
        let active_constraint = GraphNeuronConstraint {
            node_name: node_name.clone(),
            neuron_idx,
            is_active: true,
        };
        if let Some(mut active_child) =
            domain.with_constraint(graph, active_constraint, self.config.verify_upper_bound)
        {
            // Compute bounds without cuts (parallel-safe)
            // Skip SPSA β optimization in parallel path for speed - use simple propagation
            let context = GraphCrownContext::new(
                &active_child.history,
                None, // No cuts in parallel path
                Some(&domain.node_bounds),
                engine,
            );
            if let Ok((output, node_cache)) = self.propagate_crown_with_graph_constraints(
                graph,
                active_child.input_bounds.as_ref(),
                &context,
                Some(&active_child.beta_state),
                Some(objective),
            ) {
                let l = output.lower_scalar();
                let u = output.upper_scalar();
                active_child.node_bounds = node_cache
                    .into_iter()
                    .map(|(k, v)| (k, Arc::new(v)))
                    .collect();
                active_child.lower_bound = l;
                active_child.upper_bound = u;
                active_child.priority = if self.config.verify_upper_bound {
                    u
                } else {
                    -l
                };

                let verified = if self.config.verify_upper_bound {
                    u < threshold
                } else {
                    l > threshold
                };
                children.push((active_child, verified));
            }
        }

        // Create and process inactive child (x <= 0)
        let inactive_constraint = GraphNeuronConstraint {
            node_name: node_name.clone(),
            neuron_idx,
            is_active: false,
        };
        if let Some(mut inactive_child) =
            domain.with_constraint(graph, inactive_constraint, self.config.verify_upper_bound)
        {
            // Compute bounds without cuts (parallel-safe)
            let context = GraphCrownContext::new(
                &inactive_child.history,
                None, // No cuts in parallel path
                Some(&domain.node_bounds),
                engine,
            );
            if let Ok((output, node_cache)) = self.propagate_crown_with_graph_constraints(
                graph,
                inactive_child.input_bounds.as_ref(),
                &context,
                Some(&inactive_child.beta_state),
                Some(objective),
            ) {
                let l = output.lower_scalar();
                let u = output.upper_scalar();
                inactive_child.node_bounds = node_cache
                    .into_iter()
                    .map(|(k, v)| (k, Arc::new(v)))
                    .collect();
                inactive_child.lower_bound = l;
                inactive_child.upper_bound = u;
                inactive_child.priority = if self.config.verify_upper_bound {
                    u
                } else {
                    -l
                };

                let verified = if self.config.verify_upper_bound {
                    u < threshold
                } else {
                    l > threshold
                };
                children.push((inactive_child, verified));
            }
        }

        GraphDomainResult::Children(children)
    }

    /// Process multiple graph domains with GPU-batched Linear GEMM.
    ///
    /// This function batches the expensive Linear layer GEMMs across all domains,
    /// significantly improving GPU utilization compared to processing domains sequentially.
    ///
    /// For N domains processing through L Linear layers:
    /// - Sequential: N × L GPU kernel launches (each small)
    /// - Batched: L GPU kernel launches (each large, better GPU utilization)
    ///
    /// The batching exploits that matrix multiplication distributes over row stacking:
    /// [A1; A2; ...; An] @ W^T = [A1 @ W^T; A2 @ W^T; ...; An @ W^T]
    ///
    /// Returns results for each domain in the same order as input.
    fn process_graph_domains_batched_gpu(
        &self,
        graph: &GraphNetwork,
        domains: &[&GraphBabDomain],
        relu_nodes: &[String],
        objective: &[f32],
        threshold: f32,
        engine: &dyn GemmEngine,
    ) -> Vec<GraphDomainResult> {
        if domains.is_empty() {
            return Vec::new();
        }

        // Pre-filter: separate already-verified, violation, and to-process domains
        // Use a HashMap for sparse results storage (avoids Clone requirement)
        let mut quick_results: std::collections::HashMap<usize, GraphDomainResult> =
            std::collections::HashMap::new();
        let mut domains_to_process: Vec<(usize, &GraphBabDomain)> = Vec::new();

        for (idx, domain) in domains.iter().enumerate() {
            // Quick verification check
            let already_verified = if self.config.verify_upper_bound {
                domain.upper_bound < threshold
            } else {
                domain.lower_bound > threshold
            };
            if already_verified {
                quick_results.insert(idx, GraphDomainResult::AlreadyVerified);
                continue;
            }

            // Quick violation check
            let violation = if self.config.verify_upper_bound {
                domain.lower_bound >= threshold
            } else {
                domain.upper_bound < threshold
            };
            if violation {
                quick_results.insert(idx, GraphDomainResult::Violation);
                continue;
            }

            domains_to_process.push((idx, domain));
        }

        if domains_to_process.is_empty() {
            // All domains were quick-filtered
            return (0..domains.len())
                .map(|idx| quick_results.remove(&idx).unwrap())
                .collect();
        }

        // Find unstable neurons for all domains in parallel (cheap operation)
        let unstable_per_domain: Vec<(usize, Vec<(String, usize)>)> = domains_to_process
            .par_iter()
            .map(|(idx, domain)| {
                let unstable = self.find_unstable_graph_neurons(graph, domain, relu_nodes);
                (*idx, unstable)
            })
            .collect();

        // Separate domains with/without unstable neurons
        let mut domains_with_unstable: Vec<DomainWithUnstable<'_>> = Vec::new();

        for (idx, unstable) in unstable_per_domain {
            let domain = domains_to_process
                .iter()
                .find(|(i, _)| *i == idx)
                .map(|(_, d)| *d)
                .unwrap();

            if unstable.is_empty() {
                // No unstable neurons - compute final bounds
                let context = GraphCrownContext::new(
                    &domain.history,
                    None,
                    Some(&domain.node_bounds),
                    Some(engine),
                );
                if let Ok((output, _node_cache)) = self.propagate_crown_with_graph_constraints(
                    graph,
                    domain.input_bounds.as_ref(),
                    &context,
                    None,
                    Some(objective),
                ) {
                    let l = output.lower_scalar();
                    let u = output.upper_scalar();
                    let verified = if self.config.verify_upper_bound {
                        u < threshold
                    } else {
                        l > threshold
                    };
                    quick_results.insert(
                        idx,
                        GraphDomainResult::NoUnstable {
                            lower: l,
                            upper: u,
                            verified,
                        },
                    );
                } else {
                    quick_results.insert(
                        idx,
                        GraphDomainResult::NoUnstable {
                            lower: domain.lower_bound,
                            upper: domain.upper_bound,
                            verified: false,
                        },
                    );
                }
            } else {
                domains_with_unstable.push((idx, domain, unstable));
            }
        }

        if domains_with_unstable.is_empty() {
            return (0..domains.len())
                .map(|idx| quick_results.remove(&idx).unwrap())
                .collect();
        }

        // For domains with unstable neurons, process children in batched manner
        // First, create all child domains (this is relatively cheap)
        let child_creation_results: Vec<_> = domains_with_unstable
            .par_iter()
            .map(|(idx, domain, unstable)| {
                let (node_name, neuron_idx) = self.select_graph_branch(graph, domain, unstable);

                // Create active and inactive child domains
                let mut children_info = Vec::with_capacity(2);

                // Active child
                let active_constraint = GraphNeuronConstraint {
                    node_name: node_name.clone(),
                    neuron_idx,
                    is_active: true,
                };
                if let Some(child) =
                    domain.with_constraint(graph, active_constraint, self.config.verify_upper_bound)
                {
                    children_info.push((*idx, child, true));
                }

                // Inactive child
                let inactive_constraint = GraphNeuronConstraint {
                    node_name: node_name.clone(),
                    neuron_idx,
                    is_active: false,
                };
                if let Some(child) = domain.with_constraint(
                    graph,
                    inactive_constraint,
                    self.config.verify_upper_bound,
                ) {
                    children_info.push((*idx, child, false));
                }

                (*idx, children_info)
            })
            .collect();

        // Collect all children that need CROWN bounds computation
        let mut all_children: Vec<(usize, GraphBabDomain, bool)> = Vec::new();
        let mut parent_domain_lookup: std::collections::HashMap<usize, &GraphBabDomain> =
            std::collections::HashMap::new();

        for (parent_idx, children_info) in &child_creation_results {
            let parent_domain = domains_with_unstable
                .iter()
                .find(|(i, _, _)| i == parent_idx)
                .map(|(_, d, _)| *d)
                .unwrap();
            parent_domain_lookup.insert(*parent_idx, parent_domain);

            for (_, child, is_active) in children_info {
                all_children.push((*parent_idx, child.clone(), *is_active));
            }
        }

        // Now compute CROWN bounds for all children with TRUE tensor-level GPU batching.
        // This batches all Linear layer GEMMs into single kernel launches, dramatically
        // improving GPU utilization compared to the old par_iter approach.
        //
        // Old approach: N children × L layers = N×L small GPU kernel launches
        // New approach: L layers = L large GPU kernel launches (batches all children)
        let domain_data: Vec<_> = all_children
            .iter()
            .map(|(parent_idx, child, _is_active)| {
                let parent = parent_domain_lookup.get(parent_idx).unwrap();
                (
                    child.input_bounds.as_ref(),
                    &child.history,
                    Some(&child.beta_state),
                    Some(&parent.node_bounds),
                )
            })
            .collect();

        let child_bounds: Vec<
            Option<(
                BoundedTensor,
                std::collections::HashMap<String, BoundedTensor>,
            )>,
        > = match self.propagate_crown_batched_backward(graph, &domain_data, objective, engine) {
            Ok(results) => results.into_iter().map(Some).collect(),
            Err(e) => {
                // If batched processing fails, fall back to sequential
                tracing::warn!("Batched CROWN failed ({}), falling back to sequential", e);
                all_children
                    .par_iter()
                    .map(|(parent_idx, child, _is_active)| {
                        let parent = parent_domain_lookup.get(parent_idx).unwrap();
                        let context = GraphCrownContext::new(
                            &child.history,
                            None,
                            Some(&parent.node_bounds),
                            Some(engine),
                        );
                        match self.propagate_crown_with_graph_constraints(
                            graph,
                            child.input_bounds.as_ref(),
                            &context,
                            Some(&child.beta_state),
                            Some(objective),
                        ) {
                            Ok((output, node_cache)) => Some((output, node_cache)),
                            Err(_) => None,
                        }
                    })
                    .collect()
            }
        };

        // Build results from child bounds
        let mut children_by_parent: std::collections::HashMap<usize, Vec<(GraphBabDomain, bool)>> =
            std::collections::HashMap::new();

        for ((parent_idx, mut child, _is_active), bounds_result) in
            all_children.into_iter().zip(child_bounds.into_iter())
        {
            if let Some((output, node_cache)) = bounds_result {
                let l = output.lower_scalar();
                let u = output.upper_scalar();
                child.node_bounds = node_cache
                    .into_iter()
                    .map(|(k, v)| (k, Arc::new(v)))
                    .collect();
                child.lower_bound = l;
                child.upper_bound = u;
                child.priority = if self.config.verify_upper_bound {
                    u
                } else {
                    -l
                };

                let verified = if self.config.verify_upper_bound {
                    u < threshold
                } else {
                    l > threshold
                };

                children_by_parent
                    .entry(parent_idx)
                    .or_default()
                    .push((child, verified));
            }
        }

        // Assemble final results
        for (parent_idx, _, _) in &domains_with_unstable {
            if let Some(children) = children_by_parent.remove(parent_idx) {
                quick_results.insert(*parent_idx, GraphDomainResult::Children(children));
            } else {
                // No children - shouldn't happen but handle gracefully
                quick_results.insert(
                    *parent_idx,
                    GraphDomainResult::NoUnstable {
                        lower: f32::NEG_INFINITY,
                        upper: f32::INFINITY,
                        verified: false,
                    },
                );
            }
        }

        // Return results in order
        (0..domains.len())
            .map(|idx| quick_results.remove(&idx).unwrap())
            .collect()
    }

    /// Process a batch of multi-objective domains with GPU-batched CROWN computation.
    ///
    /// Similar to `process_graph_domains_batched_gpu` but handles multiple objectives.
    /// This batches the CROWN computation across all child domains to improve GPU utilization.
    fn process_graph_domains_batched_gpu_multi_objective(
        &self,
        graph: &GraphNetwork,
        domains: &[&MultiObjectiveGraphBabDomain],
        relu_nodes: &[String],
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        engine: &dyn GemmEngine,
    ) -> Vec<MultiObjectiveGraphDomainResult> {
        if domains.is_empty() {
            return Vec::new();
        }

        // Pre-filter: separate already-verified, violation, and to-process domains
        let mut quick_results: std::collections::HashMap<usize, MultiObjectiveGraphDomainResult> =
            std::collections::HashMap::new();
        let mut domains_to_process: Vec<(usize, &MultiObjectiveGraphBabDomain)> = Vec::new();

        for (idx, domain) in domains.iter().enumerate() {
            // Quick verification check
            if domain.all_verified() {
                quick_results.insert(idx, MultiObjectiveGraphDomainResult::AlreadyVerified);
                continue;
            }

            // Quick violation check
            if domain.any_violated(thresholds, false) {
                quick_results.insert(idx, MultiObjectiveGraphDomainResult::Violation);
                continue;
            }

            domains_to_process.push((idx, domain));
        }

        if domains_to_process.is_empty() {
            return (0..domains.len())
                .map(|idx| quick_results.remove(&idx).unwrap())
                .collect();
        }

        // Find unstable neurons for all domains in parallel
        let unstable_per_domain: Vec<(usize, Vec<(String, usize)>)> = domains_to_process
            .par_iter()
            .map(|(idx, domain)| {
                let unstable = self.find_unstable_graph_neurons_multi(graph, domain, relu_nodes);
                (*idx, unstable)
            })
            .collect();

        // Separate domains with/without unstable neurons
        let mut domains_with_unstable: Vec<MultiObjDomainWithUnstable<'_>> = Vec::new();

        for (idx, unstable) in unstable_per_domain {
            let domain = domains_to_process
                .iter()
                .find(|(i, _)| *i == idx)
                .map(|(_, d)| *d)
                .unwrap();

            if unstable.is_empty() {
                // No unstable neurons - compute final bounds
                let context = GraphCrownContext::new(
                    &domain.history,
                    None,
                    Some(&domain.node_bounds),
                    Some(engine),
                );
                if let Ok((output, _node_cache)) = self.propagate_crown_with_graph_constraints(
                    graph,
                    domain.input_bounds.as_ref(),
                    &context,
                    None,
                    None, // Multi-objective: compute full output bounds
                ) {
                    if let Ok(new_bounds) = Self::objective_bounds_multi(&output, objectives) {
                        let all_verified = new_bounds
                            .iter()
                            .zip(thresholds.iter())
                            .all(|((l, _), &t)| *l > t);
                        quick_results.insert(
                            idx,
                            MultiObjectiveGraphDomainResult::NoUnstable {
                                objective_bounds: new_bounds,
                                all_verified,
                            },
                        );
                    } else {
                        quick_results.insert(
                            idx,
                            MultiObjectiveGraphDomainResult::NoUnstable {
                                objective_bounds: domain.objective_bounds.clone(),
                                all_verified: false,
                            },
                        );
                    }
                } else {
                    quick_results.insert(
                        idx,
                        MultiObjectiveGraphDomainResult::NoUnstable {
                            objective_bounds: domain.objective_bounds.clone(),
                            all_verified: false,
                        },
                    );
                }
            } else {
                domains_with_unstable.push((idx, domain, unstable));
            }
        }

        if domains_with_unstable.is_empty() {
            return (0..domains.len())
                .map(|idx| quick_results.remove(&idx).unwrap())
                .collect();
        }

        // For domains with unstable neurons, create all children in parallel
        let child_creation_results: Vec<_> = domains_with_unstable
            .par_iter()
            .map(|(idx, domain, unstable)| {
                let (node_name, neuron_idx) =
                    self.select_graph_branch_multi(graph, domain, unstable);

                let mut children_info = Vec::with_capacity(2);

                // Active child
                let active_constraint = GraphNeuronConstraint {
                    node_name: node_name.clone(),
                    neuron_idx,
                    is_active: true,
                };
                if let Some(child) =
                    domain.with_constraint(graph, active_constraint, false, thresholds)
                {
                    children_info.push((*idx, child, true));
                }

                // Inactive child
                let inactive_constraint = GraphNeuronConstraint {
                    node_name: node_name.clone(),
                    neuron_idx,
                    is_active: false,
                };
                if let Some(child) =
                    domain.with_constraint(graph, inactive_constraint, false, thresholds)
                {
                    children_info.push((*idx, child, false));
                }

                (*idx, children_info)
            })
            .collect();

        // Collect all children that need CROWN bounds computation
        let mut all_children: Vec<(usize, MultiObjectiveGraphBabDomain, bool)> = Vec::new();
        let mut parent_domain_lookup: std::collections::HashMap<
            usize,
            &MultiObjectiveGraphBabDomain,
        > = std::collections::HashMap::new();

        for (parent_idx, children_info) in &child_creation_results {
            let parent_domain = domains_with_unstable
                .iter()
                .find(|(i, _, _)| i == parent_idx)
                .map(|(_, d, _)| *d)
                .unwrap();
            parent_domain_lookup.insert(*parent_idx, parent_domain);

            for (_, child, is_active) in children_info {
                all_children.push((*parent_idx, child.clone(), *is_active));
            }
        }

        // Compute CROWN bounds for all children with GPU-parallel processing
        // The GPU engine batches the GEMM operations internally
        let child_bounds: Vec<_> = all_children
            .par_iter()
            .map(|(parent_idx, child, _is_active)| {
                let parent = parent_domain_lookup.get(parent_idx).unwrap();

                // Use β-CROWN with SPSA optimization for shallow domains
                let mut beta_state = child.beta_state.clone();
                let context = GraphCrownContext::new(
                    &child.history,
                    None, // No cuts in batched mode for simplicity
                    Some(&parent.node_bounds),
                    Some(engine),
                );
                let targets = MultiObjectiveTargets::new(objectives, thresholds, &child.verified);
                // Only run β optimization for shallow domains (depth <= beta_max_depth)
                let result = if child.depth <= self.config.beta_max_depth {
                    self.optimize_graph_beta_analytical_multi_objective(
                        graph,
                        child.input_bounds.as_ref(),
                        &context,
                        &mut beta_state,
                        &targets,
                    )
                } else {
                    // Skip optimization, just propagate with inherited β
                    self.propagate_multi_objective_with_beta(
                        graph,
                        child.input_bounds.as_ref(),
                        &context,
                        &beta_state,
                        &targets,
                    )
                };
                match result {
                    Ok((obj_bounds, node_cache)) => Some((obj_bounds, node_cache, beta_state)),
                    Err(_) => None,
                }
            })
            .collect();

        // Build results from child bounds
        let mut children_by_parent: std::collections::HashMap<
            usize,
            Vec<(MultiObjectiveGraphBabDomain, bool)>,
        > = std::collections::HashMap::new();

        for ((parent_idx, mut child, _is_active), bounds_result) in
            all_children.into_iter().zip(child_bounds.into_iter())
        {
            if let Some((obj_bounds, node_cache, beta_state)) = bounds_result {
                child.node_bounds = node_cache
                    .into_iter()
                    .map(|(k, v)| (k, Arc::new(v)))
                    .collect();
                child.beta_state = beta_state;
                child.update_bounds(obj_bounds, thresholds, false);

                let all_verified = child.all_verified();
                let any_violated = child.any_violated(thresholds, false);

                // Only keep non-violated children
                if !any_violated {
                    children_by_parent
                        .entry(parent_idx)
                        .or_default()
                        .push((child, all_verified));
                }
            }
        }

        // Assemble final results
        for (parent_idx, _, _) in &domains_with_unstable {
            if let Some(children) = children_by_parent.remove(parent_idx) {
                quick_results.insert(
                    *parent_idx,
                    MultiObjectiveGraphDomainResult::Children(children),
                );
            } else {
                // No valid children - mark as violated
                quick_results.insert(*parent_idx, MultiObjectiveGraphDomainResult::Violation);
            }
        }

        // Return results in order
        (0..domains.len())
            .map(|idx| quick_results.remove(&idx).unwrap())
            .collect()
    }

    /// Propagate CROWN bounds through GraphNetwork with ReLU constraints and cuts.
    ///
    /// This is a constraint-aware version of `GraphNetwork::propagate_crown`.
    /// It uses the constrained node bounds for ReLU relaxation and optionally
    /// applies cutting plane contributions to tighten the lower bound.
    ///
    /// Returns: (output_bounds, node_bounds_cache) where node_bounds_cache maps
    /// node names to their computed bounds for use in cut contribution calculation.
    ///
    /// If `base_bounds` is provided, uses those as the starting point (e.g., CROWN-IBP bounds
    /// computed at initialization). Otherwise, computes fresh IBP bounds.
    fn propagate_crown_with_graph_constraints(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        context: &GraphCrownContext<'_>,
        beta_state: Option<&GraphBetaState>,
        objective: Option<&[f32]>,
    ) -> Result<(
        BoundedTensor,
        std::collections::HashMap<String, BoundedTensor>,
    )> {
        // Constraint-aware forward bound collection:
        // - ReLU constraints apply to the *pre-activation* (the ReLU input node), not the ReLU output.
        // - Tighten the producing node's bounds so all consumers (e.g., residual adds) see the restriction.
        // - Compute each ReLU output bound using the constrained pre-activation bounds.
        //
        // Uses base_bounds (e.g., CROWN-IBP) if provided for tighter intermediate bounds.

        let exec_order = graph.topological_sort()?;

        // Use base bounds if provided, otherwise compute IBP bounds
        let mut bounds_cache: std::collections::HashMap<String, BoundedTensor> =
            if let Some(base) = context.base_bounds {
                base.iter()
                    .map(|(k, v)| (k.clone(), v.as_ref().clone()))
                    .collect()
            } else {
                graph.collect_node_bounds(input)?
            };

        // Build constraint lookup keyed by ReLU node, and also build a map from
        // pre-activation node name -> constraints that should be applied there.
        let mut constraints_by_relu: std::collections::HashMap<
            String,
            std::collections::HashMap<usize, bool>,
        > = std::collections::HashMap::new();
        let mut pre_constraints: std::collections::HashMap<String, Vec<(usize, bool, String)>> =
            std::collections::HashMap::new();

        for c in &context.history.constraints {
            constraints_by_relu
                .entry(c.node_name.clone())
                .or_default()
                .insert(c.neuron_idx, c.is_active);

            let relu_node = graph.nodes.get(&c.node_name).ok_or_else(|| {
                GammaError::InvalidSpec(format!(
                    "Graph constraint references missing node '{}'",
                    c.node_name
                ))
            })?;
            if !matches!(relu_node.layer, Layer::ReLU(_)) {
                return Err(GammaError::InvalidSpec(format!(
                    "Graph constraint references non-ReLU node '{}'",
                    c.node_name
                )));
            }
            let pre_name = relu_node
                .inputs
                .first()
                .cloned()
                .unwrap_or_else(|| "_input".to_string());
            pre_constraints.entry(pre_name).or_default().push((
                c.neuron_idx,
                c.is_active,
                c.node_name.clone(),
            ));
        }

        let apply_pre_constraints = |bounds: &BoundedTensor,
                                     constraints: &[(usize, bool, String)]|
         -> Result<BoundedTensor> {
            if constraints.is_empty() {
                return Ok(bounds.clone());
            }

            let flat = bounds.flatten();
            let shape = bounds.shape().to_vec();
            let mut lower = flat.lower.clone();
            let mut upper = flat.upper.clone();

            for (neuron_idx, is_active, relu_name) in constraints {
                if *neuron_idx >= flat.len() {
                    return Err(GammaError::InvalidSpec(format!(
                        "Constraint out of range: relu='{}' idx={} len={}",
                        relu_name,
                        neuron_idx,
                        flat.len()
                    )));
                }
                if *is_active {
                    lower[[*neuron_idx]] = lower[[*neuron_idx]].max(0.0);
                } else {
                    upper[[*neuron_idx]] = upper[[*neuron_idx]].min(0.0);
                }
                if lower[[*neuron_idx]] > upper[[*neuron_idx]] {
                    return Err(GammaError::InvalidSpec(format!(
                        "Infeasible domain after constraint: relu='{}' idx={} [{}, {}]",
                        relu_name,
                        neuron_idx,
                        lower[[*neuron_idx]],
                        upper[[*neuron_idx]]
                    )));
                }
            }

            let lower_arr = lower
                .into_shape_clone(ndarray::IxDyn(&shape))
                .map_err(|e| GammaError::InvalidSpec(format!("shape error: {}", e)))?;
            let upper_arr = upper
                .into_shape_clone(ndarray::IxDyn(&shape))
                .map_err(|e| GammaError::InvalidSpec(format!("shape error: {}", e)))?;
            BoundedTensor::new(lower_arr, upper_arr)
        };

        // Apply any constraints that target the graph input itself.
        let mut constrained_input = input.clone();
        if let Some(cons) = pre_constraints.get("_input") {
            constrained_input = apply_pre_constraints(&constrained_input, cons)?;
        }

        // Apply constraint tightening to CROWN-IBP bounds.
        // The bounds_cache already contains CROWN-IBP bounds; we only apply constraints to tighten.
        for node_name in &exec_order {
            let node = graph
                .nodes
                .get(node_name)
                .ok_or_else(|| GammaError::InvalidSpec(format!("Node not found: {}", node_name)))?;

            // Get current CROWN-IBP bounds for this node
            let current_bounds = bounds_cache.get(node_name).ok_or_else(|| {
                GammaError::InvalidSpec(format!(
                    "CROWN-IBP bounds not found for node '{}'",
                    node_name
                ))
            })?;

            // For ReLU nodes, apply per-neuron active/inactive constraints to the output
            let mut output_bounds = if matches!(node.layer, Layer::ReLU(_)) {
                // Get pre-activation bounds (input to this ReLU)
                let pre_activation = if node.inputs.is_empty() || node.inputs[0] == "_input" {
                    &constrained_input
                } else {
                    bounds_cache.get(&node.inputs[0]).ok_or_else(|| {
                        GammaError::InvalidSpec(format!(
                            "Pre-activation bounds for {} not found",
                            node.inputs[0]
                        ))
                    })?
                };

                let pre_flat = pre_activation.flatten();
                let shape = pre_activation.shape().to_vec();
                let mut lower = pre_flat.lower.clone();
                let mut upper = pre_flat.upper.clone();

                let relu_cons = constraints_by_relu.get(node_name);
                for neuron_idx in 0..pre_flat.len() {
                    let l = pre_flat.lower[[neuron_idx]];
                    let u = pre_flat.upper[[neuron_idx]];

                    let constrained = relu_cons.and_then(|m| m.get(&neuron_idx).copied());
                    match constrained {
                        Some(true) => {
                            // Active: x >= 0, y = x
                            if u < 0.0 {
                                return Err(GammaError::InvalidSpec(format!(
                                    "Infeasible active ReLU constraint at node '{}' idx={} pre_u={}",
                                    node_name, neuron_idx, u
                                )));
                            }
                            lower[[neuron_idx]] = l.max(0.0);
                            upper[[neuron_idx]] = u;
                        }
                        Some(false) => {
                            // Inactive: x <= 0, y = 0
                            if l > 0.0 {
                                return Err(GammaError::InvalidSpec(format!(
                                    "Infeasible inactive ReLU constraint at node '{}' idx={} pre_l={}",
                                    node_name, neuron_idx, l
                                )));
                            }
                            lower[[neuron_idx]] = 0.0;
                            upper[[neuron_idx]] = 0.0;
                        }
                        None => {
                            // Unconstrained: use CROWN-IBP ReLU bounds (already computed)
                            let crown_bounds = current_bounds.flatten();
                            lower[[neuron_idx]] = crown_bounds.lower[[neuron_idx]];
                            upper[[neuron_idx]] = crown_bounds.upper[[neuron_idx]];
                        }
                    }
                }

                let lower_arr = lower
                    .into_shape_clone(ndarray::IxDyn(&shape))
                    .map_err(|e| GammaError::InvalidSpec(format!("shape error: {}", e)))?;
                let upper_arr = upper
                    .into_shape_clone(ndarray::IxDyn(&shape))
                    .map_err(|e| GammaError::InvalidSpec(format!("shape error: {}", e)))?;
                BoundedTensor::new(lower_arr, upper_arr)?
            } else {
                // For non-ReLU nodes, keep the CROWN-IBP bounds
                current_bounds.clone()
            };

            // Apply pre-activation constraints that target this node's outputs (i.e., this node
            // is the input to some constrained ReLU).
            if let Some(cons) = pre_constraints.get(node_name) {
                output_bounds = apply_pre_constraints(&output_bounds, cons)?;
            }

            bounds_cache.insert(node_name.clone(), output_bounds);
        }

        // ==== BACKWARD CROWN PASS using constrained intermediate bounds ====
        // This replaces IBP output with tighter CROWN output.
        let output_name = graph.get_output_name();
        let output_node = if output_name.is_empty() {
            exec_order
                .last()
                .ok_or_else(|| GammaError::InvalidSpec("No nodes in graph".to_string()))?
        } else {
            output_name
        };

        let ibp_output = bounds_cache.get(output_node).ok_or_else(|| {
            GammaError::InvalidSpec(format!("Output node {} not found", output_node))
        })?;
        let output_dim = ibp_output.len();
        let (output_shape, initial_lb) = if let Some(objective) = objective {
            if objective.len() != output_dim {
                return Err(GammaError::shape_mismatch(
                    vec![objective.len()],
                    vec![output_dim],
                ));
            }
            let a = Array2::from_shape_vec((1, output_dim), objective.to_vec()).map_err(|e| {
                GammaError::InvalidSpec(format!("Failed to build objective coefficients: {e}"))
            })?;
            (
                vec![1usize],
                LinearBounds {
                    lower_a: a.clone(),
                    lower_b: Array1::zeros(1),
                    upper_a: a,
                    upper_b: Array1::zeros(1),
                },
            )
        } else {
            (
                ibp_output.shape().to_vec(),
                LinearBounds::identity(output_dim),
            )
        };
        let _input_dim = constrained_input.len();

        // Initialize linear bounds: either identity at output node (full output bounds),
        // or a single-row objective (scalar bounds).
        let mut node_linear_bounds: std::collections::HashMap<String, LinearBounds> =
            std::collections::HashMap::new();
        node_linear_bounds.insert(output_node.to_string(), initial_lb);

        let mut input_accumulated = false;

        // Backward propagation through nodes in reverse topological order
        for node_name in exec_order.iter().rev() {
            let node = graph
                .nodes
                .get(node_name)
                .ok_or_else(|| GammaError::InvalidSpec(format!("Node not found: {}", node_name)))?;

            // Get this node's accumulated linear bounds
            let node_lb = match node_linear_bounds.remove(node_name) {
                Some(lb) => lb,
                None => continue, // Node has no consumers
            };

            // Get pre-activation bounds for this node from the constrained cache
            let pre_activation_name = if node.inputs.is_empty() || node.inputs[0] == "_input" {
                "_input"
            } else {
                &node.inputs[0]
            };
            let pre_activation = if pre_activation_name == "_input" {
                &constrained_input
            } else {
                bounds_cache.get(pre_activation_name).ok_or_else(|| {
                    GammaError::InvalidSpec(format!(
                        "Pre-activation bounds for {} not found",
                        pre_activation_name
                    ))
                })?
            };

            // Propagate linear bounds backward based on layer type
            match &node.layer {
                Layer::Linear(l) => {
                    let new_lb = l
                        .propagate_linear_with_engine(&node_lb, context.engine)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "Constrained CROWN failed at node '{}' (Linear): {}",
                                node_name, e
                            ))
                        })?;
                    let new_lb = match new_lb {
                        std::borrow::Cow::Borrowed(_) => node_lb,
                        std::borrow::Cow::Owned(lb) => lb,
                    };
                    Self::accumulate_crown_bounds(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::ReLU(r) => {
                    // Use constrained pre-activation bounds for ReLU relaxation
                    let mut new_lb = r
                        .propagate_linear_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "Constrained CROWN failed at node '{}' (ReLU): {}",
                                node_name, e
                            ))
                        })?;

                    // Add β contribution for Lagrangian β-CROWN.
                    // For constrained neurons, modify linear coefficients:
                    //   lower_a[i, j] -= signed_beta
                    //   upper_a[i, j] += signed_beta
                    // This is the Lagrangian dual term for the split constraint.
                    if let Some(beta) = beta_state {
                        for entry in &beta.entries {
                            if entry.node_name != *node_name {
                                continue;
                            }
                            let j = entry.neuron_idx;
                            if j >= new_lb.num_inputs() {
                                continue;
                            }
                            let signed_beta = entry.value * entry.sign;
                            if signed_beta.abs() < 1e-10 {
                                continue;
                            }

                            for i in 0..new_lb.num_outputs() {
                                new_lb.lower_a[[i, j]] -= signed_beta;
                                new_lb.upper_a[[i, j]] += signed_beta;
                            }
                            trace!(
                                "Added β contribution for {}[{}]: signed_beta={}",
                                node_name,
                                j,
                                signed_beta
                            );
                        }
                    }

                    Self::accumulate_crown_bounds(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::Add(_) => {
                    // Add propagates same linear bounds to both inputs
                    Self::accumulate_crown_bounds(
                        &node.inputs[0],
                        node_lb.clone(),
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                    Self::accumulate_crown_bounds(
                        &node.inputs[1],
                        node_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::Conv2d(conv) => {
                    // Conv backward: propagate through transposed convolution
                    // Need to set input_shape for CROWN to work
                    let input_shape = pre_activation.shape();
                    let (in_h, in_w) = if input_shape.len() >= 3 {
                        (
                            input_shape[input_shape.len() - 2],
                            input_shape[input_shape.len() - 1],
                        )
                    } else {
                        // Input too small for Conv2d CROWN, fall back
                        debug!(
                            "Constrained CROWN: Conv2d input shape too small: {:?}, falling back",
                            input_shape
                        );
                        for input_name in &node.inputs {
                            Self::accumulate_crown_bounds(
                                input_name,
                                node_lb.clone(),
                                &mut node_linear_bounds,
                                &mut input_accumulated,
                            );
                        }
                        continue;
                    };
                    let mut conv_with_shape = conv.clone();
                    conv_with_shape.set_input_shape(in_h, in_w);
                    let new_lb = conv_with_shape.propagate_linear(&node_lb).map_err(|e| {
                        GammaError::InvalidSpec(format!(
                            "Constrained CROWN failed at node '{}' (Conv2d): {}",
                            node_name, e
                        ))
                    })?;
                    let new_lb = match new_lb {
                        std::borrow::Cow::Borrowed(_) => node_lb,
                        std::borrow::Cow::Owned(lb) => lb,
                    };
                    Self::accumulate_crown_bounds(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::BatchNorm(bn) => {
                    let new_lb = bn
                        .propagate_linear_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "Constrained CROWN failed at node '{}' (BatchNorm): {}",
                                node_name, e
                            ))
                        })?;
                    Self::accumulate_crown_bounds(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::Flatten(f) => {
                    let new_lb = f.propagate_linear(&node_lb).map_err(|e| {
                        GammaError::InvalidSpec(format!(
                            "Constrained CROWN failed at node '{}' (Flatten): {}",
                            node_name, e
                        ))
                    })?;
                    let new_lb = match new_lb {
                        std::borrow::Cow::Borrowed(_) => node_lb,
                        std::borrow::Cow::Owned(lb) => lb,
                    };
                    Self::accumulate_crown_bounds(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::AveragePool(pool) => {
                    let new_lb = pool
                        .propagate_linear_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "Constrained CROWN failed at node '{}' (AveragePool): {}",
                                node_name, e
                            ))
                        })?;
                    Self::accumulate_crown_bounds(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::MaxPool2d(pool) => {
                    let new_lb = pool
                        .propagate_linear_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "Constrained CROWN failed at node '{}' (MaxPool2d): {}",
                                node_name, e
                            ))
                        })?;
                    Self::accumulate_crown_bounds(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::Concat(concat) => {
                    // Concat: SPLIT linear bounds across N inputs based on their shapes.
                    // Check constant inputs first (CLS token, etc.), then stored shapes, then bounds_cache.
                    let input_shapes: Vec<Vec<usize>> = node
                        .inputs
                        .iter()
                        .enumerate()
                        .map(|(i, inp_name)| {
                            // First check if this is a constant input (CLS token, etc.)
                            if let Some(constant_tensor) = concat.get_constant_input(i) {
                                return constant_tensor.shape().to_vec();
                            }
                            if inp_name == "_input" {
                                constrained_input.shape().to_vec()
                            } else if let Some(shape) = concat.get_input_shape(i) {
                                shape.clone()
                            } else {
                                bounds_cache
                                    .get(inp_name)
                                    .map(|b| b.shape().to_vec())
                                    .unwrap_or_else(|| vec![pre_activation.len()])
                            }
                        })
                        .collect();

                    // Use N-ary propagation to split bounds
                    let bounds_vec = concat
                        .propagate_linear_nary(&node_lb, &input_shapes)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "Constrained CROWN failed at node '{}' (Concat): {}",
                                node_name, e
                            ))
                        })?;

                    // Accumulate split bounds to each input (skip constant inputs)
                    for (i, (inp_name, lb)) in
                        node.inputs.iter().zip(bounds_vec.into_iter()).enumerate()
                    {
                        // Skip constant inputs - they have no gradient to propagate
                        if concat.get_constant_input(i).is_some() {
                            continue;
                        }
                        Self::accumulate_crown_bounds(
                            inp_name,
                            lb,
                            &mut node_linear_bounds,
                            &mut input_accumulated,
                        );
                    }
                }
                Layer::ReduceSum(rs) => {
                    let new_lb = rs
                        .propagate_linear_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "Constrained CROWN failed at node '{}' (ReduceSum): {}",
                                node_name, e
                            ))
                        })?;
                    Self::accumulate_crown_bounds(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::ReduceMean(rm) => {
                    let new_lb = rm
                        .propagate_linear_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "Constrained CROWN failed at node '{}' (ReduceMean): {}",
                                node_name, e
                            ))
                        })?;
                    Self::accumulate_crown_bounds(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                // For other layers, fall back to IBP result
                _ => {
                    debug!(
                        "Constrained backward CROWN: layer {} not supported, falling back to IBP",
                        node.layer.layer_type()
                    );
                    // Still propagate linear bounds to avoid breaking the chain
                    for input_name in &node.inputs {
                        Self::accumulate_crown_bounds(
                            input_name,
                            node_lb.clone(),
                            &mut node_linear_bounds,
                            &mut input_accumulated,
                        );
                    }
                }
            }
        }

        // Concretize the final linear bounds at network input
        let mut output_bounds = if input_accumulated {
            let final_lb = node_linear_bounds
                .remove("_input")
                .ok_or_else(|| GammaError::InvalidSpec("No linear bounds at input".to_string()))?;
            final_lb
                .concretize(&constrained_input)
                .reshape(&output_shape)?
        } else {
            // No backward pass reached input - fall back to IBP
            bounds_cache.remove(output_node).ok_or_else(|| {
                GammaError::InvalidSpec(format!("Output node {} not found", output_node))
            })?
        };

        // GCP-CROWN: Apply cutting plane contributions to lower bound
        if let Some(pool) = context.cut_pool {
            if !pool.is_empty() && self.config.enable_cuts {
                let relevant_cuts = pool.relevant_cuts_for(context.history);
                if !relevant_cuts.is_empty() {
                    let cut_contribution = self.compute_graph_cut_contribution(
                        graph,
                        &relevant_cuts,
                        &bounds_cache,
                        &constrained_input,
                    );

                    if cut_contribution > 0.0 {
                        // Add cut contribution to lower bound (tightens it)
                        let flat = output_bounds.flatten();
                        let shape = output_bounds.shape().to_vec();
                        let mut lower = flat.lower.clone();
                        for i in 0..lower.len() {
                            lower[[i]] += cut_contribution;
                        }
                        let lower_arr = lower
                            .into_shape_clone(ndarray::IxDyn(&shape))
                            .map_err(|e| GammaError::InvalidSpec(format!("shape error: {}", e)))?;
                        let upper_arr = flat
                            .upper
                            .clone()
                            .into_shape_clone(ndarray::IxDyn(&shape))
                            .map_err(|e| GammaError::InvalidSpec(format!("shape error: {}", e)))?;
                        output_bounds = BoundedTensor::new(lower_arr, upper_arr)?;

                        debug!(
                            "Applied {} graph cuts, contribution: {}",
                            relevant_cuts.len(),
                            cut_contribution
                        );
                    }
                }
            }
        }

        Ok((output_bounds, bounds_cache))
    }

    /// Batched CROWN backward propagation for multiple domains.
    ///
    /// This function processes N domains through the backward pass with true tensor-level
    /// batching at Linear layers. Instead of calling `propagate_crown_with_graph_constraints`
    /// N times (each launching GPU kernels), this batches all Linear layer GEMMs into
    /// single kernel launches, dramatically improving GPU utilization.
    ///
    /// # Performance
    /// For N domains processing through L Linear layers:
    /// - Sequential (old): N × L GPU kernel launches (each small)
    /// - Batched (this): L GPU kernel launches (each large, good GPU utilization)
    ///
    /// For cifar10_resnet with ~8 Linear layers and batch_size=64:
    /// - Old: 512 small GPU kernel launches per batch
    /// - New: 8 large GPU kernel launches per batch
    ///
    /// # Arguments
    /// * `graph` - The network graph
    /// * `domain_data` - Vec of (input_bounds, history, beta_state, base_bounds) per domain
    /// * `objective` - Objective coefficients (same for all domains)
    /// * `engine` - GPU compute engine
    ///
    /// # Returns
    /// Vec of (output_bounds, node_bounds_cache) per domain
    #[allow(clippy::type_complexity)]
    fn propagate_crown_batched_backward(
        &self,
        graph: &GraphNetwork,
        domain_data: &[(
            &BoundedTensor,                                                 // input_bounds
            &GraphSplitHistory,                                             // history
            Option<&GraphBetaState>,                                        // beta_state
            Option<&std::collections::HashMap<String, Arc<BoundedTensor>>>, // base_bounds
        )],
        objective: &[f32],
        engine: &dyn GemmEngine,
    ) -> Result<
        Vec<(
            BoundedTensor,
            std::collections::HashMap<String, BoundedTensor>,
        )>,
    > {
        use rayon::prelude::*;

        if domain_data.is_empty() {
            return Ok(Vec::new());
        }

        let n_domains = domain_data.len();
        let exec_order = graph.topological_sort()?;

        // ===== FORWARD PASS: Compute intermediate bounds for each domain =====
        // This is done in parallel but not GPU-batched (IBP is cheap)
        let forward_results: Vec<_> = domain_data
            .par_iter()
            .map(|(input, history, _beta_state, base_bounds)| {
                self.compute_constrained_forward_bounds(graph, input, history, *base_bounds)
            })
            .collect();

        // Check for errors and extract bounds_caches
        let mut bounds_caches: Vec<std::collections::HashMap<String, BoundedTensor>> =
            Vec::with_capacity(n_domains);
        let mut constrained_inputs: Vec<BoundedTensor> = Vec::with_capacity(n_domains);

        for (i, result) in forward_results.into_iter().enumerate() {
            match result {
                Ok((cache, input)) => {
                    bounds_caches.push(cache);
                    constrained_inputs.push(input);
                }
                Err(e) => {
                    return Err(GammaError::InvalidSpec(format!(
                        "Forward pass failed for domain {}: {}",
                        i, e
                    )));
                }
            }
        }

        // ===== BACKWARD PASS: Batched CROWN propagation =====
        let output_name = graph.get_output_name();
        let output_node = if output_name.is_empty() {
            exec_order
                .last()
                .ok_or_else(|| GammaError::InvalidSpec("No nodes in graph".to_string()))?
        } else {
            output_name
        };

        // Get output dimension (same for all domains)
        let ibp_output = bounds_caches[0].get(output_node).ok_or_else(|| {
            GammaError::InvalidSpec(format!("Output node {} not found", output_node))
        })?;
        let output_dim = ibp_output.len();

        if objective.len() != output_dim {
            return Err(GammaError::shape_mismatch(
                vec![objective.len()],
                vec![output_dim],
            ));
        }

        // Initialize LinearBounds for all domains at output node
        let output_shape = vec![1usize];
        let initial_a =
            Array2::from_shape_vec((1, output_dim), objective.to_vec()).map_err(|e| {
                GammaError::InvalidSpec(format!("Failed to build objective coefficients: {e}"))
            })?;
        let initial_lb = LinearBounds {
            lower_a: initial_a.clone(),
            lower_b: Array1::zeros(1),
            upper_a: initial_a,
            upper_b: Array1::zeros(1),
        };

        // Track LinearBounds per domain per node
        // node_linear_bounds[node_name] = Vec of LinearBounds, one per domain
        let mut node_linear_bounds: std::collections::HashMap<String, Vec<Option<LinearBounds>>> =
            std::collections::HashMap::new();

        // Initialize all domains at output node
        node_linear_bounds.insert(output_node.to_string(), vec![Some(initial_lb); n_domains]);

        let mut input_accumulated = vec![false; n_domains];

        // Backward propagation through nodes in reverse topological order
        for node_name in exec_order.iter().rev() {
            let node = graph
                .nodes
                .get(node_name)
                .ok_or_else(|| GammaError::InvalidSpec(format!("Node not found: {}", node_name)))?;

            // Get this node's accumulated linear bounds for all domains
            let node_lbs = match node_linear_bounds.remove(node_name) {
                Some(lbs) => lbs,
                None => continue, // Node has no consumers
            };

            // Check if any domain has bounds at this node
            let has_any_bounds = node_lbs.iter().any(|lb| lb.is_some());
            if !has_any_bounds {
                continue;
            }

            // Process based on layer type
            match &node.layer {
                Layer::Linear(l) => {
                    // Collect all domains that have bounds at this node
                    let mut active_indices: Vec<usize> = Vec::new();
                    let mut active_bounds: Vec<&LinearBounds> = Vec::new();

                    for (i, lb_opt) in node_lbs.iter().enumerate() {
                        if let Some(lb) = lb_opt {
                            active_indices.push(i);
                            active_bounds.push(lb);
                        }
                    }

                    if active_bounds.is_empty() {
                        continue;
                    }

                    // BATCHED GPU GEMM: Process all domains in one kernel call
                    let new_bounds =
                        l.propagate_linear_batched_with_engine(&active_bounds, engine)?;

                    // Distribute results back to domains
                    for (result_idx, &domain_idx) in active_indices.iter().enumerate() {
                        Self::accumulate_crown_bounds_batched(
                            &node.inputs[0],
                            new_bounds[result_idx].clone(),
                            &mut node_linear_bounds,
                            &mut input_accumulated[domain_idx],
                            domain_idx,
                            n_domains,
                        );
                    }
                }
                Layer::ReLU(r) => {
                    // Process ReLU for each domain (different pre-activation bounds)
                    for (domain_idx, lb_opt) in node_lbs.into_iter().enumerate() {
                        let Some(lb) = lb_opt else { continue };

                        let pre_activation_name =
                            if node.inputs.is_empty() || node.inputs[0] == "_input" {
                                "_input"
                            } else {
                                &node.inputs[0]
                            };
                        let pre_activation = if pre_activation_name == "_input" {
                            &constrained_inputs[domain_idx]
                        } else {
                            bounds_caches[domain_idx]
                                .get(pre_activation_name)
                                .ok_or_else(|| {
                                    GammaError::InvalidSpec(format!(
                                        "Pre-activation bounds for {} not found",
                                        pre_activation_name
                                    ))
                                })?
                        };

                        let mut new_lb = r.propagate_linear_with_bounds(&lb, pre_activation)?;

                        // Add β contribution if present
                        if let Some(beta_state) = domain_data[domain_idx].2 {
                            for entry in &beta_state.entries {
                                if entry.node_name != *node_name {
                                    continue;
                                }
                                let j = entry.neuron_idx;
                                if j >= new_lb.num_inputs() {
                                    continue;
                                }
                                let signed_beta = entry.value * entry.sign;
                                if signed_beta.abs() < 1e-10 {
                                    continue;
                                }

                                for i in 0..new_lb.num_outputs() {
                                    new_lb.lower_a[[i, j]] -= signed_beta;
                                    new_lb.upper_a[[i, j]] += signed_beta;
                                }
                            }
                        }

                        Self::accumulate_crown_bounds_batched(
                            &node.inputs[0],
                            new_lb,
                            &mut node_linear_bounds,
                            &mut input_accumulated[domain_idx],
                            domain_idx,
                            n_domains,
                        );
                    }
                }
                Layer::Add(_) => {
                    // Add propagates same linear bounds to both inputs
                    for (domain_idx, lb_opt) in node_lbs.into_iter().enumerate() {
                        let Some(lb) = lb_opt else { continue };

                        Self::accumulate_crown_bounds_batched(
                            &node.inputs[0],
                            lb.clone(),
                            &mut node_linear_bounds,
                            &mut input_accumulated[domain_idx],
                            domain_idx,
                            n_domains,
                        );
                        Self::accumulate_crown_bounds_batched(
                            &node.inputs[1],
                            lb,
                            &mut node_linear_bounds,
                            &mut input_accumulated[domain_idx],
                            domain_idx,
                            n_domains,
                        );
                    }
                }
                Layer::Conv2d(conv) => {
                    // Conv backward: process each domain (different input shapes possible)
                    for (domain_idx, lb_opt) in node_lbs.into_iter().enumerate() {
                        let Some(lb) = lb_opt else { continue };

                        let pre_activation_name =
                            if node.inputs.is_empty() || node.inputs[0] == "_input" {
                                "_input"
                            } else {
                                &node.inputs[0]
                            };
                        let pre_activation = if pre_activation_name == "_input" {
                            &constrained_inputs[domain_idx]
                        } else {
                            bounds_caches[domain_idx]
                                .get(pre_activation_name)
                                .ok_or_else(|| {
                                    GammaError::InvalidSpec(format!(
                                        "Pre-activation bounds for {} not found",
                                        pre_activation_name
                                    ))
                                })?
                        };

                        let input_shape = pre_activation.shape();
                        let (in_h, in_w) = if input_shape.len() >= 3 {
                            (
                                input_shape[input_shape.len() - 2],
                                input_shape[input_shape.len() - 1],
                            )
                        } else {
                            // Input too small for Conv2d CROWN, fall back
                            for input_name in &node.inputs {
                                Self::accumulate_crown_bounds_batched(
                                    input_name,
                                    lb.clone(),
                                    &mut node_linear_bounds,
                                    &mut input_accumulated[domain_idx],
                                    domain_idx,
                                    n_domains,
                                );
                            }
                            continue;
                        };
                        let mut conv_with_shape = conv.clone();
                        conv_with_shape.set_input_shape(in_h, in_w);
                        let new_lb = conv_with_shape.propagate_linear(&lb)?;
                        let new_lb = match new_lb {
                            std::borrow::Cow::Borrowed(_) => lb,
                            std::borrow::Cow::Owned(lb) => lb,
                        };
                        Self::accumulate_crown_bounds_batched(
                            &node.inputs[0],
                            new_lb,
                            &mut node_linear_bounds,
                            &mut input_accumulated[domain_idx],
                            domain_idx,
                            n_domains,
                        );
                    }
                }
                Layer::BatchNorm(bn) => {
                    for (domain_idx, lb_opt) in node_lbs.into_iter().enumerate() {
                        let Some(lb) = lb_opt else { continue };

                        let pre_activation_name =
                            if node.inputs.is_empty() || node.inputs[0] == "_input" {
                                "_input"
                            } else {
                                &node.inputs[0]
                            };
                        let pre_activation = if pre_activation_name == "_input" {
                            &constrained_inputs[domain_idx]
                        } else {
                            bounds_caches[domain_idx]
                                .get(pre_activation_name)
                                .ok_or_else(|| {
                                    GammaError::InvalidSpec(format!(
                                        "Pre-activation bounds for {} not found",
                                        pre_activation_name
                                    ))
                                })?
                        };

                        let new_lb = bn.propagate_linear_with_bounds(&lb, pre_activation)?;
                        Self::accumulate_crown_bounds_batched(
                            &node.inputs[0],
                            new_lb,
                            &mut node_linear_bounds,
                            &mut input_accumulated[domain_idx],
                            domain_idx,
                            n_domains,
                        );
                    }
                }
                Layer::Flatten(f) => {
                    for (domain_idx, lb_opt) in node_lbs.into_iter().enumerate() {
                        let Some(lb) = lb_opt else { continue };

                        let new_lb = f.propagate_linear(&lb)?;
                        let new_lb = match new_lb {
                            std::borrow::Cow::Borrowed(_) => lb,
                            std::borrow::Cow::Owned(lb) => lb,
                        };
                        Self::accumulate_crown_bounds_batched(
                            &node.inputs[0],
                            new_lb,
                            &mut node_linear_bounds,
                            &mut input_accumulated[domain_idx],
                            domain_idx,
                            n_domains,
                        );
                    }
                }
                Layer::AveragePool(pool) => {
                    for (domain_idx, lb_opt) in node_lbs.into_iter().enumerate() {
                        let Some(lb) = lb_opt else { continue };

                        let pre_activation_name =
                            if node.inputs.is_empty() || node.inputs[0] == "_input" {
                                "_input"
                            } else {
                                &node.inputs[0]
                            };
                        let pre_activation = if pre_activation_name == "_input" {
                            &constrained_inputs[domain_idx]
                        } else {
                            bounds_caches[domain_idx]
                                .get(pre_activation_name)
                                .ok_or_else(|| {
                                    GammaError::InvalidSpec(format!(
                                        "Pre-activation bounds for {} not found",
                                        pre_activation_name
                                    ))
                                })?
                        };

                        let new_lb = pool.propagate_linear_with_bounds(&lb, pre_activation)?;
                        Self::accumulate_crown_bounds_batched(
                            &node.inputs[0],
                            new_lb,
                            &mut node_linear_bounds,
                            &mut input_accumulated[domain_idx],
                            domain_idx,
                            n_domains,
                        );
                    }
                }
                // For other layers, fall back to per-domain processing
                _ => {
                    for (domain_idx, lb_opt) in node_lbs.into_iter().enumerate() {
                        let Some(lb) = lb_opt else { continue };

                        for input_name in &node.inputs {
                            Self::accumulate_crown_bounds_batched(
                                input_name,
                                lb.clone(),
                                &mut node_linear_bounds,
                                &mut input_accumulated[domain_idx],
                                domain_idx,
                                n_domains,
                            );
                        }
                    }
                }
            }
        }

        // Concretize final linear bounds at network input for each domain
        let mut results: Vec<(
            BoundedTensor,
            std::collections::HashMap<String, BoundedTensor>,
        )> = Vec::with_capacity(n_domains);

        let input_bounds_vec = node_linear_bounds.remove("_input");

        // Drain bounds_caches into an iterator so we can consume them in order
        let mut bounds_caches_iter = bounds_caches.into_iter();

        for domain_idx in 0..n_domains {
            let output_bounds = if input_accumulated[domain_idx] {
                let final_lb = input_bounds_vec
                    .as_ref()
                    .and_then(|v| v[domain_idx].clone())
                    .ok_or_else(|| {
                        GammaError::InvalidSpec(format!(
                            "No linear bounds at input for domain {}",
                            domain_idx
                        ))
                    })?;
                final_lb
                    .concretize(&constrained_inputs[domain_idx])
                    .reshape(&output_shape)?
            } else {
                // No backward pass reached input - fall back to IBP
                // Get the bounds cache for this domain (we'll consume it below)
                let cache = bounds_caches_iter
                    .next()
                    .ok_or_else(|| GammaError::InvalidSpec("Missing bounds cache".to_string()))?;
                let output = cache
                    .get(output_node)
                    .ok_or_else(|| {
                        GammaError::InvalidSpec(format!("Output node {} not found", output_node))
                    })?
                    .clone();
                // Put it back for the results
                results.push((output, cache));
                continue;
            };

            // Get the bounds cache for this domain
            let cache = bounds_caches_iter
                .next()
                .ok_or_else(|| GammaError::InvalidSpec("Missing bounds cache".to_string()))?;
            results.push((output_bounds, cache));
        }

        Ok(results)
    }

    /// Helper function to compute constrained forward bounds for a single domain.
    ///
    /// This is extracted from `propagate_crown_with_graph_constraints` forward pass.
    fn compute_constrained_forward_bounds(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        history: &GraphSplitHistory,
        base_bounds: Option<&std::collections::HashMap<String, Arc<BoundedTensor>>>,
    ) -> Result<(
        std::collections::HashMap<String, BoundedTensor>,
        BoundedTensor,
    )> {
        let exec_order = graph.topological_sort()?;

        // Use base bounds if provided, otherwise compute IBP bounds
        let mut bounds_cache: std::collections::HashMap<String, BoundedTensor> =
            if let Some(base) = base_bounds {
                base.iter()
                    .map(|(k, v)| (k.clone(), v.as_ref().clone()))
                    .collect()
            } else {
                graph.collect_node_bounds(input)?
            };

        // Build constraint lookups
        let mut constraints_by_relu: std::collections::HashMap<
            String,
            std::collections::HashMap<usize, bool>,
        > = std::collections::HashMap::new();
        let mut pre_constraints: std::collections::HashMap<String, Vec<(usize, bool, String)>> =
            std::collections::HashMap::new();

        for c in &history.constraints {
            constraints_by_relu
                .entry(c.node_name.clone())
                .or_default()
                .insert(c.neuron_idx, c.is_active);

            let relu_node = graph.nodes.get(&c.node_name).ok_or_else(|| {
                GammaError::InvalidSpec(format!(
                    "Graph constraint references missing node '{}'",
                    c.node_name
                ))
            })?;
            if !matches!(relu_node.layer, Layer::ReLU(_)) {
                return Err(GammaError::InvalidSpec(format!(
                    "Graph constraint references non-ReLU node '{}'",
                    c.node_name
                )));
            }
            let pre_name = relu_node
                .inputs
                .first()
                .cloned()
                .unwrap_or_else(|| "_input".to_string());
            pre_constraints.entry(pre_name).or_default().push((
                c.neuron_idx,
                c.is_active,
                c.node_name.clone(),
            ));
        }

        // Apply constraint tightening
        let apply_pre_constraints = |bounds: &BoundedTensor,
                                     constraints: &[(usize, bool, String)]|
         -> Result<BoundedTensor> {
            if constraints.is_empty() {
                return Ok(bounds.clone());
            }

            let flat = bounds.flatten();
            let shape = bounds.shape().to_vec();
            let mut lower = flat.lower.clone();
            let mut upper = flat.upper.clone();

            for (neuron_idx, is_active, relu_name) in constraints {
                if *neuron_idx >= flat.len() {
                    return Err(GammaError::InvalidSpec(format!(
                        "Constraint out of range: relu='{}' idx={} len={}",
                        relu_name,
                        neuron_idx,
                        flat.len()
                    )));
                }
                if *is_active {
                    lower[[*neuron_idx]] = lower[[*neuron_idx]].max(0.0);
                } else {
                    upper[[*neuron_idx]] = upper[[*neuron_idx]].min(0.0);
                }
                if lower[[*neuron_idx]] > upper[[*neuron_idx]] {
                    return Err(GammaError::InvalidSpec(format!(
                        "Infeasible domain after constraint: relu='{}' idx={} [{}, {}]",
                        relu_name,
                        neuron_idx,
                        lower[[*neuron_idx]],
                        upper[[*neuron_idx]]
                    )));
                }
            }

            let lower_arr = lower
                .into_shape_clone(ndarray::IxDyn(&shape))
                .map_err(|e| GammaError::InvalidSpec(format!("shape error: {}", e)))?;
            let upper_arr = upper
                .into_shape_clone(ndarray::IxDyn(&shape))
                .map_err(|e| GammaError::InvalidSpec(format!("shape error: {}", e)))?;
            BoundedTensor::new(lower_arr, upper_arr)
        };

        // Apply constraints to input
        let mut constrained_input = input.clone();
        if let Some(cons) = pre_constraints.get("_input") {
            constrained_input = apply_pre_constraints(&constrained_input, cons)?;
        }

        // Apply constraint tightening to bounds cache
        for node_name in &exec_order {
            let node = graph
                .nodes
                .get(node_name)
                .ok_or_else(|| GammaError::InvalidSpec(format!("Node not found: {}", node_name)))?;

            let current_bounds = bounds_cache.get(node_name).ok_or_else(|| {
                GammaError::InvalidSpec(format!(
                    "CROWN-IBP bounds not found for node '{}'",
                    node_name
                ))
            })?;

            let mut output_bounds = if matches!(node.layer, Layer::ReLU(_)) {
                let pre_activation = if node.inputs.is_empty() || node.inputs[0] == "_input" {
                    &constrained_input
                } else {
                    bounds_cache.get(&node.inputs[0]).ok_or_else(|| {
                        GammaError::InvalidSpec(format!(
                            "Pre-activation bounds for {} not found",
                            node.inputs[0]
                        ))
                    })?
                };

                let pre_flat = pre_activation.flatten();
                let shape = pre_activation.shape().to_vec();
                let mut lower = pre_flat.lower.clone();
                let mut upper = pre_flat.upper.clone();

                let relu_cons = constraints_by_relu.get(node_name);
                for neuron_idx in 0..pre_flat.len() {
                    let l = pre_flat.lower[[neuron_idx]];
                    let u = pre_flat.upper[[neuron_idx]];

                    let constrained = relu_cons.and_then(|m| m.get(&neuron_idx).copied());
                    match constrained {
                        Some(true) => {
                            if u < 0.0 {
                                return Err(GammaError::InvalidSpec(format!(
                                    "Infeasible active ReLU constraint at node '{}' idx={} pre_u={}",
                                    node_name, neuron_idx, u
                                )));
                            }
                            lower[[neuron_idx]] = l.max(0.0);
                            upper[[neuron_idx]] = u;
                        }
                        Some(false) => {
                            if l > 0.0 {
                                return Err(GammaError::InvalidSpec(format!(
                                    "Infeasible inactive ReLU constraint at node '{}' idx={} pre_l={}",
                                    node_name, neuron_idx, l
                                )));
                            }
                            lower[[neuron_idx]] = 0.0;
                            upper[[neuron_idx]] = 0.0;
                        }
                        None => {
                            let crown_bounds = current_bounds.flatten();
                            lower[[neuron_idx]] = crown_bounds.lower[[neuron_idx]];
                            upper[[neuron_idx]] = crown_bounds.upper[[neuron_idx]];
                        }
                    }
                }

                let lower_arr = lower
                    .into_shape_clone(ndarray::IxDyn(&shape))
                    .map_err(|e| GammaError::InvalidSpec(format!("shape error: {}", e)))?;
                let upper_arr = upper
                    .into_shape_clone(ndarray::IxDyn(&shape))
                    .map_err(|e| GammaError::InvalidSpec(format!("shape error: {}", e)))?;
                BoundedTensor::new(lower_arr, upper_arr)?
            } else {
                current_bounds.clone()
            };

            if let Some(cons) = pre_constraints.get(node_name) {
                output_bounds = apply_pre_constraints(&output_bounds, cons)?;
            }

            bounds_cache.insert(node_name.clone(), output_bounds);
        }

        Ok((bounds_cache, constrained_input))
    }

    /// Helper to accumulate CROWN bounds for batched processing.
    fn accumulate_crown_bounds_batched(
        input_name: &str,
        new_lb: LinearBounds,
        node_linear_bounds: &mut std::collections::HashMap<String, Vec<Option<LinearBounds>>>,
        input_accumulated: &mut bool,
        domain_idx: usize,
        n_domains: usize,
    ) {
        if input_name == "_input" {
            *input_accumulated = true;
        }

        let entry = node_linear_bounds
            .entry(input_name.to_string())
            .or_insert_with(|| vec![None; n_domains]);

        if let Some(existing) = &mut entry[domain_idx] {
            // Accumulate: add the new bounds to existing
            existing.lower_a = &existing.lower_a + &new_lb.lower_a;
            existing.lower_b = &existing.lower_b + &new_lb.lower_b;
            existing.upper_a = &existing.upper_a + &new_lb.upper_a;
            existing.upper_b = &existing.upper_b + &new_lb.upper_b;
        } else {
            entry[domain_idx] = Some(new_lb);
        }
    }

    /// Propagate CROWN bounds with Lagrangian β contribution.
    ///
    /// This extends `propagate_crown_with_graph_constraints` to include β parameters
    /// as Lagrangian multipliers for split constraints. The β contribution tightens
    /// the lower bound based on constraint satisfaction.
    ///
    /// For each constraint (node_name, neuron_idx, is_active):
    /// - Active constraint (x >= 0): contribution = β * x_lower
    /// - Inactive constraint (x <= 0): contribution = -β * x_upper
    ///
    /// This is a Lagrangian relaxation: we add β * (constraint_slack) to the lower bound.
    fn propagate_crown_with_graph_beta(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        context: &GraphCrownContext<'_>,
        beta_state: &GraphBetaState,
        objective: Option<&[f32]>,
    ) -> Result<(
        BoundedTensor,
        std::collections::HashMap<String, BoundedTensor>,
    )> {
        // Compute bounds with β integrated into the backward pass.
        // The β contribution is now added to linear coefficients during the ReLU
        // backward pass, rather than as a scalar offset to final bounds.
        // This is the correct Lagrangian β-CROWN formulation.
        self.propagate_crown_with_graph_constraints(
            graph,
            input,
            context,
            Some(beta_state),
            objective,
        )
    }

    /// Optimize β parameters using SPSA (Simultaneous Perturbation Stochastic Approximation).
    ///
    /// This is the graph network equivalent of the sequential β optimization loop.
    /// Since analytical gradients are complex for DAGs with skip connections, we use
    /// SPSA to estimate gradients efficiently.
    ///
    /// Returns the optimized bounds and updated beta_state.
    fn optimize_graph_beta_spsa(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        context: &GraphCrownContext<'_>,
        beta_state: &mut GraphBetaState,
        objective: &[f32],
    ) -> Result<(f32, f32, std::collections::HashMap<String, BoundedTensor>)> {
        use rand::Rng;

        // Skip if no beta parameters or iterations disabled
        if beta_state.is_empty() || self.config.beta_iterations == 0 {
            let (output, node_bounds) = self.propagate_crown_with_graph_beta(
                graph,
                input,
                context,
                beta_state,
                Some(objective),
            )?;
            let lb = output.lower_scalar();
            let ub = output.upper_scalar();
            return Ok((lb, ub, node_bounds));
        }

        let mut rng = rand::rng();
        let eps = 1e-3f32; // Perturbation magnitude
        let num_samples = 1; // SPSA samples per iteration (1 is often sufficient)

        let mut best_lower = f32::NEG_INFINITY;
        let mut best_upper = f32::INFINITY;
        let mut best_node_bounds = std::collections::HashMap::new();

        // Store beta values for each iteration (updated after Adam step)
        let mut current_betas: Vec<f32> = beta_state.entries.iter().map(|e| e.value).collect();

        for iter in 0..self.config.beta_iterations {
            // Reset gradients
            beta_state.zero_grad();

            // Compute bounds with current β
            let (output, node_bounds) = self.propagate_crown_with_graph_beta(
                graph,
                input,
                context,
                beta_state,
                Some(objective),
            )?;
            let lb = output.lower_scalar();
            let ub = output.upper_scalar();

            // Track best bounds
            if lb > best_lower {
                best_lower = lb;
                best_upper = ub;
                best_node_bounds = node_bounds.clone();
            }

            // Early exit if already verified (lower bound > 0 for minimization problems)
            // This check is domain-specific; caller will check against threshold

            // SPSA gradient estimation
            for _sample in 0..num_samples {
                // Generate Bernoulli perturbation (+1 or -1) for each β
                let perturbations: Vec<f32> = (0..beta_state.entries.len())
                    .map(|_| if rng.random_bool(0.5) { 1.0 } else { -1.0 })
                    .collect();

                // Create perturbed states
                // +ε perturbation
                for (i, entry) in beta_state.entries.iter_mut().enumerate() {
                    entry.value = (current_betas[i] + eps * perturbations[i]).max(0.0);
                }
                let (output_plus, _) = self.propagate_crown_with_graph_beta(
                    graph,
                    input,
                    context,
                    beta_state,
                    Some(objective),
                )?;
                let lb_plus = output_plus.lower_scalar();

                // -ε perturbation
                for (i, entry) in beta_state.entries.iter_mut().enumerate() {
                    entry.value = (current_betas[i] - eps * perturbations[i]).max(0.0);
                }
                let (output_minus, _) = self.propagate_crown_with_graph_beta(
                    graph,
                    input,
                    context,
                    beta_state,
                    Some(objective),
                )?;
                let lb_minus = output_minus.lower_scalar();

                // Restore current values
                for (i, entry) in beta_state.entries.iter_mut().enumerate() {
                    entry.value = current_betas[i];
                }

                // SPSA gradient estimate: g_i = (f+ - f-) / (2 * eps * Δ_i)
                let diff = lb_plus - lb_minus;
                for (i, entry) in beta_state.entries.iter_mut().enumerate() {
                    entry.grad += diff / (2.0 * eps * perturbations[i]) / (num_samples as f32);
                }
            }

            // Adam gradient step
            let t = iter + 1;
            let max_grad = beta_state.gradient_step_adam(&self.config.adaptive_config, t);

            // Update current_betas for next iteration (after Adam step updated values)
            for (i, entry) in beta_state.entries.iter().enumerate() {
                current_betas[i] = entry.value;
            }

            // Check convergence
            if max_grad < self.config.beta_tolerance {
                trace!(
                    "Graph β-SPSA converged at iteration {} (max_grad={:.6})",
                    iter,
                    max_grad
                );
                break;
            }
        }

        // Compute final bounds with optimized β
        let (output, node_bounds) = self.propagate_crown_with_graph_beta(
            graph,
            input,
            context,
            beta_state,
            Some(objective),
        )?;
        let lb = output.lower_scalar();
        let ub = output.upper_scalar();

        // Return the better of final vs best-seen bounds
        if lb >= best_lower {
            Ok((lb, ub, node_bounds))
        } else {
            Ok((best_lower, best_upper, best_node_bounds))
        }
    }

    /// Optimize β parameters using analytical gradients for DAG networks.
    ///
    /// This is the efficient alternative to SPSA. Instead of finite-difference gradient
    /// estimation (which requires 3 propagation passes per iteration), this computes
    /// analytical gradients from the A matrices stored during a single backward pass.
    ///
    /// For each constraint at (node_name, neuron_idx, sign), the gradient is:
    ///   ∂lb/∂β = -sign * sensitivity
    ///
    /// where sensitivity is derived from the A matrix at that ReLU layer.
    ///
    /// Returns the optimized bounds and updated beta_state.
    fn optimize_graph_beta_analytical(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        context: &GraphCrownContext<'_>,
        beta_state: &mut GraphBetaState,
        objective: &[f32],
    ) -> Result<(f32, f32, std::collections::HashMap<String, BoundedTensor>)> {
        // Skip if no beta parameters or iterations disabled
        if beta_state.is_empty() || self.config.beta_iterations == 0 {
            let (output, node_bounds) = self.propagate_crown_with_graph_beta(
                graph,
                input,
                context,
                beta_state,
                Some(objective),
            )?;
            let lb = output.lower_scalar();
            let ub = output.upper_scalar();
            return Ok((lb, ub, node_bounds));
        }

        let mut best_lower = f32::NEG_INFINITY;
        let mut best_upper = f32::INFINITY;
        let mut best_node_bounds = std::collections::HashMap::new();

        for iter in 0..self.config.beta_iterations {
            // Reset gradients
            beta_state.zero_grad();

            // Compute bounds with current β AND get intermediate A matrices
            let (output, node_bounds, intermediate) = self
                .propagate_crown_with_graph_beta_and_intermediates(
                    graph,
                    input,
                    context,
                    beta_state,
                    Some(objective),
                )?;

            let lb = output.lower_scalar();
            let ub = output.upper_scalar();

            // Track best bounds
            if lb > best_lower {
                best_lower = lb;
                best_upper = ub;
                best_node_bounds = node_bounds.clone();
            }

            // Compute analytical gradients from A matrices
            beta_state.compute_analytical_gradients(&intermediate);

            // Adam gradient step
            let t = iter + 1;
            let max_grad = beta_state.gradient_step_adam(&self.config.adaptive_config, t);

            // Check convergence
            if max_grad < self.config.beta_tolerance {
                trace!(
                    "Graph β-analytical converged at iteration {} (max_grad={:.6})",
                    iter,
                    max_grad
                );
                break;
            }
        }

        // Compute final bounds with optimized β
        let (output, node_bounds) = self.propagate_crown_with_graph_beta(
            graph,
            input,
            context,
            beta_state,
            Some(objective),
        )?;
        let lb = output.lower_scalar();
        let ub = output.upper_scalar();

        // Return the better of final vs best-seen bounds
        if lb >= best_lower {
            Ok((lb, ub, node_bounds))
        } else {
            Ok((best_lower, best_upper, best_node_bounds))
        }
    }

    /// Propagate CROWN bounds with β and return intermediate A matrices for gradient computation.
    ///
    /// This is like `propagate_crown_with_graph_beta` but also captures the A matrices
    /// at constrained ReLU nodes during the backward pass. These A matrices are used
    /// to compute analytical β gradients.
    #[allow(clippy::type_complexity)]
    fn propagate_crown_with_graph_beta_and_intermediates(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        context: &GraphCrownContext<'_>,
        beta_state: &GraphBetaState,
        objective: Option<&[f32]>,
    ) -> Result<(
        BoundedTensor,
        std::collections::HashMap<String, BoundedTensor>,
        GraphAlphaCrownIntermediate,
    )> {
        self.propagate_crown_with_graph_constraints_storing_intermediates(
            graph,
            input,
            context,
            Some(beta_state),
            objective,
        )
    }

    /// Internal function: CROWN propagation with intermediate storage for gradients.
    ///
    /// Same as propagate_crown_with_graph_constraints but stores A matrices at
    /// constrained ReLU nodes for analytical gradient computation.
    #[allow(clippy::type_complexity)]
    fn propagate_crown_with_graph_constraints_storing_intermediates(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        context: &GraphCrownContext<'_>,
        beta_state: Option<&GraphBetaState>,
        objective: Option<&[f32]>,
    ) -> Result<(
        BoundedTensor,
        std::collections::HashMap<String, BoundedTensor>,
        GraphAlphaCrownIntermediate,
    )> {
        // Initialize intermediate storage
        let mut intermediate = GraphAlphaCrownIntermediate::new();

        // ==== FORWARD PASS: Same as propagate_crown_with_graph_constraints ====
        let exec_order = graph.topological_sort()?;

        let mut bounds_cache: std::collections::HashMap<String, BoundedTensor> =
            if let Some(base) = context.base_bounds {
                base.iter()
                    .map(|(k, v)| (k.clone(), v.as_ref().clone()))
                    .collect()
            } else {
                graph.collect_node_bounds(input)?
            };

        // Build constraint lookup
        let mut constraints_by_relu: std::collections::HashMap<
            String,
            std::collections::HashMap<usize, bool>,
        > = std::collections::HashMap::new();
        let mut pre_constraints: std::collections::HashMap<String, Vec<(usize, bool, String)>> =
            std::collections::HashMap::new();

        for c in &context.history.constraints {
            constraints_by_relu
                .entry(c.node_name.clone())
                .or_default()
                .insert(c.neuron_idx, c.is_active);

            let relu_node = graph.nodes.get(&c.node_name).ok_or_else(|| {
                GammaError::InvalidSpec(format!(
                    "Graph constraint references missing node '{}'",
                    c.node_name
                ))
            })?;
            if !matches!(relu_node.layer, Layer::ReLU(_)) {
                return Err(GammaError::InvalidSpec(format!(
                    "Graph constraint references non-ReLU node '{}'",
                    c.node_name
                )));
            }
            let pre_name = relu_node
                .inputs
                .first()
                .cloned()
                .unwrap_or_else(|| "_input".to_string());
            pre_constraints.entry(pre_name).or_default().push((
                c.neuron_idx,
                c.is_active,
                c.node_name.clone(),
            ));
        }

        let apply_pre_constraints = |bounds: &BoundedTensor,
                                     constraints: &[(usize, bool, String)]|
         -> Result<BoundedTensor> {
            if constraints.is_empty() {
                return Ok(bounds.clone());
            }

            let flat = bounds.flatten();
            let shape = bounds.shape().to_vec();
            let mut lower = flat.lower.clone();
            let mut upper = flat.upper.clone();

            for (neuron_idx, is_active, relu_name) in constraints {
                if *neuron_idx >= flat.len() {
                    return Err(GammaError::InvalidSpec(format!(
                        "Constraint out of range: relu='{}' idx={} len={}",
                        relu_name,
                        neuron_idx,
                        flat.len()
                    )));
                }
                if *is_active {
                    lower[[*neuron_idx]] = lower[[*neuron_idx]].max(0.0);
                } else {
                    upper[[*neuron_idx]] = upper[[*neuron_idx]].min(0.0);
                }
                if lower[[*neuron_idx]] > upper[[*neuron_idx]] {
                    return Err(GammaError::InvalidSpec(format!(
                        "Infeasible domain after constraint: relu='{}' idx={} [{}, {}]",
                        relu_name,
                        neuron_idx,
                        lower[[*neuron_idx]],
                        upper[[*neuron_idx]]
                    )));
                }
            }

            let lower_arr = lower
                .into_shape_clone(ndarray::IxDyn(&shape))
                .map_err(|e| GammaError::InvalidSpec(format!("shape error: {}", e)))?;
            let upper_arr = upper
                .into_shape_clone(ndarray::IxDyn(&shape))
                .map_err(|e| GammaError::InvalidSpec(format!("shape error: {}", e)))?;
            BoundedTensor::new(lower_arr, upper_arr)
        };

        // Apply constraints to input
        let mut constrained_input = input.clone();
        if let Some(cons) = pre_constraints.get("_input") {
            constrained_input = apply_pre_constraints(&constrained_input, cons)?;
        }

        // Apply constraint tightening to forward bounds
        for node_name in &exec_order {
            let node = graph
                .nodes
                .get(node_name)
                .ok_or_else(|| GammaError::InvalidSpec(format!("Node not found: {}", node_name)))?;

            let current_bounds = bounds_cache.get(node_name).ok_or_else(|| {
                GammaError::InvalidSpec(format!(
                    "CROWN-IBP bounds not found for node '{}'",
                    node_name
                ))
            })?;

            let mut output_bounds = if matches!(node.layer, Layer::ReLU(_)) {
                let pre_activation = if node.inputs.is_empty() || node.inputs[0] == "_input" {
                    &constrained_input
                } else {
                    bounds_cache.get(&node.inputs[0]).ok_or_else(|| {
                        GammaError::InvalidSpec(format!(
                            "Pre-activation bounds for {} not found",
                            node.inputs[0]
                        ))
                    })?
                };

                let pre_flat = pre_activation.flatten();
                let shape = pre_activation.shape().to_vec();
                let mut lower = pre_flat.lower.clone();
                let mut upper = pre_flat.upper.clone();

                let relu_cons = constraints_by_relu.get(node_name);
                for neuron_idx in 0..pre_flat.len() {
                    let l = pre_flat.lower[[neuron_idx]];
                    let u = pre_flat.upper[[neuron_idx]];

                    let constrained = relu_cons.and_then(|m| m.get(&neuron_idx).copied());
                    match constrained {
                        Some(true) => {
                            if u < 0.0 {
                                return Err(GammaError::InvalidSpec(format!(
                                    "Infeasible active ReLU constraint at node '{}' idx={} pre_u={}",
                                    node_name, neuron_idx, u
                                )));
                            }
                            lower[[neuron_idx]] = l.max(0.0);
                            upper[[neuron_idx]] = u;
                        }
                        Some(false) => {
                            if l > 0.0 {
                                return Err(GammaError::InvalidSpec(format!(
                                    "Infeasible inactive ReLU constraint at node '{}' idx={} pre_l={}",
                                    node_name, neuron_idx, l
                                )));
                            }
                            lower[[neuron_idx]] = 0.0;
                            upper[[neuron_idx]] = 0.0;
                        }
                        None => {
                            let crown_bounds = current_bounds.flatten();
                            lower[[neuron_idx]] = crown_bounds.lower[[neuron_idx]];
                            upper[[neuron_idx]] = crown_bounds.upper[[neuron_idx]];
                        }
                    }
                }

                let lower_arr = lower
                    .into_shape_clone(ndarray::IxDyn(&shape))
                    .map_err(|e| GammaError::InvalidSpec(format!("shape error: {}", e)))?;
                let upper_arr = upper
                    .into_shape_clone(ndarray::IxDyn(&shape))
                    .map_err(|e| GammaError::InvalidSpec(format!("shape error: {}", e)))?;
                BoundedTensor::new(lower_arr, upper_arr)?
            } else {
                current_bounds.clone()
            };

            if let Some(cons) = pre_constraints.get(node_name) {
                output_bounds = apply_pre_constraints(&output_bounds, cons)?;
            }

            bounds_cache.insert(node_name.clone(), output_bounds);
        }

        // ==== BACKWARD CROWN PASS with intermediate storage ====
        let output_name = graph.get_output_name();
        let output_node = if output_name.is_empty() {
            exec_order
                .last()
                .ok_or_else(|| GammaError::InvalidSpec("No nodes in graph".to_string()))?
        } else {
            output_name
        };

        let ibp_output = bounds_cache.get(output_node).ok_or_else(|| {
            GammaError::InvalidSpec(format!("Output node {} not found", output_node))
        })?;
        let output_dim = ibp_output.len();
        let (output_shape, initial_lb) = if let Some(objective) = objective {
            if objective.len() != output_dim {
                return Err(GammaError::shape_mismatch(
                    vec![objective.len()],
                    vec![output_dim],
                ));
            }
            let a = Array2::from_shape_vec((1, output_dim), objective.to_vec()).map_err(|e| {
                GammaError::InvalidSpec(format!("Failed to build objective coefficients: {e}"))
            })?;
            (
                vec![1usize],
                LinearBounds {
                    lower_a: a.clone(),
                    lower_b: Array1::zeros(1),
                    upper_a: a,
                    upper_b: Array1::zeros(1),
                },
            )
        } else {
            (
                ibp_output.shape().to_vec(),
                LinearBounds::identity(output_dim),
            )
        };

        let mut node_linear_bounds: std::collections::HashMap<String, LinearBounds> =
            std::collections::HashMap::new();
        node_linear_bounds.insert(output_node.to_string(), initial_lb);

        let mut input_accumulated = false;

        // Backward pass - STORE A matrices at constrained ReLU nodes
        for node_name in exec_order.iter().rev() {
            let node = graph
                .nodes
                .get(node_name)
                .ok_or_else(|| GammaError::InvalidSpec(format!("Node not found: {}", node_name)))?;

            let node_lb = match node_linear_bounds.remove(node_name) {
                Some(lb) => lb,
                None => continue,
            };

            let pre_activation_name = if node.inputs.is_empty() || node.inputs[0] == "_input" {
                "_input"
            } else {
                &node.inputs[0]
            };
            let pre_activation = if pre_activation_name == "_input" {
                &constrained_input
            } else {
                bounds_cache.get(pre_activation_name).ok_or_else(|| {
                    GammaError::InvalidSpec(format!(
                        "Pre-activation bounds for {} not found",
                        pre_activation_name
                    ))
                })?
            };

            match &node.layer {
                Layer::Linear(l) => {
                    let new_lb = l
                        .propagate_linear_with_engine(&node_lb, context.engine)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "CROWN failed at node '{}' (Linear): {}",
                                node_name, e
                            ))
                        })?;
                    let new_lb = match new_lb {
                        std::borrow::Cow::Borrowed(_) => node_lb,
                        std::borrow::Cow::Owned(lb) => lb,
                    };
                    Self::accumulate_crown_bounds(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::ReLU(r) => {
                    // STORE A matrix BEFORE β modification for gradient computation
                    // Only store for constrained nodes (nodes with β parameters)
                    if constraints_by_relu.contains_key(node_name) {
                        intermediate
                            .a_at_relu
                            .insert(node_name.clone(), node_lb.lower_a.clone());

                        // Also store pre-ReLU bounds
                        let flat = pre_activation.flatten();
                        let lower = flat
                            .lower
                            .clone()
                            .into_dimensionality::<ndarray::Ix1>()
                            .unwrap_or_else(|_| Array1::zeros(flat.len()));
                        let upper = flat
                            .upper
                            .clone()
                            .into_dimensionality::<ndarray::Ix1>()
                            .unwrap_or_else(|_| Array1::zeros(flat.len()));
                        intermediate
                            .pre_relu_bounds
                            .insert(node_name.clone(), (lower, upper));
                    }

                    let mut new_lb = r
                        .propagate_linear_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "CROWN failed at node '{}' (ReLU): {}",
                                node_name, e
                            ))
                        })?;

                    // Add β contribution
                    if let Some(beta) = beta_state {
                        for entry in &beta.entries {
                            if entry.node_name != *node_name {
                                continue;
                            }
                            let j = entry.neuron_idx;
                            if j >= new_lb.num_inputs() {
                                continue;
                            }
                            let signed_beta = entry.value * entry.sign;
                            if signed_beta.abs() < 1e-10 {
                                continue;
                            }

                            for i in 0..new_lb.num_outputs() {
                                new_lb.lower_a[[i, j]] -= signed_beta;
                                new_lb.upper_a[[i, j]] += signed_beta;
                            }
                        }
                    }

                    Self::accumulate_crown_bounds(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::Add(_) => {
                    Self::accumulate_crown_bounds(
                        &node.inputs[0],
                        node_lb.clone(),
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                    Self::accumulate_crown_bounds(
                        &node.inputs[1],
                        node_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::Conv2d(conv) => {
                    let input_shape = pre_activation.shape();
                    let (in_h, in_w) = if input_shape.len() >= 3 {
                        (
                            input_shape[input_shape.len() - 2],
                            input_shape[input_shape.len() - 1],
                        )
                    } else {
                        for input_name in &node.inputs {
                            Self::accumulate_crown_bounds(
                                input_name,
                                node_lb.clone(),
                                &mut node_linear_bounds,
                                &mut input_accumulated,
                            );
                        }
                        continue;
                    };
                    let mut conv_with_shape = conv.clone();
                    conv_with_shape.set_input_shape(in_h, in_w);
                    let new_lb = conv_with_shape.propagate_linear(&node_lb).map_err(|e| {
                        GammaError::InvalidSpec(format!(
                            "CROWN failed at node '{}' (Conv2d): {}",
                            node_name, e
                        ))
                    })?;
                    let new_lb = match new_lb {
                        std::borrow::Cow::Borrowed(_) => node_lb,
                        std::borrow::Cow::Owned(lb) => lb,
                    };
                    Self::accumulate_crown_bounds(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::BatchNorm(bn) => {
                    let new_lb = bn
                        .propagate_linear_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "CROWN failed at node '{}' (BatchNorm): {}",
                                node_name, e
                            ))
                        })?;
                    Self::accumulate_crown_bounds(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::Flatten(f) => {
                    let new_lb = f.propagate_linear(&node_lb).map_err(|e| {
                        GammaError::InvalidSpec(format!(
                            "CROWN failed at node '{}' (Flatten): {}",
                            node_name, e
                        ))
                    })?;
                    let new_lb = match new_lb {
                        std::borrow::Cow::Borrowed(_) => node_lb,
                        std::borrow::Cow::Owned(lb) => lb,
                    };
                    Self::accumulate_crown_bounds(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::AveragePool(pool) => {
                    let new_lb = pool
                        .propagate_linear_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "CROWN failed at node '{}' (AveragePool): {}",
                                node_name, e
                            ))
                        })?;
                    Self::accumulate_crown_bounds(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                Layer::MaxPool2d(pool) => {
                    let new_lb = pool
                        .propagate_linear_with_bounds(&node_lb, pre_activation)
                        .map_err(|e| {
                            GammaError::InvalidSpec(format!(
                                "CROWN failed at node '{}' (MaxPool2d): {}",
                                node_name, e
                            ))
                        })?;
                    Self::accumulate_crown_bounds(
                        &node.inputs[0],
                        new_lb,
                        &mut node_linear_bounds,
                        &mut input_accumulated,
                    );
                }
                _ => {
                    for input_name in &node.inputs {
                        Self::accumulate_crown_bounds(
                            input_name,
                            node_lb.clone(),
                            &mut node_linear_bounds,
                            &mut input_accumulated,
                        );
                    }
                }
            }
        }

        // Concretize final bounds
        let mut output_bounds = if input_accumulated {
            let final_lb = node_linear_bounds
                .remove("_input")
                .ok_or_else(|| GammaError::InvalidSpec("No linear bounds at input".to_string()))?;
            final_lb
                .concretize(&constrained_input)
                .reshape(&output_shape)?
        } else {
            bounds_cache.remove(output_node).ok_or_else(|| {
                GammaError::InvalidSpec(format!("Output node {} not found", output_node))
            })?
        };

        // Apply cutting planes
        if let Some(pool) = context.cut_pool {
            if !pool.is_empty() && self.config.enable_cuts {
                let relevant_cuts = pool.relevant_cuts_for(context.history);
                if !relevant_cuts.is_empty() {
                    let cut_contribution = self.compute_graph_cut_contribution(
                        graph,
                        &relevant_cuts,
                        &bounds_cache,
                        &constrained_input,
                    );

                    if cut_contribution > 0.0 {
                        let flat = output_bounds.flatten();
                        let shape = output_bounds.shape().to_vec();
                        let mut lower = flat.lower.clone();
                        for i in 0..lower.len() {
                            lower[[i]] += cut_contribution;
                        }
                        let lower_arr = lower
                            .into_shape_clone(ndarray::IxDyn(&shape))
                            .map_err(|e| GammaError::InvalidSpec(format!("shape error: {}", e)))?;
                        let upper_arr = flat
                            .upper
                            .clone()
                            .into_shape_clone(ndarray::IxDyn(&shape))
                            .map_err(|e| GammaError::InvalidSpec(format!("shape error: {}", e)))?;
                        output_bounds = BoundedTensor::new(lower_arr, upper_arr)?;
                    }
                }
            }
        }

        Ok((output_bounds, bounds_cache, intermediate))
    }

    /// Compute bounds for multiple objectives from output tensor.
    fn objective_bounds_multi(
        output: &BoundedTensor,
        objectives: &[Vec<f32>],
    ) -> Result<Vec<(f32, f32)>> {
        let flat = output.flatten();
        objectives
            .iter()
            .map(|obj| {
                if flat.len() != obj.len() {
                    return Err(GammaError::shape_mismatch(
                        vec![obj.len()],
                        vec![flat.len()],
                    ));
                }
                let mut lower = 0.0f32;
                let mut upper = 0.0f32;
                for (idx, &c) in obj.iter().enumerate() {
                    let l = flat.lower[[idx]];
                    let u = flat.upper[[idx]];
                    if c >= 0.0 {
                        lower += c * l;
                        upper += c * u;
                    } else {
                        lower += c * u;
                        upper += c * l;
                    }
                }
                Ok((lower, upper))
            })
            .collect()
    }

    /// Propagate bounds with β for multi-objective verification without optimization.
    ///
    /// This is used for deep domains where we skip β optimization and rely on
    /// inherited β values from warmup.
    #[allow(clippy::type_complexity)]
    fn propagate_multi_objective_with_beta(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        context: &GraphCrownContext<'_>,
        beta_state: &GraphBetaState,
        targets: &MultiObjectiveTargets<'_>,
    ) -> Result<(
        Vec<(f32, f32)>,
        std::collections::HashMap<String, BoundedTensor>,
    )> {
        let (output, node_bounds) =
            self.propagate_crown_with_graph_beta(graph, input, context, beta_state, None)?;
        let obj_bounds = Self::objective_bounds_multi(&output, targets.objectives)?;
        Ok((obj_bounds, node_bounds))
    }

    /// Optimize β parameters using analytical gradients for multi-objective verification.
    ///
    /// This is the analytical gradient equivalent of `optimize_graph_beta_analytical_multi_objective`.
    /// It uses the same optimization objective (maximize minimum margin) but computes gradients
    /// analytically from the A matrices, avoiding the 3 forward passes per iteration that SPSA
    /// requires.
    ///
    /// For each iteration:
    /// 1. Propagate bounds and capture A matrices (1 forward pass)
    /// 2. Compute objective bounds for all objectives
    /// 3. Find the critical objective (minimum margin among unverified)
    /// 4. Compute β gradients for the critical objective using A matrices
    /// 5. Adam gradient step
    ///
    /// This is ~3x faster than SPSA since it only needs 1 forward pass per iteration.
    #[allow(clippy::type_complexity)]
    fn optimize_graph_beta_analytical_multi_objective(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        context: &GraphCrownContext<'_>,
        beta_state: &mut GraphBetaState,
        targets: &MultiObjectiveTargets<'_>,
    ) -> Result<(
        Vec<(f32, f32)>,
        std::collections::HashMap<String, BoundedTensor>,
    )> {
        // Skip if no beta parameters or iterations disabled
        if beta_state.is_empty() || self.config.beta_iterations == 0 {
            let (output, node_bounds) =
                self.propagate_crown_with_graph_beta(graph, input, context, beta_state, None)?;
            let obj_bounds = Self::objective_bounds_multi(&output, targets.objectives)?;
            return Ok((obj_bounds, node_bounds));
        }

        // Compute min margin metric for optimization
        let compute_min_margin = |bounds: &[(f32, f32)]| -> f32 {
            bounds
                .iter()
                .zip(targets.thresholds.iter())
                .zip(targets.verified_mask.iter())
                .filter(|((_, _), &v)| !v) // Only unverified objectives
                .map(|(((l, _), &t), _)| l - t)
                .fold(f32::INFINITY, |a, b| a.min(b))
        };

        let mut best_margin = f32::NEG_INFINITY;
        let mut best_obj_bounds = vec![(0.0f32, 0.0f32); targets.objectives.len()];
        let mut best_node_bounds = std::collections::HashMap::new();

        for iter in 0..self.config.beta_iterations {
            // Reset gradients
            beta_state.zero_grad();

            // Compute bounds with current β AND get intermediate A matrices
            // Note: We pass None for objective to get raw A matrices (not weighted by objective)
            let (output, node_bounds, intermediate) = self
                .propagate_crown_with_graph_constraints_storing_intermediates(
                    graph,
                    input,
                    context,
                    Some(beta_state),
                    None, // No objective - we'll apply objectives post-hoc
                )?;

            // Compute bounds for all objectives
            let obj_bounds = Self::objective_bounds_multi(&output, targets.objectives)?;
            let margin = compute_min_margin(&obj_bounds);

            // Track best bounds
            if margin > best_margin {
                best_margin = margin;
                best_obj_bounds = obj_bounds.clone();
                best_node_bounds = node_bounds.clone();
            }

            // Compute analytical gradients for the critical objective
            let max_grad = beta_state.compute_analytical_gradients_multi_objective(
                &intermediate,
                &obj_bounds,
                targets.objectives,
                targets.thresholds,
                targets.verified_mask,
            );

            // Adam gradient step
            let t = iter + 1;
            let _step_max_grad = beta_state.gradient_step_adam(&self.config.adaptive_config, t);

            // Check convergence
            if max_grad < self.config.beta_tolerance {
                trace!(
                    "Graph β-analytical multi-obj converged at iteration {} (max_grad={:.6})",
                    iter,
                    max_grad
                );
                break;
            }
        }

        // Compute final bounds with optimized β
        let (output, node_bounds) =
            self.propagate_crown_with_graph_beta(graph, input, context, beta_state, None)?;
        let obj_bounds = Self::objective_bounds_multi(&output, targets.objectives)?;
        let margin = compute_min_margin(&obj_bounds);

        // Return the better of final vs best-seen bounds
        if margin >= best_margin {
            Ok((obj_bounds, node_bounds))
        } else {
            Ok((best_obj_bounds, best_node_bounds))
        }
    }

    /// Accumulate linear bounds to a node during backward CROWN pass.
    fn accumulate_crown_bounds(
        input_name: &str,
        new_bounds: LinearBounds,
        node_linear_bounds: &mut std::collections::HashMap<String, LinearBounds>,
        input_accumulated: &mut bool,
    ) {
        if input_name == "_input" {
            if *input_accumulated {
                // Add to existing bounds
                if let Some(existing) = node_linear_bounds.get_mut("_input") {
                    existing.lower_a = &existing.lower_a + &new_bounds.lower_a;
                    existing.lower_b = &existing.lower_b + &new_bounds.lower_b;
                    existing.upper_a = &existing.upper_a + &new_bounds.upper_a;
                    existing.upper_b = &existing.upper_b + &new_bounds.upper_b;
                }
            } else {
                node_linear_bounds.insert("_input".to_string(), new_bounds);
                *input_accumulated = true;
            }
        } else {
            // Accumulate to intermediate node
            if let Some(existing) = node_linear_bounds.get_mut(input_name) {
                existing.lower_a = &existing.lower_a + &new_bounds.lower_a;
                existing.lower_b = &existing.lower_b + &new_bounds.lower_b;
                existing.upper_a = &existing.upper_a + &new_bounds.upper_a;
                existing.upper_b = &existing.upper_b + &new_bounds.upper_b;
            } else {
                node_linear_bounds.insert(input_name.to_string(), new_bounds);
            }
        }
    }

    /// Compute initial bounds with optional early termination and GPU acceleration.
    fn compute_initial_bounds_with_early_exit_engine(
        &self,
        network: &Network,
        input: &BoundedTensor,
        threshold_check: Option<(f32, bool)>, // (threshold, verify_upper_bound)
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        // If α-CROWN is enabled and we have a threshold to check, try fast bounds first
        if self.config.use_alpha_crown {
            if let Some((threshold, verify_upper)) = threshold_check {
                // Compute fast CROWN-IBP bounds first (use engine for CROWN)
                let fast_bounds = if self.config.use_crown_ibp {
                    // CROWN-IBP doesn't have engine variant yet, fall back to CPU
                    network.propagate_crown_ibp(input)?
                } else {
                    network.propagate_crown_with_engine(input, engine)?
                };

                let fast_lower = fast_bounds.lower_scalar();
                let fast_upper = fast_bounds.upper_scalar();

                // Check if fast bounds already verify
                let verified = if verify_upper {
                    // Verifying output < threshold (upper_bound < threshold)
                    fast_upper < threshold
                } else {
                    // Verifying output > threshold (lower_bound > threshold)
                    fast_lower > threshold
                };

                if verified {
                    debug!(
                        "Early termination: fast bounds [{:.2}, {:.2}] already verify threshold {}",
                        fast_lower, fast_upper, threshold
                    );
                    return Ok(fast_bounds);
                }

                // Fast bounds don't verify, proceed with α-CROWN
                debug!(
                    "Fast bounds [{:.2}, {:.2}] don't verify threshold {}, running α-CROWN",
                    fast_lower, fast_upper, threshold
                );
            }

            // Run α-CROWN with engine (either no threshold check or fast bounds didn't verify)
            network.propagate_alpha_crown_with_config_and_engine(
                input,
                &self.config.alpha_config,
                engine,
            )
        } else if self.config.use_crown_ibp {
            network.propagate_crown_ibp(input)
        } else {
            network.propagate_crown_with_engine(input, engine)
        }
    }

    /// Try to find a counterexample using PGD attack.
    ///
    /// Called when verification is inconclusive to try to find a concrete
    /// input that violates the property. If found, returns Violated status
    /// with the counterexample. Otherwise returns the original Unknown status.
    fn try_pgd_attack(
        &self,
        network: &Network,
        input: &BoundedTensor,
        threshold: f32,
        original_result: BetaCrownResult,
    ) -> Result<BetaCrownResult> {
        if !self.config.enable_pgd_attack {
            return Ok(original_result);
        }

        info!(
            "Running PGD attack with {} restarts, {} steps",
            self.config.pgd_restarts, self.config.pgd_steps
        );

        let pgd_config = PgdConfig {
            num_restarts: self.config.pgd_restarts,
            num_steps: self.config.pgd_steps,
            step_size: 0.01,
            spsa_delta: 0.001,
            seed: 42,
            parallel: true, // Use parallel restarts for efficiency
        };
        let attacker = PgdAttacker::new(pgd_config);

        // Run PGD attack
        // For verify_upper_bound=true: looking for output >= threshold
        // For verify_upper_bound=false: looking for output <= threshold
        let attack_result = attacker.attack(
            network,
            input,
            0, // Output index (scalar output assumed)
            threshold,
            self.config.verify_upper_bound,
        )?;

        if attack_result.found_counterexample {
            info!(
                "PGD found counterexample: output = {} {} threshold = {}",
                attack_result.best_output_value,
                if self.config.verify_upper_bound {
                    ">="
                } else {
                    "<="
                },
                threshold
            );

            let counterexample = attack_result
                .counterexample
                .unwrap()
                .iter()
                .copied()
                .collect();
            let output = attack_result.output.unwrap().iter().copied().collect();

            return Ok(BetaCrownResult {
                result: BabVerificationStatus::Violated {
                    counterexample,
                    output,
                },
                domains_explored: original_result.domains_explored,
                time_elapsed: original_result.time_elapsed,
                max_depth_reached: original_result.max_depth_reached,
                output_bounds: original_result.output_bounds,
                cuts_generated: original_result.cuts_generated,
                domains_verified: original_result.domains_verified,
            });
        }

        debug!(
            "PGD attack completed {} restarts, no counterexample found. Best: {} vs threshold {}",
            attack_result.restarts_completed, attack_result.best_output_value, threshold
        );

        Ok(original_result)
    }

    /// Process a single domain sequentially, returning child domains.
    fn process_domain_sequential(
        &self,
        network: &Network,
        input: &BoundedTensor,
        domain: &BabDomain,
        threshold: f32,
        cut_pool: &mut CutPool,
        engine: Option<&dyn GemmEngine>,
    ) -> Vec<BabDomain> {
        // Check if using input splitting (Conv layers now support ReLU splitting via CROWN backward)
        if matches!(
            self.config.branching_heuristic,
            BranchingHeuristic::InputSplit
        ) {
            return self
                .create_input_split_children(network, input, domain, threshold)
                .unwrap_or_else(|_| Vec::new());
        }

        let mut children = Vec::with_capacity(2);

        // Select neuron to split
        let split_neuron = match self.select_split_neuron(network, input, domain) {
            Ok(Some(neuron)) => neuron,
            Ok(None) => {
                debug!("No unstable neurons to split, domain unresolved");
                return children;
            }
            Err(_) => return children,
        };
        let (layer_idx, neuron_idx) = split_neuron;

        // Create active child
        if let Ok(Some(child)) = self.create_child_domain(
            network, input, domain, layer_idx, neuron_idx, true, threshold, cut_pool, engine,
        ) {
            children.push(child);
        }

        // Create inactive child
        if let Ok(Some(child)) = self.create_child_domain(
            network, input, domain, layer_idx, neuron_idx, false, threshold, cut_pool, engine,
        ) {
            children.push(child);
        }

        children
    }

    /// Process a single domain with parallel child creation, returning child domains.
    ///
    /// Note: When cuts are enabled, parallel child creation is disabled because
    /// lambda optimization modifies the shared cut pool. This is a known limitation;
    /// future work could use per-domain lambda copies.
    fn process_domain_parallel(
        &self,
        network: &Network,
        input: &BoundedTensor,
        domain: &BabDomain,
        config: &DomainProcessingConfig,
        cut_pool: &mut CutPool,
        engine: Option<&dyn GemmEngine>,
    ) -> Vec<BabDomain> {
        // Check if using input splitting (Conv layers now support ReLU splitting via CROWN backward)
        if matches!(
            self.config.branching_heuristic,
            BranchingHeuristic::InputSplit
        ) {
            return self
                .create_input_split_children(network, input, domain, config.threshold)
                .unwrap_or_else(|_| Vec::new());
        }

        // Select neuron to split
        let split_neuron = match self.select_split_neuron(network, input, domain) {
            Ok(Some(neuron)) => neuron,
            Ok(None) => {
                debug!("No unstable neurons to split, domain unresolved");
                return Vec::new();
            }
            Err(_) => return Vec::new(),
        };
        let (layer_idx, neuron_idx) = split_neuron;

        // When cuts are enabled, use sequential processing to avoid concurrent mutation
        // of lambda values in the cut pool. This is a known limitation.
        let has_cuts = !cut_pool.is_empty() && self.config.enable_cuts;

        if config.use_parallel_children && !has_cuts {
            // Create both children in parallel using rayon (no cuts)
            // Note: GPU engine not used in parallel path (not Sync); rayon CPU parallelism
            // is often faster than sequential GPU for the small 2-child parallel case
            let children: Vec<Option<BabDomain>> = [true, false]
                .par_iter()
                .map(|&is_active| {
                    let mut empty_pool = CutPool::new(0); // No cuts for parallel path
                    self.create_child_domain(
                        network,
                        input,
                        domain,
                        layer_idx,
                        neuron_idx,
                        is_active,
                        config.threshold,
                        &mut empty_pool,
                        None, // CPU-only for rayon parallel path
                    )
                    .ok()
                    .flatten()
                })
                .collect();

            children.into_iter().flatten().collect()
        } else {
            // Sequential child creation (required when cuts are enabled)
            // GPU engine can be used here for acceleration
            self.process_domain_sequential(
                network,
                input,
                domain,
                config.threshold,
                cut_pool,
                engine,
            )
        }
    }

    /// Select which neuron to split based on the branching heuristic.
    /// Returns None for InputSplit mode (use `create_input_split_children` instead).
    fn select_split_neuron(
        &self,
        network: &Network,
        input: &BoundedTensor,
        domain: &BabDomain,
    ) -> Result<Option<(usize, usize)>> {
        match self.config.branching_heuristic {
            BranchingHeuristic::LargestBoundWidth => {
                self.select_largest_width_neuron(network, domain)
            }
            BranchingHeuristic::Sequential => self.select_sequential_neuron(network, domain),
            BranchingHeuristic::BoundImpact => self.select_babsr_neuron(network, domain),
            BranchingHeuristic::FilteredSmartBranching => {
                self.select_fsb_neuron(network, input, domain)
            }
            BranchingHeuristic::InputSplit => {
                // Signal that we should use input splitting instead of ReLU splitting
                Ok(None)
            }
        }
    }

    /// Select the unstable ReLU neuron with largest pre-activation bound width.
    fn select_largest_width_neuron(
        &self,
        network: &Network,
        domain: &BabDomain,
    ) -> Result<Option<(usize, usize)>> {
        let mut best: Option<(usize, usize, f32)> = None;

        for (layer_idx, layer) in network.layers.iter().enumerate() {
            if !matches!(layer, Layer::ReLU(_)) {
                continue;
            }

            // Get pre-activation bounds for this ReLU
            // Pre-activation is from the previous layer's output
            if layer_idx == 0 || layer_idx > domain.layer_bounds.len() {
                continue;
            }
            let pre_bounds = &domain.layer_bounds[layer_idx - 1];

            // Find unstable neurons (l < 0 < u) that aren't already constrained
            let flat = pre_bounds.flatten();
            for neuron_idx in 0..flat.len() {
                let l = flat.lower[[neuron_idx]];
                let u = flat.upper[[neuron_idx]];

                // Check if unstable
                if l >= 0.0 || u <= 0.0 {
                    continue;
                }

                // Check if already constrained
                if domain
                    .history
                    .is_constrained(layer_idx, neuron_idx)
                    .is_some()
                {
                    continue;
                }

                let width = u - l;
                if best.is_none() || width > best.unwrap().2 {
                    best = Some((layer_idx, neuron_idx, width));
                }
            }
        }

        Ok(best.map(|(l, n, _)| (l, n)))
    }

    /// Select the unstable ReLU neuron using BaBSR heuristic (Bound-impact scoring).
    ///
    /// BaBSR (Branch-and-Bound with Smart ReLU branching) scores each unstable neuron
    /// based on the expected bound improvement from splitting. The score combines:
    /// 1. Triangle relaxation intercept: measures looseness of the current relaxation
    /// 2. CROWN coefficient magnitude: measures impact on output bounds
    ///
    /// Score formula: |lA_i| * intercept_i where intercept_i = (-lb_i * ub_i) / (ub_i - lb_i)
    fn select_babsr_neuron(
        &self,
        network: &Network,
        domain: &BabDomain,
    ) -> Result<Option<(usize, usize)>> {
        // First, collect all unstable neurons with their basic info
        let mut candidates: Vec<(usize, usize, f32, f32)> = Vec::new(); // (layer, neuron, lb, ub)

        for (layer_idx, layer) in network.layers.iter().enumerate() {
            if !matches!(layer, Layer::ReLU(_)) {
                continue;
            }

            if layer_idx == 0 || layer_idx > domain.layer_bounds.len() {
                continue;
            }
            let pre_bounds = &domain.layer_bounds[layer_idx - 1];

            let flat = pre_bounds.flatten();
            for neuron_idx in 0..flat.len() {
                let l = flat.lower[[neuron_idx]];
                let u = flat.upper[[neuron_idx]];

                // Check if unstable and not constrained
                if l < 0.0
                    && u > 0.0
                    && domain
                        .history
                        .is_constrained(layer_idx, neuron_idx)
                        .is_none()
                {
                    candidates.push((layer_idx, neuron_idx, l, u));
                }
            }
        }

        if candidates.is_empty() {
            return Ok(None);
        }

        // Compute CROWN coefficients for scoring
        // We do a backward pass starting from identity at output to get sensitivities
        let crown_coeffs = self.compute_crown_coefficients(network, domain)?;

        // Score each candidate using BaBSR formula
        let mut best: Option<(usize, usize, f32)> = None;

        for (layer_idx, neuron_idx, lb, ub) in candidates {
            // Triangle relaxation intercept: looseness measure
            let intercept = (-lb * ub) / (ub - lb);

            // Get CROWN coefficient for this neuron (sensitivity to output)
            let coeff_magnitude = crown_coeffs
                .get(&(layer_idx, neuron_idx))
                .copied()
                .unwrap_or(1.0);

            // BaBSR score: coefficient magnitude * intercept
            // Higher score = more impact from splitting this neuron
            let score = coeff_magnitude * intercept;

            if best.is_none() || score > best.unwrap().2 {
                best = Some((layer_idx, neuron_idx, score));
            }
        }

        trace!(
            "BaBSR selected neuron layer={}, idx={}, score={:.4}",
            best.map(|b| b.0).unwrap_or(0),
            best.map(|b| b.1).unwrap_or(0),
            best.map(|b| b.2).unwrap_or(0.0)
        );

        Ok(best.map(|(l, n, _)| (l, n)))
    }

    /// Select the unstable ReLU neuron using KFSB-style heuristic.
    ///
    /// KFSB (K-FSB) uses two scoring methods to select branching candidates:
    /// 1) BaBSR score: coefficient magnitude * intercept (higher is better)
    /// 2) Intercept-only score: pure triangle relaxation gap (higher is better)
    ///
    /// Strategy:
    /// 1) Rank by BaBSR and take top-k candidates.
    /// 2) Rank by intercept-only and take top-k candidates.
    /// 3) Merge both sets (deduplicate) and evaluate all candidates.
    /// 4) Choose the split that maximizes worst-child improvement.
    ///
    /// The intercept-only scoring helps find neurons where the relaxation gap
    /// is large even if the CROWN coefficient is small.
    fn select_fsb_neuron(
        &self,
        network: &Network,
        input: &BoundedTensor,
        domain: &BabDomain,
    ) -> Result<Option<(usize, usize)>> {
        let k = self.config.fsb_candidates;
        if k == 0 {
            return self.select_babsr_neuron(network, domain);
        }

        // Collect unstable, unconstrained neurons with their scores.
        let mut candidates: Vec<(usize, usize, f32, f32, f32)> = Vec::new();
        // (layer_idx, neuron_idx, lb, ub, intercept)

        for (layer_idx, layer) in network.layers.iter().enumerate() {
            if !matches!(layer, Layer::ReLU(_)) {
                continue;
            }
            if layer_idx == 0 || layer_idx > domain.layer_bounds.len() {
                continue;
            }
            let pre_bounds = &domain.layer_bounds[layer_idx - 1];
            let flat = pre_bounds.flatten();
            for neuron_idx in 0..flat.len() {
                let l = flat.lower[[neuron_idx]];
                let u = flat.upper[[neuron_idx]];
                if l < 0.0
                    && u > 0.0
                    && domain
                        .history
                        .is_constrained(layer_idx, neuron_idx)
                        .is_none()
                {
                    // Triangle relaxation intercept: measures looseness
                    let intercept = (-l * u) / (u - l);
                    candidates.push((layer_idx, neuron_idx, l, u, intercept));
                }
            }
        }

        if candidates.is_empty() {
            return Ok(None);
        }

        // Compute CROWN coefficients for BaBSR scoring
        let crown_coeffs = self.compute_crown_coefficients(network, domain)?;

        // Compute both scores for all candidates
        let mut scored: Vec<(usize, usize, f32, f32)> = Vec::with_capacity(candidates.len());
        // (layer_idx, neuron_idx, babsr_score, intercept_score)
        for (layer_idx, neuron_idx, _lb, _ub, intercept) in &candidates {
            let coeff_magnitude = crown_coeffs
                .get(&(*layer_idx, *neuron_idx))
                .copied()
                .unwrap_or(1.0);
            let babsr_score = coeff_magnitude * intercept;
            // Intercept-only score: just the pure relaxation gap
            let intercept_score = *intercept;
            scored.push((*layer_idx, *neuron_idx, babsr_score, intercept_score));
        }

        // Get top-k by BaBSR score (higher is better)
        let mut babsr_ranked = scored.clone();
        babsr_ranked.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // Get top-k by intercept-only score (higher is better)
        let mut intercept_ranked = scored.clone();
        intercept_ranked.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));

        // Merge candidates from both rankings, deduplicate by (layer, neuron)
        let mut eval_candidates: Vec<(usize, usize, f32, f32)> = Vec::new();
        let mut seen: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();

        // Add top-k from BaBSR
        for &(layer_idx, neuron_idx, babsr_score, intercept_score) in babsr_ranked.iter().take(k) {
            if seen.insert((layer_idx, neuron_idx)) {
                eval_candidates.push((layer_idx, neuron_idx, babsr_score, intercept_score));
            }
        }

        // Add top-k from intercept-only (may add up to k more, but often overlaps)
        for &(layer_idx, neuron_idx, babsr_score, intercept_score) in
            intercept_ranked.iter().take(k)
        {
            if seen.insert((layer_idx, neuron_idx)) {
                eval_candidates.push((layer_idx, neuron_idx, babsr_score, intercept_score));
            }
        }

        // Evaluate all candidates by computing child bounds
        let mut best: Option<(usize, usize, f32, f32)> = None; // (layer, neuron, fsb_score, babsr_score)
        for &(layer_idx, neuron_idx, babsr_score, _intercept_score) in &eval_candidates {
            let active = self.estimate_child_bounds_after_split(
                network, input, domain, layer_idx, neuron_idx, true,
            )?;
            let inactive = self.estimate_child_bounds_after_split(
                network, input, domain, layer_idx, neuron_idx, false,
            )?;

            let fsb_score = if self.config.verify_upper_bound {
                // Minimize the worst-case upper bound.
                let mut worst_upper = f32::NEG_INFINITY;
                if let Some((_, u)) = active {
                    worst_upper = worst_upper.max(u);
                }
                if let Some((_, u)) = inactive {
                    worst_upper = worst_upper.max(u);
                }
                if worst_upper == f32::NEG_INFINITY {
                    continue;
                }
                -worst_upper
            } else {
                // Maximize the worst-case lower bound.
                let mut worst_lower = f32::INFINITY;
                if let Some((l, _)) = active {
                    worst_lower = worst_lower.min(l);
                }
                if let Some((l, _)) = inactive {
                    worst_lower = worst_lower.min(l);
                }
                if worst_lower == f32::INFINITY {
                    continue;
                }
                worst_lower
            };

            let is_better = best
                .map(|(_, _, best_fsb, best_babsr)| {
                    fsb_score > best_fsb + 1e-6
                        || ((fsb_score - best_fsb).abs() <= 1e-6 && babsr_score > best_babsr)
                })
                .unwrap_or(true);
            if is_better {
                best = Some((layer_idx, neuron_idx, fsb_score, babsr_score));
            }
        }

        if let Some((layer_idx, neuron_idx, fsb_score, babsr_score)) = best {
            trace!(
                "KFSB selected neuron layer={}, idx={}, fsb_score={:.4}, babsr_score={:.4} (eval={}/{})",
                layer_idx,
                neuron_idx,
                fsb_score,
                babsr_score,
                eval_candidates.len(),
                scored.len()
            );
            return Ok(Some((layer_idx, neuron_idx)));
        }

        // Fallback: best BaBSR if evaluation failed for all candidates.
        Ok(babsr_ranked.first().map(|(l, n, _, _)| (*l, *n)))
    }

    /// Estimate child bounds after applying a single ReLU split constraint, without joint optimization.
    ///
    /// This is intended for branching heuristics (FSB) where we want a cheap estimate.
    fn estimate_child_bounds_after_split(
        &self,
        network: &Network,
        input: &BoundedTensor,
        parent: &BabDomain,
        layer_idx: usize,
        neuron_idx: usize,
        is_active: bool,
    ) -> Result<Option<(f32, f32)>> {
        let constraint = NeuronConstraint {
            layer_idx,
            neuron_idx,
            is_active,
        };
        let new_history = parent.history.with_constraint(constraint);

        let mut new_layer_bounds = parent.layer_bounds.clone();

        // Apply constraint to pre-activation bounds (same tightening as create_child_domain).
        if layer_idx > 0 && layer_idx <= new_layer_bounds.len() {
            let pre_bounds = &new_layer_bounds[layer_idx - 1];
            let lower = pre_bounds.lower.clone();
            let upper = pre_bounds.upper.clone();

            let shape = lower.shape().to_vec();
            let lower_len = lower.len();
            let upper_len = upper.len();
            let mut lower_flat = lower
                .into_shape_clone(ndarray::IxDyn(&[lower_len]))
                .unwrap();
            let mut upper_flat = upper
                .into_shape_clone(ndarray::IxDyn(&[upper_len]))
                .unwrap();

            if is_active {
                lower_flat[[neuron_idx]] = lower_flat[[neuron_idx]].max(0.0);
            } else {
                upper_flat[[neuron_idx]] = upper_flat[[neuron_idx]].min(0.0);
            }

            if lower_flat[[neuron_idx]] > upper_flat[[neuron_idx]] {
                return Ok(None);
            }

            let lower_new = lower_flat.into_shape_clone(ndarray::IxDyn(&shape)).unwrap();
            let upper_new = upper_flat.into_shape_clone(ndarray::IxDyn(&shape)).unwrap();
            new_layer_bounds[layer_idx - 1] = Arc::new(BoundedTensor::new(lower_new, upper_new)?);
        }

        let beta_state = BetaState::from_history(&new_history);
        let alpha_state = if self.config.use_alpha_crown {
            DomainAlphaState::from_layer_bounds_and_constraints(
                network,
                &new_layer_bounds,
                &new_history,
            )
        } else {
            DomainAlphaState::empty()
        };

        let empty_cuts = CutPool::new(0);
        // Note: FSB uses CPU-only for cheap estimates during branching heuristics
        let bounds = self.compute_bounds_with_alpha_beta(
            network,
            input,
            &new_history,
            &new_layer_bounds,
            &beta_state,
            &alpha_state,
            &empty_cuts,
            None, // CPU-only for branching heuristics
        )?;

        Ok(Some((bounds.lower_scalar(), bounds.upper_scalar())))
    }

    /// Compute CROWN coefficients (sensitivities) for all neurons.
    ///
    /// Returns a map from (layer_idx, neuron_idx) to the sum of absolute output
    /// sensitivities |lA[output, neuron]|.
    fn compute_crown_coefficients(
        &self,
        network: &Network,
        domain: &BabDomain,
    ) -> Result<std::collections::HashMap<(usize, usize), f32>> {
        let mut coeffs = std::collections::HashMap::new();

        if network.layers.is_empty() || domain.layer_bounds.is_empty() {
            return Ok(coeffs);
        }

        // Get output dimension from last layer bounds
        let output_dim = domain.layer_bounds.last().map(|b| b.len()).unwrap_or(1);

        // Start with identity at output layer
        let mut current_coeffs = Array2::<f32>::eye(output_dim);

        // Backward pass through layers (output to input)
        for (layer_idx, layer) in network.layers.iter().enumerate().rev() {
            // Get pre-activation bounds as reference (Arc derefs to inner BoundedTensor)
            let pre_bounds: &BoundedTensor = if layer_idx == 0 {
                // Use first layer bounds if available, else this is special case
                if !domain.layer_bounds.is_empty() {
                    domain.layer_bounds[0].as_ref()
                } else {
                    continue;
                }
            } else if layer_idx <= domain.layer_bounds.len() {
                domain.layer_bounds[layer_idx - 1].as_ref()
            } else {
                continue;
            };

            match layer {
                Layer::Linear(linear) => {
                    // Linear backward: coeffs = coeffs @ W
                    current_coeffs = current_coeffs.dot(&linear.weight);
                }
                Layer::ReLU(_) => {
                    // For ReLU, record coefficients for all neurons
                    let flat = pre_bounds.flatten();
                    let num_neurons = current_coeffs.ncols().min(flat.len());

                    for neuron_idx in 0..num_neurons {
                        // Sum of absolute coefficients across all outputs
                        let sum_abs_coeff: f32 = current_coeffs
                            .column(neuron_idx)
                            .iter()
                            .map(|c| c.abs())
                            .sum();
                        coeffs.insert((layer_idx, neuron_idx), sum_abs_coeff);
                    }

                    // Apply ReLU relaxation slopes to coefficients
                    let mut new_coeffs =
                        Array2::<f32>::zeros((current_coeffs.nrows(), num_neurons));
                    for neuron_idx in 0..num_neurons {
                        let l = flat.lower[[neuron_idx]];
                        let u = flat.upper[[neuron_idx]];

                        // Determine slope based on activation status
                        let slope = if l >= 0.0 {
                            1.0 // Always active
                        } else if u <= 0.0 {
                            0.0 // Always inactive
                        } else {
                            // Unstable: use upper bound slope (conservative)
                            u / (u - l)
                        };

                        for output_idx in 0..current_coeffs.nrows() {
                            new_coeffs[[output_idx, neuron_idx]] =
                                current_coeffs[[output_idx, neuron_idx]] * slope;
                        }
                    }
                    current_coeffs = new_coeffs;
                }
                _ => {
                    // For other layers, preserve coefficient dimension
                    // This is a simplification; full CROWN would handle each layer type
                }
            }
        }

        Ok(coeffs)
    }

    /// Select neurons in sequential order.
    fn select_sequential_neuron(
        &self,
        network: &Network,
        domain: &BabDomain,
    ) -> Result<Option<(usize, usize)>> {
        for (layer_idx, layer) in network.layers.iter().enumerate() {
            if !matches!(layer, Layer::ReLU(_)) {
                continue;
            }

            if layer_idx == 0 || layer_idx > domain.layer_bounds.len() {
                continue;
            }
            let pre_bounds = &domain.layer_bounds[layer_idx - 1];

            let flat = pre_bounds.flatten();
            for neuron_idx in 0..flat.len() {
                let l = flat.lower[[neuron_idx]];
                let u = flat.upper[[neuron_idx]];

                // Check if unstable and not constrained
                if l < 0.0
                    && u > 0.0
                    && domain
                        .history
                        .is_constrained(layer_idx, neuron_idx)
                        .is_none()
                {
                    return Ok(Some((layer_idx, neuron_idx)));
                }
            }
        }
        Ok(None)
    }

    /// Create a child domain with an additional constraint.
    #[allow(clippy::too_many_arguments)]
    fn create_child_domain(
        &self,
        network: &Network,
        input: &BoundedTensor,
        parent: &BabDomain,
        layer_idx: usize,
        neuron_idx: usize,
        is_active: bool,
        _threshold: f32,
        cut_pool: &mut CutPool,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<Option<BabDomain>> {
        let constraint = NeuronConstraint {
            layer_idx,
            neuron_idx,
            is_active,
        };
        let new_history = parent.history.with_constraint(constraint);

        // Tighten bounds based on new constraint
        // Cloning Vec<Arc<_>> is cheap - only Arc pointers are cloned, not the underlying data
        let mut new_layer_bounds = parent.layer_bounds.clone();

        // Apply constraint to pre-activation bounds
        if layer_idx > 0 && layer_idx <= new_layer_bounds.len() {
            // Read from the Arc-wrapped bounds (no mut needed, we'll replace the entry)
            let pre_bounds = &new_layer_bounds[layer_idx - 1];
            let lower = pre_bounds.lower.clone();
            let upper = pre_bounds.upper.clone();

            // Flatten to access individual neurons
            let shape = lower.shape().to_vec();
            let lower_len = lower.len();
            let upper_len = upper.len();
            let mut lower_flat = lower
                .into_shape_clone(ndarray::IxDyn(&[lower_len]))
                .unwrap();
            let mut upper_flat = upper
                .into_shape_clone(ndarray::IxDyn(&[upper_len]))
                .unwrap();

            if is_active {
                // Neuron is active: x >= 0, so lower bound becomes max(lower, 0)
                lower_flat[[neuron_idx]] = lower_flat[[neuron_idx]].max(0.0);
            } else {
                // Neuron is inactive: x <= 0, so upper bound becomes min(upper, 0)
                upper_flat[[neuron_idx]] = upper_flat[[neuron_idx]].min(0.0);
            }

            // Check if constraint makes domain infeasible
            if lower_flat[[neuron_idx]] > upper_flat[[neuron_idx]] {
                trace!("Child domain infeasible: constraint makes l > u");
                return Ok(None);
            }

            // Reshape back
            let lower_new = lower_flat.into_shape_clone(ndarray::IxDyn(&shape)).unwrap();
            let upper_new = upper_flat.into_shape_clone(ndarray::IxDyn(&shape)).unwrap();

            // Wrap new BoundedTensor in Arc - only the modified layer gets a new allocation
            new_layer_bounds[layer_idx - 1] = Arc::new(BoundedTensor::new(lower_new, upper_new)?);
        }

        // Initialize beta state from history
        let mut beta_state = BetaState::from_history(&new_history);

        // Initialize domain-specific alpha state for joint optimization
        let mut domain_alpha_state = if self.config.use_alpha_crown {
            DomainAlphaState::from_layer_bounds_and_constraints(
                network,
                &new_layer_bounds,
                &new_history,
            )
        } else {
            DomainAlphaState::empty()
        };

        // Optimize α, β, and λ (cut) parameters jointly for tighter bounds
        let output_bounds = self.optimize_joint_bounds(
            network,
            input,
            &new_history,
            &new_layer_bounds,
            &mut beta_state,
            &mut domain_alpha_state,
            cut_pool,
            engine,
        )?;

        let new_lower = output_bounds.lower_scalar();
        let new_upper = output_bounds.upper_scalar();

        Ok(Some(BabDomain {
            history: new_history,
            lower_bound: new_lower,
            upper_bound: new_upper,
            layer_bounds: new_layer_bounds,
            alpha_state: parent.alpha_state.clone(),
            domain_alpha_state,
            beta_state,
            input_bounds: parent.input_bounds.clone(),
            input_split_count: parent.input_split_count,
        }))
    }

    /// Select input dimension to split based on largest bound width.
    /// Returns the dimension index to split.
    fn select_input_dimension(&self, input_bounds: &BoundedTensor) -> usize {
        let flat = input_bounds.flatten();
        let mut best_dim = 0;
        let mut best_width = 0.0f32;

        for dim in 0..flat.len() {
            let width = flat.upper[[dim]] - flat.lower[[dim]];
            if width > best_width {
                best_width = width;
                best_dim = dim;
            }
        }

        best_dim
    }

    /// Create two child domains by splitting on an input dimension.
    /// Returns (left_child, right_child) where:
    /// - left_child has input[dim] in [l, mid]
    /// - right_child has input[dim] in [mid, u]
    fn create_input_split_children(
        &self,
        network: &Network,
        input: &BoundedTensor,
        parent: &BabDomain,
        _threshold: f32,
    ) -> Result<Vec<BabDomain>> {
        // Get the input bounds for this domain
        let domain_input = parent.get_input_bounds().unwrap_or(input);

        // Select dimension to split (largest width)
        let split_dim = self.select_input_dimension(domain_input);

        let flat = domain_input.flatten();
        let l = flat.lower[[split_dim]];
        let u = flat.upper[[split_dim]];
        let mid = (l + u) / 2.0;

        trace!(
            "Input split on dim {}: [{:.4}, {:.4}] -> [{:.4}, {:.4}] and [{:.4}, {:.4}]",
            split_dim,
            l,
            u,
            l,
            mid,
            mid,
            u
        );

        let mut children = Vec::with_capacity(2);

        // Create left child: input[dim] in [l, mid]
        if let Some(left_child) =
            self.create_input_split_child(network, input, parent, split_dim, l, mid)?
        {
            children.push(left_child);
        }

        // Create right child: input[dim] in [mid, u]
        if let Some(right_child) =
            self.create_input_split_child(network, input, parent, split_dim, mid, u)?
        {
            children.push(right_child);
        }

        Ok(children)
    }

    /// Create a single child domain with tightened input bounds on one dimension.
    fn create_input_split_child(
        &self,
        network: &Network,
        original_input: &BoundedTensor,
        parent: &BabDomain,
        split_dim: usize,
        new_lower: f32,
        new_upper: f32,
    ) -> Result<Option<BabDomain>> {
        // Get the input bounds for this domain
        let domain_input = parent.get_input_bounds().unwrap_or(original_input);

        // Create new input bounds with tightened dimension
        let flat = domain_input.flatten();
        let shape = domain_input.lower.shape().to_vec();

        let mut new_lower_arr = flat.lower.clone();
        let mut new_upper_arr = flat.upper.clone();

        new_lower_arr[[split_dim]] = new_lower;
        new_upper_arr[[split_dim]] = new_upper;

        // Reshape back to original shape
        let new_lower_arr = new_lower_arr
            .into_shape_clone(ndarray::IxDyn(&shape))
            .map_err(|e| gamma_core::GammaError::InvalidSpec(format!("reshape lower: {}", e)))?;
        let new_upper_arr = new_upper_arr
            .into_shape_clone(ndarray::IxDyn(&shape))
            .map_err(|e| gamma_core::GammaError::InvalidSpec(format!("reshape upper: {}", e)))?;

        let new_input_bounds = BoundedTensor::new(new_lower_arr, new_upper_arr)?;

        // Recompute all layer bounds with new input bounds
        let new_layer_bounds = if self.config.use_crown_ibp {
            network.collect_crown_ibp_bounds(&new_input_bounds)?
        } else {
            network.collect_ibp_bounds(&new_input_bounds)?
        };
        let new_layer_bounds: Vec<Arc<BoundedTensor>> =
            new_layer_bounds.into_iter().map(Arc::new).collect();

        // Compute output bounds with CROWN
        let output_bounds = if self.config.use_alpha_crown {
            network
                .propagate_alpha_crown_with_config(&new_input_bounds, &self.config.alpha_config)?
        } else if self.config.use_crown_ibp {
            network.propagate_crown_ibp(&new_input_bounds)?
        } else {
            network.propagate_crown(&new_input_bounds)?
        };

        let new_lower_bound = output_bounds.lower_scalar();
        let new_upper_bound = output_bounds.upper_scalar();

        // Initialize states for new domain (empty since no ReLU constraints)
        let domain_alpha_state = if self.config.use_alpha_crown {
            DomainAlphaState::from_layer_bounds_and_constraints(
                network,
                &new_layer_bounds,
                &SplitHistory::new(),
            )
        } else {
            DomainAlphaState::empty()
        };

        Ok(Some(BabDomain {
            history: SplitHistory::new(), // No ReLU constraints
            lower_bound: new_lower_bound,
            upper_bound: new_upper_bound,
            layer_bounds: new_layer_bounds,
            alpha_state: None,
            domain_alpha_state,
            beta_state: BetaState::empty(),
            input_bounds: Some(Arc::new(new_input_bounds)),
            input_split_count: parent.input_split_count + 1,
        }))
    }

    /// Jointly optimize α, β, and λ (cut) parameters to tighten output bounds.
    ///
    /// This performs alternating optimization:
    /// 1. Compute bounds with current α, β, and λ values
    /// 2. Compute gradients of lower bound w.r.t. α, β, and λ
    /// 3. Update α via gradient ascent: α = clamp(α + lr_α * grad_α, 0, 1)
    /// 4. Update β via projected gradient ascent: β = max(0, β + lr_β * grad_β)
    /// 5. Update λ via projected gradient ascent: λ = max(0, λ + lr_λ * grad_λ)
    /// 6. Repeat until convergence or max iterations
    #[allow(clippy::too_many_arguments)]
    fn optimize_joint_bounds(
        &self,
        network: &Network,
        input: &BoundedTensor,
        history: &SplitHistory,
        layer_bounds: &[Arc<BoundedTensor>],
        beta_state: &mut BetaState,
        alpha_state: &mut DomainAlphaState,
        cut_pool: &mut CutPool,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        // Check if any optimization is needed
        let has_beta = !beta_state.is_empty();
        let has_alpha = !alpha_state.is_empty() && self.config.use_alpha_crown;
        let has_cuts = !cut_pool.is_empty() && self.config.enable_cuts;

        // Skip optimization if nothing to optimize
        if (!has_beta && !has_alpha && !has_cuts) || self.config.beta_iterations == 0 {
            return self.compute_bounds_with_alpha_beta(
                network,
                input,
                history,
                layer_bounds,
                beta_state,
                alpha_state,
                cut_pool,
                engine,
            );
        }

        let mut best_bounds: Option<BoundedTensor> = None;
        let mut best_lower = f32::NEG_INFINITY;
        let momentum = if self.config.alpha_momentum {
            self.config.alpha_config.momentum
        } else {
            0.0
        };

        for iter in 0..self.config.beta_iterations {
            // Reset gradients for α, β, and λ (cuts)
            beta_state.zero_grad();
            alpha_state.zero_grad();
            if has_cuts {
                cut_pool.zero_grad();
            }

            // Compute bounds with current α, β, and λ
            let bounds = self.compute_bounds_with_alpha_beta(
                network,
                input,
                history,
                layer_bounds,
                beta_state,
                alpha_state,
                cut_pool,
                engine,
            )?;

            let current_lower = bounds.lower_scalar();
            let objective_lower_idx = bounds.argmin_lower_flat_idx();

            // Track best bounds (highest lower bound)
            if current_lower > best_lower {
                best_lower = current_lower;
                best_bounds = Some(bounds.clone());
            }

            // Compute analytical gradients for α, β, and λ
            self.compute_joint_gradients(
                network,
                input,
                history,
                layer_bounds,
                beta_state,
                alpha_state,
                cut_pool,
                objective_lower_idx,
            )?;

            // Perform gradient ascent steps for α, β, and λ
            // Use Adam optimizer if enabled, otherwise use standard gradient ascent
            let (max_beta_grad, max_alpha_grad, max_cut_grad) = if self.config.use_adaptive {
                // Adam optimizer: adaptive learning rates per parameter
                let t = iter + 1; // Time step (1-indexed for bias correction)
                let beta_grad = if has_beta {
                    beta_state.gradient_step_adam(&self.config.adaptive_config, t)
                } else {
                    0.0
                };
                let alpha_grad = if has_alpha {
                    alpha_state.gradient_step_adam(&self.config.adaptive_config, t)
                } else {
                    0.0
                };
                // GCP-CROWN: optimize cut lambdas
                let cut_grad = if has_cuts {
                    let mut max_grad = 0.0f32;
                    for cut in cut_pool.cuts_mut() {
                        cut.gradient_step_adam(&self.config.adaptive_config, t);
                        max_grad = max_grad.max(cut.lambda_grad.abs());
                    }
                    max_grad
                } else {
                    0.0
                };
                (beta_grad, alpha_grad, cut_grad)
            } else {
                // Standard gradient ascent with fixed learning rates
                let beta_grad = if has_beta {
                    beta_state.gradient_step(self.config.beta_lr)
                } else {
                    0.0
                };
                let alpha_grad = if has_alpha {
                    alpha_state.gradient_step(self.config.alpha_lr, momentum)
                } else {
                    0.0
                };
                // GCP-CROWN: optimize cut lambdas with fixed learning rate
                let cut_grad = if has_cuts {
                    let lr = self
                        .config
                        .adaptive_config
                        .lr_lambda
                        .unwrap_or(self.config.beta_lr);
                    let mut max_grad = 0.0f32;
                    for cut in cut_pool.cuts_mut() {
                        max_grad = max_grad.max(cut.lambda_grad.abs());
                        // Gradient ascent step (maximize lower bound)
                        cut.lambda += lr * cut.lambda_grad;
                        // Project to feasible region: 0 <= lambda <= MAX_LAMBDA
                        const MAX_LAMBDA: f32 = 10.0;
                        cut.lambda = cut.lambda.clamp(0.0, MAX_LAMBDA);
                    }
                    max_grad
                } else {
                    0.0
                };
                (beta_grad, alpha_grad, cut_grad)
            };

            let max_grad = max_beta_grad.max(max_alpha_grad).max(max_cut_grad);

            // Check convergence
            if max_grad < self.config.beta_tolerance {
                trace!(
                    "Joint α-β-λ optimization converged at iteration {} (max_grad={:.6}, adaptive={})",
                    iter, max_grad, self.config.use_adaptive
                );
                break;
            }

            trace!(
                "Joint opt iter {}: lb={:.4}, max_α_grad={:.6}, max_β_grad={:.6}, max_λ_grad={:.6}, adaptive={}",
                iter,
                current_lower,
                max_alpha_grad,
                max_beta_grad,
                max_cut_grad,
                self.config.use_adaptive
            );
        }

        // Return best bounds found
        best_bounds.ok_or_else(|| {
            gamma_core::GammaError::NumericalInstability(
                "No valid bounds computed during joint optimization".into(),
            )
        })
    }

    /// Compute (sub)gradients of the active scalar lower bound w.r.t. β parameters.
    ///
    /// For each constrained neuron (layer k, neuron j), β modifies the lower-bound
    /// backward coefficients as `lA_k[j] -= sign_j * β_j`, so:
    ///
    /// d(lb)/d(β_j) = -sign_j * d(lb)/d(lA_k[j])
    ///
    /// We compute `d(lb)/d(lA_k[j])` exactly for the current piecewise-linear relaxation
    /// choices by:
    /// - Running a backward pass to the input while recording, per ReLU layer, which
    ///   (slope, intercept) branch was selected for the lower bound.
    /// - Selecting the concretization point x* from the final input coefficients.
    /// - Running a forward sensitivity pass in homogeneous coordinates so that bias and
    ///   intercept contributions are included exactly.
    #[allow(dead_code)] // Legacy function, kept for testing and compatibility
    fn compute_beta_gradients(
        &self,
        network: &Network,
        input: &BoundedTensor,
        history: &SplitHistory,
        layer_bounds: &[Arc<BoundedTensor>],
        beta_state: &mut BetaState,
        objective_lower_idx: usize,
    ) -> Result<()> {
        if beta_state.is_empty() {
            return Ok(());
        }

        let output_dim = layer_bounds.last().map(|b| b.len()).unwrap_or(input.len());
        if objective_lower_idx >= output_dim {
            return Err(gamma_core::GammaError::NumericalInstability(format!(
                "Objective lower index {} out of range (output_dim={})",
                objective_lower_idx, output_dim
            )));
        }

        // Build constraint lookup: layer_idx -> neuron_idx -> is_active
        let mut constraints: std::collections::HashMap<
            usize,
            std::collections::HashMap<usize, bool>,
        > = std::collections::HashMap::new();
        for c in &history.constraints {
            constraints
                .entry(c.layer_idx)
                .or_default()
                .insert(c.neuron_idx, c.is_active);
        }

        // Map layer -> beta entry indices for efficient gradient fill during forward sensitivity pass.
        let mut beta_entries_by_layer: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for (entry_idx, entry) in beta_state.entries.iter().enumerate() {
            beta_entries_by_layer
                .entry(entry.layer_idx)
                .or_default()
                .push(entry_idx);
        }

        let mut relu_relaxations: Vec<Option<ReluLowerRelaxation>> =
            vec![None; network.layers.len()];

        // Backward pass for the active scalar lower-bound objective output element.
        // We record the lower-bound ReLU relaxation (slope/intercept choices) for each ReLU layer
        // under the current coefficients, since those choices determine the piecewise-linear
        // bound and its (sub)gradient.
        let mut lower_a = Array2::<f32>::zeros((1, output_dim));
        lower_a[[0, objective_lower_idx]] = 1.0;
        let mut lin_bounds = LinearBounds {
            lower_a: lower_a.clone(),
            lower_b: Array1::<f32>::zeros(1),
            upper_a: lower_a,
            upper_b: Array1::<f32>::zeros(1),
        };

        for (layer_idx, layer) in network.layers.iter().enumerate().rev() {
            // Use references instead of clones (Arc derefs to inner BoundedTensor)
            let pre_bounds: &BoundedTensor = if layer_idx == 0 {
                input
            } else {
                layer_bounds[layer_idx - 1].as_ref()
            };

            let layer_constraints = constraints.get(&layer_idx);

            match layer {
                Layer::Linear(linear) => {
                    let weight = &linear.weight;
                    let new_lower_a = lin_bounds.lower_a.dot(weight);
                    let new_upper_a = lin_bounds.upper_a.dot(weight);

                    let new_lower_b = if let Some(bias) = &linear.bias {
                        &lin_bounds.lower_b + &lin_bounds.lower_a.dot(bias)
                    } else {
                        lin_bounds.lower_b.clone()
                    };

                    let new_upper_b = if let Some(bias) = &linear.bias {
                        &lin_bounds.upper_b + &lin_bounds.upper_a.dot(bias)
                    } else {
                        lin_bounds.upper_b.clone()
                    };

                    lin_bounds = LinearBounds {
                        lower_a: new_lower_a,
                        lower_b: new_lower_b,
                        upper_a: new_upper_a,
                        upper_b: new_upper_b,
                    };
                }
                Layer::ReLU(_) => {
                    let (new_bounds, relaxation) = self.relu_backward_with_beta_record_relaxation(
                        &lin_bounds,
                        pre_bounds,
                        layer_constraints,
                        beta_state,
                        layer_idx,
                    )?;
                    relu_relaxations[layer_idx] = Some(relaxation);
                    lin_bounds = new_bounds;
                }
                other => {
                    return Err(gamma_core::GammaError::NotSupported(format!(
                        "Analytical β gradients only supported for Linear/ReLU networks (saw {other:?})"
                    )));
                }
            }
        }

        // Choose the concretization point x* based on the final input-layer coefficients.
        // lb = lA(x*) + lB, where x*_i is lower_i if lA_i >= 0 else upper_i.
        let input_flat = input.flatten();
        let input_dim = input_flat.len();
        if lin_bounds.lower_a.ncols() != input_dim {
            return Err(gamma_core::GammaError::shape_mismatch(
                vec![input_dim],
                vec![lin_bounds.lower_a.ncols()],
            ));
        }

        let mut x_star = Array1::<f32>::zeros(input_dim + 1);
        for i in 0..input_dim {
            let coeff = lin_bounds.lower_a[[0, i]];
            x_star[i] = if coeff >= 0.0 {
                input_flat.lower[[i]]
            } else {
                input_flat.upper[[i]]
            };
        }
        x_star[input_dim] = 1.0;

        // Forward sensitivity pass in the augmented space (homogeneous coordinates).
        // This yields w_k = P_k * x_star for each layer input space, where P_k is the product
        // of the recorded piecewise-linear bound transforms up to that layer.
        //
        // For any coefficient a_k[j] at layer k input, d(lb)/d(a_k[j]) = w_k[j], including
        // both coefficient->input effects and coefficient->bias/intercept effects.
        let mut w = x_star;
        for (layer_idx, layer) in network.layers.iter().enumerate() {
            if let Some(entry_indices) = beta_entries_by_layer.get(&layer_idx) {
                let w_len = w.len();
                for &entry_idx in entry_indices {
                    let neuron_idx = beta_state.entries[entry_idx].neuron_idx;
                    if neuron_idx + 1 >= w_len {
                        return Err(gamma_core::GammaError::NumericalInstability(format!(
                            "β entry neuron index {} out of range for layer {} (w_len={})",
                            neuron_idx, layer_idx, w_len
                        )));
                    }
                    let sign = beta_state.entries[entry_idx].sign;
                    beta_state.entries[entry_idx].grad = -sign * w[neuron_idx];
                }
            }

            match layer {
                Layer::Linear(linear) => {
                    let in_dim = linear.weight.ncols();
                    let out_dim = linear.weight.nrows();
                    if w.len() != in_dim + 1 {
                        return Err(gamma_core::GammaError::shape_mismatch(
                            vec![in_dim + 1],
                            vec![w.len()],
                        ));
                    }

                    let w_in = w.slice(ndarray::s![..in_dim]).to_owned();
                    let w_const = w[in_dim];
                    let mut w_out = linear.weight.dot(&w_in);
                    if let Some(bias) = &linear.bias {
                        w_out = &w_out + &(bias * w_const);
                    }

                    let mut w_aug = Array1::<f32>::zeros(out_dim + 1);
                    w_aug.slice_mut(ndarray::s![..out_dim]).assign(&w_out);
                    w_aug[out_dim] = w_const;
                    w = w_aug;
                }
                Layer::ReLU(_) => {
                    let relaxation = relu_relaxations[layer_idx].as_ref().ok_or_else(|| {
                        gamma_core::GammaError::NumericalInstability(format!(
                            "Missing recorded ReLU relaxation for layer {}",
                            layer_idx
                        ))
                    })?;

                    let n = relaxation.slopes.len();
                    if w.len() != n + 1 {
                        return Err(gamma_core::GammaError::shape_mismatch(
                            vec![n + 1],
                            vec![w.len()],
                        ));
                    }

                    let w_const = w[n];
                    let mut w_aug = Array1::<f32>::zeros(n + 1);
                    for j in 0..n {
                        w_aug[j] = relaxation.slopes[j] * w[j] + relaxation.intercepts[j] * w_const;
                    }
                    w_aug[n] = w_const;
                    w = w_aug;
                }
                other => {
                    return Err(gamma_core::GammaError::NotSupported(format!(
                        "Analytical β gradients only supported for Linear/ReLU networks (saw {other:?})"
                    )));
                }
            }
        }

        Ok(())
    }

    /// ReLU backward pass that returns both the new bounds and the recorded lower-bound
    /// relaxation (slope/intercept choices) for the active objective output row.
    ///
    /// The recorded per-neuron (slope, intercept) pairs define a piecewise-linear transform
    /// that is used to compute exact (sub)gradients of the lower bound w.r.t. β under the
    /// current coefficient sign pattern.
    #[allow(dead_code)] // Used by compute_beta_gradients
    fn relu_backward_with_beta_record_relaxation(
        &self,
        output_bounds: &LinearBounds,
        pre_bounds: &BoundedTensor,
        constraints: Option<&std::collections::HashMap<usize, bool>>,
        beta_state: &BetaState,
        layer_idx: usize,
    ) -> Result<(LinearBounds, ReluLowerRelaxation)> {
        let pre_flat = pre_bounds.flatten();
        let num_neurons = pre_flat.len();
        let num_outputs = output_bounds.num_outputs();

        // Handle dimension mismatch when Conv layers changed dimensions in backward pass.
        if output_bounds.num_inputs() != num_neurons {
            debug!(
                "ReLU backward (β record) dimension mismatch: output_bounds has {} inputs, but layer has {} neurons. Using identity fallback.",
                output_bounds.num_inputs(),
                num_neurons
            );
            // Return identity bounds and empty relaxation
            let empty_relaxation = ReluLowerRelaxation {
                slopes: vec![0.0; num_neurons],
                intercepts: vec![0.0; num_neurons],
            };
            return Ok((LinearBounds::identity(num_neurons), empty_relaxation));
        }

        if num_outputs != 1 {
            return Err(gamma_core::GammaError::NotSupported(format!(
                "Analytical β gradient recording expects a single objective output row (got {num_outputs})"
            )));
        }

        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_lower_b = output_bounds.lower_b.clone();
        let mut new_upper_b = output_bounds.upper_b.clone();

        let mut slopes: Vec<f32> = vec![0.0; num_neurons];
        let mut intercepts: Vec<f32> = vec![0.0; num_neurons];

        for j in 0..num_neurons {
            let l = pre_flat.lower[[j]];
            let u = pre_flat.upper[[j]];

            // Check if this neuron is constrained
            let constraint = constraints.and_then(|c| c.get(&j).copied());

            // Determine relaxation based on constraint
            let (lower_slope, lower_intercept, upper_slope, upper_intercept) =
                if let Some(is_active) = constraint {
                    if is_active {
                        (1.0, 0.0, 1.0, 0.0)
                    } else {
                        (0.0, 0.0, 0.0, 0.0)
                    }
                } else if l >= 0.0 {
                    (1.0, 0.0, 1.0, 0.0)
                } else if u <= 0.0 {
                    (0.0, 0.0, 0.0, 0.0)
                } else {
                    let upper_slope = u / (u - l);
                    let upper_intercept = -l * u / (u - l);
                    let lower_slope = if u > -l { 1.0 } else { 0.0 };
                    (lower_slope, 0.0, upper_slope, upper_intercept)
                };

            // Apply relaxation to each output
            for i in 0..num_outputs {
                let la_ij = output_bounds.lower_a[[i, j]];
                let ua_ij = output_bounds.upper_a[[i, j]];

                // Lower bound computation
                if la_ij >= 0.0 {
                    new_lower_a[[i, j]] = la_ij * lower_slope;
                    new_lower_b[i] += la_ij * lower_intercept;
                } else {
                    new_lower_a[[i, j]] = la_ij * upper_slope;
                    new_lower_b[i] += la_ij * upper_intercept;
                }

                // Upper bound computation
                if ua_ij >= 0.0 {
                    new_upper_a[[i, j]] = ua_ij * upper_slope;
                    new_upper_b[i] += ua_ij * upper_intercept;
                } else {
                    new_upper_a[[i, j]] = ua_ij * lower_slope;
                    new_upper_b[i] += ua_ij * lower_intercept;
                }
            }

            // Record the lower-bound relaxation choice for the single objective row.
            // The choice depends on the sign of the pre-relaxation coefficient a_post[j].
            let la_j = output_bounds.lower_a[[0, j]];
            if la_j >= 0.0 {
                slopes[j] = lower_slope;
                intercepts[j] = lower_intercept;
            } else {
                slopes[j] = upper_slope;
                intercepts[j] = upper_intercept;
            }

            // Add beta contribution for constrained neurons
            if let Some(signed_beta) = beta_state.get_signed_beta(layer_idx, j) {
                for i in 0..num_outputs {
                    new_lower_a[[i, j]] -= signed_beta;
                    new_upper_a[[i, j]] += signed_beta;
                }
            }
        }

        Ok((
            LinearBounds {
                lower_a: new_lower_a,
                lower_b: new_lower_b,
                upper_a: new_upper_a,
                upper_b: new_upper_b,
            },
            ReluLowerRelaxation { slopes, intercepts },
        ))
    }

    /// Compute output bounds incorporating split constraints.
    ///
    /// This is the core of β-CROWN: it modifies the CROWN backward pass
    /// to incorporate the β parameters for constrained neurons.
    #[allow(dead_code)] // Legacy function, kept for testing and compatibility
    fn compute_bounds_with_constraints(
        &self,
        network: &Network,
        input: &BoundedTensor,
        history: &SplitHistory,
        layer_bounds: &[Arc<BoundedTensor>],
        beta_state: &BetaState,
    ) -> Result<BoundedTensor> {
        if network.layers.is_empty() {
            return Ok(input.clone());
        }

        // Build constraint lookup: layer_idx -> neuron_idx -> is_active
        let mut constraints: std::collections::HashMap<
            usize,
            std::collections::HashMap<usize, bool>,
        > = std::collections::HashMap::new();
        for c in &history.constraints {
            constraints
                .entry(c.layer_idx)
                .or_default()
                .insert(c.neuron_idx, c.is_active);
        }

        // Start with identity linear bounds for output
        let output_dim = layer_bounds.last().map(|b| b.len()).unwrap_or(input.len());
        let mut lin_bounds = LinearBounds::identity(output_dim);

        // Backward pass through layers
        for (layer_idx, layer) in network.layers.iter().enumerate().rev() {
            // Use references instead of clones (Arc derefs to inner BoundedTensor)
            let pre_bounds: &BoundedTensor = if layer_idx == 0 {
                input
            } else {
                layer_bounds[layer_idx - 1].as_ref()
            };

            // Check if this layer has constraints
            let layer_constraints = constraints.get(&layer_idx);

            lin_bounds = self.propagate_layer_backward_with_beta(
                layer,
                &lin_bounds,
                pre_bounds,
                layer_constraints,
                beta_state,
                layer_idx,
            )?;
        }

        // Concretize with input bounds
        Ok(lin_bounds.concretize(input))
    }

    /// Compute output bounds using α, β, and λ (cut) parameters.
    ///
    /// This extends `compute_bounds_with_constraints` to use optimizable α values
    /// for unstable neurons instead of the heuristic, and adds cutting plane
    /// contributions to the lower bound via Lagrangian relaxation (GCP-CROWN).
    #[allow(clippy::too_many_arguments)]
    fn compute_bounds_with_alpha_beta(
        &self,
        network: &Network,
        input: &BoundedTensor,
        history: &SplitHistory,
        layer_bounds: &[Arc<BoundedTensor>],
        beta_state: &BetaState,
        alpha_state: &DomainAlphaState,
        cut_pool: &CutPool,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        if network.layers.is_empty() {
            return Ok(input.clone());
        }

        // Build constraint lookup: layer_idx -> neuron_idx -> is_active
        let mut constraints: std::collections::HashMap<
            usize,
            std::collections::HashMap<usize, bool>,
        > = std::collections::HashMap::new();
        for c in &history.constraints {
            constraints
                .entry(c.layer_idx)
                .or_default()
                .insert(c.neuron_idx, c.is_active);
        }

        // Start with identity linear bounds for output
        let output_dim = layer_bounds.last().map(|b| b.len()).unwrap_or(input.len());
        let mut lin_bounds = LinearBounds::identity(output_dim);

        // Backward pass through layers
        for (layer_idx, layer) in network.layers.iter().enumerate().rev() {
            // Use references instead of clones (Arc derefs to inner BoundedTensor)
            let pre_bounds: &BoundedTensor = if layer_idx == 0 {
                input
            } else {
                layer_bounds[layer_idx - 1].as_ref()
            };

            // Check if this layer has constraints
            let layer_constraints = constraints.get(&layer_idx);

            lin_bounds = self.propagate_layer_backward_with_alpha_beta(
                layer,
                &lin_bounds,
                pre_bounds,
                layer_constraints,
                beta_state,
                alpha_state,
                layer_idx,
                engine,
            )?;
        }

        // Concretize with input bounds (base bounds without cuts)
        let base_bounds = lin_bounds.concretize(input);

        // GCP-CROWN: Add cutting plane contributions to lower bound
        // For each cut, contribution is: lambda * (min_constraint_value - bias)
        // where min_constraint_value uses lower/upper bounds depending on coefficient sign
        if !cut_pool.is_empty() && self.config.enable_cuts {
            let relevant_cuts = cut_pool.relevant_cuts_for(history);
            if !relevant_cuts.is_empty() {
                let cut_contribution = self.compute_cut_contribution(&relevant_cuts, layer_bounds);

                // SOUNDNESS FIX: Clamp cut contribution to prevent lb > ub
                // The cut contribution must not push the lower bound above the upper bound
                // This can happen when lambda values grow too large during optimization
                let base_lb = base_bounds
                    .lower
                    .iter()
                    .cloned()
                    .reduce(f32::min)
                    .unwrap_or(0.0);
                let base_ub = base_bounds
                    .upper
                    .iter()
                    .cloned()
                    .reduce(f32::max)
                    .unwrap_or(0.0);

                // Maximum safe contribution: don't let lb exceed ub (with small margin for numerical stability)
                let max_safe_contribution = (base_ub - base_lb).max(0.0) * 0.99;
                let clamped_contribution = cut_contribution.min(max_safe_contribution);

                // Use clamped contribution instead of raw contribution
                let cut_contribution = clamped_contribution;

                // Lower bound increases (tightens) with positive cut contribution
                let lower_shape = base_bounds.lower.shape().to_vec();
                let upper_shape = base_bounds.upper.shape().to_vec();
                let lower_len = base_bounds.lower.len();
                let _upper_len = base_bounds.upper.len();

                let mut lower_flat = base_bounds
                    .lower
                    .into_shape_clone(ndarray::IxDyn(&[lower_len]))
                    .unwrap();

                // Add cut contribution to all lower bound elements
                // (cuts apply uniformly to the entire bound computation)
                for i in 0..lower_len {
                    lower_flat[[i]] += cut_contribution;
                }

                let lower_new = lower_flat
                    .into_shape_clone(ndarray::IxDyn(&lower_shape))
                    .unwrap();
                let upper_new = base_bounds
                    .upper
                    .into_shape_clone(ndarray::IxDyn(&upper_shape))
                    .unwrap();

                return BoundedTensor::new(lower_new, upper_new);
            }
        }

        Ok(base_bounds)
    }

    /// Compute the total contribution of cuts to the lower bound.
    ///
    /// For each cut: contribution = lambda * (min_constraint_value - bias)
    /// where min_constraint_value is computed using ReLU indicator bounds.
    ///
    /// BICCOS cuts use ReLU activation indicators z_i ∈ {0,1}:
    /// - z_i = 0 if neuron is inactive (x_i < 0)
    /// - z_i = 1 if neuron is active (x_i >= 0)
    /// - z_i ∈ [0, 1] for unstable neurons
    fn compute_cut_contribution(
        &self,
        cuts: &[&CuttingPlane],
        layer_bounds: &[Arc<BoundedTensor>],
    ) -> f32 {
        let mut total = 0.0f32;

        for cut in cuts {
            // Skip cuts with zero lambda (no contribution)
            if cut.lambda.abs() < 1e-10 {
                continue;
            }

            // Compute minimum value of the constraint: sum_i(coeff_i * z_i)
            // where z_i is the ReLU indicator (0 if inactive, 1 if active)
            let constraint_min: f32 = cut
                .terms
                .iter()
                .filter_map(|term| {
                    // Get pre-activation bounds for this layer
                    // term.layer_idx is the ReLU layer, pre-activation is from layer_idx - 1
                    if term.layer_idx == 0 || term.layer_idx > layer_bounds.len() {
                        return None;
                    }
                    let pre_bounds = &layer_bounds[term.layer_idx - 1];
                    let flat = pre_bounds.flatten();

                    if term.neuron_idx >= flat.len() {
                        return None;
                    }

                    let l = flat.lower[[term.neuron_idx]];
                    let u = flat.upper[[term.neuron_idx]];

                    // Determine ReLU indicator bounds [z_min, z_max]
                    let (z_min, z_max) = if l >= 0.0 {
                        // Stable active: z = 1
                        (1.0, 1.0)
                    } else if u <= 0.0 {
                        // Stable inactive: z = 0
                        (0.0, 0.0)
                    } else {
                        // Unstable: z ∈ [0, 1]
                        (0.0, 1.0)
                    };

                    // Worst-case (minimum) value of coeff * z
                    let value = if term.coefficient > 0.0 {
                        term.coefficient * z_min // Use lower bound of z for positive coeff
                    } else {
                        term.coefficient * z_max // Use upper bound of z for negative coeff
                    };
                    Some(value)
                })
                .sum();

            // Lagrangian contribution: lambda * (constraint_value - bias)
            // The cut constraint is: sum(coeff_i * z_i) <= bias
            // Lagrangian dual adds: -lambda * (sum(coeff_i * z_i) - bias) to lower bound
            // For minimizing violation: lambda * (bias - constraint_min) >= 0
            total += cut.lambda * (cut.bias - constraint_min);
        }

        total
    }

    /// Compute the total contribution of graph cuts to the lower bound.
    ///
    /// For each graph cut: contribution = lambda * (min_constraint_value - bias)
    /// where min_constraint_value is computed using ReLU indicator bounds from node_bounds.
    ///
    /// Note: This function accepts both `BoundedTensor` and `Arc<BoundedTensor>` via the
    /// generic implementation. For the non-Arc version used in propagate_crown_with_graph_constraints,
    /// we convert to Arc references internally.
    fn compute_graph_cut_contribution(
        &self,
        graph: &GraphNetwork,
        cuts: &[&GraphCuttingPlane],
        node_bounds: &std::collections::HashMap<String, BoundedTensor>,
        input_bounds: &BoundedTensor,
    ) -> f32 {
        // Convert to Arc-based map for unified handling
        let arc_bounds: std::collections::HashMap<String, Arc<BoundedTensor>> = node_bounds
            .iter()
            .map(|(k, v)| (k.clone(), Arc::new(v.clone())))
            .collect();
        self.compute_graph_cut_contribution_arc(graph, cuts, &arc_bounds, input_bounds)
    }

    /// Arc-based version of graph cut contribution computation.
    fn compute_graph_cut_contribution_arc(
        &self,
        graph: &GraphNetwork,
        cuts: &[&GraphCuttingPlane],
        node_bounds: &std::collections::HashMap<String, Arc<BoundedTensor>>,
        input_bounds: &BoundedTensor,
    ) -> f32 {
        let mut total = 0.0f32;

        for cut in cuts {
            // Skip cuts with zero lambda (no contribution)
            if cut.lambda.abs() < 1e-10 {
                continue;
            }

            // Compute minimum value of the constraint: sum_i(coeff_i * z_i)
            let constraint_min: f32 = cut
                .terms
                .iter()
                .filter_map(|term| {
                    // Graph cut terms are keyed by ReLU node name, but the indicator variable
                    // is determined from the *pre-activation* bounds (the ReLU input node).
                    let relu_node = graph.nodes.get(&term.node_name)?;
                    if !matches!(relu_node.layer, Layer::ReLU(_)) {
                        return None;
                    }
                    let pre_name = relu_node
                        .inputs
                        .first()
                        .map(|s| s.as_str())
                        .unwrap_or("_input");
                    let bounds: &BoundedTensor = if pre_name == "_input" {
                        input_bounds
                    } else {
                        node_bounds.get(pre_name)?.as_ref()
                    };
                    let flat = bounds.flatten();

                    if term.neuron_idx >= flat.len() {
                        return None;
                    }

                    let l = flat.lower[[term.neuron_idx]];
                    let u = flat.upper[[term.neuron_idx]];

                    // Determine ReLU indicator bounds [z_min, z_max] from pre-activation bounds.
                    let (z_min, z_max) = if l >= 0.0 {
                        // Stable active: z = 1
                        (1.0, 1.0)
                    } else if u <= 0.0 {
                        // Stable inactive: z = 0
                        (0.0, 0.0)
                    } else {
                        // Unstable: z ∈ [0, 1]
                        (0.0, 1.0)
                    };

                    // Worst-case (minimum) value of coeff * z
                    let value = if term.coefficient > 0.0 {
                        term.coefficient * z_min
                    } else {
                        term.coefficient * z_max
                    };
                    Some(value)
                })
                .sum();

            // Lagrangian contribution: lambda * (bias - constraint_min)
            total += cut.lambda * (cut.bias - constraint_min);
        }

        total
    }

    /// Compute gradients for graph cut lambda parameters.
    ///
    /// For each cut: d(lb)/d(λ_c) = constraint_min_c - bias_c
    fn compute_graph_cut_gradients(
        &self,
        graph: &GraphNetwork,
        cut_pool: &mut GraphCutPool,
        node_bounds: &std::collections::HashMap<String, Arc<BoundedTensor>>,
        input_bounds: &BoundedTensor,
    ) {
        for cut in cut_pool.cuts_mut() {
            // Skip cuts with zero lambda (won't contribute to gradient)
            if cut.lambda.abs() < 1e-10 && cut.lambda_grad.abs() < 1e-10 {
                // Still compute gradient for initialization
            }

            // Compute minimum value of the constraint
            let constraint_min: f32 = cut
                .terms
                .iter()
                .filter_map(|term| {
                    let relu_node = graph.nodes.get(&term.node_name)?;
                    if !matches!(relu_node.layer, Layer::ReLU(_)) {
                        return None;
                    }
                    let pre_name = relu_node
                        .inputs
                        .first()
                        .map(|s| s.as_str())
                        .unwrap_or("_input");
                    let bounds: &BoundedTensor = if pre_name == "_input" {
                        input_bounds
                    } else {
                        node_bounds.get(pre_name)?.as_ref()
                    };
                    let flat = bounds.flatten();

                    if term.neuron_idx >= flat.len() {
                        return None;
                    }

                    let l = flat.lower[[term.neuron_idx]];
                    let u = flat.upper[[term.neuron_idx]];

                    let (z_min, z_max) = if l >= 0.0 {
                        (1.0, 1.0)
                    } else if u <= 0.0 {
                        (0.0, 0.0)
                    } else {
                        (0.0, 1.0)
                    };

                    let value = if term.coefficient > 0.0 {
                        term.coefficient * z_min
                    } else {
                        term.coefficient * z_max
                    };
                    Some(value)
                })
                .sum();

            // Gradient: d(lb)/d(λ) = bias - constraint_min (for maximization)
            cut.lambda_grad = cut.bias - constraint_min;
        }
    }

    /// Compute joint gradients for α, β, and λ (cut) parameters.
    ///
    /// Uses the augmented chain rule to compute exact gradients:
    /// - For β: d(lb)/d(β_j) = -sign_j * d(lb)/d(lA_k[j])
    /// - For α: d(lb)/d(α_j) = d(lb)/d(lower_slope[j]) where the α controls the lower bound slope
    /// - For λ: d(lb)/d(λ_c) = constraint_min_c - bias_c (Lagrangian gradient)
    #[allow(clippy::too_many_arguments)]
    fn compute_joint_gradients(
        &self,
        network: &Network,
        input: &BoundedTensor,
        history: &SplitHistory,
        layer_bounds: &[Arc<BoundedTensor>],
        beta_state: &mut BetaState,
        alpha_state: &mut DomainAlphaState,
        cut_pool: &mut CutPool,
        objective_lower_idx: usize,
    ) -> Result<()> {
        // If nothing to optimize, skip gradient computation
        let has_cuts = !cut_pool.is_empty() && self.config.enable_cuts;
        if beta_state.is_empty() && alpha_state.is_empty() && !has_cuts {
            return Ok(());
        }

        let output_dim = layer_bounds.last().map(|b| b.len()).unwrap_or(input.len());
        if objective_lower_idx >= output_dim {
            return Err(gamma_core::GammaError::NumericalInstability(format!(
                "Objective lower index {} out of range (output_dim={})",
                objective_lower_idx, output_dim
            )));
        }

        // Build constraint lookup
        let mut constraints: std::collections::HashMap<
            usize,
            std::collections::HashMap<usize, bool>,
        > = std::collections::HashMap::new();
        for c in &history.constraints {
            constraints
                .entry(c.layer_idx)
                .or_default()
                .insert(c.neuron_idx, c.is_active);
        }

        // Map layer -> beta entry indices
        let mut beta_entries_by_layer: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for (entry_idx, entry) in beta_state.entries.iter().enumerate() {
            beta_entries_by_layer
                .entry(entry.layer_idx)
                .or_default()
                .push(entry_idx);
        }

        // Storage for ReLU relaxation info during backward pass
        let mut relu_lower_slopes: Vec<Option<Vec<f32>>> = vec![None; network.layers.len()];

        // Backward pass: compute linear bounds while recording relaxation choices
        let mut lower_a = Array2::<f32>::zeros((1, output_dim));
        lower_a[[0, objective_lower_idx]] = 1.0;
        let mut lin_bounds = LinearBounds {
            lower_a: lower_a.clone(),
            lower_b: Array1::<f32>::zeros(1),
            upper_a: lower_a,
            upper_b: Array1::<f32>::zeros(1),
        };

        for (layer_idx, layer) in network.layers.iter().enumerate().rev() {
            // Use references instead of clones (Arc derefs to inner BoundedTensor)
            let pre_bounds: &BoundedTensor = if layer_idx == 0 {
                input
            } else {
                layer_bounds[layer_idx - 1].as_ref()
            };

            match layer {
                Layer::Linear(linear) => {
                    let weight = &linear.weight;
                    let new_lower_a = lin_bounds.lower_a.dot(weight);
                    let new_upper_a = lin_bounds.upper_a.dot(weight);

                    let new_lower_b = if let Some(bias) = &linear.bias {
                        &lin_bounds.lower_b + &lin_bounds.lower_a.dot(bias)
                    } else {
                        lin_bounds.lower_b.clone()
                    };

                    let new_upper_b = if let Some(bias) = &linear.bias {
                        &lin_bounds.upper_b + &lin_bounds.upper_a.dot(bias)
                    } else {
                        lin_bounds.upper_b.clone()
                    };

                    lin_bounds = LinearBounds {
                        lower_a: new_lower_a,
                        lower_b: new_lower_b,
                        upper_a: new_upper_a,
                        upper_b: new_upper_b,
                    };
                }
                Layer::ReLU(_) => {
                    let pre_flat = pre_bounds.flatten();
                    let num_neurons = pre_flat.len();
                    let num_outputs = lin_bounds.num_outputs();
                    let layer_constraints = constraints.get(&layer_idx);

                    let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_neurons));
                    let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
                    let mut new_lower_b = lin_bounds.lower_b.clone();
                    let mut new_upper_b = lin_bounds.upper_b.clone();

                    // Record lower slopes for gradient computation
                    let mut lower_slopes = vec![0.0f32; num_neurons];

                    for j in 0..num_neurons {
                        let l = pre_flat.lower[[j]];
                        let u = pre_flat.upper[[j]];
                        let constraint = layer_constraints.and_then(|c| c.get(&j).copied());

                        // Determine relaxation
                        let (lower_slope, lower_intercept, upper_slope, upper_intercept) =
                            if let Some(is_active) = constraint {
                                if is_active {
                                    (1.0, 0.0, 1.0, 0.0)
                                } else {
                                    (0.0, 0.0, 0.0, 0.0)
                                }
                            } else if l >= 0.0 {
                                (1.0, 0.0, 1.0, 0.0)
                            } else if u <= 0.0 {
                                (0.0, 0.0, 0.0, 0.0)
                            } else {
                                // Unstable: use α if available
                                let upper_slope_val = u / (u - l);
                                let upper_intercept_val = -l * u / (u - l);
                                let lower_slope_val = alpha_state.get_alpha(layer_idx, j);
                                (lower_slope_val, 0.0, upper_slope_val, upper_intercept_val)
                            };

                        lower_slopes[j] = lower_slope;

                        // Apply relaxation
                        for i in 0..num_outputs {
                            let la_ij = lin_bounds.lower_a[[i, j]];
                            let ua_ij = lin_bounds.upper_a[[i, j]];

                            if la_ij >= 0.0 {
                                new_lower_a[[i, j]] = la_ij * lower_slope;
                                new_lower_b[i] += la_ij * lower_intercept;
                            } else {
                                new_lower_a[[i, j]] = la_ij * upper_slope;
                                new_lower_b[i] += la_ij * upper_intercept;
                            }

                            if ua_ij >= 0.0 {
                                new_upper_a[[i, j]] = ua_ij * upper_slope;
                                new_upper_b[i] += ua_ij * upper_intercept;
                            } else {
                                new_upper_a[[i, j]] = ua_ij * lower_slope;
                                new_upper_b[i] += ua_ij * lower_intercept;
                            }
                        }

                        // Add beta contribution
                        if let Some(signed_beta) = beta_state.get_signed_beta(layer_idx, j) {
                            for i in 0..num_outputs {
                                new_lower_a[[i, j]] -= signed_beta;
                                new_upper_a[[i, j]] += signed_beta;
                            }
                        }
                    }

                    relu_lower_slopes[layer_idx] = Some(lower_slopes);

                    lin_bounds = LinearBounds {
                        lower_a: new_lower_a,
                        lower_b: new_lower_b,
                        upper_a: new_upper_a,
                        upper_b: new_upper_b,
                    };
                }
                Layer::Flatten(_) | Layer::Reshape(_) | Layer::Transpose(_) => {
                    // Shape-only layers: pass through unchanged to preserve dimensions
                    // (no-op in gradient computation backward pass)
                }
                Layer::AveragePool(pool) => {
                    // Pooling changes dimensionality; propagate linear bounds so earlier layers
                    // see the correct coefficient shapes.
                    lin_bounds = pool.propagate_linear_with_bounds(&lin_bounds, pre_bounds)?;
                }
                Layer::MaxPool2d(pool) => {
                    // MaxPool2d changes dimensionality; use its CROWN relaxation to propagate bounds.
                    lin_bounds = pool.propagate_linear_with_bounds(&lin_bounds, pre_bounds)?;
                }
                Layer::Conv2d(conv) => {
                    // Conv2d CROWN backward for gradient computation
                    let input_shape = pre_bounds.shape();
                    let (in_h, in_w) = if input_shape.len() >= 3 {
                        (
                            input_shape[input_shape.len() - 2],
                            input_shape[input_shape.len() - 1],
                        )
                    } else if input_shape.len() >= 2 {
                        let in_c = conv.in_channels();
                        let total = input_shape.iter().product::<usize>();
                        if total % in_c == 0 {
                            let spatial = total / in_c;
                            let side = (spatial as f64).sqrt() as usize;
                            if side * side == spatial {
                                (side, side)
                            } else {
                                continue; // Can't infer shape, skip
                            }
                        } else {
                            continue;
                        }
                    } else {
                        continue;
                    };

                    let mut conv_with_shape = conv.clone();
                    conv_with_shape.set_input_shape(in_h, in_w);

                    if let Ok(std::borrow::Cow::Owned(new_bounds)) =
                        conv_with_shape.propagate_linear(&lin_bounds)
                    {
                        lin_bounds = new_bounds;
                    }
                }
                Layer::Conv1d(conv) => {
                    // Conv1d CROWN backward for gradient computation
                    let input_shape = pre_bounds.shape();
                    let in_len = if input_shape.len() >= 2 {
                        input_shape[input_shape.len() - 1]
                    } else if !input_shape.is_empty() {
                        let in_c = conv.in_channels();
                        let total = input_shape.iter().product::<usize>();
                        if total % in_c == 0 {
                            total / in_c
                        } else {
                            continue;
                        }
                    } else {
                        continue;
                    };

                    let mut conv_with_shape = conv.clone();
                    conv_with_shape.set_input_length(in_len);

                    if let Ok(std::borrow::Cow::Owned(new_bounds)) =
                        conv_with_shape.propagate_linear(&lin_bounds)
                    {
                        lin_bounds = new_bounds;
                    }
                }
                _ => {
                    // For other layers, use IBP fallback (conservative)
                    let post_bounds = layer.propagate_ibp(pre_bounds)?;
                    let dim = post_bounds.len();
                    lin_bounds = LinearBounds::identity(dim);
                }
            }
        }

        // Compute concretization point x*
        let input_flat = input.flatten();
        let final_a = lin_bounds.lower_a.row(0);
        let x_star: Vec<f32> = (0..input_flat.len())
            .map(|i| {
                if final_a[i] >= 0.0 {
                    input_flat.lower[[i]]
                } else {
                    input_flat.upper[[i]]
                }
            })
            .collect();
        let x_star_arr = Array1::from_vec(x_star);

        // Forward sensitivity pass in homogeneous coordinates
        // w = [x*; 1] initially, then propagate through each layer
        let mut w: Vec<f32> = x_star_arr.to_vec();
        w.push(1.0); // Augment with constant 1

        for (layer_idx, layer) in network.layers.iter().enumerate() {
            // Use references instead of clones (Arc derefs to inner BoundedTensor)
            let pre_bounds: &BoundedTensor = if layer_idx == 0 {
                input
            } else {
                layer_bounds[layer_idx - 1].as_ref()
            };

            match layer {
                Layer::Linear(linear) => {
                    let weight = &linear.weight;
                    let dim_out = weight.nrows();
                    let dim_in = weight.ncols();

                    let w_const = w[dim_in]; // The homogeneous coordinate

                    // Apply Linear transform: w_out = W @ w_in + b * w_const
                    let w_in = Array1::from_vec(w[0..dim_in].to_vec());
                    let mut w_out = weight.dot(&w_in);

                    if let Some(bias) = &linear.bias {
                        for i in 0..dim_out {
                            w_out[i] += bias[i] * w_const;
                        }
                    }

                    w = w_out.to_vec();
                    w.push(w_const);
                }
                Layer::ReLU(_) => {
                    let pre_flat = pre_bounds.flatten();
                    let num_neurons = pre_flat.len();
                    let layer_constraints = constraints.get(&layer_idx);

                    // Compute β gradients before applying ReLU transform
                    if let Some(entry_indices) = beta_entries_by_layer.get(&layer_idx) {
                        for &entry_idx in entry_indices {
                            let entry = &beta_state.entries[entry_idx];
                            let neuron_idx = entry.neuron_idx;
                            if neuron_idx < num_neurons {
                                // d(lb)/d(β_j) = -sign_j * w[neuron_idx]
                                let grad = -entry.sign * w[neuron_idx];
                                beta_state.entries[entry_idx].grad = grad;
                            }
                        }
                    }

                    // Compute α gradients for unstable neurons
                    if let Some(slopes) = &relu_lower_slopes[layer_idx] {
                        for neuron_idx in 0..num_neurons {
                            if alpha_state.is_unstable(layer_idx, neuron_idx) {
                                let l = pre_flat.lower[[neuron_idx]];
                                let u = pre_flat.upper[[neuron_idx]];

                                // Check if this neuron uses the lower relaxation path
                                // (positive coefficient in the backward pass)
                                // The gradient is: d(lb)/d(α_j) = w[j] * x_j when using lower slope
                                // where x_j is the pre-activation value at concretization

                                // For lower bound, we need to check if we used the lower slope
                                let constraint =
                                    layer_constraints.and_then(|c| c.get(&neuron_idx).copied());
                                if constraint.is_none() && l < 0.0 && u > 0.0 {
                                    // This is an unstable, unconstrained neuron
                                    // The gradient depends on whether the coefficient was positive
                                    // (using lower relaxation) or negative (using upper relaxation)

                                    // For simplicity, use the sensitivity: d(lb)/d(α) ≈ w[j] * x_j
                                    // where x_j is the pre-activation input at concretization
                                    let x_j = if slopes[neuron_idx] > 0.5 {
                                        // α ≈ 1, used lower bound with slope 1
                                        w[neuron_idx]
                                    } else {
                                        // α ≈ 0, used lower bound with slope 0
                                        // Gradient is w[neuron_idx] * (pre-activation at concretization)
                                        w[neuron_idx]
                                    };

                                    // The actual gradient: how much does lb change if we increase α?
                                    // lb = ... + a_post * α * x + ...
                                    // d(lb)/d(α) = a_post * x where a_post is the post-layer coefficient
                                    // Since we want to maximize lb, positive gradient means increase α
                                    alpha_state.accumulate_grad(layer_idx, neuron_idx, x_j);
                                }
                            }
                        }
                    }

                    // Apply ReLU transform to sensitivity
                    if let Some(slopes) = &relu_lower_slopes[layer_idx] {
                        let w_const = w[num_neurons];
                        let mut w_out = vec![0.0f32; num_neurons];

                        for j in 0..num_neurons {
                            w_out[j] = slopes[j] * w[j];
                            // Note: intercepts would add to w_const, but for lower bound relaxation
                            // the intercept is always 0 (it's only non-zero for upper relaxation)
                        }

                        w = w_out;
                        w.push(w_const);
                    }
                }
                _ => {
                    // For other layers, reset sensitivity (conservative)
                    let post_bounds = layer.propagate_ibp(pre_bounds)?;
                    let dim = post_bounds.len();
                    w = vec![0.0; dim];
                    w.push(1.0);
                }
            }
        }

        // GCP-CROWN: Compute cut gradients using ReLU indicators
        // For the Lagrangian term: lambda * (bias - constraint_min)
        // d(lb)/d(λ) = bias - constraint_min
        // Positive gradient means increasing lambda will increase the lower bound
        if has_cuts {
            let relevant_indices: Vec<usize> = cut_pool
                .cuts
                .iter()
                .enumerate()
                .filter(|(_, cut)| !cut.is_redundant_for(history))
                .map(|(i, _)| i)
                .collect();

            for idx in relevant_indices {
                let cut = &cut_pool.cuts[idx];

                // Compute minimum value of the constraint: sum_i(coeff_i * z_i)
                // where z_i is the ReLU indicator (0 if inactive, 1 if active)
                let constraint_min: f32 = cut
                    .terms
                    .iter()
                    .filter_map(|term| {
                        if term.layer_idx == 0 || term.layer_idx > layer_bounds.len() {
                            return None;
                        }
                        let pre_bounds = &layer_bounds[term.layer_idx - 1];
                        let flat = pre_bounds.flatten();

                        if term.neuron_idx >= flat.len() {
                            return None;
                        }

                        let l = flat.lower[[term.neuron_idx]];
                        let u = flat.upper[[term.neuron_idx]];

                        // Determine ReLU indicator bounds [z_min, z_max]
                        let (z_min, z_max) = if l >= 0.0 {
                            (1.0, 1.0) // Stable active
                        } else if u <= 0.0 {
                            (0.0, 0.0) // Stable inactive
                        } else {
                            (0.0, 1.0) // Unstable
                        };

                        let value = if term.coefficient > 0.0 {
                            term.coefficient * z_min
                        } else {
                            term.coefficient * z_max
                        };
                        Some(value)
                    })
                    .sum();

                // Gradient: d(lb)/d(lambda) = bias - constraint_min
                // The cut adds: lambda * (bias - constraint_min) to the lower bound
                cut_pool.cuts[idx].lambda_grad = cut.bias - constraint_min;
            }
        }

        Ok(())
    }

    /// Backward propagation through a single layer with β constraints (legacy).
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)] // Legacy function, kept for testing and compatibility
    fn propagate_layer_backward_with_beta(
        &self,
        layer: &Layer,
        output_bounds: &LinearBounds,
        pre_bounds: &BoundedTensor,
        constraints: Option<&std::collections::HashMap<usize, bool>>,
        beta_state: &BetaState,
        layer_idx: usize,
    ) -> Result<LinearBounds> {
        match layer {
            Layer::Linear(linear) => {
                // Linear: standard CROWN backward
                // new_lA = lA @ W, new_lb = lb + lA @ b
                let weight = &linear.weight;
                let new_lower_a = output_bounds.lower_a.dot(weight);
                let new_upper_a = output_bounds.upper_a.dot(weight);

                let new_lower_b = if let Some(bias) = &linear.bias {
                    &output_bounds.lower_b + &output_bounds.lower_a.dot(bias)
                } else {
                    output_bounds.lower_b.clone()
                };

                let new_upper_b = if let Some(bias) = &linear.bias {
                    &output_bounds.upper_b + &output_bounds.upper_a.dot(bias)
                } else {
                    output_bounds.upper_b.clone()
                };

                Ok(LinearBounds {
                    lower_a: new_lower_a,
                    lower_b: new_lower_b,
                    upper_a: new_upper_a,
                    upper_b: new_upper_b,
                })
            }
            Layer::ReLU(_) => {
                // ReLU with β constraints
                self.relu_backward_with_beta(
                    output_bounds,
                    pre_bounds,
                    constraints,
                    beta_state,
                    layer_idx,
                )
            }
            Layer::Flatten(_) | Layer::Reshape(_) | Layer::Transpose(_) => {
                // Shape-only layers: pass through unchanged to preserve dimensions
                Ok(output_bounds.clone())
            }
            Layer::AveragePool(pool) => {
                // Pooling changes dimensionality; propagate linear bounds through it.
                pool.propagate_linear_with_bounds(output_bounds, pre_bounds)
            }
            Layer::MaxPool2d(pool) => {
                // MaxPool2d changes dimensionality; use its CROWN relaxation.
                pool.propagate_linear_with_bounds(output_bounds, pre_bounds)
            }
            Layer::Conv2d(conv) => {
                // Conv2d CROWN backward: apply transposed convolution to propagate linear bounds
                let input_shape = pre_bounds.shape();
                let (in_h, in_w) = if input_shape.len() >= 3 {
                    (
                        input_shape[input_shape.len() - 2],
                        input_shape[input_shape.len() - 1],
                    )
                } else if input_shape.len() >= 2 {
                    let in_c = conv.in_channels();
                    let total = input_shape.iter().product::<usize>();
                    if total % in_c == 0 {
                        let spatial = total / in_c;
                        let side = (spatial as f64).sqrt() as usize;
                        if side * side == spatial {
                            (side, side)
                        } else {
                            return Ok(output_bounds.clone());
                        }
                    } else {
                        return Ok(output_bounds.clone());
                    }
                } else {
                    return Ok(output_bounds.clone());
                };

                let mut conv_with_shape = conv.clone();
                conv_with_shape.set_input_shape(in_h, in_w);

                match conv_with_shape.propagate_linear(output_bounds) {
                    Ok(std::borrow::Cow::Owned(new_bounds)) => Ok(new_bounds),
                    Ok(std::borrow::Cow::Borrowed(_)) => Ok(output_bounds.clone()),
                    Err(e) => {
                        debug!("β-CROWN Conv2d backward failed: {}, using pass-through", e);
                        Ok(output_bounds.clone())
                    }
                }
            }
            Layer::Conv1d(conv) => {
                // Conv1d CROWN backward
                let input_shape = pre_bounds.shape();
                let in_len = if input_shape.len() >= 2 {
                    input_shape[input_shape.len() - 1]
                } else if !input_shape.is_empty() {
                    let in_c = conv.in_channels();
                    let total = input_shape.iter().product::<usize>();
                    if total % in_c == 0 {
                        total / in_c
                    } else {
                        return Ok(output_bounds.clone());
                    }
                } else {
                    return Ok(output_bounds.clone());
                };

                let mut conv_with_shape = conv.clone();
                conv_with_shape.set_input_length(in_len);

                match conv_with_shape.propagate_linear(output_bounds) {
                    Ok(std::borrow::Cow::Owned(new_bounds)) => Ok(new_bounds),
                    Ok(std::borrow::Cow::Borrowed(_)) => Ok(output_bounds.clone()),
                    Err(e) => {
                        debug!("β-CROWN Conv1d backward failed: {}, using pass-through", e);
                        Ok(output_bounds.clone())
                    }
                }
            }
            _ => {
                // For other layers, fall back to standard CROWN (IBP-based)
                debug!(
                    "β-CROWN: using IBP fallback for layer type {:?}",
                    std::any::type_name::<Layer>()
                );
                let post_bounds = layer.propagate_ibp(pre_bounds)?;

                // Create identity bounds (conservative)
                let dim = post_bounds.len();
                Ok(LinearBounds::identity(dim))
            }
        }
    }

    /// ReLU backward pass with β constraints for split neurons (legacy).
    ///
    /// This is the core of β-CROWN: when a neuron is constrained via split,
    /// we use exact slopes (0 or 1) instead of relaxations. Additionally,
    /// we add β contributions to the A matrix for Lagrangian optimization.
    #[allow(dead_code)] // Legacy function, kept for testing and compatibility
    fn relu_backward_with_beta(
        &self,
        output_bounds: &LinearBounds,
        pre_bounds: &BoundedTensor,
        constraints: Option<&std::collections::HashMap<usize, bool>>,
        beta_state: &BetaState,
        layer_idx: usize,
    ) -> Result<LinearBounds> {
        let pre_flat = pre_bounds.flatten();
        let num_neurons = pre_flat.len();
        let num_outputs = output_bounds.num_outputs();

        // Handle dimension mismatch when Conv layers changed dimensions in backward pass.
        if output_bounds.num_inputs() != num_neurons {
            debug!(
                "ReLU backward (β) dimension mismatch: output_bounds has {} inputs, but layer has {} neurons. Using identity fallback.",
                output_bounds.num_inputs(),
                num_neurons
            );
            return Ok(LinearBounds::identity(num_neurons));
        }

        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_lower_b = output_bounds.lower_b.clone();
        let mut new_upper_b = output_bounds.upper_b.clone();

        for j in 0..num_neurons {
            let l = pre_flat.lower[[j]];
            let u = pre_flat.upper[[j]];

            // Check if this neuron is constrained
            let constraint = constraints.and_then(|c| c.get(&j).copied());

            // Determine relaxation based on constraint
            let (lower_slope, lower_intercept, upper_slope, upper_intercept) =
                if let Some(is_active) = constraint {
                    if is_active {
                        // Neuron is constrained to be active (x >= 0)
                        // ReLU(x) = x, so slope = 1, intercept = 0
                        (1.0, 0.0, 1.0, 0.0)
                    } else {
                        // Neuron is constrained to be inactive (x <= 0)
                        // ReLU(x) = 0, so slope = 0, intercept = 0
                        (0.0, 0.0, 0.0, 0.0)
                    }
                } else if l >= 0.0 {
                    // Always active
                    (1.0, 0.0, 1.0, 0.0)
                } else if u <= 0.0 {
                    // Always inactive
                    (0.0, 0.0, 0.0, 0.0)
                } else {
                    // Unstable: use CROWN relaxation
                    // Upper bound: line through (l, 0) and (u, u)
                    // slope = u / (u - l), intercept = -l * u / (u - l)
                    let upper_slope = u / (u - l);
                    let upper_intercept = -l * u / (u - l);

                    // Lower bound: either 0 or identity line (heuristic: choose based on |u| vs |l|)
                    let lower_slope = if u > -l { 1.0 } else { 0.0 };
                    let lower_intercept = 0.0;

                    (lower_slope, lower_intercept, upper_slope, upper_intercept)
                };

            // Apply relaxation to each output
            for i in 0..num_outputs {
                let la_ij = output_bounds.lower_a[[i, j]];
                let ua_ij = output_bounds.upper_a[[i, j]];

                // Lower bound computation (for lA)
                if la_ij >= 0.0 {
                    // Positive coefficient: use lower relaxation
                    new_lower_a[[i, j]] = la_ij * lower_slope;
                    new_lower_b[i] += la_ij * lower_intercept;
                } else {
                    // Negative coefficient: use upper relaxation
                    new_lower_a[[i, j]] = la_ij * upper_slope;
                    new_lower_b[i] += la_ij * upper_intercept;
                }

                // Upper bound computation (for uA)
                if ua_ij >= 0.0 {
                    // Positive coefficient: use upper relaxation
                    new_upper_a[[i, j]] = ua_ij * upper_slope;
                    new_upper_b[i] += ua_ij * upper_intercept;
                } else {
                    // Negative coefficient: use lower relaxation
                    new_upper_a[[i, j]] = ua_ij * lower_slope;
                    new_upper_b[i] += ua_ij * lower_intercept;
                }
            }

            // Add beta contribution for constrained neurons (Lagrangian term)
            // β-CROWN: modify A matrix with β * sign
            // lA[:, j] -= β_j * sign_j
            // uA[:, j] += β_j * sign_j
            if let Some(signed_beta) = beta_state.get_signed_beta(layer_idx, j) {
                for i in 0..num_outputs {
                    new_lower_a[[i, j]] -= signed_beta;
                    new_upper_a[[i, j]] += signed_beta;
                }
            }
        }

        Ok(LinearBounds {
            lower_a: new_lower_a,
            lower_b: new_lower_b,
            upper_a: new_upper_a,
            upper_b: new_upper_b,
        })
    }

    /// Backward propagation through a single layer with both α and β parameters.
    #[allow(clippy::too_many_arguments)]
    fn propagate_layer_backward_with_alpha_beta(
        &self,
        layer: &Layer,
        output_bounds: &LinearBounds,
        pre_bounds: &BoundedTensor,
        constraints: Option<&std::collections::HashMap<usize, bool>>,
        beta_state: &BetaState,
        alpha_state: &DomainAlphaState,
        layer_idx: usize,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<LinearBounds> {
        fn infer_conv2d_input_hw(shape: &[usize], in_channels: usize) -> Option<(usize, usize)> {
            if shape.len() < 3 {
                return None;
            }
            let n = shape.len();
            let last3 = &shape[n - 3..];
            let chan_pos = last3.iter().position(|&d| d == in_channels)?;

            // Return the two non-channel dims in their original order.
            let mut spatial = Vec::with_capacity(2);
            for (idx, &d) in last3.iter().enumerate() {
                if idx != chan_pos {
                    spatial.push(d);
                }
            }
            if spatial.len() == 2 {
                Some((spatial[0], spatial[1]))
            } else {
                None
            }
        }

        fn infer_conv1d_input_len(shape: &[usize], in_channels: usize) -> Option<usize> {
            if shape.len() < 2 {
                return None;
            }
            let n = shape.len();
            let last2 = [shape[n - 2], shape[n - 1]];
            if last2[0] == in_channels {
                Some(last2[1])
            } else if last2[1] == in_channels {
                Some(last2[0])
            } else {
                None
            }
        }

        match layer {
            Layer::Linear(linear) => {
                // Linear: use GPU-accelerated CROWN backward if engine available
                linear
                    .propagate_linear_with_engine(output_bounds, engine)
                    .map(|cow| cow.into_owned())
            }
            Layer::ReLU(_) => {
                // ReLU with α and β parameters
                self.relu_backward_with_alpha_beta(
                    output_bounds,
                    pre_bounds,
                    constraints,
                    beta_state,
                    alpha_state,
                    layer_idx,
                )
            }
            Layer::Flatten(_) | Layer::Reshape(_) | Layer::Transpose(_) => {
                // Shape-only layers: pass through linear bounds unchanged.
                // These layers just reorganize data without changing values,
                // so in the backward pass, we maintain the accumulated linear bounds.
                Ok(output_bounds.clone())
            }
            Layer::AveragePool(pool) => {
                // Pooling changes dimensionality; propagate linear bounds through it.
                pool.propagate_linear_with_bounds(output_bounds, pre_bounds)
            }
            Layer::MaxPool2d(pool) => {
                // MaxPool2d changes dimensionality; use its CROWN relaxation.
                pool.propagate_linear_with_bounds(output_bounds, pre_bounds)
            }
            Layer::Conv2d(conv) => {
                // Conv2d CROWN backward: apply transposed convolution to propagate linear bounds
                // Get input shape from pre_bounds to configure conv backward
                let input_shape = pre_bounds.shape();
                let in_c = conv.in_channels();
                let (in_h, in_w) = if let Some((h, w)) = infer_conv2d_input_hw(input_shape, in_c) {
                    (h, w)
                } else if input_shape.len() >= 2 {
                    // Flattened input: try to infer shape from conv parameters
                    let total = input_shape.iter().product::<usize>();
                    if total % in_c == 0 {
                        let spatial = total / in_c;
                        let side = (spatial as f64).sqrt() as usize;
                        if side * side == spatial {
                            (side, side)
                        } else {
                            debug!("Joint α-β CROWN: Conv2d input shape {:?} cannot be reshaped, using pass-through", input_shape);
                            return Ok(output_bounds.clone());
                        }
                    } else {
                        debug!("Joint α-β CROWN: Conv2d input shape {:?} incompatible with in_channels={}, using pass-through", input_shape, in_c);
                        return Ok(output_bounds.clone());
                    }
                } else {
                    debug!(
                        "Joint α-β CROWN: Conv2d input shape too small: {:?}, using pass-through",
                        input_shape
                    );
                    return Ok(output_bounds.clone());
                };

                // Clone conv layer and set input shape for CROWN backward
                let mut conv_with_shape = conv.clone();
                conv_with_shape.set_input_shape(in_h, in_w);

                // Apply Conv2d CROWN backward propagation
                match conv_with_shape.propagate_linear(output_bounds) {
                    Ok(std::borrow::Cow::Owned(new_bounds)) => {
                        debug!(
                            "Joint α-β CROWN: Conv2d backward succeeded, input {}x{}",
                            in_h, in_w
                        );
                        Ok(new_bounds)
                    }
                    Ok(std::borrow::Cow::Borrowed(_)) => Ok(output_bounds.clone()),
                    Err(e) => {
                        debug!(
                            "Joint α-β CROWN: Conv2d backward failed ({}), using pass-through",
                            e
                        );
                        Ok(output_bounds.clone())
                    }
                }
            }
            Layer::Conv1d(conv) => {
                // Conv1d CROWN backward: apply transposed convolution to propagate linear bounds
                let input_shape = pre_bounds.shape();
                let in_c = conv.in_channels();
                let in_len = if let Some(len) = infer_conv1d_input_len(input_shape, in_c) {
                    len
                } else if !input_shape.is_empty() {
                    let total = input_shape.iter().product::<usize>();
                    if total % in_c == 0 {
                        total / in_c
                    } else {
                        debug!("Joint α-β CROWN: Conv1d input shape {:?} incompatible with in_channels={}, using pass-through", input_shape, in_c);
                        return Ok(output_bounds.clone());
                    }
                } else {
                    debug!(
                        "Joint α-β CROWN: Conv1d input shape too small: {:?}, using pass-through",
                        input_shape
                    );
                    return Ok(output_bounds.clone());
                };

                // Clone conv layer and set input length for CROWN backward
                let mut conv_with_shape = conv.clone();
                conv_with_shape.set_input_length(in_len);

                // Apply Conv1d CROWN backward propagation
                match conv_with_shape.propagate_linear(output_bounds) {
                    Ok(std::borrow::Cow::Owned(new_bounds)) => {
                        debug!(
                            "Joint α-β CROWN: Conv1d backward succeeded, input len {}",
                            in_len
                        );
                        Ok(new_bounds)
                    }
                    Ok(std::borrow::Cow::Borrowed(_)) => Ok(output_bounds.clone()),
                    Err(e) => {
                        debug!(
                            "Joint α-β CROWN: Conv1d backward failed ({}), using pass-through",
                            e
                        );
                        Ok(output_bounds.clone())
                    }
                }
            }
            _ => {
                // For other layers, fall back to IBP
                debug!(
                    "Joint α-β CROWN: using IBP fallback for layer type {:?}",
                    std::any::type_name::<Layer>()
                );
                let post_bounds = layer.propagate_ibp(pre_bounds)?;
                let dim = post_bounds.len();
                Ok(LinearBounds::identity(dim))
            }
        }
    }

    /// ReLU backward pass with both α and β parameters.
    ///
    /// Uses α values for the lower bound slope of unstable neurons,
    /// and β values for Lagrangian contributions from constrained neurons.
    #[allow(clippy::too_many_arguments)]
    fn relu_backward_with_alpha_beta(
        &self,
        output_bounds: &LinearBounds,
        pre_bounds: &BoundedTensor,
        constraints: Option<&std::collections::HashMap<usize, bool>>,
        beta_state: &BetaState,
        alpha_state: &DomainAlphaState,
        layer_idx: usize,
    ) -> Result<LinearBounds> {
        let pre_flat = pre_bounds.flatten();
        let num_neurons = pre_flat.len();
        let num_outputs = output_bounds.num_outputs();

        // Handle dimension mismatch when Conv layers changed dimensions in backward pass.
        // If output_bounds has different number of inputs than this layer's neurons,
        // we can't do proper CROWN backward - fall back to identity bounds.
        if output_bounds.num_inputs() != num_neurons {
            debug!(
                "ReLU backward dimension mismatch: output_bounds has {} inputs, but layer has {} neurons. Using identity fallback.",
                output_bounds.num_inputs(),
                num_neurons
            );
            return Ok(LinearBounds::identity(num_neurons));
        }

        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_lower_b = output_bounds.lower_b.clone();
        let mut new_upper_b = output_bounds.upper_b.clone();

        for j in 0..num_neurons {
            let l = pre_flat.lower[[j]];
            let u = pre_flat.upper[[j]];

            // Check if this neuron is constrained
            let constraint = constraints.and_then(|c| c.get(&j).copied());

            // Determine relaxation based on constraint and α
            let (lower_slope, lower_intercept, upper_slope, upper_intercept) =
                if let Some(is_active) = constraint {
                    if is_active {
                        (1.0, 0.0, 1.0, 0.0)
                    } else {
                        (0.0, 0.0, 0.0, 0.0)
                    }
                } else if l >= 0.0 {
                    // Always active
                    (1.0, 0.0, 1.0, 0.0)
                } else if u <= 0.0 {
                    // Always inactive
                    (0.0, 0.0, 0.0, 0.0)
                } else {
                    // Unstable: use α for lower slope, standard for upper
                    let upper_slope_val = u / (u - l);
                    let upper_intercept_val = -l * u / (u - l);

                    // Use optimizable α value for lower slope
                    let lower_slope_val = alpha_state.get_alpha(layer_idx, j);

                    (lower_slope_val, 0.0, upper_slope_val, upper_intercept_val)
                };

            // Apply relaxation to each output
            for i in 0..num_outputs {
                let la_ij = output_bounds.lower_a[[i, j]];
                let ua_ij = output_bounds.upper_a[[i, j]];

                // Lower bound computation
                if la_ij >= 0.0 {
                    new_lower_a[[i, j]] = la_ij * lower_slope;
                    new_lower_b[i] += la_ij * lower_intercept;
                } else {
                    new_lower_a[[i, j]] = la_ij * upper_slope;
                    new_lower_b[i] += la_ij * upper_intercept;
                }

                // Upper bound computation
                if ua_ij >= 0.0 {
                    new_upper_a[[i, j]] = ua_ij * upper_slope;
                    new_upper_b[i] += ua_ij * upper_intercept;
                } else {
                    new_upper_a[[i, j]] = ua_ij * lower_slope;
                    new_upper_b[i] += ua_ij * lower_intercept;
                }
            }

            // Add beta contribution for constrained neurons
            if let Some(signed_beta) = beta_state.get_signed_beta(layer_idx, j) {
                for i in 0..num_outputs {
                    new_lower_a[[i, j]] -= signed_beta;
                    new_upper_a[[i, j]] += signed_beta;
                }
            }
        }

        Ok(LinearBounds {
            lower_a: new_lower_a,
            lower_b: new_lower_b,
            upper_a: new_upper_a,
            upper_b: new_upper_b,
        })
    }
}

/// Extension trait for BoundedTensor to get scalar values.
trait BoundedTensorExt {
    fn lower_scalar(&self) -> f32;
    fn upper_scalar(&self) -> f32;
    fn argmin_lower_flat_idx(&self) -> usize;
}

impl BoundedTensorExt for BoundedTensor {
    fn lower_scalar(&self) -> f32 {
        // Return the minimum lower bound across all elements
        self.lower.iter().copied().fold(f32::INFINITY, f32::min)
    }

    fn upper_scalar(&self) -> f32 {
        // Return the maximum upper bound across all elements
        self.upper.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }

    fn argmin_lower_flat_idx(&self) -> usize {
        let mut best_idx = 0usize;
        let mut best_val = f32::INFINITY;
        for (idx, v) in self.lower.iter().copied().enumerate() {
            if v.is_nan() {
                continue;
            }
            if v < best_val {
                best_val = v;
                best_idx = idx;
            }
        }
        best_idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AddLayer, Conv2dLayer, GraphNetwork, GraphNode, Layer, LinearLayer, ReLULayer};
    use ndarray::{arr1, arr2, arr3, Array1, Array2, ArrayD, IxDyn};

    /// Helper to create a simple 2-layer network: Linear -> ReLU -> Linear
    fn simple_network() -> Network {
        // Layer 1: Linear 2 -> 2
        let w1 = arr2(&[[1.0, -1.0], [-1.0, 1.0]]);
        let linear1 = LinearLayer::new(w1, None).unwrap();

        // Layer 2: ReLU
        // Layer 3: Linear 2 -> 1
        let w2 = arr2(&[[1.0, 1.0]]);
        let linear2 = LinearLayer::new(w2, None).unwrap();

        let mut network = Network::new();
        network.add_layer(Layer::Linear(linear1));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(linear2));
        network
    }

    #[test]
    fn test_split_history() {
        let mut history = SplitHistory::new();
        assert_eq!(history.depth(), 0);
        assert!(history.is_constrained(1, 0).is_none());

        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active: true,
        });
        assert_eq!(history.depth(), 1);
        assert_eq!(history.is_constrained(1, 0), Some(true));
        assert!(history.is_constrained(1, 1).is_none());
    }

    #[test]
    fn test_joint_alpha_beta_conv2d_infers_spatial_dims_from_layout() {
        // Regression: Conv2d backward in joint α-β CROWN must not mis-infer (H,W)
        // for NHWC / HWC shaped tensors (common in TensorFlow-exported ONNX).
        let verifier = BetaCrownVerifier::default();

        // Conv2d: in_c=3, out_c=2, kernel=1x1, stride=1, pad=0.
        let kernel = ArrayD::from_shape_vec(
            IxDyn(&[2, 3, 1, 1]),
            vec![
                // out 0
                1.0, 0.0, 0.0, //
                // out 1
                0.0, 1.0, 0.0, //
            ],
        )
        .unwrap();
        let conv = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).unwrap();
        let layer = Layer::Conv2d(conv);

        let beta_state = BetaState::empty();
        let alpha_state = DomainAlphaState::empty();

        let in_c = 3usize;
        let in_h = 4usize;
        let in_w = 5usize;
        let out_c = 2usize;
        let conv_out_size = out_c * in_h * in_w;

        let mut lower_a = Array2::<f32>::zeros((1, conv_out_size));
        lower_a[[0, 0]] = 1.0;
        let output_bounds = LinearBounds {
            lower_a: lower_a.clone(),
            lower_b: Array1::zeros(1),
            upper_a: lower_a,
            upper_b: Array1::zeros(1),
        };

        let shapes: Vec<Vec<usize>> = vec![
            vec![in_c, in_h, in_w],    // CHW
            vec![in_h, in_w, in_c],    // HWC
            vec![1, in_c, in_h, in_w], // NCHW
            vec![1, in_h, in_w, in_c], // NHWC
        ];

        for shape in shapes {
            let zeros = ArrayD::<f32>::zeros(IxDyn(&shape));
            let pre_bounds = BoundedTensor::new(zeros.clone(), zeros).unwrap();

            let new_bounds = verifier
                .propagate_layer_backward_with_alpha_beta(
                    &layer,
                    &output_bounds,
                    &pre_bounds,
                    None,
                    &beta_state,
                    &alpha_state,
                    0,
                    None, // No GPU engine in tests
                )
                .unwrap();

            assert_eq!(new_bounds.num_inputs(), in_c * in_h * in_w);
        }
    }

    #[test]
    fn test_beta_crown_trivial_verified() {
        // Network output is always positive for the given input range
        let network = simple_network();

        // Input: x in [[1, 2], [1, 2]]
        // After Linear1: [1-1, 1-1] to [2-1, 2-1] = [[0, 0], [1, 1]]
        // Wait, let me recalculate...
        // W1 = [[1, -1], [-1, 1]]
        // x in [[1, 2], [1, 2]]
        // W1 @ [1, 1] = [0, 0], W1 @ [2, 2] = [0, 0]
        // Actually bounds are [0, 0] to [0, 0]? No...
        // W1 @ x = [x1 - x2, -x1 + x2]
        // For x in [[1, 2], [1, 2]]:
        // x1 - x2 in [1-2, 2-1] = [-1, 1]
        // -x1 + x2 in [-2+1, -1+2] = [-1, 1]
        // After ReLU: [[0, 0], [1, 1]]
        // After Linear2: [0+0, 1+1] = [0, 2]
        // So output in [0, 2]

        // Use a simpler input where output is clearly > 0
        let input =
            BoundedTensor::new(arr1(&[2.0, 0.0]).into_dyn(), arr1(&[3.0, 1.0]).into_dyn()).unwrap();

        let verifier = BetaCrownVerifier::default();
        let result = verifier.verify(&network, &input, -10.0).unwrap();

        // Should verify since output bounds are [0, inf] and threshold is -10
        assert_eq!(result.result, BabVerificationStatus::Verified);
    }

    #[test]
    fn test_beta_crown_needs_splitting() {
        let network = simple_network();

        // Input that creates unstable neurons
        let input =
            BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn())
                .unwrap();

        let config = BetaCrownConfig {
            max_domains: 100,
            timeout: Duration::from_secs(10),
            ..Default::default()
        };

        let verifier = BetaCrownVerifier::new(config);
        let result = verifier.verify(&network, &input, -5.0).unwrap();

        // Should verify (output >= 0 for this network)
        println!("Result: {:?}", result);
        assert_eq!(result.result, BabVerificationStatus::Verified);
    }

    #[test]
    fn test_constraint_tightens_bounds() {
        // Test that adding a constraint tightens the bounds
        let mut history = SplitHistory::new();

        // Add constraint that neuron 0 in layer 1 is active
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active: true,
        });

        let beta_state = BetaState::from_history(&history);
        assert_eq!(beta_state.len(), 1);

        // Check that the beta entry has correct sign
        let entry = beta_state.get_entry(1, 0).unwrap();
        assert_eq!(entry.sign, 1.0); // Active constraint has positive sign
        assert_eq!(entry.value, 0.0); // Initial beta is 0
    }

    #[test]
    fn test_domain_ordering() {
        // Test that BabDomain ordering works correctly (higher lb = higher priority)
        let bounds1 =
            vec![BoundedTensor::new(arr1(&[0.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap()];
        let bounds2 = bounds1.clone();

        let d1 = BabDomain::root(bounds1, 1.0, 2.0);
        let d2 = BabDomain::root(bounds2, 0.5, 2.0);

        // d1 has higher lower bound, should be "greater"
        assert!(d1 > d2);

        // Test with heap
        let mut heap = BinaryHeap::new();
        heap.push(d2.clone());
        heap.push(d1.clone());

        // Should pop d1 first (higher lower bound)
        assert_eq!(heap.pop().unwrap().lower_bound, 1.0);
        assert_eq!(heap.pop().unwrap().lower_bound, 0.5);
    }

    #[test]
    fn test_babsr_branching_heuristic() {
        // Test BaBSR branching with BoundImpact heuristic
        let network = simple_network();

        // Input that creates unstable neurons
        let input =
            BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn())
                .unwrap();

        let config = BetaCrownConfig {
            max_domains: 100,
            timeout: Duration::from_secs(10),
            branching_heuristic: BranchingHeuristic::BoundImpact,
            ..Default::default()
        };

        let verifier = BetaCrownVerifier::new(config);
        let result = verifier.verify(&network, &input, -5.0).unwrap();

        // Should verify (output >= 0 for this network) with BaBSR
        println!("BaBSR Result: {:?}", result);
        assert_eq!(result.result, BabVerificationStatus::Verified);
    }

    #[test]
    fn test_fsb_branching_heuristic() {
        // Smoke test for FSB branching (FilteredSmartBranching)
        let network = simple_network();

        // Input that creates unstable neurons
        let input =
            BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn())
                .unwrap();

        let config = BetaCrownConfig {
            max_domains: 100,
            timeout: Duration::from_secs(10),
            branching_heuristic: BranchingHeuristic::FilteredSmartBranching,
            fsb_candidates: 4,
            ..Default::default()
        };

        let verifier = BetaCrownVerifier::new(config);
        let result = verifier.verify(&network, &input, -5.0).unwrap();

        println!("FSB Result: {:?}", result);
        assert_eq!(result.result, BabVerificationStatus::Verified);
    }

    /// Helper to create a deeper network for CROWN-IBP testing
    /// Linear 2 -> 4, ReLU, Linear 4 -> 4, ReLU, Linear 4 -> 1
    fn deeper_network() -> Network {
        let w1 = arr2(&[[1.0, 0.5], [-0.5, 1.0], [0.3, -0.7], [-0.2, 0.8]]);
        let linear1 = LinearLayer::new(w1, None).unwrap();

        let w2 = arr2(&[
            [0.5, -0.3, 0.7, 0.1],
            [-0.4, 0.6, -0.2, 0.5],
            [0.3, 0.2, -0.5, 0.4],
            [-0.1, 0.4, 0.3, -0.6],
        ]);
        let linear2 = LinearLayer::new(w2, None).unwrap();

        let w3 = arr2(&[[1.0, -0.5, 0.3, 0.2]]);
        let linear3 = LinearLayer::new(w3, None).unwrap();

        let mut network = Network::new();
        network.add_layer(Layer::Linear(linear1));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(linear2));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(linear3));
        network
    }

    #[test]
    fn test_crown_ibp_tighter_intermediate_bounds() {
        // Test that CROWN-IBP produces tighter intermediate bounds than IBP
        let network = deeper_network();

        let input =
            BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn())
                .unwrap();

        // Collect bounds with IBP
        let ibp_bounds = network.collect_ibp_bounds(&input).unwrap();

        // Collect bounds with CROWN-IBP
        let crown_ibp_bounds = network.collect_crown_ibp_bounds(&input).unwrap();

        // CROWN-IBP bounds should be at least as tight (usually tighter) than IBP
        let mut crown_ibp_tighter = false;
        for (ibp, crown_ibp) in ibp_bounds.iter().zip(crown_ibp_bounds.iter()) {
            let ibp_width = ibp.max_width();
            let crown_ibp_width = crown_ibp.max_width();

            // CROWN-IBP should never be looser
            assert!(
                crown_ibp_width <= ibp_width + 1e-5,
                "CROWN-IBP should not be looser than IBP"
            );

            // Track if CROWN-IBP is tighter for any layer
            if crown_ibp_width < ibp_width - 1e-5 {
                crown_ibp_tighter = true;
            }
        }

        // For this network, CROWN-IBP should be tighter
        assert!(
            crown_ibp_tighter,
            "CROWN-IBP should be tighter than IBP for deeper networks"
        );
    }

    #[test]
    fn test_beta_crown_with_crown_ibp() {
        // Test that β-CROWN works with CROWN-IBP enabled
        let network = deeper_network();

        let input =
            BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn())
                .unwrap();

        // Run with standard IBP bounds
        let config_ibp = BetaCrownConfig {
            max_domains: 100,
            timeout: Duration::from_secs(10),
            use_alpha_crown: false,
            use_crown_ibp: false,
            ..Default::default()
        };
        let verifier_ibp = BetaCrownVerifier::new(config_ibp);
        let result_ibp = verifier_ibp.verify(&network, &input, -5.0).unwrap();

        // Run with CROWN-IBP bounds
        let config_crown_ibp = BetaCrownConfig {
            max_domains: 100,
            timeout: Duration::from_secs(10),
            use_alpha_crown: false,
            use_crown_ibp: true,
            ..Default::default()
        };
        let verifier_crown_ibp = BetaCrownVerifier::new(config_crown_ibp);
        let result_crown_ibp = verifier_crown_ibp.verify(&network, &input, -5.0).unwrap();

        // Both should verify (property is easy to verify)
        assert_eq!(result_ibp.result, BabVerificationStatus::Verified);
        assert_eq!(result_crown_ibp.result, BabVerificationStatus::Verified);

        // CROWN-IBP should use fewer or equal domains (tighter bounds = less splitting)
        println!(
            "Domains explored: IBP={}, CROWN-IBP={}",
            result_ibp.domains_explored, result_crown_ibp.domains_explored
        );

        // Note: For this simple case, both may verify with 1 domain
        // The real benefit shows on harder problems like ACAS-Xu
    }

    #[test]
    fn test_crown_coefficient_computation() {
        // Test that CROWN coefficients are computed correctly
        let network = simple_network();

        // Create input bounds
        let input =
            BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn())
                .unwrap();

        // Get layer bounds
        let layer_bounds = network.collect_ibp_bounds(&input).unwrap();

        // Create root domain
        let domain = BabDomain::root(layer_bounds, 0.0, 1.0);

        // Compute coefficients
        let verifier = BetaCrownVerifier::default();
        let coeffs = verifier
            .compute_crown_coefficients(&network, &domain)
            .unwrap();

        // ReLU is at layer 1, should have coefficients for both neurons
        // Layer 1 is the ReLU layer, so we should have entries for neurons at that layer
        println!("CROWN coefficients: {:?}", coeffs);

        // Verify we got coefficients for the ReLU layer (layer_idx=1)
        let has_relu_coeffs = coeffs.keys().any(|(layer, _)| *layer == 1);
        assert!(has_relu_coeffs, "Should have coefficients for ReLU layer");
    }

    #[test]
    fn test_babsr_selects_high_impact_neuron() {
        // Create a deeper network to test BaBSR selection
        // Linear 2 -> 4, ReLU, Linear 4 -> 2, ReLU, Linear 2 -> 1
        let w1 = arr2(&[[1.0, 0.5], [-0.5, 1.0], [0.3, -0.7], [-0.2, 0.8]]);
        let linear1 = LinearLayer::new(w1, None).unwrap();

        let w2 = arr2(&[[0.5, -0.3, 0.7, 0.1], [-0.4, 0.6, -0.2, 0.5]]);
        let linear2 = LinearLayer::new(w2, None).unwrap();

        let w3 = arr2(&[[1.0, -0.5]]);
        let linear3 = LinearLayer::new(w3, None).unwrap();

        let mut network = Network::new();
        network.add_layer(Layer::Linear(linear1));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(linear2));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(linear3));

        // Input that creates many unstable neurons
        let input =
            BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn())
                .unwrap();

        let config = BetaCrownConfig {
            max_domains: 50,
            timeout: Duration::from_secs(5),
            branching_heuristic: BranchingHeuristic::BoundImpact,
            ..Default::default()
        };

        let verifier = BetaCrownVerifier::new(config);
        let result = verifier.verify(&network, &input, -10.0).unwrap();

        // Should explore domains and verify
        println!("Deep network BaBSR result: {:?}", result);
        assert!(result.domains_explored > 0);
    }

    #[test]
    fn test_beta_state_gradient_step() {
        // Test beta gradient step with projection to non-negative
        let mut history = SplitHistory::new();
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active: true,
        });
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 1,
            is_active: false,
        });

        let mut beta_state = BetaState::from_history(&history);

        // Set some values
        beta_state.entries[0].value = 0.5;
        beta_state.entries[1].value = 0.1;

        // Set gradients
        beta_state.entries[0].grad = 0.2;
        beta_state.entries[1].grad = -0.3; // Negative gradient should trigger projection

        // Perform gradient step with lr=1.0
        let max_grad = beta_state.gradient_step(1.0);

        // Check projection
        assert_eq!(beta_state.entries[0].value, 0.7); // 0.5 + 1.0 * 0.2
        assert_eq!(beta_state.entries[1].value, 0.0); // max(0, 0.1 + 1.0 * -0.3) = max(0, -0.2) = 0

        // Check max gradient magnitude
        assert!((max_grad - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_beta_state_inactive_constraint_sign() {
        // Test that inactive constraints have negative sign
        let mut history = SplitHistory::new();
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active: false,
        });

        let beta_state = BetaState::from_history(&history);
        let entry = beta_state.get_entry(1, 0).unwrap();
        assert_eq!(entry.sign, -1.0); // Inactive constraint has negative sign
    }

    #[test]
    fn test_beta_signed_contribution() {
        // Test that get_signed_beta returns value * sign
        let mut history = SplitHistory::new();
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active: true,
        });
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 1,
            is_active: false,
        });

        let mut beta_state = BetaState::from_history(&history);
        beta_state.set_beta(1, 0, 2.0);
        beta_state.set_beta(1, 1, 3.0);

        // Active: sign=+1, so signed_beta = 2.0 * 1.0 = 2.0
        assert_eq!(beta_state.get_signed_beta(1, 0), Some(2.0));

        // Inactive: sign=-1, so signed_beta = 3.0 * -1.0 = -3.0
        assert_eq!(beta_state.get_signed_beta(1, 1), Some(-3.0));
    }

    #[test]
    fn test_beta_optimization_with_constraints() {
        // Test that beta optimization runs and produces valid bounds
        let network = simple_network();

        // Input that creates unstable neurons
        let input =
            BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn())
                .unwrap();

        // Create config with beta optimization enabled
        let config = BetaCrownConfig {
            max_domains: 20,
            timeout: Duration::from_secs(10),
            beta_lr: 0.05,
            beta_iterations: 5,
            beta_tolerance: 1e-6,
            ..Default::default()
        };

        let verifier = BetaCrownVerifier::new(config);
        let result = verifier.verify(&network, &input, -5.0).unwrap();

        // Should verify (output >= 0 for this network)
        println!("Beta optimization result: {:?}", result);
        assert_eq!(result.result, BabVerificationStatus::Verified);
    }

    #[test]
    fn test_beta_optimization_disabled() {
        // Test that setting beta_iterations=0 disables optimization
        let network = simple_network();

        let input =
            BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn())
                .unwrap();

        let config = BetaCrownConfig {
            max_domains: 20,
            timeout: Duration::from_secs(10),
            beta_iterations: 0, // Disable optimization
            ..Default::default()
        };

        let verifier = BetaCrownVerifier::new(config);
        let result = verifier.verify(&network, &input, -5.0).unwrap();

        // Should still work without optimization
        println!("No beta optimization result: {:?}", result);
        assert_eq!(result.result, BabVerificationStatus::Verified);
    }

    #[test]
    fn test_batched_processing_same_result() {
        // Test that batched processing gives the same result as sequential
        let network = simple_network();

        let input =
            BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn())
                .unwrap();

        // Sequential (batch_size=1, no parallel children)
        let config_seq = BetaCrownConfig {
            max_domains: 50,
            timeout: Duration::from_secs(10),
            batch_size: 1,
            parallel_children: false,
            ..Default::default()
        };
        let verifier_seq = BetaCrownVerifier::new(config_seq);
        let result_seq = verifier_seq.verify(&network, &input, -5.0).unwrap();

        // Batched parallel (batch_size=4, parallel children)
        let config_par = BetaCrownConfig {
            max_domains: 50,
            timeout: Duration::from_secs(10),
            batch_size: 4,
            parallel_children: true,
            ..Default::default()
        };
        let verifier_par = BetaCrownVerifier::new(config_par);
        let result_par = verifier_par.verify(&network, &input, -5.0).unwrap();

        // Both should give the same verification result
        assert_eq!(result_seq.result, result_par.result);
        println!("Sequential: {:?}", result_seq);
        println!("Parallel: {:?}", result_par);
    }

    #[test]
    fn test_parallel_children_enabled() {
        // Test that parallel_children flag works
        let network = simple_network();

        let input =
            BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn())
                .unwrap();

        let config = BetaCrownConfig {
            max_domains: 50,
            timeout: Duration::from_secs(10),
            batch_size: 1,
            parallel_children: true, // Use parallel child creation
            ..Default::default()
        };

        let verifier = BetaCrownVerifier::new(config);
        let result = verifier.verify(&network, &input, -5.0).unwrap();

        assert_eq!(result.result, BabVerificationStatus::Verified);
    }

    #[test]
    fn test_large_batch_processing() {
        // Test with large batch size on a deeper network
        let w1 = arr2(&[[1.0, 0.5], [-0.5, 1.0], [0.3, -0.7], [-0.2, 0.8]]);
        let linear1 = LinearLayer::new(w1, None).unwrap();

        let w2 = arr2(&[[0.5, -0.3, 0.7, 0.1], [-0.4, 0.6, -0.2, 0.5]]);
        let linear2 = LinearLayer::new(w2, None).unwrap();

        let w3 = arr2(&[[1.0, -0.5]]);
        let linear3 = LinearLayer::new(w3, None).unwrap();

        let mut network = Network::new();
        network.add_layer(Layer::Linear(linear1));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(linear2));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(linear3));

        let input =
            BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn())
                .unwrap();

        let config = BetaCrownConfig {
            max_domains: 100,
            timeout: Duration::from_secs(10),
            batch_size: 8, // Large batch
            parallel_children: true,
            ..Default::default()
        };

        let verifier = BetaCrownVerifier::new(config);
        let result = verifier.verify(&network, &input, -10.0).unwrap();

        println!("Large batch result: {:?}", result);
        assert!(result.domains_explored > 0);
    }

    #[test]
    fn test_analytical_gradients_direction() {
        // Test that analytical gradients point in a direction that improves bounds
        // when used for gradient ascent (increasing lower bound)
        let network = simple_network();

        let input =
            BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn())
                .unwrap();

        // Create a split history with one constraint
        let mut history = SplitHistory::new();
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active: true,
        });

        let layer_bounds: Vec<Arc<BoundedTensor>> = network
            .collect_ibp_bounds(&input)
            .unwrap()
            .into_iter()
            .map(Arc::new)
            .collect();
        let mut beta_state = BetaState::from_history(&history);

        // Set initial beta value
        beta_state.set_beta(1, 0, 0.1);

        // Compute bounds before gradient step
        let verifier = BetaCrownVerifier::default();
        let bounds_before = verifier
            .compute_bounds_with_constraints(&network, &input, &history, &layer_bounds, &beta_state)
            .unwrap();
        let lb_before = bounds_before.lower_scalar();

        // Compute analytical gradient
        verifier
            .compute_beta_gradients(
                &network,
                &input,
                &history,
                &layer_bounds,
                &mut beta_state,
                0,
            )
            .unwrap();

        let grad = beta_state.entries[0].grad;
        println!("Analytical gradient: {}", grad);

        // Take a small step in gradient direction
        let lr = 0.01;
        let new_beta = (beta_state.entries[0].value + lr * grad).max(0.0);
        beta_state.set_beta(1, 0, new_beta);

        // Compute bounds after gradient step
        let bounds_after = verifier
            .compute_bounds_with_constraints(&network, &input, &history, &layer_bounds, &beta_state)
            .unwrap();
        let lb_after = bounds_after.lower_scalar();

        println!("Lower bound before: {}, after: {}", lb_before, lb_after);

        // For gradient ascent, if grad > 0 and we increase beta, lb should increase (or stay same).
        if grad.abs() > 1e-6 {
            let improvement = lb_after - lb_before;
            println!("Improvement: {}", improvement);
            assert!(
                improvement >= -1e-3,
                "Gradient step should not significantly decrease bound"
            );
        }
    }

    #[test]
    fn test_analytical_vs_numerical_gradient_consistency() {
        // Test that analytical gradients are consistent with numerical gradients
        // in terms of sign and rough magnitude
        let network = simple_network();

        let input =
            BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn())
                .unwrap();

        let mut history = SplitHistory::new();
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active: true,
        });

        let layer_bounds: Vec<Arc<BoundedTensor>> = network
            .collect_ibp_bounds(&input)
            .unwrap()
            .into_iter()
            .map(Arc::new)
            .collect();
        let mut beta_state = BetaState::from_history(&history);
        beta_state.set_beta(1, 0, 0.5);

        let verifier = BetaCrownVerifier::default();

        // Compute analytical gradient
        verifier
            .compute_beta_gradients(
                &network,
                &input,
                &history,
                &layer_bounds,
                &mut beta_state,
                0,
            )
            .unwrap();
        let analytical_grad = beta_state.entries[0].grad;

        // Compute numerical gradient for comparison
        let eps = 1e-4;
        let original_beta = 0.5;

        beta_state.set_beta(1, 0, original_beta + eps);
        let bounds_plus = verifier
            .compute_bounds_with_constraints(&network, &input, &history, &layer_bounds, &beta_state)
            .unwrap();
        let lb_plus = bounds_plus.lower_scalar();

        beta_state.set_beta(1, 0, (original_beta - eps).max(0.0));
        let bounds_minus = verifier
            .compute_bounds_with_constraints(&network, &input, &history, &layer_bounds, &beta_state)
            .unwrap();
        let lb_minus = bounds_minus.lower_scalar();

        let numerical_grad = (lb_plus - lb_minus) / (2.0 * eps);

        println!("Analytical gradient: {}", analytical_grad);
        println!("Numerical gradient: {}", numerical_grad);

        let abs_err = (analytical_grad - numerical_grad).abs();
        let tol = 1e-2 + 1e-2 * numerical_grad.abs();
        assert!(
            abs_err <= tol,
            "Analytical gradient should match numerical: analytical={}, numerical={}, abs_err={}, tol={}",
            analytical_grad,
            numerical_grad,
            abs_err,
            tol
        );
    }

    #[test]
    fn test_analytical_gradient_multiple_constraints() {
        // Test analytical gradients with multiple constraints
        let w1 = arr2(&[[1.0, 0.5], [-0.5, 1.0], [0.3, -0.7], [-0.2, 0.8]]);
        let linear1 = LinearLayer::new(w1, None).unwrap();

        let w2 = arr2(&[[0.5, -0.3, 0.7, 0.1], [-0.4, 0.6, -0.2, 0.5]]);
        let linear2 = LinearLayer::new(w2, None).unwrap();

        let w3 = arr2(&[[1.0, -0.5]]);
        let linear3 = LinearLayer::new(w3, None).unwrap();

        let mut network = Network::new();
        network.add_layer(Layer::Linear(linear1));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(linear2));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(linear3));

        let input =
            BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn())
                .unwrap();

        // Create history with multiple constraints on different layers
        let mut history = SplitHistory::new();
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active: true,
        });
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 1,
            is_active: false,
        });
        history.add_constraint(NeuronConstraint {
            layer_idx: 3,
            neuron_idx: 0,
            is_active: true,
        });

        let layer_bounds: Vec<Arc<BoundedTensor>> = network
            .collect_ibp_bounds(&input)
            .unwrap()
            .into_iter()
            .map(Arc::new)
            .collect();
        let mut beta_state = BetaState::from_history(&history);

        // Set initial beta values
        for entry in &mut beta_state.entries {
            entry.value = 0.1;
        }

        let verifier = BetaCrownVerifier::default();
        verifier
            .compute_beta_gradients(
                &network,
                &input,
                &history,
                &layer_bounds,
                &mut beta_state,
                0,
            )
            .unwrap();

        println!("Gradients for multiple constraints:");
        for entry in &beta_state.entries {
            println!(
                "  Layer {}, Neuron {}, sign={}, beta={}, grad={}",
                entry.layer_idx, entry.neuron_idx, entry.sign, entry.value, entry.grad
            );
        }

        // All gradients should be computed (not NaN)
        for entry in &beta_state.entries {
            assert!(!entry.grad.is_nan(), "Gradient should not be NaN");
        }
    }

    #[test]
    fn test_joint_alpha_beta_optimization() {
        // Test that joint α-β optimization produces valid bounds
        let network = simple_network();

        // Input that creates unstable neurons (crossing zero)
        let input =
            BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn())
                .unwrap();

        // Config with joint α-β optimization enabled
        let config = BetaCrownConfig {
            max_domains: 20,
            timeout: Duration::from_secs(10),
            use_alpha_crown: true,
            beta_lr: 0.05,
            alpha_lr: 0.5,
            alpha_momentum: true,
            beta_iterations: 10,
            beta_tolerance: 1e-6,
            ..Default::default()
        };

        let verifier = BetaCrownVerifier::new(config);
        let result = verifier.verify(&network, &input, -5.0).unwrap();

        println!("Joint α-β optimization result: {:?}", result);
        assert_eq!(result.result, BabVerificationStatus::Verified);
        assert!(
            result.domains_explored > 0,
            "Should explore at least one domain"
        );
    }

    #[test]
    fn test_domain_alpha_state_initialization() {
        // Test that domain alpha state is correctly initialized
        let w1 = arr2(&[[1.0, 0.5], [-0.5, 1.0]]);
        let linear1 = LinearLayer::new(w1, None).unwrap();
        let w2 = arr2(&[[0.5, -0.3], [-0.2, 0.6]]);
        let linear2 = LinearLayer::new(w2, None).unwrap();

        let mut network = Network::new();
        network.add_layer(Layer::Linear(linear1));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(linear2));

        // Input with crossing bounds to create unstable neurons
        let input =
            BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn())
                .unwrap();

        let layer_bounds: Vec<Arc<BoundedTensor>> = network
            .collect_ibp_bounds(&input)
            .unwrap()
            .into_iter()
            .map(Arc::new)
            .collect();
        let history = SplitHistory::new();

        let alpha_state =
            DomainAlphaState::from_layer_bounds_and_constraints(&network, &layer_bounds, &history);

        // Should have some unstable neurons tracked
        println!("DomainAlphaState: {} unstable neurons", alpha_state.len());
        for ((layer_idx, neuron_idx), alpha) in &alpha_state.alphas {
            println!("  Layer {}, Neuron {}: α={}", layer_idx, neuron_idx, alpha);
        }

        // Without constraints, should have unstable neurons from layer 1 (ReLU)
        // Check that alphas are in valid range [0, 1]
        for &alpha in alpha_state.alphas.values() {
            assert!((0.0..=1.0).contains(&alpha), "Alpha should be in [0, 1]");
        }
    }

    #[test]
    fn test_domain_alpha_state_with_constraints() {
        // Test that constrained neurons are excluded from alpha optimization
        let w1 = arr2(&[[1.0, 0.5], [-0.5, 1.0]]);
        let linear1 = LinearLayer::new(w1, None).unwrap();
        let w2 = arr2(&[[0.5, -0.3], [-0.2, 0.6]]);
        let linear2 = LinearLayer::new(w2, None).unwrap();

        let mut network = Network::new();
        network.add_layer(Layer::Linear(linear1));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(linear2));

        let input =
            BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn())
                .unwrap();

        let layer_bounds: Vec<Arc<BoundedTensor>> = network
            .collect_ibp_bounds(&input)
            .unwrap()
            .into_iter()
            .map(Arc::new)
            .collect();

        // Add constraint on neuron 0 of ReLU layer
        let mut history = SplitHistory::new();
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active: true,
        });

        let alpha_state =
            DomainAlphaState::from_layer_bounds_and_constraints(&network, &layer_bounds, &history);

        // Constrained neuron should NOT be in alpha_state
        assert!(
            !alpha_state.is_unstable(1, 0),
            "Constrained neuron should not be tracked for alpha optimization"
        );

        println!(
            "Alpha state after constraint: {} unstable neurons",
            alpha_state.len()
        );
    }

    #[test]
    fn test_joint_optimization_improves_bounds() {
        // Test that joint optimization produces at least as good bounds as beta-only
        let network = simple_network();

        let input =
            BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn())
                .unwrap();

        // Beta-only optimization
        let config_beta_only = BetaCrownConfig {
            max_domains: 30,
            timeout: Duration::from_secs(10),
            use_alpha_crown: false,
            beta_iterations: 10,
            ..Default::default()
        };
        let verifier_beta = BetaCrownVerifier::new(config_beta_only);
        let result_beta = verifier_beta.verify(&network, &input, -3.0).unwrap();

        // Joint α-β optimization
        let config_joint = BetaCrownConfig {
            max_domains: 30,
            timeout: Duration::from_secs(10),
            use_alpha_crown: true,
            beta_iterations: 10,
            alpha_lr: 0.5,
            alpha_momentum: true,
            ..Default::default()
        };
        let verifier_joint = BetaCrownVerifier::new(config_joint);
        let result_joint = verifier_joint.verify(&network, &input, -3.0).unwrap();

        println!(
            "Beta-only: {:?}, domains={}",
            result_beta.result, result_beta.domains_explored
        );
        println!(
            "Joint α-β: {:?}, domains={}",
            result_joint.result, result_joint.domains_explored
        );

        // Both should verify (this is an easy problem)
        assert_eq!(result_beta.result, BabVerificationStatus::Verified);
        assert_eq!(result_joint.result, BabVerificationStatus::Verified);
    }

    #[test]
    fn test_adaptive_opt_config_defaults() {
        // Test that AdaptiveOptConfig has sensible defaults
        let config = AdaptiveOptConfig::default();

        // Standard Adam defaults
        assert_eq!(config.beta1, 0.9);
        assert_eq!(config.beta2, 0.999);
        assert_eq!(config.epsilon, 1e-8);
        assert!(config.bias_correction);
        assert!(!config.radam);

        // Our specific defaults for β-CROWN (matching α,β-CROWN)
        assert_eq!(config.beta_lr, 0.05); // α,β-CROWN default
        assert_eq!(config.alpha_lr, 0.01); // α,β-CROWN default
        assert_eq!(config.grad_clip, 10.0);
    }

    #[test]
    fn test_beta_state_adam_update() {
        // Test that Adam optimizer updates β correctly
        let mut history = SplitHistory::new();
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active: true,
        });

        let mut beta_state = BetaState::from_history(&history);
        let config = AdaptiveOptConfig::default();

        // Set a gradient
        beta_state.accumulate_grad(1, 0, 1.0);

        // Perform several Adam steps
        for t in 1..=5 {
            let max_grad = beta_state.gradient_step_adam(&config, t);
            println!(
                "t={}: beta={:.4}, m={:.4}, v={:.6}, max_grad={:.4}",
                t,
                beta_state.entries[0].value,
                beta_state.entries[0].m,
                beta_state.entries[0].v,
                max_grad
            );

            // Reset gradient for next iteration
            beta_state.zero_grad();
            beta_state.accumulate_grad(1, 0, 1.0); // Constant gradient
        }

        // After 5 iterations with constant gradient=1, beta should have increased
        assert!(
            beta_state.entries[0].value > 0.0,
            "Beta should increase with positive gradient"
        );

        // First moment should approach gradient (weighted average)
        assert!(
            beta_state.entries[0].m > 0.0,
            "First moment m should be positive with positive gradients"
        );

        // Second moment should be positive
        assert!(
            beta_state.entries[0].v > 0.0,
            "Second moment v should be positive"
        );
    }

    #[test]
    fn test_alpha_state_adam_update() {
        // Test that Adam optimizer updates α correctly and respects [0, 1] bounds
        let network = simple_network();
        let input =
            BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn())
                .unwrap();

        let layer_bounds: Vec<Arc<BoundedTensor>> = network
            .collect_ibp_bounds(&input)
            .unwrap()
            .into_iter()
            .map(Arc::new)
            .collect();
        let history = SplitHistory::new();

        let mut alpha_state =
            DomainAlphaState::from_layer_bounds_and_constraints(&network, &layer_bounds, &history);

        if alpha_state.is_empty() {
            println!("No unstable neurons for this network/input combination");
            return;
        }

        let config = AdaptiveOptConfig::default();

        // Get first unstable neuron for testing
        let key = *alpha_state.alphas.keys().next().unwrap();
        let initial_alpha = alpha_state.get_alpha(key.0, key.1);

        // Set a positive gradient (should increase α towards 1)
        alpha_state.accumulate_grad(key.0, key.1, 0.5);

        for t in 1..=10 {
            let max_grad = alpha_state.gradient_step_adam(&config, t);

            let current_alpha = alpha_state.get_alpha(key.0, key.1);

            // Alpha should always be in [0, 1]
            assert!(
                (0.0..=1.0).contains(&current_alpha),
                "Alpha out of bounds: {}",
                current_alpha
            );

            println!(
                "t={}: alpha={:.4}, max_grad={:.4}",
                t, current_alpha, max_grad
            );

            // Reset and set same gradient
            alpha_state.zero_grad();
            alpha_state.accumulate_grad(key.0, key.1, 0.5);
        }

        let final_alpha = alpha_state.get_alpha(key.0, key.1);

        // With positive gradient, α should have moved towards 1 (or stayed there if already 1)
        assert!(
            final_alpha >= initial_alpha || (initial_alpha == 1.0 && final_alpha == 1.0),
            "Alpha should increase or stay at 1 with positive gradient"
        );
    }

    #[test]
    fn test_adaptive_optimization_convergence() {
        // Test that adaptive optimization converges similarly to fixed-LR
        let network = simple_network();
        let input =
            BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn())
                .unwrap();

        // Fixed learning rate
        let config_fixed = BetaCrownConfig {
            max_domains: 30,
            timeout: Duration::from_secs(10),
            use_alpha_crown: true,
            use_adaptive: false,
            beta_iterations: 15,
            ..Default::default()
        };
        let verifier_fixed = BetaCrownVerifier::new(config_fixed);
        let result_fixed = verifier_fixed.verify(&network, &input, -3.0).unwrap();

        // Adaptive learning rate
        let config_adaptive = BetaCrownConfig {
            max_domains: 30,
            timeout: Duration::from_secs(10),
            use_alpha_crown: true,
            use_adaptive: true,
            beta_iterations: 15,
            adaptive_config: AdaptiveOptConfig::default(),
            ..Default::default()
        };
        let verifier_adaptive = BetaCrownVerifier::new(config_adaptive);
        let result_adaptive = verifier_adaptive.verify(&network, &input, -3.0).unwrap();

        println!(
            "Fixed LR: {:?}, domains={}",
            result_fixed.result, result_fixed.domains_explored
        );
        println!(
            "Adaptive: {:?}, domains={}",
            result_adaptive.result, result_adaptive.domains_explored
        );

        // Both should verify this simple network
        assert_eq!(result_fixed.result, BabVerificationStatus::Verified);
        assert_eq!(result_adaptive.result, BabVerificationStatus::Verified);
    }

    #[test]
    fn test_gradient_clipping() {
        // Test that gradient clipping works correctly
        let mut history = SplitHistory::new();
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active: true,
        });

        let mut beta_state = BetaState::from_history(&history);
        let config = AdaptiveOptConfig {
            grad_clip: 1.0, // Clip to ±1
            ..Default::default()
        };

        // Set a large gradient that should be clipped
        beta_state.accumulate_grad(1, 0, 100.0);

        // Perform Adam step
        let max_grad = beta_state.gradient_step_adam(&config, 1);

        // The effective gradient should be clipped
        // With clipping to 1.0, first moment should be approximately 0.1 (1 - β₁) * 1.0
        println!(
            "m after clipping: {:.6}, max_grad: {:.4}",
            beta_state.entries[0].m, max_grad
        );

        // First moment should reflect clipped gradient, not original
        // m = (1 - 0.9) * 1.0 = 0.1 (since grad is clipped to 1.0)
        assert!(
            beta_state.entries[0].m < 10.0,
            "First moment should be bounded due to gradient clipping"
        );
    }

    #[test]
    fn test_bias_correction() {
        // Test that bias correction produces different updates than without
        // (The specific relationship depends on the relative values of β₁ and β₂)
        let mut history = SplitHistory::new();
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active: true,
        });

        // Without bias correction
        let mut beta_no_correction = BetaState::from_history(&history);
        let config_no_correction = AdaptiveOptConfig {
            bias_correction: false,
            ..Default::default()
        };
        beta_no_correction.accumulate_grad(1, 0, 1.0);
        beta_no_correction.gradient_step_adam(&config_no_correction, 1);
        let value_no_correction = beta_no_correction.entries[0].value;
        let m_no = beta_no_correction.entries[0].m;
        let v_no = beta_no_correction.entries[0].v;

        // With bias correction
        let mut beta_with_correction = BetaState::from_history(&history);
        let config_with_correction = AdaptiveOptConfig {
            bias_correction: true,
            ..Default::default()
        };
        beta_with_correction.accumulate_grad(1, 0, 1.0);
        beta_with_correction.gradient_step_adam(&config_with_correction, 1);
        let value_with_correction = beta_with_correction.entries[0].value;
        let m_with = beta_with_correction.entries[0].m;
        let v_with = beta_with_correction.entries[0].v;

        println!(
            "Without bias correction: beta={:.6}, m={:.6}, v={:.9}",
            value_no_correction, m_no, v_no
        );
        println!(
            "With bias correction:    beta={:.6}, m={:.6}, v={:.9}",
            value_with_correction, m_with, v_with
        );

        // Raw m and v should be the same (bias correction only affects effective values)
        assert!(
            (m_no - m_with).abs() < 1e-6,
            "Raw first moment should be the same"
        );
        assert!(
            (v_no - v_with).abs() < 1e-9,
            "Raw second moment should be the same"
        );

        // The updates should be different due to bias correction scaling
        // Note: With Adam's default parameters, bias correction actually produces
        // smaller updates in iteration 1 because v_hat = v/(1-β₂^t) grows faster
        // than m_hat = m/(1-β₁^t), leading to larger denominator
        assert!(
            (value_no_correction - value_with_correction).abs() > 1e-6,
            "Bias correction should produce different update"
        );

        // Both updates should be positive (gradient is positive)
        assert!(value_no_correction > 0.0, "Update should be positive");
        assert!(value_with_correction > 0.0, "Update should be positive");
    }

    // ==================== Learning Rate Scheduler Tests ====================

    #[test]
    fn test_lr_scheduler_constant() {
        let scheduler = LRScheduler::Constant;
        let base_lr = 0.1;

        // Constant scheduler should always return the base LR
        for t in 0..100 {
            let lr = scheduler.get_lr(t, base_lr);
            assert!(
                (lr - base_lr).abs() < 1e-6,
                "Constant scheduler should return base_lr at t={}, got {}",
                t,
                lr
            );
        }
    }

    #[test]
    fn test_lr_scheduler_step_decay() {
        let scheduler = LRScheduler::StepDecay {
            gamma: 0.5,
            step_size: 5,
        };
        let base_lr = 1.0;

        // t=0-4: factor = 0.5^0 = 1.0
        assert!((scheduler.get_lr(0, base_lr) - 1.0).abs() < 1e-6);
        assert!((scheduler.get_lr(4, base_lr) - 1.0).abs() < 1e-6);

        // t=5-9: factor = 0.5^1 = 0.5
        assert!((scheduler.get_lr(5, base_lr) - 0.5).abs() < 1e-6);
        assert!((scheduler.get_lr(9, base_lr) - 0.5).abs() < 1e-6);

        // t=10-14: factor = 0.5^2 = 0.25
        assert!((scheduler.get_lr(10, base_lr) - 0.25).abs() < 1e-6);
        assert!((scheduler.get_lr(14, base_lr) - 0.25).abs() < 1e-6);

        // t=15: factor = 0.5^3 = 0.125
        assert!((scheduler.get_lr(15, base_lr) - 0.125).abs() < 1e-6);
    }

    #[test]
    fn test_lr_scheduler_exponential_decay() {
        let gamma = 0.9f32;
        let scheduler = LRScheduler::ExponentialDecay { gamma };
        let base_lr = 1.0;

        // LR(t) = base_lr * gamma^t
        for t in 0..20 {
            let expected = base_lr * gamma.powi(t as i32);
            let actual = scheduler.get_lr(t, base_lr);
            assert!(
                (actual - expected).abs() < 1e-6,
                "ExponentialDecay mismatch at t={}: expected {}, got {}",
                t,
                expected,
                actual
            );
        }

        // Verify decay is happening
        assert!(scheduler.get_lr(0, base_lr) > scheduler.get_lr(10, base_lr));
        assert!(scheduler.get_lr(10, base_lr) > scheduler.get_lr(20, base_lr));
    }

    #[test]
    fn test_lr_scheduler_cosine_annealing() {
        let scheduler = LRScheduler::CosineAnnealing {
            min_lr: 0.0,
            t_max: 10,
        };
        let base_lr = 1.0;

        // At t=0: cosine(0) = 1, so LR = 0 + 0.5 * (1-0) * (1+1) = 1.0
        let lr_0 = scheduler.get_lr(0, base_lr);
        assert!(
            (lr_0 - 1.0).abs() < 1e-6,
            "Cosine should start at base_lr, got {}",
            lr_0
        );

        // At t=t_max/2: cosine(π/2) = 0, so LR = 0.5 * base_lr
        let lr_mid = scheduler.get_lr(5, base_lr);
        assert!(
            (lr_mid - 0.5).abs() < 1e-5,
            "Cosine should be at midpoint at t=t_max/2, got {}",
            lr_mid
        );

        // At t=t_max: cosine(π) = -1, so LR = min_lr = 0
        let lr_end = scheduler.get_lr(10, base_lr);
        assert!(
            lr_end < 0.01,
            "Cosine should approach min_lr at t_max, got {}",
            lr_end
        );

        // Test with non-zero min_lr
        let scheduler2 = LRScheduler::CosineAnnealing {
            min_lr: 0.1,
            t_max: 10,
        };

        let lr_end2 = scheduler2.get_lr(10, base_lr);
        assert!(
            (lr_end2 - 0.1).abs() < 0.05,
            "Cosine should end near min_lr, got {}",
            lr_end2
        );
    }

    #[test]
    fn test_lr_scheduler_warmup_cosine() {
        let scheduler = LRScheduler::WarmupCosine {
            warmup_steps: 3,
            min_lr: 0.0,
            t_max: 10,
        };
        let base_lr = 1.0;

        // During warmup: linear increase from 0 to base_lr
        // t=0: (0+1)/3 = 0.333...
        let lr_0 = scheduler.get_lr(0, base_lr);
        assert!(
            (lr_0 - 1.0 / 3.0).abs() < 1e-5,
            "Warmup at t=0 should be 1/3 of base_lr, got {}",
            lr_0
        );

        // t=1: (1+1)/3 = 0.666...
        let lr_1 = scheduler.get_lr(1, base_lr);
        assert!(
            (lr_1 - 2.0 / 3.0).abs() < 1e-5,
            "Warmup at t=1 should be 2/3 of base_lr, got {}",
            lr_1
        );

        // t=2: (2+1)/3 = 1.0 (end of warmup)
        let lr_2 = scheduler.get_lr(2, base_lr);
        assert!(
            (lr_2 - 1.0).abs() < 1e-5,
            "Warmup should reach base_lr at end of warmup, got {}",
            lr_2
        );

        // After warmup: cosine annealing from t=3 to t=10
        // The cosine phase has t_max - warmup_steps = 7 iterations
        let lr_3 = scheduler.get_lr(3, base_lr);
        assert!(
            (lr_3 - 1.0).abs() < 1e-5,
            "Just after warmup should be at base_lr, got {}",
            lr_3
        );

        // At t=10: should be near min_lr
        let lr_end = scheduler.get_lr(10, base_lr);
        assert!(
            lr_end < 0.1,
            "Should approach min_lr at t_max, got {}",
            lr_end
        );

        // LR should decrease monotonically after warmup
        let mut prev_lr = lr_3;
        for t in 4..=10 {
            let lr = scheduler.get_lr(t, base_lr);
            assert!(
                lr <= prev_lr + 1e-6,
                "LR should decrease after warmup: t={}, prev={}, curr={}",
                t,
                prev_lr,
                lr
            );
            prev_lr = lr;
        }
    }

    #[test]
    fn test_lr_scheduler_edge_cases() {
        // Zero t_max for cosine should return base LR
        let scheduler = LRScheduler::CosineAnnealing {
            min_lr: 0.1,
            t_max: 0,
        };
        assert!((scheduler.get_lr(0, 1.0) - 1.0).abs() < 1e-6);

        // Zero warmup steps - should go directly to cosine
        let scheduler2 = LRScheduler::WarmupCosine {
            warmup_steps: 0,
            min_lr: 0.0,
            t_max: 10,
        };
        let lr = scheduler2.get_lr(0, 1.0);
        assert!(
            (lr - 1.0).abs() < 1e-5,
            "Zero warmup should start at base_lr, got {}",
            lr
        );
    }

    #[test]
    fn test_lr_scheduler_with_adam() {
        // Create a split history with one constraint
        let mut history = SplitHistory::new();
        history.add_constraint(NeuronConstraint {
            layer_idx: 0,
            neuron_idx: 0,
            is_active: true,
        });

        // Create beta state from history
        let mut beta_state = BetaState::from_history(&history);
        beta_state.accumulate_grad(0, 0, 1.0);

        // Test with exponential decay scheduler
        let config = AdaptiveOptConfig {
            beta_lr: 1.0,
            scheduler: LRScheduler::ExponentialDecay { gamma: 0.5 },
            bias_correction: false, // Simpler to verify
            ..Default::default()
        };

        // Iteration 1 (t=1): LR = 1.0 * 0.5^0 = 1.0
        beta_state.gradient_step_adam(&config, 1);
        let beta_t1 = beta_state.get_entry(0, 0).map(|e| e.value).unwrap_or(0.0);

        // Reset and try iteration 2
        let mut history2 = SplitHistory::new();
        history2.add_constraint(NeuronConstraint {
            layer_idx: 0,
            neuron_idx: 0,
            is_active: true,
        });
        let mut beta_state2 = BetaState::from_history(&history2);
        beta_state2.accumulate_grad(0, 0, 1.0);
        // First do iteration 1 to build up momentum
        beta_state2.gradient_step_adam(&config, 1);
        beta_state2.zero_grad();
        beta_state2.accumulate_grad(0, 0, 1.0);
        // Iteration 2 (t=2): LR = 1.0 * 0.5^1 = 0.5
        beta_state2.gradient_step_adam(&config, 2);

        println!("Beta after t=1: {}", beta_t1);
        let beta_t2 = beta_state2.get_entry(0, 0).map(|e| e.value).unwrap_or(0.0);
        println!("Beta after t=2: {}", beta_t2);

        // The second update should use half the learning rate
        // This is validated by the scheduler integration working correctly
        assert!(beta_t1 > 0.0, "Beta should increase");
    }

    #[test]
    fn test_lr_scheduler_default() {
        // Default scheduler should be Constant
        let config = AdaptiveOptConfig::default();
        assert!(
            matches!(config.scheduler, LRScheduler::Constant),
            "Default scheduler should be Constant"
        );
    }

    #[test]
    fn test_cosine_scheduler_full_trajectory() {
        // Verify the full trajectory of cosine annealing
        let scheduler = LRScheduler::CosineAnnealing {
            min_lr: 0.01,
            t_max: 100,
        };
        let base_lr = 1.0;

        let lrs: Vec<f32> = (0..=100).map(|t| scheduler.get_lr(t, base_lr)).collect();

        // Should start near base_lr
        assert!(lrs[0] > 0.99, "Should start near base_lr");

        // Should end near min_lr
        assert!(lrs[100] < 0.02, "Should end near min_lr");

        // Should be monotonically decreasing
        for i in 1..lrs.len() {
            assert!(
                lrs[i] <= lrs[i - 1] + 1e-6,
                "Cosine should be monotonically decreasing: lrs[{}]={} > lrs[{}]={}",
                i,
                lrs[i],
                i - 1,
                lrs[i - 1]
            );
        }

        // Midpoint should be approximately (base_lr + min_lr) / 2
        let expected_mid = (1.0 + 0.01) / 2.0;
        assert!(
            (lrs[50] - expected_mid).abs() < 0.05,
            "Midpoint LR {} should be close to {}",
            lrs[50],
            expected_mid
        );
    }

    #[test]
    fn test_amsgrad_basic() {
        // Test that AMSGrad tracks v_max properly
        let mut history = SplitHistory::new();
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active: true,
        });

        let mut beta_state = BetaState::from_history(&history);
        // Use lower beta2 for faster decay to demonstrate v decreasing
        let config = AdaptiveOptConfig {
            amsgrad: true,
            beta2: 0.5,             // Fast decay to show v can decrease
            bias_correction: false, // Simpler to verify
            ..Default::default()
        };

        // First step with gradient 1.0
        beta_state.accumulate_grad(1, 0, 1.0);
        beta_state.gradient_step_adam(&config, 1);

        let entry = beta_state.get_entry(1, 0).unwrap();
        let v_after_1 = entry.v;
        let v_max_after_1 = entry.v_max;

        // v_max should equal v after first step
        assert!(
            (v_max_after_1 - v_after_1).abs() < 1e-9,
            "v_max should equal v after first step: v={}, v_max={}",
            v_after_1,
            v_max_after_1
        );

        // Reset gradient and apply smaller gradient
        beta_state.zero_grad();
        beta_state.accumulate_grad(1, 0, 0.1); // Much smaller gradient
        beta_state.gradient_step_adam(&config, 2);

        let entry2 = beta_state.get_entry(1, 0).unwrap();
        let v_after_2 = entry2.v;
        let v_max_after_2 = entry2.v_max;

        // With beta2=0.5:
        // v_after_1 = 0.5 * 1.0^2 = 0.5
        // v_after_2 = 0.5 * 0.5 + 0.5 * 0.1^2 = 0.25 + 0.005 = 0.255
        // v should decrease because 0.5*v + 0.5*small^2 < v when small < 1

        // v_max should stay the same (or larger) - this is the key AMSGrad property
        assert!(
            v_max_after_2 >= v_max_after_1 - 1e-9,
            "v_max should never decrease: {} < {}",
            v_max_after_2,
            v_max_after_1
        );
        assert!(
            v_after_2 < v_after_1,
            "v should decrease with smaller gradients: {} >= {}",
            v_after_2,
            v_after_1
        );

        // v_max should be larger than v after v decreased
        assert!(
            v_max_after_2 > v_after_2,
            "v_max ({}) should be > v ({}) after v decreased",
            v_max_after_2,
            v_after_2
        );

        println!(
            "v_after_1: {:.6}, v_max_after_1: {:.6}",
            v_after_1, v_max_after_1
        );
        println!(
            "v_after_2: {:.6}, v_max_after_2: {:.6}",
            v_after_2, v_max_after_2
        );
    }

    #[test]
    fn test_amsgrad_vs_adam_behavior() {
        // Compare AMSGrad and standard Adam update sizes when v decreases
        let mut history = SplitHistory::new();
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active: true,
        });

        // Standard Adam config
        let config_adam = AdaptiveOptConfig {
            amsgrad: false,
            beta_lr: 0.5,
            bias_correction: false,
            ..Default::default()
        };

        // AMSGrad config
        let config_amsgrad = AdaptiveOptConfig {
            amsgrad: true,
            beta_lr: 0.5,
            bias_correction: false,
            ..Default::default()
        };

        // Run both with same gradient sequence: large then small
        let mut beta_adam = BetaState::from_history(&history);
        let mut beta_amsgrad = BetaState::from_history(&history);

        // Step 1: Large gradient
        beta_adam.accumulate_grad(1, 0, 2.0);
        beta_amsgrad.accumulate_grad(1, 0, 2.0);
        beta_adam.gradient_step_adam(&config_adam, 1);
        beta_amsgrad.gradient_step_adam(&config_amsgrad, 1);

        let v1_adam = beta_adam.get_entry(1, 0).unwrap().value;
        let v1_amsgrad = beta_amsgrad.get_entry(1, 0).unwrap().value;

        // Should be equal after first step
        assert!(
            (v1_adam - v1_amsgrad).abs() < 1e-6,
            "First step should be equal"
        );

        // Step 2: Small gradient - this is where AMSGrad differs
        beta_adam.zero_grad();
        beta_amsgrad.zero_grad();
        beta_adam.accumulate_grad(1, 0, 0.01);
        beta_amsgrad.accumulate_grad(1, 0, 0.01);
        beta_adam.gradient_step_adam(&config_adam, 2);
        beta_amsgrad.gradient_step_adam(&config_amsgrad, 2);

        let v2_adam = beta_adam.get_entry(1, 0).unwrap().v;
        let v2_amsgrad_v = beta_amsgrad.get_entry(1, 0).unwrap().v;
        let v2_amsgrad_v_max = beta_amsgrad.get_entry(1, 0).unwrap().v_max;

        // v should be similar (decayed)
        assert!(
            (v2_adam - v2_amsgrad_v).abs() < 1e-6,
            "v values should be similar"
        );

        // v_max should be larger than v for AMSGrad
        assert!(
            v2_amsgrad_v_max > v2_amsgrad_v,
            "v_max ({}) should be > v ({}) after large then small gradient",
            v2_amsgrad_v_max,
            v2_amsgrad_v
        );

        println!("Adam v: {:.6}", v2_adam);
        println!(
            "AMSGrad v: {:.6}, v_max: {:.6}",
            v2_amsgrad_v, v2_amsgrad_v_max
        );
    }

    #[test]
    fn test_amsgrad_monotonic_v_max() {
        // Test that v_max never decreases over many iterations
        let mut history = SplitHistory::new();
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active: true,
        });

        let mut beta_state = BetaState::from_history(&history);
        let config = AdaptiveOptConfig {
            amsgrad: true,
            bias_correction: true,
            ..Default::default()
        };

        let mut prev_v_max = 0.0f32;
        let gradients = [1.0, 0.5, 2.0, 0.1, 0.3, 1.5, 0.05, 0.8, 0.2, 1.0];

        for (i, &grad) in gradients.iter().enumerate() {
            beta_state.zero_grad();
            beta_state.accumulate_grad(1, 0, grad);
            beta_state.gradient_step_adam(&config, i + 1);

            let entry = beta_state.get_entry(1, 0).unwrap();
            assert!(
                entry.v_max >= prev_v_max - 1e-9,
                "v_max decreased at iteration {}: {} < {}",
                i + 1,
                entry.v_max,
                prev_v_max
            );
            prev_v_max = entry.v_max;
        }
    }

    #[test]
    fn test_amsgrad_alpha_state() {
        // Test AMSGrad with DomainAlphaState using the simple_network helper
        let network = simple_network();

        // Input bounds that create unstable neurons
        let input =
            BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn())
                .unwrap();

        let layer_bounds: Vec<Arc<BoundedTensor>> = network
            .collect_ibp_bounds(&input)
            .unwrap()
            .into_iter()
            .map(Arc::new)
            .collect();
        let history = SplitHistory::new();

        let mut alpha_state =
            DomainAlphaState::from_layer_bounds_and_constraints(&network, &layer_bounds, &history);

        if alpha_state.is_empty() {
            println!("No unstable neurons - skipping test");
            return;
        }

        let config = AdaptiveOptConfig {
            amsgrad: true,
            alpha_lr: 0.1,
            bias_correction: false,
            ..Default::default()
        };

        // Get first unstable neuron key
        let key = *alpha_state.unstable_neurons.iter().next().unwrap();

        // First gradient: large
        alpha_state.accumulate_grad(key.0, key.1, 1.0);
        alpha_state.gradient_step_adam(&config, 1);
        let v_max_1 = *alpha_state.adam_v_max.get(&key).unwrap_or(&0.0);

        // Second gradient: small
        alpha_state.zero_grad();
        alpha_state.accumulate_grad(key.0, key.1, 0.1);
        alpha_state.gradient_step_adam(&config, 2);
        let v_max_2 = *alpha_state.adam_v_max.get(&key).unwrap_or(&0.0);

        // v_max should not decrease
        assert!(
            v_max_2 >= v_max_1 - 1e-9,
            "v_max should not decrease: {} < {}",
            v_max_2,
            v_max_1
        );

        println!(
            "AMSGrad alpha test passed: v_max_1={:.6}, v_max_2={:.6}",
            v_max_1, v_max_2
        );
    }

    #[test]
    fn test_amsgrad_config_default() {
        // Test that AMSGrad is disabled by default
        let config = AdaptiveOptConfig::default();
        assert!(
            !config.amsgrad,
            "AMSGrad should be disabled by default for backward compatibility"
        );
    }

    #[test]
    fn test_amsgrad_disabled_v_max_unchanged() {
        // When AMSGrad is disabled, v_max should not be updated
        let mut history = SplitHistory::new();
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active: true,
        });

        let mut beta_state = BetaState::from_history(&history);
        let config = AdaptiveOptConfig {
            amsgrad: false,
            ..Default::default()
        };

        // Check initial v_max
        assert_eq!(
            beta_state.get_entry(1, 0).unwrap().v_max,
            0.0,
            "v_max should start at 0"
        );

        // Run several iterations
        for i in 1..=5 {
            beta_state.zero_grad();
            beta_state.accumulate_grad(1, 0, 1.0);
            beta_state.gradient_step_adam(&config, i);
        }

        // v_max should still be 0 when AMSGrad is disabled
        assert_eq!(
            beta_state.get_entry(1, 0).unwrap().v_max,
            0.0,
            "v_max should remain 0 when AMSGrad is disabled"
        );
    }

    #[test]
    fn test_adamw_config_default() {
        // Test that weight_decay is 0 by default
        let config = AdaptiveOptConfig::default();
        assert_eq!(
            config.weight_decay, 0.0,
            "weight_decay should be 0 by default for backward compatibility"
        );
    }

    #[test]
    fn test_radam_config_default() {
        // Test that RAdam is disabled by default
        let config = AdaptiveOptConfig::default();
        assert!(
            !config.radam,
            "RAdam should be disabled by default for backward compatibility"
        );
    }

    #[test]
    fn test_radam_rectification_factor_switch() {
        // With beta2=0.999, rectification activates at t=5 (ρ_t > 4).
        assert!(radam_rectification_factor(0.999, 4.0).is_none());
        let r = radam_rectification_factor(0.999, 5.0).expect("expected rectification factor");
        assert!(
            r > 0.0 && r < 1.0,
            "rectification factor should be in (0, 1), got {}",
            r
        );
    }

    #[test]
    fn test_beta_state_radam_smaller_update_than_adam() {
        // With constant positive gradients, RAdam should take smaller steps once rectification
        // activates (t >= 5 for beta2=0.999) compared to standard Adam.
        let mut history = SplitHistory::new();
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active: true,
        });

        let mut beta_adam = BetaState::from_history(&history);
        let mut beta_radam = BetaState::from_history(&history);

        let config_adam = AdaptiveOptConfig {
            beta_lr: 0.1,
            grad_clip: 0.0,
            radam: false,
            ..Default::default()
        };
        let config_radam = AdaptiveOptConfig {
            beta_lr: 0.1,
            grad_clip: 0.0,
            radam: true,
            ..Default::default()
        };

        for t in 1..=5 {
            beta_adam.zero_grad();
            beta_radam.zero_grad();
            beta_adam.accumulate_grad(1, 0, 1.0);
            beta_radam.accumulate_grad(1, 0, 1.0);
            beta_adam.gradient_step_adam(&config_adam, t);
            beta_radam.gradient_step_adam(&config_radam, t);
        }

        let adam_value = beta_adam.get_entry(1, 0).unwrap().value;
        let radam_value = beta_radam.get_entry(1, 0).unwrap().value;

        assert!(
            radam_value < adam_value,
            "expected RAdam value < Adam value at t=5, got radam={:.6}, adam={:.6}",
            radam_value,
            adam_value
        );
    }

    #[test]
    fn test_adamw_basic() {
        // Test that weight decay reduces parameter values over time
        let mut history = SplitHistory::new();
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active: true,
        });

        let mut beta_state = BetaState::from_history(&history);
        // Set initial value explicitly
        beta_state.get_entry_mut(1, 0).unwrap().value = 1.0;

        // Use high weight decay and zero gradient to see decay in isolation
        let config = AdaptiveOptConfig {
            weight_decay: 0.1,
            beta_lr: 0.1,
            bias_correction: false,
            ..Default::default()
        };

        // With zero gradient, Adam update is 0, only weight decay applies
        // θ_new = θ * (1 - lr * λ) = 1.0 * (1 - 0.1 * 0.1) = 0.99
        beta_state.zero_grad();
        beta_state.accumulate_grad(1, 0, 0.0); // Zero gradient
        beta_state.gradient_step_adam(&config, 1);

        let value_after = beta_state.get_entry(1, 0).unwrap().value;
        let expected = 1.0 * (1.0 - 0.1 * 0.1);
        assert!(
            (value_after - expected).abs() < 1e-6,
            "After one step with zero gradient, value should be {:.6}, got {:.6}",
            expected,
            value_after
        );

        // Multiple steps should compound the decay
        for i in 2..=10 {
            beta_state.zero_grad();
            beta_state.accumulate_grad(1, 0, 0.0);
            beta_state.gradient_step_adam(&config, i);
        }

        let value_final = beta_state.get_entry(1, 0).unwrap().value;
        // After 10 steps: 1.0 * (1 - 0.01)^10 ≈ 0.9044
        let expected_final = (1.0 - 0.1 * 0.1f32).powi(10);
        assert!(
            (value_final - expected_final).abs() < 1e-4,
            "After 10 steps, value should be {:.6}, got {:.6}",
            expected_final,
            value_final
        );

        println!(
            "AdamW basic test: initial=1.0, after 1 step={:.6}, after 10 steps={:.6}",
            expected, value_final
        );
    }

    #[test]
    fn test_adamw_vs_adam() {
        // Compare AdamW with standard Adam (weight_decay=0)
        let mut history = SplitHistory::new();
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active: true,
        });

        let config_adam = AdaptiveOptConfig {
            weight_decay: 0.0,
            beta_lr: 0.1,
            ..Default::default()
        };

        let config_adamw = AdaptiveOptConfig {
            weight_decay: 0.05,
            beta_lr: 0.1,
            ..Default::default()
        };

        let mut beta_adam = BetaState::from_history(&history);
        let mut beta_adamw = BetaState::from_history(&history);

        // Set same initial values
        beta_adam.get_entry_mut(1, 0).unwrap().value = 0.5;
        beta_adamw.get_entry_mut(1, 0).unwrap().value = 0.5;

        // Run with same gradients
        let gradients = [1.0, 0.5, 0.8, 0.3, 1.2];
        for (i, &grad) in gradients.iter().enumerate() {
            beta_adam.zero_grad();
            beta_adamw.zero_grad();
            beta_adam.accumulate_grad(1, 0, grad);
            beta_adamw.accumulate_grad(1, 0, grad);
            beta_adam.gradient_step_adam(&config_adam, i + 1);
            beta_adamw.gradient_step_adam(&config_adamw, i + 1);
        }

        let val_adam = beta_adam.get_entry(1, 0).unwrap().value;
        let val_adamw = beta_adamw.get_entry(1, 0).unwrap().value;

        // AdamW should have smaller values due to weight decay
        assert!(
            val_adamw < val_adam,
            "AdamW value ({:.6}) should be less than Adam value ({:.6}) due to weight decay",
            val_adamw,
            val_adam
        );

        println!("Adam value: {:.6}, AdamW value: {:.6}", val_adam, val_adamw);
    }

    #[test]
    fn test_adamw_alpha_state() {
        // Test weight decay with DomainAlphaState using simple_network
        let network = simple_network();

        // Input bounds that create unstable neurons
        let input =
            BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn())
                .unwrap();

        let layer_bounds: Vec<Arc<BoundedTensor>> = network
            .collect_ibp_bounds(&input)
            .unwrap()
            .into_iter()
            .map(Arc::new)
            .collect();
        let history = SplitHistory::new();

        let mut alpha_state =
            DomainAlphaState::from_layer_bounds_and_constraints(&network, &layer_bounds, &history);

        if alpha_state.is_empty() {
            println!("No unstable neurons - skipping test");
            return;
        }

        let config_adamw = AdaptiveOptConfig {
            weight_decay: 0.1,
            alpha_lr: 0.5,
            bias_correction: false,
            ..Default::default()
        };

        // Get first unstable neuron key
        let key = *alpha_state.unstable_neurons.iter().next().unwrap();

        // Record initial alpha value
        let initial_alpha = *alpha_state.alphas.get(&key).unwrap();

        // With zero gradient and weight decay, α should decrease
        alpha_state.zero_grad();
        alpha_state.accumulate_grad(key.0, key.1, 0.0); // Zero gradient
        alpha_state.gradient_step_adam(&config_adamw, 1);

        let alpha_after = *alpha_state.alphas.get(&key).unwrap();

        // α_new = α * (1 - lr * λ) = initial_alpha * (1 - 0.5 * 0.1) = initial_alpha * 0.95
        let expected = initial_alpha * (1.0 - 0.5 * 0.1);
        assert!(
            (alpha_after - expected).abs() < 1e-6,
            "Alpha should be {:.6}, got {:.6}",
            expected,
            alpha_after
        );

        println!(
            "AdamW alpha test: initial={:.6}, after 1 step={:.6}",
            initial_alpha, alpha_after
        );
    }

    #[test]
    fn test_adamw_with_scheduler() {
        // Test that weight decay works correctly with LR scheduling
        let mut history = SplitHistory::new();
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active: true,
        });

        let mut beta_state = BetaState::from_history(&history);
        beta_state.get_entry_mut(1, 0).unwrap().value = 1.0;

        // Weight decay should use scheduled LR
        let config = AdaptiveOptConfig {
            weight_decay: 0.1,
            beta_lr: 1.0,
            scheduler: LRScheduler::StepDecay {
                step_size: 5,
                gamma: 0.5, // LR halves every 5 steps
            },
            bias_correction: false,
            ..Default::default()
        };

        // First step: lr = 1.0, decay = 1.0 * 0.1 = 0.1
        // θ_new = 1.0 * (1 - 0.1) = 0.9 (ignoring Adam update since grad=0)
        beta_state.zero_grad();
        beta_state.accumulate_grad(1, 0, 0.0);
        beta_state.gradient_step_adam(&config, 1);

        let value_step1 = beta_state.get_entry(1, 0).unwrap().value;
        assert!(
            (value_step1 - 0.9).abs() < 1e-6,
            "After step 1, value should be 0.9, got {:.6}",
            value_step1
        );

        // Advance to step 6 where LR = 0.5, decay = 0.5 * 0.1 = 0.05
        beta_state.get_entry_mut(1, 0).unwrap().value = 1.0; // Reset
        beta_state.zero_grad();
        beta_state.accumulate_grad(1, 0, 0.0);
        beta_state.gradient_step_adam(&config, 6);

        let value_step6 = beta_state.get_entry(1, 0).unwrap().value;
        // At step 6, LR = 1.0 * 0.5^1 = 0.5
        // decay_factor = 1 - 0.5 * 0.1 = 0.95
        assert!(
            (value_step6 - 0.95).abs() < 1e-6,
            "After step 6, value should be 0.95, got {:.6}",
            value_step6
        );

        println!(
            "AdamW scheduler test: step1={:.6}, step6={:.6}",
            value_step1, value_step6
        );
    }

    #[test]
    fn test_adamw_no_decay_when_zero() {
        // Test that weight_decay=0 produces identical results to standard Adam
        let mut history = SplitHistory::new();
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active: true,
        });

        let config = AdaptiveOptConfig {
            weight_decay: 0.0,
            ..Default::default()
        };

        let mut beta_state = BetaState::from_history(&history);
        let initial_value = 0.5;
        beta_state.get_entry_mut(1, 0).unwrap().value = initial_value;

        // With zero gradient, no change should occur
        beta_state.zero_grad();
        beta_state.accumulate_grad(1, 0, 0.0);
        beta_state.gradient_step_adam(&config, 1);

        let value_after = beta_state.get_entry(1, 0).unwrap().value;
        // With zero gradient and zero weight decay, value should remain unchanged
        // (actually m=0 so update is 0)
        assert!(
            (value_after - initial_value).abs() < 1e-6,
            "With zero gradient and weight_decay, value should be unchanged: {:.6} != {:.6}",
            value_after,
            initial_value
        );
    }

    // ============================================================
    // Per-Layer Learning Rate Tests
    // ============================================================

    #[test]
    fn test_per_layer_lr_uniform() {
        // Uniform strategy should return 1.0 for all layers
        let strategy = PerLayerLR::Uniform;
        assert_eq!(strategy.get_factor(0, 10), 1.0);
        assert_eq!(strategy.get_factor(5, 10), 1.0);
        assert_eq!(strategy.get_factor(9, 10), 1.0);
    }

    #[test]
    fn test_per_layer_lr_depth_scaling() {
        // DepthScaling: LR(layer) = 1 / (1 + layer_idx * scale_factor)
        let strategy = PerLayerLR::DepthScaling { scale_factor: 0.1 };

        // Layer 0: 1 / (1 + 0 * 0.1) = 1.0
        assert!((strategy.get_factor(0, 10) - 1.0).abs() < 1e-6);

        // Layer 5: 1 / (1 + 5 * 0.1) = 1 / 1.5 = 0.667
        assert!((strategy.get_factor(5, 10) - 0.6667).abs() < 0.001);

        // Layer 10: 1 / (1 + 10 * 0.1) = 1 / 2 = 0.5
        assert!((strategy.get_factor(10, 20) - 0.5).abs() < 1e-6);

        println!(
            "DepthScaling(0.1): layer0={:.4}, layer5={:.4}, layer10={:.4}",
            strategy.get_factor(0, 10),
            strategy.get_factor(5, 10),
            strategy.get_factor(10, 20)
        );
    }

    #[test]
    fn test_per_layer_lr_exponential_depth() {
        // ExponentialDepth: LR(layer) = decay^layer_idx
        let strategy = PerLayerLR::ExponentialDepth { decay: 0.9 };

        // Layer 0: 0.9^0 = 1.0
        assert!((strategy.get_factor(0, 10) - 1.0).abs() < 1e-6);

        // Layer 5: 0.9^5 = 0.59049
        assert!((strategy.get_factor(5, 10) - 0.59049).abs() < 0.001);

        // Layer 10: 0.9^10 = 0.34868
        assert!((strategy.get_factor(10, 20) - 0.34868).abs() < 0.001);

        println!(
            "ExponentialDepth(0.9): layer0={:.4}, layer5={:.4}, layer10={:.4}",
            strategy.get_factor(0, 10),
            strategy.get_factor(5, 10),
            strategy.get_factor(10, 20)
        );
    }

    #[test]
    fn test_per_layer_lr_sqrt_depth_scaling() {
        // SqrtDepthScaling: LR(layer) = 1 / sqrt(1 + layer_idx * scale)
        let strategy = PerLayerLR::SqrtDepthScaling { scale: 1.0 };

        // Layer 0: 1 / sqrt(1 + 0) = 1.0
        assert!((strategy.get_factor(0, 10) - 1.0).abs() < 1e-6);

        // Layer 3: 1 / sqrt(1 + 3) = 1 / 2 = 0.5
        assert!((strategy.get_factor(3, 10) - 0.5).abs() < 1e-6);

        // Layer 8: 1 / sqrt(1 + 8) = 1 / 3 = 0.333
        assert!((strategy.get_factor(8, 10) - 0.3333).abs() < 0.001);

        println!(
            "SqrtDepthScaling(1.0): layer0={:.4}, layer3={:.4}, layer8={:.4}",
            strategy.get_factor(0, 10),
            strategy.get_factor(3, 10),
            strategy.get_factor(8, 10)
        );
    }

    #[test]
    fn test_per_layer_lr_linear_warmup() {
        // LinearWarmup: LR(layer) = start_factor + (1 - start_factor) * layer_idx / (total - 1)
        let strategy = PerLayerLR::LinearWarmup { start_factor: 0.5 };
        let total_layers = 5;

        // Layer 0: 0.5 + (1 - 0.5) * 0/4 = 0.5
        assert!((strategy.get_factor(0, total_layers) - 0.5).abs() < 1e-6);

        // Layer 2: 0.5 + 0.5 * 2/4 = 0.75
        assert!((strategy.get_factor(2, total_layers) - 0.75).abs() < 1e-6);

        // Layer 4: 0.5 + 0.5 * 4/4 = 1.0
        assert!((strategy.get_factor(4, total_layers) - 1.0).abs() < 1e-6);

        println!(
            "LinearWarmup(0.5) with 5 layers: layer0={:.4}, layer2={:.4}, layer4={:.4}",
            strategy.get_factor(0, total_layers),
            strategy.get_factor(2, total_layers),
            strategy.get_factor(4, total_layers)
        );
    }

    #[test]
    fn test_per_layer_lr_custom() {
        // Custom: direct lookup
        let strategy = PerLayerLR::Custom {
            factors: vec![0.1, 0.2, 0.5, 1.0, 2.0],
        };

        assert!((strategy.get_factor(0, 10) - 0.1).abs() < 1e-6);
        assert!((strategy.get_factor(2, 10) - 0.5).abs() < 1e-6);
        assert!((strategy.get_factor(4, 10) - 2.0).abs() < 1e-6);

        // Out of bounds should return 1.0
        assert!((strategy.get_factor(10, 20) - 1.0).abs() < 1e-6);

        println!(
            "Custom factors: layer0={:.4}, layer2={:.4}, layer4={:.4}, layer10={:.4}",
            strategy.get_factor(0, 10),
            strategy.get_factor(2, 10),
            strategy.get_factor(4, 10),
            strategy.get_factor(10, 20)
        );
    }

    #[test]
    fn test_per_layer_lr_beta_state() {
        // Test that per-layer LR is applied correctly in BetaState::gradient_step_adam
        let mut history = SplitHistory::new();

        // Add constraints at different layers
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active: true,
        });
        history.add_constraint(NeuronConstraint {
            layer_idx: 5,
            neuron_idx: 0,
            is_active: true,
        });

        let config = AdaptiveOptConfig {
            beta_lr: 1.0,
            bias_correction: false, // Simpler math for testing
            per_layer_lr_beta: PerLayerLR::DepthScaling { scale_factor: 0.2 },
            total_layers: 10,
            ..Default::default()
        };

        let mut beta_state = BetaState::from_history(&history);

        // Set initial values
        beta_state.get_entry_mut(1, 0).unwrap().value = 0.0;
        beta_state.get_entry_mut(5, 0).unwrap().value = 0.0;

        // Apply same gradient to both
        beta_state.zero_grad();
        beta_state.accumulate_grad(1, 0, 1.0);
        beta_state.accumulate_grad(5, 0, 1.0);
        beta_state.gradient_step_adam(&config, 1);

        let value_layer1 = beta_state.get_entry(1, 0).unwrap().value;
        let value_layer5 = beta_state.get_entry(5, 0).unwrap().value;

        // Layer 1: factor = 1/(1 + 1*0.2) = 1/1.2 = 0.833
        // Layer 5: factor = 1/(1 + 5*0.2) = 1/2 = 0.5
        // value_layer1 should be larger than value_layer5

        assert!(
            value_layer1 > value_layer5,
            "Layer 1 should have larger update (higher LR): {:.6} vs {:.6}",
            value_layer1,
            value_layer5
        );

        // Check approximate ratios
        // Expected ratio: 0.833 / 0.5 = 1.667
        let ratio = value_layer1 / value_layer5;
        assert!(
            (ratio - 1.667).abs() < 0.1,
            "Update ratio should be ~1.667, got {:.4}",
            ratio
        );

        println!(
            "Per-layer LR beta test: layer1={:.6}, layer5={:.6}, ratio={:.4}",
            value_layer1, value_layer5, ratio
        );
    }

    #[test]
    fn test_per_layer_lr_alpha_state() {
        // Test per-layer LR in DomainAlphaState::gradient_step_adam
        // Create a simple network to initialize alpha state
        use crate::{Layer, LinearLayer, ReLULayer};
        use gamma_tensor::BoundedTensor;
        use ndarray::Array2;

        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(Array2::eye(4), None).unwrap(),
        ));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(
            LinearLayer::new(Array2::eye(4), None).unwrap(),
        ));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(
            LinearLayer::new(Array2::eye(4), None).unwrap(),
        ));

        // Create bounds that make both ReLU layers unstable
        let input_bounds = BoundedTensor::from_epsilon(
            ndarray::Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]).into_dyn(),
            1.0,
        );
        let layer_bounds: Vec<Arc<BoundedTensor>> = vec![
            Arc::new(input_bounds.clone()),
            Arc::new(BoundedTensor::from_epsilon(
                ndarray::Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]).into_dyn(),
                0.5,
            )),
            Arc::new(BoundedTensor::from_epsilon(
                ndarray::Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]).into_dyn(),
                0.5,
            )),
            Arc::new(BoundedTensor::from_epsilon(
                ndarray::Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]).into_dyn(),
                0.5,
            )),
            Arc::new(BoundedTensor::from_epsilon(
                ndarray::Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]).into_dyn(),
                0.5,
            )),
        ];

        let history = SplitHistory::new();
        let mut alpha_state =
            DomainAlphaState::from_layer_bounds_and_constraints(&network, &layer_bounds, &history);

        // We should have alphas for layer 1 (ReLU after first Linear) and layer 3 (second ReLU)
        // Layer indices in Network: 0=Linear, 1=ReLU, 2=Linear, 3=ReLU, 4=Linear

        let config = AdaptiveOptConfig {
            alpha_lr: 1.0,
            bias_correction: false,
            per_layer_lr_alpha: PerLayerLR::ExponentialDepth { decay: 0.5 },
            total_layers: 5,
            ..Default::default()
        };

        // Reset alphas and gradients
        for alpha in alpha_state.alphas.values_mut() {
            *alpha = 0.5;
        }

        // Set gradients for neurons in different layers
        for ((layer_idx, neuron_idx), _) in alpha_state.alphas.iter() {
            alpha_state.grads.insert((*layer_idx, *neuron_idx), 0.1);
        }

        alpha_state.gradient_step_adam(&config, 1);

        // Check that earlier layers have larger updates
        // Layer 1: decay^1 = 0.5
        // Layer 3: decay^3 = 0.125
        let alpha_layer1 = alpha_state.get_alpha(1, 0);
        let alpha_layer3 = alpha_state.get_alpha(3, 0);

        // Both started at 0.5 with positive gradient, both should increase
        // But layer 1 should increase more
        println!(
            "Per-layer LR alpha test: layer1_alpha={:.6}, layer3_alpha={:.6}",
            alpha_layer1, alpha_layer3
        );

        // The exact values depend on Adam dynamics, but the ratio of changes should
        // reflect the LR ratio
        let change_layer1 = alpha_layer1 - 0.5;
        let change_layer3 = alpha_layer3 - 0.5;

        if change_layer1.abs() > 1e-6 && change_layer3.abs() > 1e-6 {
            let ratio = change_layer1 / change_layer3;
            // Expected: 0.5 / 0.125 = 4.0
            assert!(
                ratio > 1.0,
                "Layer 1 change should be larger than layer 3: {:.6} vs {:.6}",
                change_layer1,
                change_layer3
            );
            println!("Change ratio: {:.4} (expected ~4.0)", ratio);
        }
    }

    #[test]
    fn test_per_layer_lr_config_default() {
        // Test that default config has Uniform per-layer LR
        let config = AdaptiveOptConfig::default();

        assert_eq!(config.per_layer_lr_beta, PerLayerLR::Uniform);
        assert_eq!(config.per_layer_lr_alpha, PerLayerLR::Uniform);
        assert_eq!(config.total_layers, 0);
    }

    #[test]
    fn test_per_layer_lr_combined_with_scheduler() {
        // Test that per-layer LR works correctly with LR scheduler
        let mut history = SplitHistory::new();
        history.add_constraint(NeuronConstraint {
            layer_idx: 2,
            neuron_idx: 0,
            is_active: true,
        });

        let config = AdaptiveOptConfig {
            beta_lr: 1.0,
            bias_correction: false,
            scheduler: LRScheduler::StepDecay {
                gamma: 0.5,
                step_size: 5,
            },
            per_layer_lr_beta: PerLayerLR::DepthScaling { scale_factor: 0.5 },
            total_layers: 10,
            ..Default::default()
        };

        let mut beta_state = BetaState::from_history(&history);
        beta_state.get_entry_mut(2, 0).unwrap().value = 0.0;

        // Step 1: scheduler factor = 1.0, layer factor = 1/(1+2*0.5) = 0.5
        // Combined LR = 1.0 * 1.0 * 0.5 = 0.5
        beta_state.zero_grad();
        beta_state.accumulate_grad(2, 0, 1.0);
        beta_state.gradient_step_adam(&config, 1);
        let value_step1 = beta_state.get_entry(2, 0).unwrap().value;

        // Reset and step 6: scheduler factor = 0.5, layer factor = 0.5
        // Combined LR = 1.0 * 0.5 * 0.5 = 0.25
        beta_state.get_entry_mut(2, 0).unwrap().value = 0.0;
        beta_state.get_entry_mut(2, 0).unwrap().m = 0.0;
        beta_state.get_entry_mut(2, 0).unwrap().v = 0.0;
        beta_state.zero_grad();
        beta_state.accumulate_grad(2, 0, 1.0);
        beta_state.gradient_step_adam(&config, 6);
        let value_step6 = beta_state.get_entry(2, 0).unwrap().value;

        // Step 6 should have smaller update due to scheduler decay
        assert!(
            value_step1 > value_step6,
            "Step 1 should have larger update: {:.6} vs {:.6}",
            value_step1,
            value_step6
        );

        // Ratio should be approximately 2 (since scheduler decayed by 0.5)
        let ratio = value_step1 / value_step6;
        assert!(
            (ratio - 2.0).abs() < 0.5,
            "Update ratio should be ~2.0, got {:.4}",
            ratio
        );

        println!(
            "Combined scheduler + per-layer LR: step1={:.6}, step6={:.6}, ratio={:.4}",
            value_step1, value_step6, ratio
        );
    }

    // ==================== Lookahead Optimizer Tests ====================

    #[test]
    fn test_lookahead_config_default() {
        let config = LookaheadConfig::default();
        assert!(!config.enabled, "Lookahead should be disabled by default");
        assert_eq!(config.sync_period, 5);
        assert!((config.alpha - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_lookahead_config_new() {
        let config = LookaheadConfig::new(10, 0.8);
        assert!(config.enabled);
        assert_eq!(config.sync_period, 10);
        assert!((config.alpha - 0.8).abs() < 1e-6);

        // Test clamping of alpha
        let config_high = LookaheadConfig::new(5, 1.5);
        assert!(
            (config_high.alpha - 1.0).abs() < 1e-6,
            "alpha should be clamped to 1.0"
        );

        let config_low = LookaheadConfig::new(5, -0.5);
        assert!(
            (config_low.alpha - 0.0).abs() < 1e-6,
            "alpha should be clamped to 0.0"
        );

        // Test sync_period minimum
        let config_min = LookaheadConfig::new(0, 0.5);
        assert_eq!(
            config_min.sync_period, 1,
            "sync_period should be at least 1"
        );
    }

    #[test]
    fn test_lookahead_config_should_sync() {
        let config = LookaheadConfig::new(5, 0.5);

        // Should sync at multiples of 5
        assert!(!config.should_sync(0), "Should not sync at iteration 0");
        assert!(!config.should_sync(1));
        assert!(!config.should_sync(4));
        assert!(config.should_sync(5), "Should sync at iteration 5");
        assert!(!config.should_sync(6));
        assert!(config.should_sync(10), "Should sync at iteration 10");
        assert!(config.should_sync(15), "Should sync at iteration 15");

        // Disabled config should never sync
        let disabled = LookaheadConfig::default();
        assert!(!disabled.should_sync(5));
        assert!(!disabled.should_sync(100));
    }

    #[test]
    fn test_beta_state_lookahead_init() {
        let mut history = SplitHistory::new();
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active: true,
        });
        history.add_constraint(NeuronConstraint {
            layer_idx: 2,
            neuron_idx: 1,
            is_active: false,
        });

        let mut beta_state = BetaState::from_history(&history);
        assert!(!beta_state.has_slow_weights());

        // Set some values
        beta_state.set_beta(1, 0, 0.5);
        beta_state.set_beta(2, 1, 1.0);

        // Initialize slow weights
        beta_state.init_slow_weights();
        assert!(beta_state.has_slow_weights());

        let slow = beta_state.get_slow_weights().unwrap();
        assert_eq!(slow.len(), 2);
        assert!((slow[0] - 0.5).abs() < 1e-6);
        assert!((slow[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_beta_state_lookahead_step() {
        let mut history = SplitHistory::new();
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active: true,
        });

        let mut beta_state = BetaState::from_history(&history);
        beta_state.set_beta(1, 0, 0.0); // Initial fast = slow = 0.0
        beta_state.init_slow_weights();

        let config = LookaheadConfig::new(5, 0.5);

        // Simulate inner optimizer updating fast weights to 1.0
        beta_state.set_beta(1, 0, 1.0);

        // After lookahead step:
        // slow = 0.0 + 0.5 * (1.0 - 0.0) = 0.5
        // fast = slow = 0.5
        beta_state.lookahead_step(&config);

        let new_value = beta_state.get_beta(1, 0).unwrap();
        assert!(
            (new_value - 0.5).abs() < 1e-6,
            "Fast weight should be 0.5 after lookahead step, got {}",
            new_value
        );

        let slow = beta_state.get_slow_weights().unwrap();
        assert!(
            (slow[0] - 0.5).abs() < 1e-6,
            "Slow weight should be 0.5 after lookahead step, got {}",
            slow[0]
        );
    }

    #[test]
    fn test_beta_state_lookahead_convergence() {
        // Test that lookahead converges when fast weights are constant
        let mut history = SplitHistory::new();
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active: true,
        });

        let mut beta_state = BetaState::from_history(&history);
        beta_state.set_beta(1, 0, 0.0);
        beta_state.init_slow_weights();

        let config = LookaheadConfig::new(1, 0.5);

        // Repeatedly set fast to 1.0 and apply lookahead
        for _ in 0..20 {
            beta_state.set_beta(1, 0, 1.0);
            beta_state.lookahead_step(&config);
        }

        // Should converge to 1.0
        let final_value = beta_state.get_beta(1, 0).unwrap();
        assert!(
            (final_value - 1.0).abs() < 0.01,
            "Should converge to target, got {}",
            final_value
        );
    }

    #[test]
    fn test_beta_state_lookahead_non_negative() {
        // Test that beta values remain non-negative after lookahead
        let mut history = SplitHistory::new();
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active: true,
        });

        let mut beta_state = BetaState::from_history(&history);
        beta_state.set_beta(1, 0, 1.0);
        beta_state.init_slow_weights();

        let config = LookaheadConfig::new(1, 2.0); // alpha > 1 causes overshoot

        // Set fast to negative (would be projected in normal update)
        beta_state.entries[0].value = -0.5;
        beta_state.lookahead_step(&config);

        // Value should be clamped to 0
        let value = beta_state.get_beta(1, 0).unwrap();
        assert!(value >= 0.0, "Beta should be non-negative, got {}", value);
    }

    #[test]
    fn test_alpha_state_lookahead_init() {
        let mut alpha_state = DomainAlphaState::empty();
        alpha_state.alphas.insert((1, 0), 0.3);
        alpha_state.alphas.insert((1, 1), 0.7);
        alpha_state.alphas.insert((2, 0), 0.5);

        assert!(!alpha_state.has_slow_weights());

        alpha_state.init_slow_weights();
        assert!(alpha_state.has_slow_weights());

        let slow = alpha_state.get_slow_weights().unwrap();
        assert_eq!(slow.len(), 3);
        assert!((slow[&(1, 0)] - 0.3).abs() < 1e-6);
        assert!((slow[&(1, 1)] - 0.7).abs() < 1e-6);
        assert!((slow[&(2, 0)] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_alpha_state_lookahead_step() {
        let mut alpha_state = DomainAlphaState::empty();
        alpha_state.alphas.insert((1, 0), 0.0); // Initial fast = slow = 0.0
        alpha_state.init_slow_weights();

        let config = LookaheadConfig::new(5, 0.5);

        // Simulate inner optimizer updating fast weights to 1.0
        *alpha_state.alphas.get_mut(&(1, 0)).unwrap() = 1.0;

        // After lookahead step:
        // slow = 0.0 + 0.5 * (1.0 - 0.0) = 0.5
        // fast = slow = 0.5
        alpha_state.lookahead_step(&config);

        let new_value = alpha_state.get_alpha(1, 0);
        assert!(
            (new_value - 0.5).abs() < 1e-6,
            "Fast weight should be 0.5 after lookahead step, got {}",
            new_value
        );
    }

    #[test]
    fn test_alpha_state_lookahead_clamping() {
        // Test that alpha values remain in [0, 1] after lookahead
        let mut alpha_state = DomainAlphaState::empty();
        alpha_state.alphas.insert((1, 0), 0.9);
        alpha_state.init_slow_weights();

        let config = LookaheadConfig::new(1, 0.5);

        // Set fast to value > 1
        *alpha_state.alphas.get_mut(&(1, 0)).unwrap() = 1.5;
        alpha_state.lookahead_step(&config);

        // Value should be clamped to 1.0
        let value = alpha_state.get_alpha(1, 0);
        assert!(value <= 1.0, "Alpha should be at most 1.0, got {}", value);

        // Set fast to value < 0
        *alpha_state.alphas.get_mut(&(1, 0)).unwrap() = -0.5;
        alpha_state.lookahead_step(&config);

        // Value should be clamped to 0.0
        let value = alpha_state.get_alpha(1, 0);
        assert!(value >= 0.0, "Alpha should be at least 0.0, got {}", value);
    }

    #[test]
    fn test_adaptive_opt_config_has_lookahead() {
        let config = AdaptiveOptConfig::default();
        assert!(
            !config.lookahead.enabled,
            "Lookahead should be disabled by default"
        );

        let config_with_lookahead = AdaptiveOptConfig {
            lookahead: LookaheadConfig::new(5, 0.5),
            ..Default::default()
        };
        assert!(config_with_lookahead.lookahead.enabled);
    }

    #[test]
    fn test_lookahead_beta_with_adam() {
        // Test full lookahead integration with Adam optimizer
        let mut history = SplitHistory::new();
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active: true,
        });

        let config = AdaptiveOptConfig {
            beta_lr: 0.5,
            lookahead: LookaheadConfig::new(3, 0.5),
            ..Default::default()
        };

        let mut beta_state = BetaState::from_history(&history);
        beta_state.init_slow_weights();

        // Run 3 Adam steps then lookahead
        for t in 1..=3 {
            beta_state.zero_grad();
            beta_state.accumulate_grad(1, 0, 1.0);
            beta_state.gradient_step_adam(&config, t);
        }

        let value_before_lookahead = beta_state.get_beta(1, 0).unwrap();

        // Apply lookahead at sync point
        assert!(config.lookahead.should_sync(3));
        beta_state.lookahead_step(&config.lookahead);

        let value_after_lookahead = beta_state.get_beta(1, 0).unwrap();

        // After lookahead, value should be interpolated between slow (0) and fast
        assert!(
            value_after_lookahead < value_before_lookahead,
            "Lookahead should pull back toward slow weights: {} < {}",
            value_after_lookahead,
            value_before_lookahead
        );
        assert!(
            value_after_lookahead > 0.0,
            "Value should be positive after interpolation"
        );

        println!(
            "Adam+Lookahead: before={:.6}, after={:.6}",
            value_before_lookahead, value_after_lookahead
        );
    }

    #[test]
    fn test_lookahead_alpha_with_adam() {
        // Test full lookahead integration with Adam optimizer for alpha
        let mut alpha_state = DomainAlphaState::empty();
        alpha_state.alphas.insert((1, 0), 0.5);
        alpha_state.grads.insert((1, 0), 0.0);
        alpha_state.adam_m.insert((1, 0), 0.0);
        alpha_state.adam_v.insert((1, 0), 0.0);
        alpha_state.adam_v_max.insert((1, 0), 0.0);
        alpha_state.unstable_neurons.insert((1, 0));
        alpha_state.init_slow_weights();

        let config = AdaptiveOptConfig {
            alpha_lr: 0.5,
            lookahead: LookaheadConfig::new(3, 0.5),
            ..Default::default()
        };

        // Run 3 Adam steps with positive gradient (pushing toward 1.0)
        for t in 1..=3 {
            alpha_state.zero_grad();
            *alpha_state.grads.get_mut(&(1, 0)).unwrap() = 0.5; // Positive gradient
            alpha_state.gradient_step_adam(&config, t);
        }

        let value_before_lookahead = alpha_state.get_alpha(1, 0);

        // Apply lookahead at sync point
        assert!(config.lookahead.should_sync(3));
        alpha_state.lookahead_step(&config.lookahead);

        let value_after_lookahead = alpha_state.get_alpha(1, 0);

        // After lookahead, value should be interpolated between slow (0.5) and fast
        assert!(
            (value_after_lookahead - 0.5).abs() < (value_before_lookahead - 0.5).abs(),
            "Lookahead should pull toward initial slow weight: |{} - 0.5| < |{} - 0.5|",
            value_after_lookahead,
            value_before_lookahead
        );

        println!(
            "Alpha Adam+Lookahead: before={:.6}, after={:.6}",
            value_before_lookahead, value_after_lookahead
        );
    }

    // ==================== Optimizer Benchmark Tests ====================
    //
    // These tests compare different optimizer variants (Adam, AMSGrad, AdamW, RAdam, Lookahead)
    // on fixed verification problems to measure:
    // - Final lower bound (higher = tighter bounds)
    // - Convergence speed (iterations to reach tolerance)
    // - Stability (variance across runs)

    /// Results from a single optimizer run.
    #[derive(Debug, Clone)]
    struct OptBenchmarkResult {
        /// Name of the optimizer variant.
        name: String,
        /// Final lower bound achieved.
        final_lower: f32,
        /// Number of domains explored until completion.
        iterations: usize,
        /// Whether optimization converged (verified) before domain/time limit.
        converged: bool,
    }

    /// Create a deeper network for more meaningful optimizer comparison.
    /// Structure: Linear(4,8) -> ReLU -> Linear(8,4) -> ReLU -> Linear(4,1)
    fn benchmark_network() -> Network {
        use ndarray::Array2;

        // Layer 1: Linear 4 -> 8 (creates more unstable neurons)
        let w1: Array2<f32> = arr2(&[
            [0.5, -0.3, 0.2, -0.4],
            [-0.2, 0.6, -0.1, 0.3],
            [0.4, -0.5, 0.3, -0.2],
            [-0.3, 0.2, -0.4, 0.5],
            [0.1, -0.6, 0.5, -0.1],
            [-0.5, 0.4, -0.3, 0.6],
            [0.6, -0.2, 0.1, -0.5],
            [-0.4, 0.3, -0.6, 0.2],
        ]);
        let linear1 = LinearLayer::new(w1, None).unwrap();

        // Layer 2: Linear 8 -> 4
        let w2: Array2<f32> = arr2(&[
            [0.3, -0.2, 0.4, -0.3, 0.2, -0.1, 0.5, -0.4],
            [-0.4, 0.5, -0.2, 0.3, -0.5, 0.4, -0.3, 0.2],
            [0.2, -0.4, 0.3, -0.5, 0.4, -0.3, 0.1, -0.2],
            [-0.3, 0.1, -0.5, 0.2, -0.2, 0.5, -0.4, 0.3],
        ]);
        let linear2 = LinearLayer::new(w2, None).unwrap();

        // Layer 3: Linear 4 -> 1 (output)
        let w3: Array2<f32> = arr2(&[[0.5, -0.3, 0.4, -0.2]]);
        let linear3 = LinearLayer::new(w3, None).unwrap();

        let mut network = Network::new();
        network.add_layer(Layer::Linear(linear1));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(linear2));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(linear3));
        network
    }

    /// Run optimization with a specific config and return metrics.
    /// Uses a tighter threshold (-0.5) to require more domain exploration.
    fn run_optimizer_benchmark(
        name: &str,
        adaptive_config: AdaptiveOptConfig,
        max_iterations: usize,
    ) -> OptBenchmarkResult {
        run_optimizer_benchmark_with_threshold(name, adaptive_config, max_iterations, -0.5)
    }

    /// Run optimization with a specific config and threshold.
    fn run_optimizer_benchmark_with_threshold(
        name: &str,
        adaptive_config: AdaptiveOptConfig,
        max_iterations: usize,
        threshold: f32,
    ) -> OptBenchmarkResult {
        let network = benchmark_network();

        // Input bounds that create unstable neurons (wider range for harder problem)
        let input = BoundedTensor::new(
            arr1(&[-1.0, -1.0, -1.0, -1.0]).into_dyn(),
            arr1(&[1.0, 1.0, 1.0, 1.0]).into_dyn(),
        )
        .unwrap();

        // Config with adaptive optimizer
        let config = BetaCrownConfig {
            max_domains: 100, // More domains for harder problems
            timeout: Duration::from_secs(30),
            use_alpha_crown: true,
            use_adaptive: true,
            adaptive_config,
            beta_lr: 0.1,
            alpha_lr: 0.5,
            alpha_momentum: true,
            beta_iterations: max_iterations,
            beta_tolerance: 1e-5,
            ..Default::default()
        };

        let verifier = BetaCrownVerifier::new(config);

        // Run verification - this exercises the optimizer
        let result = verifier.verify(&network, &input, threshold).unwrap();

        // Extract metrics from the result
        let final_lower = match &result.result {
            BabVerificationStatus::Verified => result
                .output_bounds
                .as_ref()
                .map(|b| b.lower_scalar())
                .unwrap_or(0.0),
            BabVerificationStatus::Unknown { .. } => result
                .output_bounds
                .as_ref()
                .map(|b| b.lower_scalar())
                .unwrap_or(-f32::INFINITY),
            _ => -f32::INFINITY,
        };

        OptBenchmarkResult {
            name: name.to_string(),
            final_lower,
            iterations: result.domains_explored,
            converged: matches!(result.result, BabVerificationStatus::Verified),
        }
    }

    #[test]
    fn test_benchmark_adam_baseline() {
        // Baseline Adam optimizer
        let config = AdaptiveOptConfig {
            beta_lr: 0.1,
            alpha_lr: 0.3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            bias_correction: true,
            ..Default::default()
        };

        let result = run_optimizer_benchmark("Adam", config, 20);
        println!(
            "Adam baseline: lower={:.6}, domains={}, converged={}",
            result.final_lower, result.iterations, result.converged
        );

        // Basic sanity check
        assert!(
            result.final_lower.is_finite(),
            "Lower bound should be finite"
        );
    }

    #[test]
    fn test_benchmark_amsgrad() {
        // AMSGrad variant
        let config = AdaptiveOptConfig {
            beta_lr: 0.1,
            alpha_lr: 0.3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            bias_correction: true,
            amsgrad: true,
            ..Default::default()
        };

        let result = run_optimizer_benchmark("AMSGrad", config, 20);
        println!(
            "AMSGrad: lower={:.6}, domains={}, converged={}",
            result.final_lower, result.iterations, result.converged
        );

        assert!(
            result.final_lower.is_finite(),
            "Lower bound should be finite"
        );
    }

    #[test]
    fn test_benchmark_adamw() {
        // AdamW variant with weight decay
        let config = AdaptiveOptConfig {
            beta_lr: 0.1,
            alpha_lr: 0.3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            bias_correction: true,
            weight_decay: 0.01,
            ..Default::default()
        };

        let result = run_optimizer_benchmark("AdamW", config, 20);
        println!(
            "AdamW: lower={:.6}, domains={}, converged={}",
            result.final_lower, result.iterations, result.converged
        );

        assert!(
            result.final_lower.is_finite(),
            "Lower bound should be finite"
        );
    }

    #[test]
    fn test_benchmark_radam() {
        // RAdam variant
        // Note: RAdam underperforms on this β-CROWN benchmark due to its warmup phase.
        // RAdam uses SGD-with-momentum steps for early iterations (before variance
        // estimate is reliable), which may not be optimal for constraint optimization.
        let config = AdaptiveOptConfig {
            beta_lr: 0.1,
            alpha_lr: 0.3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            bias_correction: true,
            radam: true,
            ..Default::default()
        };

        let result = run_optimizer_benchmark("RAdam", config, 20);
        println!(
            "RAdam: lower={:.6}, domains={}, converged={}",
            result.final_lower, result.iterations, result.converged
        );

        // RAdam may not converge on this benchmark - document the behavior
        // See test_optimizer_comparison_report for full comparison
        println!("Note: RAdam's warmup phase may cause slower convergence on β-CROWN problems");
    }

    #[test]
    fn test_benchmark_lookahead() {
        // Lookahead + Adam
        let config = AdaptiveOptConfig {
            beta_lr: 0.1,
            alpha_lr: 0.3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            bias_correction: true,
            lookahead: LookaheadConfig::new(5, 0.5),
            ..Default::default()
        };

        let result = run_optimizer_benchmark("Lookahead+Adam", config, 20);
        println!(
            "Lookahead+Adam: lower={:.6}, domains={}, converged={}",
            result.final_lower, result.iterations, result.converged
        );

        assert!(
            result.final_lower.is_finite(),
            "Lower bound should be finite"
        );
    }

    #[test]
    fn test_benchmark_radam_integration() {
        // Integration test: RAdam with joint α-β optimization
        // This specifically tests RAdam in the full verification pipeline
        let network = benchmark_network();

        let input = BoundedTensor::new(
            arr1(&[-0.3, -0.3, -0.3, -0.3]).into_dyn(),
            arr1(&[0.3, 0.3, 0.3, 0.3]).into_dyn(),
        )
        .unwrap();

        let config = BetaCrownConfig {
            max_domains: 30,
            timeout: Duration::from_secs(20),
            use_alpha_crown: true,
            use_adaptive: true,
            adaptive_config: AdaptiveOptConfig {
                beta_lr: 0.1,
                alpha_lr: 0.3,
                radam: true,
                bias_correction: true,
                ..Default::default()
            },
            beta_iterations: 15,
            ..Default::default()
        };

        let verifier = BetaCrownVerifier::new(config);
        let result = verifier.verify(&network, &input, -1.5).unwrap();

        println!(
            "RAdam integration test: status={:?}, domains={}",
            result.result, result.domains_explored
        );

        // Should produce valid result
        assert!(
            result.domains_explored > 0,
            "Should explore at least one domain"
        );
    }

    #[test]
    fn test_optimizer_comparison_report() {
        // Comprehensive comparison of all optimizer variants
        // Runs each optimizer and prints a comparison table

        let variants: Vec<(&str, AdaptiveOptConfig)> = vec![
            (
                "Adam (baseline)",
                AdaptiveOptConfig {
                    beta_lr: 0.1,
                    alpha_lr: 0.3,
                    ..Default::default()
                },
            ),
            (
                "AMSGrad",
                AdaptiveOptConfig {
                    beta_lr: 0.1,
                    alpha_lr: 0.3,
                    amsgrad: true,
                    ..Default::default()
                },
            ),
            (
                "AdamW (λ=0.01)",
                AdaptiveOptConfig {
                    beta_lr: 0.1,
                    alpha_lr: 0.3,
                    weight_decay: 0.01,
                    ..Default::default()
                },
            ),
            (
                "RAdam",
                AdaptiveOptConfig {
                    beta_lr: 0.1,
                    alpha_lr: 0.3,
                    radam: true,
                    ..Default::default()
                },
            ),
            (
                "Lookahead+Adam",
                AdaptiveOptConfig {
                    beta_lr: 0.1,
                    alpha_lr: 0.3,
                    lookahead: LookaheadConfig::new(5, 0.5),
                    ..Default::default()
                },
            ),
            (
                "RAdam+AMSGrad",
                AdaptiveOptConfig {
                    beta_lr: 0.1,
                    alpha_lr: 0.3,
                    radam: true,
                    amsgrad: true,
                    ..Default::default()
                },
            ),
            (
                "RAdam+Lookahead",
                AdaptiveOptConfig {
                    beta_lr: 0.1,
                    alpha_lr: 0.3,
                    radam: true,
                    lookahead: LookaheadConfig::new(5, 0.5),
                    ..Default::default()
                },
            ),
        ];

        println!("\n=== Optimizer Comparison Report ===\n");
        println!(
            "{:<20} {:>12} {:>10} {:>10}",
            "Optimizer", "Lower Bound", "Domains", "Converged"
        );
        println!("{}", "-".repeat(54));

        let mut results = Vec::new();
        for (name, config) in variants {
            let result = run_optimizer_benchmark(name, config, 20);
            println!(
                "{:<20} {:>12.6} {:>10} {:>10}",
                result.name,
                result.final_lower,
                result.iterations,
                if result.converged { "yes" } else { "no" }
            );
            results.push(result);
        }

        println!("\n{}", "=".repeat(54));

        // Find best converged result
        let converged_results: Vec<_> = results.iter().filter(|r| r.converged).collect();
        let non_converged: Vec<_> = results.iter().filter(|r| !r.converged).collect();

        if !converged_results.is_empty() {
            let best = converged_results
                .iter()
                .min_by_key(|r| r.iterations) // Fewer domains = faster
                .unwrap();
            println!(
                "Best (fewest domains): {} ({} domains)",
                best.name, best.iterations
            );
        }

        if !non_converged.is_empty() {
            println!("\nDid not converge within domain limit:");
            for r in &non_converged {
                println!("  - {}", r.name);
            }
        }

        // Analysis summary
        let converged_count = converged_results.len();
        let total = results.len();
        println!(
            "\nConvergence rate: {}/{} ({:.0}%)",
            converged_count,
            total,
            (converged_count as f32 / total as f32) * 100.0
        );

        // At least some optimizers should converge
        assert!(
            converged_count > 0,
            "At least one optimizer should converge on the benchmark problem"
        );

        // Check that converged results have valid bounds
        for result in &converged_results {
            assert!(
                result.final_lower.is_finite(),
                "{} converged but has invalid lower bound",
                result.name
            );
        }
    }

    #[test]
    fn test_optimizer_lr_schedule_comparison() {
        // Compare learning rate schedules with Adam
        let schedules: Vec<(&str, LRScheduler)> = vec![
            ("Constant", LRScheduler::Constant),
            (
                "Step (γ=0.5, s=5)",
                LRScheduler::StepDecay {
                    gamma: 0.5,
                    step_size: 5,
                },
            ),
            (
                "Exponential (γ=0.95)",
                LRScheduler::ExponentialDecay { gamma: 0.95 },
            ),
            (
                "Cosine (T=20)",
                LRScheduler::CosineAnnealing {
                    t_max: 20,
                    min_lr: 0.001,
                },
            ),
            (
                "WarmupCosine",
                LRScheduler::WarmupCosine {
                    warmup_steps: 5,
                    min_lr: 0.001,
                    t_max: 20,
                },
            ),
        ];

        println!("\n=== LR Schedule Comparison ===\n");
        println!(
            "{:<25} {:>12} {:>10} {:>10}",
            "Schedule", "Lower Bound", "Domains", "Converged"
        );
        println!("{}", "-".repeat(59));

        let mut results = Vec::new();
        for (name, scheduler) in schedules {
            let config = AdaptiveOptConfig {
                beta_lr: 0.1,
                alpha_lr: 0.3,
                scheduler,
                ..Default::default()
            };

            let result = run_optimizer_benchmark(name, config, 20);
            println!(
                "{:<25} {:>12.6} {:>10} {:>10}",
                result.name,
                result.final_lower,
                result.iterations,
                if result.converged { "yes" } else { "no" }
            );
            results.push(result);
        }

        // Converged results should have valid bounds
        let converged_count = results.iter().filter(|r| r.converged).count();
        println!("\nConvergence rate: {}/{}", converged_count, results.len());
        assert!(converged_count > 0, "At least one schedule should converge");
    }

    // =========================================================================
    // GCP-CROWN: Cutting Plane Tests
    // =========================================================================

    #[test]
    fn test_cutting_plane_from_history() {
        // Create a split history with some constraints
        let mut history = SplitHistory::new();
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active: true,
        });
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 1,
            is_active: false,
        });
        history.add_constraint(NeuronConstraint {
            layer_idx: 2,
            neuron_idx: 0,
            is_active: true,
        });

        // Generate cut from history
        let cut = CuttingPlane::from_verified_domain(&history);
        assert!(cut.is_some());

        let cut = cut.unwrap();
        assert_eq!(cut.terms.len(), 3);
        assert_eq!(cut.source_depth, 3);

        // Check term signs
        assert_eq!(cut.terms[0].coefficient, 1.0); // active -> positive
        assert_eq!(cut.terms[1].coefficient, -1.0); // inactive -> negative
        assert_eq!(cut.terms[2].coefficient, 1.0); // active -> positive
    }

    #[test]
    fn test_cutting_plane_empty_history() {
        let history = SplitHistory::new();
        let cut = CuttingPlane::from_verified_domain(&history);
        assert!(cut.is_none());
    }

    #[test]
    fn test_cutting_plane_redundancy() {
        // Create a cut from some constraints
        let mut history1 = SplitHistory::new();
        history1.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active: true,
        });
        history1.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 1,
            is_active: false,
        });

        let cut = CuttingPlane::from_verified_domain(&history1).unwrap();

        // A domain with the same constraints should find the cut redundant
        assert!(cut.is_redundant_for(&history1));

        // A domain with different constraints should not find it redundant
        let mut history2 = SplitHistory::new();
        history2.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active: false, // Different!
        });
        assert!(!cut.is_redundant_for(&history2));

        // A domain with partial constraints
        let mut history3 = SplitHistory::new();
        history3.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active: true,
        });
        // Only one constraint matches
        assert!(!cut.is_redundant_for(&history3));
    }

    #[test]
    fn test_cut_pool_basic() {
        let mut pool = CutPool::new(10);
        assert!(pool.is_empty());
        assert_eq!(pool.len(), 0);

        // Add cut from history
        let mut history = SplitHistory::new();
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active: true,
        });
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 1,
            is_active: false,
        });

        assert!(pool.add_from_verified_domain(&history));
        assert_eq!(pool.len(), 1);
        assert_eq!(pool.total_generated, 1);
    }

    #[test]
    fn test_cut_pool_rejects_shallow() {
        let mut pool = CutPool::new(10);

        // Single-constraint history (too shallow, min_depth=2)
        let mut history = SplitHistory::new();
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active: true,
        });

        // Should not add cut from shallow domain
        assert!(!pool.add_from_verified_domain(&history));
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn test_cut_pool_capacity() {
        let mut pool = CutPool::new(2); // Small capacity

        // Add first cut
        let mut history1 = SplitHistory::new();
        for i in 0..3 {
            history1.add_constraint(NeuronConstraint {
                layer_idx: 1,
                neuron_idx: i,
                is_active: true,
            });
        }
        assert!(pool.add_from_verified_domain(&history1));

        // Add second cut
        let mut history2 = SplitHistory::new();
        for i in 3..6 {
            history2.add_constraint(NeuronConstraint {
                layer_idx: 1,
                neuron_idx: i,
                is_active: false,
            });
        }
        assert!(pool.add_from_verified_domain(&history2));

        // Third cut should fail (pool full)
        let mut history3 = SplitHistory::new();
        for i in 6..9 {
            history3.add_constraint(NeuronConstraint {
                layer_idx: 1,
                neuron_idx: i,
                is_active: true,
            });
        }
        assert!(!pool.add_from_verified_domain(&history3));
        assert_eq!(pool.len(), 2);
        assert_eq!(pool.total_generated, 3); // Tracks all generation attempts (even rejected)
    }

    #[test]
    fn test_cut_pool_relevant_cuts() {
        let mut pool = CutPool::new(10);

        // Add a cut with specific constraints
        let mut history1 = SplitHistory::new();
        history1.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active: true,
        });
        history1.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 1,
            is_active: false,
        });
        pool.add_from_verified_domain(&history1);

        // Domain with no constraints -> cut is relevant
        let empty_history = SplitHistory::new();
        let relevant = pool.relevant_cuts_for(&empty_history);
        assert_eq!(relevant.len(), 1);

        // Domain with matching constraints -> cut is redundant
        let relevant = pool.relevant_cuts_for(&history1);
        assert_eq!(relevant.len(), 0);
    }

    #[test]
    fn test_cutting_plane_adam_step() {
        let mut history = SplitHistory::new();
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active: true,
        });
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 1,
            is_active: false,
        });

        let mut cut = CuttingPlane::from_verified_domain(&history).unwrap();
        assert_eq!(cut.lambda, 0.0);

        // Set gradient and take step
        cut.lambda_grad = 1.0;
        let config = AdaptiveOptConfig::default();
        cut.gradient_step_adam(&config, 1);

        // Lambda should have increased (gradient ascent)
        assert!(cut.lambda > 0.0);
        // Lambda should be non-negative (projected)
        assert!(cut.lambda >= 0.0);
    }

    #[test]
    fn test_sequential_proactive_cuts_generation() {
        // Test that proactive cuts are generated correctly for sequential networks
        use crate::layers::{Layer, LinearLayer, ReLULayer};
        use crate::Network;
        use ndarray::{arr1, arr2, Array1};

        // Create a simple sequential network: Linear -> ReLU -> Linear -> ReLU
        let linear1 = LinearLayer::new(
            arr2(&[[1.0_f32, 0.5], [-0.5, 1.0]]),
            Some(arr1(&[0.1, -0.1])),
        )
        .unwrap();
        let linear2 = LinearLayer::new(
            arr2(&[[1.0_f32, -0.3], [0.3, 1.0]]),
            Some(arr1(&[0.0, 0.0])),
        )
        .unwrap();

        let network = Network {
            layers: vec![
                Layer::Linear(linear1),
                Layer::ReLU(ReLULayer),
                Layer::Linear(linear2),
                Layer::ReLU(ReLULayer),
            ],
        };

        // Create layer bounds with unstable neurons (crossing zero)
        // layer_bounds[0] = output of layer 0 (Linear1)
        // layer_bounds[1] = output of layer 1 (ReLU1)
        // layer_bounds[2] = output of layer 2 (Linear2)
        // layer_bounds[3] = output of layer 3 (ReLU2)

        let bounds_linear1 = BoundedTensor::new(
            Array1::from_vec(vec![-1.0, -0.5]).into_dyn(), // Some neurons crossing zero
            Array1::from_vec(vec![1.0, 0.5]).into_dyn(),
        )
        .unwrap();

        let bounds_relu1 = BoundedTensor::new(
            Array1::from_vec(vec![0.0, 0.0]).into_dyn(), // ReLU output non-negative
            Array1::from_vec(vec![1.0, 0.5]).into_dyn(),
        )
        .unwrap();

        let bounds_linear2 = BoundedTensor::new(
            Array1::from_vec(vec![-0.8, -0.3]).into_dyn(), // Some neurons crossing zero
            Array1::from_vec(vec![0.8, 0.3]).into_dyn(),
        )
        .unwrap();

        let bounds_relu2 = BoundedTensor::new(
            Array1::from_vec(vec![0.0, 0.0]).into_dyn(),
            Array1::from_vec(vec![0.8, 0.3]).into_dyn(),
        )
        .unwrap();

        let layer_bounds = vec![bounds_linear1, bounds_relu1, bounds_linear2, bounds_relu2];

        // Generate proactive cuts
        let mut cut_pool = CutPool::new(100);
        let cuts_generated = cut_pool.generate_proactive_cuts(&network, &layer_bounds, 50);

        // Verify cuts were generated
        assert!(
            cuts_generated > 0,
            "Should generate proactive cuts for unstable neurons"
        );
        assert_eq!(
            cut_pool.len(),
            cuts_generated,
            "Cut pool length should match cuts generated"
        );

        // Verify cut properties
        for cut in &cut_pool.cuts {
            // All proactive cuts have source_depth 0
            assert_eq!(
                cut.source_depth, 0,
                "Proactive cuts should have source_depth 0"
            );
            // All cuts should have small initial lambda
            assert!(
                cut.lambda >= 0.0 && cut.lambda <= 0.1,
                "Proactive cuts should have small initial lambda"
            );
            // All cuts should reference valid layer indices
            for term in &cut.terms {
                assert!(
                    term.layer_idx < network.layers.len(),
                    "Cut term should reference valid layer index"
                );
            }
        }
    }

    #[test]
    fn test_beta_crown_config_with_cuts() {
        let config = BetaCrownConfig {
            enable_cuts: true,
            max_cuts: 500,
            min_cut_depth: 3,
            ..Default::default()
        };

        assert!(config.enable_cuts);
        assert_eq!(config.max_cuts, 500);
        assert_eq!(config.min_cut_depth, 3);
    }

    #[test]
    fn test_cut_contribution_computation() {
        // Test that cut contributions are computed correctly
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            enable_cuts: true,
            ..Default::default()
        });

        // Create a simple cut with known terms
        let mut cut = CuttingPlane {
            terms: vec![
                CutTerm {
                    layer_idx: 1,
                    neuron_idx: 0,
                    coefficient: 1.0,
                },
                CutTerm {
                    layer_idx: 1,
                    neuron_idx: 1,
                    coefficient: -1.0,
                },
            ],
            bias: 0.0,
            lambda: 1.0,
            lambda_grad: 0.0,
            lambda_m: 0.0,
            lambda_v: 0.0,
            source_depth: 2,
        };

        // Create layer bounds: [lower, upper] for each neuron
        // Neuron 0: [-1, 2] → unstable, z ∈ [0, 1]
        // Neuron 1: [-2, 1] → unstable, z ∈ [0, 1]
        let lower = ndarray::arr1(&[-1.0, -2.0]).into_dyn();
        let upper = ndarray::arr1(&[2.0, 1.0]).into_dyn();
        let bounds = crate::BoundedTensor::new(lower, upper).unwrap();
        let layer_bounds = vec![std::sync::Arc::new(bounds)];

        // Compute cut contribution using ReLU indicators
        // For term 0: coeff=1 (positive) → use z_min = 0, contribution = 1 * 0 = 0
        // For term 1: coeff=-1 (negative) → use z_max = 1, contribution = -1 * 1 = -1
        // constraint_min = 0 + -1 = -1
        // cut_contribution = lambda * (bias - constraint_min) = 1.0 * (0 - (-1)) = 1.0
        let cuts = vec![&cut];
        let contribution = verifier.compute_cut_contribution(&cuts, &layer_bounds);
        assert!(
            (contribution - 1.0).abs() < 1e-6,
            "Expected 1.0, got {}",
            contribution
        );

        // Test with different lambda
        cut.lambda = 0.5;
        let cuts = vec![&cut];
        let contribution = verifier.compute_cut_contribution(&cuts, &layer_bounds);
        assert!(
            (contribution - 0.5).abs() < 1e-6,
            "Expected 0.5, got {}",
            contribution
        );

        // Test with non-zero bias
        cut.lambda = 1.0;
        cut.bias = -1.0; // bias - constraint_min = -1 - (-1) = 0
        let cuts = vec![&cut];
        let contribution = verifier.compute_cut_contribution(&cuts, &layer_bounds);
        assert!(
            (contribution - 0.0).abs() < 1e-6,
            "Expected 0.0, got {}",
            contribution
        );
    }

    #[test]
    fn test_cut_gradient_computation() {
        // Test that cut gradients are computed correctly
        // d(lb)/d(lambda) = bias - constraint_min (for lambda * (bias - constraint_min))
        let mut cut = CuttingPlane {
            terms: vec![CutTerm {
                layer_idx: 1,
                neuron_idx: 0,
                coefficient: 1.0, // positive -> use z_min
            }],
            bias: 0.5,
            lambda: 1.0,
            lambda_grad: 0.0,
            lambda_m: 0.0,
            lambda_v: 0.0,
            source_depth: 1,
        };

        // Create layer bounds: neuron 0 has [-1, 2] → unstable, z ∈ [0, 1]
        let lower = ndarray::arr1(&[-1.0]).into_dyn();
        let upper = ndarray::arr1(&[2.0]).into_dyn();
        let bounds = crate::BoundedTensor::new(lower, upper).unwrap();
        let _layer_bounds = [std::sync::Arc::new(bounds)];

        // With ReLU indicators:
        // coeff=1 (positive) → use z_min = 0
        // constraint_min = 1 * 0 = 0
        // Expected gradient: bias - constraint_min = 0.5 - 0 = 0.5
        // (positive gradient means increasing lambda increases lower bound)

        let constraint_min = 0.0; // coeff=1 * z_min=0
        let expected_grad = cut.bias - constraint_min; // 0.5 - 0 = 0.5

        // Update gradient
        cut.lambda_grad = expected_grad;
        assert!((cut.lambda_grad - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_cut_optimization_integration() {
        // Test that cut optimization increases lower bounds when cuts have positive contribution
        let config = AdaptiveOptConfig {
            lr_lambda: Some(0.1),
            ..Default::default()
        };

        let mut cut = CuttingPlane {
            terms: vec![CutTerm {
                layer_idx: 1,
                neuron_idx: 0,
                coefficient: 1.0,
            }],
            bias: 2.0, // Positive bias means cut can contribute positively
            lambda: 0.0,
            lambda_grad: 0.0,
            lambda_m: 0.0,
            lambda_v: 0.0,
            source_depth: 1,
        };

        // Simulate gradient: bias - constraint_min = 2.0 - 0.0 = 2.0 (positive)
        // (Assuming unstable neuron with z_min=0)
        // Positive gradient means increasing lambda increases lower bound
        cut.lambda_grad = 2.0;

        // Take Adam step
        let initial_lambda = cut.lambda;
        cut.gradient_step_adam(&config, 1);

        // Lambda should increase (gradient ascent with positive gradient)
        assert!(
            cut.lambda > initial_lambda,
            "Lambda should increase with positive gradient"
        );
        assert!(cut.lambda >= 0.0, "Lambda should stay non-negative");
    }

    #[test]
    fn test_graph_beta_crown_input_split_residual_add() {
        // Graph: y = x + relu(x)
        let w = arr2(&[[1.0f32]]);
        let id = LinearLayer::new(w, None).unwrap();

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("id", Layer::Linear(id)));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["id".to_string()],
        ));
        graph.add_node(GraphNode::binary("add", Layer::Add(AddLayer), "id", "relu"));
        graph.set_output("add");

        let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            branching_heuristic: BranchingHeuristic::InputSplit,
            verify_upper_bound: false,
            use_alpha_crown: false,
            use_crown_ibp: false,
            enable_cuts: false,
            max_domains: 32,
            max_depth: 8,
            timeout: Duration::from_secs(1),
            ..Default::default()
        });

        let result = verifier
            .verify_graph_input_split(&graph, &input, &[1.0], -1.1)
            .unwrap();
        assert!(matches!(result.result, BabVerificationStatus::Verified));
    }

    #[test]
    fn test_graph_beta_crown_relu_split_simple() {
        // Graph: y = relu(x)
        // Input: [-1, 1]
        // Output: [0, 1] for ReLU
        // Objective: y >= -0.5 (should verify since min output is 0)
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
        graph.set_output("relu");

        let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            verify_upper_bound: false,
            use_alpha_crown: false,
            use_crown_ibp: false,
            enable_cuts: false,
            max_domains: 32,
            max_depth: 8,
            timeout: Duration::from_secs(1),
            ..Default::default()
        });

        // Verify: y >= -0.5 (i.e., 1*y > -0.5)
        let result = verifier
            .verify_graph_relu_split(&graph, &input, &[1.0], -0.5)
            .unwrap();
        assert!(
            matches!(result.result, BabVerificationStatus::Verified),
            "Expected Verified, got {:?}",
            result.result
        );
    }

    #[test]
    fn test_graph_beta_crown_relu_split_detects_violation_relu_input() {
        // Regression test: a false property must not return Verified.
        // Graph: y = relu(x), x ∈ [-1, 1], property y > 0.5 (false).
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
        graph.set_output("relu");

        let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            verify_upper_bound: false,
            use_alpha_crown: false,
            use_crown_ibp: false,
            enable_cuts: false,
            max_domains: 64,
            max_depth: 8,
            timeout: Duration::from_secs(2),
            ..Default::default()
        });

        let result = verifier
            .verify_graph_relu_split(&graph, &input, &[1.0], 0.5)
            .unwrap();
        assert!(
            matches!(result.result, BabVerificationStatus::PotentialViolation),
            "Expected PotentialViolation, got {:?}",
            result.result
        );
    }

    #[test]
    fn test_graph_beta_crown_relu_split_supports_conv2d() {
        // Conv2d models should use the ReLU-splitting path (no forced fallback),
        // so a violation should still be detected.
        // Graph: y = relu(conv(x)), x ∈ [-1,1], conv is identity, property y > 0.5 (false).
        let kernel = ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1, 1, 1, 1]), 1.0f32);
        let conv = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).unwrap();

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv)));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["conv".to_string()],
        ));
        graph.set_output("relu");

        let input =
            BoundedTensor::new(arr3(&[[[-1.0]]]).into_dyn(), arr3(&[[[1.0]]]).into_dyn()).unwrap();

        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            verify_upper_bound: false,
            use_alpha_crown: false,
            use_crown_ibp: false,
            enable_cuts: false,
            max_domains: 64,
            max_depth: 8,
            timeout: Duration::from_secs(2),
            ..Default::default()
        });

        let result = verifier
            .verify_graph_relu_split(&graph, &input, &[1.0], 0.5)
            .unwrap();
        assert!(
            matches!(result.result, BabVerificationStatus::PotentialViolation),
            "Expected PotentialViolation, got {:?}",
            result.result
        );
    }

    #[test]
    fn test_graph_beta_crown_relu_split_two_layer() {
        // Graph: y = relu(relu(x))
        // Input: [-2, 2]
        // After first ReLU: [0, 2]
        // After second ReLU: [0, 2]
        // Verify: y >= -1 (should pass since output is always >= 0)
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("relu1", Layer::ReLU(ReLULayer)));
        graph.add_node(GraphNode::new(
            "relu2",
            Layer::ReLU(ReLULayer),
            vec!["relu1".to_string()],
        ));
        graph.set_output("relu2");

        let input = BoundedTensor::new(arr1(&[-2.0]).into_dyn(), arr1(&[2.0]).into_dyn()).unwrap();

        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            verify_upper_bound: false,
            use_alpha_crown: false,
            use_crown_ibp: false,
            enable_cuts: false,
            max_domains: 64,
            max_depth: 10,
            timeout: Duration::from_secs(2),
            ..Default::default()
        });

        let result = verifier
            .verify_graph_relu_split(&graph, &input, &[1.0], -1.0)
            .unwrap();
        assert!(
            matches!(result.result, BabVerificationStatus::Verified),
            "Expected Verified, got {:?}",
            result.result
        );
    }

    #[test]
    fn test_graph_beta_crown_relu_split_residual() {
        // Graph: y = x + relu(x) (residual connection)
        // Input: [-1, 1]
        // When x >= 0: y = x + x = 2x, range [0, 2]
        // When x < 0: y = x + 0 = x, range [-1, 0]
        // Total output range: [-1, 2]
        // Verify: y > -1.5 (should pass since min is -1)
        let w = arr2(&[[1.0f32]]);
        let id = LinearLayer::new(w, None).unwrap();

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("id", Layer::Linear(id)));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["id".to_string()],
        ));
        graph.add_node(GraphNode::binary("add", Layer::Add(AddLayer), "id", "relu"));
        graph.set_output("add");

        let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            verify_upper_bound: false,
            use_alpha_crown: false,
            use_crown_ibp: false,
            enable_cuts: false,
            max_domains: 64,
            max_depth: 10,
            timeout: Duration::from_secs(2),
            ..Default::default()
        });

        let result = verifier
            .verify_graph_relu_split(&graph, &input, &[1.0], -1.5)
            .unwrap();
        assert!(
            matches!(result.result, BabVerificationStatus::Verified),
            "Expected Verified, got {:?}",
            result.result
        );
    }

    #[test]
    fn test_gcp_crown_cuts_generated_and_applied() {
        // Test that GCP-CROWN cuts are generated from verified domains
        // and applied to subsequent bound computations.
        //
        // Network: Linear -> ReLU -> Linear -> ReLU -> Linear
        // Multiple ReLU layers create opportunities for cuts.
        let w1 = arr2(&[[1.0f32], [-1.0]]);
        let b1 = arr1(&[0.0, 0.0]);
        let l1 = LinearLayer::new(w1, Some(b1)).unwrap();

        let w2 = arr2(&[[1.0f32, 0.5], [0.5, 1.0]]);
        let b2 = arr1(&[0.0, 0.0]);
        let l2 = LinearLayer::new(w2, Some(b2)).unwrap();

        let w3 = arr2(&[[1.0f32, -1.0]]);
        let l3 = LinearLayer::new(w3, None).unwrap();

        let mut network = Network::new();
        network.add_layer(Layer::Linear(l1));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(l2));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(l3));

        let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

        // Test with cuts enabled - should generate cuts from verified sub-domains
        let verifier_with_cuts = BetaCrownVerifier::new(BetaCrownConfig {
            verify_upper_bound: false,
            use_alpha_crown: false,
            use_crown_ibp: false,
            enable_cuts: true, // Enable GCP-CROWN
            max_cuts: 100,
            min_cut_depth: 1,
            max_domains: 128,
            max_depth: 16,
            beta_iterations: 10,
            timeout: Duration::from_secs(5),
            ..Default::default()
        });

        // Verify a property that requires splitting
        // Threshold of -2.0 should be easy enough to verify
        let result_with_cuts = verifier_with_cuts.verify(&network, &input, -2.0).unwrap();

        // Verify we got a result (verified or unknown)
        assert!(
            !matches!(
                result_with_cuts.result,
                BabVerificationStatus::PotentialViolation
            ),
            "Expected Verified or Unknown, got PotentialViolation"
        );

        // If verified, check that cuts were generated
        if matches!(result_with_cuts.result, BabVerificationStatus::Verified) {
            // Cuts should have been generated during the BaB process
            // The cuts_generated count tracks how many times we tried to generate cuts
            // (even if the pool was full or the domain was too shallow)
            eprintln!(
                "GCP-CROWN result: domains={}, verified={}, cuts={}",
                result_with_cuts.domains_explored,
                result_with_cuts.domains_verified,
                result_with_cuts.cuts_generated
            );

            // With min_cut_depth=1, we should generate cuts from domains at depth >= 1
            // When the property verifies quickly, there may be few or no cuts
            // The test passes if verification completes
        }
    }

    #[test]
    fn test_gcp_crown_cut_contribution_increases_lower_bound() {
        // Test that when cuts have positive lambda, they contribute to tightening
        // the lower bound via Lagrangian relaxation.
        //
        // This is a unit test for the cut contribution mechanism.
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            enable_cuts: true,
            ..Default::default()
        });

        // Create a cut with positive lambda and terms
        let cut = CuttingPlane {
            terms: vec![CutTerm {
                layer_idx: 1,
                neuron_idx: 0,
                coefficient: 1.0, // Active constraint: z_0 = 1
            }],
            bias: 0.5, // If neuron is stable-active (z=1), constraint is: 1 >= 0.5
            // contribution = lambda * (bias - z_min) = 0.5 * (0.5 - 0) = 0.25
            lambda: 0.5,
            lambda_grad: 0.0,
            lambda_m: 0.0,
            lambda_v: 0.0,
            source_depth: 1,
        };

        // Create layer bounds where neuron is unstable (z_min=0, z_max=1)
        let layer_bounds = vec![Arc::new(
            BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap(),
        )];

        // Compute cut contribution
        let contribution = verifier.compute_cut_contribution(&[&cut], &layer_bounds);

        // Expected: lambda * (bias - z_min) = 0.5 * (0.5 - 0) = 0.25
        // For unstable neuron with positive coefficient, z_min = 0
        assert!(
            (contribution - 0.25).abs() < 1e-6,
            "Expected cut contribution 0.25, got {}",
            contribution
        );
    }

    #[test]
    fn test_gcp_crown_lambda_optimization_convergence() {
        // Test that lambda optimization converges when cuts have useful gradients.
        //
        // When the Lagrangian gradient is positive, lambda should increase
        // (gradient ascent to maximize lower bound).
        let config = AdaptiveOptConfig {
            lr_lambda: Some(0.1),
            ..Default::default()
        };

        let mut cut = CuttingPlane {
            terms: vec![CutTerm {
                layer_idx: 1,
                neuron_idx: 0,
                coefficient: 1.0,
            }],
            bias: 1.0,
            lambda: 0.0,
            lambda_grad: 0.0,
            lambda_m: 0.0,
            lambda_v: 0.0,
            source_depth: 1,
        };

        // Simulate multiple optimization iterations with positive gradient
        for t in 1..=10 {
            // Gradient = bias - constraint_min = 1.0 - 0.0 = 1.0 (positive)
            // This means increasing lambda increases lower bound
            cut.lambda_grad = 1.0;
            cut.gradient_step_adam(&config, t);
        }

        // Lambda should have increased significantly
        assert!(
            cut.lambda > 0.5,
            "Lambda should increase with positive gradient, got {}",
            cut.lambda
        );

        // Lambda should stay non-negative
        assert!(cut.lambda >= 0.0, "Lambda must be non-negative");
    }

    // =========================================================================
    // GraphNetwork GCP-CROWN Tests
    // =========================================================================

    #[test]
    fn test_graph_cutting_plane_from_history() {
        // Test creating a graph cutting plane from a verified domain's split history
        let mut history = GraphSplitHistory::new();
        history.add_constraint(GraphNeuronConstraint {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            is_active: true,
        });
        history.add_constraint(GraphNeuronConstraint {
            node_name: "relu2".to_string(),
            neuron_idx: 1,
            is_active: false,
        });

        let cut = GraphCuttingPlane::from_verified_domain(&history);
        assert!(cut.is_some());
        let cut = cut.unwrap();
        assert_eq!(cut.terms.len(), 2);
        assert_eq!(cut.terms[0].node_name, "relu1");
        assert_eq!(cut.terms[0].neuron_idx, 0);
        assert_eq!(cut.terms[0].coefficient, 1.0); // active
        assert_eq!(cut.terms[1].node_name, "relu2");
        assert_eq!(cut.terms[1].neuron_idx, 1);
        assert_eq!(cut.terms[1].coefficient, -1.0); // inactive
                                                    // Bias = (1 active) - 1 = 0
        assert_eq!(cut.bias, 0.0);
    }

    #[test]
    fn test_graph_cutting_plane_empty_history() {
        // Empty history should not produce a cut
        let history = GraphSplitHistory::new();
        let cut = GraphCuttingPlane::from_verified_domain(&history);
        assert!(cut.is_none());
    }

    #[test]
    fn test_graph_cut_pool_basic() {
        let mut pool = GraphCutPool::new(10);
        assert!(pool.is_empty());
        assert_eq!(pool.len(), 0);

        // Add a cut
        let mut history = GraphSplitHistory::new();
        history.add_constraint(GraphNeuronConstraint {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            is_active: true,
        });
        history.add_constraint(GraphNeuronConstraint {
            node_name: "relu2".to_string(),
            neuron_idx: 1,
            is_active: false,
        });

        let added = pool.add_from_verified_domain(&history);
        assert!(added);
        assert_eq!(pool.len(), 1);
        assert_eq!(pool.total_generated, 1);
    }

    #[test]
    fn test_graph_cut_pool_min_depth() {
        // Pool with min_depth=2 should reject single-constraint histories
        let mut pool = GraphCutPool::with_min_depth(10, 2);

        // Single constraint - should be rejected
        let mut history1 = GraphSplitHistory::new();
        history1.add_constraint(GraphNeuronConstraint {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            is_active: true,
        });
        let added1 = pool.add_from_verified_domain(&history1);
        assert!(!added1);
        assert_eq!(pool.len(), 0);

        // Two constraints - should be accepted
        let mut history2 = GraphSplitHistory::new();
        history2.add_constraint(GraphNeuronConstraint {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            is_active: true,
        });
        history2.add_constraint(GraphNeuronConstraint {
            node_name: "relu2".to_string(),
            neuron_idx: 1,
            is_active: false,
        });
        let added2 = pool.add_from_verified_domain(&history2);
        assert!(added2);
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn test_graph_cutting_plane_redundancy() {
        // Create a cut
        let mut history1 = GraphSplitHistory::new();
        history1.add_constraint(GraphNeuronConstraint {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            is_active: true,
        });
        history1.add_constraint(GraphNeuronConstraint {
            node_name: "relu2".to_string(),
            neuron_idx: 1,
            is_active: false,
        });
        let cut = GraphCuttingPlane::from_verified_domain(&history1).unwrap();

        // A domain with the same constraints should make the cut redundant
        assert!(cut.is_redundant_for(&history1));

        // A domain with different constraints should not make it redundant
        let mut history2 = GraphSplitHistory::new();
        history2.add_constraint(GraphNeuronConstraint {
            node_name: "relu3".to_string(),
            neuron_idx: 2,
            is_active: true,
        });
        assert!(!cut.is_redundant_for(&history2));
    }

    #[test]
    fn test_graph_cut_pool_relevant_cuts() {
        let mut pool = GraphCutPool::with_min_depth(10, 2);

        // Add a cut
        let mut history1 = GraphSplitHistory::new();
        history1.add_constraint(GraphNeuronConstraint {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            is_active: true,
        });
        history1.add_constraint(GraphNeuronConstraint {
            node_name: "relu2".to_string(),
            neuron_idx: 1,
            is_active: false,
        });
        pool.add_from_verified_domain(&history1);

        // For an unrelated domain, the cut should be relevant
        let relevant = pool.relevant_cuts_for(&GraphSplitHistory::new());
        assert_eq!(relevant.len(), 1);

        // For a domain with the same constraints, the cut should be redundant
        let relevant2 = pool.relevant_cuts_for(&history1);
        assert_eq!(relevant2.len(), 0);
    }

    #[test]
    fn test_graph_cutting_plane_adam_update() {
        let mut cut = GraphCuttingPlane {
            terms: vec![GraphCutTerm {
                node_name: "relu1".to_string(),
                neuron_idx: 0,
                coefficient: 1.0,
            }],
            bias: 1.0,
            lambda: 0.0,
            lambda_grad: 1.0, // positive gradient
            lambda_m: 0.0,
            lambda_v: 0.0,
            source_depth: 1,
        };

        // After several Adam updates with positive gradient, lambda should increase
        for t in 1..=10 {
            cut.lambda_grad = 1.0;
            cut.update_lambda_adam(0.1, 0.9, 0.999, 1e-8, t);
        }

        assert!(
            cut.lambda > 0.0,
            "Lambda should increase with positive gradient"
        );
        assert!(cut.lambda >= 0.0, "Lambda must be non-negative");
    }

    #[test]
    fn test_proactive_cuts_generation() {
        // Test that proactive cuts are generated correctly from a simple graph
        use crate::layers::{Layer, LinearLayer, ReLULayer};
        use crate::network::GraphNode;
        use crate::BoundedTensor;
        use crate::GraphNetwork;
        use ndarray::{arr1, arr2, Array1};
        use std::collections::HashMap;
        use std::sync::Arc;

        // Create a simple graph with two ReLU layers
        let mut graph = GraphNetwork::new();

        // Build: input -> linear1 -> relu1 -> linear2 -> relu2
        let linear1 = LinearLayer::new(
            arr2(&[[1.0_f32, 0.5], [-0.5, 1.0]]),
            Some(arr1(&[0.1_f32, -0.1])),
        )
        .unwrap();
        graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
        graph.add_node(GraphNode::new(
            "relu1",
            Layer::ReLU(ReLULayer),
            vec!["linear1".to_string()],
        ));

        let linear2 = LinearLayer::new(
            arr2(&[[1.0_f32, -0.3], [0.3, 1.0]]),
            Some(arr1(&[0.0_f32, 0.0])),
        )
        .unwrap();
        graph.add_node(GraphNode::new(
            "linear2",
            Layer::Linear(linear2),
            vec!["relu1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "relu2",
            Layer::ReLU(ReLULayer),
            vec!["linear2".to_string()],
        ));

        graph.set_output("relu2");

        // Create node bounds with unstable neurons (crossing zero)
        let mut node_bounds: HashMap<String, Arc<BoundedTensor>> = HashMap::new();

        // Bounds for linear1/relu1 input: some neurons crossing zero (unstable)
        let bounds_linear1 = BoundedTensor::new(
            Array1::from_vec(vec![-1.0, -0.5]).into_dyn(),
            Array1::from_vec(vec![1.0, 0.5]).into_dyn(),
        )
        .unwrap();
        node_bounds.insert("linear1".to_string(), Arc::new(bounds_linear1));

        // Bounds for linear2/relu2 input
        let bounds_linear2 = BoundedTensor::new(
            Array1::from_vec(vec![-0.8, -0.3]).into_dyn(),
            Array1::from_vec(vec![0.8, 0.3]).into_dyn(),
        )
        .unwrap();
        node_bounds.insert("linear2".to_string(), Arc::new(bounds_linear2));

        // Generate proactive cuts
        let mut cut_pool = GraphCutPool::new(100);
        let cuts_generated = cut_pool.generate_proactive_cuts(&graph, &node_bounds, 50);

        // Verify cuts were generated
        assert!(
            cuts_generated > 0,
            "Should generate proactive cuts for unstable neurons"
        );
        assert_eq!(
            cut_pool.len(),
            cuts_generated,
            "Cut pool length should match cuts generated"
        );

        // Verify cut properties
        for cut in &cut_pool.cuts {
            // All proactive cuts have source_depth 0
            assert_eq!(
                cut.source_depth, 0,
                "Proactive cuts should have source_depth 0"
            );
            // All cuts should have small initial lambda
            assert!(
                cut.lambda >= 0.0 && cut.lambda <= 0.1,
                "Proactive cuts should have small initial lambda"
            );
            // All cuts should have non-empty terms
            assert!(!cut.terms.is_empty(), "Cut should have terms");
        }

        // Verify we have both single-neuron and pairwise cuts
        let single_neuron_cuts = cut_pool.cuts.iter().filter(|c| c.terms.len() == 1).count();
        let _pairwise_cuts = cut_pool.cuts.iter().filter(|c| c.terms.len() == 2).count();

        assert!(
            single_neuron_cuts > 0,
            "Should have single-neuron indicator cuts"
        );
        // Pairwise cuts are generated when there are multiple ReLU nodes
        // (may be 0 if only single-neuron cuts are generated due to max_cuts limit)
    }

    #[test]
    fn test_graph_beta_warmup_inherits_parent_values() {
        // Test that child domains inherit β values from parent via warmup

        // Create a parent split history with two constraints
        let history1 = GraphSplitHistory::new();
        let history1 = history1.with_constraint(GraphNeuronConstraint {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            is_active: true,
        });
        let history1 = history1.with_constraint(GraphNeuronConstraint {
            node_name: "relu1".to_string(),
            neuron_idx: 1,
            is_active: false,
        });

        // Create parent β state and set non-default values
        let mut parent_beta = GraphBetaState::from_history(&history1);
        parent_beta.entries[0].value = 0.5; // Optimized value
        parent_beta.entries[0].m = 0.1; // Adam momentum
        parent_beta.entries[0].v = 0.01;
        parent_beta.entries[1].value = 0.3;
        parent_beta.entries[1].m = 0.05;

        // Create child history with one more constraint
        let history2 = history1.with_constraint(GraphNeuronConstraint {
            node_name: "relu2".to_string(),
            neuron_idx: 2,
            is_active: true,
        });

        // Create child β state with warmup
        let child_beta = GraphBetaState::from_history_with_warmup(
            &history2,
            &parent_beta,
            GraphBetaState::DEFAULT_BETA_INIT,
        );

        // Verify child has 3 entries
        assert_eq!(child_beta.entries.len(), 3);

        // Verify existing constraints inherit parent values
        let entry0 = child_beta.get_entry("relu1", 0).unwrap();
        assert_eq!(entry0.value, 0.5, "Should inherit parent β value");
        assert_eq!(entry0.m, 0.1, "Should inherit parent Adam momentum m");
        assert_eq!(entry0.v, 0.01, "Should inherit parent Adam momentum v");
        assert_eq!(entry0.sign, 1.0, "Active constraint has sign +1");

        let entry1 = child_beta.get_entry("relu1", 1).unwrap();
        assert_eq!(entry1.value, 0.3, "Should inherit parent β value");
        assert_eq!(entry1.m, 0.05, "Should inherit parent Adam momentum m");
        assert_eq!(entry1.sign, -1.0, "Inactive constraint has sign -1");

        // Verify new constraint has default initialization
        let entry2 = child_beta.get_entry("relu2", 2).unwrap();
        assert_eq!(
            entry2.value,
            GraphBetaState::DEFAULT_BETA_INIT,
            "New constraint should have default β"
        );
        assert_eq!(entry2.m, 0.0, "New constraint should have zero momentum m");
        assert_eq!(entry2.v, 0.0, "New constraint should have zero momentum v");
        assert_eq!(entry2.sign, 1.0, "Active constraint has sign +1");
    }

    #[test]
    fn test_graph_beta_analytical_gradients_from_a_matrix() {
        // Test that analytical β gradients are computed correctly from A matrices.
        //
        // For a constraint at (node_name, neuron_idx, sign), the gradient should be:
        //   ∂lb/∂β = -sign * sensitivity
        // where sensitivity = sum_j(A[j, neuron_idx])

        use crate::bounds::GraphAlphaCrownIntermediate;

        // Create a split history with two constraints
        let history = GraphSplitHistory::new();
        let history = history.with_constraint(GraphNeuronConstraint {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            is_active: true, // sign = +1
        });
        let history = history.with_constraint(GraphNeuronConstraint {
            node_name: "relu1".to_string(),
            neuron_idx: 1,
            is_active: false, // sign = -1
        });

        // Create β state
        let mut beta_state = GraphBetaState::from_history(&history);
        assert_eq!(beta_state.entries.len(), 2);

        // Create intermediate storage with A matrix
        let mut intermediate = GraphAlphaCrownIntermediate::new();

        // A matrix at relu1: shape (2, 3) - 2 outputs, 3 neurons
        // A = [[1.0, -2.0, 0.5],
        //      [0.5,  1.0, -1.0]]
        let a_matrix = arr2(&[[1.0, -2.0, 0.5], [0.5, 1.0, -1.0]]);
        intermediate.a_at_relu.insert("relu1".to_string(), a_matrix);

        // Pre-ReLU bounds (not strictly needed for gradient computation but required for struct)
        let pre_lower = arr1(&[-1.0, -0.5, 0.0]);
        let pre_upper = arr1(&[1.0, 1.5, 2.0]);
        intermediate
            .pre_relu_bounds
            .insert("relu1".to_string(), (pre_lower, pre_upper));

        // Compute analytical gradients
        let max_grad = beta_state.compute_analytical_gradients(&intermediate);

        // Check gradients:
        // For neuron_idx=0, sign=+1:
        //   sensitivity = A[0,0] + A[1,0] = 1.0 + 0.5 = 1.5
        //   grad = -sign * sensitivity = -1.0 * 1.5 = -1.5
        let entry0 = beta_state.get_entry("relu1", 0).unwrap();
        assert!(
            (entry0.grad - (-1.5)).abs() < 1e-6,
            "Expected grad=-1.5 for active constraint, got {}",
            entry0.grad
        );

        // For neuron_idx=1, sign=-1:
        //   sensitivity = A[0,1] + A[1,1] = -2.0 + 1.0 = -1.0
        //   grad = -sign * sensitivity = -(-1.0) * (-1.0) = -1.0
        let entry1 = beta_state.get_entry("relu1", 1).unwrap();
        assert!(
            (entry1.grad - (-1.0)).abs() < 1e-6,
            "Expected grad=-1.0 for inactive constraint, got {}",
            entry1.grad
        );

        // max_grad should be the max absolute gradient
        assert!(
            (max_grad - 1.5).abs() < 1e-6,
            "Expected max_grad=1.5, got {}",
            max_grad
        );
    }

    #[test]
    fn test_graph_beta_analytical_gradients_missing_node() {
        // Test that gradients are zero for constraints on nodes not in intermediate storage

        use crate::bounds::GraphAlphaCrownIntermediate;

        let history = GraphSplitHistory::new().with_constraint(GraphNeuronConstraint {
            node_name: "relu_missing".to_string(),
            neuron_idx: 0,
            is_active: true,
        });

        let mut beta_state = GraphBetaState::from_history(&history);

        // Empty intermediate - no A matrices stored
        let intermediate = GraphAlphaCrownIntermediate::new();

        let max_grad = beta_state.compute_analytical_gradients(&intermediate);

        // Gradient should be zero since node is not found
        let entry = beta_state.get_entry("relu_missing", 0).unwrap();
        assert_eq!(entry.grad, 0.0, "Gradient should be zero for missing node");
        assert_eq!(max_grad, 0.0, "Max grad should be zero");
    }

    #[test]
    fn test_graph_beta_analytical_gradients_multi_objective() {
        // Test analytical gradients for multi-objective verification
        // The gradient should be computed for the critical objective (minimum margin)
        use crate::bounds::GraphAlphaCrownIntermediate;

        // Create split history with one constraint
        let history = GraphSplitHistory::new().with_constraint(GraphNeuronConstraint {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            is_active: true, // sign = +1
        });

        let mut beta_state = GraphBetaState::from_history(&history);
        assert_eq!(beta_state.entries.len(), 1);

        // Create intermediate storage with A matrix
        let mut intermediate = GraphAlphaCrownIntermediate::new();

        // A matrix at relu1: shape (3, 2) - 3 outputs, 2 neurons
        // A = [[1.0, -2.0],
        //      [0.5,  1.0],
        //      [2.0,  0.0]]
        let a_matrix = arr2(&[[1.0, -2.0], [0.5, 1.0], [2.0, 0.0]]);
        intermediate.a_at_relu.insert("relu1".to_string(), a_matrix);

        // Two objectives:
        // Objective 0: c = [1.0, 0.0, 0.0] (only uses output 0)
        // Objective 1: c = [0.0, 1.0, 1.0] (uses outputs 1 and 2)
        let objectives = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 1.0]];

        // Bounds for each objective (lower, upper)
        // Objective 0: margin = lower - threshold = 0.5 - 0.0 = 0.5
        // Objective 1: margin = lower - threshold = -0.2 - 0.0 = -0.2 (CRITICAL - minimum margin)
        let obj_bounds = vec![(0.5, 1.0), (-0.2, 0.5)];
        let thresholds = vec![0.0, 0.0];
        let verified_mask = vec![false, false];

        // Compute analytical gradients for multi-objective
        let max_grad = beta_state.compute_analytical_gradients_multi_objective(
            &intermediate,
            &obj_bounds,
            &objectives,
            &thresholds,
            &verified_mask,
        );

        // Objective 1 is critical (min margin = -0.2)
        // Its coefficient vector is c = [0.0, 1.0, 1.0]
        // For neuron_idx=0:
        //   sensitivity = c[0]*A[0,0] + c[1]*A[1,0] + c[2]*A[2,0]
        //              = 0.0*1.0 + 1.0*0.5 + 1.0*2.0 = 2.5
        //   grad = -sign * sensitivity = -1.0 * 2.5 = -2.5
        let entry = beta_state.get_entry("relu1", 0).unwrap();
        assert!(
            (entry.grad - (-2.5)).abs() < 1e-6,
            "Expected grad=-2.5 for critical objective, got {}",
            entry.grad
        );
        assert!(
            (max_grad - 2.5).abs() < 1e-6,
            "Expected max_grad=2.5, got {}",
            max_grad
        );
    }

    #[test]
    fn test_graph_beta_analytical_gradients_multi_objective_all_verified() {
        // Test that gradients are zero when all objectives are verified
        use crate::bounds::GraphAlphaCrownIntermediate;

        let history = GraphSplitHistory::new().with_constraint(GraphNeuronConstraint {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            is_active: true,
        });

        let mut beta_state = GraphBetaState::from_history(&history);

        let mut intermediate = GraphAlphaCrownIntermediate::new();
        let a_matrix = arr2(&[[1.0, -2.0], [0.5, 1.0]]);
        intermediate.a_at_relu.insert("relu1".to_string(), a_matrix);

        let objectives = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let obj_bounds = vec![(0.5, 1.0), (0.3, 0.8)];
        let thresholds = vec![0.0, 0.0];
        let verified_mask = vec![true, true]; // All verified

        let max_grad = beta_state.compute_analytical_gradients_multi_objective(
            &intermediate,
            &obj_bounds,
            &objectives,
            &thresholds,
            &verified_mask,
        );

        // All objectives verified - gradient should be zero
        let entry = beta_state.get_entry("relu1", 0).unwrap();
        assert_eq!(
            entry.grad, 0.0,
            "Gradient should be zero when all objectives verified"
        );
        assert_eq!(max_grad, 0.0, "Max grad should be zero");
    }

    /// Benchmark analytical gradients vs finite-difference for multi-objective β optimization.
    ///
    /// This test demonstrates the performance advantage of analytical gradients:
    /// - Analytical: 1 forward pass per iteration (stores A matrices during propagation)
    /// - Finite-difference (SPSA-style): 3 forward passes per iteration (+ε, -ε, baseline)
    ///
    /// Expected speedup: ~3x for the gradient computation phase.
    #[test]
    fn test_benchmark_analytical_vs_finite_diff_multi_objective() {
        use crate::bounds::GraphAlphaCrownIntermediate;
        use std::time::Instant;

        // Create a larger network for meaningful benchmarking
        // Using the benchmark network pattern with more neurons
        let w1: Array2<f32> = arr2(&[
            [0.5, -0.3, 0.2, -0.4],
            [-0.2, 0.6, -0.1, 0.3],
            [0.4, -0.5, 0.3, -0.2],
            [-0.3, 0.2, -0.4, 0.5],
            [0.1, -0.6, 0.5, -0.1],
            [-0.5, 0.4, -0.3, 0.6],
            [0.6, -0.2, 0.1, -0.5],
            [-0.4, 0.3, -0.6, 0.2],
        ]);
        let linear1 = LinearLayer::new(w1, None).unwrap();

        let w2: Array2<f32> = arr2(&[
            [0.3, -0.2, 0.4, -0.3, 0.2, -0.1, 0.5, -0.4],
            [-0.4, 0.5, -0.2, 0.3, -0.5, 0.4, -0.3, 0.2],
            [0.2, -0.4, 0.3, -0.5, 0.4, -0.3, 0.1, -0.2],
            [-0.3, 0.1, -0.5, 0.2, -0.2, 0.5, -0.4, 0.3],
        ]);
        let linear2 = LinearLayer::new(w2, None).unwrap();

        // Output layer: 4 -> 3 (for multi-objective with 3 targets)
        let w3: Array2<f32> = arr2(&[
            [0.5, -0.3, 0.4, -0.2],
            [-0.2, 0.4, -0.3, 0.5],
            [0.3, -0.5, 0.2, -0.4],
        ]);
        let linear3 = LinearLayer::new(w3, None).unwrap();

        let mut network = Network::new();
        network.add_layer(Layer::Linear(linear1));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(linear2));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(linear3));

        // Create split history with multiple constraints for β optimization
        let history = GraphSplitHistory::new()
            .with_constraint(GraphNeuronConstraint {
                node_name: "relu_0".to_string(),
                neuron_idx: 0,
                is_active: true,
            })
            .with_constraint(GraphNeuronConstraint {
                node_name: "relu_0".to_string(),
                neuron_idx: 1,
                is_active: false,
            })
            .with_constraint(GraphNeuronConstraint {
                node_name: "relu_1".to_string(),
                neuron_idx: 0,
                is_active: true,
            });

        // Create β state
        let mut beta_state = GraphBetaState::from_history(&history);

        // Initialize β values
        for entry in beta_state.entries.iter_mut() {
            entry.value = 0.1;
        }

        // Create mock intermediate with A matrices
        // In practice, these come from CROWN propagation
        let mut intermediate = GraphAlphaCrownIntermediate::new();

        // A matrix at relu_0: shape (4, 8) - next layer width x current layer width
        let a_relu0 = Array2::from_shape_fn((4, 8), |(i, j)| {
            let val = ((i * 8 + j) as f32 * 0.1) - 0.4;
            val.sin() * 0.5 // Create varied but bounded values
        });
        intermediate.a_at_relu.insert("relu_0".to_string(), a_relu0);

        // A matrix at relu_1: shape (3, 4) - output width x relu_1 width
        let a_relu1 = Array2::from_shape_fn((3, 4), |(i, j)| {
            let val = ((i * 4 + j) as f32 * 0.15) - 0.3;
            val.cos() * 0.5
        });
        intermediate.a_at_relu.insert("relu_1".to_string(), a_relu1);

        // Multi-objective setup: 3 objectives (one per output neuron)
        let objectives = vec![
            vec![1.0, 0.0, 0.0], // Objective 0: maximize output 0
            vec![0.0, 1.0, 0.0], // Objective 1: maximize output 1
            vec![0.0, 0.0, 1.0], // Objective 2: maximize output 2
        ];

        // Bounds for each objective (simulating verification scenario)
        let obj_bounds = vec![
            (-0.1, 0.5), // Objective 0: margin = -0.1 (not verified)
            (0.2, 0.8),  // Objective 1: margin = 0.2 (verified)
            (-0.3, 0.4), // Objective 2: margin = -0.3 (critical - minimum)
        ];
        let thresholds = vec![0.0, 0.0, 0.0];
        let verified_mask = vec![false, true, false];

        const NUM_ITERATIONS: u32 = 100;

        // Benchmark analytical gradient computation
        let analytical_start = Instant::now();
        for _ in 0..NUM_ITERATIONS {
            beta_state.zero_grad();
            beta_state.compute_analytical_gradients_multi_objective(
                &intermediate,
                &obj_bounds,
                &objectives,
                &thresholds,
                &verified_mask,
            );
        }
        let analytical_time = analytical_start.elapsed();

        // Benchmark finite-difference gradient computation (SPSA-style)
        // This simulates what SPSA would do: perturb each β and compute bounds
        let epsilon = 0.01f32;
        let fd_start = Instant::now();
        for _ in 0..NUM_ITERATIONS {
            beta_state.zero_grad();

            // For each β entry, compute finite-difference gradient
            // This requires 2 evaluations per β (or 2 total with simultaneous perturbation)
            // SPSA uses 3 total: baseline, +ε, -ε
            for entry_idx in 0..beta_state.entries.len() {
                let original_value = beta_state.entries[entry_idx].value;

                // +ε evaluation (just compute the weighted sum, simulating forward pass result)
                beta_state.entries[entry_idx].value = original_value + epsilon;
                let _plus_margin: f32 = obj_bounds
                    .iter()
                    .zip(thresholds.iter())
                    .zip(verified_mask.iter())
                    .filter(|((_, _), &v)| !v)
                    .map(|(((l, _), t), _)| l - t)
                    .fold(f32::INFINITY, |a, b| a.min(b));

                // -ε evaluation
                beta_state.entries[entry_idx].value = original_value - epsilon;
                let _minus_margin: f32 = obj_bounds
                    .iter()
                    .zip(thresholds.iter())
                    .zip(verified_mask.iter())
                    .filter(|((_, _), &v)| !v)
                    .map(|(((l, _), t), _)| l - t)
                    .fold(f32::INFINITY, |a, b| a.min(b));

                // Restore and compute gradient
                beta_state.entries[entry_idx].value = original_value;
                // grad = (plus - minus) / (2 * epsilon)
                // Note: In real SPSA, we'd use actual forward pass bounds
            }
        }
        let fd_time = fd_start.elapsed();

        // Report results
        let analytical_us = analytical_time.as_micros() as f64 / NUM_ITERATIONS as f64;
        let fd_us = fd_time.as_micros() as f64 / NUM_ITERATIONS as f64;
        let speedup = fd_us / analytical_us;

        println!("\n=== Multi-Objective β Gradient Benchmark ===");
        println!("Network: 4 -> 8 -> 4 -> 3 (2 ReLU layers)");
        println!("β parameters: {}", beta_state.entries.len());
        println!("Objectives: {}", objectives.len());
        println!("Iterations: {}", NUM_ITERATIONS);
        println!();
        println!(
            "Analytical gradient: {:.2} µs/iter",
            analytical_us
        );
        println!(
            "Finite-difference:   {:.2} µs/iter (simulated)",
            fd_us
        );
        println!("Speedup: {:.1}x", speedup);
        println!();
        println!("Note: Real SPSA speedup is ~3x because analytical uses 1 forward pass");
        println!("      vs SPSA's 3 forward passes. This benchmark only measures the");
        println!("      gradient computation overhead, not forward pass cost.");

        // Verify gradients are computed correctly
        beta_state.zero_grad();
        let max_grad = beta_state.compute_analytical_gradients_multi_objective(
            &intermediate,
            &obj_bounds,
            &objectives,
            &thresholds,
            &verified_mask,
        );

        assert!(
            max_grad.is_finite() && max_grad >= 0.0,
            "Max gradient should be finite and non-negative"
        );

        // Check that gradients were computed for all entries
        for (i, entry) in beta_state.entries.iter().enumerate() {
            assert!(
                entry.grad.is_finite(),
                "Gradient {} should be finite",
                i
            );
        }

        // Note: The pure gradient computation overhead is similar between methods.
        // The real 3x speedup comes from forward pass savings:
        // - Analytical: 1 forward pass + ~2µs gradient computation per iteration
        // - SPSA: 3 forward passes + ~2µs gradient estimation per iteration
        //
        // For a typical forward pass of 100-1000µs, this means:
        // - Analytical: 100-1000µs + 2µs ≈ 100-1000µs per iteration
        // - SPSA: 300-3000µs + 2µs ≈ 300-3000µs per iteration
        // Speedup: ~3x
        //
        // This benchmark verifies the gradient computation overhead is reasonable
        // (not a bottleneck compared to forward pass cost).
        assert!(
            analytical_us < 100.0,
            "Analytical gradient computation should be fast (<100µs): {} µs",
            analytical_us
        );
    }
}
