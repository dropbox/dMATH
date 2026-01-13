//! Bound propagation algorithms for neural network verification.
//!
//! Implements multiple propagation strategies with increasing precision:
//! - IBP (Interval Bound Propagation): Fastest, loosest bounds
//! - CROWN: Linear relaxation, tighter bounds
//! - α-CROWN: Optimized CROWN with learnable parameters
//! - β-CROWN: Branch and bound for complete verification
//!
//! # Parallel Verification
//!
//! For sequence models, use [`parallel::ParallelVerifier`] to parallelize
//! verification across positions for near-linear speedup with cores.

pub mod beta_crown;
pub mod bounds;
pub mod domain_clip;
pub mod layers;
pub mod network;
pub mod parallel;
pub mod pgd_attack;
pub mod pooled;
pub mod sdp_crown;
pub mod streaming;
pub mod types;
pub mod verifier;

pub use beta_crown::{
    AdaptiveOptConfig, BabDomain, BabVerificationStatus, BetaCrownConfig, BetaCrownResult,
    BetaCrownVerifier, BranchingHeuristic, CutPool, CutTerm, CuttingPlane, GraphPrecomputedBounds,
    LRScheduler, NeuronConstraint, SplitHistory,
};

pub use domain_clip::{
    ClipStrategy, ClipperSummary, DomainClipConfig, DomainClipper, LayerStatistics,
};

// Re-export types for backward compatibility
pub use types::{
    compute_model_hash, truncate_name, BlockBoundsInfo, BlockProgress, BlockWiseResult,
    LayerByLayerResult, LayerProgress, NodeBoundsInfo, PropagationConfig, PropagationMethod,
    VerificationCheckpoint,
};

// Re-export layers for backward compatibility
pub use layers::{
    adaptive_gelu_linear_relaxation, conv1d_single, conv1d_transpose, conv2d_single,
    conv2d_transpose, gelu_eval, gelu_linear_relaxation, AbsLayer, AddConstantLayer, AddLayer,
    AveragePoolLayer, BatchNormLayer, BoundPropagation, CausalSoftmaxLayer, CeilLayer, CeluLayer,
    ClipLayer, ConcatLayer, Conv1dLayer, Conv2dLayer, CosLayer, DivConstantLayer, DivLayer,
    EluLayer, ExpLayer, FlattenLayer, FloorLayer, GELULayer, GeluApproximation, HardSigmoidLayer,
    HardSwishLayer, Layer, LayerNormLayer, LeakyReLULayer, LinearLayer, LogLayer, LogSoftmaxLayer,
    MatMulLayer, MaxPool2dLayer, MishLayer, MulBinaryLayer, MulConstantLayer, NonZeroLayer,
    PReluLayer, PowConstantLayer, ReLULayer, ReciprocalLayer, ReduceMeanLayer, ReduceSumLayer,
    RelaxationMode, ReshapeLayer, RoundLayer, SeluLayer, ShrinkLayer, SigmoidLayer, SignLayer,
    SinLayer, SliceLayer, SoftmaxLayer, SoftplusLayer, SoftsignLayer, SqrtLayer, SubConstantLayer,
    SubLayer, TanhLayer, ThresholdedReluLayer, TileLayer, TransposeLayer, WhereLayer,
};

// Re-export bounds for backward compatibility
pub use bounds::{
    batched_matvec, safe_add_for_bounds, safe_add_for_bounds_with_polarity, safe_array_add,
    safe_mul_for_bounds, AlphaCrownConfig, AlphaCrownIntermediate, AlphaState, BatchedLinearBounds,
    GradientMethod, GraphAlphaState, LinearBounds, Optimizer,
};

// Re-export network types for backward compatibility
pub use network::{
    broadcast_shapes, relu_crown_relaxation, relu_ibp, AttentionGraphBuilder, GraphNetwork,
    GraphNode, Network,
};

// Re-export verifier for backward compatibility
pub use verifier::Verifier;

// Re-export parallel verification
pub use parallel::{
    verify_parallel, verify_parallel_with_method, ParallelConfig, ParallelVerificationResult,
    ParallelVerifier,
};

// Re-export streaming computation
pub use streaming::{
    estimate_memory_savings, CheckpointedBounds, StreamingConfig, StreamingVerifier,
};

// Re-export PGD attack
pub use pgd_attack::{PgdAttacker, PgdConfig, PgdResult};

// Re-export gamma_tensor and gamma_core types for tests and downstream use
pub use gamma_core::{Bound, GammaError, Result, VerificationResult, VerificationSpec};
pub use gamma_tensor::BoundedTensor;

#[cfg(test)]
mod tests;
