//! Core types and traits for γ-CROWN neural network verification.
//!
//! This crate provides the foundational abstractions for bound propagation
//! in neural networks, enabling formal verification of properties like
//! robustness and equivalence.

use serde::{Deserialize, Serialize};
use std::ops::RangeInclusive;

/// A bound on a scalar value: [lower, upper].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Bound {
    pub lower: f32,
    pub upper: f32,
}

impl Bound {
    /// Create a new bound.
    #[inline]
    pub fn new(lower: f32, upper: f32) -> Self {
        debug_assert!(lower <= upper, "Invalid bound: {lower} > {upper}");
        Self { lower, upper }
    }

    /// Create a concrete (point) bound.
    #[inline]
    pub fn concrete(value: f32) -> Self {
        Self {
            lower: value,
            upper: value,
        }
    }

    /// Check if this bound contains a value.
    #[inline]
    pub fn contains(&self, value: f32) -> bool {
        self.lower <= value && value <= self.upper
    }

    /// Width of the bound interval.
    #[inline]
    pub fn width(&self) -> f32 {
        self.upper - self.lower
    }

    /// Check if bounds are tight (width below threshold).
    #[inline]
    pub fn is_tight(&self, epsilon: f32) -> bool {
        self.width() <= epsilon
    }

    /// Check if bounds have exploded to infinity.
    #[inline]
    pub fn is_unbounded(&self) -> bool {
        self.lower.is_infinite() || self.upper.is_infinite()
    }

    /// Intersect two bounds.
    #[inline]
    pub fn intersect(&self, other: &Bound) -> Option<Bound> {
        let lower = self.lower.max(other.lower);
        let upper = self.upper.min(other.upper);
        if lower <= upper {
            Some(Bound { lower, upper })
        } else {
            None
        }
    }

    /// Union of two bounds (convex hull).
    #[inline]
    pub fn union(&self, other: &Bound) -> Bound {
        Bound {
            lower: self.lower.min(other.lower),
            upper: self.upper.max(other.upper),
        }
    }
}

impl From<RangeInclusive<f32>> for Bound {
    fn from(range: RangeInclusive<f32>) -> Self {
        Self::new(*range.start(), *range.end())
    }
}

/// Output values at a specific layer in the network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerOutput {
    /// Layer index (0-based).
    pub layer_idx: usize,
    /// Layer name (if available).
    pub layer_name: Option<String>,
    /// Layer type (e.g., "Linear", "ReLU").
    pub layer_type: String,
    /// Output values at this layer (flattened).
    pub values: Vec<f32>,
    /// Minimum output value.
    pub min_value: f32,
    /// Maximum output value.
    pub max_value: f32,
}

impl LayerOutput {
    /// Create a new LayerOutput from values.
    pub fn new(
        layer_idx: usize,
        layer_name: Option<String>,
        layer_type: String,
        values: Vec<f32>,
    ) -> Self {
        let min_value = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max_value = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        Self {
            layer_idx,
            layer_name,
            layer_type,
            values,
            min_value,
            max_value,
        }
    }
}

/// Information about which constraint was violated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViolatedConstraint {
    /// Index of the output dimension that was violated.
    pub output_idx: usize,
    /// The output value that violated the constraint.
    pub actual_value: f32,
    /// The required bound that was violated.
    pub required_bound: Bound,
    /// Direction of violation.
    pub violation_type: ViolationType,
    /// Amount by which the bound was violated.
    pub violation_amount: f32,
}

/// Type of constraint violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationType {
    /// Output value below lower bound.
    BelowLower,
    /// Output value above upper bound.
    AboveUpper,
}

impl ViolatedConstraint {
    /// Detect which constraint was violated given output values and bounds.
    pub fn detect(output: &[f32], bounds: &[Bound]) -> Option<Self> {
        for (idx, (&value, bound)) in output.iter().zip(bounds.iter()).enumerate() {
            if value < bound.lower {
                return Some(Self {
                    output_idx: idx,
                    actual_value: value,
                    required_bound: *bound,
                    violation_type: ViolationType::BelowLower,
                    violation_amount: bound.lower - value,
                });
            }
            if value > bound.upper {
                return Some(Self {
                    output_idx: idx,
                    actual_value: value,
                    required_bound: *bound,
                    violation_type: ViolationType::AboveUpper,
                    violation_amount: value - bound.upper,
                });
            }
        }
        None
    }

    /// Human-readable description of the violation.
    pub fn explain(&self) -> String {
        match self.violation_type {
            ViolationType::BelowLower => format!(
                "Output[{}] = {:.6} < lower bound {:.6} (by {:.6})",
                self.output_idx,
                self.actual_value,
                self.required_bound.lower,
                self.violation_amount
            ),
            ViolationType::AboveUpper => format!(
                "Output[{}] = {:.6} > upper bound {:.6} (by {:.6})",
                self.output_idx,
                self.actual_value,
                self.required_bound.upper,
                self.violation_amount
            ),
        }
    }
}

/// Detailed counterexample with layer-by-layer trace and explanation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformativeCounterexample {
    /// Concrete input values that violate the property.
    pub input: Vec<f32>,
    /// Output values at the counterexample input.
    pub output: Vec<f32>,
    /// Layer-by-layer trace showing values at each layer.
    pub trace: Vec<LayerOutput>,
    /// Which constraint was violated.
    pub violated_constraint: Option<ViolatedConstraint>,
    /// Human-readable explanation of the counterexample.
    pub explanation: String,
}

impl InformativeCounterexample {
    /// Create an informative counterexample from basic inputs/outputs.
    ///
    /// If bounds are provided, detects which constraint was violated.
    pub fn new(input: Vec<f32>, output: Vec<f32>, output_bounds: Option<&[Bound]>) -> Self {
        let violated_constraint =
            output_bounds.and_then(|bounds| ViolatedConstraint::detect(&output, bounds));

        let explanation = match &violated_constraint {
            Some(vc) => format!(
                "Property violated: {}. Input: {:?}",
                vc.explain(),
                &input[..input.len().min(5)]
            ),
            None => format!(
                "Counterexample found. Input: {:?}, Output: {:?}",
                &input[..input.len().min(5)],
                &output[..output.len().min(5)]
            ),
        };

        Self {
            input,
            output,
            trace: Vec::new(),
            violated_constraint,
            explanation,
        }
    }

    /// Add a layer output to the trace.
    pub fn add_layer_output(&mut self, layer_output: LayerOutput) {
        self.trace.push(layer_output);
    }

    /// Set the layer trace.
    pub fn with_trace(mut self, trace: Vec<LayerOutput>) -> Self {
        self.trace = trace;
        self
    }

    /// Get a formatted string showing the layer-by-layer trace.
    pub fn format_trace(&self) -> String {
        if self.trace.is_empty() {
            return "No layer trace available.".to_string();
        }

        let mut result = String::from("Layer-by-layer trace:\n");
        for layer in &self.trace {
            let name_part = layer
                .layer_name
                .as_ref()
                .map(|n| format!(" ({})", n))
                .unwrap_or_default();
            result.push_str(&format!(
                "  Layer {:3}{}: {:10} | min={:10.4} max={:10.4} | {} values\n",
                layer.layer_idx,
                name_part,
                layer.layer_type,
                layer.min_value,
                layer.max_value,
                layer.values.len()
            ));
        }
        result
    }
}

/// Proof certificate for verified (UNSAT) results.
///
/// When verification succeeds, the solver can optionally produce a proof
/// that the property holds. This proof can be exported in various formats
/// for independent verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationProof {
    /// Proof format identifier.
    pub format: ProofFormat,
    /// Raw proof bytes (format-specific encoding).
    pub data: Vec<u8>,
    /// Number of proof steps (if available).
    pub num_steps: Option<usize>,
    /// Summary of proof statistics.
    pub stats: Option<ProofStats>,
}

/// Proof format for UNSAT certificates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProofFormat {
    /// Alethe format (SMT-LIB extension, checkable by carcara).
    Alethe,
    /// LFSC (Logical Framework with Side Conditions).
    Lfsc,
    /// DRAT (Delete Resolution Asymmetric Tautology) - for SAT.
    Drat,
    /// Custom format for bound propagation proofs.
    BoundTrace,
}

/// Statistics about the proof.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProofStats {
    /// Number of assumptions (input assertions).
    pub num_assumptions: usize,
    /// Number of resolution steps.
    pub num_resolutions: usize,
    /// Number of theory lemmas.
    pub num_theory_lemmas: usize,
    /// Total proof size in bytes.
    pub size_bytes: usize,
}

impl VerificationProof {
    /// Create a new Alethe format proof.
    pub fn alethe(proof_text: String) -> Self {
        let data = proof_text.into_bytes();
        Self {
            format: ProofFormat::Alethe,
            num_steps: None,
            stats: Some(ProofStats {
                size_bytes: data.len(),
                ..Default::default()
            }),
            data,
        }
    }

    /// Create an Alethe proof with statistics.
    pub fn alethe_with_stats(proof_text: String, num_steps: usize, stats: ProofStats) -> Self {
        let data = proof_text.into_bytes();
        Self {
            format: ProofFormat::Alethe,
            num_steps: Some(num_steps),
            stats: Some(ProofStats {
                size_bytes: data.len(),
                ..stats
            }),
            data,
        }
    }

    /// Get the proof as a UTF-8 string (for text-based formats like Alethe).
    pub fn as_text(&self) -> Option<&str> {
        match self.format {
            ProofFormat::Alethe | ProofFormat::Lfsc => std::str::from_utf8(&self.data).ok(),
            _ => None,
        }
    }

    /// Get the raw proof bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Export proof to a file.
    pub fn save_to_file(&self, path: &std::path::Path) -> std::io::Result<()> {
        std::fs::write(path, &self.data)
    }
}

/// Result of a verification query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationResult {
    /// Property verified: all outputs within bounds for all inputs in region.
    Verified {
        /// Certified output bounds.
        output_bounds: Vec<Bound>,
        /// Optional UNSAT proof certificate.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        proof: Option<Box<VerificationProof>>,
    },
    /// Property violated: counterexample found.
    Violated {
        /// Concrete counterexample input.
        counterexample: Vec<f32>,
        /// Output at counterexample.
        output: Vec<f32>,
        /// Detailed counterexample information (if available).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        details: Option<Box<InformativeCounterexample>>,
    },
    /// Verification inconclusive: bounds too loose.
    Unknown {
        /// Best bounds achieved.
        bounds: Vec<Bound>,
        /// Reason verification couldn't complete.
        reason: String,
    },
    /// Verification timed out.
    Timeout {
        /// Partial bounds at timeout.
        partial_bounds: Option<Vec<Bound>>,
    },
}

impl VerificationResult {
    pub fn is_verified(&self) -> bool {
        matches!(self, VerificationResult::Verified { .. })
    }
}

/// Specification of a property to verify.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationSpec {
    /// Input bounds (per element, in row-major order).
    pub input_bounds: Vec<Bound>,
    /// Required output bounds (property holds if outputs within these).
    pub output_bounds: Vec<Bound>,
    /// Optional timeout in milliseconds.
    pub timeout_ms: Option<u64>,
    /// Optional input shape (if None, input is treated as 1D).
    /// This allows proper reshaping for Conv1d/Conv2d inputs.
    #[serde(default)]
    pub input_shape: Option<Vec<usize>>,
}

/// Layer types supported by γ-CROWN.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerType {
    // Basic layers
    Linear,
    Conv1d,
    Conv2d,
    /// Average pooling over spatial dimensions
    AveragePool,
    /// Max pooling over spatial dimensions
    MaxPool,

    // Activations
    ReLU,
    /// Leaky ReLU: y = x if x >= 0, else alpha * x (typically alpha = 0.01)
    LeakyRelu,
    GELU,
    /// SiLU (Swish): x * sigmoid(x), commonly used in LLMs
    SiLU,
    Sigmoid,
    Tanh,
    /// Softplus: ln(1 + exp(x))
    Softplus,
    Softmax,
    /// Softmax with causal mask (for decoder attention)
    CausalSoftmax,
    /// Clip: clamp values to [min, max] range
    Clip,
    /// ELU (Exponential Linear Unit): x if x >= 0, else alpha * (exp(x) - 1)
    Elu,
    /// SELU (Scaled ELU): lambda * (x if x >= 0, else alpha * (exp(x) - 1))
    /// Uses fixed constants: alpha ≈ 1.6733, lambda ≈ 1.0507
    Selu,
    /// PRelu (Parametric ReLU): y = x if x >= 0, else slope * x
    /// Unlike LeakyRelu, slope is a learned per-channel parameter
    PRelu,
    /// HardSigmoid: y = max(0, min(1, alpha * x + beta))
    /// Default: alpha = 0.2, beta = 0.5. More efficient than Sigmoid.
    HardSigmoid,
    /// HardSwish: y = x * HardSigmoid(x)
    /// Used in MobileNetV3 for efficiency
    HardSwish,
    /// Exponential: y = exp(x)
    Exp,
    /// Natural logarithm: y = ln(x), requires x > 0
    Log,
    /// CELU (Continuous ELU): max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
    /// Continuous at x=0 with continuous derivatives
    Celu,
    /// Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    /// Self-regularizing non-monotonic activation (YOLOv4, etc)
    Mish,
    /// LogSoftmax: log(softmax(x)) = x - logsumexp(x)
    /// More numerically stable than log(softmax(x))
    LogSoftmax,
    /// ThresholdedRelu: y = x if x > alpha, else 0
    /// Default alpha = 1.0 (unlike ReLU which uses 0)
    ThresholdedRelu,
    /// Shrink: soft thresholding / shrinkage operation
    /// y = x - bias if x > lambd, y = x + bias if x < -lambd, else 0
    /// Default: bias = 0.0, lambd = 0.5
    Shrink,
    /// Softsign: y = x / (1 + |x|)
    /// Output range (-1, 1), similar to tanh but computationally cheaper
    Softsign,

    // Rounding operations (for quantization checks)
    /// Floor: y = floor(x) - rounds towards negative infinity
    Floor,
    /// Ceil: y = ceil(x) - rounds towards positive infinity
    Ceil,
    /// Round: y = round(x) - rounds to nearest integer (0.5 rounds away from zero)
    Round,

    // Mathematical operations
    /// Sign: y = -1 if x < 0, 0 if x == 0, 1 if x > 0
    Sign,
    /// Reciprocal: y = 1/x (requires x != 0)
    Reciprocal,

    // Trigonometric (for positional encodings)
    /// Sine function: y = sin(x)
    Sin,
    /// Cosine function: y = cos(x)
    Cos,

    // Normalization
    LayerNorm,
    /// RMSNorm: x / sqrt(mean(x^2) + eps), used in LLMs
    RMSNorm,
    BatchNorm,

    // Transformer components
    MultiHeadAttention,
    /// Token embedding lookup: indices -> embeddings
    Embedding,

    // Structural
    Add,
    Concat,
    Reshape,
    Flatten,
    /// Tensor transpose (permute axes)
    Transpose,

    // Bounded operations (both inputs are bounded)
    /// Matrix multiplication of two bounded tensors (e.g., Q @ K^T in attention)
    MatMul,
    /// Element-wise multiplication (e.g., attention scaling by constant)
    Mul,

    // Element-wise arithmetic
    /// Element-wise negation: y = -x
    Neg,
    /// Element-wise absolute value: y = |x|
    Abs,
    /// Element-wise square root: y = sqrt(x). Assumes x >= 0.
    Sqrt,
    /// Element-wise division: y = x / divisor. Divisor is a constant.
    Div,
    /// Element-wise subtraction: y = x - constant or y = constant - x
    Sub,
    /// Element-wise power: y = x^p where p is a constant.
    Pow,

    // Reduction operations
    /// Mean reduction over specified axes.
    ReduceMean,
    /// Sum reduction over specified axes.
    ReduceSum,

    // Tiling/broadcasting operations
    /// Repeat tensor along specified axis: tile(x, reps) repeats x reps times.
    /// Used for GQA (Grouped Query Attention) to expand KV heads to match Q heads.
    /// Attributes: "axis" (i64) - axis to repeat along, "reps" (i64) - repetition count.
    Tile,

    // Conditional operations
    /// Element-wise conditional: Where(condition, x, y) = x if condition else y.
    /// For interval bounds, takes union of x and y bounds.
    Where,

    // Index/selection operations
    /// NonZero: returns indices of non-zero elements.
    /// Output shape is data-dependent: [rank(input), num_nonzero].
    /// For bound propagation, returns conservative bounds on possible indices.
    NonZero,
    /// Slice: extracts a contiguous range along an axis.
    /// Used to implement Split (multi-output op) as multiple Slice layers.
    /// Attributes: "axis" (i64), "start" (i64), "end" (i64).
    Slice,

    // Placeholder for unsupported ops
    Unknown,
}

/// Macro to create ShapeMismatch error with debug assertion.
#[macro_export]
macro_rules! shape_mismatch_err {
    ($expected:expr, $got:expr) => {{
        let exp: Vec<usize> = $expected;
        let got: Vec<usize> = $got;
        debug_assert!(
            exp != got,
            "BUG: ShapeMismatch with identical shapes {:?} at {}:{}:{}",
            exp,
            file!(),
            line!(),
            column!()
        );
        GammaError::ShapeMismatch { expected: exp, got }
    }};
}

/// Error types for γ-CROWN operations.
#[derive(Debug)]
pub enum GammaError {
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    UnsupportedLayer(String),

    UnsupportedOp(String),

    BoundsExplosion {
        layer: String,
    },

    ModelLoad(String),

    InvalidSpec(String),

    NumericalInstability(String),

    NotSupported(String),

    LayerError {
        layer_index: usize,
        layer_type: String,
        source: Box<GammaError>,
    },
}

impl GammaError {
    /// Create a ShapeMismatch error, panicking if shapes are identical (bug indicator).
    #[track_caller]
    pub fn shape_mismatch(expected: Vec<usize>, got: Vec<usize>) -> Self {
        if expected == got {
            let loc = std::panic::Location::caller();
            panic!(
                "BUG at {}:{}:{}: ShapeMismatch created with identical shapes: {:?}. \
                 This indicates a bug in the calling code - shapes match but error was raised.",
                loc.file(),
                loc.line(),
                loc.column(),
                expected
            );
        }
        GammaError::ShapeMismatch { expected, got }
    }

    /// Create a ShapeMismatch error with panic on identical shapes.
    /// Use this for map_err and other closure-based error creation.
    #[track_caller]
    #[inline]
    pub fn shape_mismatch_checked(expected: Vec<usize>, got: Vec<usize>) -> Self {
        // Always panic on identical shapes - this is a bug indicator
        if expected == got {
            let loc = std::panic::Location::caller();
            panic!(
                "BUG at {}:{}:{}: ShapeMismatch with identical shapes: {:?}.",
                loc.file(),
                loc.line(),
                loc.column(),
                expected
            );
        }
        GammaError::ShapeMismatch { expected, got }
    }
}

impl std::fmt::Display for GammaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GammaError::ShapeMismatch { expected, got } => {
                // Debug check: if shapes are identical, this is a bug - panic to get full stack
                if expected == got {
                    panic!(
                        "BUG: ShapeMismatch with identical shapes {:?}. \
                         Error was created somewhere that bypassed the shape_mismatch() helper. \
                         Check direct GammaError::ShapeMismatch {{ ... }} constructions.",
                        expected
                    );
                }
                write!(f, "Shape mismatch: expected {:?}, got {:?}", expected, got)
            }
            GammaError::UnsupportedLayer(s) => write!(f, "Unsupported layer type: {}", s),
            GammaError::UnsupportedOp(s) => write!(f, "Unsupported operation: {}", s),
            GammaError::BoundsExplosion { layer } => {
                write!(
                    f,
                    "Bounds explosion: layer {} produced infinite bounds",
                    layer
                )
            }
            GammaError::ModelLoad(s) => write!(f, "Model loading failed: {}", s),
            GammaError::InvalidSpec(s) => write!(f, "Invalid specification: {}", s),
            GammaError::NumericalInstability(s) => write!(f, "Numerical instability: {}", s),
            GammaError::NotSupported(s) => write!(f, "Not supported: {}", s),
            GammaError::LayerError {
                layer_index,
                layer_type,
                source,
            } => write!(
                f,
                "Layer {} ({}) failed: {}",
                layer_index, layer_type, source
            ),
        }
    }
}

impl std::error::Error for GammaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            GammaError::LayerError { source, .. } => Some(source.as_ref()),
            _ => None,
        }
    }
}

pub type Result<T> = std::result::Result<T, GammaError>;

// =============================================================================
// Optional Acceleration Hooks
// =============================================================================

/// Minimal GEMM interface for accelerating CROWN/α-CROWN linear backprop.
///
/// Computes `C = A @ B` for f32 row-major matrices:
/// - `A`: shape (m, k)
/// - `B`: shape (k, n)
/// - `C`: shape (m, n)
///
/// Implementations may run on CPU, GPU, or remote accelerators. Callers must be
/// prepared to fall back to a local implementation if this returns an error.
///
/// The trait requires `Sync + Send` to allow use in rayon parallel contexts
/// (e.g., parallel domain processing in BaB).
pub trait GemmEngine: Sync + Send {
    fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bound_operations() {
        let a = Bound::new(0.0, 1.0);
        let b = Bound::new(0.5, 1.5);

        assert!(a.contains(0.5));
        assert!(!a.contains(1.5));

        let intersection = a.intersect(&b).unwrap();
        assert_eq!(intersection.lower, 0.5);
        assert_eq!(intersection.upper, 1.0);

        let union = a.union(&b);
        assert_eq!(union.lower, 0.0);
        assert_eq!(union.upper, 1.5);
    }

    #[test]
    fn test_concrete_bound() {
        let b = Bound::concrete(0.5);
        assert_eq!(b.width(), 0.0);
        assert!(b.is_tight(0.001));
    }

    // Tests to catch surviving mutations
    #[test]
    fn test_bound_width_computation() {
        // Test width = upper - lower (not lower - upper or any other formula)
        let b = Bound::new(1.0, 3.0);
        assert_eq!(b.width(), 2.0); // 3.0 - 1.0 = 2.0

        // Verify width is positive for valid bounds
        let b2 = Bound::new(-5.0, 5.0);
        assert_eq!(b2.width(), 10.0); // 5.0 - (-5.0) = 10.0

        // Test with negative bounds
        let b3 = Bound::new(-10.0, -3.0);
        assert_eq!(b3.width(), 7.0); // -3.0 - (-10.0) = 7.0

        // Ensure width distinguishes different bounds
        let narrow = Bound::new(0.0, 0.1);
        let wide = Bound::new(0.0, 10.0);
        assert!(narrow.width() < wide.width());
        assert_eq!(narrow.width(), 0.1);
        assert_eq!(wide.width(), 10.0);
    }

    #[test]
    fn test_bound_is_tight_boundary_conditions() {
        // Test boundary conditions for is_tight
        let b = Bound::new(0.0, 0.5);
        assert_eq!(b.width(), 0.5);

        // Exactly at epsilon boundary - should be tight (width <= epsilon)
        assert!(b.is_tight(0.5));

        // Epsilon just below width - should NOT be tight
        assert!(!b.is_tight(0.49));
        assert!(!b.is_tight(0.4));
        assert!(!b.is_tight(0.1));

        // Epsilon just above width - should be tight
        assert!(b.is_tight(0.51));
        assert!(b.is_tight(1.0));

        // Concrete bounds are always tight
        let concrete = Bound::concrete(5.0);
        assert!(concrete.is_tight(0.0));
        assert!(concrete.is_tight(0.0001));
        assert!(concrete.is_tight(f32::EPSILON));

        // Wide bounds need large epsilon
        let wide = Bound::new(0.0, 100.0);
        assert!(!wide.is_tight(99.0));
        assert!(wide.is_tight(100.0));
        assert!(wide.is_tight(101.0));
    }

    #[test]
    fn test_bound_is_unbounded_all_cases() {
        // Finite bounds - NOT unbounded
        let finite = Bound::new(-1e10, 1e10);
        assert!(!finite.is_unbounded());

        // Lower is infinite
        let lower_inf = Bound::new(f32::NEG_INFINITY, 0.0);
        assert!(lower_inf.is_unbounded());

        // Upper is infinite
        let upper_inf = Bound::new(0.0, f32::INFINITY);
        assert!(upper_inf.is_unbounded());

        // Both infinite
        let both_inf = Bound::new(f32::NEG_INFINITY, f32::INFINITY);
        assert!(both_inf.is_unbounded());

        // Edge case: very large but finite
        let large = Bound::new(-f32::MAX, f32::MAX);
        assert!(!large.is_unbounded());

        // Zero bounds - NOT unbounded
        let zero = Bound::concrete(0.0);
        assert!(!zero.is_unbounded());
    }

    #[test]
    fn test_verification_result_is_verified_all_variants() {
        // Verified variant - MUST return true
        let verified = VerificationResult::Verified {
            output_bounds: vec![Bound::new(0.0, 1.0)],
            proof: None,
        };
        assert!(verified.is_verified());

        // Violated variant - MUST return false
        let violated = VerificationResult::Violated {
            counterexample: vec![0.5],
            output: vec![1.5],
            details: None,
        };
        assert!(!violated.is_verified());

        // Unknown variant - MUST return false
        let unknown = VerificationResult::Unknown {
            bounds: vec![Bound::new(-1.0, 2.0)],
            reason: "Bounds too loose".to_string(),
        };
        assert!(!unknown.is_verified());

        // Timeout variant with partial bounds - MUST return false
        let timeout_partial = VerificationResult::Timeout {
            partial_bounds: Some(vec![Bound::new(0.0, 1.0)]),
        };
        assert!(!timeout_partial.is_verified());

        // Timeout variant without partial bounds - MUST return false
        let timeout_none = VerificationResult::Timeout {
            partial_bounds: None,
        };
        assert!(!timeout_none.is_verified());
    }

    #[test]
    fn test_bound_width_distinguishes_bounds() {
        // Ensure width is sensitive to both lower and upper changes
        let base = Bound::new(0.0, 1.0);
        let wider_upper = Bound::new(0.0, 2.0);
        let wider_lower = Bound::new(-1.0, 1.0);

        assert!(wider_upper.width() > base.width());
        assert!(wider_lower.width() > base.width());
        assert_eq!(wider_upper.width(), wider_lower.width()); // Both are width 2.0
    }

    #[test]
    fn test_bound_from_range_inclusive() {
        let range = 0.5f32..=1.5f32;
        let bound: Bound = range.into();
        assert_eq!(bound.lower, 0.5);
        assert_eq!(bound.upper, 1.5);
        assert_eq!(bound.width(), 1.0);
    }

    #[test]
    fn test_intersect_disjoint_returns_none() {
        let a = Bound::new(0.0, 1.0);
        let b = Bound::new(2.0, 3.0);
        assert!(a.intersect(&b).is_none());
    }

    #[test]
    fn test_contains_edge_cases() {
        let b = Bound::new(0.0, 1.0);
        // Boundary values should be contained (inclusive)
        assert!(b.contains(0.0));
        assert!(b.contains(1.0));
        // Just outside should not be contained
        assert!(!b.contains(-0.0001));
        assert!(!b.contains(1.0001));
    }

    // Tests for informative counterexample types
    #[test]
    fn test_violated_constraint_detect_below_lower() {
        let output = vec![0.5, -0.1, 0.8];
        let bounds = vec![
            Bound::new(0.0, 1.0), // ok
            Bound::new(0.0, 1.0), // violated: -0.1 < 0.0
            Bound::new(0.0, 1.0), // ok
        ];

        let vc = ViolatedConstraint::detect(&output, &bounds).unwrap();
        assert_eq!(vc.output_idx, 1);
        assert_eq!(vc.actual_value, -0.1);
        assert_eq!(vc.violation_type, ViolationType::BelowLower);
        assert!((vc.violation_amount - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_violated_constraint_detect_above_upper() {
        let output = vec![0.5, 0.3, 1.5];
        let bounds = vec![
            Bound::new(0.0, 1.0), // ok
            Bound::new(0.0, 1.0), // ok
            Bound::new(0.0, 1.0), // violated: 1.5 > 1.0
        ];

        let vc = ViolatedConstraint::detect(&output, &bounds).unwrap();
        assert_eq!(vc.output_idx, 2);
        assert_eq!(vc.actual_value, 1.5);
        assert_eq!(vc.violation_type, ViolationType::AboveUpper);
        assert!((vc.violation_amount - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_violated_constraint_detect_no_violation() {
        let output = vec![0.5, 0.3, 0.8];
        let bounds = vec![
            Bound::new(0.0, 1.0),
            Bound::new(0.0, 1.0),
            Bound::new(0.0, 1.0),
        ];

        assert!(ViolatedConstraint::detect(&output, &bounds).is_none());
    }

    #[test]
    fn test_violated_constraint_explain() {
        let vc = ViolatedConstraint {
            output_idx: 0,
            actual_value: 1.5,
            required_bound: Bound::new(0.0, 1.0),
            violation_type: ViolationType::AboveUpper,
            violation_amount: 0.5,
        };

        let explanation = vc.explain();
        assert!(explanation.contains("Output[0]"));
        assert!(explanation.contains("1.5"));
        assert!(explanation.contains("upper bound"));
        assert!(explanation.contains("1.0"));
    }

    #[test]
    fn test_layer_output_new() {
        let values = vec![1.0, -2.0, 3.0, 0.5];
        let layer = LayerOutput::new(0, Some("relu1".to_string()), "ReLU".to_string(), values);

        assert_eq!(layer.layer_idx, 0);
        assert_eq!(layer.layer_name, Some("relu1".to_string()));
        assert_eq!(layer.layer_type, "ReLU");
        assert_eq!(layer.min_value, -2.0);
        assert_eq!(layer.max_value, 3.0);
        assert_eq!(layer.values.len(), 4);
    }

    #[test]
    fn test_informative_counterexample_new() {
        let input = vec![0.5, 0.5];
        let output = vec![1.5]; // violates [0, 1]
        let bounds = vec![Bound::new(0.0, 1.0)];

        let ce = InformativeCounterexample::new(input.clone(), output.clone(), Some(&bounds));

        assert_eq!(ce.input, input);
        assert_eq!(ce.output, output);
        assert!(ce.violated_constraint.is_some());
        assert!(ce.explanation.contains("Property violated"));
    }

    #[test]
    fn test_informative_counterexample_format_trace() {
        let mut ce = InformativeCounterexample::new(vec![1.0], vec![2.0], None);

        // Empty trace
        assert!(ce.format_trace().contains("No layer trace"));

        // Add layers
        ce.add_layer_output(LayerOutput::new(0, None, "Linear".to_string(), vec![1.5]));
        ce.add_layer_output(LayerOutput::new(
            1,
            Some("relu".to_string()),
            "ReLU".to_string(),
            vec![2.0],
        ));

        let trace_str = ce.format_trace();
        assert!(trace_str.contains("Layer   0"));
        assert!(trace_str.contains("Layer   1"));
        assert!(trace_str.contains("Linear"));
        assert!(trace_str.contains("ReLU"));
    }

    #[test]
    fn test_informative_counterexample_with_trace() {
        let trace = vec![
            LayerOutput::new(0, None, "Linear".to_string(), vec![1.0]),
            LayerOutput::new(1, None, "ReLU".to_string(), vec![1.0]),
        ];

        let ce = InformativeCounterexample::new(vec![0.5], vec![1.0], None).with_trace(trace);

        assert_eq!(ce.trace.len(), 2);
    }

    // Tests for VerificationProof
    #[test]
    fn test_verification_proof_alethe() {
        let proof_text = "(assume h1 (> x 0))".to_string();
        let proof = VerificationProof::alethe(proof_text.clone());

        assert_eq!(proof.format, ProofFormat::Alethe);
        assert!(proof.num_steps.is_none());
        assert!(proof.stats.is_some());
        assert_eq!(proof.stats.as_ref().unwrap().size_bytes, proof_text.len());
        assert_eq!(proof.data, proof_text.as_bytes());
    }

    #[test]
    fn test_verification_proof_alethe_with_stats() {
        let proof_text = "(step s1 (resolution h1 h2))".to_string();
        let stats = ProofStats {
            num_assumptions: 2,
            num_resolutions: 1,
            num_theory_lemmas: 0,
            size_bytes: 0, // Will be overwritten
        };

        let proof = VerificationProof::alethe_with_stats(proof_text.clone(), 3, stats);

        assert_eq!(proof.format, ProofFormat::Alethe);
        assert_eq!(proof.num_steps, Some(3));
        assert!(proof.stats.is_some());
        let stats = proof.stats.unwrap();
        assert_eq!(stats.num_assumptions, 2);
        assert_eq!(stats.num_resolutions, 1);
        assert_eq!(stats.size_bytes, proof_text.len());
    }

    #[test]
    fn test_verification_proof_as_text() {
        // Alethe format should return text
        let proof_text = "(step s1 (resolution))";
        let proof = VerificationProof::alethe(proof_text.to_string());
        assert_eq!(proof.as_text(), Some(proof_text));

        // Binary format (Drat) should return None
        let binary_proof = VerificationProof {
            format: ProofFormat::Drat,
            data: vec![0x01, 0x02, 0x03],
            num_steps: None,
            stats: None,
        };
        assert!(binary_proof.as_text().is_none());

        // BoundTrace format should return None
        let bound_proof = VerificationProof {
            format: ProofFormat::BoundTrace,
            data: vec![0xDE, 0xAD, 0xBE, 0xEF],
            num_steps: None,
            stats: None,
        };
        assert!(bound_proof.as_text().is_none());

        // LFSC format should return text
        let lfsc_proof = VerificationProof {
            format: ProofFormat::Lfsc,
            data: "(check (holds true))".as_bytes().to_vec(),
            num_steps: None,
            stats: None,
        };
        assert!(lfsc_proof.as_text().is_some());
    }

    #[test]
    fn test_verification_proof_as_bytes() {
        let proof_text = "proof content";
        let proof = VerificationProof::alethe(proof_text.to_string());
        assert_eq!(proof.as_bytes(), proof_text.as_bytes());

        // Binary data
        let binary_data = vec![0x01, 0x02, 0x03, 0x04];
        let binary_proof = VerificationProof {
            format: ProofFormat::Drat,
            data: binary_data.clone(),
            num_steps: None,
            stats: None,
        };
        assert_eq!(binary_proof.as_bytes(), &binary_data);
    }

    #[test]
    fn test_proof_stats_default() {
        let stats = ProofStats::default();
        assert_eq!(stats.num_assumptions, 0);
        assert_eq!(stats.num_resolutions, 0);
        assert_eq!(stats.num_theory_lemmas, 0);
        assert_eq!(stats.size_bytes, 0);
    }

    #[test]
    fn test_proof_format_equality() {
        assert_eq!(ProofFormat::Alethe, ProofFormat::Alethe);
        assert_ne!(ProofFormat::Alethe, ProofFormat::Lfsc);
        assert_ne!(ProofFormat::Drat, ProofFormat::BoundTrace);
    }

    #[test]
    fn test_verification_proof_save_to_file() {
        use std::io::Read;
        let proof_text = "test proof data";
        let proof = VerificationProof::alethe(proof_text.to_string());

        // Create temp file
        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join("test_proof_export.alethe");

        // Save and read back
        proof.save_to_file(&temp_path).unwrap();
        let mut file = std::fs::File::open(&temp_path).unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();

        assert_eq!(contents, proof_text);

        // Cleanup
        std::fs::remove_file(&temp_path).ok();
    }

    // Tests for GammaError
    #[test]
    fn test_gamma_error_shape_mismatch_display() {
        let err = GammaError::ShapeMismatch {
            expected: vec![2, 3],
            got: vec![2, 4],
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Shape mismatch"));
        assert!(msg.contains("[2, 3]"));
        assert!(msg.contains("[2, 4]"));
    }

    #[test]
    fn test_gamma_error_unsupported_layer_display() {
        let err = GammaError::UnsupportedLayer("CustomOp".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("Unsupported layer type"));
        assert!(msg.contains("CustomOp"));
    }

    #[test]
    fn test_gamma_error_unsupported_op_display() {
        let err = GammaError::UnsupportedOp("Einsum".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("Unsupported operation"));
        assert!(msg.contains("Einsum"));
    }

    #[test]
    fn test_gamma_error_bounds_explosion_display() {
        let err = GammaError::BoundsExplosion {
            layer: "relu_5".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Bounds explosion"));
        assert!(msg.contains("relu_5"));
        assert!(msg.contains("infinite"));
    }

    #[test]
    fn test_gamma_error_model_load_display() {
        let err = GammaError::ModelLoad("file not found".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("Model loading failed"));
        assert!(msg.contains("file not found"));
    }

    #[test]
    fn test_gamma_error_invalid_spec_display() {
        let err = GammaError::InvalidSpec("empty bounds".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("Invalid specification"));
        assert!(msg.contains("empty bounds"));
    }

    #[test]
    fn test_gamma_error_numerical_instability_display() {
        let err = GammaError::NumericalInstability("NaN detected".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("Numerical instability"));
        assert!(msg.contains("NaN"));
    }

    #[test]
    fn test_gamma_error_not_supported_display() {
        let err = GammaError::NotSupported("dynamic shapes".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("Not supported"));
        assert!(msg.contains("dynamic shapes"));
    }

    #[test]
    fn test_gamma_error_layer_error_display() {
        let inner = Box::new(GammaError::UnsupportedOp("op1".to_string()));
        let err = GammaError::LayerError {
            layer_index: 5,
            layer_type: "MatMul".to_string(),
            source: inner,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Layer 5"));
        assert!(msg.contains("MatMul"));
        assert!(msg.contains("failed"));
    }

    #[test]
    fn test_gamma_error_source_returns_inner_for_layer_error() {
        let inner = Box::new(GammaError::UnsupportedOp("op1".to_string()));
        let err = GammaError::LayerError {
            layer_index: 0,
            layer_type: "Test".to_string(),
            source: inner,
        };
        assert!(std::error::Error::source(&err).is_some());
    }

    #[test]
    fn test_gamma_error_source_returns_none_for_non_layer_error() {
        let err = GammaError::UnsupportedOp("test".to_string());
        assert!(std::error::Error::source(&err).is_none());

        let err2 = GammaError::ModelLoad("test".to_string());
        assert!(std::error::Error::source(&err2).is_none());
    }

    #[test]
    fn test_gamma_error_shape_mismatch_constructor() {
        // Different shapes should work fine
        let err = GammaError::shape_mismatch(vec![1, 2], vec![3, 4]);
        match err {
            GammaError::ShapeMismatch { expected, got } => {
                assert_eq!(expected, vec![1, 2]);
                assert_eq!(got, vec![3, 4]);
            }
            _ => panic!("Expected ShapeMismatch variant"),
        }
    }

    #[test]
    #[should_panic(expected = "BUG")]
    fn test_gamma_error_shape_mismatch_panics_on_identical() {
        // Identical shapes should panic - this is a bug indicator
        let _ = GammaError::shape_mismatch(vec![1, 2, 3], vec![1, 2, 3]);
    }

    #[test]
    fn test_gamma_error_shape_mismatch_checked_different_shapes() {
        let err = GammaError::shape_mismatch_checked(vec![10], vec![20]);
        match err {
            GammaError::ShapeMismatch { expected, got } => {
                assert_eq!(expected, vec![10]);
                assert_eq!(got, vec![20]);
            }
            _ => panic!("Expected ShapeMismatch variant"),
        }
    }

    #[test]
    #[should_panic(expected = "BUG")]
    fn test_gamma_error_shape_mismatch_checked_panics_on_identical() {
        let _ = GammaError::shape_mismatch_checked(vec![5, 5], vec![5, 5]);
    }

    #[test]
    fn test_gamma_error_debug_impl() {
        let err = GammaError::UnsupportedLayer("Test".to_string());
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("UnsupportedLayer"));
        assert!(debug_str.contains("Test"));
    }

    // Tests for VerificationSpec
    #[test]
    fn test_verification_spec_basic() {
        let spec = VerificationSpec {
            input_bounds: vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
            output_bounds: vec![Bound::new(0.0, 1.0)],
            timeout_ms: Some(5000),
            input_shape: None,
        };

        assert_eq!(spec.input_bounds.len(), 2);
        assert_eq!(spec.output_bounds.len(), 1);
        assert_eq!(spec.timeout_ms, Some(5000));
        assert!(spec.input_shape.is_none());
    }

    #[test]
    fn test_verification_spec_with_input_shape() {
        let spec = VerificationSpec {
            input_bounds: vec![Bound::new(0.0, 1.0); 6],
            output_bounds: vec![Bound::new(0.0, 1.0)],
            timeout_ms: None,
            input_shape: Some(vec![2, 3]),
        };

        assert_eq!(spec.input_shape, Some(vec![2, 3]));
        assert!(spec.timeout_ms.is_none());
    }

    #[test]
    fn test_verification_spec_serialization() {
        let spec = VerificationSpec {
            input_bounds: vec![Bound::new(-1.0, 1.0)],
            output_bounds: vec![Bound::new(0.0, 2.0)],
            timeout_ms: Some(1000),
            input_shape: Some(vec![1]),
        };

        // Serialize to JSON
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains("input_bounds"));
        assert!(json.contains("output_bounds"));
        assert!(json.contains("timeout_ms"));

        // Deserialize back
        let deserialized: VerificationSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.input_bounds.len(), 1);
        assert_eq!(deserialized.output_bounds.len(), 1);
        assert_eq!(deserialized.timeout_ms, Some(1000));
    }

    // Tests for LayerType
    #[test]
    fn test_layer_type_debug() {
        assert_eq!(format!("{:?}", LayerType::Linear), "Linear");
        assert_eq!(format!("{:?}", LayerType::ReLU), "ReLU");
        assert_eq!(format!("{:?}", LayerType::Softmax), "Softmax");
        assert_eq!(
            format!("{:?}", LayerType::MultiHeadAttention),
            "MultiHeadAttention"
        );
    }

    #[test]
    fn test_layer_type_equality() {
        assert_eq!(LayerType::Linear, LayerType::Linear);
        assert_ne!(LayerType::Linear, LayerType::ReLU);
        assert_eq!(LayerType::GELU, LayerType::GELU);
        assert_ne!(LayerType::Conv1d, LayerType::Conv2d);
    }

    #[test]
    fn test_layer_type_clone() {
        let layer = LayerType::LayerNorm;
        let cloned = layer.clone();
        assert_eq!(layer, cloned);
    }

    #[test]
    fn test_layer_type_serialization() {
        let layer = LayerType::Softmax;
        let json = serde_json::to_string(&layer).unwrap();
        assert_eq!(json, "\"Softmax\"");

        let deserialized: LayerType = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, LayerType::Softmax);
    }

    #[test]
    fn test_layer_type_all_activations_distinct() {
        // Ensure all activation types are distinct
        let activations = vec![
            LayerType::ReLU,
            LayerType::LeakyRelu,
            LayerType::GELU,
            LayerType::SiLU,
            LayerType::Sigmoid,
            LayerType::Tanh,
            LayerType::Softplus,
            LayerType::Elu,
            LayerType::Selu,
            LayerType::PRelu,
            LayerType::HardSigmoid,
            LayerType::HardSwish,
            LayerType::Celu,
            LayerType::Mish,
            LayerType::ThresholdedRelu,
            LayerType::Softsign,
        ];

        // Check all pairs are distinct
        for (i, a) in activations.iter().enumerate() {
            for (j, b) in activations.iter().enumerate() {
                if i != j {
                    assert_ne!(
                        a, b,
                        "Activation types {:?} and {:?} should be distinct",
                        a, b
                    );
                }
            }
        }
    }

    // Tests for ViolationType
    #[test]
    fn test_violation_type_equality() {
        assert_eq!(ViolationType::BelowLower, ViolationType::BelowLower);
        assert_eq!(ViolationType::AboveUpper, ViolationType::AboveUpper);
        assert_ne!(ViolationType::BelowLower, ViolationType::AboveUpper);
    }

    #[test]
    fn test_violation_type_serialization() {
        let below = ViolationType::BelowLower;
        let json = serde_json::to_string(&below).unwrap();
        let deserialized: ViolationType = serde_json::from_str(&json).unwrap();
        assert_eq!(below, deserialized);
    }

    // Tests for shape_mismatch_err! macro
    #[test]
    fn test_shape_mismatch_err_macro() {
        let err = shape_mismatch_err!(vec![1, 2], vec![3, 4]);
        match err {
            GammaError::ShapeMismatch { expected, got } => {
                assert_eq!(expected, vec![1, 2]);
                assert_eq!(got, vec![3, 4]);
            }
            _ => panic!("Expected ShapeMismatch"),
        }
    }

    // Tests for Bound serde
    #[test]
    fn test_bound_serialization() {
        let bound = Bound::new(-1.5, 2.5);
        let json = serde_json::to_string(&bound).unwrap();
        assert!(json.contains("-1.5"));
        assert!(json.contains("2.5"));

        let deserialized: Bound = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.lower, -1.5);
        assert_eq!(deserialized.upper, 2.5);
    }

    // Tests for VerificationResult serialization
    #[test]
    fn test_verification_result_verified_serialization() {
        let result = VerificationResult::Verified {
            output_bounds: vec![Bound::new(0.0, 1.0)],
            proof: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: VerificationResult = serde_json::from_str(&json).unwrap();
        assert!(deserialized.is_verified());
    }

    #[test]
    fn test_verification_result_violated_serialization() {
        let result = VerificationResult::Violated {
            counterexample: vec![0.5],
            output: vec![1.5],
            details: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: VerificationResult = serde_json::from_str(&json).unwrap();
        assert!(!deserialized.is_verified());
    }

    #[test]
    fn test_verification_result_unknown_serialization() {
        let result = VerificationResult::Unknown {
            bounds: vec![Bound::new(-1.0, 2.0)],
            reason: "too loose".to_string(),
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("Unknown"));
        assert!(json.contains("too loose"));
    }

    #[test]
    fn test_verification_result_timeout_serialization() {
        let result = VerificationResult::Timeout {
            partial_bounds: Some(vec![Bound::new(0.0, 1.0)]),
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("Timeout"));
    }
}
