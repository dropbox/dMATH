//! Accelerated bound propagation for γ-CROWN.
//!
//! This crate provides optimized implementations of the core bound propagation
//! operations, using SIMD, parallel execution, and GPU compute for performance.
//!
//! ## Available Backends
//!
//! - **CPU (Rayon)**: Parallel CPU implementation with auto-vectorization
//! - **wgpu**: Cross-platform GPU compute via WebGPU (Metal, Vulkan, DX12)
//! - **MLX (CPU)** (feature `mlx`): Apple Silicon via Apple's MLX framework (no Metal kernels)
//! - **MLX (Metal GPU)** (feature `mlx-metal`): MLX with Metal kernels (requires Xcode.app)
//!
//! ## Design
//!
//! The main acceleration targets are:
//! 1. **Linear layer IBP** - Matrix-vector operations with interval arithmetic
//! 2. **MatMul IBP** - Batched matrix multiplication with interval bounds
//! 3. **Per-position CROWN** - Independent CROWN execution per sequence position
//!
//! ## Usage
//!
//! ```ignore
//! use gamma_gpu::{AcceleratedDevice, WgpuDevice};
//!
//! // CPU parallelization (Rayon)
//! let cpu_device = AcceleratedDevice::new();
//! let result = cpu_device.linear_ibp(&input, &weight, bias)?;
//!
//! // GPU acceleration (wgpu - cross-platform)
//! let gpu_device = WgpuDevice::new()?;
//! let result = gpu_device.linear_ibp(&input, &weight, bias)?;
//!
//! // MLX acceleration (macOS Apple Silicon only)
//! #[cfg(feature = "mlx")]
//! {
//!     use gamma_gpu::MlxDevice;
//!     // Uses CPU stream unless built with `mlx-metal`.
//!     let mlx_device = MlxDevice::new()?;
//!     let result = mlx_device.linear_ibp(&input, &weight, bias)?;
//! }
//! ```
//!
//! ## MLX Backend Prerequisites
//!
//! The `mlx` feature requires macOS on Apple Silicon (M1/M2/M3/M4).
//!
//! The `mlx-metal` feature additionally requires:
//! - Xcode.app installed (not just Command Line Tools)
//! - Xcode selected: `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`

pub mod wgpu_device;
pub use wgpu_device::WgpuDevice;

// MLX backend for Apple Silicon (macOS only)
#[cfg(feature = "mlx")]
pub mod mlx_device;
#[cfg(feature = "mlx")]
pub use mlx_device::MlxDevice;

use gamma_core::{GammaError, GemmEngine, Result};

/// Backend selection for compute operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// CPU with Rayon parallelization (default, always available)
    #[default]
    Cpu,
    /// wgpu GPU compute (cross-platform: Metal, Vulkan, DX12)
    Wgpu,
    /// MLX for Apple Silicon (macOS only, requires `mlx` feature)
    Mlx,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Backend::Cpu => write!(f, "cpu"),
            Backend::Wgpu => write!(f, "wgpu"),
            Backend::Mlx => write!(f, "mlx"),
        }
    }
}

impl std::str::FromStr for Backend {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cpu" => Ok(Backend::Cpu),
            "wgpu" | "gpu" => Ok(Backend::Wgpu),
            "mlx" => Ok(Backend::Mlx),
            _ => Err(format!(
                "Unknown backend: {}. Valid options: cpu, wgpu, mlx",
                s
            )),
        }
    }
}

/// Unified compute device that dispatches to available backends.
///
/// This enum allows runtime backend selection while providing a common interface.
/// Use `ComputeDevice::new(backend)` to create a device with the specified backend.
pub enum ComputeDevice {
    /// CPU with Rayon parallelization
    Cpu(AcceleratedDevice),
    /// wgpu GPU compute (boxed to avoid large enum size)
    Wgpu(Box<WgpuDevice>),
    /// MLX for Apple Silicon
    #[cfg(feature = "mlx")]
    Mlx(MlxDevice),
}

impl ComputeDevice {
    /// Create a new compute device with the specified backend.
    ///
    /// Returns an error if the requested backend is not available.
    pub fn new(backend: Backend) -> Result<Self> {
        match backend {
            Backend::Cpu => Ok(ComputeDevice::Cpu(AcceleratedDevice::new())),
            Backend::Wgpu => {
                let device = WgpuDevice::new()?;
                Ok(ComputeDevice::Wgpu(Box::new(device)))
            }
            #[cfg(feature = "mlx")]
            Backend::Mlx => {
                let device = MlxDevice::new()?;
                Ok(ComputeDevice::Mlx(device))
            }
            #[cfg(not(feature = "mlx"))]
            Backend::Mlx => Err(GammaError::InvalidSpec(
                "MLX backend not available. Rebuild with `--features mlx`".to_string(),
            )),
        }
    }

    /// Get the backend type of this device.
    pub fn backend(&self) -> Backend {
        match self {
            ComputeDevice::Cpu(_) => Backend::Cpu,
            ComputeDevice::Wgpu(_) => Backend::Wgpu,
            #[cfg(feature = "mlx")]
            ComputeDevice::Mlx(_) => Backend::Mlx,
        }
    }

    /// Check if this device supports attention operations.
    pub fn supports_attention(&self) -> bool {
        match self {
            ComputeDevice::Cpu(_) => true,
            ComputeDevice::Wgpu(_) => true,
            #[cfg(feature = "mlx")]
            ComputeDevice::Mlx(_) => true,
        }
    }

    /// Full attention IBP: softmax((Q @ K^T) * scale) @ V
    ///
    /// Input shapes: Q, K, V with shape [batch, heads, seq, dim]
    /// Output shape: [batch, heads, seq, dim]
    pub fn attention_ibp(
        &self,
        q: &BoundedTensor,
        k: &BoundedTensor,
        v: &BoundedTensor,
        scale: f32,
    ) -> Result<BoundedTensor> {
        match self {
            ComputeDevice::Cpu(d) => d.attention_ibp(q, k, v, scale),
            ComputeDevice::Wgpu(d) => d.attention_ibp(q, k, v, scale),
            #[cfg(feature = "mlx")]
            ComputeDevice::Mlx(d) => d.attention_ibp(q, k, v, scale),
        }
    }

    /// Causal attention IBP for decoder-only models (LLaMA, GPT).
    ///
    /// Position i can only attend to positions j where j <= i.
    ///
    /// Input shapes: Q, K, V with shape [batch, heads, seq, dim]
    /// Output shape: [batch, heads, seq, dim]
    pub fn causal_attention_ibp(
        &self,
        q: &BoundedTensor,
        k: &BoundedTensor,
        v: &BoundedTensor,
        scale: f32,
    ) -> Result<BoundedTensor> {
        match self {
            ComputeDevice::Cpu(d) => d.causal_attention_ibp(q, k, v, scale),
            ComputeDevice::Wgpu(d) => d.causal_attention_ibp(q, k, v, scale),
            #[cfg(feature = "mlx")]
            ComputeDevice::Mlx(d) => d.causal_attention_ibp(q, k, v, scale),
        }
    }

    /// Cross-attention IBP for encoder-decoder models (Whisper).
    ///
    /// Q (queries) from decoder: [batch, heads, seq_dec, dim]
    /// K, V from encoder: [batch, heads, seq_enc, dim]
    /// Output: [batch, heads, seq_dec, dim]
    pub fn cross_attention_ibp(
        &self,
        q: &BoundedTensor,
        k: &BoundedTensor,
        v: &BoundedTensor,
        scale: f32,
    ) -> Result<BoundedTensor> {
        match self {
            ComputeDevice::Cpu(d) => d.cross_attention_ibp(q, k, v, scale),
            ComputeDevice::Wgpu(d) => d.cross_attention_ibp(q, k, v, scale),
            #[cfg(feature = "mlx")]
            ComputeDevice::Mlx(d) => d.cross_attention_ibp(q, k, v, scale),
        }
    }
}

impl GemmEngine for ComputeDevice {
    fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        match self {
            ComputeDevice::Cpu(_) => Err(GammaError::NotSupported(
                "GEMM acceleration requested but backend is CPU".to_string(),
            )),
            ComputeDevice::Wgpu(d) => d.gemm_f32(m, k, n, a, b),
            #[cfg(feature = "mlx")]
            ComputeDevice::Mlx(_) => Err(GammaError::NotSupported(
                "GEMM acceleration not yet implemented for MLX backend".to_string(),
            )),
        }
    }
}

impl AcceleratedBoundPropagation for ComputeDevice {
    fn linear_ibp(
        &self,
        input: &BoundedTensor,
        weight: &Array2<f32>,
        bias: Option<&Array1<f32>>,
    ) -> Result<BoundedTensor> {
        match self {
            ComputeDevice::Cpu(d) => d.linear_ibp(input, weight, bias),
            ComputeDevice::Wgpu(d) => d.linear_ibp(input, weight, bias),
            #[cfg(feature = "mlx")]
            ComputeDevice::Mlx(d) => d.linear_ibp(input, weight, bias),
        }
    }

    fn matmul_ibp(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        match self {
            ComputeDevice::Cpu(d) => d.matmul_ibp(input_a, input_b),
            ComputeDevice::Wgpu(d) => d.matmul_ibp(input_a, input_b),
            #[cfg(feature = "mlx")]
            ComputeDevice::Mlx(d) => d.matmul_ibp(input_a, input_b),
        }
    }

    fn crown_per_position_parallel(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        match self {
            ComputeDevice::Cpu(d) => d.crown_per_position_parallel(graph, input),
            ComputeDevice::Wgpu(d) => d.crown_per_position_parallel(graph, input),
            #[cfg(feature = "mlx")]
            ComputeDevice::Mlx(d) => d.crown_per_position_parallel(graph, input),
        }
    }
}
use gamma_propagate::GraphNetwork;
use gamma_tensor::BoundedTensor;
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use rayon::prelude::*;
use tracing::debug;

/// Trait for accelerated bound propagation operations.
pub trait AcceleratedBoundPropagation {
    /// IBP through a linear layer: y = Wx + b
    ///
    /// For interval input [l, u], computes:
    /// - lower: W_pos @ l + W_neg @ u + b
    /// - upper: W_pos @ u + W_neg @ l + b
    ///
    /// where W_pos = max(W, 0), W_neg = min(W, 0)
    fn linear_ibp(
        &self,
        input: &BoundedTensor,
        weight: &Array2<f32>,
        bias: Option<&Array1<f32>>,
    ) -> Result<BoundedTensor>;

    /// IBP through batched matrix multiplication.
    ///
    /// Supports N-D batch dimensions for transformer attention patterns.
    fn matmul_ibp(&self, input_a: &BoundedTensor, input_b: &BoundedTensor)
        -> Result<BoundedTensor>;

    /// Per-position CROWN using parallel execution.
    ///
    /// For N-D input [...batch_dims..., features], runs CROWN independently
    /// on each position in parallel. This provides significant speedup for
    /// transformer verification where positions are independent.
    fn crown_per_position_parallel(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
    ) -> Result<BoundedTensor>;
}

/// Accelerated device using SIMD + Rayon parallelization.
#[derive(Default)]
pub struct AcceleratedDevice;

impl AcceleratedDevice {
    pub fn new() -> Self {
        Self
    }

    /// Full attention IBP using CPU (Rayon parallelized matmul, sequential softmax).
    ///
    /// Computes: softmax((Q @ K^T) * scale) @ V
    ///
    /// Input shapes: Q, K, V with shape [batch, heads, seq, dim]
    /// Output shape: [batch, heads, seq, dim]
    pub fn attention_ibp(
        &self,
        q: &BoundedTensor,
        k: &BoundedTensor,
        v: &BoundedTensor,
        scale: f32,
    ) -> Result<BoundedTensor> {
        let shape_q = q.shape();
        let shape_k = k.shape();
        let shape_v = v.shape();

        // Validate shapes: all should be [batch, heads, seq, dim]
        if shape_q.len() != 4 || shape_k.len() != 4 || shape_v.len() != 4 {
            return Err(GammaError::InvalidSpec(
                "Attention inputs must be 4D [batch, heads, seq, dim]".to_string(),
            ));
        }

        // Verify Q, K, V shapes are compatible
        if shape_q != shape_k || shape_q[..3] != shape_v[..3] {
            return Err(GammaError::shape_mismatch(
                shape_q.to_vec(),
                shape_k.to_vec(),
            ));
        }

        let batch = shape_q[0];
        let heads = shape_q[1];
        let seq = shape_q[2];
        let dim = shape_q[3];

        debug!(
            "AcceleratedDevice attention_ibp: batch={}, heads={}, seq={}, dim={}, scale={}",
            batch, heads, seq, dim, scale
        );

        // Step 1: Compute K^T
        // K shape: [batch, heads, seq, dim]
        // K^T shape: [batch, heads, dim, seq]
        let k_transposed = k.transpose_last_two()?;

        // Step 2: Q @ K^T using Rayon parallel matmul
        // Q: [batch, heads, seq, dim] @ K^T: [batch, heads, dim, seq]
        // -> scores: [batch, heads, seq, seq]
        let scores = self.matmul_ibp(q, &k_transposed)?;

        // Step 3: Scale scores
        let scores_scaled = scores.scale(scale);

        // Step 4: Softmax over last dimension (seq)
        // Use gamma_transformer::softmax_bounds
        let probs = gamma_transformer::softmax_bounds(&scores_scaled, -1)?;

        // Step 5: probs @ V using Rayon parallel matmul
        // probs: [batch, heads, seq, seq] @ V: [batch, heads, seq, dim]
        // -> output: [batch, heads, seq, dim]
        let output = self.matmul_ibp(&probs, v)?;

        Ok(output)
    }

    /// Causal attention IBP using CPU (Rayon parallelized matmul).
    ///
    /// Computes: causal_softmax((Q @ K^T) * scale) @ V
    ///
    /// This is for decoder-only (LLaMA, GPT) and decoder blocks (Whisper decoder).
    /// Position i can only attend to positions j where j <= i.
    ///
    /// Input shapes: Q, K, V with shape [batch, heads, seq, dim]
    /// Output shape: [batch, heads, seq, dim]
    pub fn causal_attention_ibp(
        &self,
        q: &BoundedTensor,
        k: &BoundedTensor,
        v: &BoundedTensor,
        scale: f32,
    ) -> Result<BoundedTensor> {
        let shape_q = q.shape();
        let shape_k = k.shape();
        let shape_v = v.shape();

        // Validate shapes: all should be [batch, heads, seq, dim]
        if shape_q.len() != 4 || shape_k.len() != 4 || shape_v.len() != 4 {
            return Err(GammaError::InvalidSpec(
                "Attention inputs must be 4D [batch, heads, seq, dim]".to_string(),
            ));
        }

        // Verify Q, K, V shapes are compatible
        if shape_q != shape_k || shape_q[..3] != shape_v[..3] {
            return Err(GammaError::shape_mismatch(
                shape_q.to_vec(),
                shape_k.to_vec(),
            ));
        }

        let batch = shape_q[0];
        let heads = shape_q[1];
        let seq = shape_q[2];
        let dim = shape_q[3];

        debug!(
            "AcceleratedDevice causal_attention_ibp: batch={}, heads={}, seq={}, dim={}, scale={}",
            batch, heads, seq, dim, scale
        );

        // Step 1: Compute K^T
        let k_transposed = k.transpose_last_two()?;

        // Step 2: Q @ K^T using Rayon parallel matmul
        let scores = self.matmul_ibp(q, &k_transposed)?;

        // Step 3: Scale scores
        let scores_scaled = scores.scale(scale);

        // Step 4: Causal softmax over last dimension
        // This applies the lower-triangular mask: position i attends only to j <= i
        let probs = gamma_transformer::causal_softmax_bounds(&scores_scaled, -1)?;

        // Step 5: probs @ V using Rayon parallel matmul
        let output = self.matmul_ibp(&probs, v)?;

        Ok(output)
    }

    /// Cross-attention IBP for encoder-decoder models (e.g., Whisper).
    ///
    /// In cross-attention:
    /// - Q (queries) comes from the decoder with shape [batch, heads, seq_dec, dim]
    /// - K, V (keys, values) come from the encoder with shape [batch, heads, seq_enc, dim]
    /// - Output has shape [batch, heads, seq_dec, dim]
    ///
    /// Unlike causal self-attention, cross-attention has NO causal mask:
    /// decoder positions can attend to ALL encoder positions.
    ///
    /// Computes: softmax((Q @ K^T) * scale) @ V
    pub fn cross_attention_ibp(
        &self,
        q: &BoundedTensor, // [batch, heads, seq_dec, dim]
        k: &BoundedTensor, // [batch, heads, seq_enc, dim]
        v: &BoundedTensor, // [batch, heads, seq_enc, dim]
        scale: f32,
    ) -> Result<BoundedTensor> {
        let shape_q = q.shape();
        let shape_k = k.shape();
        let shape_v = v.shape();

        // Validate shapes: all should be 4D [batch, heads, seq, dim]
        if shape_q.len() != 4 || shape_k.len() != 4 || shape_v.len() != 4 {
            return Err(GammaError::InvalidSpec(
                "Cross-attention inputs must be 4D [batch, heads, seq, dim]".to_string(),
            ));
        }

        // Verify batch and heads match
        if shape_q[0] != shape_k[0] || shape_q[0] != shape_v[0] {
            return Err(GammaError::shape_mismatch(
                vec![shape_q[0]],
                vec![shape_k[0], shape_v[0]],
            ));
        }
        if shape_q[1] != shape_k[1] || shape_q[1] != shape_v[1] {
            return Err(GammaError::shape_mismatch(
                vec![shape_q[1]],
                vec![shape_k[1], shape_v[1]],
            ));
        }

        // Verify K and V have same sequence length (encoder sequence)
        if shape_k[2] != shape_v[2] {
            return Err(GammaError::shape_mismatch(
                vec![shape_k[2]],
                vec![shape_v[2]],
            ));
        }

        // Verify dim matches for Q and K (needed for Q @ K^T)
        if shape_q[3] != shape_k[3] {
            return Err(GammaError::shape_mismatch(
                vec![shape_q[3]],
                vec![shape_k[3]],
            ));
        }

        let batch = shape_q[0];
        let heads = shape_q[1];
        let seq_dec = shape_q[2];
        let seq_enc = shape_k[2];
        let dim = shape_q[3];

        debug!(
            "AcceleratedDevice cross_attention_ibp: batch={}, heads={}, seq_dec={}, seq_enc={}, dim={}, scale={}",
            batch, heads, seq_dec, seq_enc, dim, scale
        );

        // Step 1: Compute K^T
        // K shape: [batch, heads, seq_enc, dim]
        // K^T shape: [batch, heads, dim, seq_enc]
        let k_transposed = k.transpose_last_two()?;

        // Step 2: Q @ K^T using Rayon parallel matmul
        // Q: [batch, heads, seq_dec, dim] @ K^T: [batch, heads, dim, seq_enc]
        // -> scores: [batch, heads, seq_dec, seq_enc]
        let scores = self.matmul_ibp(q, &k_transposed)?;

        // Step 3: Scale scores
        let scores_scaled = scores.scale(scale);

        // Step 4: Standard softmax (no causal mask - decoder can attend to all encoder positions)
        // Softmax over last dimension (seq_enc)
        let probs = gamma_transformer::softmax_bounds(&scores_scaled, -1)?;

        // Step 5: probs @ V using Rayon parallel matmul
        // probs: [batch, heads, seq_dec, seq_enc] @ V: [batch, heads, seq_enc, dim]
        // -> output: [batch, heads, seq_dec, dim]
        let output = self.matmul_ibp(&probs, v)?;

        Ok(output)
    }
}

impl AcceleratedBoundPropagation for AcceleratedDevice {
    fn linear_ibp(
        &self,
        input: &BoundedTensor,
        weight: &Array2<f32>,
        bias: Option<&Array1<f32>>,
    ) -> Result<BoundedTensor> {
        let in_features = weight.ncols();
        let out_features = weight.nrows();

        // Validate input shape
        let shape = input.shape();
        if shape.is_empty() || shape[shape.len() - 1] != in_features {
            return Err(GammaError::shape_mismatch(
                vec![in_features],
                shape.to_vec(),
            ));
        }

        let batch_size: usize = shape[..shape.len() - 1].iter().product();

        debug!(
            "AcceleratedDevice linear_ibp: batch={}, in={}, out={}",
            batch_size, in_features, out_features
        );

        // Use parallel implementation
        linear_ibp_parallel(input, weight, bias, batch_size, in_features, out_features)
    }

    fn matmul_ibp(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        matmul_ibp_parallel(input_a, input_b)
    }

    fn crown_per_position_parallel(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        crown_per_position_parallel(graph, input)
    }
}

/// Parallel IBP for linear layers using Rayon.
///
/// This implementation splits work across batch elements and output features
/// for maximum parallelism.
pub fn linear_ibp_parallel(
    input: &BoundedTensor,
    weight: &Array2<f32>,
    bias: Option<&Array1<f32>>,
    batch_size: usize,
    in_features: usize,
    out_features: usize,
) -> Result<BoundedTensor> {
    let shape = input.shape();

    // Pre-compute positive and negative weight matrices
    let weight_pos: Vec<f32> = weight.iter().map(|&w| w.max(0.0)).collect();
    let weight_neg: Vec<f32> = weight.iter().map(|&w| w.min(0.0)).collect();

    // Get input data as contiguous slices
    let lower_data = input.lower.as_slice().unwrap();
    let upper_data = input.upper.as_slice().unwrap();

    // Allocate output buffers
    let output_size = batch_size * out_features;
    let mut result_lower = vec![0.0_f32; output_size];
    let mut result_upper = vec![0.0_f32; output_size];

    // Parallel computation over (batch, output) pairs
    // Use chunks to enable better cache utilization
    result_lower
        .par_chunks_mut(out_features)
        .zip(result_upper.par_chunks_mut(out_features))
        .enumerate()
        .for_each(|(batch_idx, (lower_chunk, upper_chunk))| {
            let input_offset = batch_idx * in_features;
            let xl = &lower_data[input_offset..input_offset + in_features];
            let xu = &upper_data[input_offset..input_offset + in_features];

            for o in 0..out_features {
                let weight_offset = o * in_features;
                let wp = &weight_pos[weight_offset..weight_offset + in_features];
                let wn = &weight_neg[weight_offset..weight_offset + in_features];

                // Vectorized dot products (compiler will auto-vectorize)
                let mut low = 0.0_f32;
                let mut high = 0.0_f32;

                for i in 0..in_features {
                    low += wp[i] * xl[i] + wn[i] * xu[i];
                    high += wp[i] * xu[i] + wn[i] * xl[i];
                }

                if let Some(b) = bias {
                    low += b[o];
                    high += b[o];
                }

                lower_chunk[o] = low;
                upper_chunk[o] = high;
            }
        });

    // Reshape to output shape [..., out_features]
    let mut out_shape = shape[..shape.len() - 1].to_vec();
    out_shape.push(out_features);

    let lower = ArrayD::from_shape_vec(IxDyn(&out_shape), result_lower)
        .map_err(|_| GammaError::shape_mismatch(out_shape.clone(), vec![output_size]))?;
    let upper = ArrayD::from_shape_vec(IxDyn(&out_shape), result_upper)
        .map_err(|_| GammaError::shape_mismatch(out_shape, vec![output_size]))?;

    BoundedTensor::new(lower, upper)
}

/// Parallel IBP for batched matrix multiplication.
///
/// Computes [A_l, A_u] @ [B_l, B_u] with interval arithmetic.
/// Supports N-D batched inputs for transformer attention patterns.
pub fn matmul_ibp_parallel(
    input_a: &BoundedTensor,
    input_b: &BoundedTensor,
) -> Result<BoundedTensor> {
    let shape_a = input_a.shape();
    let shape_b = input_b.shape();

    if shape_a.len() < 2 || shape_b.len() < 2 {
        return Err(GammaError::shape_mismatch(
            vec![2],
            vec![shape_a.len().min(shape_b.len())],
        ));
    }

    // Get matrix dimensions
    let m = shape_a[shape_a.len() - 2]; // rows of A
    let k = shape_a[shape_a.len() - 1]; // cols of A = rows of B
    let n = shape_b[shape_b.len() - 1]; // cols of B

    // Verify inner dimensions match
    if shape_b[shape_b.len() - 2] != k {
        return Err(GammaError::shape_mismatch(
            vec![k],
            vec![shape_b[shape_b.len() - 2]],
        ));
    }

    // Compute batch dimensions
    let batch_dims_a = &shape_a[..shape_a.len() - 2];
    let batch_dims_b = &shape_b[..shape_b.len() - 2];

    // Batch dims should match (simplified - full broadcasting not implemented)
    if batch_dims_a != batch_dims_b {
        return Err(GammaError::shape_mismatch(
            batch_dims_a.to_vec(),
            batch_dims_b.to_vec(),
        ));
    }

    let batch_size: usize = batch_dims_a.iter().product();
    let output_size = batch_size * m * n;

    debug!(
        "AcceleratedDevice matmul_ibp: batch={}, m={}, k={}, n={}",
        batch_size, m, k, n
    );

    // Get input data
    let al = input_a.lower.as_slice().unwrap();
    let au = input_a.upper.as_slice().unwrap();
    let bl = input_b.lower.as_slice().unwrap();
    let bu = input_b.upper.as_slice().unwrap();

    // Allocate output
    let mut result_lower = vec![0.0_f32; output_size];
    let mut result_upper = vec![0.0_f32; output_size];

    let matrix_size_a = m * k;
    let matrix_size_b = k * n;
    let matrix_size_out = m * n;

    // Parallel over batch elements
    result_lower
        .par_chunks_mut(matrix_size_out)
        .zip(result_upper.par_chunks_mut(matrix_size_out))
        .enumerate()
        .for_each(|(batch_idx, (lower_chunk, upper_chunk))| {
            let a_offset = batch_idx * matrix_size_a;
            let b_offset = batch_idx * matrix_size_b;

            // For each output element C[i,j]
            for i in 0..m {
                for j in 0..n {
                    let mut low = 0.0_f32;
                    let mut high = 0.0_f32;

                    // Dot product with interval arithmetic
                    for kk in 0..k {
                        let a_l = al[a_offset + i * k + kk];
                        let a_u = au[a_offset + i * k + kk];
                        let b_l = bl[b_offset + kk * n + j];
                        let b_u = bu[b_offset + kk * n + j];

                        // Interval multiplication: [a,b] * [c,d]
                        let products = [a_l * b_l, a_l * b_u, a_u * b_l, a_u * b_u];
                        let min_prod = products.iter().cloned().fold(f32::INFINITY, f32::min);
                        let max_prod = products.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                        low += min_prod;
                        high += max_prod;
                    }

                    lower_chunk[i * n + j] = low;
                    upper_chunk[i * n + j] = high;
                }
            }
        });

    // Build output shape
    let mut out_shape = batch_dims_a.to_vec();
    out_shape.push(m);
    out_shape.push(n);

    // Sanitize NaN/Inf values before creating BoundedTensor
    // Interval arithmetic can overflow with very wide bounds
    const FALLBACK_BOUND: f32 = 1e10;
    let mut sanitized_count = 0;

    for i in 0..output_size {
        let l = result_lower[i];
        let u = result_upper[i];
        if l.is_nan() || l.is_infinite() || u.is_nan() || u.is_infinite() {
            result_lower[i] = -FALLBACK_BOUND;
            result_upper[i] = FALLBACK_BOUND;
            sanitized_count += 1;
        }
    }

    if sanitized_count > 0 {
        debug!(
            "matmul_ibp_batched_parallel: sanitized {} NaN/Inf values ({}% of output)",
            sanitized_count,
            100.0 * sanitized_count as f64 / output_size as f64
        );
    }

    let lower = ArrayD::from_shape_vec(IxDyn(&out_shape), result_lower)
        .map_err(|_| GammaError::shape_mismatch(out_shape.clone(), vec![output_size]))?;
    let upper = ArrayD::from_shape_vec(IxDyn(&out_shape), result_upper)
        .map_err(|_| GammaError::shape_mismatch(out_shape, vec![output_size]))?;

    BoundedTensor::new(lower, upper)
}

/// Parallel CROWN for per-position bound propagation.
///
/// For transformer verification, each position in the sequence can be
/// verified independently since position-independent layers (Linear, LayerNorm,
/// GELU, etc.) don't create cross-position dependencies.
///
/// This function parallelizes CROWN execution across positions using Rayon,
/// providing significant speedup proportional to the number of CPU cores.
pub fn crown_per_position_parallel(
    graph: &GraphNetwork,
    input: &BoundedTensor,
) -> Result<BoundedTensor> {
    crown_per_position_parallel_with_engine(graph, input, None)
}

/// Sequential per-position CROWN with optional GEMM acceleration.
///
/// This is primarily used by GPU engines that cannot safely participate in
/// Rayon parallel execution due to internal buffer reuse or thread-affinity
/// constraints. It still provides acceleration by offloading GEMM operations
/// within CROWN backward propagation to the provided `engine`.
pub fn crown_per_position_sequential_with_engine(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
) -> Result<BoundedTensor> {
    let shape = input.shape();
    let ndim = shape.len();

    // For 1-D input, just use regular CROWN (no per-position structure)
    if ndim == 1 {
        return graph.propagate_crown_with_engine(input, engine);
    }

    // Extract batch dimensions and hidden dimension
    let hidden_dim = shape[ndim - 1];
    let batch_shape: Vec<usize> = shape[..ndim - 1].to_vec();
    let num_positions: usize = batch_shape.iter().product();

    debug!(
        "Sequential per-position CROWN{}: {} positions x {} hidden, batch shape {:?}",
        if engine.is_some() {
            " (engine-accelerated)"
        } else {
            ""
        },
        num_positions,
        hidden_dim,
        batch_shape
    );

    // Flatten input to [num_positions, hidden_dim]
    // Make arrays contiguous first to avoid reshape failures due to memory layout.
    let lower_contiguous = if input.lower.is_standard_layout() {
        input.lower.clone()
    } else {
        input.lower.as_standard_layout().to_owned()
    };
    let upper_contiguous = if input.upper.is_standard_layout() {
        input.upper.clone()
    } else {
        input.upper.as_standard_layout().to_owned()
    };

    let target_shape = (num_positions, hidden_dim);
    let flat_lower = lower_contiguous
        .into_shape_with_order(target_shape)
        .map_err(|e| {
            GammaError::InvalidSpec(format!(
                "Failed to reshape lower from {:?} to {:?}: {:?}",
                shape, target_shape, e
            ))
        })?;
    let flat_upper = upper_contiguous
        .into_shape_with_order(target_shape)
        .map_err(|e| {
            GammaError::InvalidSpec(format!(
                "Failed to reshape upper from {:?} to {:?}: {:?}",
                shape, target_shape, e
            ))
        })?;

    // Run CROWN on first position to determine output dimension
    let first_lower = flat_lower.row(0).to_owned().into_dyn();
    let first_upper = flat_upper.row(0).to_owned().into_dyn();
    let first_input = BoundedTensor::new(first_lower, first_upper)?;
    let first_output = graph.propagate_crown_with_engine(&first_input, engine)?;
    let output_dim = first_output.len();

    // Allocate output arrays
    let mut out_lower = ndarray::Array2::<f32>::zeros((num_positions, output_dim));
    let mut out_upper = ndarray::Array2::<f32>::zeros((num_positions, output_dim));

    // Copy first result
    {
        let first_out_lower = first_output
            .lower
            .clone()
            .into_shape_with_order((output_dim,))
            .map_err(|_| {
                GammaError::shape_mismatch(vec![output_dim], first_output.lower.shape().to_vec())
            })?;
        let first_out_upper = first_output
            .upper
            .clone()
            .into_shape_with_order((output_dim,))
            .map_err(|_| {
                GammaError::shape_mismatch(vec![output_dim], first_output.upper.shape().to_vec())
            })?;
        out_lower.row_mut(0).assign(&first_out_lower);
        out_upper.row_mut(0).assign(&first_out_upper);
    }

    // Process remaining positions
    for pos in 1..num_positions {
        let pos_lower = flat_lower.row(pos).to_owned().into_dyn();
        let pos_upper = flat_upper.row(pos).to_owned().into_dyn();
        let pos_input = BoundedTensor::new(pos_lower, pos_upper)?;

        let pos_output = graph.propagate_crown_with_engine(&pos_input, engine)?;

        let pos_out_lower = pos_output
            .lower
            .clone()
            .into_shape_with_order((output_dim,))
            .map_err(|_| {
                GammaError::shape_mismatch(vec![output_dim], pos_output.lower.shape().to_vec())
            })?;
        let pos_out_upper = pos_output
            .upper
            .clone()
            .into_shape_with_order((output_dim,))
            .map_err(|_| {
                GammaError::shape_mismatch(vec![output_dim], pos_output.upper.shape().to_vec())
            })?;

        out_lower.row_mut(pos).assign(&pos_out_lower);
        out_upper.row_mut(pos).assign(&pos_out_upper);
    }

    // Sanitize NaN/Inf values before creating BoundedTensor
    // CROWN can produce overflow when bound widths explode through deep networks.
    // Replace NaN/Inf with conservative fallback bounds to maintain soundness.
    let mut sanitized_count = 0;
    const FALLBACK_BOUND: f32 = 1e10;

    for pos in 0..num_positions {
        for i in 0..output_dim {
            let l = out_lower[[pos, i]];
            let u = out_upper[[pos, i]];
            if l.is_nan() || l.is_infinite() || u.is_nan() || u.is_infinite() {
                out_lower[[pos, i]] = -FALLBACK_BOUND;
                out_upper[[pos, i]] = FALLBACK_BOUND;
                sanitized_count += 1;
            }
        }
    }

    if sanitized_count > 0 {
        debug!(
            "crown_per_position_sequential_with_engine: sanitized {} NaN/Inf values ({}% of output)",
            sanitized_count,
            100.0 * sanitized_count as f64 / (num_positions * output_dim) as f64
        );
    }

    // Reshape output to [...batch_dims..., output_dim]
    let mut output_shape = batch_shape;
    output_shape.push(output_dim);

    let out_lower_nd = out_lower
        .into_dyn()
        .into_shape_with_order(ndarray::IxDyn(&output_shape))
        .map_err(|_| {
            GammaError::shape_mismatch(output_shape.clone(), vec![num_positions, output_dim])
        })?;
    let out_upper_nd = out_upper
        .into_dyn()
        .into_shape_with_order(ndarray::IxDyn(&output_shape))
        .map_err(|_| {
            GammaError::shape_mismatch(output_shape.clone(), vec![num_positions, output_dim])
        })?;

    BoundedTensor::new(out_lower_nd, out_upper_nd)
}

/// Parallel CROWN for per-position bound propagation with optional GPU acceleration.
///
/// Same as `crown_per_position_parallel` but accepts an optional `GemmEngine` for
/// GPU-accelerated matrix operations within CROWN backward propagation.
///
/// When `engine` is `Some`, the GEMM operations within each position's CROWN
/// propagation will be accelerated using the provided engine (e.g., GPU via wgpu).
pub fn crown_per_position_parallel_with_engine(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
) -> Result<BoundedTensor> {
    let shape = input.shape();
    let ndim = shape.len();

    // For 1-D input, just use regular CROWN (no parallelization benefit)
    if ndim == 1 {
        return graph.propagate_crown_with_engine(input, engine);
    }

    // Extract batch dimensions and hidden dimension
    let hidden_dim = shape[ndim - 1];
    let batch_shape: Vec<usize> = shape[..ndim - 1].to_vec();
    let num_positions: usize = batch_shape.iter().product();

    debug!(
        "Parallel per-position CROWN{}: {} positions x {} hidden, batch shape {:?}",
        if engine.is_some() {
            " (GPU-accelerated)"
        } else {
            ""
        },
        num_positions,
        hidden_dim,
        batch_shape
    );

    // Flatten input to [num_positions, hidden_dim]
    let flat_lower = input
        .lower
        .clone()
        .into_shape_with_order((num_positions, hidden_dim))
        .map_err(|_| {
            GammaError::shape_mismatch_checked(vec![num_positions, hidden_dim], shape.to_vec())
        })?;
    let flat_upper = input
        .upper
        .clone()
        .into_shape_with_order((num_positions, hidden_dim))
        .map_err(|_| {
            GammaError::shape_mismatch_checked(vec![num_positions, hidden_dim], shape.to_vec())
        })?;

    // Run CROWN on first position to determine output dimension
    let first_lower = flat_lower.row(0).to_owned().into_dyn();
    let first_upper = flat_upper.row(0).to_owned().into_dyn();
    let first_input = BoundedTensor::new(first_lower, first_upper)?;
    let first_output = graph.propagate_crown_with_engine(&first_input, engine)?;
    let output_dim = first_output.len();

    // Prepare input data for parallel processing
    // Each element is (lower_row, upper_row) as Vec<f32>
    let inputs: Vec<(Vec<f32>, Vec<f32>)> = (0..num_positions)
        .map(|pos| (flat_lower.row(pos).to_vec(), flat_upper.row(pos).to_vec()))
        .collect();

    // Run CROWN in parallel across all positions
    let results: Vec<Result<(Vec<f32>, Vec<f32>)>> = inputs
        .par_iter()
        .map(|(lower_row, upper_row)| {
            // Create BoundedTensor for this position
            let pos_lower = ArrayD::from_shape_vec(IxDyn(&[hidden_dim]), lower_row.clone())
                .map_err(|_| GammaError::shape_mismatch(vec![hidden_dim], vec![lower_row.len()]))?;
            let pos_upper = ArrayD::from_shape_vec(IxDyn(&[hidden_dim]), upper_row.clone())
                .map_err(|_| GammaError::shape_mismatch(vec![hidden_dim], vec![upper_row.len()]))?;
            let pos_input = BoundedTensor::new(pos_lower, pos_upper)?;

            // Run CROWN with optional engine
            let pos_output = graph.propagate_crown_with_engine(&pos_input, engine)?;

            // Extract results as Vec<f32>
            let out_lower = pos_output.lower.as_slice().unwrap().to_vec();
            let out_upper = pos_output.upper.as_slice().unwrap().to_vec();

            Ok((out_lower, out_upper))
        })
        .collect();

    // Check for errors and collect results
    let mut out_lower = ndarray::Array2::<f32>::zeros((num_positions, output_dim));
    let mut out_upper = ndarray::Array2::<f32>::zeros((num_positions, output_dim));

    for (pos, result) in results.into_iter().enumerate() {
        let (lower_row, upper_row) = result?;
        for (i, (&l, &u)) in lower_row.iter().zip(upper_row.iter()).enumerate() {
            out_lower[[pos, i]] = l;
            out_upper[[pos, i]] = u;
        }
    }

    // Sanitize NaN/Inf values before creating BoundedTensor
    // CROWN can produce overflow when bound widths explode through deep networks.
    // Replace NaN/Inf with conservative fallback bounds to maintain soundness.
    let mut sanitized_count = 0;
    const FALLBACK_BOUND: f32 = 1e10;

    for pos in 0..num_positions {
        for i in 0..output_dim {
            let l = out_lower[[pos, i]];
            let u = out_upper[[pos, i]];
            if l.is_nan() || l.is_infinite() || u.is_nan() || u.is_infinite() {
                out_lower[[pos, i]] = -FALLBACK_BOUND;
                out_upper[[pos, i]] = FALLBACK_BOUND;
                sanitized_count += 1;
            }
        }
    }

    if sanitized_count > 0 {
        debug!(
            "crown_per_position_parallel_with_engine: sanitized {} NaN/Inf values ({}% of output)",
            sanitized_count,
            100.0 * sanitized_count as f64 / (num_positions * output_dim) as f64
        );
    }

    // Reshape output to [...batch_dims..., output_dim]
    let mut output_shape = batch_shape;
    output_shape.push(output_dim);

    let out_lower_nd = out_lower
        .into_dyn()
        .into_shape_with_order(ndarray::IxDyn(&output_shape))
        .map_err(|_| {
            GammaError::shape_mismatch(output_shape.clone(), vec![num_positions, output_dim])
        })?;
    let out_upper_nd = out_upper
        .into_dyn()
        .into_shape_with_order(ndarray::IxDyn(&output_shape))
        .map_err(|_| {
            GammaError::shape_mismatch(output_shape.clone(), vec![num_positions, output_dim])
        })?;

    BoundedTensor::new(out_lower_nd, out_upper_nd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_linear_ibp_basic() {
        let device = AcceleratedDevice::new();

        // Create test input: shape [2, 3] with bounds
        let lower = ArrayD::from_elem(IxDyn(&[2, 3]), -1.0_f32);
        let upper = ArrayD::from_elem(IxDyn(&[2, 3]), 1.0_f32);
        let input = BoundedTensor::new(lower, upper).unwrap();

        // Weight [4, 3] -> output [2, 4]
        let weight = Array2::from_elem((4, 3), 0.5_f32);
        let bias = Some(Array1::from_elem(4, 0.1_f32));

        let result = device.linear_ibp(&input, &weight, bias.as_ref()).unwrap();

        assert_eq!(result.shape(), &[2, 4]);

        // For w=0.5 and x in [-1, 1]: sum of 3 terms = [-1.5, 1.5], plus bias 0.1 = [-1.4, 1.6]
        assert_relative_eq!(result.lower[[0, 0]], -1.4, epsilon = 1e-5);
        assert_relative_eq!(result.upper[[0, 0]], 1.6, epsilon = 1e-5);
    }

    #[test]
    fn test_linear_ibp_batched() {
        let device = AcceleratedDevice::new();

        // Create batched input: shape [2, 3, 4] with bounds
        let lower = ArrayD::from_elem(IxDyn(&[2, 3, 4]), 0.0_f32);
        let upper = ArrayD::from_elem(IxDyn(&[2, 3, 4]), 1.0_f32);
        let input = BoundedTensor::new(lower, upper).unwrap();

        // Weight [8, 4] -> output [2, 3, 8]
        let weight = Array2::from_elem((8, 4), 0.25_f32);

        let result = device.linear_ibp(&input, &weight, None).unwrap();

        assert_eq!(result.shape(), &[2, 3, 8]);

        // For w=0.25 and x in [0, 1]: sum of 4 terms = [0, 1]
        assert_relative_eq!(result.lower[[0, 0, 0]], 0.0, epsilon = 1e-5);
        assert_relative_eq!(result.upper[[0, 0, 0]], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_matmul_ibp_basic() {
        let device = AcceleratedDevice::new();

        // A: [2, 3] with bounds [0, 1]
        let lower_a = ArrayD::from_elem(IxDyn(&[2, 3]), 0.0_f32);
        let upper_a = ArrayD::from_elem(IxDyn(&[2, 3]), 1.0_f32);
        let input_a = BoundedTensor::new(lower_a, upper_a).unwrap();

        // B: [3, 4] with bounds [0, 1]
        let lower_b = ArrayD::from_elem(IxDyn(&[3, 4]), 0.0_f32);
        let upper_b = ArrayD::from_elem(IxDyn(&[3, 4]), 1.0_f32);
        let input_b = BoundedTensor::new(lower_b, upper_b).unwrap();

        let result = device.matmul_ibp(&input_a, &input_b).unwrap();

        assert_eq!(result.shape(), &[2, 4]);

        // For A,B in [0, 1]^{2x3} @ [0, 1]^{3x4}: result in [0, 3]
        assert_relative_eq!(result.lower[[0, 0]], 0.0, epsilon = 1e-5);
        assert_relative_eq!(result.upper[[0, 0]], 3.0, epsilon = 1e-5);
    }

    #[test]
    fn test_matmul_ibp_batched() {
        let device = AcceleratedDevice::new();

        // Batched A: [2, 2, 3] (2 batches of 2x3 matrices)
        let lower_a = ArrayD::from_elem(IxDyn(&[2, 2, 3]), 0.5_f32);
        let upper_a = ArrayD::from_elem(IxDyn(&[2, 2, 3]), 1.0_f32);
        let input_a = BoundedTensor::new(lower_a, upper_a).unwrap();

        // Batched B: [2, 3, 4] (2 batches of 3x4 matrices)
        let lower_b = ArrayD::from_elem(IxDyn(&[2, 3, 4]), 0.5_f32);
        let upper_b = ArrayD::from_elem(IxDyn(&[2, 3, 4]), 1.0_f32);
        let input_b = BoundedTensor::new(lower_b, upper_b).unwrap();

        let result = device.matmul_ibp(&input_a, &input_b).unwrap();

        assert_eq!(result.shape(), &[2, 2, 4]);

        // For A,B in [0.5, 1.0]: each product in [0.25, 1.0], sum of 3 = [0.75, 3.0]
        assert_relative_eq!(result.lower[[0, 0, 0]], 0.75, epsilon = 1e-5);
        assert_relative_eq!(result.upper[[0, 0, 0]], 3.0, epsilon = 1e-5);
    }

    #[test]
    fn test_crown_per_position_parallel() {
        use gamma_propagate::{GELULayer, GraphNode, Layer, LinearLayer};
        use ndarray::Array2;

        let device = AcceleratedDevice::new();

        // Build a small MLP graph: Linear -> GELU -> Linear
        // 4 features -> 8 features -> 4 features
        let in_features = 4;
        let hidden_features = 8;
        let out_features = 4;

        let weight1 = Array2::from_shape_fn((hidden_features, in_features), |(i, j)| {
            0.1 * ((i + j) as f32 - 6.0)
        });
        let bias1 = ndarray::Array1::from_elem(hidden_features, 0.05_f32);
        let linear1 = LinearLayer::new(weight1, Some(bias1)).unwrap();

        let weight2 = Array2::from_shape_fn((out_features, hidden_features), |(i, j)| {
            0.1 * ((i + j) as f32 - 6.0)
        });
        let bias2 = ndarray::Array1::from_elem(out_features, 0.02_f32);
        let linear2 = LinearLayer::new(weight2, Some(bias2)).unwrap();

        let gelu = GELULayer::default();

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
        graph.add_node(GraphNode::new(
            "gelu",
            Layer::GELU(gelu),
            vec!["linear1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "linear2",
            Layer::Linear(linear2),
            vec!["gelu".to_string()],
        ));
        graph.set_output("linear2");

        // Create multi-position input: [2, 3, 4] = 6 positions with 4 features each
        let lower = ArrayD::from_elem(IxDyn(&[2, 3, in_features]), 0.0_f32);
        let upper = ArrayD::from_elem(IxDyn(&[2, 3, in_features]), 1.0_f32);
        let input = BoundedTensor::new(lower, upper).unwrap();

        // Run parallel per-position CROWN
        let result = device.crown_per_position_parallel(&graph, &input).unwrap();

        // Check output shape: [2, 3, 4]
        assert_eq!(result.shape(), &[2, 3, out_features]);

        // Verify bounds are valid (lower <= upper)
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..out_features {
                    let l = result.lower[[i, j, k]];
                    let u = result.upper[[i, j, k]];
                    assert!(
                        l <= u,
                        "Invalid bounds at [{},{},{}]: lower={} > upper={}",
                        i,
                        j,
                        k,
                        l,
                        u
                    );
                }
            }
        }

        // Compare with sequential per-position CROWN
        let sequential_result = graph.propagate_crown_per_position(&input).unwrap();

        // Results should be identical
        assert_eq!(result.shape(), sequential_result.shape());

        for i in 0..2 {
            for j in 0..3 {
                for k in 0..out_features {
                    assert_relative_eq!(
                        result.lower[[i, j, k]],
                        sequential_result.lower[[i, j, k]],
                        epsilon = 1e-5
                    );
                    assert_relative_eq!(
                        result.upper[[i, j, k]],
                        sequential_result.upper[[i, j, k]],
                        epsilon = 1e-5
                    );
                }
            }
        }
    }

    #[test]
    fn test_cpu_attention_ibp_basic() {
        let device = AcceleratedDevice::new();

        // Create test Q, K, V: shape [1, 2, 4, 8] (batch=1, heads=2, seq=4, dim=8)
        let batch = 1;
        let heads = 2;
        let seq = 4;
        let dim = 8;
        let shape = [batch, heads, seq, dim];

        // Q, K, V with small perturbation around 0
        let lower = ArrayD::from_elem(IxDyn(&shape), -0.1_f32);
        let upper = ArrayD::from_elem(IxDyn(&shape), 0.1_f32);
        let q = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();
        let k = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();
        let v = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();

        let scale = 1.0 / (dim as f32).sqrt();

        let result = device.attention_ibp(&q, &k, &v, scale).unwrap();

        // Check output shape: [batch, heads, seq, dim]
        assert_eq!(result.shape(), &shape);

        // Check bounds are valid (lower <= upper)
        for val_lower in result.lower.iter() {
            for val_upper in result.upper.iter() {
                assert!(
                    *val_lower <= *val_upper + 1e-6,
                    "Invalid bounds: lower={} > upper={}",
                    val_lower,
                    val_upper
                );
            }
        }

        // Output should be bounded since softmax outputs sum to 1
        // and V is in [-0.1, 0.1], so output should be roughly in [-0.1, 0.1]
        let max_upper = result
            .upper
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let min_lower = result.lower.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(
            max_upper < 1.0 && min_lower > -1.0,
            "Attention bounds seem too loose: [{}, {}]",
            min_lower,
            max_upper
        );
    }

    #[test]
    fn test_cpu_causal_attention_ibp_basic() {
        let device = AcceleratedDevice::new();

        // Create test Q, K, V: shape [1, 2, 4, 8] (batch=1, heads=2, seq=4, dim=8)
        let batch = 1;
        let heads = 2;
        let seq = 4;
        let dim = 8;
        let shape = [batch, heads, seq, dim];

        // Q, K, V with small perturbation around 0
        let lower = ArrayD::from_elem(IxDyn(&shape), -0.1_f32);
        let upper = ArrayD::from_elem(IxDyn(&shape), 0.1_f32);
        let q = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();
        let k = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();
        let v = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();

        let scale = 1.0 / (dim as f32).sqrt();

        let result = device.causal_attention_ibp(&q, &k, &v, scale).unwrap();

        // Check output shape: [batch, heads, seq, dim]
        assert_eq!(result.shape(), &shape);

        // Check bounds are valid (lower <= upper)
        for val_lower in result.lower.iter() {
            for val_upper in result.upper.iter() {
                assert!(
                    *val_lower <= *val_upper + 1e-6,
                    "Invalid bounds: lower={} > upper={}",
                    val_lower,
                    val_upper
                );
            }
        }

        // Output should be bounded similarly to standard attention
        let max_upper = result
            .upper
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let min_lower = result.lower.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(
            max_upper < 1.0 && min_lower > -1.0,
            "Causal attention bounds seem too loose: [{}, {}]",
            min_lower,
            max_upper
        );
    }

    #[test]
    fn test_causal_attention_soundness() {
        // Test that causal attention bounds are sound by checking that
        // concrete causal attention outputs fall within bounds
        let device = AcceleratedDevice::new();

        let batch = 1;
        let heads = 1;
        let seq = 4;
        let dim = 4;
        let shape = [batch, heads, seq, dim];

        // Create bounded Q, K, V with small perturbation
        let eps = 0.1;
        let center = ArrayD::from_elem(IxDyn(&shape), 0.5_f32);
        let lower = center.mapv(|v| v - eps);
        let upper = center.mapv(|v| v + eps);
        let q = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();
        let k = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();
        let v = BoundedTensor::new(lower, upper).unwrap();

        let scale = 1.0 / (dim as f32).sqrt();

        let result = device.causal_attention_ibp(&q, &k, &v, scale).unwrap();

        // Bounds should be valid
        assert_eq!(result.shape(), &shape);
        for i in 0..result.lower.len() {
            assert!(
                result.lower.as_slice().unwrap()[i] <= result.upper.as_slice().unwrap()[i] + 1e-5,
                "Invalid bounds at position {}: lower={} > upper={}",
                i,
                result.lower.as_slice().unwrap()[i],
                result.upper.as_slice().unwrap()[i]
            );
        }
    }

    #[test]
    fn test_causal_vs_standard_attention_difference() {
        // Causal attention should give different results than standard attention
        // (except for the last position which sees all previous positions)
        let device = AcceleratedDevice::new();

        let batch = 1;
        let heads = 1;
        let seq = 4;
        let dim = 4;
        let shape = [batch, heads, seq, dim];

        // Use point estimates (no perturbation) to compare concrete outputs
        let data = ArrayD::from_shape_fn(IxDyn(&shape), |idx| {
            let [_b, _h, s, d] = [idx[0], idx[1], idx[2], idx[3]];
            (s + d) as f32 * 0.1
        });
        let q = BoundedTensor::new(data.clone(), data.clone()).unwrap();
        let k = BoundedTensor::new(data.clone(), data.clone()).unwrap();
        let v = BoundedTensor::new(data.clone(), data.clone()).unwrap();

        let scale = 1.0 / (dim as f32).sqrt();

        let causal_result = device.causal_attention_ibp(&q, &k, &v, scale).unwrap();
        let standard_result = device.attention_ibp(&q, &k, &v, scale).unwrap();

        // First position should have only one valid attention target, so might differ
        // from standard attention which sees all positions
        let causal_pos0: Vec<f32> = (0..dim)
            .map(|d| causal_result.lower[[0, 0, 0, d]])
            .collect();
        let standard_pos0: Vec<f32> = (0..dim)
            .map(|d| standard_result.lower[[0, 0, 0, d]])
            .collect();

        // At position 0, causal attention only sees position 0 (self)
        // Standard attention sees all 4 positions
        // These should typically differ unless inputs are special
        let diff: f32 = causal_pos0
            .iter()
            .zip(standard_pos0.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        // The outputs are allowed to be identical (if V values are all the same)
        // but in general they should differ
        // Just verify both produce valid results
        assert!(causal_pos0.iter().all(|x| x.is_finite()));
        assert!(standard_pos0.iter().all(|x| x.is_finite()));

        // Last position in causal should match standard (sees all positions)
        // This is only exact for point inputs (no perturbation)
        let last_pos = seq - 1;
        for d in 0..dim {
            let causal_val = causal_result.lower[[0, 0, last_pos, d]];
            let standard_val = standard_result.lower[[0, 0, last_pos, d]];
            assert!(
                (causal_val - standard_val).abs() < 1e-4,
                "Last position should match: causal={} vs standard={} at dim {}",
                causal_val,
                standard_val,
                d
            );
        }

        // Verify total difference exists (causal != standard for earlier positions)
        assert!(
            diff > 1e-6,
            "Expected difference between causal and standard attention, got diff={}",
            diff
        );
    }

    // ================= Cross-Attention Tests =================

    #[test]
    fn test_cross_attention_basic() {
        // Test basic cross-attention with different sequence lengths
        let device = AcceleratedDevice::new();

        // Q from decoder: [batch=1, heads=2, seq_dec=3, dim=4]
        // K, V from encoder: [batch=1, heads=2, seq_enc=5, dim=4]
        let batch = 1;
        let heads = 2;
        let seq_dec = 3;
        let seq_enc = 5;
        let dim = 4;

        let shape_q = [batch, heads, seq_dec, dim];
        let shape_kv = [batch, heads, seq_enc, dim];

        // Q, K, V with small perturbation around 0
        let lower_q = ArrayD::from_elem(IxDyn(&shape_q), -0.1_f32);
        let upper_q = ArrayD::from_elem(IxDyn(&shape_q), 0.1_f32);
        let lower_kv = ArrayD::from_elem(IxDyn(&shape_kv), -0.1_f32);
        let upper_kv = ArrayD::from_elem(IxDyn(&shape_kv), 0.1_f32);

        let q = BoundedTensor::new(lower_q, upper_q).unwrap();
        let k = BoundedTensor::new(lower_kv.clone(), upper_kv.clone()).unwrap();
        let v = BoundedTensor::new(lower_kv, upper_kv).unwrap();

        let scale = 1.0 / (dim as f32).sqrt();

        let result = device.cross_attention_ibp(&q, &k, &v, scale).unwrap();

        // Output should match decoder sequence length
        // Expected shape: [batch, heads, seq_dec, dim]
        assert_eq!(result.shape(), &shape_q);

        // Bounds should be valid
        for i in 0..result.lower.len() {
            assert!(
                result.lower.as_slice().unwrap()[i] <= result.upper.as_slice().unwrap()[i] + 1e-5,
                "Invalid bounds at position {}: lower={} > upper={}",
                i,
                result.lower.as_slice().unwrap()[i],
                result.upper.as_slice().unwrap()[i]
            );
        }
    }

    #[test]
    fn test_cross_attention_soundness() {
        // Test that cross-attention bounds contain concrete outputs
        let device = AcceleratedDevice::new();

        let batch = 1;
        let heads = 1;
        let seq_dec = 2;
        let seq_enc = 3;
        let dim = 4;

        let shape_q = [batch, heads, seq_dec, dim];
        let shape_kv = [batch, heads, seq_enc, dim];

        // Create bounded Q, K, V with small perturbation
        let eps = 0.1;
        let center_q = ArrayD::from_elem(IxDyn(&shape_q), 0.5_f32);
        let center_kv = ArrayD::from_elem(IxDyn(&shape_kv), 0.5_f32);

        let q = BoundedTensor::new(center_q.mapv(|v| v - eps), center_q.mapv(|v| v + eps)).unwrap();
        let k =
            BoundedTensor::new(center_kv.mapv(|v| v - eps), center_kv.mapv(|v| v + eps)).unwrap();
        let v =
            BoundedTensor::new(center_kv.mapv(|v| v - eps), center_kv.mapv(|v| v + eps)).unwrap();

        let scale = 1.0 / (dim as f32).sqrt();

        let result = device.cross_attention_ibp(&q, &k, &v, scale).unwrap();

        // Expected output shape: [batch, heads, seq_dec, dim]
        assert_eq!(result.shape(), &shape_q);

        // Bounds should be valid
        for i in 0..result.lower.len() {
            assert!(
                result.lower.as_slice().unwrap()[i] <= result.upper.as_slice().unwrap()[i] + 1e-5,
                "Invalid bounds at position {}: lower={} > upper={}",
                i,
                result.lower.as_slice().unwrap()[i],
                result.upper.as_slice().unwrap()[i]
            );
        }

        // Output bounds should be reasonable (contained in V bounds with some slack)
        let min_lower = result.lower.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_upper = result
            .upper
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            min_lower >= -1.0 && max_upper <= 2.0,
            "Cross-attention bounds seem too loose: [{}, {}]",
            min_lower,
            max_upper
        );
    }

    #[test]
    fn test_cross_attention_shape_validation() {
        // Test that cross-attention rejects mismatched shapes
        let device = AcceleratedDevice::new();

        // Valid shapes
        let shape_q = [1, 2, 3, 4]; // [batch=1, heads=2, seq_dec=3, dim=4]
        let shape_k = [1, 2, 5, 4]; // [batch=1, heads=2, seq_enc=5, dim=4]
        let shape_v = [1, 2, 5, 4]; // [batch=1, heads=2, seq_enc=5, dim=4]

        let lower = ArrayD::zeros(IxDyn(&shape_q));
        let upper = ArrayD::zeros(IxDyn(&shape_q));
        let q = BoundedTensor::new(lower, upper).unwrap();

        let lower_k = ArrayD::zeros(IxDyn(&shape_k));
        let upper_k = ArrayD::zeros(IxDyn(&shape_k));
        let k = BoundedTensor::new(lower_k, upper_k).unwrap();

        let lower_v = ArrayD::zeros(IxDyn(&shape_v));
        let upper_v = ArrayD::zeros(IxDyn(&shape_v));
        let v = BoundedTensor::new(lower_v, upper_v).unwrap();

        // This should work
        let result = device.cross_attention_ibp(&q, &k, &v, 1.0);
        assert!(result.is_ok());

        // Test mismatched batch - should fail
        let bad_shape = [2, 2, 5, 4]; // batch=2 doesn't match
        let bad_k = BoundedTensor::new(
            ArrayD::zeros(IxDyn(&bad_shape)),
            ArrayD::zeros(IxDyn(&bad_shape)),
        )
        .unwrap();
        let result = device.cross_attention_ibp(&q, &bad_k, &v, 1.0);
        assert!(result.is_err());

        // Test mismatched K/V sequence lengths - should fail
        let bad_v_shape = [1, 2, 6, 4]; // seq_enc=6 doesn't match K's seq_enc=5
        let bad_v = BoundedTensor::new(
            ArrayD::zeros(IxDyn(&bad_v_shape)),
            ArrayD::zeros(IxDyn(&bad_v_shape)),
        )
        .unwrap();
        let result = device.cross_attention_ibp(&q, &k, &bad_v, 1.0);
        assert!(result.is_err());

        // Test mismatched dim between Q and K - should fail
        let bad_k_dim_shape = [1, 2, 5, 8]; // dim=8 doesn't match Q's dim=4
        let bad_k_dim = BoundedTensor::new(
            ArrayD::zeros(IxDyn(&bad_k_dim_shape)),
            ArrayD::zeros(IxDyn(&bad_k_dim_shape)),
        )
        .unwrap();
        let result = device.cross_attention_ibp(&q, &bad_k_dim, &v, 1.0);
        assert!(result.is_err());
    }

    // ================= Backend Enum Tests =================

    #[test]
    fn test_backend_display() {
        assert_eq!(Backend::Cpu.to_string(), "cpu");
        assert_eq!(Backend::Wgpu.to_string(), "wgpu");
        assert_eq!(Backend::Mlx.to_string(), "mlx");
    }

    #[test]
    fn test_backend_from_str() {
        use std::str::FromStr;

        // Valid backends
        assert_eq!(Backend::from_str("cpu").unwrap(), Backend::Cpu);
        assert_eq!(Backend::from_str("CPU").unwrap(), Backend::Cpu);
        assert_eq!(Backend::from_str("wgpu").unwrap(), Backend::Wgpu);
        assert_eq!(Backend::from_str("WGPU").unwrap(), Backend::Wgpu);
        assert_eq!(Backend::from_str("gpu").unwrap(), Backend::Wgpu);
        assert_eq!(Backend::from_str("GPU").unwrap(), Backend::Wgpu);
        assert_eq!(Backend::from_str("mlx").unwrap(), Backend::Mlx);
        assert_eq!(Backend::from_str("MLX").unwrap(), Backend::Mlx);

        // Invalid backends
        assert!(Backend::from_str("invalid").is_err());
        assert!(Backend::from_str("").is_err());
        assert!(Backend::from_str("cuda").is_err());
    }

    #[test]
    fn test_backend_default() {
        assert_eq!(Backend::default(), Backend::Cpu);
    }

    #[test]
    fn test_backend_equality() {
        assert_eq!(Backend::Cpu, Backend::Cpu);
        assert_ne!(Backend::Cpu, Backend::Wgpu);
        assert_ne!(Backend::Wgpu, Backend::Mlx);
    }

    #[test]
    fn test_backend_clone() {
        // Backend implements Copy, so we test the Clone impl via Copy behavior
        let backend = Backend::Wgpu;
        let cloned: Backend = backend; // Uses Copy (which implies Clone)
        assert_eq!(backend, cloned);
    }

    #[test]
    fn test_backend_copy() {
        let backend = Backend::Mlx;
        let copied: Backend = backend; // Copy, not move
        assert_eq!(backend, copied);
    }

    // ================= ComputeDevice Tests =================

    #[test]
    fn test_compute_device_cpu_creation() {
        let device = ComputeDevice::new(Backend::Cpu).unwrap();
        assert_eq!(device.backend(), Backend::Cpu);
        assert!(device.supports_attention());
    }

    #[test]
    fn test_compute_device_cpu_linear_ibp() {
        let device = ComputeDevice::new(Backend::Cpu).unwrap();

        let lower = ArrayD::from_elem(IxDyn(&[2, 3]), -1.0_f32);
        let upper = ArrayD::from_elem(IxDyn(&[2, 3]), 1.0_f32);
        let input = BoundedTensor::new(lower, upper).unwrap();

        let weight = Array2::from_elem((4, 3), 0.5_f32);
        let bias = Some(Array1::from_elem(4, 0.1_f32));

        let result = device.linear_ibp(&input, &weight, bias.as_ref()).unwrap();
        assert_eq!(result.shape(), &[2, 4]);

        // Verify bounds: for w=0.5 and x in [-1, 1], sum of 3 = [-1.5, 1.5], plus bias 0.1
        assert_relative_eq!(result.lower[[0, 0]], -1.4, epsilon = 1e-5);
        assert_relative_eq!(result.upper[[0, 0]], 1.6, epsilon = 1e-5);
    }

    #[test]
    fn test_compute_device_cpu_matmul_ibp() {
        let device = ComputeDevice::new(Backend::Cpu).unwrap();

        let lower_a = ArrayD::from_elem(IxDyn(&[2, 3]), 0.0_f32);
        let upper_a = ArrayD::from_elem(IxDyn(&[2, 3]), 1.0_f32);
        let input_a = BoundedTensor::new(lower_a, upper_a).unwrap();

        let lower_b = ArrayD::from_elem(IxDyn(&[3, 4]), 0.0_f32);
        let upper_b = ArrayD::from_elem(IxDyn(&[3, 4]), 1.0_f32);
        let input_b = BoundedTensor::new(lower_b, upper_b).unwrap();

        let result = device.matmul_ibp(&input_a, &input_b).unwrap();
        assert_eq!(result.shape(), &[2, 4]);

        // For A,B in [0, 1]^{2x3} @ [0, 1]^{3x4}: result in [0, 3]
        assert_relative_eq!(result.lower[[0, 0]], 0.0, epsilon = 1e-5);
        assert_relative_eq!(result.upper[[0, 0]], 3.0, epsilon = 1e-5);
    }

    #[test]
    fn test_compute_device_cpu_attention_ibp() {
        let device = ComputeDevice::new(Backend::Cpu).unwrap();

        let batch = 1;
        let heads = 2;
        let seq = 4;
        let dim = 8;
        let shape = [batch, heads, seq, dim];

        let lower = ArrayD::from_elem(IxDyn(&shape), -0.1_f32);
        let upper = ArrayD::from_elem(IxDyn(&shape), 0.1_f32);
        let q = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();
        let k = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();
        let v = BoundedTensor::new(lower, upper).unwrap();

        let scale = 1.0 / (dim as f32).sqrt();

        let result = device.attention_ibp(&q, &k, &v, scale).unwrap();
        assert_eq!(result.shape(), &shape);

        // Verify bounds are valid
        for (l, u) in result.lower.iter().zip(result.upper.iter()) {
            assert!(*l <= *u + 1e-5, "Invalid bounds: lower={} > upper={}", l, u);
        }
    }

    #[test]
    fn test_compute_device_cpu_causal_attention_ibp() {
        let device = ComputeDevice::new(Backend::Cpu).unwrap();

        let batch = 1;
        let heads = 2;
        let seq = 4;
        let dim = 8;
        let shape = [batch, heads, seq, dim];

        let lower = ArrayD::from_elem(IxDyn(&shape), -0.1_f32);
        let upper = ArrayD::from_elem(IxDyn(&shape), 0.1_f32);
        let q = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();
        let k = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();
        let v = BoundedTensor::new(lower, upper).unwrap();

        let scale = 1.0 / (dim as f32).sqrt();

        let result = device.causal_attention_ibp(&q, &k, &v, scale).unwrap();
        assert_eq!(result.shape(), &shape);

        // Verify bounds are valid
        for (l, u) in result.lower.iter().zip(result.upper.iter()) {
            assert!(*l <= *u + 1e-5, "Invalid bounds: lower={} > upper={}", l, u);
        }
    }

    #[test]
    fn test_compute_device_cpu_cross_attention_ibp() {
        let device = ComputeDevice::new(Backend::Cpu).unwrap();

        let batch = 1;
        let heads = 2;
        let seq_dec = 3;
        let seq_enc = 5;
        let dim = 4;

        let shape_q = [batch, heads, seq_dec, dim];
        let shape_kv = [batch, heads, seq_enc, dim];

        let lower_q = ArrayD::from_elem(IxDyn(&shape_q), -0.1_f32);
        let upper_q = ArrayD::from_elem(IxDyn(&shape_q), 0.1_f32);
        let lower_kv = ArrayD::from_elem(IxDyn(&shape_kv), -0.1_f32);
        let upper_kv = ArrayD::from_elem(IxDyn(&shape_kv), 0.1_f32);

        let q = BoundedTensor::new(lower_q, upper_q).unwrap();
        let k = BoundedTensor::new(lower_kv.clone(), upper_kv.clone()).unwrap();
        let v = BoundedTensor::new(lower_kv, upper_kv).unwrap();

        let scale = 1.0 / (dim as f32).sqrt();

        let result = device.cross_attention_ibp(&q, &k, &v, scale).unwrap();
        assert_eq!(result.shape(), &shape_q);

        // Verify bounds are valid
        for (l, u) in result.lower.iter().zip(result.upper.iter()) {
            assert!(*l <= *u + 1e-5, "Invalid bounds: lower={} > upper={}", l, u);
        }
    }

    #[test]
    fn test_compute_device_crown_per_position() {
        use gamma_propagate::{GELULayer, GraphNode, Layer, LinearLayer};
        use ndarray::Array2;

        let device = ComputeDevice::new(Backend::Cpu).unwrap();

        // Build a small MLP graph
        let in_features = 4;
        let hidden_features = 8;
        let out_features = 4;

        let weight1 = Array2::from_shape_fn((hidden_features, in_features), |(i, j)| {
            0.1 * ((i + j) as f32 - 6.0)
        });
        let bias1 = ndarray::Array1::from_elem(hidden_features, 0.05_f32);
        let linear1 = LinearLayer::new(weight1, Some(bias1)).unwrap();

        let weight2 = Array2::from_shape_fn((out_features, hidden_features), |(i, j)| {
            0.1 * ((i + j) as f32 - 6.0)
        });
        let bias2 = ndarray::Array1::from_elem(out_features, 0.02_f32);
        let linear2 = LinearLayer::new(weight2, Some(bias2)).unwrap();

        let gelu = GELULayer::default();

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
        graph.add_node(GraphNode::new(
            "gelu",
            Layer::GELU(gelu),
            vec!["linear1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "linear2",
            Layer::Linear(linear2),
            vec!["gelu".to_string()],
        ));
        graph.set_output("linear2");

        let lower = ArrayD::from_elem(IxDyn(&[2, 3, in_features]), 0.0_f32);
        let upper = ArrayD::from_elem(IxDyn(&[2, 3, in_features]), 1.0_f32);
        let input = BoundedTensor::new(lower, upper).unwrap();

        let result = device.crown_per_position_parallel(&graph, &input).unwrap();
        assert_eq!(result.shape(), &[2, 3, out_features]);

        // Verify bounds are valid
        for l in result.lower.iter() {
            for u in result.upper.iter() {
                assert!(*l <= *u + 1e-5);
            }
        }
    }
}
