//! MLX-accelerated bound propagation for Apple Silicon.
//!
//! This module provides GPU acceleration using Apple's MLX framework,
//! optimized for Apple Silicon unified memory architecture.
//!
//! ## Prerequisites
//!
//! Building with the `mlx` feature requires:
//! 1. macOS on Apple Silicon (M1/M2/M3)
//!
//! To enable Metal GPU kernels, also enable `mlx-metal`, which requires:
//! - Xcode.app installed (not just Command Line Tools)
//! - Xcode selected as developer tools: `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`
//!
//! ## Usage
//!
//! ```ignore
//! use gamma_gpu::MlxDevice;
//!
//! let device = MlxDevice::new()?;
//! let result = device.linear_ibp(&input, &weight, bias)?;
//! ```
//!
//! ## Performance Characteristics
//!
//! MLX uses Apple's unified memory, which means:
//! - No explicit CPU<->GPU copies needed
//! - Lazy evaluation (operations are fused automatically)
//! - Optimized for transformer workloads
//!
//! For verification workloads with interval arithmetic, MLX provides:
//! - Fast matrix operations (leveraging AMX/GPU)
//! - Efficient softmax/attention bounds computation
//! - Low latency for small batch sizes (no kernel launch overhead)

use crate::AcceleratedBoundPropagation;
use gamma_core::{GammaError, Result};
use gamma_propagate::GraphNetwork;
use gamma_tensor::BoundedTensor;
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use tracing::{debug, info};

use mlx_rs::ops::{maximum_device, minimum_device};
use mlx_rs::{Array as MlxArray, Stream};

fn mlx_result<T>(context: &'static str, result: mlx_rs::error::Result<T>) -> Result<T> {
    result.map_err(|e| GammaError::NotSupported(format!("MLX {context}: {e}")))
}

/// MLX device for accelerated bound propagation on Apple Silicon.
///
/// This provides GPU acceleration via Apple's MLX framework, which is
/// optimized for the unified memory architecture of M1/M2/M3 chips.
pub struct MlxDevice {
    /// The MLX stream (CPU by default; GPU with `mlx-metal`)
    stream: Stream,
}

impl MlxDevice {
    /// Create a new MLX device.
    ///
    /// With feature `mlx-metal`, this uses the default Metal GPU stream.
    /// Without it, this uses the CPU stream (so `--features mlx` does not
    /// require full Xcode.app).
    ///
    /// # Errors
    ///
    /// Returns an error if MLX initialization fails (e.g., not on Apple Silicon).
    pub fn new() -> Result<Self> {
        #[cfg(feature = "mlx-metal")]
        let stream = {
            info!("Initializing MLX device (Metal GPU)");
            Stream::gpu()
        };
        #[cfg(not(feature = "mlx-metal"))]
        let stream = {
            info!("Initializing MLX device (CPU; enable feature `mlx-metal` for GPU)");
            Stream::cpu()
        };
        Ok(Self { stream })
    }

    /// Create an MLX device using the default Metal GPU stream.
    #[cfg(feature = "mlx-metal")]
    pub fn gpu() -> Result<Self> {
        info!("Initializing MLX device (Metal GPU)");
        let stream = Stream::gpu();
        Ok(Self { stream })
    }

    /// Create an MLX device with explicit CPU stream (for debugging/testing).
    pub fn cpu() -> Result<Self> {
        info!("Initializing MLX device (CPU)");
        let stream = Stream::cpu();
        Ok(Self { stream })
    }

    /// Convert ndarray to MLX Array.
    fn ndarray_to_mlx(&self, arr: &ArrayD<f32>) -> Result<MlxArray> {
        let shape: Vec<i32> = arr.shape().iter().map(|&s| s as i32).collect();
        let data: Vec<f32> = arr.iter().cloned().collect();
        Ok(MlxArray::from_slice(&data, &shape))
    }

    /// Convert MLX Array back to ndarray.
    fn mlx_to_ndarray(&self, arr: &MlxArray, shape: &[usize]) -> Result<ArrayD<f32>> {
        // Ensure computation is complete
        mlx_result("eval", arr.eval())?;

        // Get data from MLX array
        let data: Vec<f32> = arr.as_slice::<f32>().to_vec();
        let data_len = data.len();
        ArrayD::from_shape_vec(IxDyn(shape), data)
            .map_err(|_| GammaError::shape_mismatch(shape.to_vec(), vec![data_len]))
    }

    /// Full attention IBP using MLX: softmax((Q @ K^T) * scale) @ V
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

        if shape_q.len() != 4 || shape_k.len() != 4 || shape_v.len() != 4 {
            return Err(GammaError::InvalidSpec(
                "Attention inputs must be 4D [batch, heads, seq, dim]".to_string(),
            ));
        }

        debug!(
            "MlxDevice attention_ibp: shape={:?}, scale={}",
            shape_q, scale
        );

        // Step 1: Transpose K to [batch, heads, dim, seq]
        let k_transposed = k.transpose_last_two()?;

        // Step 2: Q @ K^T
        let scores = self.matmul_ibp(q, &k_transposed)?;

        // Step 3: Scale scores
        let scores_scaled = scores.scale(scale);

        // Step 4: Softmax over last dimension
        let probs = gamma_transformer::softmax_bounds(&scores_scaled, -1)?;

        // Step 5: probs @ V
        let output = self.matmul_ibp(&probs, v)?;

        Ok(output)
    }

    /// Causal attention IBP for decoder models.
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

        if shape_q.len() != 4 {
            return Err(GammaError::InvalidSpec(
                "Attention inputs must be 4D [batch, heads, seq, dim]".to_string(),
            ));
        }

        debug!(
            "MlxDevice causal_attention_ibp: shape={:?}, scale={}",
            shape_q, scale
        );

        let k_transposed = k.transpose_last_two()?;
        let scores = self.matmul_ibp(q, &k_transposed)?;
        let scores_scaled = scores.scale(scale);

        // Use causal softmax (applies lower-triangular mask)
        let probs = gamma_transformer::causal_softmax_bounds(&scores_scaled, -1)?;

        let output = self.matmul_ibp(&probs, v)?;
        Ok(output)
    }

    /// Cross-attention IBP for encoder-decoder models.
    ///
    /// Q from decoder: [batch, heads, seq_dec, dim]
    /// K, V from encoder: [batch, heads, seq_enc, dim]
    /// Output: [batch, heads, seq_dec, dim]
    pub fn cross_attention_ibp(
        &self,
        q: &BoundedTensor,
        k: &BoundedTensor,
        v: &BoundedTensor,
        scale: f32,
    ) -> Result<BoundedTensor> {
        let shape_q = q.shape();
        let shape_k = k.shape();
        let shape_v = v.shape();

        if shape_q.len() != 4 || shape_k.len() != 4 || shape_v.len() != 4 {
            return Err(GammaError::InvalidSpec(
                "Cross-attention inputs must be 4D [batch, heads, seq, dim]".to_string(),
            ));
        }

        // Verify batch and heads match
        if shape_q[0] != shape_k[0] || shape_q[1] != shape_k[1] {
            return Err(GammaError::ShapeMismatch {
                expected: vec![shape_q[0], shape_q[1]],
                got: vec![shape_k[0], shape_k[1]],
            });
        }

        // Verify K and V encoder sequence lengths match
        if shape_k[2] != shape_v[2] {
            return Err(GammaError::shape_mismatch(
                vec![shape_k[2]],
                vec![shape_v[2]],
            ));
        }

        debug!(
            "MlxDevice cross_attention_ibp: q_shape={:?}, kv_shape={:?}, scale={}",
            shape_q, shape_k, scale
        );

        let k_transposed = k.transpose_last_two()?;
        let scores = self.matmul_ibp(q, &k_transposed)?;
        let scores_scaled = scores.scale(scale);

        // Standard softmax (no causal mask for cross-attention)
        let probs = gamma_transformer::softmax_bounds(&scores_scaled, -1)?;

        let output = self.matmul_ibp(&probs, v)?;
        Ok(output)
    }
}

impl AcceleratedBoundPropagation for MlxDevice {
    fn linear_ibp(
        &self,
        input: &BoundedTensor,
        weight: &Array2<f32>,
        bias: Option<&Array1<f32>>,
    ) -> Result<BoundedTensor> {
        let in_features = weight.ncols();
        let out_features = weight.nrows();
        let shape = input.shape();

        if shape.is_empty() || shape[shape.len() - 1] != in_features {
            return Err(GammaError::shape_mismatch(
                vec![in_features],
                shape.to_vec(),
            ));
        }

        let batch_size: usize = shape[..shape.len() - 1].iter().product();

        debug!(
            "MlxDevice linear_ibp: batch={}, in={}, out={}",
            batch_size, in_features, out_features
        );

        // Convert to MLX arrays
        let input_lower = self.ndarray_to_mlx(&input.lower)?;
        let input_upper = self.ndarray_to_mlx(&input.upper)?;

        // Convert weight to MLX and compute positive/negative parts
        let weight_data: Vec<f32> = weight.iter().cloned().collect();
        let weight_shape = [weight.nrows() as i32, weight.ncols() as i32];
        let weight_mlx = MlxArray::from_slice(&weight_data, &weight_shape);

        // W_pos = max(W, 0), W_neg = min(W, 0)
        let zero = MlxArray::from_slice(&[0.0f32], &[1]);
        let weight_pos = mlx_result(
            "maximum(weight, 0)",
            maximum_device(&weight_mlx, &zero, &self.stream),
        )?;
        let weight_neg = mlx_result(
            "minimum(weight, 0)",
            minimum_device(&weight_mlx, &zero, &self.stream),
        )?;

        // Reshape input for matmul: [batch_size, in_features]
        let flat_shape = [batch_size as i32, in_features as i32];
        let input_lower_flat = mlx_result(
            "reshape(input_lower)",
            input_lower.reshape_device(&flat_shape, &self.stream),
        )?;
        let input_upper_flat = mlx_result(
            "reshape(input_upper)",
            input_upper.reshape_device(&flat_shape, &self.stream),
        )?;

        // Transpose weight for matmul: [out_features, in_features] -> [in_features, out_features]
        let weight_pos_t = mlx_result(
            "transpose(weight_pos)",
            weight_pos.transpose_device(&self.stream),
        )?;
        let weight_neg_t = mlx_result(
            "transpose(weight_neg)",
            weight_neg.transpose_device(&self.stream),
        )?;

        // lower = W_pos @ l + W_neg @ u
        // upper = W_pos @ u + W_neg @ l
        let term1_lower = mlx_result(
            "matmul(l, W_pos^T)",
            input_lower_flat.matmul_device(&weight_pos_t, &self.stream),
        )?;
        let term2_lower = mlx_result(
            "matmul(u, W_neg^T)",
            input_upper_flat.matmul_device(&weight_neg_t, &self.stream),
        )?;
        let term1_upper = mlx_result(
            "matmul(u, W_pos^T)",
            input_upper_flat.matmul_device(&weight_pos_t, &self.stream),
        )?;
        let term2_upper = mlx_result(
            "matmul(l, W_neg^T)",
            input_lower_flat.matmul_device(&weight_neg_t, &self.stream),
        )?;

        let mut output_lower = mlx_result(
            "add(term1_lower, term2_lower)",
            term1_lower.add_device(&term2_lower, &self.stream),
        )?;
        let mut output_upper = mlx_result(
            "add(term1_upper, term2_upper)",
            term1_upper.add_device(&term2_upper, &self.stream),
        )?;

        // Add bias if present
        if let Some(b) = bias {
            let bias_data: Vec<f32> = b.iter().cloned().collect();
            let bias_mlx = MlxArray::from_slice(&bias_data, &[out_features as i32]);
            output_lower = mlx_result(
                "add(output_lower, bias)",
                output_lower.add_device(&bias_mlx, &self.stream),
            )?;
            output_upper = mlx_result(
                "add(output_upper, bias)",
                output_upper.add_device(&bias_mlx, &self.stream),
            )?;
        }

        // Reshape output to [..., out_features]
        let mut out_shape = shape[..shape.len() - 1].to_vec();
        out_shape.push(out_features);

        let lower_nd = self.mlx_to_ndarray(&output_lower, &out_shape)?;
        let upper_nd = self.mlx_to_ndarray(&output_upper, &out_shape)?;

        BoundedTensor::new(lower_nd, upper_nd)
    }

    fn matmul_ibp(
        &self,
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

        let m = shape_a[shape_a.len() - 2];
        let k = shape_a[shape_a.len() - 1];
        let n = shape_b[shape_b.len() - 1];

        if shape_b[shape_b.len() - 2] != k {
            return Err(GammaError::shape_mismatch(
                vec![k],
                vec![shape_b[shape_b.len() - 2]],
            ));
        }

        let batch_dims_a = &shape_a[..shape_a.len() - 2];
        let batch_dims_b = &shape_b[..shape_b.len() - 2];

        if batch_dims_a != batch_dims_b {
            return Err(GammaError::shape_mismatch(
                batch_dims_a.to_vec(),
                batch_dims_b.to_vec(),
            ));
        }

        let batch_size: usize = batch_dims_a.iter().product();

        debug!(
            "MlxDevice matmul_ibp: batch={}, m={}, k={}, n={}",
            batch_size, m, k, n
        );

        // Convert to MLX arrays
        let al = self.ndarray_to_mlx(&input_a.lower)?;
        let au = self.ndarray_to_mlx(&input_a.upper)?;
        let bl = self.ndarray_to_mlx(&input_b.lower)?;
        let bu = self.ndarray_to_mlx(&input_b.upper)?;

        // For interval multiplication [a_l, a_u] * [b_l, b_u], we need:
        // - If both intervals contain only non-negative values: result = [a_l*b_l, a_u*b_u]
        // - If both intervals contain only non-positive values: result = [a_u*b_u, a_l*b_l]
        // - For mixed signs, we need all four products and take min/max
        //
        // For matrix multiplication, we use the standard IBP formulation:
        // For A in [A_l, A_u], B in [B_l, B_u]:
        // (A @ B)_ij = sum_k A_ik * B_kj
        //
        // This is complex with interval arithmetic. For now, use element-wise interval
        // multiplication followed by sum, which is sound but potentially loose.
        //
        // More efficient: compute all 4 products and combine
        let p1 = mlx_result("matmul(al, bl)", al.matmul_device(&bl, &self.stream))?; // al @ bl
        let p2 = mlx_result("matmul(al, bu)", al.matmul_device(&bu, &self.stream))?; // al @ bu
        let p3 = mlx_result("matmul(au, bl)", au.matmul_device(&bl, &self.stream))?; // au @ bl
        let p4 = mlx_result("matmul(au, bu)", au.matmul_device(&bu, &self.stream))?; // au @ bu

        // For each element, take min and max across all products
        // output_lower = min(p1, p2, p3, p4)
        // output_upper = max(p1, p2, p3, p4)
        let min_12 = mlx_result("minimum(p1, p2)", minimum_device(&p1, &p2, &self.stream))?;
        let min_34 = mlx_result("minimum(p3, p4)", minimum_device(&p3, &p4, &self.stream))?;
        let output_lower = mlx_result(
            "minimum(min_12, min_34)",
            minimum_device(&min_12, &min_34, &self.stream),
        )?;

        let max_12 = mlx_result("maximum(p1, p2)", maximum_device(&p1, &p2, &self.stream))?;
        let max_34 = mlx_result("maximum(p3, p4)", maximum_device(&p3, &p4, &self.stream))?;
        let output_upper = mlx_result(
            "maximum(max_12, max_34)",
            maximum_device(&max_12, &max_34, &self.stream),
        )?;

        // Build output shape
        let mut out_shape = batch_dims_a.to_vec();
        out_shape.push(m);
        out_shape.push(n);

        let lower_nd = self.mlx_to_ndarray(&output_lower, &out_shape)?;
        let upper_nd = self.mlx_to_ndarray(&output_upper, &out_shape)?;

        BoundedTensor::new(lower_nd, upper_nd)
    }

    fn crown_per_position_parallel(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        // For CROWN, we fall back to the CPU implementation since CROWN
        // involves complex backward passes that are not easily expressed
        // in MLX's forward-mode operations.
        //
        // The CPU implementation in lib.rs already uses Rayon parallelism,
        // which provides good performance.
        crate::crown_per_position_parallel(graph, input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_mlx_linear_ibp_basic() {
        let device = MlxDevice::new().expect("MLX should initialize");

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
        assert_relative_eq!(result.lower[[0, 0]], -1.4, epsilon = 1e-4);
        assert_relative_eq!(result.upper[[0, 0]], 1.6, epsilon = 1e-4);
    }

    #[test]
    fn test_mlx_matmul_ibp_basic() {
        let device = MlxDevice::new().expect("MLX should initialize");

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

        // For A,B in [0, 1]: result in [0, 3]
        assert_relative_eq!(result.lower[[0, 0]], 0.0, epsilon = 1e-4);
        assert_relative_eq!(result.upper[[0, 0]], 3.0, epsilon = 1e-4);
    }

    #[test]
    fn test_mlx_attention_ibp_basic() {
        let device = MlxDevice::new().expect("MLX should initialize");

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

        // Check bounds are valid
        for (l, u) in result.lower.iter().zip(result.upper.iter()) {
            assert!(*l <= *u + 1e-5, "Invalid bounds: lower={} > upper={}", l, u);
        }
    }

    #[test]
    fn test_mlx_cpu_constructor() {
        // Test the explicit CPU constructor
        let device = MlxDevice::cpu().expect("MLX CPU should initialize");

        // Simple test to verify device works
        let lower = ArrayD::from_elem(IxDyn(&[2, 3]), 0.0_f32);
        let upper = ArrayD::from_elem(IxDyn(&[2, 3]), 1.0_f32);
        let input = BoundedTensor::new(lower, upper).unwrap();

        let weight = Array2::from_elem((4, 3), 0.5_f32);
        let result = device.linear_ibp(&input, &weight, None).unwrap();
        assert_eq!(result.shape(), &[2, 4]);
    }

    #[test]
    fn test_mlx_linear_ibp_batched() {
        let device = MlxDevice::new().expect("MLX should initialize");

        // Create batched input: shape [2, 3, 4] with bounds
        let lower = ArrayD::from_elem(IxDyn(&[2, 3, 4]), 0.0_f32);
        let upper = ArrayD::from_elem(IxDyn(&[2, 3, 4]), 1.0_f32);
        let input = BoundedTensor::new(lower, upper).unwrap();

        // Weight [8, 4] -> output [2, 3, 8]
        let weight = Array2::from_elem((8, 4), 0.25_f32);

        let result = device.linear_ibp(&input, &weight, None).unwrap();

        assert_eq!(result.shape(), &[2, 3, 8]);

        // For w=0.25 and x in [0, 1]: sum of 4 terms = [0, 1]
        assert_relative_eq!(result.lower[[0, 0, 0]], 0.0, epsilon = 1e-4);
        assert_relative_eq!(result.upper[[0, 0, 0]], 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_mlx_matmul_ibp_batched() {
        let device = MlxDevice::new().expect("MLX should initialize");

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
        assert_relative_eq!(result.lower[[0, 0, 0]], 0.75, epsilon = 1e-4);
        assert_relative_eq!(result.upper[[0, 0, 0]], 3.0, epsilon = 1e-4);
    }

    #[test]
    fn test_mlx_causal_attention_ibp() {
        let device = MlxDevice::new().expect("MLX should initialize");

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

        // Check bounds are valid
        for (l, u) in result.lower.iter().zip(result.upper.iter()) {
            assert!(*l <= *u + 1e-5, "Invalid bounds: lower={} > upper={}", l, u);
        }
    }

    #[test]
    fn test_mlx_cross_attention_ibp() {
        let device = MlxDevice::new().expect("MLX should initialize");

        // Q from decoder: [batch=1, heads=2, seq_dec=3, dim=4]
        // K, V from encoder: [batch=1, heads=2, seq_enc=5, dim=4]
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

        // Output should match decoder sequence length
        assert_eq!(result.shape(), &shape_q);

        // Bounds should be valid
        for (l, u) in result.lower.iter().zip(result.upper.iter()) {
            assert!(*l <= *u + 1e-5, "Invalid bounds: lower={} > upper={}", l, u);
        }
    }

    #[test]
    fn test_mlx_cross_attention_shape_validation() {
        let device = MlxDevice::new().expect("MLX should initialize");

        // Valid shapes
        let shape_q = [1, 2, 3, 4];
        let shape_kv = [1, 2, 5, 4];

        let q = BoundedTensor::new(
            ArrayD::zeros(IxDyn(&shape_q)),
            ArrayD::zeros(IxDyn(&shape_q)),
        )
        .unwrap();
        let k = BoundedTensor::new(
            ArrayD::zeros(IxDyn(&shape_kv)),
            ArrayD::zeros(IxDyn(&shape_kv)),
        )
        .unwrap();
        let v = BoundedTensor::new(
            ArrayD::zeros(IxDyn(&shape_kv)),
            ArrayD::zeros(IxDyn(&shape_kv)),
        )
        .unwrap();

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
    }
}
